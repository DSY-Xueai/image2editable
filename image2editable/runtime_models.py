from __future__ import annotations

import ctypes
import errno
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import sys
from urllib.request import urlopen

from image2editable.model_receipts import (
    canonical_sha256,
    manifest_files,
    read_strict_json,
    strict_file_record,
    validate_manifest,
)


CATALOG_PATH = Path(__file__).with_name("runtime_model_catalog.json")
RECEIPT_NAME = "runtime-receipt.json"
INSTALL_COMMAND = "image2editable models install runtime"
_MODEL_NAMES = {"sam2_large", "big_lama", "grounding_dino"}


class RuntimeModelError(RuntimeError):
    pass


def download_file(url: str, descriptor: int) -> None:
    with urlopen(url) as response:
        while chunk := response.read(1024 * 1024):
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("runtime model download write failed")
                remaining = remaining[written:]


def snapshot_download(**kwargs: object) -> str:
    try:
        from huggingface_hub import snapshot_download as huggingface_download
    except ImportError as exc:
        raise RuntimeError(
            "Runtime model dependencies are missing; install project dependencies"
        ) from exc
    return str(huggingface_download(**kwargs))


def default_runtime_cache() -> Path:
    configured = os.environ.get("IMAGE2EDITABLE_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path.home() / ".cache" / "image2editable" / "models" / "runtime"
    ).resolve()


def _cache_path(cache_dir: str | Path | None) -> Path:
    if cache_dir is None:
        return default_runtime_cache()
    return Path(cache_dir).expanduser().resolve()


def load_runtime_catalog(path: str | Path | None = None) -> dict[str, object]:
    catalog_path = Path(path) if path is not None else CATALOG_PATH
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    _validate_catalog(catalog)
    return catalog


def _validate_catalog(catalog: object) -> None:
    if not isinstance(catalog, dict) or set(catalog) != {"schema_version", "models"}:
        raise ValueError("runtime model catalog fields are invalid")
    if type(catalog["schema_version"]) is not int or catalog["schema_version"] != 1:
        raise ValueError("runtime model catalog schema version is unsupported")
    models = catalog["models"]
    if not isinstance(models, dict) or set(models) != _MODEL_NAMES:
        raise ValueError("runtime model catalog models are invalid")
    for name in ("sam2_large", "big_lama"):
        entry = models[name]
        fields = {"kind", "url", "size", "sha256", "relative_path"}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"runtime model catalog {name} fields are invalid")
        if (
            entry["kind"] != "file"
            or not isinstance(entry["url"], str)
            or not entry["url"].startswith("https://")
            or type(entry["size"]) is not int
            or entry["size"] <= 0
            or not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            or not isinstance(entry["relative_path"], str)
            or not entry["relative_path"]
            or Path(entry["relative_path"]).name != entry["relative_path"]
        ):
            raise ValueError(f"runtime model catalog {name} values are invalid")
    entry = models["grounding_dino"]
    if not isinstance(entry, dict) or set(entry) != {"kind", "model_id", "revision"}:
        raise ValueError("runtime model catalog grounding_dino fields are invalid")
    if (
        entry["kind"] != "huggingface_snapshot"
        or not isinstance(entry["model_id"], str)
        or not entry["model_id"]
        or not isinstance(entry["revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", entry["revision"]) is None
    ):
        raise ValueError("runtime model catalog grounding_dino values are invalid")


def install_runtime_models(
    *,
    cache_dir: str | Path | None = None,
    confirmed: bool = False,
) -> dict[str, object]:
    if not confirmed:
        raise PermissionError("explicit confirmation required before model download")
    cache = _cache_path(cache_dir)
    catalog = load_runtime_catalog()
    _validate_catalog(catalog)
    cache.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(cache / RECEIPT_NAME):
        return _load_valid_receipt(cache, catalog)
    records = {}
    for name in ("sam2_large", "big_lama"):
        records[name] = _install_file(cache, catalog["models"][name])
    records["grounding_dino"] = _install_snapshot(
        cache,
        catalog["models"]["grounding_dino"],
    )
    receipt = {
        "schema_version": 1,
        "catalog_sha256": canonical_sha256(catalog),
        "models": records,
    }
    _publish_json(cache / RECEIPT_NAME, receipt)
    return receipt


def _private_file(
    parent: Path,
    label: str,
) -> tuple[Path, int, tuple[int, int, int]]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    for _ in range(100):
        path = parent / f".{label}-{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            continue
        status = os.fstat(descriptor)
        return path, descriptor, _private_file_identity(status)
    raise RuntimeError(f"cannot allocate a private model file for {label}")


def _private_file_identity(status: os.stat_result) -> tuple[int, int, int]:
    return status.st_dev, status.st_ino, status.st_size


def _cleanup_private(
    path: Path,
    identity: tuple[int, int, int],
) -> None:
    try:
        status = path.lstat()
    except OSError:
        return
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(status.st_mode)
        or getattr(status, "st_file_attributes", 0) & reparse
        or not stat.S_ISREG(status.st_mode)
        or _private_file_identity(status) != identity
    ):
        return
    path.unlink()


def _install_file(cache: Path, entry: dict[str, object]) -> dict[str, object]:
    target = cache / entry["relative_path"]
    if os.path.lexists(target):
        record = strict_file_record(target, cache)
        _require_file_identity(record, entry)
        return _file_receipt(entry)
    temporary, descriptor, identity = _private_file(cache, str(entry["relative_path"]))
    try:
        download_file(str(entry["url"]), descriptor)
        os.fsync(descriptor)
        downloaded = strict_file_record(temporary, cache)
        _require_file_identity(downloaded, entry)
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            existing = strict_file_record(target, cache)
            _require_file_identity(existing, entry)
        return _file_receipt(entry)
    finally:
        identity = _private_file_identity(os.fstat(descriptor))
        os.close(descriptor)
        _cleanup_private(temporary, identity)


def _require_file_identity(
    record: dict[str, object],
    entry: dict[str, object],
) -> None:
    if record["size"] != entry["size"] or record["sha256"] != entry["sha256"]:
        raise RuntimeError(
            f"runtime model integrity verification failed: {entry['relative_path']}"
        )


def _file_receipt(entry: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "file",
        "relative_path": entry["relative_path"],
        "size": entry["size"],
        "sha256": entry["sha256"],
    }


def _install_snapshot(cache: Path, entry: dict[str, object]) -> dict[str, object]:
    revision = str(entry["revision"])
    relative_path = Path("grounding_dino") / revision
    target = cache / relative_path
    if os.path.lexists(target):
        raise RuntimeError(f"runtime snapshot already exists without a receipt: {target}")
    parent = target.parent
    try:
        parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    parent_status = _directory_status(parent, "snapshot parent")
    parent_binding = _open_directory(parent)
    staging_binding = None
    try:
        _validate_parent(parent, parent_binding, parent_status)
        staging, staging_identity, staging_binding = _private_directory(
            parent,
            revision,
            parent_binding,
        )
        local_dir = _snapshot_download_path(
            staging,
            staging_identity,
            staging_binding,
        )
        returned = Path(
            snapshot_download(
                repo_id=entry["model_id"],
                revision=revision,
                local_dir=str(local_dir),
                cache_dir=str(cache / ".huggingface"),
            )
        ).resolve()
        _validate_parent(parent, parent_binding, parent_status)
        _validate_directory_binding(
            staging,
            staging_binding,
            staging_identity,
            "snapshot staging",
        )
        if returned != staging.resolve():
            raise RuntimeError("runtime snapshot download returned an unexpected path")
        staged_files = manifest_files(local_dir, cache, strict=True)
        _publish_directory(
            staging,
            target,
            parent_binding,
            parent_status,
            staging_binding,
            staging_identity,
        )
        files = manifest_files(target, cache, strict=True)
        if files != staged_files:
            raise RuntimeError("runtime snapshot changed while being published")
    except BaseException as error:
        if staging_binding is not None:
            _close_directory(staging_binding)
            staging_binding = None
        if "staging" in locals() and not _cleanup_private_directory(
            staging,
            staging_identity,
            parent,
            parent_binding,
            parent_status,
        ):
            raise RuntimeError(
                f"{error}; private staging preserved; inspect and remove: {staging.name}"
            ) from error
        raise
    finally:
        if staging_binding is not None:
            _close_directory(staging_binding)
        _close_directory(parent_binding)
    return {
        "kind": "huggingface_snapshot",
        "model_id": entry["model_id"],
        "requested_revision": revision,
        "resolved_revision": revision,
        "relative_path": relative_path.as_posix(),
        "files": files,
    }


def _directory_status(path: Path, label: str) -> os.stat_result:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"runtime {label} identity changed: {path}") from error
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(status.st_mode)
        or bool(getattr(status, "st_file_attributes", 0) & reparse)
        or not stat.S_ISDIR(status.st_mode)
    ):
        raise RuntimeError(f"runtime {label} is unsafe: {path}")
    return status


def _directory_identity(status: os.stat_result) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _require_directory_identity(
    path: Path,
    expected: tuple[int, int],
    label: str,
) -> None:
    if _directory_identity(_directory_status(path, label)) != expected:
        raise RuntimeError(f"runtime {label} identity changed: {path}")


def _open_directory(path: Path) -> tuple[int, tuple[int, int] | None]:
    if os.name == "nt":
        before = _directory_status(path, "directory")
        handle = _open_windows_directory(path)
        after = _directory_status(path, "directory")
        if _directory_identity(before) != _directory_identity(after):
            _close_windows_handle(handle)
            raise RuntimeError(f"runtime directory identity changed: {path}")
        return handle, _windows_directory_identity(handle)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    nofollow_flag = getattr(os, "O_NOFOLLOW", None)
    if directory_flag is None or nofollow_flag is None:
        raise RuntimeError("runtime snapshot parent cannot be opened safely")
    flags = os.O_RDONLY | directory_flag | nofollow_flag
    flags |= getattr(os, "O_CLOEXEC", 0)
    return os.open(path, flags), None


def _close_directory(binding: tuple[int, tuple[int, int] | None]) -> None:
    try:
        if binding[1] is None:
            os.close(binding[0])
        else:
            _close_windows_handle(binding[0])
    except OSError:
        pass


def _private_directory(
    parent: Path,
    label: str,
    parent_binding: tuple[int, tuple[int, int] | None],
) -> tuple[Path, tuple[int, int], tuple[int, tuple[int, int] | None]]:
    path = parent / f".grounding-dino-{label}.installing"
    try:
        if os.name == "nt":
            path.mkdir(mode=0o700)
        else:
            os.mkdir(path.name, mode=0o700, dir_fd=parent_binding[0])
    except FileExistsError as error:
        raise RuntimeError(
            f"runtime snapshot staging already exists; inspect and remove: {path.name}"
        ) from error
    if os.name == "nt":
        status = _directory_status(path, "snapshot staging")
        binding = _open_directory(path)
    else:
        status = os.stat(path.name, dir_fd=parent_binding[0], follow_symlinks=False)
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(f"runtime snapshot staging is unsafe: {path}")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        binding = (os.open(path.name, flags, dir_fd=parent_binding[0]), None)
    try:
        if binding[1] is None:
            opened = os.fstat(binding[0])
            if _directory_identity(opened) != _directory_identity(status):
                raise RuntimeError(f"runtime snapshot staging identity changed: {path}")
        else:
            _validate_directory_binding(
                path,
                binding,
                _directory_identity(status),
                "snapshot staging",
            )
    except BaseException:
        _close_directory(binding)
        raise
    return path, _directory_identity(status), binding


def _snapshot_download_path(
    staging: Path,
    expected: tuple[int, int],
    staging_binding: tuple[int, tuple[int, int] | None],
) -> Path:
    if sys.platform == "win32":
        return staging
    if sys.platform.startswith("linux"):
        root = Path("/proc/self/fd")
    elif sys.platform == "darwin":
        root = Path("/dev/fd")
    else:
        raise RuntimeError("runtime snapshot descriptor anchor is unavailable")
    anchor = root / str(staging_binding[0])
    try:
        resolved = anchor.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("runtime snapshot descriptor anchor is unavailable") from error
    opened = os.fstat(staging_binding[0])
    if (
        resolved != staging.resolve()
        or not stat.S_ISDIR(opened.st_mode)
        or _directory_identity(opened) != expected
    ):
        raise RuntimeError("runtime snapshot descriptor anchor identity mismatch")
    return anchor


def _validate_directory_binding(
    path: Path,
    binding: tuple[int, tuple[int, int] | None],
    expected: tuple[int, int],
    label: str,
) -> None:
    _require_directory_identity(path, expected, label)
    if binding[1] is None:
        opened = os.fstat(binding[0])
        if not stat.S_ISDIR(opened.st_mode) or _directory_identity(opened) != expected:
            raise RuntimeError(f"runtime {label} identity changed: {path}")
    elif _windows_directory_identity(binding[0]) != binding[1]:
        raise RuntimeError(f"runtime {label} identity changed: {path}")


def _open_windows_directory(path: Path) -> int:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x80000000 | 0x00010000,
        0x00000001 | 0x00000002,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _windows_directory_identity(handle: int) -> tuple[int, int]:
    from ctypes import wintypes

    class FileInformation(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("creation", wintypes.FILETIME),
            ("access", wintypes.FILETIME),
            ("write", wintypes.FILETIME),
            ("volume_serial", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        ]

    information = FileInformation()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = [wintypes.HANDLE, ctypes.POINTER(FileInformation)]
    get_information.restype = wintypes.BOOL
    if not get_information(
        wintypes.HANDLE(handle), ctypes.byref(information)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return information.index_high, information.index_low


def _close_windows_handle(handle: int) -> None:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    if not close_handle(wintypes.HANDLE(handle)):
        raise ctypes.WinError(ctypes.get_last_error())


def _rename_windows_directory(
    staging_handle: int,
    target: Path,
) -> None:
    from ctypes import wintypes

    target_name = str(target)
    class RenameInformation(ctypes.Structure):
        _fields_ = [
            ("replace", wintypes.BOOL),
            ("root", wintypes.HANDLE),
            ("name_length", wintypes.DWORD),
            ("name", wintypes.WCHAR * (len(target_name) + 1)),
        ]

    information = RenameInformation()
    information.replace = False
    information.root = None
    information.name_length = len(target_name.encode("utf-16-le"))
    information.name = target_name
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    set_information.restype = wintypes.BOOL
    if set_information(
        wintypes.HANDLE(staging_handle),
        3,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        return
    error_number = ctypes.get_last_error()
    if error_number in {80, 183}:
        raise FileExistsError(f"runtime snapshot already exists: {target}")
    raise ctypes.WinError(error_number)


def _validate_parent(
    parent: Path,
    binding: tuple[int, tuple[int, int] | None],
    expected: os.stat_result,
) -> None:
    identity = _directory_identity(expected)
    try:
        current = _directory_status(parent, "snapshot parent")
    except RuntimeError as error:
        raise RuntimeError(f"runtime snapshot parent identity changed: {parent}") from error
    if _directory_identity(current) != identity:
        raise RuntimeError(f"runtime snapshot parent identity changed: {parent}")
    if binding[1] is None:
        opened = os.fstat(binding[0])
        if not stat.S_ISDIR(opened.st_mode) or _directory_identity(opened) != identity:
            raise RuntimeError(f"runtime snapshot parent identity changed: {parent}")
    elif _windows_directory_identity(binding[0]) != binding[1]:
        raise RuntimeError(f"runtime snapshot parent identity changed: {parent}")


def _publish_directory(
    staging: Path,
    target: Path,
    parent_binding: tuple[int, tuple[int, int] | None],
    parent_status: os.stat_result,
    staging_binding: tuple[int, tuple[int, int] | None],
    staging_identity: tuple[int, int],
) -> None:
    if staging.parent != target.parent:
        raise RuntimeError("runtime snapshot publication directories differ")
    _validate_parent(staging.parent, parent_binding, parent_status)
    _validate_directory_binding(
        staging,
        staging_binding,
        staging_identity,
        "snapshot staging",
    )
    if os.name == "nt":
        try:
            _rename_windows_directory(
                staging_binding[0],
                target,
            )
        except OSError as error:
            _validate_parent(target.parent, parent_binding, parent_status)
            if os.path.lexists(target):
                raise FileExistsError(
                    f"runtime snapshot already exists: {target}"
                ) from error
            raise
    else:
        if sys.platform.startswith("linux"):
            symbol = "renameat2"
            flag = 1
        elif sys.platform == "darwin":
            symbol = "renameatx_np"
            flag = 4
        else:
            raise RuntimeError("runtime snapshot no-replace publication is unavailable")
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, symbol, None)
        if rename is None:
            raise RuntimeError("runtime snapshot no-replace publication is unavailable")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            parent_binding[0],
            os.fsencode(staging.name),
            parent_binding[0],
            os.fsencode(target.name),
            flag,
        )
        error_number = ctypes.get_errno()
        _validate_parent(target.parent, parent_binding, parent_status)
        if result != 0:
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(
                    f"runtime snapshot already exists: {target}"
                )
            raise OSError(error_number, os.strerror(error_number), str(target))
    _validate_parent(target.parent, parent_binding, parent_status)
    _validate_directory_binding(
        target,
        staging_binding,
        staging_identity,
        "published snapshot",
    )


def _cleanup_private_directory(
    staging: Path,
    expected_identity: tuple[int, int],
    parent: Path,
    parent_binding: tuple[int, tuple[int, int] | None],
    parent_status: os.stat_result,
) -> bool:
    if not os.path.lexists(staging):
        return True
    try:
        _validate_parent(parent, parent_binding, parent_status)
        _require_directory_identity(staging, expected_identity, "snapshot staging")
        for path in staging.rglob("*"):
            status = path.lstat()
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if (
                stat.S_ISLNK(status.st_mode)
                or bool(getattr(status, "st_file_attributes", 0) & reparse)
                or not (stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode))
            ):
                return False
        if os.name == "nt":
            shutil.rmtree(staging)
        elif getattr(shutil.rmtree, "avoids_symlink_attacks", False):
            shutil.rmtree(staging.name, dir_fd=parent_binding[0])
        else:
            return False
    except (OSError, RuntimeError):
        return False
    return not os.path.lexists(staging)


def _publish_json(path: Path, document: dict[str, object]) -> None:
    if os.path.lexists(path):
        raise RuntimeError(f"runtime model receipt already exists: {path}")
    temporary, descriptor, identity = _private_file(path.parent, path.name)
    try:
        payload = (
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("runtime model receipt write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise RuntimeError(f"runtime model receipt already exists: {path}") from exc
    finally:
        identity = _private_file_identity(os.fstat(descriptor))
        os.close(descriptor)
        _cleanup_private(temporary, identity)


def runtime_model_status(
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, object]:
    cache = _cache_path(cache_dir)
    receipt_path = cache / RECEIPT_NAME
    base = {
        "installed": False,
        "valid": False,
        "install_command": INSTALL_COMMAND,
    }
    if not os.path.lexists(receipt_path):
        return base
    try:
        catalog = load_runtime_catalog()
        _validate_catalog(catalog)
        receipt = _load_valid_receipt(cache, catalog)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        return {
            **base,
            "installed": True,
            "reason": str(exc),
        }
    return {
        "installed": True,
        "valid": True,
        "install_command": INSTALL_COMMAND,
        "receipt": receipt,
    }


def _load_valid_receipt(
    cache: Path,
    catalog: dict[str, object],
) -> dict[str, object]:
    receipt_path = cache / RECEIPT_NAME
    strict_file_record(receipt_path, cache)
    receipt = read_strict_json(receipt_path, cache)
    _validate_receipt(cache, catalog, receipt)
    return receipt


def _validate_receipt(
    cache: Path,
    catalog: dict[str, object],
    receipt: object,
) -> None:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema_version",
        "catalog_sha256",
        "models",
    }:
        raise RuntimeError("runtime model receipt fields are invalid")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise RuntimeError("runtime model receipt schema version is unsupported")
    if receipt["catalog_sha256"] != canonical_sha256(catalog):
        raise RuntimeError("runtime model receipt catalog hash mismatch")
    records = receipt["models"]
    if not isinstance(records, dict) or set(records) != _MODEL_NAMES:
        raise RuntimeError("runtime model receipt models are invalid")
    for name in ("sam2_large", "big_lama"):
        entry = catalog["models"][name]
        record = records[name]
        if not isinstance(record, dict) or set(record) != {
            "kind",
            "relative_path",
            "size",
            "sha256",
        }:
            raise RuntimeError(f"runtime model receipt {name} fields are invalid")
        if record != {
            "kind": "file",
            "relative_path": entry["relative_path"],
            "size": entry["size"],
            "sha256": entry["sha256"],
        }:
            raise RuntimeError(f"runtime model receipt {name} does not match catalog")
        actual = strict_file_record(cache / record["relative_path"], cache)
        _require_file_identity(actual, entry)
    entry = catalog["models"]["grounding_dino"]
    record = records["grounding_dino"]
    snapshot_fields = {
        "kind", "model_id", "requested_revision", "resolved_revision",
        "relative_path", "files",
    }
    if not isinstance(record, dict) or set(record) != snapshot_fields:
        raise RuntimeError("runtime model receipt grounding_dino fields are invalid")
    revision = entry["revision"]
    expected_relative = f"grounding_dino/{revision}"
    if (
        record["kind"] != "huggingface_snapshot"
        or record["model_id"] != entry["model_id"]
        or record["requested_revision"] != revision
        or record["resolved_revision"] != revision
        or re.fullmatch(r"[0-9a-f]{40}", record["resolved_revision"]) is None
        or record["relative_path"] != expected_relative
    ):
        raise RuntimeError("runtime model receipt grounding_dino does not match catalog")
    snapshot = (cache / record["relative_path"]).resolve()
    if not snapshot.is_relative_to(cache) or not snapshot.is_dir():
        raise RuntimeError("runtime model snapshot is missing or outside the cache")
    validate_manifest(
        snapshot,
        cache,
        record["files"],
        strict=True,
        require_sorted=True,
    )


def runtime_model_path(
    name: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    if name not in _MODEL_NAMES:
        raise RuntimeModelError(f"unknown runtime model: {name}")
    cache = _cache_path(cache_dir)
    status = runtime_model_status(cache_dir=cache)
    if not status["valid"]:
        detail = status.get("reason", "runtime models are not installed")
        raise RuntimeModelError(f"{detail}; run: {INSTALL_COMMAND}")
    relative_path = status["receipt"]["models"][name]["relative_path"]
    return (cache / relative_path).resolve()
