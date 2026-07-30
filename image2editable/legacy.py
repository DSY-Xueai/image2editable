from __future__ import annotations

from contextlib import redirect_stdout
import ctypes
import importlib
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any

from image2editable.contracts import validate_schema_version
from image2editable.inputs import sha256_file
from image2editable.store import RunStore


def _absolute_outputs(value: Any) -> Any:
    if isinstance(value, str):
        return str(Path(value).resolve())
    if isinstance(value, list):
        return [_absolute_outputs(item) for item in value]
    if isinstance(value, dict):
        return {key: _absolute_outputs(item) for key, item in value.items()}
    return value


def _source_path(store: RunStore, page_id: str) -> Path:
    request = store.read_json(
        Path("pages") / page_id / "page_request.json"
    )
    validate_schema_version(request)
    source = (store.root / request["source"]).resolve()
    if not source.is_relative_to(store.root):
        raise ValueError(f"{page_id}: source is outside run directory")
    if not source.is_file():
        raise ValueError(f"{page_id}: source is not a file")
    if sha256_file(source) != request["sha256"]:
        raise ValueError(f"{page_id}: source sha256 mismatch")
    return source


def _is_link_or_reparse(status: Any) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )


def _directory_identity(status: Any) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _validate_directory(
    path: Path,
    status: Any,
    expected_identity: tuple[int, int] | None,
) -> None:
    if (
        expected_identity is not None
        and _directory_identity(status) != expected_identity
    ):
        raise RuntimeError(f"Directory changed before cleanup: {path}")
    if _is_link_or_reparse(status):
        raise RuntimeError(f"Refusing to clean a link or reparse point: {path}")
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"Cleanup path is not a directory: {path}")


def _windows_open_bound(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    directory: bool,
) -> tuple[Any, Any, Any]:
    from ctypes import wintypes

    class FileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    status = path.lstat()
    if _directory_identity(status) != expected_identity:
        raise RuntimeError(f"Entry changed before cleanup: {path}")
    if _is_link_or_reparse(status):
        raise RuntimeError(f"Refusing to clean a link or reparse point: {path}")
    if stat.S_ISDIR(status.st_mode) != directory:
        raise RuntimeError(f"Entry type changed before cleanup: {path}")
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
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(FileInformation),
    ]
    get_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        str(path),
        0x1 if directory else 0x00010080,
        0x1 | 0x2,  # FILE_SHARE_READ | FILE_SHARE_WRITE
        None,
        3,  # OPEN_EXISTING
        (0x02000000 if directory else 0) | 0x00200000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        information = FileInformation()
        if not get_information(handle, ctypes.byref(information)):
            raise ctypes.WinError(ctypes.get_last_error())
        attributes = information.dwFileAttributes
        if bool(attributes & 0x10) != directory or attributes & 0x400:
            raise RuntimeError(
                f"Cleanup handle has unexpected attributes: {path}"
            )
        handle_identity = (
            information.dwVolumeSerialNumber,
            (information.nFileIndexHigh << 32) | information.nFileIndexLow,
        )
        path_identity = (status.st_dev & 0xFFFFFFFF, status.st_ino)
        if handle_identity != path_identity:
            raise RuntimeError(f"Directory changed while opening cleanup handle: {path}")
    except Exception:
        if not close_handle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
        raise
    return kernel32, handle, status


def _windows_close(kernel32: Any, handle: Any) -> None:
    if not kernel32.CloseHandle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_entries(
    kernel32: Any,
    handle: Any,
    path: Path,
    status: Any,
) -> list[tuple[str, int, tuple[int, int]]]:
    from ctypes import wintypes

    class FileIdBothDirectoryInformation(ctypes.Structure):
        _fields_ = [
            ("NextEntryOffset", wintypes.DWORD),
            ("FileIndex", wintypes.DWORD),
            ("CreationTime", ctypes.c_longlong),
            ("LastAccessTime", ctypes.c_longlong),
            ("LastWriteTime", ctypes.c_longlong),
            ("ChangeTime", ctypes.c_longlong),
            ("EndOfFile", ctypes.c_longlong),
            ("AllocationSize", ctypes.c_longlong),
            ("FileAttributes", wintypes.DWORD),
            ("FileNameLength", wintypes.DWORD),
            ("EaSize", wintypes.DWORD),
            ("ShortNameLength", ctypes.c_byte),
            ("ShortName", wintypes.WCHAR * 12),
            ("FileId", ctypes.c_longlong),
            ("FileName", wintypes.WCHAR),
        ]

    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    get_information.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(65536)
    results = []
    while True:
        if not get_information(handle, 10, buffer, len(buffer)):
            error = ctypes.get_last_error()
            if error == 18:  # ERROR_NO_MORE_FILES
                return results
            raise ctypes.WinError(error)
        offset = 0
        while True:
            information = FileIdBothDirectoryInformation.from_buffer(
                buffer,
                offset,
            )
            name = ctypes.wstring_at(
                ctypes.addressof(buffer)
                + offset
                + FileIdBothDirectoryInformation.FileName.offset,
                information.FileNameLength // 2,
            )
            if name not in {".", ".."}:
                results.append(
                    (
                        name,
                        information.FileAttributes,
                        (
                            status.st_dev,
                            information.FileId & 0xFFFFFFFFFFFFFFFF,
                        ),
                    )
                )
            if not information.NextEntryOffset:
                break
            offset += information.NextEntryOffset


def _windows_unlink(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    from ctypes import wintypes

    class FileDispositionInformation(ctypes.Structure):
        _fields_ = [("DeleteFile", wintypes.BOOL)]

    kernel32, handle, _ = _windows_open_bound(
        path,
        expected_identity,
        directory=False,
    )
    try:
        set_information = kernel32.SetFileInformationByHandle
        set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        disposition = FileDispositionInformation(True)
        if not set_information(
            handle,
            4,  # FileDispositionInfo
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _windows_close(kernel32, handle)


def _windows_rmtree(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    kernel32, handle, status = _windows_open_bound(
        path,
        expected_identity,
        directory=True,
    )
    try:
        for name, attributes, child_identity in _windows_entries(
            kernel32,
            handle,
            path,
            status,
        ):
            child = path / name
            if attributes & 0x400:
                raise RuntimeError(
                    f"Refusing to clean a link or reparse point: {child}"
                )
            if attributes & 0x10:
                _windows_rmtree(child, child_identity)
            else:
                _windows_unlink(child, child_identity)
    finally:
        _windows_close(kernel32, handle)

    os.rmdir(path)


def _safe_rmtree(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    status = path.lstat()
    _validate_directory(path, status, expected_identity)
    if os.name == "nt":
        _windows_rmtree(path, expected_identity)
        return
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise RuntimeError("Safe recursive directory cleanup is unavailable")
    shutil.rmtree(path)


def _prepare_work_root(
    store: RunStore,
) -> tuple[Path, tuple[int, int]]:
    work_root = store.root / "work"
    try:
        status = work_root.lstat()
    except FileNotFoundError:
        work_root.mkdir()
        status = work_root.lstat()
    if _is_link_or_reparse(status):
        raise RuntimeError(f"Run work directory is a link or reparse point: {work_root}")
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"Run work path is not a directory: {work_root}")
    resolved = work_root.resolve()
    if not resolved.is_relative_to(store.root):
        raise RuntimeError(f"Run work directory is outside run directory: {work_root}")
    if any(work_root.iterdir()):
        raise RuntimeError(f"Run work directory is not empty: {work_root}")
    return resolved, _directory_identity(status)


def execute_legacy(store: RunStore) -> dict[str, Any]:
    manifest = store.read_json("job_manifest.json")
    validate_schema_version(manifest)
    sources = [_source_path(store, page_id) for page_id in manifest["pages"]]
    options = manifest["options"]
    slide_size = options["slide_size"]
    combine_original = (
        manifest["input"].get("type") == "pdf"
        and manifest["input"].get("page_ratios_equal") is True
    )
    original_aspect_ratio = manifest["input"].get("page_aspect_ratio")
    output_path = options["output_path"]
    if output_path is None:
        output_path = str(store.root / "final" / "output.pptx")

    work_root, work_identity = _prepare_work_root(store)
    module = importlib.import_module("image_to_ppt")
    with redirect_stdout(sys.stderr):
        if len(sources) == 1 and slide_size == "both":
            result = module.convert_variants(
                sources[0],
                output_path=output_path,
                lang=options["lang"],
                _work_root=work_root,
                _resource_isolation=True,
            )
        elif len(sources) == 1:
            result = {
                slide_size: module.convert(
                    sources[0],
                    output_path=output_path,
                    lang=options["lang"],
                    slide_size=slide_size,
                    _work_root=work_root,
                    _resource_isolation=True,
                )
            }
        elif slide_size == "both":
            kwargs = {
                "output_path": output_path,
                "lang": options["lang"],
                "_work_root": work_root,
                "_resource_isolation": True,
            }
            if combine_original:
                kwargs["combine_original"] = True
                if original_aspect_ratio is not None:
                    kwargs["original_aspect_ratio"] = original_aspect_ratio
            result = module.convert_batch_variants(
                sources,
                **kwargs,
            )
        elif slide_size == "original":
            kwargs = {
                "output_path": output_path,
                "lang": options["lang"],
                "include_widescreen": False,
                "_work_root": work_root,
                "_resource_isolation": True,
            }
            if combine_original:
                kwargs["combine_original"] = True
                if original_aspect_ratio is not None:
                    kwargs["original_aspect_ratio"] = original_aspect_ratio
            result = module.convert_batch_variants(
                sources,
                **kwargs,
            )
        else:
            result = {
                "16:9": module.convert_batch(
                    sources,
                    output_path=output_path,
                    lang=options["lang"],
                    _work_root=work_root,
                    _resource_isolation=True,
                )
            }
    result = _absolute_outputs(result)
    _safe_rmtree(work_root, work_identity)
    return result
