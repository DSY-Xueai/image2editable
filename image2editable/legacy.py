from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stdout
import ctypes
import errno
import hashlib
import importlib
import io
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
from typing import Any
import uuid

from PIL import Image, ImageChops, ImageDraw, ImageOps
from pptx import Presentation

from image2editable.contracts import utc_now, validate_schema_version
from image2editable.component_contracts import (
    MAX_REPAIR_ROUNDS,
    validate_component_graph,
    validate_component_repair_state,
)
from image2editable.component_repair import (
    COMPONENT_STATE_NAME,
    EVIDENCE_NAMES,
    advance_component_repair,
    build_component_agent_request,
    execute_component_action_round,
    initialize_component_repair_state,
    record_component_execution,
    record_component_quality,
    record_next_component_request,
    require_parent_fallback,
    reject_recoverable_component_plan,
    record_parent_fallback_execution,
    record_parent_fallback_quality,
    _decode_binary_grayscale_png,
    _read_bound_file,
    _require_held_execution_lease,
    _snapshot_directory_chain,
    _validate_presentation_manifest,
    _write_exclusive,
)
from image2editable.inputs import sha256_file
from image2editable.store import RunStore
from image2editable.execution import ExecutionLease
from scripts.psd_assemble import assemble_psd


_LOGGER = logging.getLogger(__name__)


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


def _page_ocr_rotation(store: RunStore, page_id: str) -> int:
    request = store.read_json(Path("pages") / page_id / "page_request.json")
    validate_schema_version(request)
    if request.get("source_type") != "pdf":
        return 0
    render = request.get("render")
    rotation = render.get("rotation") if isinstance(render, dict) else None
    if type(rotation) is not int or rotation not in {0, 90, 180, 270}:
        raise ValueError(f"{page_id}: PDF render rotation is invalid")
    return rotation


def _native_pdf_analysis(store: RunStore, page_id: str) -> dict | None:
    try:
        request = store.read_json(Path("pages") / page_id / "page_request.json")
    except FileNotFoundError:
        return None
    analysis = request.get("pdf_analysis")
    render = request.get("render")
    if (
        request.get("source_type") != "pdf"
        or not isinstance(analysis, dict)
        or analysis.get("classification") not in {"native", "hybrid"}
        or analysis.get("requires_visual") is not False
        or not isinstance(analysis.get("objects"), list)
        or not isinstance(render, dict)
        or render.get("rotation") != 0
    ):
        return None
    for item in analysis["objects"]:
        if not isinstance(item, dict):
            return None
        if item.get("type") == "text":
            matrix = item.get("matrix")
            if (
                not isinstance(matrix, list)
                or len(matrix) != 6
                or abs(float(matrix[1])) > 1e-6
                or abs(float(matrix[2])) > 1e-6
            ):
                return None
        if item.get("type") == "image":
            transform = item.get("transform")
            if (
                not isinstance(transform, list)
                or len(transform) != 6
                or abs(float(transform[0])) <= 1e-6
                or abs(float(transform[3])) <= 1e-6
                or abs(float(transform[1])) > 1e-4
                or abs(float(transform[2])) > 1e-4
            ):
                return None
    return analysis


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


def _windows_handle_information(
    kernel32: Any, handle: Any
) -> tuple[int, int, tuple[int, int]]:
    from ctypes import wintypes

    class FileInformation(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD), ("creation", wintypes.FILETIME),
            ("access", wintypes.FILETIME), ("write", wintypes.FILETIME),
            ("volume", wintypes.DWORD), ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD),
        ]

    information = FileInformation()
    query = kernel32.GetFileInformationByHandle
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(FileInformation)]
    query.restype = wintypes.BOOL
    if not query(handle, ctypes.byref(information)):
        raise ctypes.WinError(ctypes.get_last_error())
    identity = (
        information.volume,
        (information.index_high << 32) | information.index_low,
    )
    return information.attributes, information.links, identity


def _windows_open_bound(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    directory: bool,
    desired_access: int | None = None,
    share_mode: int | None = None,
) -> tuple[Any, Any, Any]:
    from ctypes import wintypes

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
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        str(path),
        desired_access if desired_access is not None else (
            0xA1 if directory else 0x00010080
        ),
        share_mode if share_mode is not None else 0x1 | 0x2,
        None,
        3,  # OPEN_EXISTING
        (0x02000000 if directory else 0) | 0x00200000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        attributes, _, handle_identity = _windows_handle_information(
            kernel32, handle
        )
        if bool(attributes & 0x10) != directory or attributes & 0x400:
            raise RuntimeError(
                f"Cleanup handle has unexpected attributes: {path}"
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


def _rename_directory_exclusive(
    staging: Path,
    final: Path,
    expected_identity: tuple[int, int],
) -> None:
    if staging.parent != final.parent:
        raise RuntimeError("presentation publication directories differ")
    _validate_directory(staging, staging.lstat(), expected_identity)
    if os.name == "nt":
        try:
            os.rename(staging, final)
        except OSError as error:
            if final.exists() or final.is_symlink():
                raise RuntimeError(
                    "presentation assets already published"
                ) from error
            raise
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise RuntimeError("exclusive directory publication is unavailable")
        rename.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        )
        result = rename(
            -100, os.fsencode(staging), -100, os.fsencode(final), 1
        )
    elif sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise RuntimeError("exclusive directory publication is unavailable")
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        result = rename(os.fsencode(staging), os.fsencode(final), 4)
    else:
        raise RuntimeError("exclusive directory publication is unavailable")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise RuntimeError("presentation assets already published")
    raise OSError(error_number, os.strerror(error_number), str(staging), str(final))


def _rename_file_exclusive_at(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
) -> None:
    if os.name == "nt":
        raise RuntimeError("exclusive publication is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        symbol, exclusive_flag = "renameat2", 1
    elif sys.platform == "darwin":
        symbol, exclusive_flag = "renameatx_np", 4
    else:
        raise RuntimeError("exclusive publication is unavailable")
    rename = getattr(libc, symbol, None)
    if rename is None:
        raise RuntimeError("exclusive publication is unavailable")
    rename.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
    )
    result = rename(
        source_fd, os.fsencode(source_name), destination_fd,
        os.fsencode(destination_name), exclusive_flag,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise RuntimeError("publication destination already exists")
    raise OSError(error_number, os.strerror(error_number))


def _rename_windows_staging(source_handle: Any, parent_handle: Any, final_name: str) -> None:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if _windows_handle_information(kernel32, source_handle)[1] != 1:
        raise RuntimeError("background responsibility staging has unsafe links")

    class RenameInformation(ctypes.Structure):
        _fields_ = [
            ("replace", wintypes.BOOL),
            ("root", wintypes.HANDLE),
            ("name_length", wintypes.DWORD),
            ("name", wintypes.WCHAR * (len(final_name) + 1)),
        ]

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("status", wintypes.LPVOID), ("information", ctypes.c_size_t)]

    information = RenameInformation()
    information.root = parent_handle
    information.name_length = len(final_name.encode("utf-16-le"))
    information.name = final_name
    rename = ctypes.WinDLL("ntdll").NtSetInformationFile
    rename.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.LPVOID, wintypes.ULONG, ctypes.c_int]
    rename.restype = ctypes.c_long
    status = rename(
        wintypes.HANDLE(source_handle), ctypes.byref(IoStatusBlock()), ctypes.byref(information),
        ctypes.sizeof(information), 10,
    )
    if status:
        if status & 0xFFFFFFFF == 0xC0000035:
            raise RuntimeError("publication destination already exists")
        raise OSError(f"background responsibility rename failed: {status:#x}")


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


def initialize_legacy_page(
    store: RunStore, page_id: str, *, _lease: ExecutionLease,
    performance_trace=None, ocr_result=None, ocr_worker_pool=None,
    visual_worker_pool=None, visual_worker_pool_factory=None,
) -> dict[str, Any]:
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    state_path = reconstruction / "component_state.json"
    if state_path.is_file():
        return {"status": "already_initialized", "page_id": page_id}
    manifest = store.read_json("job_manifest.json")
    native_analysis = _native_pdf_analysis(store, page_id)
    if native_analysis is not None:
        reconstruction.mkdir(parents=True, exist_ok=True)
        native_path = reconstruction / "native_page.json"
        store.write_json(
            native_path.relative_to(store.root),
            {
                "schema_version": 1,
                "page_id": page_id,
                "analysis": native_analysis,
            },
        )
        store.write_json(
            state_path.relative_to(store.root),
            {
                "schema_version": 1,
                "page_id": page_id,
                "route": "pdf_native",
                "phase": "ready_for_assembly",
                "status": "ready_for_assembly",
                "native_page_ref": {
                    "path": native_path.relative_to(store.root).as_posix(),
                    "sha256": sha256_file(native_path),
                },
            },
        )
        return {"status": "initialized", "page_id": page_id}
    source = _source_path(store, page_id)
    performance = (
        performance_trace.span(
            "visual_prepare", page_id=page_id, operation_count=1
        )
        if performance_trace is not None
        else nullcontext()
    )
    pipeline_mode = manifest.get("options", {}).get("pipeline_mode", "strict")
    prepare_kwargs = {
        "lang": manifest["options"]["lang"],
        "resource_isolation": True,
        "ocr_rotation": _page_ocr_rotation(store, page_id),
    }
    if pipeline_mode == "fast":
        prepare_kwargs.update({
            "pipeline_mode": pipeline_mode,
            "source_kind": {
                "images": "image", "pdf": "pdf", "pptx": "pptx",
            }.get(manifest.get("input", {}).get("type"), "image"),
        })
    if ocr_result is not None:
        prepare_kwargs["ocr_result"] = ocr_result
    if ocr_worker_pool is not None:
        prepare_kwargs["ocr_worker_pool"] = ocr_worker_pool
    if visual_worker_pool is not None:
        prepare_kwargs["visual_worker_pool"] = visual_worker_pool
    if visual_worker_pool_factory is not None:
        prepare_kwargs["visual_worker_pool_factory"] = visual_worker_pool_factory
    if ocr_worker_pool is not None or visual_worker_pool is not None:
        prepare_kwargs["performance_trace"] = performance_trace
        prepare_kwargs["page_id"] = page_id
    with performance:
        prepared = importlib.import_module("image_to_ppt").prepare_component_layers(
            source,
            reconstruction / "initial",
            **prepare_kwargs,
        )
    session = _build_initial_page_session(
        store, page_id, prepared, reconstruction
    )
    request_path = build_component_agent_request(
        session, repair_round=1, _lease=_lease,
    )
    initialize_component_repair_state(
        store, page_id, request_path=request_path,
        initial_component_count=prepared["initial_component_count"],
        _lease=_lease,
    )
    if pipeline_mode == "fast":
        _record_deterministic_fast_plan(
            store, page_id, request_path, reconstruction, _lease=_lease
        )
    return {"status": "initialized", "page_id": page_id}


def _record_deterministic_fast_plan(
    store: RunStore, page_id: str, request_path: Path, reconstruction: Path,
    *, _lease: ExecutionLease,
) -> None:
    """Record a deterministic Fast plan without waiting for an Agent."""
    from image2editable.component_contracts import validate_component_plan
    from image2editable.component_repair import (
        load_component_agent_graph,
        load_component_agent_request,
    )
    from image2editable.host_agent import _record_plan_reference, _request_sha256

    # Advance request_published to awaiting_plan through the durable state API.
    advance_component_repair(store, page_id, _lease=_lease)
    request = load_component_agent_request(request_path)
    graph = load_component_agent_graph(request_path)
    request_sha256 = _request_sha256(request)
    strict_escalation = (
        request["repair_round"] == 2
        and (reconstruction / "strict-escalation-02" / "graph"
             / "component-graph.json").is_file()
    )
    absorbed_ids: set[str] = set()
    if request["repair_round"] == 2:
        absorbed_path = (
            reconstruction / "strict-escalation-02" / "graph"
            / "absorbed-component-ids.json"
        )
        if absorbed_path.is_file():
            metadata = json.loads(absorbed_path.read_text(encoding="utf-8"))
            component_ids = metadata.get("component_ids") if isinstance(metadata, dict) else None
            if (
                not isinstance(metadata, dict)
                or metadata.get("schema_version") != 1
                or not isinstance(component_ids, list)
                or any(type(component_id) is not str or not component_id for component_id in component_ids)
                or component_ids != sorted(set(component_ids))
                or not set(component_ids).issubset(request["candidate_ids"])
            ):
                raise ValueError("strict absorbed component metadata is invalid")
            absorbed_ids = set(component_ids)
    presentation_ref = request.get("evidence", {}).get("presentation-manifest.json")
    if presentation_ref is not None:
        _, payload = _request_evidence_ref(store, request_path, presentation_ref)
        presentation = json.loads(payload.decode("utf-8"))
        for component in presentation["components"]:
            if component["component_id"] not in request["candidate_ids"]:
                continue
            _, mask_payload = _load_legacy_ref(store, component["ownership_mask"])
            with Image.open(io.BytesIO(mask_payload)) as mask:
                if mask.getbbox() is None:
                    absorbed_ids.add(component["component_id"])
    active_candidate_ids = [
        object_id for object_id in request["candidate_ids"]
        if object_id not in absorbed_ids
    ]
    actions = [
        {
            "action": "discard" if object_id in absorbed_ids else "accept",
            "object_ids": [object_id],
            "parameters": (
                {"preserve_mask": True}
                if strict_escalation and object_id not in absorbed_ids
                else {}
            ),
            "confidence": 1.0,
            "evidence": ["deterministic page route"],
        }
        for object_id in request["candidate_ids"]
    ]
    quality_ref = request.get("evidence", {}).get("quality-report.json")
    if request["repair_round"] > 1 and quality_ref is not None:
        from image2editable.component_repair import _page_residual_owner_ids

        _, quality_payload = _request_evidence_ref(store, request_path, quality_ref)
        previous_quality = json.loads(quality_payload.decode("utf-8"))
        if "unexplained_visual_residual" in previous_quality.get("report", {}).get("violations", []):
            repair_graph = {
                **graph,
                "nodes": [node for node in graph["nodes"] if node["id"] in active_candidate_ids],
            }
            residual_ids = _page_residual_owner_ids(
                store, quality=previous_quality, graph=repair_graph,
                graph_root=request_path.parent,
            )
            actions.extend({
                "action": "absorb_residual", "object_ids": [object_id],
                "parameters": {}, "confidence": 1.0,
                "evidence": ["signed residual adjacent to component"],
            } for object_id in sorted(residual_ids))
    if active_candidate_ids:
        actions.append({
            "action": "rebuild_background",
            "object_ids": active_candidate_ids,
            "parameters": {"margin_ratio": 0.01},
            "confidence": 1.0,
            "evidence": ["deterministic page route"],
        })
    plan = {
        "schema_version": 1,
        "kind": "component_plan",
        "page_id": page_id,
        "provider": "host",
        "repair_round": request["repair_round"],
        "request_sha256": request_sha256,
        "actions": actions,
    }
    validate_component_plan(plan, request=request, graph=graph)
    payload = json.dumps(
        plan, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8") + b"\n"
    plan_path = reconstruction / (
        f"deterministic-component-plan-{page_id}-"
        f"{request['repair_round']:02d}-{request_sha256}.json"
    )
    if not plan_path.exists():
        _write_exclusive(plan_path, payload, reconstruction)
    elif plan_path.read_bytes() != payload:
        raise RuntimeError("A different deterministic component plan exists")
    _record_plan_reference(store, request, plan_path)
def _build_initial_page_session(
    store: RunStore, page_id: str, prepared: dict, reconstruction: Path
) -> dict:
    with Image.open(prepared["original_image_path"]) as image:
        page_size = image.size
    text_items = _component_text_items(prepared.get("text_items", []), page_size)
    masks = prepared["_element_mask_paths"]
    semantic_masks = prepared.get("_semantic_mask_paths")
    components = prepared["components"]
    if len(masks) != len(components):
        raise ValueError("prepared component and mask counts differ")
    if semantic_masks is not None and len(semantic_masks) != len(components):
        raise ValueError("prepared semantic mask count differs from component count")
    _ensure_component_disk_reserve(
        reconstruction,
        Path(prepared["original_image_path"]),
        node_count=(
            len(components) * (2 if semantic_masks is not None else 1)
            + len(text_items)
        ),
        repair_round=1,
    )
    evidence_root = reconstruction / "evidence-source"
    evidence_root.mkdir(parents=True, exist_ok=False)
    masks_root = evidence_root / "masks"
    masks_root.mkdir()
    source_target = evidence_root / "source.png"
    shutil.copyfile(prepared["original_image_path"], source_target)
    evidence = {"source.png": source_target}
    if prepared.get("_prepared_schema_version", 1) >= 5:
        unexplained = evidence_root / "unexplained-mask.png"
        shutil.copyfile(prepared["_foreground_evidence_mask_path"], unexplained)
        evidence["unexplained-mask.png"] = unexplained

    nodes = []
    for index, (mask_source, component) in enumerate(
        zip(masks, components, strict=True), start=1
    ):
        component_id = f"component_{index:04d}"
        if semantic_masks is None:
            mask_nodes = ((component_id, "parent", None, "pending", mask_source),)
        else:
            parent_id = f"parent_{index:04d}"
            mask_nodes = (
                (parent_id, "parent", None, "inactive", semantic_masks[index - 1]),
                (component_id, "child", parent_id, "pending", mask_source),
            )
        for node_id, kind, parent_id, state, node_mask_source in mask_nodes:
            mask_target = masks_root / f"{node_id}.png"
            shutil.copyfile(node_mask_source, mask_target)
            with Image.open(mask_target) as image:
                if image.size != page_size:
                    raise ValueError(
                        f"prepared component mask dimensions differ: {node_id}"
                    )
                grayscale = image.convert("L")
                try:
                    bbox = grayscale.getbbox()
                finally:
                    grayscale.close()
            if bbox is None:
                raise ValueError(f"prepared component mask is empty: {node_id}")
            left, top, right, bottom = bbox
            nodes.append({
                "id": node_id, "kind": kind, "parent_id": parent_id,
                "state": state, "mask": f"masks/{mask_target.name}",
                "mask_sha256": sha256_file(mask_target),
                "bbox": [left, top, right, bottom],
                "z_index": component.get("z_index", index - 1), "text_ids": [],
            })
    for index, item in enumerate(text_items, start=1):
        text_id = item["id"]
        mask_target = masks_root / f"{text_id}.png"
        mask = Image.new("L", page_size, 0)
        try:
            left, top, right, bottom = item["box"]
            ImageDraw.Draw(mask).rectangle(
                (left, top, right - 1, bottom - 1),
                fill=255,
            )
            mask.save(mask_target)
        finally:
            mask.close()
        nodes.append({
            "id": text_id, "kind": "text", "parent_id": None,
            "state": "frozen", "mask": f"masks/{mask_target.name}",
            "mask_sha256": sha256_file(mask_target),
            "bbox": item["box"], "z_index": len(components) + index - 1,
            "text_ids": [],
        })
    graph_path = evidence_root / "component-graph.json"
    graph_path.write_text(
        json.dumps({"nodes": nodes}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evidence["component-graph.json"] = graph_path
    quality_path = evidence_root / "quality-report.json"
    initial_diagnostics = prepared.get("initial_diagnostics", [])
    quality_path.write_text(
        json.dumps({
            "schema_version": 1,
            "phase": "initial_layers",
            "text_items": text_items,
            "initial_diagnostics": initial_diagnostics,
            "violations": (
                ["unowned_raster_text"] if initial_diagnostics else []
            ),
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    evidence["quality-report.json"] = quality_path
    presentation_manifest = _build_presentation_assets(
        store,
        source_path=source_target,
        text_clean_path=Path(prepared.get(
            "_text_clean_path", prepared["original_image_path"]
        )),
        graph_path=graph_path,
        output_dir=evidence_root,
    )
    evidence["presentation-manifest.json"] = presentation_manifest
    evidence.update(
        _render_component_evidence(
            source_path=source_target,
            graph={"nodes": nodes},
            text_mask_path=Path(prepared["_text_mask_path"]),
            background_path=Path(prepared["background_original_path"]),
            presentation_manifest_path=presentation_manifest,
            run_root=store.root,
            reconstruction=reconstruction,
            graph_sha256=sha256_file(graph_path),
            output_dir=evidence_root,
            text_items=text_items,
        )
    )
    expected_evidence = (
        set(EVIDENCE_NAMES)
        if prepared.get("_prepared_schema_version", 1) >= 5
        else set(EVIDENCE_NAMES) - {"unexplained-mask.png"}
    )
    if set(evidence) != expected_evidence:
        raise RuntimeError("legacy component evidence set is incomplete")
    return {
        "page_id": page_id,
        "provider": store.read_json("job_manifest.json")["options"]["agent_provider"],
        "reconstruction_dir": reconstruction,
        "evidence": evidence,
    }


def _component_text_records(items: object, page_size: tuple[int, int]) -> list[dict]:
    if not isinstance(items, list):
        return []
    width, height = page_size
    normalized = []
    used_ids = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        box = item.get("box")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(
                type(value) not in {int, float} or not math.isfinite(value)
                for value in box
            )
        ):
            continue
        x, y, box_width, box_height = box
        if (
            not item["text"].strip()
            or box_width <= 0
            or box_height <= 0
            or x >= width
            or y >= height
            or x + box_width <= 0
            or y + box_height <= 0
        ):
            continue
        left = max(0, min(width - 1, int(x)))
        top = max(0, min(height - 1, int(y)))
        right = max(left + 1, min(width, math.ceil(x + box_width)))
        bottom = max(top + 1, min(height, math.ceil(y + box_height)))
        component_id = item.get(
            "_component_id", f"text_{len(normalized) + 1:04d}"
        )
        if (
            type(component_id) is not str
            or not component_id.startswith("text_")
            or not component_id[5:].isascii()
            or not component_id[5:].isdigit()
            or component_id in used_ids
        ):
            raise ValueError("component text id is invalid")
        used_ids.add(component_id)
        rotation = item.get("rotation", 0)
        if type(rotation) is not int or rotation not in {0, 90, 180, 270}:
            raise ValueError("component text rotation is invalid")
        normalized_item = {
            "id": component_id,
            "text": item["text"],
            "box": [left, top, right, bottom],
        }
        if rotation:
            normalized_item["rotation"] = rotation
        normalized.append({
            "raw": item,
            "normalized": normalized_item,
        })
    return normalized


def _component_text_items(items: object, page_size: tuple[int, int]) -> list[dict]:
    return [
        record["normalized"]
        for record in _component_text_records(items, page_size)
    ]


def _ensure_component_disk_reserve(
    reconstruction: Path,
    source_path: Path,
    *,
    node_count: int,
    repair_round: int,
) -> None:
    with Image.open(source_path) as image:
        width, height = image.size
    pixels = width * height
    remaining_rounds = MAX_REPAIR_ROUNDS - repair_round + 1
    color_files = 16 * 4 * pixels
    mask_files = max(1, node_count) * 6 * 2 * pixels
    metadata = 8 * 1024 * 1024
    safety_margin = max(256 * 1024 * 1024, 8 * pixels)
    required = remaining_rounds * (color_files + mask_files + metadata)
    required += safety_margin
    if shutil.disk_usage(reconstruction).free < required:
        raise RuntimeError(
            "component page disk reserve is insufficient before page artifact write"
        )


def _build_presentation_assets(
    store: RunStore,
    *,
    source_path: Path,
    text_clean_path: Path,
    text_mask_path: Path | None = None,
    graph_path: Path,
    output_dir: Path,
    frozen_components: dict[str, dict] | None = None,
) -> Path:
    import numpy as np

    from image2editable.component_quality import resolve_visual_mask_ownership
    from scripts.component_underlay import build_presentation_layer

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB")).copy()
    with Image.open(text_clean_path) as image:
        text_clean = np.asarray(image.convert("RGB")).copy()
    if source.shape != text_clean.shape:
        raise ValueError("presentation text-clean dimensions differ")

    masks = {}
    for node in graph["nodes"]:
        mask_path = graph_path.parent / Path(node["mask"])
        if sha256_file(mask_path) != node["mask_sha256"]:
            raise ValueError("presentation graph mask sha256 mismatch")
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        if mask.shape != source.shape[:2]:
            raise ValueError("presentation graph mask dimensions differ")
        masks[node["id"]] = mask

    text_mask = np.zeros(source.shape[:2], dtype=bool)
    text_items = []
    for node in graph["nodes"]:
        if node["kind"] == "text" and node["state"] == "frozen":
            text_mask |= masks[node["id"]]
            left, top, right, bottom = node["bbox"]
            text_items.append({
                "box": [left, top, right - left, bottom - top],
                "component_id": node["id"],
            })
    if text_mask_path is not None:
        with Image.open(text_mask_path) as image:
            text_mask = np.asarray(image.convert("L")) > 0
        if text_mask.shape != source.shape[:2]:
            raise ValueError("presentation text mask dimensions differ")
    active_nodes = _active_visual_nodes(graph)
    frozen_components = frozen_components or {}
    if set(frozen_components) - {node["id"] for node in active_nodes}:
        raise ValueError("frozen presentation component is missing")
    text_owners = {
        text_id: component_index
        for component_index, node in enumerate(active_nodes)
        for text_id in node["text_ids"]
    }
    assigned_masks = _assign_text_regions_to_component_masks(
        [masks[node["id"]] for node in active_nodes],
        text_mask,
        text_items,
        text_owner_indices=[
            text_owners.get(item["component_id"])
            for item in text_items
        ],
    )
    frozen_ownership = {}
    frozen_coverage = np.zeros(source.shape[:2], dtype=bool) if frozen_components else None
    for component_id, record in frozen_components.items():
        _, payload = _load_legacy_ref(
            store, record["ownership_mask"],
            max_bytes=max(1024 * 1024, source.shape[0] * source.shape[1] * 2),
        )
        owned = _decode_binary_grayscale_png(
            payload, source.shape[:2], label="frozen presentation ownership",
        )
        frozen_ownership[component_id] = owned
        frozen_coverage |= owned
    ownership_masks = resolve_visual_mask_ownership(
        active_nodes,
        [
            frozen_ownership[node["id"]]
            if node["id"] in frozen_ownership else mask & ~frozen_coverage
            for node, mask in zip(active_nodes, assigned_masks, strict=True)
        ] if frozen_components else assigned_masks,
    )
    assigned_by_id = {
        node["id"]: mask
        for node, mask in zip(active_nodes, assigned_masks, strict=True)
    }

    final_dir = output_dir / "presentation-assets"
    staging = output_dir / f".pa-tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    staging_identity = _directory_identity(staging.lstat())
    try:
        components_by_id = {}
        max_asset_bytes = max(
            1024 * 1024, source.shape[0] * source.shape[1] * 8
        )
        indexed_layers = [
            (index, node, ownership)
            for index, (node, ownership) in enumerate(
                zip(active_nodes, ownership_masks, strict=True), start=1
            )
        ]
        groups = {}
        for item in indexed_layers:
            groups.setdefault(int(item[1]["z_index"]), []).append(item)
        higher = np.zeros(source.shape[:2], dtype=bool)
        for z_index in sorted(groups, reverse=True):
            group = groups[z_index]
            for index, node, ownership in group:
                if node["id"] in frozen_components:
                    components_by_id[node["id"]] = frozen_components[node["id"]]
                    continue
                semantic = assigned_by_id[node["id"]]
                if node["parent_id"] is not None:
                    semantic = masks[node["parent_id"]] | semantic
                if np.any(ownership):
                    layer = build_presentation_layer(
                        source_rgb=source,
                        text_clean_rgb=text_clean,
                        ownership_mask=ownership,
                        semantic_mask=semantic,
                        higher_layer_mask=higher,
                        text_mask=text_mask,
                    )
                else:
                    empty = np.zeros(source.shape[:2], dtype=bool)
                    layer = {
                        "rgb": np.zeros_like(source),
                        "ownership_mask": empty,
                        "presentation_alpha_mask": empty.copy(),
                        "generated_underlay_mask": empty.copy(),
                        "metrics": {
                            "boundary_color_mae": 0.0,
                            "gradient_jump_p95": 0.0,
                            "added_high_frequency_pixels": 0.0,
                        },
                    }
                filenames = {
                    "rgba": f"{index:04d}-rgba.png",
                    "ownership_mask": f"{index:04d}-ownership-mask.png",
                    "presentation_alpha_mask": (
                        f"{index:04d}-presentation-alpha-mask.png"
                    ),
                    "generated_underlay_mask": (
                        f"{index:04d}-generated-underlay-mask.png"
                    ),
                }
                paths = {
                    name: staging / filename
                    for name, filename in filenames.items()
                }
                final_paths = {
                    name: final_dir / filename
                    for name, filename in filenames.items()
                }
                encoded_rgb = layer["rgb"].copy()
                encoded_rgb[~layer["presentation_alpha_mask"]] = 0
                rgba = Image.fromarray(np.dstack((
                    encoded_rgb,
                    layer["presentation_alpha_mask"].astype(np.uint8) * 255,
                )), mode="RGBA")
                try:
                    rgba.save(paths["rgba"])
                finally:
                    rgba.close()
                for name in (
                    "ownership_mask", "presentation_alpha_mask",
                    "generated_underlay_mask",
                ):
                    image = Image.fromarray(
                        layer[name].astype(np.uint8) * 255, mode="L"
                    )
                    try:
                        image.save(paths[name])
                    finally:
                        image.close()
                metrics = {
                    name: float(value) for name, value in layer["metrics"].items()
                }
                if any(not math.isfinite(value) for value in metrics.values()):
                    raise ValueError("presentation metrics must be finite")
                payloads = {}
                hashes = {}
                for name, path in paths.items():
                    payload = _read_bound_file(
                        path,
                        output_dir,
                        max_bytes=max_asset_bytes,
                        label="presentation staging asset",
                    )
                    payloads[name] = payload
                    hashes[name] = hashlib.sha256(payload).hexdigest()
                arrays = _decode_presentation_arrays(
                    payloads,
                    page_size=(source.shape[1], source.shape[0]),
                )
                _validate_presentation_arrays(arrays)
                component = {
                    "component_id": node["id"],
                    **{
                        name: {
                            "path": final_path.resolve().relative_to(
                                store.root.resolve()
                            ).as_posix(),
                            "sha256": hashes[name],
                        }
                        for name, final_path in final_paths.items()
                    },
                    "metrics": metrics,
                }
                components_by_id[node["id"]] = component
                del arrays, encoded_rgb, layer, payloads
            for _, _, ownership in group:
                higher |= ownership
        components = [components_by_id[node["id"]] for node in active_nodes]
        manifest = {
            "schema_version": 1,
            "source_sha256": sha256_file(source_path),
            "graph_sha256": sha256_file(graph_path),
            "components": components,
        }
        staging_manifest = staging / "presentation-manifest.json"
        with staging_manifest.open("x", encoding="utf-8") as stream:
            json.dump(
                manifest, stream, ensure_ascii=False, indent=2, sort_keys=True
            )
            stream.write("\n")
        manifest_payload = _read_bound_file(
            staging_manifest,
            output_dir,
            max_bytes=16 * 1024 * 1024,
            label="presentation staging manifest",
        )
        if json.loads(manifest_payload.decode("utf-8")) != manifest:
            raise RuntimeError("presentation staging manifest mismatch")
        _rename_directory_exclusive(staging, final_dir, staging_identity)
    except BaseException:
        if staging.exists():
            _safe_rmtree(staging, staging_identity)
        raise
    return final_dir / "presentation-manifest.json"


def _active_visual_nodes(graph: dict) -> list[dict]:
    return [
        node for node in graph["nodes"]
        if node["kind"] != "text"
        and node["state"] in {"pending", "pending_gate", "frozen"}
    ]


def _decode_presentation_arrays(
    payloads: dict[str, bytes], *, page_size: tuple[int, int]
) -> dict[str, Any]:
    import numpy as np

    arrays = {}
    for name, mode in (
        ("rgba", "RGBA"),
        ("ownership_mask", "L"),
        ("presentation_alpha_mask", "L"),
        ("generated_underlay_mask", "L"),
    ):
        with Image.open(io.BytesIO(payloads[name])) as image:
            converted = image.convert(mode)
            try:
                if converted.size != page_size:
                    raise ValueError("presentation asset dimensions differ")
                arrays[name] = np.asarray(converted).copy()
            finally:
                converted.close()
    return arrays


def _validate_presentation_arrays(arrays: dict[str, Any]) -> None:
    import numpy as np

    for name in (
        "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask",
    ):
        if not np.all((arrays[name] == 0) | (arrays[name] == 255)):
            raise ValueError("presentation asset masks must be binary")
    ownership = arrays["ownership_mask"] == 255
    alpha = arrays["presentation_alpha_mask"] == 255
    generated = arrays["generated_underlay_mask"] == 255
    if np.any(ownership & generated):
        raise ValueError(
            "presentation ownership and generated underlay masks overlap"
        )
    if not np.array_equal(
        arrays["rgba"][:, :, 3], arrays["presentation_alpha_mask"]
    ):
        raise ValueError("presentation RGBA alpha does not match alpha mask")
    if not np.array_equal(alpha, ownership | generated):
        raise ValueError("presentation asset masks do not match RGBA alpha")
    if np.any(arrays["rgba"][~alpha, :3]):
        raise ValueError("presentation transparent RGB must be zero")


def _load_presentation_assets(
    *,
    run_root: Path,
    reconstruction: Path,
    manifest_path: Path,
    source_sha256: str,
    graph_sha256: str,
    graph: dict,
    page_size: tuple[int, int],
    component_ids: list[str] | None = None,
    expected_manifest_sha256: str | None = None,
):

    manifest = _validate_presentation_manifest(
        manifest_path,
        reconstruction,
        source_sha256=source_sha256,
        graph_sha256=graph_sha256,
        run_root=run_root,
        expected_component_ids=[
            node["id"] for node in _active_visual_nodes(graph)
        ],
        expected_sha256=expected_manifest_sha256,
    )
    components_by_id = {
        component["component_id"]: component
        for component in manifest["components"]
    }
    expected_ids = [node["id"] for node in _active_visual_nodes(graph)]
    ordered_ids = expected_ids if component_ids is None else component_ids
    if sorted(ordered_ids) != sorted(expected_ids):
        raise ValueError("presentation asset load order does not match graph")
    for component_id in ordered_ids:
        component = components_by_id[component_id]
        payloads = {}
        max_bytes = max(1024 * 1024, page_size[0] * page_size[1] * 8)
        for name in (
            "rgba", "ownership_mask", "presentation_alpha_mask",
            "generated_underlay_mask",
        ):
            reference = component[name]
            path = run_root / Path(*PurePosixPath(reference["path"]).parts)
            payload = _read_bound_file(
                path,
                reconstruction,
                max_bytes=max_bytes,
                label="presentation asset",
            )
            if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
                raise RuntimeError(
                    f"presentation asset hash mismatch: "
                    f"{component['component_id']}/{name}"
                )
            payloads[name] = payload
        arrays = _decode_presentation_arrays(payloads, page_size=page_size)
        _validate_presentation_arrays(arrays)
        del payloads
        layer = {"component_id": component["component_id"], **arrays}
        yield layer
        del layer, arrays


def _composite_presentation_layers(
    background: Image.Image,
    graph: dict,
    layers,
) -> Image.Image:
    composited = background.convert("RGBA")
    try:
        active_nodes = _active_visual_nodes(graph)
        indexed = {node["id"]: index for index, node in enumerate(active_nodes)}
        ordered_nodes = sorted(
            active_nodes,
            key=lambda item: (int(item["z_index"]), indexed[item["id"]]),
        )
        layer_iterator = iter(layers)
        for node in ordered_nodes:
            try:
                layer = next(layer_iterator)
            except StopIteration as error:
                raise ValueError("presentation layer stream ended early") from error
            if layer["component_id"] != node["id"]:
                raise ValueError("presentation layer stream order does not match graph")
            overlay = Image.fromarray(layer["rgba"], mode="RGBA")
            try:
                composited.alpha_composite(overlay)
            finally:
                overlay.close()
            del layer
        try:
            next(layer_iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("presentation layer stream has extra components")
        return composited.convert("RGB")
    finally:
        composited.close()


def _render_component_evidence(
    *,
    source_path: Path,
    graph: dict,
    text_mask_path: Path,
    background_path: Path,
    presentation_manifest_path: Path,
    run_root: Path,
    reconstruction: Path,
    graph_sha256: str,
    output_dir: Path,
    text_items: list[dict],
) -> dict[str, Path]:
    with ExitStack() as images:
        def keep(image: Image.Image) -> Image.Image:
            images.callback(image.close)
            return image

        with Image.open(source_path) as image:
            source = keep(image.convert("RGB"))
        with Image.open(text_mask_path) as image:
            text_mask = keep(image.convert("L"))
        if text_mask.size != source.size:
            raise ValueError("component evidence text mask dimensions differ")
        with Image.open(background_path) as image:
            background = keep(image.convert("RGB"))
        if background.size != source.size:
            raise ValueError("component evidence background dimensions differ")
        isolation_nodes = _active_visual_nodes(graph)
        indexed = {node["id"]: index for index, node in enumerate(isolation_nodes)}
        composite_ids = [
            node["id"] for node in sorted(
                isolation_nodes,
                key=lambda item: (
                    int(item["z_index"]), indexed[item["id"]]
                ),
            )
        ]
        reconstructed = keep(
            _composite_presentation_layers(
                background,
                graph,
                _load_presentation_assets(
                    run_root=run_root,
                    reconstruction=reconstruction,
                    manifest_path=presentation_manifest_path,
                    source_sha256=sha256_file(source_path),
                    graph_sha256=graph_sha256,
                    graph=graph,
                    page_size=source.size,
                    component_ids=composite_ids,
                ),
            )
        )

        numbered = keep(source.copy())
        ownership = keep(Image.new("RGB", source.size, (24, 24, 24)))
        numbered_draw = ImageDraw.Draw(numbered)
        ownership_draw = ImageDraw.Draw(ownership)
        colors = (
            (255, 80, 80),
            (70, 180, 255),
            (90, 220, 120),
            (255, 190, 60),
            (190, 100, 255),
            (60, 220, 210),
        )
        columns = max(1, min(3, len(isolation_nodes)))
        rows = max(1, math.ceil(len(isolation_nodes) / columns))
        cell_width, cell_height, label_height = 320, 240, 24
        isolation = keep(Image.new(
            "RGBA", (columns * cell_width, rows * cell_height), (0, 0, 0, 0)
        ))
        isolation_draw = ImageDraw.Draw(isolation)
        isolation_layers = _load_presentation_assets(
            run_root=run_root,
            reconstruction=reconstruction,
            manifest_path=presentation_manifest_path,
            source_sha256=sha256_file(source_path),
            graph_sha256=graph_sha256,
            graph=graph,
            page_size=source.size,
        )
        isolation_iterator = iter(isolation_layers)
        for index, node in enumerate(isolation_nodes):
            try:
                layer = next(isolation_iterator)
            except StopIteration as error:
                raise ValueError("presentation isolation stream ended early") from error
            with ExitStack() as node_images:
                def keep_node(image: Image.Image) -> Image.Image:
                    node_images.callback(image.close)
                    return image

                mask = keep_node(Image.fromarray(layer["ownership_mask"], mode="L"))
                color = colors[int(node["z_index"]) % len(colors)]
                alpha = keep_node(mask.point(lambda value: value * 96 // 255))
                presentation = keep_node(
                    Image.fromarray(layer["rgba"], mode="RGBA")
                )
                presentation_alpha = keep_node(presentation.getchannel("A"))
                bbox = presentation_alpha.getbbox()
                cell_left = (index % columns) * cell_width
                cell_top = (index // columns) * cell_height
                isolation_draw.text(
                    (cell_left + 4, cell_top + 4), node["id"], fill="white",
                    stroke_width=2, stroke_fill="black",
                )
                if bbox is not None:
                    isolated = keep_node(presentation.crop(bbox))
                    isolated.thumbnail(
                        (cell_width - 16, cell_height - label_height - 16),
                        Image.Resampling.LANCZOS,
                    )
                    isolation.alpha_composite(
                        isolated,
                        (
                            cell_left + (cell_width - isolated.width) // 2,
                            cell_top + label_height
                            + (cell_height - label_height - isolated.height) // 2,
                        ),
                    )
                color_layer = keep_node(Image.new("RGB", source.size, color))
                numbered.paste(color_layer, (0, 0), alpha)
                ownership.paste(color_layer, (0, 0), mask)
                left, top, right, bottom = node["bbox"]
                label_at = ((left + right) // 2, (top + bottom) // 2)
                for draw in (numbered_draw, ownership_draw):
                    draw.text(
                        label_at,
                        node["id"],
                        fill="white",
                        stroke_width=2,
                        stroke_fill="black",
                        anchor="mm",
                    )
            del layer
        try:
            next(isolation_iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("presentation isolation stream has extra components")

        paths = {}
        for name, evidence_image in (
            ("numbered-masks.png", numbered),
            ("ownership.png", ownership),
            ("component-isolation.png", isolation),
        ):
            path = output_dir / name
            evidence_image.save(path)
            paths[name] = path

        ocr_overlay = keep(source.copy())
        text_color = keep(Image.new("RGB", source.size, (255, 225, 0)))
        text_alpha = keep(text_mask.point(lambda value: value * 112 // 255))
        ocr_overlay.paste(text_color, (0, 0), text_alpha)
        ocr_draw = ImageDraw.Draw(ocr_overlay)
        ocr_draw.text(
            (4, 4),
            "OCR/TEXT MASK",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        for item in text_items:
            left, top, right, bottom = item["box"]
            ocr_draw.rectangle(
                (left, top, right, bottom), outline=(255, 225, 0), width=2
            )
            ocr_draw.text(
                (left, max(0, top - 12)),
                item["id"],
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
        ocr_path = output_dir / "ocr-overlay.png"
        ocr_overlay.save(ocr_path)
        paths["ocr-overlay.png"] = ocr_path

        raw_difference = keep(ImageChops.difference(source, reconstructed))
        difference = keep(ImageOps.autocontrast(raw_difference))
        for name, evidence_image in (
            ("reconstructed.png", reconstructed),
            ("difference.png", difference),
        ):
            path = output_dir / name
            evidence_image.save(path)
            paths[name] = path
        return paths


def advance_legacy_page(
    store: RunStore, page_id: str, *, _lease: ExecutionLease,
    performance_trace=None, visual_worker_pool=None,
    visual_worker_pool_factory=None,
) -> dict[str, Any]:
    state_path = Path("pages") / page_id / "reconstruction" / COMPONENT_STATE_NAME
    state = (
        store.read_json(state_path)
        if (store.root / state_path).is_file()
        else None
    )
    if isinstance(state, dict) and state.get("route") == "pdf_native":
        if (
            state.get("page_id") != page_id
            or state.get("phase") != "ready_for_assembly"
            or state.get("status") != "ready_for_assembly"
        ):
            raise ValueError("native PDF page state is invalid")
        _load_legacy_ref(store, state.get("native_page_ref"))
        return {"status": "ready_for_assembly", "page_id": page_id}
    outcome = advance_component_repair(store, page_id, _lease=_lease)
    if outcome["status"] == "awaiting_agent":
        manifest = store.read_json("job_manifest.json")
        if manifest.get("options", {}).get("pipeline_mode", "strict") == "fast":
            state = store.read_json(
                f"pages/{page_id}/reconstruction/component_state.json"
            )
            request_ref = state["current_round"]["request_ref"]["path"]
            request_path = store.root / Path(*PurePosixPath(request_ref).parts)
            _record_deterministic_fast_plan(
                store,
                page_id,
                request_path,
                store.root / "pages" / page_id / "reconstruction",
                _lease=_lease,
            )
            outcome = advance_component_repair(store, page_id, _lease=_lease)
    status = outcome["status"]
    if status == "needs_execution":
        rejected = _execute_legacy_round(
            store,
            page_id,
            _lease,
            performance_trace=performance_trace,
            visual_worker_pool=visual_worker_pool,
        )
        if rejected:
            return {
                "status": "awaiting_agent",
                "page_id": page_id,
                "repair_round": outcome["repair_round"],
            }
        return {"status": "processing", "page_id": page_id}
    if status == "needs_quality":
        record_component_quality(store, page_id, _lease=_lease)
        return {"status": "processing", "page_id": page_id}
    if status == "needs_next_round":
        if _fast_strict_escalation_exhausted(
            store, page_id, outcome["repair_round"],
        ):
            outcome = _commit_local_fidelity_result(
                store, page_id, _lease=_lease,
            )
            if outcome["status"] == "fallback_required":
                return {"status": "processing", "page_id": page_id}
            if outcome["status"] != "needs_next_round":
                return outcome
        _publish_next_legacy_request(
            store,
            page_id,
            outcome["repair_round"],
            _lease,
            visual_worker_pool=visual_worker_pool,
            visual_worker_pool_factory=visual_worker_pool_factory,
            performance_trace=performance_trace,
        )
        return {"status": "processing", "page_id": page_id}
    if status == "needs_parent_fallback":
        _execute_legacy_parent_fallback(store, page_id, _lease)
        return {"status": "processing", "page_id": page_id}
    if status == "needs_parent_quality":
        record_parent_fallback_quality(store, page_id, _lease=_lease)
        return {"status": "processing", "page_id": page_id}
    if status in {"freeze_committed", "fallback_required"}:
        return {"status": "processing", "page_id": page_id}
    return outcome


def _state_artifact(store: RunStore, reference: dict) -> Path:
    return _load_legacy_ref(store, reference)[0]


def _rebuild_canvas_background(
    *,
    source_path: Path,
    current_background_path: Path,
    restore_background_path: Path | None = None,
    repair_requests: list[tuple[set[str], float]],
    graph: dict,
    graph_dir: Path,
    text_mask_path: Path,
    output_path: Path,
    repair_all_active: bool = True,
) -> Path:
    import cv2
    import numpy as np

    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB")).copy()
    with Image.open(current_background_path) as image:
        current = np.asarray(image.convert("RGB")).copy()
    restored = None
    if restore_background_path is not None:
        with Image.open(restore_background_path) as image:
            restored = np.asarray(image.convert("RGB")).copy()
    with Image.open(text_mask_path) as image:
        text_repair = np.asarray(image.convert("L")) > 0
    if (
        source.shape != current.shape
        or (restored is not None and restored.shape != source.shape)
        or text_repair.shape != source.shape[:2]
    ):
        raise ValueError("background rebuild input dimensions differ")

    graph_root = graph_dir.resolve()
    by_id = {node["id"]: node for node in graph["nodes"]}
    masks_by_id = {}
    repairable_visual = np.zeros(text_repair.shape, dtype=bool)
    visible_coverage = np.zeros(text_repair.shape, dtype=bool)
    for object_id, node in by_id.items():
        mask_path = (graph_dir / Path(node["mask"])).resolve()
        if not mask_path.is_relative_to(graph_root):
            raise ValueError("background rebuild mask is outside graph directory")
        if sha256_file(mask_path) != node["mask_sha256"]:
            raise ValueError("background rebuild mask sha256 mismatch")
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        if mask.shape != text_repair.shape:
            raise ValueError("background rebuild mask dimensions differ")
        masks_by_id[object_id] = mask
        if node["kind"] != "text" and node["state"] in {
            "pending", "pending_gate", "frozen",
        }:
            visible_coverage |= mask
    edge_margin = max(1, round(min(text_repair.shape) * 0.01))
    inactive_page_surfaces = {
        object_id
        for object_id, node in by_id.items()
        if node["kind"] == "parent"
        and node["state"] == "inactive"
        and node["parent_id"] is None
        and node["z_index"] == 0
        and float(masks_by_id[object_id].mean()) >= 0.75
        and node["bbox"][0] <= edge_margin
        and node["bbox"][1] <= edge_margin
        and node["bbox"][2] >= text_repair.shape[1] - edge_margin
        and node["bbox"][3] >= text_repair.shape[0] - edge_margin
    }
    for object_id, node in by_id.items():
        ancestor_id = node["parent_id"]
        belongs_to_page_surface = object_id in inactive_page_surfaces
        while (
            node["state"] == "inactive"
            and ancestor_id is not None
            and ancestor_id in by_id
        ):
            if ancestor_id in inactive_page_surfaces:
                belongs_to_page_surface = True
                break
            ancestor_id = by_id[ancestor_id]["parent_id"]
        if not belongs_to_page_surface:
            repairable_visual |= masks_by_id[object_id]
    repair = (
        cv2.dilate(
            repairable_visual.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
        ) > 0
        if repair_all_active
        else np.zeros(text_repair.shape, dtype=bool)
    )
    restore_repair = np.zeros(text_repair.shape, dtype=bool)
    attached_text_ids = {
        text_id
        for node in by_id.values()
        if node["kind"] != "text"
        and node["state"] in {"pending", "pending_gate", "frozen"}
        for text_id in node["text_ids"]
    }
    for object_ids, margin_ratio in repair_requests:
        if not 0 < margin_ratio <= 0.1:
            raise ValueError("background rebuild margin_ratio is invalid")
        radius = max(1, round(min(text_repair.shape) * margin_ratio))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        for object_id in object_ids:
            request_mask = cv2.dilate(
                masks_by_id[object_id].astype(np.uint8), kernel
            ) > 0
            if (
                restored is not None
                and by_id[object_id]["kind"] == "text"
                and object_id not in attached_text_ids
            ):
                restore_repair |= request_mask
            else:
                repair |= request_mask
    if restored is None:
        repair |= text_repair
    elif np.any(text_repair):
        from image2editable.component_quality import (
            _prepare_page_quality_context, calibrate_page,
        )

        # Preserve recovered structure, but never reuse residual ink as a donor.
        text_context = _prepare_page_quality_context(
            source, restored, restored, text_repair,
            calibration=calibrate_page(source, text_repair),
        )
        _, text_labels = cv2.connectedComponents(text_repair.astype(np.uint8), 8)
        touched = np.unique(text_labels[text_context.background_residual_text_ink])
        touched = touched[touched > 0]
        residual_text_repair = np.isin(text_labels, touched)
        repair |= residual_text_repair
        restore_repair &= ~residual_text_repair
    rebuilt = current.copy()
    if restored is not None:
        restore_canvas = source.copy()
        restore_canvas[text_repair] = restored[text_repair]
        rebuilt[~repairable_visual] = restore_canvas[~repairable_visual]
        rebuilt[restore_repair] = restored[restore_repair]
        repair &= ~restore_repair
    if np.all(repair):
        # Inpainting without a donor silently returns the foreground unchanged.
        # The canvas is fully covered by movable layers; infer its base tone.
        border = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1]))
        rebuilt[:] = np.median(border, axis=0).astype(np.uint8)
    elif np.any(repair):
        from scripts.component_underlay import _choose_visual_fill

        rebuilt, _ = _choose_visual_fill(
            rgb=rebuilt,
            source_rgb=current,
            semantic_mask=repair,
            donor_mask=~repair,
            visual_hole=repair,
            allow_smooth_surface=True,
            allow_original=False,
        )
    if restored is not None:
        # Expanded donor exclusion must not erase uncovered source texture.
        uncovered = ~(visible_coverage | text_repair | restore_repair)
        rebuilt[uncovered] = source[uncovered]
    Image.fromarray(rebuilt, mode="RGB").save(output_path)
    return output_path


def _text_item_repair_padding_px(box_height: int) -> int:
    return max(2, min(4, int(round(max(box_height, 1) * 0.15))))


def _text_item_halo_px(box_height: int) -> int:
    return max(
        _text_item_repair_padding_px(box_height),
        min(12, int(round(max(box_height, 1) * 0.30))),
    )


def _assign_text_regions_to_component_masks(
    component_masks: list[Any],
    text_mask: Any,
    text_items: list[dict] | None = None,
    *,
    text_owner_indices: list[int | None] | None = None,
) -> list[Any]:
    import cv2
    import numpy as np

    text = np.asarray(text_mask, dtype=bool)
    assigned = [np.asarray(mask, dtype=bool).copy() for mask in component_masks]
    if not assigned:
        return assigned
    if any(mask.shape != text.shape for mask in assigned):
        raise ValueError("component text ownership mask dimensions differ")
    if text_owner_indices is not None and (
        text_items is None
        or len(text_owner_indices) != len(text_items)
        or any(
            value is not None
            and (type(value) is not int or not 0 <= value < len(assigned))
            for value in text_owner_indices
        )
    ):
        raise ValueError("component text owner indices are invalid")
    if text_items:
        regions = []
        height, width = text.shape
        for item_index, item in enumerate(text_items):
            box = item.get("box") if isinstance(item, dict) else None
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x, y, box_width, box_height = (int(value) for value in box)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + box_width), min(height, y + box_height)
            if x1 >= x2 or y1 >= y2:
                continue
            ownership_region = np.zeros(text.shape, dtype=bool)
            ownership_region[y1:y2, x1:x2] = text[y1:y2, x1:x2]
            if np.any(ownership_region):
                halo = _text_item_halo_px(box_height)
                box_region = np.zeros(text.shape, dtype=bool)
                box_region[y1:y2, x1:x2] = True
                fill_region = np.zeros(text.shape, dtype=bool)
                fill_region[
                    max(0, y1 - halo):min(height, y2 + halo),
                    max(0, x1 - halo):min(width, x2 + halo),
                ] = True
                regions.append((
                    ownership_region,
                    fill_region,
                    box_region,
                    None if text_owner_indices is None else text_owner_indices[item_index],
                ))
    else:
        count, labels = cv2.connectedComponents(text.astype(np.uint8), 8)
        regions = [
            (labels == label, labels == label, labels == label, None)
            for label in range(1, count)
        ]
    mask_boxes = []
    silhouette_masks = []
    for mask in assigned:
        ys, xs = np.nonzero(mask)
        mask_boxes.append(
            None if not len(xs) else (
                int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            )
        )
        silhouette = np.zeros_like(mask, dtype=np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.drawContours(silhouette, contours, -1, 1, thickness=cv2.FILLED)
        silhouette_masks.append(silhouette.astype(bool))
    for ownership_region, fill_region, box_region, explicit_owner in regions:
        pixels = int(np.count_nonzero(ownership_region))
        overlaps = [
            int(np.count_nonzero(mask & ownership_region)) for mask in assigned
        ]
        best = (
            explicit_owner
            if explicit_owner is not None
            else max(range(len(assigned)), key=overlaps.__getitem__)
        )
        overlap_ratio = overlaps[best] / max(pixels, 1)
        backing_ratio = 0.0
        if explicit_owner is None and text_items and overlap_ratio < 0.2:
            fill_pixels = int(np.count_nonzero(fill_region))
            backing = [
                int(np.count_nonzero(mask & fill_region)) for mask in assigned
            ]
            best = max(range(len(assigned)), key=backing.__getitem__)
            backing_ratio = backing[best] / max(fill_pixels, 1)
        box_nontext = box_region & ~text
        box_pixels = int(np.count_nonzero(box_nontext))
        box_backing = [
            int(np.count_nonzero(mask & box_nontext)) for mask in assigned
        ]
        if explicit_owner is None and box_pixels:
            box_best = max(range(len(assigned)), key=box_backing.__getitem__)
            if box_backing[box_best] > box_backing[best]:
                best = box_best
                overlap_ratio = overlaps[best] / max(pixels, 1)
                backing_ratio = 0.0
        box_backing_ratio = box_backing[best] / max(box_pixels, 1)
        box = mask_boxes[best]
        contained_ratio = 0.0
        box_contained_ratio = 0.0
        if box is not None:
            left, top, right, bottom = box
            contained_ratio = np.count_nonzero(
                fill_region[top:bottom, left:right]
            ) / max(int(np.count_nonzero(fill_region)), 1)
            box_contained_ratio = np.count_nonzero(
                box_region[top:bottom, left:right]
            ) / max(int(np.count_nonzero(box_region)), 1)
        dense_box_owner = bool(text_items) and (
            box_backing_ratio >= 0.5
            and box_contained_ratio >= 0.8
        )
        owns_text_region = explicit_owner is not None or (
            overlap_ratio >= (0.2 if text_items else 0.45)
            or (bool(text_items) and backing_ratio >= 0.5)
            or dense_box_owner
        )
        if owns_text_region and (
            explicit_owner is not None
            or not text_items
            or contained_ratio >= 0.8
            or dense_box_owner
        ):
            for index, mask in enumerate(assigned):
                if index != best:
                    mask[ownership_region] = False
            occupied_by_others = np.zeros(text.shape, dtype=bool)
            for index, mask in enumerate(assigned):
                if index != best:
                    occupied_by_others |= mask
            support = (
                silhouette_masks[best]
                if text_items else np.ones(text.shape, dtype=bool)
            )
            if dense_box_owner:
                support |= box_region
            assigned[best] |= fill_region & support & ~occupied_by_others
    return assigned


def _quality_text_repair_mask(text_mask, text_items: list[dict]):
    import cv2
    import numpy as np

    text = np.asarray(text_mask) > 0
    repair_mask = np.zeros(text.shape, dtype=bool)
    for item in text_items:
        box = item.get("box") if isinstance(item, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x, y, box_width, box_height = (int(value) for value in box)
        radius = max(2, min(6, int(round(max(box_height, 1) * 0.15))))
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2 = min(text.shape[1], x + box_width + radius)
        y2 = min(text.shape[0], y + box_height + radius)
        local = text[y1:y2, x1:x2].astype(np.uint8)
        repair_mask[y1:y2, x1:x2] |= cv2.dilate(
            local,
            np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8),
        ).astype(bool)
    return repair_mask


def _refine_quality_text_clean(
    source,
    text_clean,
    text_mask,
    text_items: list[dict],
):
    import numpy as np

    source = np.asarray(source, dtype=np.uint8)
    cleaned = np.asarray(text_clean, dtype=np.uint8)
    text = np.asarray(text_mask) > 0
    if cleaned.shape != source.shape or text.shape != source.shape[:2]:
        raise ValueError("quality text refinement dimensions differ")
    if not text_items or not np.any(text):
        return cleaned.copy()
    import cv2

    module = importlib.import_module("image_to_ppt")
    repair_mask = _quality_text_repair_mask(text, text_items)
    candidate = module._repair_text_with_local_planes(
        source,
        repair_mask.astype(np.uint8) * 255,
        text_items,
    )
    refined = cleaned.copy()
    local_background = cv2.inpaint(
        cleaned,
        repair_mask.astype(np.uint8) * 255,
        3,
        cv2.INPAINT_TELEA,
    )
    cleaned_error = np.sum(
        np.abs(cleaned.astype(np.int16) - local_background.astype(np.int16)),
        axis=2,
    )
    candidate_error = np.sum(
        np.abs(candidate.astype(np.int16) - local_background.astype(np.int16)),
        axis=2,
    )
    unchanged_from_source = np.all(cleaned == source, axis=2)
    source_candidate_delta = np.max(
        np.abs(source.astype(np.int16) - candidate.astype(np.int16)),
        axis=2,
    )
    use_candidate = repair_mask & (
        (unchanged_from_source & (source_candidate_delta > 32))
        | (~unchanged_from_source & (candidate_error < cleaned_error))
    )
    refined[use_candidate] = candidate[use_candidate]

    short_side = min(source.shape[:2])
    line_length = max(9, min(31, round(short_side * 0.03)))
    local_delta = np.zeros(source.shape[:2], dtype=np.uint8)
    for channel in range(3):
        local_delta = np.maximum(
            local_delta,
            cv2.absdiff(
                source[:, :, channel],
                cv2.medianBlur(source[:, :, channel], 9),
            ),
        )
    contrast = (local_delta > 12).astype(np.uint8)
    horizontal = cv2.morphologyEx(
        contrast, cv2.MORPH_OPEN,
        np.ones((1, line_length), dtype=np.uint8),
    )
    vertical = cv2.morphologyEx(
        contrast, cv2.MORPH_OPEN,
        np.ones((line_length, 1), dtype=np.uint8),
    )
    line_mask = (horizontal | vertical).astype(bool)
    line_count, line_labels = cv2.connectedComponents(
        line_mask.astype(np.uint8), 8
    )
    for line_label in range(1, line_count):
        component = line_labels == line_label
        inside = component & repair_mask
        outside = component & ~repair_mask
        if not np.any(inside) or not np.any(outside):
            continue
        refined[inside] = np.median(source[outside], axis=0).astype(np.uint8)
    return refined


def _reuse_frozen_presentation_records(
    current: dict, previous: dict, frozen_ids: set[str]
) -> dict:
    previous_by_id = {
        component["component_id"]: component
        for component in previous["components"]
    }
    current_ids = {
        component["component_id"] for component in current["components"]
    }
    if frozen_ids - set(previous_by_id) or frozen_ids - current_ids:
        raise ValueError("frozen presentation component is missing")
    return {
        **current,
        "components": [
            previous_by_id[component["component_id"]]
            if component["component_id"] in frozen_ids
            else component
            for component in current["components"]
        ],
    }


def _text_boundary_structure_mask(source, text_items: list[dict], visual_mask):
    import re

    import cv2
    import numpy as np

    protected = np.zeros(source.shape[:2], dtype=bool)
    known_text = np.zeros(source.shape[:2], dtype=bool)
    for item in text_items:
        color = item.get("color")
        x, y, width, height = (int(value) for value in item["box"])
        text_radius = _text_item_repair_padding_px(height)
        if item.get("runs") or not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            # A single color cannot describe styled runs or unknown text ink.
            known_text[
                max(0, y - text_radius):min(source.shape[0], y + height + text_radius),
                max(0, x - text_radius):min(source.shape[1], x + width + text_radius),
            ] = True
            continue
        pad = max(12, height * 2)
        left, top = max(0, x - pad), max(0, y - pad)
        right = min(source.shape[1], x + width + pad)
        bottom = min(source.shape[0], y + height + pad)
        crop = source[top:bottom, left:right]
        background = np.median(np.concatenate((
            crop[0], crop[-1], crop[:, 0], crop[:, -1],
        )), axis=0)
        target = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)])
        axis = target - background
        length = float(axis @ axis)
        if length <= 0:
            continue
        delta = crop.astype(np.float32) - background
        opacity = np.clip((delta @ axis) / length, 0, 1)
        color_match = np.linalg.norm(delta - opacity[..., None] * axis, axis=2) <= 12
        text_color = color_match & (opacity >= 0.02)
        foreground = (np.linalg.norm(delta, axis=2) > 12) & ~text_color
        box = np.zeros(foreground.shape, dtype=bool)
        box[
            max(0, y - top):min(crop.shape[0], y + height - top),
            max(0, x - left):min(crop.shape[1], x + width - left),
        ] = True
        text_support = cv2.dilate(
            box.astype(np.uint8),
            np.ones((2 * text_radius + 1, 2 * text_radius + 1), dtype=np.uint8),
        ).astype(bool)
        known_text[top:bottom, left:right] |= text_support & color_match & (opacity > 0)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8), 8
        )
        for label in range(1, count):
            component = labels == label
            inside = component & box
            if not np.any(inside) or np.count_nonzero(inside) * 4 > stats[label, 4]:
                continue
            if not np.any(component & ~box & visual_mask[top:bottom, left:right]):
                continue
            points = cv2.findNonZero(component.astype(np.uint8))
            short, long = sorted(cv2.minAreaRect(points)[1])
            if short > 4 or long < max(12, height, short * 6):
                continue
            # Only a thin, differently colored structure continuing outside the
            # OCR box can reclaim pixels; text-colored A/X strokes stay text.
            halo = cv2.dilate(
                component.astype(np.uint8), np.ones((5, 5), dtype=np.uint8),
            ).astype(bool)
            halo &= ~text_color & (~foreground | component)
            protected[top:bottom, left:right] |= halo
    # Another overlapping OCR item may own a differently colored thin glyph.
    return protected & ~known_text


def _effective_text_context(
    *,
    source,
    text_clean,
    text_mask,
    text_items: object,
    graph: dict,
    graph_dir: Path,
    refine_text_clean: bool = True,
    refine_cleanup_mask: bool = False,
) -> tuple[list[dict], Any, Any]:
    import numpy as np

    source = np.asarray(source, dtype=np.uint8)
    cleaned = np.asarray(text_clean, dtype=np.uint8)
    original_mask = np.asarray(text_mask) > 0
    if cleaned.shape != source.shape or original_mask.shape != source.shape[:2]:
        raise ValueError("effective text dimensions differ")
    records = _component_text_records(
        text_items, (source.shape[1], source.shape[0])
    )
    records_by_id = {
        record["normalized"]["id"]: record for record in records
    }
    frozen_mask = np.zeros(original_mask.shape, dtype=bool)
    suppressed_mask = np.zeros(original_mask.shape, dtype=bool)
    active_visual_mask = np.zeros(original_mask.shape, dtype=bool)
    frozen_ids = set()
    for node in graph["nodes"]:
        if (
            node["kind"] != "text"
            and node["state"] in {"pending", "pending_gate", "frozen"}
        ):
            mask_path = graph_dir / Path(node["mask"])
            if sha256_file(mask_path) != node["mask_sha256"]:
                raise ValueError("effective visual graph mask sha256 mismatch")
            with Image.open(mask_path) as image:
                node_mask = np.asarray(image.convert("L")) > 0
            if node_mask.shape != original_mask.shape:
                raise ValueError("effective visual graph mask dimensions differ")
            active_visual_mask |= node_mask
            continue
        if node["kind"] != "text" or node["id"] not in records_by_id:
            continue
        mask_path = graph_dir / Path(node["mask"])
        if sha256_file(mask_path) != node["mask_sha256"]:
            raise ValueError("effective text graph mask sha256 mismatch")
        with Image.open(mask_path) as image:
            node_mask = np.asarray(image.convert("L")) > 0
        if node_mask.shape != original_mask.shape:
            raise ValueError("effective text graph mask dimensions differ")
        if node["state"] == "frozen":
            frozen_ids.add(node["id"])
            frozen_mask |= node_mask
        elif node["state"] == "inactive":
            suppressed_mask |= node_mask
    effective_items = [
        {**record["raw"], "_component_id": component_id}
        for component_id, record in records_by_id.items()
        if component_id in frozen_ids
    ]
    effective_mask = original_mask & frozen_mask
    authenticated_mask = effective_mask.copy()
    effective_clean = cleaned.copy()
    restore = suppressed_mask & ~frozen_mask
    effective_clean[restore] = source[restore]
    if refine_cleanup_mask and effective_items and np.any(effective_mask):
        module = importlib.import_module("image_to_ppt")
        refined_mask = module._build_text_cleanup_mask(
            source,
            effective_mask.astype(np.uint8) * 255,
            effective_items,
        ) > 0
        effective_mask &= refined_mask
        from image2editable.component_quality import (
            _prepare_page_quality_context,
            calibrate_page,
        )
        import cv2

        calibration = calibrate_page(source, effective_mask)
        context = _prepare_page_quality_context(
            source,
            effective_clean,
            effective_clean,
            effective_mask,
            calibration=calibration,
            text_items=effective_items,
        )
        residual = context.background_residual_text_ink & effective_mask
        if np.any(residual):
            count, labels, _, _ = cv2.connectedComponentsWithStats(
                effective_mask.astype(np.uint8), 8
            )
            donor = effective_clean.copy()
            for label in range(1, count):
                component = labels == label
                if not np.any(component & residual):
                    continue
                repair = cv2.dilate(
                    component.astype(np.uint8),
                    np.ones((3, 3), dtype=np.uint8),
                ).astype(bool) & authenticated_mask
                area = int(np.count_nonzero(repair))
                radius = max(3, min(12, int(np.ceil(np.sqrt(area) * 0.18))))
                ring = cv2.dilate(
                    repair.astype(np.uint8),
                    np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8),
                ).astype(bool) & ~authenticated_mask
                if not np.any(ring):
                    continue
                effective_clean[repair] = np.median(
                    donor[ring], axis=0
                ).astype(np.uint8)
                effective_mask |= repair
    if refine_text_clean and effective_items and np.any(effective_mask):
        import cv2

        effective_clean = _refine_quality_text_clean(
            source, effective_clean, effective_mask, effective_items
        )
        protected_visual_mask = cv2.dilate(
            active_visual_mask.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
        ).astype(bool)
        effective_mask = _quality_text_repair_mask(
            effective_mask, effective_items
        )
        if refine_cleanup_mask:
            protected_visual_mask &= ~effective_mask
        effective_clean[protected_visual_mask] = cleaned[protected_visual_mask]
        if refine_cleanup_mask:
            structure = _text_boundary_structure_mask(
                source, effective_items, active_visual_mask,
            )
            effective_mask &= ~structure
            effective_clean[structure] = source[structure]
    return effective_items, effective_mask, effective_clean


def _decode_bound_legacy_image(
    payload: bytes,
    *,
    mode: str,
    expected_size: tuple[int, int] | None = None,
):
    import numpy as np

    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if expected_size is not None and image.size != expected_size:
                raise ValueError("legacy image dimensions differ")
            return np.asarray(image.convert(mode)).copy()
    except (OSError, ValueError):
        raise ValueError("legacy image is invalid") from None


def _verify_regular_at(
    parent_fd: int,
    name: str,
    identity: tuple[int, int],
) -> None:
    status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or (status.st_dev, status.st_ino) != identity
    ):
        raise RuntimeError("background responsibility artifact changed")


def _publish_staging_at(
    parent_fd: int, staging_name: str, final_name: str, identity: tuple[int, int]
) -> None:
    _verify_regular_at(parent_fd, staging_name, identity)
    _rename_file_exclusive_at(parent_fd, staging_name, parent_fd, final_name)


def _publish_background_responsibility_file(
    target: Path,
    payload: bytes,
    root: Path,
) -> bytes:
    chain = _snapshot_directory_chain(target.parent, root)
    staging_name = f".background-responsibility-staging-{uuid.uuid4().hex}.png"
    if os.name == "nt":
        kernel32, parent_handle, parent_status = _windows_open_bound(
            target.parent, chain[-1][1][:2], directory=True
        )
        staging = target.with_name(staging_name)
        source_handle = source_descriptor = locked_parent_handle = None
        try:
            identity = _write_exclusive(staging, payload, root)
            _, source_handle, _ = _windows_open_bound(
                staging,
                identity,
                directory=False,
                desired_access=0x00010081,
                share_mode=0x1,
            )
            import msvcrt

            source_descriptor = msvcrt.open_osfhandle(
                source_handle, os.O_RDONLY | os.O_BINARY
            )
            source_handle = None
            opened = os.fstat(source_descriptor)
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            read_back = os.read(source_descriptor, len(payload) + 1)
            stable = os.fstat(source_descriptor)
            if (
                read_back != payload
                or opened.st_nlink != 1
                or stable.st_nlink != 1
                or (opened.st_dev, opened.st_ino, stable.st_size)
                != (*identity, len(payload))
            ):
                raise RuntimeError("background responsibility staging changed")
            bound_handle = msvcrt.get_osfhandle(source_descriptor)
            _rename_windows_staging(bound_handle, parent_handle, target.name)
            _, locked_parent_handle, _ = _windows_open_bound(
                target.parent,
                chain[-1][1][:2],
                directory=True,
                share_mode=0x1,
            )
            entries = _windows_entries(
                kernel32,
                locked_parent_handle,
                target.parent,
                parent_status,
            )
            published = [entry for entry in entries if entry[0] == target.name]
            _, links, final_identity = _windows_handle_information(
                kernel32, bound_handle
            )
            if (
                len(published) != 1
                or published[0][2] != identity
                or links != 1
                or final_identity != (identity[0] & 0xFFFFFFFF, identity[1])
            ):
                raise RuntimeError("background responsibility publication changed")
            return read_back
        finally:
            try:
                if locked_parent_handle is not None:
                    _windows_close(kernel32, locked_parent_handle)
                _windows_close(kernel32, parent_handle)
            finally:
                if source_descriptor is not None:
                    os.close(source_descriptor)
                elif source_handle is not None:
                    _windows_close(kernel32, source_handle)

    parent_fd = os.open(
        target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        opened_parent = os.fstat(parent_fd)
        if (
            opened_parent.st_dev,
            opened_parent.st_ino,
            opened_parent.st_mode,
            getattr(opened_parent, "st_file_attributes", 0),
        ) != chain[-1][1]:
            raise RuntimeError("background responsibility parent changed")
        descriptor = os.open(
            staging_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(descriptor)
            identity = opened.st_dev, opened.st_ino
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise RuntimeError("background responsibility staging is unsafe")
            view = memoryview(payload)
            while view:
                view = view[os.write(descriptor, view):]
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            read_back = bytearray()
            while chunk := os.read(descriptor, len(payload) + 1 - len(read_back)):
                read_back.extend(chunk)
                if len(read_back) > len(payload):
                    break
            stable = os.fstat(descriptor)
            if (
                bytes(read_back) != payload
                or stable.st_nlink != 1
                or (stable.st_dev, stable.st_ino, stable.st_size)
                != (*identity, len(payload))
            ):
                raise RuntimeError("background responsibility staging changed")
            _publish_staging_at(parent_fd, staging_name, target.name, identity)
            _verify_regular_at(parent_fd, target.name, identity)
            return bytes(read_back)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _publish_background_responsibility(
    store: RunStore,
    output_dir: Path,
    *,
    allowed,
    previous_ref: dict | None,
    background_rebuilt: bool,
) -> dict | None:
    import numpy as np

    if allowed is None:
        return None
    next_mask = allowed
    previous_mask = None
    if not background_rebuilt:
        if previous_ref is None:
            return None
        _, payload = _load_legacy_ref(store, previous_ref)
        previous_mask = _decode_binary_grayscale_png(
            payload,
            allowed.shape,
            label="legacy background responsibility",
        )
        next_mask = previous_mask & allowed
    if not np.any(next_mask):
        return None
    if float(np.asarray(next_mask, dtype=bool).mean()) > 0.05:
        return None
    if previous_mask is not None and np.array_equal(next_mask, previous_mask):
        return dict(previous_ref)
    encoded = io.BytesIO()
    Image.fromarray(next_mask.astype(np.uint8) * 255, mode="L").save(
        encoded, format="PNG"
    )
    payload = encoded.getvalue()
    target = output_dir / "background-responsibility.png"
    try:
        read_back = _publish_background_responsibility_file(
            target, payload, store.root
        )
    except (OSError, RuntimeError):
        raise ValueError("background responsibility could not be published") from None
    return {
        "path": target.relative_to(store.root).as_posix(),
        "sha256": hashlib.sha256(read_back).hexdigest(),
    }


def _quality_assets(
    store: RunStore,
    page_id: str,
    graph: dict,
    graph_dir: Path,
    output_dir: Path,
    *,
    background_path_override: Path | None = None,
    previous_quality_refs: dict | None = None,
    background_rebuilt: bool = False,
    frozen_manifest_path: Path | None = None,
    frozen_component_ids: set[str] | None = None,
) -> dict:
    import numpy as np

    from image2editable.component_quality import (
        contained_active_parent_pairs,
    )

    module = importlib.import_module("image_to_ppt")
    prepared = module.load_component_layers(
        store.root / "pages" / page_id / "reconstruction/initial/prepared_page.json"
    )
    previous_refs = (
        previous_quality_refs
        if isinstance(previous_quality_refs, dict)
        else {}
    )
    previous_responsibility_ref = (
        previous_refs.get("background_responsibility")
        if isinstance(previous_refs.get("background_responsibility"), dict)
        else None
    )
    source_ref = previous_refs.get("source")
    source_path = Path(prepared["original_image_path"])
    source_payload = None
    if isinstance(source_ref, dict):
        source_path, source_payload = _load_legacy_ref(store, source_ref)
    text_clean_path = Path(prepared.get("_text_clean_path", source_path))
    background_payload = None
    if background_rebuilt:
        background_path = output_dir / "background-rebuilt.png"
        background_payload = _read_bound_legacy_file(
            store,
            background_path,
            max_bytes=256 * 1024 * 1024,
            label="rebuilt background",
        )
    elif isinstance(previous_refs.get("background"), dict):
        background_path, background_payload = _load_legacy_ref(
            store, previous_refs["background"]
        )
    else:
        background_path = (
            Path(prepared["background_original_path"])
            if background_path_override is None
            else background_path_override
        )
    foreground_ref = previous_refs.get("foreground_evidence")
    compute_responsibility_allowed = (
        (background_rebuilt or previous_responsibility_ref is not None)
        and source_payload is not None
        and background_payload is not None
        and isinstance(foreground_ref, dict)
    )
    text_mask_path = Path(prepared.get(
        "_text_cleanup_mask_path", prepared["_text_mask_path"]
    ))
    if source_payload is None:
        with Image.open(source_path) as image:
            source = np.asarray(image.convert("RGB")).copy()
    else:
        source = _decode_bound_legacy_image(source_payload, mode="RGB")
    with Image.open(text_clean_path) as image:
        text_clean = np.asarray(image.convert("RGB")).copy()
    with Image.open(text_mask_path) as image:
        text_mask = np.asarray(image.convert("L")) > 0
    effective_items, text_mask, text_clean = _effective_text_context(
        source=source,
        text_clean=text_clean,
        text_mask=text_mask,
        text_items=prepared.get("text_items", []),
        graph=graph,
        graph_dir=graph_dir,
        refine_text_clean=True,
        refine_cleanup_mask="_text_cleanup_mask_path" in prepared,
    )
    text_clean_output = output_dir / "text-clean.png"
    Image.fromarray(text_clean, mode="RGB").save(text_clean_output)
    text_mask_output = output_dir / "text-mask.png"
    Image.fromarray(text_mask.astype(np.uint8) * 255, mode="L").save(
        text_mask_output
    )
    text_mask = _decode_binary_grayscale_png(
        _read_bound_legacy_file(
            store,
            text_mask_output,
            max_bytes=max(1024 * 1024, source.shape[0] * source.shape[1] * 2),
            label="effective text mask",
        ),
        source.shape[:2],
        label="effective text mask",
    )
    component_nodes = []
    component_masks = []
    semantic_ownership = (
        np.zeros(source.shape[:2], dtype=bool)
        if compute_responsibility_allowed
        else None
    )
    for node in graph["nodes"]:
        if node["kind"] == "text" or node["state"] not in {
            "pending", "pending_gate", "frozen"
        }:
            continue
        component_nodes.append(node)
        mask_path = graph_dir / Path(node["mask"])
        mask_payload = _read_bound_legacy_file(
            store,
            mask_path,
            max_bytes=max(1024 * 1024, source.shape[0] * source.shape[1] * 2),
            label="execution graph mask",
        )
        if hashlib.sha256(mask_payload).hexdigest() != node["mask_sha256"]:
            raise ValueError("execution graph mask sha256 mismatch")
        mask = _decode_bound_legacy_image(
            mask_payload,
            mode="L",
            expected_size=(source.shape[1], source.shape[0]),
        ) > 0
        component_masks.append(mask)
        if semantic_ownership is not None:
            semantic_ownership |= mask
    contained_parent_pairs = contained_active_parent_pairs(
        component_nodes, component_masks
    )
    graph_path = graph_dir / "component-graph.json"
    frozen_ids = set() if frozen_component_ids is None else frozen_component_ids
    presentation_options = {}
    if frozen_ids:
        if frozen_manifest_path is None:
            raise ValueError("frozen presentation manifest is missing")
        previous_manifest = json.loads(
            frozen_manifest_path.read_text(encoding="utf-8")
        )
        frozen_records = {
            component["component_id"]: component
            for component in previous_manifest["components"]
            if component["component_id"] in frozen_ids
        }
        if frozen_ids - set(frozen_records):
            raise ValueError("frozen presentation component is missing")
        presentation_options["frozen_components"] = frozen_records
    presentation_manifest = _build_presentation_assets(
        store,
        source_path=source_path,
        text_clean_path=text_clean_output,
        text_mask_path=text_mask_output,
        graph_path=graph_path,
        output_dir=output_dir,
        **presentation_options,
    )
    if frozen_ids:
        current_manifest = json.loads(
            presentation_manifest.read_text(encoding="utf-8")
        )
        reused_manifest = _reuse_frozen_presentation_records(
            current_manifest, previous_manifest, frozen_ids
        )
        presentation_manifest.write_text(
            json.dumps(
                reused_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    if background_payload is None:
        with Image.open(background_path) as image:
            background_image = image.convert("RGB")
    else:
        background_image = Image.fromarray(
            _decode_bound_legacy_image(
                background_payload,
                mode="RGB",
                expected_size=(source.shape[1], source.shape[0]),
            ),
            mode="RGB",
        )
    try:
        active_nodes = _active_visual_nodes(graph)
        indexed = {node["id"]: index for index, node in enumerate(active_nodes)}
        component_ids = [
            node["id"] for node in sorted(
                active_nodes,
                key=lambda item: (
                    int(item["z_index"]), indexed[item["id"]]
                ),
            )
        ]
        presentation_ownership = (
            np.zeros(source.shape[:2], dtype=bool)
            if compute_responsibility_allowed
            else None
        )

        def presentation_layers():
            for layer in _load_presentation_assets(
                run_root=store.root,
                reconstruction=(
                    store.root / "pages" / page_id / "reconstruction"
                ),
                manifest_path=presentation_manifest,
                source_sha256=sha256_file(source_path),
                graph_sha256=sha256_file(graph_path),
                graph=graph,
                page_size=background_image.size,
                component_ids=component_ids,
            ):
                if presentation_ownership is not None:
                    presentation_ownership[:] |= layer["ownership_mask"] > 0
                yield layer

        reconstructed_image = _composite_presentation_layers(
            background_image,
            graph,
            presentation_layers(),
        )
    finally:
        background_image.close()
    reconstructed = np.asarray(reconstructed_image).copy()
    reconstructed_image.close()
    allowed = None
    if compute_responsibility_allowed:
        _, foreground_payload = _load_legacy_ref(store, foreground_ref)
        foreground = _decode_bound_legacy_image(
            foreground_payload,
            mode="L",
            expected_size=(source.shape[1], source.shape[0]),
        ) > 0
        background_pixels = _decode_bound_legacy_image(
            background_payload,
            mode="RGB",
            expected_size=(source.shape[1], source.shape[0]),
        )
        from image2editable.component_quality import (
            _background_responsibility_geometry,
            calibrate_page,
            refine_material_foreground,
        )

        material_foreground = refine_material_foreground(
            foreground,
            source,
            background_pixels,
            calibrate_page(source, text_mask),
        )
        candidate = (
            material_foreground
            & ~text_mask
            & ~(semantic_ownership | presentation_ownership)
            & np.all(source == background_pixels, axis=2)
        )
        allowed = _background_responsibility_geometry(candidate)
    responsibility_ref = _publish_background_responsibility(
        store,
        output_dir,
        allowed=allowed,
        previous_ref=previous_responsibility_ref,
        background_rebuilt=background_rebuilt,
    )
    assets = {
        "background": output_dir / "background.png",
        "reconstructed": output_dir / "reconstructed.png",
        "text_mask": output_dir / "text-mask.png",
        "native_check": output_dir / "native-check.json",
        "presentation_manifest": presentation_manifest,
    }
    if background_payload is None:
        shutil.copyfile(background_path, assets["background"])
    else:
        try:
            _write_exclusive(assets["background"], background_payload, store.root)
        except (OSError, RuntimeError):
            raise ValueError("legacy background could not be published") from None
    Image.fromarray(reconstructed, mode="RGB").save(assets["reconstructed"])
    Image.fromarray(text_mask.astype(np.uint8) * 255, mode="L").save(
        assets["text_mask"]
    )
    assets["native_check"].write_text(json.dumps({
        "schema_version": 1, "page_id": page_id,
        "source_sha256": sha256_file(source_path),
        "protected_native_overlap": "pass",
        "contained_parent_pairs": [
            list(pair) for pair in sorted(contained_parent_pairs)
        ],
        "text_items": effective_items,
        "initial_diagnostics": prepared.get("initial_diagnostics", []),
    }, ensure_ascii=False), encoding="utf-8")
    refs = {}
    for name, path in assets.items():
        refs[name] = {
            "path": path.resolve().relative_to(store.root.resolve()).as_posix(),
            "sha256": sha256_file(path),
        }
    if isinstance(foreground_ref, dict) and source_payload is not None:
        refs["foreground_evidence"] = dict(foreground_ref)
    if responsibility_ref is not None:
        refs["background_responsibility"] = responsibility_ref
    return refs


def _record_inference_performance(
    performance_trace, started: float, page_id: str,
    operation_count: int, status: str,
) -> None:
    if performance_trace is None:
        return
    try:
        performance_trace.event(
            "inference_finish",
            page_id=page_id,
            stage="component_sam",
            model="sam",
            operation_count=operation_count,
            duration_ms=round((time.perf_counter() - started) * 1000),
            status=status,
        )
    except Exception:
        _LOGGER.warning("Performance trace recording failed")


def _rebind_strict_pending_masks(
    graph: dict,
    strict_slide_data: dict,
    *,
    old_graph_dir: Path,
    strict_graph_dir: Path,
    pending_ids: list[str],
) -> dict:
    """Map strict visual masks back onto the existing pending component IDs."""
    import numpy as np

    validated = validate_component_graph(graph)
    nodes = {node["id"]: node for node in validated["nodes"]}
    if pending_ids != sorted(set(pending_ids)) or any(
        component_id not in nodes
        or nodes[component_id]["state"] not in {"pending", "pending_gate"}
        for component_id in pending_ids
    ):
        raise ValueError("strict pending component IDs are invalid")
    components = strict_slide_data.get("components")
    mask_paths = strict_slide_data.get("_element_mask_paths")
    semantic_paths = strict_slide_data.get("_semantic_mask_paths")
    if (
        not isinstance(components, list)
        or not isinstance(mask_paths, list)
        or len(components) != len(mask_paths)
        or not mask_paths
    ):
        raise ValueError("strict visual candidate count is invalid")

    def load_mask(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8).copy() > 0
        if mask.ndim != 2 or not np.any(mask):
            raise ValueError("strict visual candidate mask is invalid")
        return mask

    old_masks = {}
    for component_id in pending_ids:
        node = nodes[component_id]
        old_path = old_graph_dir / Path(*PurePosixPath(node["mask"]).parts)
        old_masks[component_id] = load_mask(old_path)
    strict_masks = [load_mask(Path(path)) for path in mask_paths]
    shape = next(iter(old_masks.values())).shape
    if any(mask.shape != shape for mask in strict_masks):
        raise ValueError("strict visual candidate dimensions differ")
    child_ids = [
        component_id for component_id in pending_ids
        if nodes[component_id]["parent_id"] is not None
    ]
    strict_semantic_masks = None
    if child_ids:
        if (
            not isinstance(semantic_paths, list)
            or len(semantic_paths) != len(strict_masks)
        ):
            raise ValueError("strict semantic candidate masks are invalid")
        strict_semantic_masks = [load_mask(Path(path)) for path in semantic_paths]
        if any(mask.shape != shape for mask in strict_semantic_masks):
            raise ValueError("strict semantic candidate dimensions differ")

    scores = []
    for component_id in pending_ids:
        source = old_masks[component_id]
        for index, candidate in enumerate(strict_masks):
            intersection = int(np.count_nonzero(source & candidate))
            union = int(np.count_nonzero(source | candidate))
            if intersection:
                scores.append((
                    intersection / max(1, union), -index, component_id, index,
                ))
    assignments = {}
    used = set()
    for _, _, component_id, index in sorted(scores, reverse=True):
        if component_id in assignments or index in used:
            continue
        assignments[component_id] = index
        used.add(index)
    assigned_candidates = set(assignments.values())
    absorbed_ids = []
    for component_id in pending_ids:
        if component_id in assignments:
            continue
        source = old_masks[component_id]
        source_pixels = max(1, int(np.count_nonzero(source)))
        if any(
            index in assigned_candidates
            and np.count_nonzero(source & candidate) / source_pixels >= 0.98
            for index, candidate in enumerate(strict_masks)
        ):
            absorbed_ids.append(component_id)
    # Unmatched components retain their original masks and still face the gate.
    replacements = {
        component_id: strict_masks[index]
        for component_id, index in assignments.items()
    }
    parent_assignments = {}
    for component_id, index in assignments.items():
        parent_id = nodes[component_id]["parent_id"]
        if parent_id is None:
            continue
        previous = parent_assignments.setdefault(parent_id, index)
        if previous != index:
            raise ValueError("strict semantic parent mapping is ambiguous")
    if strict_semantic_masks is not None:
        replacements.update({
            parent_id: strict_semantic_masks[index]
            for parent_id, index in parent_assignments.items()
        })

    strict_graph_dir.mkdir(parents=True, exist_ok=False)
    masks_dir = strict_graph_dir / "masks"
    masks_dir.mkdir()
    (strict_graph_dir / "absorbed-component-ids.json").write_text(
        json.dumps({
            "schema_version": 1,
            "component_ids": absorbed_ids,
        }, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rebound = {"nodes": []}
    for index, original in enumerate(validated["nodes"], start=1):
        node = dict(original)
        target_name = (
            f"masks/{index:04d}-{original['id']}.png"
            if original["id"] in assignments
            else original["mask"]
        )
        target = strict_graph_dir / Path(*PurePosixPath(target_name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        mask = replacements.get(original["id"])
        if mask is not None:
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(target)
        else:
            source_path = old_graph_dir / Path(
                *PurePosixPath(original["mask"]).parts
            )
            shutil.copyfile(source_path, target)
            rebound["nodes"].append(node)
            continue
        ys, xs = np.nonzero(mask)
        node.update({
            "mask": target_name,
            "mask_sha256": sha256_file(target),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        })
        rebound["nodes"].append(node)
    return rebound


def _prepare_fast_strict_round(
    store: RunStore,
    page_id: str,
    repair_round: int,
    pending_ids: list[str],
    *,
    visual_worker_pool=None,
    performance_trace=None,
) -> Path | None:
    """Build one strict visual round for a failed Fast page."""
    import numpy as np

    manifest = store.read_json("job_manifest.json")
    if manifest.get("options", {}).get("pipeline_mode", "strict") != "fast":
        return None
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    prepared_path = reconstruction / "initial" / "prepared_page.json"
    module = importlib.import_module("image_to_ppt")
    prepared = module.load_component_layers(prepared_path)
    policy = prepared.get("_page_policy", {})
    if policy.get("route") not in {"direct", "local_refine"}:
        return None
    root = reconstruction / f"strict-escalation-{repair_round:02d}"
    if root.exists():
        graph_path = root / "graph" / "component-graph.json"
        if graph_path.is_file():
            validate_component_graph(json.loads(graph_path.read_text(encoding="utf-8")))
            return graph_path
        shutil.rmtree(root)
    root.mkdir()
    strict_prepared_dir = root / "prepared"
    source_value = prepared.get("original_image_path")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("prepared source image is invalid")
    source = Path(source_value)
    if not source.is_file():
        raise ValueError("prepared source image is missing")
    with Image.open(prepared["_text_mask_path"]) as image:
        text_mask = np.asarray(image.convert("L")).copy()
    try:
        prepare_kwargs = {
            "lang": manifest["options"]["lang"],
            "resource_isolation": True,
            "ocr_result": (prepared.get("text_items", []), text_mask),
        }
        if visual_worker_pool is not None:
            prepare_kwargs.update({
                "visual_worker_pool": visual_worker_pool,
                "performance_trace": performance_trace,
                "page_id": page_id,
            })
        strict_slide_data = module.prepare_component_layers(
            source,
            strict_prepared_dir,
            **prepare_kwargs,
        )
        current_state = store.read_json(
            f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}"
        )
        current_graph_path = _state_artifact(
            store, current_state["graph_ref"]
        )
        current_graph = json.loads(current_graph_path.read_text(encoding="utf-8"))
        graph_dir = root / "graph"
        rebound = _rebind_strict_pending_masks(
            current_graph,
            strict_slide_data,
            old_graph_dir=current_graph_path.parent,
            strict_graph_dir=graph_dir,
            pending_ids=pending_ids,
        )
        graph_path = graph_dir / "component-graph.json"
        graph_path.write_text(
            json.dumps(rebound, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return graph_path
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        text_mask = None
        shutil.rmtree(strict_prepared_dir, ignore_errors=True)


def _fast_strict_escalation_exhausted(
    store: RunStore, page_id: str, next_repair_round: int,
) -> bool:
    manifest = store.read_json("job_manifest.json")
    if manifest.get("options", {}).get("pipeline_mode", "strict") != "fast":
        return False
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    prepared = importlib.import_module("image_to_ppt").load_component_layers(
        reconstruction / "initial" / "prepared_page.json"
    )
    route = prepared.get("_page_policy", {}).get("route")
    if route not in {"direct", "local_refine"}:
        return False
    # The former local-fidelity boundary now returns to full quality repair.
    if (
        next_repair_round == 2
        and manifest.get("input", {}).get("type") == "pdf"
        and route in {"direct", "local_refine"}
    ):
        return True
    if next_repair_round != 3:
        return False
    graph_path = reconstruction / "strict-escalation-02/graph/component-graph.json"
    if not graph_path.is_file():
        return False
    validate_component_graph(json.loads(graph_path.read_text(encoding="utf-8")))
    return True


def _commit_local_fidelity_result(
    store: RunStore,
    page_id: str,
    *,
    _lease: ExecutionLease,
) -> dict[str, Any]:
    _require_held_execution_lease(store, _lease)
    state_relative = (
        Path("pages") / page_id / "reconstruction" / COMPONENT_STATE_NAME
    )
    state = validate_component_repair_state(store.read_json(state_relative))
    if (
        state["page_id"] != page_id
        or state["phase"] != "freeze_committed"
        or state["fallback"] != {"status": "none", "parent_ids": []}
    ):
        raise RuntimeError("component repair is not ready for local fidelity")

    # Only the component quality state machine may authorize assembly.
    return advance_component_repair(store, page_id, _lease=_lease)


def _request_evidence_ref(
    store: RunStore,
    request_path: Path,
    record: dict,
) -> tuple[dict, bytes]:
    request_dir = request_path.parent.relative_to(store.root).as_posix()
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("legacy request evidence reference is invalid")
    reference = {
        "path": f"{request_dir}/{record.get('path', '')}",
        "sha256": record.get("sha256"),
    }
    _, payload = _load_legacy_ref(store, reference)
    return reference, payload


def _execute_legacy_round(
    store: RunStore, page_id: str, lease: ExecutionLease,
    *, performance_trace=None, visual_worker_pool=None,
) -> bool:
    import numpy as np
    from scripts.sam_worker import (
        component_prompt_batch_request_bytes,
        read_component_prompt_batch_result,
        run_component_prompt_batch_worker,
    )
    from scripts.visual_segment import RecoverableComponentPlanError

    state = store.read_json(
        f"pages/{page_id}/reconstruction/component_state.json"
    )
    request_path, request_payload = _load_legacy_ref(
        store, state["current_round"]["request_ref"]
    )
    plan_path = _state_artifact(store, state["current_round"]["plan_ref"])
    graph_path = _state_artifact(store, state["graph_ref"])
    request = json.loads(request_payload.decode("utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    source_ref, _ = _request_evidence_ref(
        store, request_path, request["evidence"]["source.png"]
    )
    source = _legacy_ref_path(store, source_ref)
    quality_record = request["evidence"]["quality-report.json"]
    if (
        not isinstance(quality_record, dict)
        or set(quality_record) != {"path", "sha256"}
        or not isinstance(quality_record["path"], str)
        or not quality_record["path"]
        or "\\" in quality_record["path"]
        or ":" in quality_record["path"]
        or PurePosixPath(quality_record["path"]).is_absolute()
        or any(
            part in {"", ".", ".."}
            for part in PurePosixPath(quality_record["path"]).parts
        )
        or not isinstance(quality_record["sha256"], str)
        or len(quality_record["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in quality_record["sha256"]
        )
    ):
        raise ValueError("previous component quality reference is invalid")
    quality_evidence = request_path.parent / Path(
        *PurePosixPath(quality_record["path"]).parts
    )
    quality_payload = _read_bound_legacy_file(
        store,
        quality_evidence,
        max_bytes=4 * 1024 * 1024,
        label="previous component quality",
    )
    if hashlib.sha256(quality_payload).hexdigest() != quality_record["sha256"]:
        raise ValueError("previous component quality sha256 mismatch")
    previous_quality = json.loads(quality_payload.decode("utf-8"))
    previous_refs = {}
    if isinstance(previous_quality.get("input_refs"), dict):
        previous_refs = dict(previous_quality["input_refs"])
    else:
        for evidence_name, ref_name in (
            ("background.png", "background"),
            ("unexplained-mask.png", "foreground_evidence"),
            ("presentation-manifest.json", "presentation_manifest"),
        ):
            record = request["evidence"].get(evidence_name)
            if isinstance(record, dict):
                previous_refs[ref_name] = _request_evidence_ref(
                    store, request_path, record
                )[0]
    previous_refs["source"] = source_ref
    current_background = (
        _state_artifact(store, previous_refs["background"])
        if isinstance(previous_refs.get("background"), dict)
        else None
    )
    previous_presentation_manifest = (
        _state_artifact(store, previous_refs["presentation_manifest"])
        if isinstance(previous_refs.get("presentation_manifest"), dict)
        else None
    )
    projected_nodes = len(graph["nodes"])
    for action in plan["actions"]:
        if action["action"] == "split":
            projected_nodes += action["parameters"]["parts"]
        elif action["action"] == "merge":
            projected_nodes += 1
    _ensure_component_disk_reserve(
        store.root / "pages" / page_id / "reconstruction",
        source,
        node_count=projected_nodes,
        repair_round=state["repair_round"],
    )
    with Image.open(source) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    output_dir = reconstruction / f"execution-{state['repair_round']:02d}"

    def sam_batch_runner(*, image, prompts):
        started = time.perf_counter()
        try:
            if visual_worker_pool is None:
                masks = run_component_prompt_batch_worker(
                    image,
                    prompts,
                    work_dir=output_dir.parent,
                )
            else:
                with tempfile.TemporaryDirectory(
                    prefix="component-sam-batch-",
                    dir=output_dir.parent,
                ) as temporary:
                    root = Path(temporary)
                    image_path = root / "image.png"
                    request_path = root / "request.json"
                    result_path = root / "result.json"
                    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
                        image_path
                    )
                    request_path.write_bytes(
                        component_prompt_batch_request_bytes(
                            tuple(image.shape[:2]),
                            prompts,
                        )
                    )
                    visual_worker_pool.request(
                        {
                            "kind": "component_prompts",
                            "request": str(request_path),
                            "result": str(result_path),
                        },
                        performance_trace=performance_trace,
                        page_id=page_id,
                    )
                    try:
                        masks = read_component_prompt_batch_result(
                            request_path,
                            result_path,
                        )
                    except RuntimeError as error:
                        if str(error) != (
                            "SAM component worker returned an invalid result batch"
                        ):
                            raise
                        masks = []
        except BaseException:
            _record_inference_performance(
                performance_trace, started, page_id, len(prompts), "error"
            )
            raise
        if type(masks) is not list or len(masks) != len(prompts):
            _record_inference_performance(
                performance_trace, started, page_id, len(prompts), "failed"
            )
            raise RuntimeError("SAM component worker returned an invalid mask count")
        if any(
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.bool_
            or mask.shape != image.shape[:2]
            or not mask.any()
            for mask in masks
        ):
            _record_inference_performance(
                performance_trace, started, page_id, len(prompts), "failed"
            )
            raise RuntimeError("SAM component worker returned an invalid mask")
        _record_inference_performance(
            performance_trace, started, page_id, len(prompts), "success"
        )
        return [
            {"component_id": prompt["component_id"], "mask": mask}
            for prompt, mask in zip(prompts, masks)
        ]

    try:
        next_graph = execute_component_action_round(
            pixels, graph, plan["actions"], sam_batch_runner=sam_batch_runner,
            input_dir=graph_path.parent, output_dir=output_dir,
        )
    except RecoverableComponentPlanError as error:
        reject_recoverable_component_plan(
            store,
            page_id,
            repair_round=state["repair_round"],
            request_ref=state["current_round"]["request_ref"],
            plan_ref=state["current_round"]["plan_ref"],
            reason=error.reason,
            _lease=lease,
        )
        return True
    output_graph = output_dir / "component-graph.json"
    rebuild_actions = [
        action for action in plan["actions"]
        if action["action"] == "rebuild_background"
    ]
    if rebuild_actions:
        repair_requests = [
            (set(action["object_ids"]), action["parameters"]["margin_ratio"])
            for action in rebuild_actions
        ]
        module = importlib.import_module("image_to_ppt")
        prepared = module.load_component_layers(
            reconstruction / "initial/prepared_page.json"
        )
        with Image.open(Path(prepared.get("_text_clean_path", source))) as image:
            text_clean = np.asarray(image.convert("RGB")).copy()
        with Image.open(Path(prepared.get(
            "_text_cleanup_mask_path", prepared["_text_mask_path"]
        ))) as image:
            text_mask = np.asarray(image.convert("L")) > 0
        _, effective_text_mask, effective_text_clean = _effective_text_context(
            source=pixels,
            text_clean=text_clean,
            text_mask=text_mask,
            text_items=prepared.get("text_items", []),
            graph=next_graph,
            graph_dir=output_dir,
            refine_text_clean=True,
            refine_cleanup_mask="_text_cleanup_mask_path" in prepared,
        )
        effective_text_mask_path = output_dir / "background-text-mask.png"
        Image.fromarray(
            effective_text_mask.astype(np.uint8) * 255, mode="L"
        ).save(effective_text_mask_path)
        effective_text_clean_path = output_dir / "background-text-clean.png"
        Image.fromarray(effective_text_clean, mode="RGB").save(
            effective_text_clean_path
        )
        _rebuild_canvas_background(
            source_path=source,
            current_background_path=(
                current_background
                if current_background is not None
                else Path(prepared["background_original_path"])
            ),
            restore_background_path=effective_text_clean_path,
            repair_requests=repair_requests,
            graph=next_graph,
            graph_dir=output_dir,
            text_mask_path=effective_text_mask_path,
            output_path=output_dir / "background-rebuilt.png",
            repair_all_active=(
                current_background is None
                or sha256_file(current_background)
                == sha256_file(Path(prepared["background_original_path"]))
            ),
        )
    refs = _quality_assets(
        store, page_id, next_graph, output_dir, output_dir,
        previous_quality_refs=previous_refs,
        background_rebuilt=bool(rebuild_actions),
        frozen_manifest_path=previous_presentation_manifest,
        frozen_component_ids=set(state["frozen"]),
    )
    execution = {
        "schema_version": 1, "page_id": page_id,
        "provider": state["provider"], "repair_round": state["repair_round"],
        "request_sha256": state["current_round"]["request_ref"]["sha256"],
        "input_graph_sha256": state["graph_ref"]["sha256"],
        "output_graph_sha256": sha256_file(output_graph),
        "executable_action_count": len(plan["actions"]),
        "quality_input_refs": refs,
    }
    execution_path = output_dir / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    record_component_execution(
        store, page_id, execution_path=execution_path,
        output_graph_path=output_graph, _lease=lease,
    )
    return False


def _publish_next_legacy_request(
    store: RunStore,
    page_id: str,
    repair_round: int,
    lease: ExecutionLease,
    *,
    visual_worker_pool=None,
    visual_worker_pool_factory=None,
    performance_trace=None,
) -> None:
    state = store.read_json(
        f"pages/{page_id}/reconstruction/component_state.json"
    )
    graph_path = _state_artifact(store, state["graph_ref"])
    if repair_round == 2:
        if visual_worker_pool is None and visual_worker_pool_factory is not None:
            visual_worker_pool = visual_worker_pool_factory()
        strict_graph_path = _prepare_fast_strict_round(
            store,
            page_id,
            repair_round,
            list(state["failed_ids"]),
            visual_worker_pool=visual_worker_pool,
            performance_trace=performance_trace,
        )
        if strict_graph_path is not None:
            graph_path = strict_graph_path
    quality_path = _state_artifact(
        store, state["current_round"]["quality_ref"]
    )
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    refs = quality["input_refs"]
    # Every repair round must keep the exact source snapshot bound into the
    # component state.  PPTX media may have been losslessly re-encoded during
    # deterministic initialization, so reopening the original candidate file
    # can produce a different byte hash even though its pixels are identical.
    source = _state_artifact(store, refs["source"])
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    native_check = json.loads(
        _state_artifact(store, refs["native_check"]).read_text(encoding="utf-8")
    )
    raw_text_items = native_check.get("text_items")
    if not isinstance(raw_text_items, list):
        raise ValueError("legacy native text_items are invalid")
    with Image.open(source) as image:
        text_items = _component_text_items(
            raw_text_items, image.size
        )
    _ensure_component_disk_reserve(
        reconstruction,
        source,
        node_count=len(graph["nodes"]),
        repair_round=repair_round,
    )
    evidence_root = reconstruction / f"evidence-round-{repair_round:02d}"
    evidence_root.mkdir(exist_ok=False)
    evidence = {}
    copies = {
        "source.png": source,
        "component-graph.json": graph_path,
        "quality-report.json": quality_path,
    }
    if "foreground_evidence" in refs:
        copies["unexplained-mask.png"] = quality_path.parent / "unexplained-mask.png"
    for name, source_path in copies.items():
        target = evidence_root / name
        shutil.copyfile(source_path, target)
        evidence[name] = target
    previous_manifest_path = _state_artifact(store, refs["presentation_manifest"])
    execution = json.loads(_state_artifact(
        store, state["current_round"]["execution_ref"]
    ).read_text(encoding="utf-8"))
    previous_manifest = _validate_presentation_manifest(
        previous_manifest_path,
        reconstruction,
        source_sha256=state["source_sha256"],
        graph_sha256=execution["output_graph_sha256"],
    )
    previous_manifest["graph_sha256"] = sha256_file(graph_path)
    presentation_manifest = evidence_root / "presentation-manifest.json"
    with presentation_manifest.open("x", encoding="utf-8") as stream:
        json.dump(
            previous_manifest,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    evidence["presentation-manifest.json"] = presentation_manifest
    shutil.copytree(graph_path.parent / "masks", evidence_root / "masks")
    evidence.update(
        _render_component_evidence(
            source_path=evidence["source.png"],
            graph=graph,
            text_mask_path=_state_artifact(store, refs["text_mask"]),
            background_path=_state_artifact(store, refs["background"]),
            presentation_manifest_path=evidence["presentation-manifest.json"],
            run_root=store.root,
            reconstruction=reconstruction,
            graph_sha256=sha256_file(evidence["component-graph.json"]),
            output_dir=evidence_root,
            text_items=text_items,
        )
    )
    session = {
        "page_id": page_id, "provider": state["provider"],
        "reconstruction_dir": reconstruction, "evidence": evidence,
    }
    request_path = build_component_agent_request(
        session, repair_round=repair_round, _lease=lease,
    )
    record_next_component_request(
        store, page_id, request_path=request_path, _lease=lease
    )


def _execute_legacy_parent_fallback(
    store: RunStore, page_id: str, lease: ExecutionLease
) -> None:
    import numpy as np

    state = store.read_json(
        f"pages/{page_id}/reconstruction/component_state.json"
    )
    graph_path = _state_artifact(store, state["graph_ref"])
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    source = _source_path(store, page_id)
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    output_dir = reconstruction / f"pf-{uuid.uuid4().hex[:12]}"
    _ensure_component_disk_reserve(
        output_dir.parent,
        source,
        node_count=len(graph["nodes"]),
        repair_round=state["repair_round"],
    )
    with Image.open(source) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    fallback_parent_ids = set(state["fallback"]["parent_ids"])
    actions = [{
        "action": "discard", "object_ids": [component_id],
        "parameters": {}, "confidence": 1.0,
        "evidence": ["deterministic parent fallback"],
    } for component_id in state["failed_ids"]
      if component_id not in fallback_parent_ids] + [{
        "action": "collapse_to_parent", "object_ids": [parent_id],
        "parameters": {}, "confidence": 1.0,
        "evidence": ["deterministic parent fallback"],
    } for parent_id in state["fallback"]["parent_ids"]]
    next_graph = execute_component_action_round(
        pixels, graph, actions, sam_runner=None,
        input_dir=graph_path.parent, output_dir=output_dir,
    )
    parent_ids = set(state["fallback"]["parent_ids"])
    for node in next_graph["nodes"]:
        if node["id"] in parent_ids:
            initial_mask = _state_artifact(store, state["parent_assets"][node["id"]])
            restored_mask = output_dir / Path(node["mask"])
            shutil.copyfile(initial_mask, restored_mask)
            with Image.open(restored_mask) as image:
                bbox = image.convert("L").getbbox()
            if bbox is None:
                raise ValueError("initial parent fallback mask is empty")
            left, top, right, bottom = bbox
            node["mask_sha256"] = sha256_file(restored_mask)
            node["bbox"] = [left, top, right, bottom]
            node["state"] = "pending_gate"
    output_graph = output_dir / "component-graph.json"
    output_graph.write_text(
        json.dumps(next_graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    quality_options = {"background_rebuilt": False}
    quality_ref = state["current_round"].get("quality_ref")
    trusted_quality = isinstance(quality_ref, dict)
    if trusted_quality:
        previous_quality_payload = _load_legacy_ref(
            store, quality_ref, max_bytes=4 * 1024 * 1024
        )[1]
    else:
        previous_quality_path = graph_path.with_name("quality-report.json")
        previous_quality_payload = (
            _read_bound_legacy_file(
                store,
                previous_quality_path,
                max_bytes=4 * 1024 * 1024,
                label="previous component quality",
            )
            if previous_quality_path.is_file()
            else None
        )
    trusted_refs = {}
    if trusted_quality and previous_quality_payload is not None:
        previous_quality = json.loads(previous_quality_payload.decode("utf-8"))
        trusted_refs.update(previous_quality["input_refs"])
    request_ref = state["current_round"].get("request_ref")
    if isinstance(request_ref, dict):
        request_path, request_payload = _load_legacy_ref(store, request_ref)
        request = json.loads(request_payload.decode("utf-8"))
        for evidence_name, ref_name in (
            ("source.png", "source"),
            ("unexplained-mask.png", "foreground_evidence"),
            ("background.png", "background"),
            ("presentation-manifest.json", "presentation_manifest"),
        ):
            record = request["evidence"].get(evidence_name)
            if ref_name not in trusted_refs and isinstance(record, dict):
                trusted_refs[ref_name] = _request_evidence_ref(
                    store, request_path, record
                )[0]
    if trusted_refs:
        quality_options["previous_quality_refs"] = trusted_refs
    frozen_ids = set(state["frozen"])
    if frozen_ids and isinstance(trusted_refs.get("presentation_manifest"), dict):
        quality_options["frozen_manifest_path"] = _state_artifact(
            store, trusted_refs["presentation_manifest"]
        )
        quality_options["frozen_component_ids"] = frozen_ids
    refs = _quality_assets(
        store, page_id, next_graph, output_dir, output_dir,
        **quality_options,
    )
    record_parent_fallback_execution(
        store, page_id, graph_path=output_graph,
        quality_input_refs=refs, _lease=lease,
    )


def _native_pdf_box(
    value: object, page_height: float, *, allow_line: bool = False,
) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            type(number) not in {int, float} or not math.isfinite(number)
            for number in value
        )
    ):
        raise ValueError("native PDF object bbox is invalid")
    left, bottom, right, top = (float(number) for number in value)
    if (
        right < left
        or top < bottom
        or (not allow_line and (right == left or top == bottom))
        or (allow_line and right == left and top == bottom)
    ):
        raise ValueError("native PDF object bbox is empty")
    return [left, page_height - top, right, page_height - bottom]


def _native_pdf_slide_data(
    store: RunStore, page_id: str, state: dict, reconstruction: Path,
) -> dict:
    _, payload = _load_legacy_ref(store, state.get("native_page_ref"))
    document = json.loads(payload.decode("utf-8"))
    if document.get("page_id") != page_id or not isinstance(
        document.get("analysis"), dict
    ):
        raise ValueError("native PDF page artifact is invalid")
    analysis = document["analysis"]
    width = analysis.get("width_pt")
    height = analysis.get("height_pt")
    if (
        type(width) not in {int, float}
        or type(height) not in {int, float}
        or not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("native PDF page dimensions are invalid")

    output_dir = Path(tempfile.mkdtemp(prefix="native-assembly-", dir=reconstruction))
    output_identity = _directory_identity(output_dir.lstat())
    try:
        background = output_dir / "background.png"
        Image.new("RGB", (1, 1), "white").save(background)
        elements = []
        for item in sorted(analysis["objects"], key=lambda value: value["z_index"]):
            object_id = item["id"]
            z_index = item["z_index"]
            object_type = item["type"]
            is_line = (
                object_type == "shape" and item.get("shape_type") == "line"
            )
            box = _native_pdf_box(
                item["bbox_pt"], float(height), allow_line=is_line
            )
            left, top, right, bottom = box
            if object_type in {"image", "patch"}:
                _, image_payload = _load_legacy_ref(store, {
                    "path": item["asset_path"],
                    "sha256": item["asset_sha256"],
                })
                elements.append({
                    "object_id": object_id,
                    "route": "native_image",
                    "z_index": z_index,
                    "component": {
                        "blob": image_payload,
                        "x": left,
                        "y": top,
                        "w": right - left,
                        "h": bottom - top,
                        **({"crop": item["crop"]} if "crop" in item else {}),
                    },
                })
            elif object_type == "text":
                color = item.get("color_rgb", [0, 0, 0])
                elements.append({
                    "object_id": object_id,
                    "route": "native_text",
                    "z_index": z_index,
                    "text": {
                        "box": [left, top, right - left, bottom - top],
                        "text": item["text"],
                        "font": str(item.get("font") or "Arial").lstrip("/"),
                        "font_size_pt": item["font_size"],
                        "character_spacing_pt": item.get(
                            "character_spacing_pt", 0.0
                        ),
                        "fill_opacity": item.get("fill_opacity", 1.0),
                        "bold": item.get("bold", False),
                        "italic": item.get("italic", False),
                        "color": "#" + "".join(
                            f"{int(channel):02x}" for channel in color
                        ),
                        "align": 0,
                    },
                })
            elif object_type == "shape":
                shape = {
                    "shape_type": item["shape_type"],
                    "fill_rgb": (
                        item.get("stroke_rgb", [0, 0, 0])
                        if item["shape_type"] == "line"
                        else item.get("fill_rgb")
                    ),
                    "stroke_rgb": item.get("stroke_rgb"),
                    "line_width": item.get("line_width", 1.0),
                    "fill_opacity": item.get("fill_opacity", 1.0),
                    "stroke_opacity": item.get("stroke_opacity", 1.0),
                }
                if item["shape_type"] == "line":
                    shape["line_start"] = [
                        item["line_start"][0],
                        float(height) - item["line_start"][1],
                    ]
                    shape["line_end"] = [
                        item["line_end"][0],
                        float(height) - item["line_end"][1],
                    ]
                elements.append({
                    "object_id": object_id,
                    "route": "native_shape",
                    "z_index": z_index,
                    "bbox": box,
                    "shape": shape,
                })
        return {
            "img_width": float(width),
            "img_height": float(height),
            "background_path": str(background),
            "background_original_path": str(background),
            "background_widescreen_path": str(background),
            "background_rgb": [255, 255, 255],
            "original_image_path": str(background),
            "components": [],
            "text_items": [],
            "visual_elements": elements,
            "_assembly_assets_dir": str(output_dir),
        }
    except Exception:
        _safe_rmtree(output_dir, output_identity)
        raise


class _PublishedLegacyOutputs(dict[str, Any]):
    def __init__(self, outputs: dict[str, Any], records: list[tuple]) -> None:
        super().__init__(outputs)
        self.records = records


def assemble_legacy_results(store: RunStore) -> dict[str, Any]:
    manifest = store.read_json("job_manifest.json")
    page_ids = manifest["pages"]
    output_format = manifest.get("output_format", "pptx")
    output_path = manifest["options"]["output_path"]
    if output_path is None:
        output_path = (
            store.root / "final" / "output.psd"
            if output_format == "psd" and len(page_ids) == 1
            else store.root / "final"
            if output_format == "psd"
            else store.root / "final" / "output.pptx"
        )
    output_path = Path(output_path).resolve()
    slide_size = manifest["options"]["slide_size"]
    targets = (
        _legacy_psd_output_targets(manifest, output_path)
        if output_format == "psd"
        else _legacy_output_targets(output_path, slide_size)
    )
    if any(path.exists() or path.is_symlink() for path in targets.values()):
        existing = next(
            path for path in targets.values()
            if path.exists() or path.is_symlink()
        )
        raise RuntimeError(f"Refusing to overwrite existing output: {existing}")
    warning_pages = [
        page_id for page_id in page_ids
        if store.read_json(
            f"pages/{page_id}/reconstruction/component_state.json"
        )["status"] == "preserved_with_warning"
    ]
    if (
        warning_pages
        and output_format == "pptx"
        and manifest["input"]["type"] in {"images", "pdf"}
    ):
        raise RuntimeError(
            "editable reconstruction incomplete; no PPTX was created for "
            + ", ".join(warning_pages)
        )
    module = importlib.import_module("image_to_ppt")
    slides = []
    page_records = []
    assembly_asset_dirs = []
    try:
        for page_id in page_ids:
            reconstruction = store.root / "pages" / page_id / "reconstruction"
            state = store.read_json(
                f"pages/{page_id}/reconstruction/component_state.json"
            )
            if state.get("route") == "pdf_native":
                slide = _native_pdf_slide_data(
                    store, page_id, state, reconstruction
                )
                asset_dir = Path(slide.pop("_assembly_assets_dir"))
                assembly_asset_dirs.append(
                    (asset_dir, _directory_identity(asset_dir.lstat()))
                )
                slides.append(slide)
                page_records.append((page_id, state, None, None))
                continue
            if output_format == "psd" and state["status"] == "preserved_with_warning":
                raise RuntimeError(
                    f"PSD output requires every page to pass the quality gate: {page_id}"
                )
            prepared = module.load_component_layers(
                reconstruction / "initial" / "prepared_page.json"
            )
            if state["status"] == "preserved_with_warning":
                source = _source_path(store, page_id)
                slides.append({
                    **prepared,
                    "background_path": str(source),
                    "background_original_path": str(source),
                    "background_widescreen_path": str(source),
                    "original_image_path": str(source),
                    "components": [],
                    "text_items": [],
                })
                page_records.append((page_id, state, None, None))
                continue
            result_path, result_payload = _load_legacy_ref(
                store, state["result_ref"]
            )
            result = json.loads(result_payload.decode("utf-8"))
            if result["status"] != "ready_for_assembly":
                raise ValueError("component result is not ready for assembly")
            slide = _accepted_slide_data(
                store,
                reconstruction,
                prepared,
                result,
                component_result_path=result_path,
            )
            route_result_ref = slide.pop("_route_result_ref")
            asset_dir = Path(slide.pop("_assembly_assets_dir"))
            assembly_asset_dirs.append(
                (asset_dir, _directory_identity(asset_dir.lstat()))
            )
            slides.append(slide)
            page_records.append((page_id, state, result, route_result_ref))
    except Exception:
        _cleanup_legacy_assembly_assets(assembly_asset_dirs)
        raise

    staged = {}
    published = {}
    planned_records = {}
    published_records = []
    staging_records = {}
    try:
        for index, (variant, target) in enumerate(targets.items()):
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, staging_name = tempfile.mkstemp(
                prefix=f".{target.stem}.",
                suffix=".staging.psd" if output_format == "psd" else ".staging",
                dir=target.parent,
            )
            os.close(fd)
            staging = Path(staging_name)
            staging.unlink()
            try:
                if output_format == "psd":
                    slide = slides[index]
                    assemble_psd(
                        background_path=slide["background_original_path"],
                        components=slide["components"],
                        text_items=slide["text_items"],
                        img_width=slide["img_width"],
                        img_height=slide["img_height"],
                        output_path=staging,
                    )
                    if not staging.is_file() or staging.stat().st_size == 0:
                        raise RuntimeError("PSD assembler did not produce output")
                elif len(slides) == 1:
                    module._assemble_prepared_slide(
                        slides[0], staging, False, variant
                    )
                else:
                    module.assemble_pptx_multi(
                        slides, staging, add_reference=False,
                        slide_size=variant,
                        original_aspect_ratio=manifest["input"].get(
                            "page_aspect_ratio"
                        ),
                    )
                if output_format != "psd":
                    presentation = Presentation(staging)
                    if len(presentation.slides) != len(slides):
                        raise RuntimeError("PPTX reopen slide count mismatch")
                staged[variant] = staging
            except Exception:
                if staging.exists():
                    staging.unlink()
                raise

        for variant, target in targets.items():
            staging = staged[variant]
            identity = _legacy_output_identity(staging)
            digest = sha256_file(staging)
            if _legacy_output_identity(staging) != identity:
                raise RuntimeError("Legacy staged output changed before publication")
            planned_records[variant] = (target, identity, digest)
            staging_records[variant] = (staging, identity, digest)
        _write_legacy_staging_records(store, list(staged.values()))
        _write_legacy_output_records(store, planned_records)

        for variant, target in targets.items():
            staging = staged[variant]
            try:
                os.link(staging, target)
            except Exception:
                raise
            record = planned_records[variant]
            published_records.append(record)
            _verify_legacy_output_record(record)
            staging_record = staging_records[variant]
            _remove_legacy_outputs([staging_record])
            del staging_records[variant]
            published[variant] = str(target)
        _record_legacy_delivery(
            store,
            page_records,
            published,
            published_records,
            output_format=output_format,
        )
        for record in published_records:
            _verify_legacy_output_record(record)
        _clear_legacy_staging_records(store)
    except Exception as error:
        cleanup_error = None
        try:
            _remove_legacy_outputs(list(reversed(published_records)))
        except Exception as caught:
            cleanup_error = caught
        if cleanup_error is None:
            _clear_legacy_output_records(store)
            try:
                _remove_legacy_outputs(
                    list(reversed(list(staging_records.values())))
                )
            except Exception as caught:
                cleanup_error = caught
            if cleanup_error is None:
                _clear_legacy_staging_records(store)
        if cleanup_error is not None:
            raise error from cleanup_error
        raise
    finally:
        for staging in staged.values():
            if staging.exists():
                record = next(
                    (
                        value for value in staging_records.values()
                        if value[0] == staging
                    ),
                    None,
                )
                if record is not None:
                    try:
                        _remove_legacy_outputs([record])
                    except Exception:
                        pass
                else:
                    staging.unlink()
        _cleanup_legacy_assembly_assets(assembly_asset_dirs)

    return _PublishedLegacyOutputs(published, published_records)


def _cleanup_native_pdf_sources(
    store: RunStore,
    page_ids: list[str],
) -> None:
    for page_id in page_ids:
        state = store.read_json(
            f"pages/{page_id}/reconstruction/component_state.json"
        )
        if state.get("route") != "pdf_native":
            continue
        page = store.root / "pages" / page_id
        for name in ("source.png", "source_detail.png"):
            try:
                (page / name).unlink(missing_ok=True)
            except OSError:
                _LOGGER.warning("Native PDF render cleanup failed: %s", page / name)
        assets = page / "pdf-assets"
        try:
            status = assets.lstat()
        except FileNotFoundError:
            continue
        try:
            _safe_rmtree(assets, _directory_identity(status))
        except (OSError, RuntimeError):
            _LOGGER.warning("Native PDF asset cleanup failed: %s", assets)


def _cleanup_legacy_assembly_assets(
    directories: list[tuple[Path, tuple[int, int]]],
) -> None:
    for path, identity in reversed(directories):
        _safe_rmtree(path, identity)


def _legacy_output_identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or _is_link_or_reparse(status):
        raise RuntimeError(f"Legacy output is not a regular file: {path}")
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_legacy_output_records(
    store: RunStore,
    records: dict[str, tuple[Path, tuple[int, int, int, int, int], str]],
) -> None:
    _write_legacy_file_records(store, "legacy_output_records.json", records)


def _write_legacy_file_records(
    store: RunStore,
    filename: str,
    records: dict[str, tuple[Path, tuple[int, int, int, int, int], str]],
) -> None:
    store.write_json(
        filename,
        {
            "schema_version": 1,
            "outputs": {
                name: {
                    "path": str(path),
                    "dev": identity[0],
                    "ino": identity[1],
                    "mode": identity[2],
                    "size": identity[3],
                    "mtime_ns": identity[4],
                    "sha256": digest,
                }
                for name, (path, identity, digest) in records.items()
            },
        },
    )


def _write_legacy_staging_records(
    store: RunStore,
    paths: list[Path],
) -> None:
    records = {
        str(index): (
            path,
            _legacy_output_identity(path),
            sha256_file(path),
        )
        for index, path in enumerate(paths)
    }
    _write_legacy_file_records(store, "legacy_staging_records.json", records)


def _load_legacy_output_records(
    store: RunStore,
    *,
    expected_outputs: dict[str, str] | None = None,
) -> list[tuple[Path, tuple[int, int, int, int, int], str]]:
    return _load_legacy_file_records(
        store, "legacy_output_records.json", expected_outputs=expected_outputs
    )


def _load_legacy_file_records(
    store: RunStore,
    filename: str,
    *,
    expected_outputs: dict[str, str] | None = None,
) -> list[tuple[Path, tuple[int, int, int, int, int], str]]:
    try:
        document = store.read_json(filename)
    except FileNotFoundError:
        if expected_outputs:
            raise RuntimeError("Legacy output records are missing")
        return []
    outputs = document.get("outputs") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or not isinstance(outputs, dict)
        or not outputs
        or any(type(name) is not str or not name for name in outputs)
    ):
        raise RuntimeError("Legacy output records are invalid")
    if expected_outputs is not None and {
        name: record.get("path") if isinstance(record, dict) else None
        for name, record in outputs.items()
    } != expected_outputs:
        raise RuntimeError("Legacy output records do not match outputs")
    records = []
    expected_keys = {
        "path", "dev", "ino", "mode", "size", "mtime_ns", "sha256"
    }
    for record in outputs.values():
        if (
            not isinstance(record, dict)
            or set(record) != expected_keys
            or type(record["path"]) is not str
            or any(
                type(record[name]) is not int
                for name in ("dev", "ino", "mode", "size", "mtime_ns")
            )
            or not _is_sha256(record["sha256"])
        ):
            raise RuntimeError("Legacy output record is invalid")
        records.append((
            Path(record["path"]),
            (
                record["dev"],
                record["ino"],
                record["mode"],
                record["size"],
                record["mtime_ns"],
            ),
            record["sha256"],
        ))
    return records


def _load_legacy_staging_records(
    store: RunStore,
) -> list[tuple[Path, tuple[int, int, int, int, int], str]]:
    return _load_legacy_file_records(store, "legacy_staging_records.json")


def _clear_legacy_output_records(store: RunStore) -> None:
    (store.root / "legacy_output_records.json").unlink(missing_ok=True)


def _clear_legacy_staging_records(store: RunStore) -> None:
    (store.root / "legacy_staging_records.json").unlink(missing_ok=True)


def _verify_legacy_output_record(
    record: tuple[Path, tuple[int, int, int, int, int], str],
) -> None:
    output, expected_identity, expected_hash = record
    if _legacy_output_identity(output) != expected_identity:
        raise RuntimeError("Legacy output identity changed after publication")
    if sha256_file(output) != expected_hash:
        raise RuntimeError("Legacy output hash changed after publication")
    if _legacy_output_identity(output) != expected_identity:
        raise RuntimeError("Legacy output changed during verification")


def _remove_legacy_output(
    record: tuple[Path, tuple[int, int, int, int, int], str],
) -> None:
    _remove_legacy_outputs([record])


def _remove_legacy_outputs(
    records: list[tuple[Path, tuple[int, int, int, int, int], str]],
) -> None:
    isolated_records = []
    try:
        for output, expected_identity, expected_hash in records:
            descriptor, isolated_value = tempfile.mkstemp(
                dir=output.parent,
                prefix=f".{output.name}.recovery-",
                suffix=".tmp",
            )
            os.close(descriptor)
            isolated = Path(isolated_value)
            isolated.unlink()
            try:
                os.replace(output, isolated)
            except FileNotFoundError as error:
                isolated.unlink(missing_ok=True)
                raise RuntimeError(
                    "Legacy output disappeared during removal"
                ) from error
            isolated_records.append((isolated, output))
            identity = _legacy_output_identity(isolated)
            digest = sha256_file(isolated)
            stable_identity = _legacy_output_identity(isolated)
            if (
                identity != expected_identity
                or stable_identity != expected_identity
                or digest != expected_hash
            ):
                raise RuntimeError("Legacy output changed and was not removed")

        for isolated, _ in isolated_records:
            isolated.unlink()
        for _, output in isolated_records:
            if output.exists() or output.is_symlink():
                raise RuntimeError("A new legacy output appeared during removal")
    except Exception as error:
        restoration_error = None
        for isolated, output in reversed(isolated_records):
            if not isolated.exists():
                continue
            try:
                if output.exists() or output.is_symlink():
                    raise RuntimeError(
                        f"Concurrent legacy output was preserved at {isolated}"
                    )
                os.replace(isolated, output)
            except Exception as caught:
                if restoration_error is None:
                    restoration_error = caught
        if restoration_error is not None:
            raise error from restoration_error
        raise


def _legacy_output_targets(output_path: Path, slide_size: str) -> dict[str, Path]:
    if slide_size != "both":
        return {slide_size: output_path}
    base = output_path.with_suffix("")
    return {
        "original": Path(f"{base}_original.pptx"),
        "16:9": Path(f"{base}_16x9.pptx"),
    }


def _legacy_psd_output_targets(
    manifest: dict[str, Any], output_path: Path
) -> dict[str, Path]:
    page_ids = manifest["pages"]
    if len(page_ids) == 1:
        return {page_ids[0]: output_path}
    items = manifest.get("input", {}).get("items", [])
    if len(items) != len(page_ids):
        raise ValueError("PSD image manifest does not match page count")
    stems = [Path(item["original_path"]).stem for item in items]
    duplicate_stems = {stem for stem in stems if stems.count(stem) > 1}
    return {
        page_id: output_path / (
            f"{index:03d}_{stem}.psd" if stem in duplicate_stems else f"{stem}.psd"
        )
        for index, (page_id, stem) in enumerate(zip(page_ids, stems), start=1)
    }


def _record_legacy_delivery(
    store: RunStore,
    page_records: list[tuple[str, dict, dict | None, dict | None]],
    outputs: dict[str, str],
    output_records: list[tuple[Path, tuple[int, int, int, int, int], str]],
    *,
    output_format: str = "pptx",
) -> None:
    hashes = {str(path): digest for path, _, digest in output_records}
    output_refs = {
        name: {"path": path, "sha256": hashes[path]}
        for name, path in outputs.items()
    }
    for page_id, state, result, route_result_ref in page_records:
        delivery = {
            "schema_version": 1,
            "page_id": page_id,
            "status": state["status"],
            "delivery_checks": {
                "psd_save" if output_format == "psd" else "pptx_reopen": "pass"
            },
            "outputs": output_refs,
        }
        if result is None and state["status"] == "preserved_with_warning":
            delivery["warning"] = (
                "Component reconstruction did not pass the parent gate; "
                "the full source image was preserved."
            )
        if route_result_ref is not None:
            delivery["route_result"] = route_result_ref
        store.write_json(
            f"pages/{page_id}/reconstruction/component_delivery.json",
            delivery,
        )


def _read_bound_legacy_file(
    store: RunStore,
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    try:
        return _read_bound_file(
            path,
            store.root,
            max_bytes=max_bytes,
            label=label,
        )
    except (OSError, RuntimeError):
        raise ValueError("legacy artifact could not be read") from None


def _load_legacy_ref(
    store: RunStore,
    reference: dict,
    *,
    max_bytes: int = 256 * 1024 * 1024,
) -> tuple[Path, bytes]:
    path = _legacy_ref_path(store, reference)
    payload = _read_bound_legacy_file(
        store,
        path,
        max_bytes=max_bytes,
        label="legacy artifact",
    )
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError("legacy artifact sha256 mismatch")
    return path, payload


def _legacy_ref_path(store: RunStore, reference: dict) -> Path:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("legacy artifact reference is invalid")
    if (
        not isinstance(reference["path"], str)
        or not isinstance(reference["sha256"], str)
        or len(reference["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in reference["sha256"]
        )
    ):
        raise ValueError("legacy artifact reference is invalid")
    raw_path = reference["path"]
    parts = raw_path.split("/")
    windows_devices = {"CON", "PRN", "AUX", "NUL", "CLOCK$"} | {
        f"{prefix}{suffix}"
        for prefix in ("COM", "LPT")
        for suffix in (*"123456789", "¹", "²", "³")
    }
    if (
        not raw_path
        or "\\" in raw_path
        or ":" in raw_path
        or any(
            not part
            or part in {".", ".."}
            or part[-1] in {".", " "}
            or part.split(".", 1)[0].rstrip(" ").upper() in windows_devices
            for part in parts
        )
    ):
        raise ValueError("legacy artifact reference is invalid")
    return store.root.joinpath(*parts)


def _accepted_reconstruction_inputs(
    store: RunStore,
    *,
    prepared: dict,
    result: dict,
    graph: dict,
    components: list[dict],
) -> dict:
    root = store.root.resolve()
    component_assets = {}
    for component in components:
        path = Path(component["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("accepted component asset escapes Run directory")
        component_assets[component["component_id"]] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
    return {
        "page_id": result["page_id"],
        "canvas": (prepared["img_width"], prepared["img_height"]),
        "graph": graph,
        "component_assets": component_assets,
        "text_items": result.get("text_items", prepared.get("text_items", [])),
    }


def _load_local_fidelity_components(
    store: RunStore,
    result: dict,
    *,
    page_size: tuple[int, int],
    output_dir: Path,
) -> list[dict]:
    reference = result.get("local_fidelity_ref")
    route = result.get("route")
    if reference is None and route is None:
        return []
    if route != "local_fidelity" or reference is None:
        raise ValueError("local fidelity result binding is invalid")
    _, payload = _load_legacy_ref(store, reference, max_bytes=16 * 1024 * 1024)
    manifest = json.loads(payload.decode("utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "page_id", "source_sha256", "components",
        "residual_pixels", "uncovered_pixels",
    }:
        raise ValueError("local fidelity manifest is invalid")
    width, height = page_size
    page_area = width * height
    if (
        manifest["schema_version"] != 1
        or type(manifest["schema_version"]) is not int
        or manifest["page_id"] != result.get("page_id")
        or manifest["source_sha256"]
        != result["accepted_asset_refs"]["source"]["sha256"]
        or type(manifest["residual_pixels"]) is not int
        or not 0 <= manifest["residual_pixels"] <= page_area
        or manifest["uncovered_pixels"] != 0
        or type(manifest["uncovered_pixels"]) is not int
        or not isinstance(manifest["components"], list)
        or len(manifest["components"]) > page_area
    ):
        raise ValueError("local fidelity manifest is invalid")

    import numpy as np

    covered = np.zeros((height, width), dtype=bool)
    components = []
    for index, item in enumerate(manifest["components"], start=1):
        if not isinstance(item, dict) or set(item) != {
            "ref", "bbox", "coverage"
        }:
            raise ValueError("local fidelity component is invalid")
        bbox = item["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
        ):
            raise ValueError("local fidelity component bbox is invalid")
        left, top, right, bottom = bbox
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("local fidelity component bbox is invalid")
        expected_coverage = (right - left) * (bottom - top) / page_area
        coverage = item["coverage"]
        if (
            type(coverage) not in {int, float}
            or not math.isfinite(coverage)
            or not math.isclose(
                float(coverage), expected_coverage, rel_tol=1e-9, abs_tol=1e-12
            )
            or not 0 < coverage <= 0.35
        ):
            raise ValueError("local fidelity component coverage is invalid")
        _, patch_payload = _load_legacy_ref(
            store,
            item["ref"],
            max_bytes=max(1024 * 1024, (right - left) * (bottom - top) * 8),
        )
        with Image.open(io.BytesIO(patch_payload)) as image:
            if image.format != "PNG" or image.mode != "RGBA" or image.size != (
                right - left, bottom - top
            ):
                raise ValueError("local fidelity component image is invalid")
            alpha = np.asarray(image.getchannel("A")) > 0
        if not np.any(alpha) or np.any(covered[top:bottom, left:right] & alpha):
            raise ValueError("local fidelity component alpha is invalid")
        covered[top:bottom, left:right] |= alpha
        snapshot = output_dir / f"local-fidelity-{index:04d}.png"
        snapshot.write_bytes(patch_payload)
        components.append({
            "component_id": f"local_fidelity_{index:04d}",
            "path": str(snapshot),
            "x": left,
            "y": top,
            "w": right - left,
            "h": bottom - top,
        })
    if int(np.count_nonzero(covered)) != manifest["residual_pixels"]:
        raise ValueError("local fidelity residual count is invalid")
    return components


def _replace_visual_routes(elements: list[dict], routed: list[dict]) -> list[dict]:
    # Routing covers frozen objects; locally preserved objects still need delivery.
    replacements = {item["object_id"]: item for item in routed}
    return [replacements.get(item["object_id"], item) for item in elements]


def _accepted_slide_data(
    store: RunStore,
    reconstruction: Path,
    prepared: dict,
    result: dict,
    *,
    component_result_path: Path | None = None,
) -> dict:
    import numpy as np

    refs = result["accepted_asset_refs"]
    expected_refs = {
        "source", "background", "reconstructed", "text_mask",
        "native_check", "presentation_manifest",
    }
    if not isinstance(refs, dict) or frozenset(refs) not in {
        frozenset(expected_refs),
        frozenset({*expected_refs, "foreground_evidence"}),
    }:
        raise ValueError("accepted presentation references are invalid")
    for reference in refs.values():
        _legacy_ref_path(store, reference)
    asset_payloads = {
        name: _load_legacy_ref(store, refs[name])[1]
        for name in ("source", "background")
    }
    manifest_path = _legacy_ref_path(store, refs["presentation_manifest"])
    graph_path, graph_payload = _load_legacy_ref(store, result["graph_ref"])
    graph = validate_component_graph(json.loads(graph_payload.decode("utf-8")))
    accepted_graph_sha256 = result.get("accepted_graph_sha256")
    if (
        not isinstance(accepted_graph_sha256, str)
        or len(accepted_graph_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in accepted_graph_sha256
        )
    ):
        raise ValueError("accepted presentation graph hash is invalid")
    by_id = {node["id"]: node for node in graph["nodes"]}
    final_ids = result["final_component_ids"]
    if len(final_ids) != len(set(final_ids)) or any(
        component_id not in by_id for component_id in final_ids
    ):
        raise ValueError("component result final IDs are invalid")
    active_nodes = _active_visual_nodes(graph)
    active_ids = [node["id"] for node in active_nodes]
    if set(final_ids) != set(active_ids):
        raise ValueError("component result final IDs do not match graph")
    with Image.open(io.BytesIO(asset_payloads["source"])) as image:
        page_size = image.size
    for node in active_nodes:
        mask_path = (graph_path.parent / Path(node["mask"])).resolve()
        if not mask_path.is_relative_to(store.root.resolve()):
            raise ValueError("final component mask escapes Run directory")
        _, mask_payload = _load_legacy_ref(store, {
            "path": mask_path.relative_to(store.root.resolve()).as_posix(),
            "sha256": node["mask_sha256"],
        })
        with Image.open(io.BytesIO(mask_payload)) as image:
            mask = image.convert("L")
            try:
                if mask.size != page_size or mask.getbbox() != tuple(node["bbox"]):
                    raise ValueError(
                        f"final component bbox is invalid: {node['id']}"
                    )
            finally:
                mask.close()
    output_dir = Path(tempfile.mkdtemp(prefix="assembly-assets-", dir=reconstruction))
    output_identity = _directory_identity(output_dir.lstat())
    try:
        asset_paths = {}
        for name in ("source", "background"):
            payload = asset_payloads[name]
            snapshot = output_dir / f"accepted-{name}.asset"
            snapshot.write_bytes(payload)
            asset_paths[name] = snapshot
        components = []
        layers = _load_presentation_assets(
            run_root=store.root,
            reconstruction=reconstruction,
            manifest_path=manifest_path,
            source_sha256=refs["source"]["sha256"],
            graph_sha256=accepted_graph_sha256,
            graph=graph,
            page_size=page_size,
            component_ids=active_ids,
            expected_manifest_sha256=refs["presentation_manifest"]["sha256"],
        )
        for index, layer in enumerate(layers, start=1):
            component_id = layer["component_id"]
            node = by_id[component_id]
            alpha = layer["rgba"][:, :, 3] == 255
            ys, xs = np.nonzero(alpha)
            if not len(xs):
                raise ValueError(f"final component became empty: {component_id}")
            left, right = int(xs.min()), int(xs.max()) + 1
            top, bottom = int(ys.min()), int(ys.max()) + 1
            component_path = output_dir / f"component-{index:04d}.png"
            Image.fromarray(
                layer["rgba"][top:bottom, left:right], mode="RGBA"
            ).save(component_path)
            components.append({
                "component_id": component_id,
                "path": str(component_path), "x": left, "y": top,
                "w": right - left, "h": bottom - top,
                "z_index": node["z_index"],
            })
        reconstruction_inputs = _accepted_reconstruction_inputs(
            store,
            prepared={
                **prepared,
                "img_width": prepared.get("img_width", page_size[0]),
                "img_height": prepared.get("img_height", page_size[1]),
            },
            result={
                **result,
                "page_id": result.get("page_id", reconstruction.parent.name),
            },
            graph=graph,
            components=components,
        )
        visual_elements = [
            {
                "object_id": component["component_id"],
                "route": "raster_component",
                "z_index": component["z_index"],
                "component": component,
            }
            for component in components
        ]
        route_result_ref = None
        if component_result_path is not None:
            from image2editable.route_execution import (
                load_published_route,
                route_visual_elements,
            )

            published_route = load_published_route(
                store,
                component_result_path,
                page_id=result["page_id"],
            )
            if published_route is not None:
                visual_elements = _replace_visual_routes(
                    visual_elements,
                    route_visual_elements(
                        store, published_route["ir"], published_route["plan"],
                    ),
                )
                route_result_ref = published_route["result_ref"]
        fidelity_components = _load_local_fidelity_components(
            store,
            result,
            page_size=page_size,
            output_dir=output_dir,
        )
        next_z_index = max(
            (item["z_index"] for item in visual_elements), default=-1
        ) + 1
        for index, component in enumerate(fidelity_components):
            component["z_index"] = next_z_index + index
            visual_elements.append({
                "object_id": component["component_id"],
                "route": "raster_component",
                "z_index": component["z_index"],
                "component": component,
            })
        components.extend(fidelity_components)
        return {
            **prepared,
            "text_items": result.get(
                "text_items", prepared.get("text_items", [])
            ),
            "background_path": str(asset_paths["background"]),
            "background_original_path": str(asset_paths["background"]),
            "background_widescreen_path": str(asset_paths["background"]),
            "original_image_path": str(asset_paths["source"]),
            "components": sorted(components, key=lambda item: item["z_index"]),
            "visual_elements": visual_elements,
            "_reconstruction_ir_inputs": reconstruction_inputs,
            "_assembly_assets_dir": str(output_dir),
            "_route_result_ref": route_result_ref,
        }
    except Exception:
        _safe_rmtree(output_dir, output_identity)
        raise


def assemble_route_candidate(
    store: RunStore,
    component_result_path: Path,
    plan: dict,
    output_path: Path,
) -> None:
    """Assemble one unpublished candidate used only by authoritative render QA."""

    reconstruction = component_result_path.parent
    component_payload = _read_bound_file(
        component_result_path,
        store.root,
        max_bytes=256 * 1024 * 1024,
        label="component result",
    )
    result = json.loads(component_payload.decode("utf-8"))
    module = importlib.import_module("image_to_ppt")
    prepared = module.load_component_layers(
        reconstruction / "initial" / "prepared_page.json"
    )
    slide = _accepted_slide_data(store, reconstruction, prepared, result)
    asset_dir = Path(slide["_assembly_assets_dir"])
    asset_identity = _directory_identity(asset_dir.lstat())
    try:
        ir_payload = _read_bound_file(
            reconstruction / "route" / "reconstruction-ir.json",
            store.root,
            max_bytes=256 * 1024 * 1024,
            label="reconstruction IR",
        )
        from image2editable.route_execution import route_visual_elements

        slide["visual_elements"] = _replace_visual_routes(
            slide["visual_elements"],
            route_visual_elements(store, json.loads(ir_payload.decode("utf-8")), plan),
        )
        module._assemble_prepared_slide(slide, output_path, False, "original")
    finally:
        _safe_rmtree(asset_dir, asset_identity)
