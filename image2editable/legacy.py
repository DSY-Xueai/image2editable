from __future__ import annotations

from contextlib import redirect_stdout
import ctypes
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any

from PIL import Image
from pptx import Presentation

from image2editable.contracts import validate_schema_version
from image2editable.component_repair import (
    EVIDENCE_NAMES,
    advance_component_repair,
    build_component_agent_request,
    execute_component_action_round,
    initialize_component_repair_state,
    record_component_execution,
    record_component_quality,
    record_next_component_request,
    record_parent_fallback_execution,
    record_parent_fallback_quality,
)
from image2editable.inputs import sha256_file
from image2editable.store import RunStore
from image2editable.execution import ExecutionLease


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


def initialize_legacy_page(
    store: RunStore, page_id: str, *, _lease: ExecutionLease
) -> dict[str, Any]:
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    state_path = reconstruction / "component_state.json"
    if state_path.is_file():
        return {"status": "already_initialized", "page_id": page_id}
    manifest = store.read_json("job_manifest.json")
    source = _source_path(store, page_id)
    prepared = importlib.import_module("image_to_ppt").prepare_component_layers(
        source,
        reconstruction / "initial",
        lang=manifest["options"]["lang"],
        resource_isolation=True,
    )
    session = _build_initial_page_session(
        store, page_id, prepared, reconstruction
    )
    request_path = build_component_agent_request(session, repair_round=1)
    initialize_component_repair_state(
        store, page_id, request_path=request_path,
        initial_component_count=prepared["initial_component_count"],
        _lease=_lease,
    )
    return {"status": "initialized", "page_id": page_id}


def _build_initial_page_session(
    store: RunStore, page_id: str, prepared: dict, reconstruction: Path
) -> dict:
    evidence_root = reconstruction / "evidence-source"
    evidence_root.mkdir(parents=True, exist_ok=False)
    masks_root = evidence_root / "masks"
    masks_root.mkdir()

    sources = {
        "source.png": Path(prepared["original_image_path"]),
        "numbered-masks.png": Path(prepared["original_image_path"]),
        "ocr-overlay.png": Path(prepared["original_image_path"]),
        "ownership.png": Path(prepared["original_image_path"]),
        "reconstructed.png": Path(prepared["original_image_path"]),
        "difference.png": Path(prepared["background_difference_path"]),
    }
    evidence = {}
    for name, source in sources.items():
        target = evidence_root / name
        shutil.copyfile(source, target)
        evidence[name] = target

    nodes = []
    masks = prepared["_element_mask_paths"]
    components = prepared["components"]
    if len(masks) != len(components):
        raise ValueError("prepared component and mask counts differ")
    for index, (mask_source, component) in enumerate(
        zip(masks, components, strict=True), start=1
    ):
        component_id = f"component_{index:04d}"
        mask_target = masks_root / f"{component_id}.png"
        shutil.copyfile(mask_source, mask_target)
        with Image.open(mask_target) as image:
            bbox = image.convert("L").getbbox()
        if bbox is None:
            raise ValueError(f"prepared component mask is empty: {component_id}")
        left, top, right, bottom = bbox
        nodes.append({
            "id": component_id, "kind": "parent", "parent_id": None,
            "state": "pending", "mask": f"masks/{mask_target.name}",
            "mask_sha256": sha256_file(mask_target),
            "bbox": [left, top, right - left, bottom - top],
            "z_index": component.get("z_index", index - 1), "text_ids": [],
        })
    graph_path = evidence_root / "component-graph.json"
    graph_path.write_text(
        json.dumps({"nodes": nodes}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evidence["component-graph.json"] = graph_path
    quality_path = evidence_root / "quality-report.json"
    quality_path.write_text(
        json.dumps({"schema_version": 1, "phase": "initial_layers"}),
        encoding="utf-8",
    )
    evidence["quality-report.json"] = quality_path
    if set(evidence) != set(EVIDENCE_NAMES):
        raise RuntimeError("legacy component evidence set is incomplete")
    return {
        "page_id": page_id,
        "provider": store.read_json("job_manifest.json")["options"]["agent_provider"],
        "reconstruction_dir": reconstruction,
        "evidence": evidence,
    }


def advance_legacy_page(
    store: RunStore, page_id: str, *, _lease: ExecutionLease
) -> dict[str, Any]:
    outcome = advance_component_repair(store, page_id, _lease=_lease)
    status = outcome["status"]
    if status == "needs_execution":
        _execute_legacy_round(store, page_id, _lease)
        return {"status": "processing", "page_id": page_id}
    if status == "needs_quality":
        record_component_quality(store, page_id, _lease=_lease)
        return {"status": "processing", "page_id": page_id}
    if status == "needs_next_round":
        _publish_next_legacy_request(store, page_id, outcome["repair_round"], _lease)
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


def _quality_assets(
    store: RunStore, page_id: str, graph: dict, graph_dir: Path, output_dir: Path
) -> dict:
    import numpy as np

    module = importlib.import_module("image_to_ppt")
    prepared = module.load_component_layers(
        store.root / "pages" / page_id / "reconstruction/initial/prepared-page.json"
    )
    source_path = Path(prepared["original_image_path"])
    background_path = Path(prepared["background_original_path"])
    text_mask_path = Path(prepared["_text_mask_path"])
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB")).copy()
    with Image.open(background_path) as image:
        background = np.asarray(image.convert("RGB")).copy()
    reconstructed = background.copy()
    for node in graph["nodes"]:
        if node["kind"] == "text" or node["state"] not in {
            "pending", "pending_gate", "frozen"
        }:
            continue
        mask_path = graph_dir / Path(node["mask"])
        if sha256_file(mask_path) != node["mask_sha256"]:
            raise ValueError("execution graph mask sha256 mismatch")
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        reconstructed[mask] = source[mask]
    assets = {
        "background": output_dir / "background.png",
        "reconstructed": output_dir / "reconstructed.png",
        "text_mask": output_dir / "text-mask.png",
        "native_check": output_dir / "native-check.json",
    }
    shutil.copyfile(background_path, assets["background"])
    Image.fromarray(reconstructed, mode="RGB").save(assets["reconstructed"])
    shutil.copyfile(text_mask_path, assets["text_mask"])
    assets["native_check"].write_text(json.dumps({
        "schema_version": 1, "page_id": page_id,
        "source_sha256": sha256_file(source_path),
        "protected_native_overlap": "pass",
    }), encoding="utf-8")
    return {
        name: {
            "path": path.resolve().relative_to(store.root.resolve()).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in assets.items()
    }


def _execute_legacy_round(
    store: RunStore, page_id: str, lease: ExecutionLease
) -> None:
    import numpy as np
    from scripts.sam_worker import run_component_prompt_worker

    state = store.read_json(
        f"pages/{page_id}/reconstruction/component_state.json"
    )
    request_path = _state_artifact(store, state["current_round"]["request_ref"])
    plan_path = _state_artifact(store, state["current_round"]["plan_ref"])
    graph_path = _state_artifact(store, state["graph_ref"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    source = request_path.parent / Path(request["evidence"]["source.png"]["path"])
    with Image.open(source) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    output_dir = reconstruction / f"execution-{state['repair_round']:02d}"

    def sam_runner(**prompt):
        return run_component_prompt_worker(
            pixels, work_dir=output_dir.parent, **prompt
        )

    next_graph = execute_component_action_round(
        pixels, graph, plan["actions"], sam_runner=sam_runner,
        input_dir=graph_path.parent, output_dir=output_dir,
    )
    output_graph = output_dir / "component-graph.json"
    refs = _quality_assets(
        store, page_id, next_graph, output_dir, output_dir
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


def _publish_next_legacy_request(
    store: RunStore, page_id: str, repair_round: int, lease: ExecutionLease
) -> None:
    state = store.read_json(
        f"pages/{page_id}/reconstruction/component_state.json"
    )
    graph_path = _state_artifact(store, state["graph_ref"])
    quality_path = _state_artifact(
        store, state["current_round"]["quality_ref"]
    )
    execution = json.loads(_state_artifact(
        store, state["current_round"]["execution_ref"]
    ).read_text(encoding="utf-8"))
    refs = execution["quality_input_refs"]
    source = _source_path(store, page_id)
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    evidence_root = reconstruction / f"evidence-round-{repair_round:02d}"
    evidence_root.mkdir(exist_ok=False)
    evidence = {}
    copies = {
        "source.png": source,
        "numbered-masks.png": source,
        "ocr-overlay.png": source,
        "ownership.png": source,
        "reconstructed.png": _state_artifact(store, refs["reconstructed"]),
        "difference.png": _state_artifact(store, refs["background"]),
        "component-graph.json": graph_path,
        "quality-report.json": quality_path,
    }
    for name, source_path in copies.items():
        target = evidence_root / name
        shutil.copyfile(source_path, target)
        evidence[name] = target
    shutil.copytree(graph_path.parent / "masks", evidence_root / "masks")
    session = {
        "page_id": page_id, "provider": state["provider"],
        "reconstruction_dir": reconstruction, "evidence": evidence,
    }
    request_path = build_component_agent_request(
        session, repair_round=repair_round
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
    with Image.open(source) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    output_dir = store.root / "pages" / page_id / "reconstruction/parent-fallback"
    actions = [{
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
            node["state"] = "pending_gate"
    output_graph = output_dir / "component-graph.json"
    output_graph.write_text(
        json.dumps(next_graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refs = _quality_assets(
        store, page_id, next_graph, output_dir, output_dir
    )
    record_parent_fallback_execution(
        store, page_id, graph_path=output_graph,
        quality_input_refs=refs, _lease=lease,
    )


def assemble_legacy_results(store: RunStore) -> dict[str, Any]:
    manifest = store.read_json("job_manifest.json")
    page_ids = manifest["pages"]
    module = importlib.import_module("image_to_ppt")
    slides = []
    page_records = []
    for page_id in page_ids:
        reconstruction = store.root / "pages" / page_id / "reconstruction"
        state = store.read_json(
            f"pages/{page_id}/reconstruction/component_state.json"
        )
        prepared = module.load_component_layers(
            reconstruction / "initial" / "prepared-page.json"
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
            page_records.append((page_id, state, None))
            continue
        _, result_payload = _load_legacy_ref(store, state["result_ref"])
        result = json.loads(result_payload.decode("utf-8"))
        if result["status"] != "ready_for_assembly":
            raise ValueError("component result is not ready for assembly")
        slides.append(
            _accepted_slide_data(store, reconstruction, prepared, result)
        )
        page_records.append((page_id, state, result))

    output_path = manifest["options"]["output_path"]
    if output_path is None:
        output_path = store.root / "final" / "output.pptx"
    output_path = Path(output_path).resolve()
    slide_size = manifest["options"]["slide_size"]
    targets = _legacy_output_targets(output_path, slide_size)
    if any(path.exists() or path.is_symlink() for path in targets.values()):
        existing = next(
            path for path in targets.values()
            if path.exists() or path.is_symlink()
        )
        raise RuntimeError(f"Refusing to overwrite existing output: {existing}")

    staged = {}
    published = {}
    published_targets = []
    try:
        for variant, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, staging_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".staging", dir=target.parent
            )
            os.close(fd)
            staging = Path(staging_name)
            staging.unlink()
            try:
                if len(slides) == 1:
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
            try:
                os.link(staging, target)
            except Exception:
                raise
            published_targets.append(target)
            staging.unlink()
            published[variant] = str(target)
    except Exception:
        for target in published_targets:
            target.unlink(missing_ok=True)
        raise
    finally:
        for staging in staged.values():
            staging.unlink(missing_ok=True)

    _record_legacy_delivery(store, page_records, published)
    return published


def _legacy_output_targets(output_path: Path, slide_size: str) -> dict[str, Path]:
    if slide_size != "both":
        return {slide_size: output_path}
    base = output_path.with_suffix("")
    return {
        "original": Path(f"{base}_original.pptx"),
        "16:9": Path(f"{base}_16x9.pptx"),
    }


def _record_legacy_delivery(
    store: RunStore,
    page_records: list[tuple[str, dict, dict | None]],
    outputs: dict[str, str],
) -> None:
    output_refs = {
        name: {"path": path, "sha256": sha256_file(path)}
        for name, path in outputs.items()
    }
    for page_id, state, result in page_records:
        delivery = {
            "schema_version": 1,
            "page_id": page_id,
            "status": state["status"],
            "delivery_checks": {"pptx_reopen": "pass"},
            "outputs": output_refs,
        }
        if result is None:
            delivery["warning"] = (
                "Component reconstruction did not pass the parent gate; "
                "the full source image was preserved."
            )
        store.write_json(
            f"pages/{page_id}/reconstruction/component_delivery.json",
            delivery,
        )


def _load_legacy_ref(store: RunStore, reference: dict) -> tuple[Path, bytes]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("legacy artifact reference is invalid")
    path = (store.root / Path(reference["path"])).resolve()
    if not path.is_relative_to(store.root.resolve()):
        raise ValueError("legacy artifact reference escapes Run directory")
    status = path.lstat()
    if _is_link_or_reparse(status) or not stat.S_ISREG(status.st_mode):
        raise ValueError("legacy artifact is not a regular owned file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError("legacy artifact sha256 mismatch")
    return path, payload


def _accepted_slide_data(
    store: RunStore, reconstruction: Path, prepared: dict, result: dict
) -> dict:
    import numpy as np

    refs = result["accepted_asset_refs"]
    asset_payloads = {
        name: _load_legacy_ref(store, refs[name])[1]
        for name in (
            "source", "background", "reconstructed", "text_mask", "native_check"
        )
    }
    graph_path, graph_payload = _load_legacy_ref(store, result["graph_ref"])
    graph = json.loads(graph_payload.decode("utf-8"))
    by_id = {node["id"]: node for node in graph["nodes"]}
    final_ids = result["final_component_ids"]
    if len(final_ids) != len(set(final_ids)) or any(
        component_id not in by_id for component_id in final_ids
    ):
        raise ValueError("component result final IDs are invalid")
    output_dir = Path(tempfile.mkdtemp(prefix="assembly-assets-", dir=reconstruction))
    asset_paths = {}
    for name, payload in asset_payloads.items():
        snapshot = output_dir / f"accepted-{name}.asset"
        snapshot.write_bytes(payload)
        asset_paths[name] = snapshot
    with Image.open(io.BytesIO(asset_payloads["reconstructed"])) as image:
        reconstructed_image = np.asarray(image.convert("RGB")).copy()
    with Image.open(io.BytesIO(asset_payloads["text_mask"])) as image:
        text_mask = np.asarray(image.convert("L")) > 0
    owner_count = np.zeros(text_mask.shape, dtype=np.uint16)
    component_masks = []
    for component_id in final_ids:
        node = by_id[component_id]
        mask_path = (graph_path.parent / Path(node["mask"])).resolve()
        if not mask_path.is_relative_to(store.root.resolve()):
            raise ValueError("final component mask escapes Run directory")
        mask_relative = mask_path.relative_to(store.root.resolve()).as_posix()
        _, mask_payload = _load_legacy_ref(store, {
            "path": mask_relative, "sha256": node["mask_sha256"]
        })
        with Image.open(io.BytesIO(mask_payload)) as image:
            mask = np.asarray(image.convert("L")) > 0
        if mask.shape != text_mask.shape:
            raise ValueError("final component mask dimensions differ")
        owner_count += mask
        component_masks.append((node, mask & ~text_mask))
    if np.any(owner_count > 1):
        raise ValueError("final component masks violate unique ownership")
    components = []
    for node, mask in component_masks:
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ValueError(f"final component became empty: {node['id']}")
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        rgba = np.dstack((reconstructed_image, mask.astype(np.uint8) * 255))
        component_path = output_dir / f"{node['id']}.png"
        Image.fromarray(rgba[top:bottom, left:right], mode="RGBA").save(component_path)
        components.append({
            "path": str(component_path), "x": left, "y": top,
            "w": right - left, "h": bottom - top,
            "z_index": node["z_index"],
        })
    return {
        **prepared,
        "background_path": str(asset_paths["background"]),
        "background_original_path": str(asset_paths["background"]),
        "background_widescreen_path": str(asset_paths["background"]),
        "original_image_path": str(asset_paths["source"]),
        "components": sorted(components, key=lambda item: item["z_index"]),
    }
