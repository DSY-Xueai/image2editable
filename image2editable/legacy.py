from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import ctypes
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageOps
from pptx import Presentation

from image2editable.contracts import validate_schema_version
from image2editable.component_contracts import MAX_REPAIR_ROUNDS
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
    page_request = store.read_json(Path("pages") / page_id / "page_request.json")
    if _is_full_page_candidate(page_request):
        prepared = _prepare_full_page_layers(source, reconstruction / "initial")
    else:
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


def _is_full_page_candidate(request: dict[str, Any]) -> bool:
    return (
        request.get("full_page_candidate") is True
        and type(request.get("slide_coverage")) is float
        and request["slide_coverage"] == 1.0
    )


def _prepare_full_page_layers(source: Path, work_dir: Path) -> dict[str, Any]:
    """Create a deterministic one-parent layer for an approved full-page image."""
    work_dir.mkdir(parents=True, exist_ok=False)
    with Image.open(source) as opened:
        rgb = opened.convert("RGB")
        rgba = opened.convert("RGBA")
        width, height = rgb.size
        rgb.save(work_dir / "source-image.png")
        rgb.save(work_dir / "background-original.png")
        rgb.save(work_dir / "background-16x9.png")
        Image.new("L", (width, height), 0).save(work_dir / "source-text-mask.png")
        Image.new("L", (width, height), 255).save(work_dir / "element-mask-0001.png")
        Image.new("L", (width, height), 255).save(work_dir / "background-removal-mask.png")
        Image.new("RGB", (width, height), (0, 0, 0)).save(work_dir / "background-difference.png")
        rgba.save(work_dir / "component-0001.png")

    def asset(path: str) -> dict[str, str]:
        payload = (work_dir / path).read_bytes()
        return {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}

    component = {
        "asset": asset("component-0001.png"),
        "metadata": {
            "x": 0, "y": 0, "w": width, "h": height,
            "area": width * height, "z_index": 0,
        },
    }
    prepared_manifest = {
        "schema_version": 1,
        "phase": "initial_layers",
        "resource_isolation": True,
        "initial_component_count": 1,
        "components": [component],
        "text_items": [],
        "dimensions": {
            "img_width": width, "img_height": height,
            "canvas_width": width, "canvas_height": height,
            "content_offset_x": 0, "content_offset_y": 0,
            "widescreen_background_method": "identity",
        },
        "assets": {
            "source_image": asset("source-image.png"),
            "ocr_mask": asset("source-text-mask.png"),
            "text_clean": None,
            "element_masks": [asset("element-mask-0001.png")],
            "background_original": asset("background-original.png"),
            "background_widescreen": asset("background-16x9.png"),
            "background_removal_mask": asset("background-removal-mask.png"),
            "background_difference": asset("background-difference.png"),
        },
    }
    payload = json.dumps(
        prepared_manifest, ensure_ascii=False, indent=2
    ).encode("utf-8")
    (work_dir / "prepared_page.json").write_bytes(payload)
    (work_dir / "prepared_page.sha256").write_bytes(
        (hashlib.sha256(payload).hexdigest() + "\n").encode("ascii")
    )
    return {
        "original_image_path": work_dir / "source-image.png",
        "background_original_path": work_dir / "background-original.png",
        "background_difference_path": work_dir / "background-difference.png",
        "_text_mask_path": work_dir / "source-text-mask.png",
        "_element_mask_paths": [work_dir / "element-mask-0001.png"],
        "components": [{
            "path": work_dir / "component-0001.png",
            "x": 0, "y": 0, "w": width, "h": height,
            "area": width * height, "z_index": 0,
        }],
        "initial_component_count": 1,
    }


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
    quality_path.write_text(
        json.dumps({
            "schema_version": 1,
            "phase": "initial_layers",
            "text_items": text_items,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    evidence["quality-report.json"] = quality_path
    evidence.update(
        _render_component_evidence(
            source_path=source_target,
            graph={"nodes": nodes},
            graph_dir=evidence_root,
            text_mask_path=Path(prepared["_text_mask_path"]),
            background_path=Path(prepared["background_original_path"]),
            reconstructed_path=None,
            output_dir=evidence_root,
            text_items=text_items,
        )
    )
    if set(evidence) != set(EVIDENCE_NAMES):
        raise RuntimeError("legacy component evidence set is incomplete")
    return {
        "page_id": page_id,
        "provider": store.read_json("job_manifest.json")["options"]["agent_provider"],
        "reconstruction_dir": reconstruction,
        "evidence": evidence,
    }


def _component_text_items(items: object, page_size: tuple[int, int]) -> list[dict]:
    if not isinstance(items, list):
        return []
    width, height = page_size
    normalized = []
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
        normalized.append({
            "id": f"text_{len(normalized) + 1:04d}",
            "text": item["text"],
            "box": [left, top, right, bottom],
        })
    return normalized


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


def _render_component_evidence(
    *,
    source_path: Path,
    graph: dict,
    graph_dir: Path,
    text_mask_path: Path,
    background_path: Path,
    reconstructed_path: Path | None,
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
        if reconstructed_path is None:
            with Image.open(background_path) as image:
                reconstructed = keep(image.convert("RGB"))
            if reconstructed.size != source.size:
                raise ValueError("component evidence background dimensions differ")
        else:
            with Image.open(reconstructed_path) as image:
                reconstructed = keep(image.convert("RGB"))
            if reconstructed.size != source.size:
                raise ValueError("component evidence reconstruction dimensions differ")

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
        for node in graph["nodes"]:
            if node["kind"] == "text" or node["state"] not in {
                "pending",
                "pending_gate",
                "frozen",
            }:
                continue
            mask_path = graph_dir / Path(node["mask"])
            if sha256_file(mask_path) != node["mask_sha256"]:
                raise ValueError("component evidence mask sha256 mismatch")
            with ExitStack() as node_images:
                def keep_node(image: Image.Image) -> Image.Image:
                    node_images.callback(image.close)
                    return image

                with Image.open(mask_path) as image:
                    mask = keep_node(image.convert("L"))
                if mask.size != source.size:
                    raise ValueError("component evidence mask dimensions differ")
                color = colors[int(node["z_index"]) % len(colors)]
                alpha = keep_node(mask.point(lambda value: value * 96 // 255))
                render_mask = keep_node(ImageChops.subtract(mask, text_mask))
                color_layer = keep_node(Image.new("RGB", source.size, color))
                numbered.paste(color_layer, (0, 0), alpha)
                ownership.paste(color_layer, (0, 0), render_mask)
                if reconstructed_path is None:
                    reconstructed.paste(source, (0, 0), render_mask)
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

        paths = {}
        for name, evidence_image in (
            ("numbered-masks.png", numbered),
            ("ownership.png", ownership),
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
            label = f'{item["id"]}: {item["text"].replace(chr(10), " ")[:40]}'
            ocr_draw.text(
                (left, max(0, top - 12)),
                label,
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
        store.root / "pages" / page_id / "reconstruction/initial/prepared_page.json"
    )
    source_path = Path(prepared["original_image_path"])
    background_path = Path(prepared["background_original_path"])
    text_mask_path = Path(prepared["_text_mask_path"])
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB")).copy()
    with Image.open(background_path) as image:
        background = np.asarray(image.convert("RGB")).copy()
    with Image.open(text_mask_path) as image:
        text_mask = np.asarray(image.convert("L")) > 0
    if text_mask.shape != source.shape[:2]:
        raise ValueError("quality text mask dimensions differ")
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
        render_mask = mask & ~text_mask
        reconstructed[render_mask] = source[render_mask]
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
        "text_items": prepared.get("text_items", []),
    }, ensure_ascii=False), encoding="utf-8")
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
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    refs = quality["input_refs"]
    # Every repair round must keep the exact source snapshot bound into the
    # component state.  PPTX media may have been losslessly re-encoded during
    # deterministic initialization, so reopening the original candidate file
    # can produce a different byte hash even though its pixels are identical.
    source = _state_artifact(store, refs["source"])
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    module = importlib.import_module("image_to_ppt")
    prepared = module.load_component_layers(
        reconstruction / "initial" / "prepared_page.json"
    )
    with Image.open(source) as image:
        text_items = _component_text_items(
            prepared.get("text_items", []), image.size
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
        "reconstructed.png": _state_artifact(store, refs["reconstructed"]),
        "component-graph.json": graph_path,
        "quality-report.json": quality_path,
    }
    for name, source_path in copies.items():
        target = evidence_root / name
        shutil.copyfile(source_path, target)
        evidence[name] = target
    shutil.copytree(graph_path.parent / "masks", evidence_root / "masks")
    evidence.update(
        _render_component_evidence(
            source_path=evidence["source.png"],
            graph=graph,
            graph_dir=evidence_root,
            text_mask_path=_state_artifact(store, refs["text_mask"]),
            background_path=_state_artifact(store, refs["background"]),
            reconstructed_path=evidence["reconstructed.png"],
            output_dir=evidence_root,
            text_items=text_items,
        )
    )
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
    output_dir = store.root / "pages" / page_id / "reconstruction/parent-fallback"
    _ensure_component_disk_reserve(
        output_dir.parent,
        source,
        node_count=len(graph["nodes"]),
        repair_round=state["repair_round"],
    )
    with Image.open(source) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
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
