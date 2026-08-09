from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Callable

import numpy as np
from PIL import Image

from image2editable.reconstruction_contracts import (
    validate_reconstruction_ir,
    validate_reconstruction_plan,
    validate_route_result,
)
from image2editable.reconstruction_ir import build_reconstruction_ir
from image2editable.reconstruction_router import route_reconstruction
from image2editable.render_qa import compare_rendered_page
from image2editable.shape_analysis import analyze_shape_candidate
from image2editable.store import RunStore


@dataclass(frozen=True)
class RouteContext:
    store: RunStore
    page_id: str
    component_result_path: Path
    adapter: str
    capabilities: frozenset[str]
    source_image_path: Path
    assemble_page: Callable[[dict, Path], None]
    rendered_text_reader: Callable[[Path], list[dict]] | None = None


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _plain_path(path: Path, root: Path) -> Path:
    root = root.resolve()
    absolute = Path(os.path.abspath(path))
    if not absolute.is_relative_to(root):
        raise ValueError("route artifact path escapes Run directory")
    relative = absolute.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if not current.exists() and not current.is_symlink():
            continue
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise RuntimeError("route artifact path contains a link or reparse point")
    return absolute


def _read_file(path: Path, root: Path, label: str) -> bytes:
    contained = _plain_path(path, root)
    info = contained.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file")
    if info.st_size > 256 * 1024 * 1024:
        raise ValueError(f"{label} exceeds the size limit")
    return contained.read_bytes()


def _reference_path(store: RunStore, reference: object) -> Path:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("route artifact ref is invalid")
    path = reference["path"]
    digest = reference["sha256"]
    if (
        type(path) is not str
        or not path
        or "\\" in path
        or ":" in path
        or PurePosixPath(path).is_absolute()
        or ".." in PurePosixPath(path).parts
        or type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("route artifact ref is invalid")
    return _plain_path(store.root / Path(*PurePosixPath(path).parts), store.root)


def _read_reference(
    store: RunStore, reference: object, label: str
) -> tuple[Path, bytes]:
    path = _reference_path(store, reference)
    payload = _read_file(path, store.root, label)
    if _sha256(payload) != reference["sha256"]:
        raise RuntimeError(f"{label} hash mismatch")
    return path, payload


def _artifact_ref(store: RunStore, path: Path, payload: bytes) -> dict:
    contained = _plain_path(path, store.root)
    return {
        "path": contained.relative_to(store.root).as_posix(),
        "sha256": _sha256(payload),
    }


def _write_exclusive(store: RunStore, path: Path, payload: bytes) -> dict:
    path = _plain_path(path, store.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _plain_path(path.parent, store.root)
    if path.exists() or path.is_symlink():
        existing = _read_file(path, store.root, "existing route artifact")
        if existing != payload:
            raise RuntimeError(f"route artifact already exists with different bytes: {path}")
        return _artifact_ref(store, path, payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
    finally:
        os.close(descriptor)
    return _artifact_ref(store, path, payload)


def _publish_json(store: RunStore, path: Path, value: dict) -> dict:
    return _write_exclusive(store, path, _json_bytes(value))


def _json_reference(store: RunStore, reference: object, label: str) -> tuple[Path, dict]:
    path, payload = _read_reference(store, reference, label)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must be an object")
    return path, value


def _load_existing_result(
    store: RunStore,
    result_path: Path,
    component_sha256: str,
    *,
    page_id: str,
) -> dict | None:
    if not result_path.exists() and not result_path.is_symlink():
        return None
    payload = _read_file(result_path, store.root, "route result")
    try:
        result = validate_route_result(json.loads(payload.decode("utf-8")))
    except json.JSONDecodeError as error:
        raise ValueError("route result JSON is invalid") from error
    if result["component_result_sha256"] != component_sha256:
        raise RuntimeError("route result component hash mismatch")
    if result["page_id"] != page_id:
        raise RuntimeError("route result page identity mismatch")
    _, ir = _json_reference(store, result["ir_ref"], "reconstruction IR")
    _, plan = _json_reference(
        store, result["plan_ref"], "reconstruction plan"
    )
    validate_reconstruction_plan(plan, ir=ir)
    if result["qa_ref"] is not None:
        _json_reference(store, result["qa_ref"], "render QA")
    return result


def load_published_route(
    store: RunStore,
    component_result_path: str | Path,
    *,
    page_id: str,
) -> dict | None:
    """Load a published sidecar only when every binding still verifies."""

    component_path = Path(component_result_path)
    component_payload = _read_file(component_path, store.root, "component result")
    component_sha256 = _sha256(component_payload)
    result_path = component_path.parent / "route" / "route_result.json"
    result = _load_existing_result(
        store, result_path, component_sha256, page_id=page_id
    )
    if result is None:
        return None
    ir_path, ir = _json_reference(store, result["ir_ref"], "reconstruction IR")
    plan_path, plan = _json_reference(
        store, result["plan_ref"], "reconstruction plan"
    )
    ir = validate_reconstruction_ir(ir)
    plan = validate_reconstruction_plan(plan, ir=ir)
    result_payload = _read_file(result_path, store.root, "route result")
    return {
        "result": result,
        "result_ref": _artifact_ref(store, result_path, result_payload),
        "ir": ir,
        "ir_path": ir_path,
        "plan": plan,
        "plan_path": plan_path,
    }


def route_visual_elements(store: RunStore, ir: dict, plan: dict) -> list[dict]:
    """Convert one validated route plan into the shared visual Adapter input."""

    validated_ir = validate_reconstruction_ir(ir)
    validated_plan = validate_reconstruction_plan(plan, ir=validated_ir)
    objects = {item["id"]: item for item in validated_ir["objects"]}
    elements = []
    for route in validated_plan["routes"]:
        item = objects[route["object_id"]]
        selected = next(
            candidate
            for candidate in item["candidate_representations"]
            if candidate["kind"] == route["selected_route"]
        )
        if route["selected_route"] == "editable_text":
            continue
        if route["selected_route"] == "native_shape":
            elements.append(
                {
                    "object_id": item["id"],
                    "route": "native_shape",
                    "z_index": item["z_index"],
                    "bbox": deepcopy(item["bbox"]),
                    "shape": deepcopy(selected["payload"]),
                }
            )
            continue
        asset_ref = selected["payload"]["asset_ref"]
        asset_path, _ = _read_reference(store, asset_ref, "route raster asset")
        left, top, right, bottom = item["bbox"]
        elements.append(
            {
                "object_id": item["id"],
                "route": "raster_component",
                "z_index": item["z_index"],
                "component": {
                    "component_id": item["id"],
                    "path": str(asset_path),
                    "x": left,
                    "y": top,
                    "w": right - left,
                    "h": bottom - top,
                    "z_index": item["z_index"],
                },
            }
        )
    return sorted(elements, key=lambda element: element["z_index"])


def _build_ir(context: RouteContext, component_result: dict) -> dict:
    graph_path, graph = _json_reference(
        context.store, component_result.get("graph_ref"), "component graph"
    )
    accepted_refs = component_result.get("accepted_asset_refs")
    if not isinstance(accepted_refs, dict):
        raise ValueError("component result accepted refs are invalid")
    _, manifest = _json_reference(
        context.store,
        accepted_refs.get("presentation_manifest"),
        "presentation manifest",
    )
    components = manifest.get("components")
    if not isinstance(components, list):
        raise ValueError("presentation manifest components are invalid")
    manifest_assets = {}
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("presentation manifest component is invalid")
        component_id = component.get("component_id")
        rgba_ref = component.get("rgba")
        _read_reference(context.store, rgba_ref, "component RGBA")
        if (
            type(component_id) is not str
            or not component_id
            or any(character in component_id for character in "/\\:")
            or component_id in manifest_assets
        ):
            raise ValueError("presentation manifest component ID is invalid")
        manifest_assets[component_id] = deepcopy(rgba_ref)

    route_graph = deepcopy(graph)
    nodes = route_graph.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("component graph nodes are invalid")
    component_assets = {}
    shape_candidates = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("component graph node is invalid")
        if "mask" in node and "mask_sha256" in node:
            mask_path = graph_path.parent / Path(*PurePosixPath(node["mask"]).parts)
            mask_payload = _read_file(mask_path, context.store.root, "component mask")
            if _sha256(mask_payload) != node["mask_sha256"]:
                raise RuntimeError("component mask hash mismatch")
            mask_ref = _artifact_ref(context.store, mask_path, mask_payload)
            node["mask"] = mask_ref["path"]
        else:
            mask_ref = None
        if (
            node.get("kind") not in {"parent", "child"}
            or node.get("state") != "frozen"
        ):
            continue
        asset_ref = manifest_assets.get(node.get("id"))
        if asset_ref is None or mask_ref is None:
            raise ValueError("accepted visual evidence is incomplete")
        rgba_path, _ = _read_reference(
            context.store, asset_ref, "component RGBA"
        )
        with Image.open(rgba_path) as image:
            rgba = np.asarray(image.convert("RGBA")).copy()
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")).copy()
        bbox = node.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or not (0 <= bbox[0] < bbox[2] <= rgba.shape[1])
            or not (0 <= bbox[1] < bbox[3] <= rgba.shape[0])
        ):
            raise ValueError("accepted visual bbox is invalid")
        cropped = rgba[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        buffer = io.BytesIO()
        Image.fromarray(cropped, mode="RGBA").save(buffer, format="PNG")
        component_assets[node["id"]] = _write_exclusive(
            context.store,
            context.component_result_path.parent
            / "route"
            / "assets"
            / f"{node['id']}.png",
            buffer.getvalue(),
        )
        measurement = analyze_shape_candidate(rgba, mask)
        if measurement is not None:
            shape_candidates[node["id"]] = [
                {
                    "kind": "native_shape",
                    "confidence": measurement["geometry_score"],
                    "payload": measurement,
                    "evidence_refs": [mask_ref],
                    "required_qa_checks": [
                        "pptx_native_shape",
                        "render_difference",
                    ],
                }
            ]
    source_payload = _read_file(
        context.source_image_path, context.store.root, "route source image"
    )
    with Image.open(context.source_image_path) as image:
        canvas = image.size
    if not source_payload:
        raise ValueError("route source image is empty")
    return build_reconstruction_ir(
        page_id=context.page_id,
        canvas=canvas,
        graph=route_graph,
        component_assets=component_assets,
        text_items=component_result.get("text_items", []),
        shape_candidates=shape_candidates,
    )


def _fallback_plan(ir: dict, plan: dict, object_ids: set[str]) -> dict:
    objects = {item["id"]: item for item in ir["objects"]}
    updated = deepcopy(plan)
    for route in updated["routes"]:
        if route["object_id"] not in object_ids:
            continue
        raster = next(
            (
                candidate
                for candidate in objects[route["object_id"]][
                    "candidate_representations"
                ]
                if candidate["kind"] == "raster_component"
            ),
            None,
        )
        if raster is None:
            raise ValueError("route fallback raster_component is missing")
        route.update(
            selected_route="raster_component",
            fallback_route=None,
            candidate_confidence=raster["confidence"],
            evidence_refs=deepcopy(raster["evidence_refs"]),
            qa_requirements=deepcopy(raster["required_qa_checks"]),
        )
    return validate_reconstruction_plan(updated, ir=ir)


def _render_plan(
    context: RouteContext,
    renderer,
    plan: dict,
    route_root: Path,
    name: str,
    *,
    width: int,
    height: int,
    pptx_name: str | None = None,
    assemble: bool = True,
) -> np.ndarray:
    pptx_path = route_root / f"{pptx_name or name}.pptx"
    output_path = route_root / f"{name}.png"
    if assemble:
        context.assemble_page(plan, pptx_path)
    rendered = renderer.render_page(
        pptx_path, 1, output_path, width=width, height=height
    )
    rendered_path = Path(rendered["path"])
    with Image.open(rendered_path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _result_document(
    context: RouteContext,
    *,
    component_sha256: str,
    ir_ref: dict,
    plan_ref: dict,
    qa_ref: dict | None,
    status: str,
    reason: str | None,
) -> dict:
    return validate_route_result(
        {
            "schema_version": 1,
            "page_id": context.page_id,
            "status": status,
            "component_result_sha256": component_sha256,
            "ir_ref": ir_ref,
            "plan_ref": plan_ref,
            "qa_ref": qa_ref,
            "reason": reason,
        }
    )


def finalize_page_route(
    context: RouteContext,
    *,
    renderer,
    policy: dict,
) -> dict:
    """Publish one hash-bound route sidecar with a deterministic Raster fallback."""

    component_payload = _read_file(
        context.component_result_path,
        context.store.root,
        "component result",
    )
    component_sha256 = _sha256(component_payload)
    try:
        component_result = json.loads(component_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("component result JSON is invalid") from error
    if (
        not isinstance(component_result, dict)
        or component_result.get("page_id") != context.page_id
        or component_result.get("status") != "ready_for_assembly"
    ):
        raise ValueError("component result is not ready for route execution")
    route_root = context.component_result_path.parent / "route"
    route_root.mkdir(parents=True, exist_ok=True)
    _plain_path(route_root, context.store.root)
    result_path = route_root / "route_result.json"
    existing = _load_existing_result(
        context.store,
        result_path,
        component_sha256,
        page_id=context.page_id,
    )
    if existing is not None:
        return existing

    ir = _build_ir(context, component_result)
    ir_ref = _publish_json(
        context.store, route_root / "reconstruction-ir.json", ir
    )
    candidate_plan = route_reconstruction(
        ir,
        adapter=context.adapter,
        capabilities=set(context.capabilities),
        policy=policy,
        authoritative_render_qa=True,
    )
    _publish_json(
        context.store,
        route_root / "reconstruction-plan-candidate.json",
        candidate_plan,
    )
    native_ids = {
        route["object_id"]
        for route in candidate_plan["routes"]
        if route["selected_route"] == "native_shape"
    }
    raster_plan = _fallback_plan(ir, candidate_plan, native_ids)

    qa_document = None
    qa_ref = None
    if not native_ids:
        final_plan = candidate_plan
        status = "raster_fallback"
        reason = "native_not_selected"
    elif not renderer.available():
        final_plan = raster_plan
        status = "raster_fallback"
        reason = "renderer_unavailable"
    else:
        with Image.open(context.source_image_path) as image:
            source = np.asarray(image.convert("RGB")).copy()
        height, width = source.shape[:2]
        baseline = _render_plan(
            context,
            renderer,
            raster_plan,
            route_root,
            "raster-baseline",
            width=width,
            height=height,
        )
        repeated = _render_plan(
            context,
            renderer,
            raster_plan,
            route_root,
            "raster-baseline-repeat",
            width=width,
            height=height,
            pptx_name="raster-baseline",
            assemble=False,
        )
        candidate = _render_plan(
            context,
            renderer,
            candidate_plan,
            route_root,
            "native-candidate",
            width=width,
            height=height,
        )
        regions = {
            item["id"]: item["bbox"] for item in ir["objects"] if item["id"] in native_ids
        }
        rendered_text = (
            context.rendered_text_reader(route_root / "native-candidate.png")
            if context.rendered_text_reader is not None
            else None
        )
        initial_qa = compare_rendered_page(
            source,
            baseline,
            candidate,
            object_regions=regions,
            repeated_baseline=repeated,
            expected_text_items=(component_result.get("text_items", []) if rendered_text is not None else None),
            rendered_text_items=rendered_text,
        )
        qa_document = {
            "schema_version": 1,
            "renderer": renderer.identity(),
            "initial": initial_qa,
            "fallback": None,
        }
        if initial_qa["accepted"]:
            final_plan = candidate_plan
            status = "native_accepted"
            reason = None
        else:
            failed_ids = set(initial_qa["failed_object_ids"]) & native_ids
            if not failed_ids:
                failed_ids = set(native_ids)
            fallback_plan = _fallback_plan(ir, candidate_plan, failed_ids)
            fallback_render = _render_plan(
                context,
                renderer,
                fallback_plan,
                route_root,
                "native-fallback",
                width=width,
                height=height,
            )
            fallback_text = (
                context.rendered_text_reader(route_root / "native-fallback.png")
                if context.rendered_text_reader is not None
                else None
            )
            fallback_qa = compare_rendered_page(
                source,
                baseline,
                fallback_render,
                object_regions=regions,
                repeated_baseline=repeated,
                expected_text_items=(component_result.get("text_items", []) if fallback_text is not None else None),
                rendered_text_items=fallback_text,
            )
            qa_document["fallback"] = fallback_qa
            if fallback_qa["accepted"]:
                final_plan = fallback_plan
                reason = "native_render_qa_failed"
            else:
                final_plan = raster_plan
                reason = "fallback_render_qa_failed"
            status = "raster_fallback"

    plan_ref = _publish_json(
        context.store, route_root / "reconstruction-plan.json", final_plan
    )
    if qa_document is not None:
        qa_ref = _publish_json(
            context.store, route_root / "render-qa.json", qa_document
        )
    result = _result_document(
        context,
        component_sha256=component_sha256,
        ir_ref=ir_ref,
        plan_ref=plan_ref,
        qa_ref=qa_ref,
        status=status,
        reason=reason,
    )
    _publish_json(context.store, result_path, result)
    return result
