from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from image2editable.route_execution import (
    RouteContext,
    finalize_page_route,
    load_published_route,
    route_visual_elements,
)
from image2editable.store import RunStore


def _write_json(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_image(root: Path, relative: str, image: Image.Image) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class _UnavailableRenderer:
    def available(self) -> bool:
        return False

    def identity(self) -> dict:
        return {"renderer": "powerpoint", "available": False}


class _FakeRenderer:
    def __init__(
        self,
        source: Path,
        plans: dict[Path, dict],
        *,
        native_fails: set[str] | None = None,
        forced_failures: set[str] | None = None,
    ) -> None:
        self.source = source
        self.plans = plans
        self.native_fails = native_fails or set()
        self.forced_failures = forced_failures or set()
        self.calls = 0

    def available(self) -> bool:
        return True

    def identity(self) -> dict:
        return {"renderer": "powerpoint", "available": True, "version": "fake"}

    def render_page(
        self,
        pptx_path: Path,
        page_number: int,
        output_path: Path,
        *,
        width: int,
        height: int,
    ) -> dict:
        assert page_number == 1
        self.calls += 1
        image = np.asarray(Image.open(self.source).convert("RGB")).copy()
        plan = self.plans[Path(pptx_path).resolve()]
        if Path(pptx_path).stem in self.forced_failures:
            image[:] = 0
        for route in plan["routes"]:
            if (
                route["selected_route"] == "native_shape"
                and route["object_id"] in self.native_fails
            ):
                if route["object_id"] == "shape_1":
                    image[10:35, 10:40] = 0
                else:
                    image[40:70, 55:90] = 0
        Image.fromarray(image).save(output_path)
        return {
            "renderer": "powerpoint",
            "version": "fake",
            "width": width,
            "height": height,
            "path": str(Path(output_path).resolve()),
        }


@pytest.fixture
def route_context(tmp_path: Path):
    store = RunStore(tmp_path)
    page_root = tmp_path / "pages/page_001/reconstruction"
    graph_root = page_root / "evidence"
    source = tmp_path / "pages/page_001/source.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(source)
    nodes = []
    manifest_components = []
    for index, (object_id, bbox) in enumerate(
        (("shape_1", [10, 10, 40, 35]), ("shape_2", [55, 40, 90, 70])),
        start=1,
    ):
        mask = np.zeros((80, 100), dtype=np.uint8)
        cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2] - 1, bbox[3] - 1), 255, -1)
        mask_ref = _write_image(
            tmp_path,
            f"pages/page_001/reconstruction/evidence/masks/{object_id}.png",
            Image.fromarray(mask),
        )
        rgba = np.zeros((80, 100, 4), dtype=np.uint8)
        rgba[mask > 0] = (30 * index, 90, 180, 255)
        rgba_ref = _write_image(
            tmp_path,
            f"pages/page_001/reconstruction/assets/{object_id}.png",
            Image.fromarray(rgba, mode="RGBA"),
        )
        nodes.append(
            {
                "id": object_id,
                "kind": "child",
                "state": "frozen",
                "bbox": bbox,
                "z_index": index,
                "mask": f"masks/{object_id}.png",
                "mask_sha256": mask_ref["sha256"],
                "parent_id": None,
                "text_ids": [],
            }
        )
        manifest_components.append({"component_id": object_id, "rgba": rgba_ref})
    graph_ref = _write_json(graph_root / "component-graph.json", {"nodes": nodes})
    graph_ref["path"] = Path(graph_ref["path"]).relative_to(tmp_path).as_posix()
    manifest_ref = _write_json(
        page_root / "presentation-manifest.json",
        {"components": manifest_components},
    )
    manifest_ref["path"] = Path(manifest_ref["path"]).relative_to(tmp_path).as_posix()
    component_result = {
        "schema_version": 1,
        "page_id": "page_001",
        "status": "ready_for_assembly",
        "graph_ref": graph_ref,
        "accepted_asset_refs": {"presentation_manifest": manifest_ref},
        "text_items": [],
    }
    component_result_path = page_root / "component_result.json"
    component_result_path.parent.mkdir(parents=True, exist_ok=True)
    component_result_path.write_text(
        json.dumps(component_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plans: dict[Path, dict] = {}
    assemble_calls = []

    def assemble(plan: dict, path: Path) -> None:
        resolved = Path(path).resolve()
        plans[resolved] = json.loads(json.dumps(plan))
        assemble_calls.append(resolved)
        resolved.write_bytes(b"fake pptx")

    context = RouteContext(
        store=store,
        page_id="page_001",
        component_result_path=component_result_path,
        adapter="pptx",
        capabilities=frozenset(
            {"editable_text", "native_shape", "raster_component"}
        ),
        source_image_path=source,
        assemble_page=assemble,
    )
    policy = {
        "schema_version": 1,
        "native_shape_enabled": True,
        "allowed_shapes": ["rectangle", "rounded_rectangle", "ellipse", "line"],
        "min_geometry_score": 0.99,
        "max_color_mad": 3.0,
    }
    return context, policy, plans, assemble_calls


def _load_ref(context: RouteContext, reference: dict) -> dict:
    path = context.store.root / reference["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    return json.loads(path.read_text(encoding="utf-8"))


def _route_for(plan: dict, object_id: str) -> str:
    return next(
        route["selected_route"]
        for route in plan["routes"]
        if route["object_id"] == object_id
    )


def test_missing_renderer_publishes_raster_fallback(route_context) -> None:
    context, policy, _, assemble_calls = route_context

    result = finalize_page_route(
        context, renderer=_UnavailableRenderer(), policy=policy
    )

    assert result["status"] == "raster_fallback"
    assert result["reason"] == "renderer_unavailable"
    assert result["qa_ref"] is None
    plan = _load_ref(context, result["plan_ref"])
    assert {route["selected_route"] for route in plan["routes"]} == {
        "raster_component"
    }
    assert assemble_calls == []


def test_failed_native_candidate_reassembles_only_failed_object(route_context) -> None:
    context, policy, plans, assemble_calls = route_context
    renderer = _FakeRenderer(
        context.source_image_path, plans, native_fails={"shape_1"}
    )

    result = finalize_page_route(context, renderer=renderer, policy=policy)

    assert result["status"] == "raster_fallback"
    assert result["reason"] == "native_render_qa_failed"
    plan = _load_ref(context, result["plan_ref"])
    assert _route_for(plan, "shape_1") == "raster_component"
    assert _route_for(plan, "shape_2") == "native_shape"
    published = load_published_route(
        context.store, context.component_result_path, page_id=context.page_id
    )
    elements = route_visual_elements(
        context.store, published["ir"], published["plan"]
    )
    assert [element["route"] for element in elements] == [
        "raster_component",
        "native_shape",
    ]
    assert published["result"] == result
    assert published["result_ref"]["path"].endswith("route_result.json")
    assert len(assemble_calls) == 3
    assert renderer.calls == 4


def test_successful_native_route_binds_qa_and_component_result(route_context) -> None:
    context, policy, plans, _ = route_context
    renderer = _FakeRenderer(context.source_image_path, plans)

    result = finalize_page_route(context, renderer=renderer, policy=policy)

    assert result["status"] == "native_accepted"
    assert result["reason"] is None
    assert result["component_result_sha256"] == hashlib.sha256(
        context.component_result_path.read_bytes()
    ).hexdigest()
    assert result["qa_ref"] is not None
    ir = _load_ref(context, result["ir_ref"])
    raster = ir["objects"][0]["candidate_representations"][0]
    asset_path = context.store.root / raster["payload"]["asset_ref"]["path"]
    with Image.open(asset_path) as image:
        assert image.size == (30, 25)


def test_published_native_route_requires_bound_accepted_qa(route_context) -> None:
    context, policy, plans, _ = route_context
    renderer = _FakeRenderer(context.source_image_path, plans)
    finalize_page_route(context, renderer=renderer, policy=policy)
    result_path = context.component_result_path.parent / "route/route_result.json"
    document = json.loads(result_path.read_text(encoding="utf-8"))
    document["qa_ref"] = None
    result_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="QA"):
        load_published_route(
            context.store,
            context.component_result_path,
            page_id=context.page_id,
        )


def test_published_native_route_rejects_qa_for_another_plan(route_context) -> None:
    context, policy, plans, _ = route_context
    renderer = _FakeRenderer(context.source_image_path, plans)
    result = finalize_page_route(context, renderer=renderer, policy=policy)
    qa_path = context.store.root / result["qa_ref"]["path"]
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["final_plan_sha256"] = "0" * 64
    qa_path.write_text(json.dumps(qa), encoding="utf-8")
    result_path = context.component_result_path.parent / "route/route_result.json"
    document = json.loads(result_path.read_text(encoding="utf-8"))
    document["qa_ref"]["sha256"] = hashlib.sha256(
        qa_path.read_bytes()
    ).hexdigest()
    result_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Plan hash"):
        load_published_route(
            context.store,
            context.component_result_path,
            page_id=context.page_id,
        )


def test_native_writer_error_publishes_full_raster_fallback(route_context) -> None:
    context, policy, plans, _ = route_context

    def fail_assembly(plan: dict, path: Path) -> None:
        raise RuntimeError("native writer failed")

    result = finalize_page_route(
        replace(context, assemble_page=fail_assembly),
        renderer=_FakeRenderer(context.source_image_path, plans),
        policy=policy,
    )

    assert result["status"] == "raster_fallback"
    assert result["reason"] == "native_render_error"
    plan = _load_ref(context, result["plan_ref"])
    assert {route["selected_route"] for route in plan["routes"]} == {
        "raster_component"
    }


def test_renderer_error_publishes_full_raster_fallback(route_context) -> None:
    context, policy, plans, _ = route_context

    class BrokenRenderer(_FakeRenderer):
        def render_page(self, *args, **kwargs):
            raise RuntimeError("PowerPoint failed to start")

    result = finalize_page_route(
        context,
        renderer=BrokenRenderer(context.source_image_path, plans),
        policy=policy,
    )

    assert result["status"] == "raster_fallback"
    assert result["reason"] == "native_render_error"


def test_failed_fallback_publishes_full_raster_without_changing_component_result(
    route_context,
) -> None:
    context, policy, plans, _ = route_context
    component_bytes = context.component_result_path.read_bytes()
    renderer = _FakeRenderer(
        context.source_image_path,
        plans,
        native_fails={"shape_1"},
        forced_failures={"native-fallback"},
    )

    result = finalize_page_route(context, renderer=renderer, policy=policy)

    assert result["reason"] == "fallback_render_qa_failed"
    plan = _load_ref(context, result["plan_ref"])
    assert {route["selected_route"] for route in plan["routes"]} == {
        "raster_component"
    }
    assert context.component_result_path.read_bytes() == component_bytes


@pytest.mark.parametrize("artifact", ["ir_ref", "plan_ref", "qa_ref"])
def test_reentry_rejects_tampered_bound_artifact(route_context, artifact: str) -> None:
    context, policy, plans, _ = route_context
    renderer = _FakeRenderer(context.source_image_path, plans)
    result = finalize_page_route(context, renderer=renderer, policy=policy)
    path = context.store.root / result[artifact]["path"]
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="hash"):
        finalize_page_route(context, renderer=renderer, policy=policy)


def test_reentry_returns_existing_result_without_reassembly(route_context) -> None:
    context, policy, plans, assemble_calls = route_context
    renderer = _FakeRenderer(context.source_image_path, plans)
    first = finalize_page_route(context, renderer=renderer, policy=policy)
    call_count = len(assemble_calls)

    second = finalize_page_route(context, renderer=renderer, policy=policy)

    assert second == first
    assert len(assemble_calls) == call_count


@pytest.mark.parametrize("path", ["../outside.json", "C:/outside.json"])
def test_route_result_path_escape_is_rejected(route_context, path: str) -> None:
    context, policy, plans, _ = route_context
    renderer = _FakeRenderer(context.source_image_path, plans)
    finalize_page_route(context, renderer=renderer, policy=policy)
    result_path = (
        context.component_result_path.parent / "route/route_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["plan_ref"]["path"] = path
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact ref"):
        finalize_page_route(context, renderer=renderer, policy=policy)


def test_route_result_reparse_point_is_rejected(route_context) -> None:
    context, policy, plans, _ = route_context
    renderer = _FakeRenderer(context.source_image_path, plans)
    result = finalize_page_route(context, renderer=renderer, policy=policy)
    plan_path = context.store.root / result["plan_ref"]["path"]
    link = plan_path.with_name("linked-plan.json")
    try:
        link.symlink_to(plan_path)
    except OSError as error:
        pytest.skip(f"Cannot create file link: {error}")
    result_path = context.component_result_path.parent / "route/route_result.json"
    document = json.loads(result_path.read_text(encoding="utf-8"))
    document["plan_ref"]["path"] = link.relative_to(context.store.root).as_posix()
    result_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="link|reparse"):
        finalize_page_route(context, renderer=renderer, policy=policy)


def test_candidate_temp_file_is_not_treated_as_result(route_context) -> None:
    context, policy, _, _ = route_context
    route_root = context.component_result_path.parent / "route"
    route_root.mkdir()
    temporary = route_root / "route_result.json.tmp"
    temporary.write_text("candidate", encoding="utf-8")

    result = finalize_page_route(
        context, renderer=_UnavailableRenderer(), policy=policy
    )

    assert result["status"] == "raster_fallback"
    assert temporary.read_text(encoding="utf-8") == "candidate"
