import hashlib
from pathlib import Path

from image2editable import legacy
from image2editable.reconstruction_ir import build_reconstruction_ir
from image2editable.store import RunStore


def _ref(path: str, digit: str) -> dict:
    return {"path": path, "sha256": digit * 64}


def test_ir_keeps_raster_candidate_and_graph_z_order() -> None:
    graph = {
        "nodes": [
            {
                "id": "front",
                "kind": "child",
                "state": "frozen",
                "bbox": [5, 5, 15, 15],
                "z_index": 2,
            },
            {
                "id": "back",
                "kind": "parent",
                "state": "frozen",
                "bbox": [0, 0, 20, 20],
                "z_index": 1,
            },
        ]
    }
    assets = {
        "back": _ref("assets/back.png", "1"),
        "front": _ref("assets/front.png", "2"),
    }

    ir = build_reconstruction_ir(
        page_id="page_001",
        canvas=(100, 80),
        graph=graph,
        component_assets=assets,
        text_items=[],
        shape_candidates={},
    )

    assert [item["id"] for item in ir["objects"]] == ["back", "front"]
    assert all(
        item["candidate_representations"][0]["kind"] == "raster_component"
        for item in ir["objects"]
    )


def test_ir_keeps_frozen_text_once_and_excludes_nonfrozen_visuals() -> None:
    graph = {
        "nodes": [
            {
                "id": "parent_0001",
                "kind": "parent",
                "state": "inactive",
                "bbox": [0, 0, 90, 70],
                "z_index": 0,
                "parent_id": None,
                "text_ids": [],
            },
            {
                "id": "component_0001",
                "kind": "child",
                "state": "frozen",
                "bbox": [10, 10, 70, 60],
                "z_index": 1,
                "parent_id": "parent_0001",
                "text_ids": ["text_0001"],
            },
            {
                "id": "pending_0001",
                "kind": "child",
                "state": "pending",
                "bbox": [75, 10, 85, 20],
                "z_index": 2,
                "parent_id": None,
                "text_ids": [],
            },
            {
                "id": "text_0001",
                "kind": "text",
                "state": "frozen",
                "bbox": [20, 20, 60, 35],
                "z_index": 3,
                "parent_id": None,
                "text_ids": [],
            },
        ]
    }

    ir = build_reconstruction_ir(
        page_id="page_001",
        canvas=(100, 80),
        graph=graph,
        component_assets={
            "component_0001": _ref("assets/component.png", "3")
        },
        text_items=[
            {"id": "text_0001", "text": "Editable", "box": [20, 20, 60, 35]}
        ],
        shape_candidates={},
    )

    assert [item["id"] for item in ir["objects"]] == [
        "component_0001",
        "text_0001",
    ]
    component, text = ir["objects"]
    assert {tuple(relation.values()) for relation in component["relations"]} >= {
        ("parent", "parent_0001"),
        ("text_owner", "text_0001"),
    }
    assert text["candidate_representations"][0]["kind"] == "editable_text"


def test_shape_candidates_are_appended_after_raster_without_mutation() -> None:
    candidate = {
        "kind": "native_shape",
        "confidence": 0.95,
        "payload": {"shape": "rectangle"},
        "evidence_refs": [_ref("evidence/shape.json", "4")],
        "required_qa_checks": ["pptx_native_shape", "render_difference"],
    }
    graph = {
        "nodes": [
            {
                "id": "component_0001",
                "kind": "child",
                "state": "frozen",
                "bbox": [10, 10, 70, 60],
                "z_index": 1,
            }
        ]
    }

    ir = build_reconstruction_ir(
        page_id="page_001",
        canvas=(100, 80),
        graph=graph,
        component_assets={
            "component_0001": _ref("assets/component.png", "3")
        },
        text_items=[],
        shape_candidates={"component_0001": [candidate]},
    )

    assert [
        item["kind"] for item in ir["objects"][0]["candidate_representations"]
    ] == ["raster_component", "native_shape"]
    assert candidate["payload"] == {"shape": "rectangle"}


def test_accepted_reconstruction_inputs_bind_component_assets(tmp_path: Path) -> None:
    component = tmp_path / "pages/page_001/reconstruction/component.png"
    component.parent.mkdir(parents=True)
    component.write_bytes(b"component")
    graph = {"nodes": []}

    inputs = legacy._accepted_reconstruction_inputs(
        RunStore(tmp_path),
        prepared={"img_width": 100, "img_height": 80},
        result={
            "page_id": "page_001",
            "text_items": [{"id": "text_0001"}],
        },
        graph=graph,
        components=[
            {"component_id": "component_0001", "path": str(component)}
        ],
    )

    assert inputs == {
        "page_id": "page_001",
        "canvas": (100, 80),
        "graph": graph,
        "component_assets": {
            "component_0001": {
                "path": "pages/page_001/reconstruction/component.png",
                "sha256": hashlib.sha256(b"component").hexdigest(),
            }
        },
        "text_items": [{"id": "text_0001"}],
    }
