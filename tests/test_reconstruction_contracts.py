from copy import deepcopy

import pytest

from image2editable.reconstruction_contracts import (
    validate_reconstruction_ir,
    validate_reconstruction_plan,
    validate_route_result,
)


SHA = "0" * 64


def _ir() -> dict:
    return {
        "schema_version": 1,
        "page_id": "page_001",
        "canvas": {"width": 1600, "height": 900},
        "objects": [
            {
                "id": "component_0001",
                "bbox": [10, 20, 110, 120],
                "z_index": 1,
                "source_refs": [
                    {"path": "assets/component.png", "sha256": SHA}
                ],
                "mask_ref": {"path": "masks/component.png", "sha256": SHA},
                "relations": [],
                "candidate_representations": [
                    {
                        "kind": "raster_component",
                        "confidence": 1.0,
                        "payload": {
                            "asset_ref": {
                                "path": "assets/component.png",
                                "sha256": SHA,
                            }
                        },
                        "evidence_refs": [
                            {"path": "masks/component.png", "sha256": SHA}
                        ],
                        "required_qa_checks": [
                            "ownership",
                            "render_difference",
                        ],
                    }
                ],
            }
        ],
    }


def _plan() -> dict:
    return {
        "schema_version": 1,
        "page_id": "page_001",
        "ir_sha256": SHA,
        "adapter": "pptx",
        "routes": [
            {
                "object_id": "component_0001",
                "selected_route": "raster_component",
                "fallback_route": None,
                "candidate_confidence": 1.0,
                "evidence_refs": [
                    {"path": "masks/component.png", "sha256": SHA}
                ],
                "qa_requirements": ["ownership", "render_difference"],
            }
        ],
    }


def test_reconstruction_ir_is_strict_and_immutable() -> None:
    value = _ir()
    original = deepcopy(value)

    validated = validate_reconstruction_ir(value)

    assert validated == original
    assert validated is not value
    assert value == original
    invalid = deepcopy(value)
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        validate_reconstruction_ir(invalid)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_schema_version_must_be_exact_integer(schema_version: object) -> None:
    value = _ir()
    value["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema_version"):
        validate_reconstruction_ir(value)


@pytest.mark.parametrize("path", ["../component.png", "C:/component.png"])
def test_ir_rejects_malformed_artifact_ref(path: str) -> None:
    value = _ir()
    value["objects"][0]["source_refs"][0]["path"] = path

    with pytest.raises(ValueError, match="artifact ref"):
        validate_reconstruction_ir(value)


def test_relation_may_reference_non_rendering_parent() -> None:
    value = _ir()
    value["objects"][0]["relations"] = [
        {"kind": "parent", "target_id": "parent_0001"}
    ]

    assert validate_reconstruction_ir(value) == value


def test_ir_rejects_duplicate_object_id() -> None:
    value = _ir()
    value["objects"].append(deepcopy(value["objects"][0]))

    with pytest.raises(ValueError, match="duplicate object"):
        validate_reconstruction_ir(value)


def test_ir_rejects_bbox_outside_canvas() -> None:
    value = _ir()
    value["objects"][0]["bbox"] = [10, 20, 1601, 120]

    with pytest.raises(ValueError, match="bbox"):
        validate_reconstruction_ir(value)


def test_plan_requires_raster_fallback_for_native_shape() -> None:
    ir = _ir()
    ir["objects"][0]["candidate_representations"].append(
        {
            "kind": "native_shape",
            "confidence": 0.99,
            "payload": {"shape": "rectangle"},
            "evidence_refs": [
                {"path": "masks/component.png", "sha256": SHA}
            ],
            "required_qa_checks": [
                "pptx_native_shape",
                "render_difference",
            ],
        }
    )
    plan = _plan()
    plan["routes"][0].update(
        selected_route="native_shape",
        candidate_confidence=0.99,
        qa_requirements=["pptx_native_shape", "render_difference"],
    )

    with pytest.raises(ValueError, match="fallback"):
        validate_reconstruction_plan(plan, ir=ir)


def test_plan_routes_must_match_ir_objects() -> None:
    plan = _plan()
    plan["routes"] = []

    with pytest.raises(ValueError, match="routes"):
        validate_reconstruction_plan(plan, ir=_ir())


def test_plan_is_strict_and_immutable() -> None:
    value = _plan()
    original = deepcopy(value)

    validated = validate_reconstruction_plan(value, ir=_ir())

    assert validated == original
    assert validated is not value
    assert value == original


def test_route_result_binds_component_result_and_plan_hashes() -> None:
    result = {
        "schema_version": 1,
        "page_id": "page_001",
        "status": "raster_fallback",
        "component_result_sha256": SHA,
        "ir_ref": {"path": "route/reconstruction-ir.json", "sha256": SHA},
        "plan_ref": {
            "path": "route/reconstruction-plan.json",
            "sha256": SHA,
        },
        "qa_ref": None,
        "reason": "renderer_unavailable",
    }
    original = deepcopy(result)

    validated = validate_route_result(result)

    assert validated == original
    assert validated is not result
    assert result == original
