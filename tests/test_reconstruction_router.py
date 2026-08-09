from copy import deepcopy

import pytest

from image2editable.reconstruction_router import route_reconstruction


SHA = "0" * 64


def _ref(path: str) -> dict:
    return {"path": path, "sha256": SHA}


def _ir_with_shape(*, geometry_score: float = 0.99, color_mad: float = 3.0) -> dict:
    asset = _ref("assets/component.png")
    evidence = [_ref("masks/component.png")]
    return {
        "schema_version": 1,
        "page_id": "page_001",
        "canvas": {"width": 1600, "height": 900},
        "objects": [
            {
                "id": "component_0001",
                "bbox": [10, 20, 110, 120],
                "z_index": 1,
                "source_refs": [asset],
                "mask_ref": evidence[0],
                "relations": [],
                "candidate_representations": [
                    {
                        "kind": "raster_component",
                        "confidence": 1.0,
                        "payload": {"asset_ref": asset},
                        "evidence_refs": evidence,
                        "required_qa_checks": [
                            "ownership",
                            "render_difference",
                        ],
                    },
                    {
                        "kind": "native_shape",
                        "confidence": geometry_score,
                        "payload": {
                            "shape_type": "rectangle",
                            "bbox": [10, 20, 110, 120],
                            "geometry_score": geometry_score,
                            "fill_rgb": [30, 90, 180],
                            "color_mad": color_mad,
                        },
                        "evidence_refs": evidence,
                        "required_qa_checks": [
                            "pptx_native_shape",
                            "render_difference",
                        ],
                    },
                ],
            }
        ],
    }


def _policy(*, enabled: bool = True) -> dict:
    return {
        "schema_version": 1,
        "native_shape_enabled": enabled,
        "allowed_shapes": ["rectangle", "rounded_rectangle", "ellipse", "line"],
        "min_geometry_score": 0.99,
        "max_color_mad": 3.0,
    }


def _route(ir: dict, **overrides) -> dict:
    arguments = {
        "adapter": "pptx",
        "capabilities": {"editable_text", "native_shape", "raster_component"},
        "policy": _policy(),
        "authoritative_render_qa": True,
    }
    arguments.update(overrides)
    return route_reconstruction(ir, **arguments)


def test_native_route_is_disabled_without_enabled_profile() -> None:
    plan = _route(_ir_with_shape(), policy=_policy(enabled=False))

    assert plan["routes"][0]["selected_route"] == "raster_component"


def test_native_route_requires_authoritative_render_qa() -> None:
    plan = _route(_ir_with_shape(), authoritative_render_qa=False)

    assert plan["routes"][0]["selected_route"] == "raster_component"


def test_native_route_requires_adapter_capability() -> None:
    plan = _route(
        _ir_with_shape(), capabilities={"editable_text", "raster_component"}
    )

    assert plan["routes"][0]["selected_route"] == "raster_component"


def test_native_route_accepts_inclusive_policy_thresholds() -> None:
    plan = _route(_ir_with_shape(geometry_score=0.99, color_mad=3.0))

    route = plan["routes"][0]
    assert route["selected_route"] == "native_shape"
    assert route["fallback_route"] == "raster_component"


@pytest.mark.parametrize(
    ("geometry_score", "color_mad"),
    [(0.989, 3.0), (0.99, 3.001)],
)
def test_native_route_rejects_failed_policy_threshold(
    geometry_score: float, color_mad: float
) -> None:
    plan = _route(
        _ir_with_shape(geometry_score=geometry_score, color_mad=color_mad)
    )

    assert plan["routes"][0]["selected_route"] == "raster_component"


def test_editable_text_is_selected_when_supported() -> None:
    ir = _ir_with_shape()
    ir["objects"] = [
        {
            "id": "text_0001",
            "bbox": [10, 20, 110, 50],
            "z_index": 1,
            "source_refs": [],
            "mask_ref": None,
            "relations": [],
            "candidate_representations": [
                {
                    "kind": "editable_text",
                    "confidence": 1.0,
                    "payload": {"text": "Editable", "box": [10, 20, 110, 50]},
                    "evidence_refs": [],
                    "required_qa_checks": ["editable_text", "render_difference"],
                }
            ],
        }
    ]

    plan = _route(ir)

    assert plan["routes"][0]["selected_route"] == "editable_text"
    assert plan["routes"][0]["fallback_route"] is None


def test_route_fails_when_raster_fallback_is_missing() -> None:
    ir = _ir_with_shape()
    ir["objects"][0]["candidate_representations"] = [
        ir["objects"][0]["candidate_representations"][1]
    ]

    with pytest.raises(ValueError, match="raster_component"):
        _route(ir)


def test_router_does_not_modify_ir_or_policy() -> None:
    ir = _ir_with_shape()
    policy = _policy()
    original_ir = deepcopy(ir)
    original_policy = deepcopy(policy)

    _route(ir, policy=policy)

    assert ir == original_ir
    assert policy == original_policy
