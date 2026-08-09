from __future__ import annotations

from copy import deepcopy
import math

from image2editable.reconstruction_contracts import (
    REPRESENTATION_KINDS,
    reconstruction_ir_sha256,
    validate_reconstruction_ir,
    validate_reconstruction_plan,
)


ROUTE_POLICY_SCHEMA_VERSION = 1
SHAPE_TYPES = frozenset({"rectangle", "rounded_rectangle", "ellipse", "line"})


def _number(value: object, label: str, *, maximum: float | None = None) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError(f"route policy {label} is invalid")
    if maximum is not None and value > maximum:
        raise ValueError(f"route policy {label} is invalid")
    return float(value)


def _validate_policy(value: object) -> dict:
    fields = {
        "schema_version",
        "native_shape_enabled",
        "allowed_shapes",
        "min_geometry_score",
        "max_color_mad",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("route policy fields are invalid")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("route policy schema_version is invalid")
    if type(value["native_shape_enabled"]) is not bool:
        raise ValueError("route policy native_shape_enabled is invalid")
    allowed = value["allowed_shapes"]
    if (
        not isinstance(allowed, list)
        or any(type(shape) is not str or shape not in SHAPE_TYPES for shape in allowed)
        or len(allowed) != len(set(allowed))
    ):
        raise ValueError("route policy allowed_shapes is invalid")
    _number(value["min_geometry_score"], "min_geometry_score", maximum=1.0)
    _number(value["max_color_mad"], "max_color_mad")
    return deepcopy(value)


def _native_shape_is_allowed(
    candidate: dict,
    *,
    policy: dict,
    authoritative_render_qa: bool,
) -> bool:
    if not policy["native_shape_enabled"] or not authoritative_render_qa:
        return False
    payload = candidate["payload"]
    if not isinstance(payload, dict):
        return False
    shape_type = payload.get("shape_type")
    geometry_score = payload.get("geometry_score")
    color_mad = payload.get("color_mad")
    return (
        shape_type in policy["allowed_shapes"]
        and type(geometry_score) in {int, float}
        and math.isfinite(geometry_score)
        and geometry_score >= policy["min_geometry_score"]
        and type(color_mad) in {int, float}
        and math.isfinite(color_mad)
        and color_mad <= policy["max_color_mad"]
    )


def route_reconstruction(
    ir: dict,
    *,
    adapter: str,
    capabilities: set[str],
    policy: dict,
    authoritative_render_qa: bool,
) -> dict:
    """Choose one conservative target representation for every IR object."""

    validated_ir = validate_reconstruction_ir(ir)
    validated_policy = _validate_policy(policy)
    if type(adapter) is not str or not adapter:
        raise ValueError("reconstruction adapter is invalid")
    if (
        not isinstance(capabilities, set)
        or any(capability not in REPRESENTATION_KINDS for capability in capabilities)
    ):
        raise ValueError("reconstruction capabilities are invalid")
    if type(authoritative_render_qa) is not bool:
        raise ValueError("authoritative_render_qa is invalid")

    routes = []
    for item in validated_ir["objects"]:
        candidates = {
            candidate["kind"]: candidate
            for candidate in item["candidate_representations"]
        }
        selected = None
        fallback = None
        if "editable_text" in candidates and "editable_text" in capabilities:
            selected = candidates["editable_text"]
        elif (
            "native_shape" in candidates
            and "native_shape" in capabilities
            and _native_shape_is_allowed(
                candidates["native_shape"],
                policy=validated_policy,
                authoritative_render_qa=authoritative_render_qa,
            )
        ):
            if "raster_component" not in candidates:
                raise ValueError("native_shape requires raster_component fallback")
            selected = candidates["native_shape"]
            fallback = "raster_component"
        elif (
            "raster_component" in candidates
            and "raster_component" in capabilities
        ):
            selected = candidates["raster_component"]
        if selected is None:
            raise ValueError(
                f"no supported reconstruction route for object: {item['id']}"
            )
        routes.append(
            {
                "object_id": item["id"],
                "selected_route": selected["kind"],
                "fallback_route": fallback,
                "candidate_confidence": selected["confidence"],
                "evidence_refs": deepcopy(selected["evidence_refs"]),
                "qa_requirements": deepcopy(selected["required_qa_checks"]),
            }
        )

    plan = {
        "schema_version": 1,
        "page_id": validated_ir["page_id"],
        "ir_sha256": reconstruction_ir_sha256(validated_ir),
        "adapter": adapter,
        "routes": routes,
    }
    return validate_reconstruction_plan(plan, ir=validated_ir)
