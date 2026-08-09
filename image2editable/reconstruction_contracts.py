from __future__ import annotations

from copy import deepcopy
import math
from pathlib import PurePosixPath


IR_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
ROUTE_RESULT_SCHEMA_VERSION = 1

REPRESENTATION_KINDS = frozenset(
    {"editable_text", "native_shape", "raster_component"}
)
ROUTE_STATUSES = frozenset({"native_accepted", "raster_fallback"})
RELATION_KINDS = frozenset(
    {"parent", "child", "contains", "text_owner", "overlaps", "z_order", "aligned_with"}
)


def _exact_fields(value: object, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _schema_version(value: object, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} schema_version is invalid")


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or "/" in value or "\\" in value:
        raise ValueError(f"{label} is invalid")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _artifact_ref(value: object) -> dict:
    reference = _exact_fields(value, {"path", "sha256"}, "artifact ref")
    path = reference["path"]
    if type(path) is not str or not path or "\\" in path or ":" in path:
        raise ValueError("artifact ref path is invalid")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError("artifact ref path is invalid")
    _sha256(reference["sha256"], "artifact ref sha256")
    return reference


def _artifact_refs(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("artifact refs are invalid")
    for reference in value:
        _artifact_ref(reference)
    return value


def _confidence(value: object) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError("candidate confidence is invalid")
    return float(value)


def _strings(value: object, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(type(item) is not str or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _candidate(value: object) -> dict:
    candidate = _exact_fields(
        value,
        {"kind", "confidence", "payload", "evidence_refs", "required_qa_checks"},
        "candidate representation",
    )
    if candidate["kind"] not in REPRESENTATION_KINDS:
        raise ValueError("candidate representation kind is invalid")
    _confidence(candidate["confidence"])
    if not isinstance(candidate["payload"], dict):
        raise ValueError("candidate representation payload is invalid")
    if candidate["kind"] == "raster_component":
        payload = _exact_fields(
            candidate["payload"], {"asset_ref"}, "raster candidate payload"
        )
        _artifact_ref(payload["asset_ref"])
    _artifact_refs(candidate["evidence_refs"])
    _strings(candidate["required_qa_checks"], "candidate QA checks")
    return candidate


def validate_reconstruction_ir(value: object) -> dict:
    """Validate and copy a versioned reconstruction intermediate representation."""

    document = _exact_fields(
        value, {"schema_version", "page_id", "canvas", "objects"}, "reconstruction IR"
    )
    _schema_version(document["schema_version"], IR_SCHEMA_VERSION, "reconstruction IR")
    _identifier(document["page_id"], "reconstruction IR page_id")
    canvas = _exact_fields(document["canvas"], {"width", "height"}, "canvas")
    if any(type(canvas[name]) is not int or canvas[name] <= 0 for name in canvas):
        raise ValueError("canvas dimensions are invalid")
    objects = document["objects"]
    if not isinstance(objects, list):
        raise ValueError("reconstruction IR objects are invalid")

    object_ids: set[str] = set()
    for item in objects:
        item = _exact_fields(
            item,
            {
                "id",
                "bbox",
                "z_index",
                "source_refs",
                "mask_ref",
                "relations",
                "candidate_representations",
            },
            "reconstruction object",
        )
        object_id = _identifier(item["id"], "reconstruction object id")
        if object_id in object_ids:
            raise ValueError("duplicate object id")
        object_ids.add(object_id)
        bbox = item["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(coordinate) is not int for coordinate in bbox)
            or not (0 <= bbox[0] < bbox[2] <= canvas["width"])
            or not (0 <= bbox[1] < bbox[3] <= canvas["height"])
        ):
            raise ValueError("reconstruction object bbox is invalid")
        if type(item["z_index"]) is not int or item["z_index"] < 0:
            raise ValueError("reconstruction object z_index is invalid")
        _artifact_refs(item["source_refs"])
        if item["mask_ref"] is not None:
            _artifact_ref(item["mask_ref"])
        if not isinstance(item["relations"], list):
            raise ValueError("reconstruction object relations are invalid")
        for relation in item["relations"]:
            relation = _exact_fields(relation, {"kind", "target_id"}, "relation")
            if relation["kind"] not in RELATION_KINDS:
                raise ValueError("relation kind is invalid")
            _identifier(relation["target_id"], "relation target_id")
        candidates = item["candidate_representations"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidate representations are invalid")
        kinds = []
        for candidate in candidates:
            kinds.append(_candidate(candidate)["kind"])
        if len(kinds) != len(set(kinds)):
            raise ValueError("candidate representation kinds are duplicated")
        if "native_shape" in kinds and "raster_component" not in kinds:
            raise ValueError("native_shape requires raster_component candidate")

    return deepcopy(document)


def validate_reconstruction_plan(value: object, *, ir: dict) -> dict:
    """Validate and copy a route plan bound to a reconstruction IR."""

    validated_ir = validate_reconstruction_ir(ir)
    document = _exact_fields(
        value,
        {"schema_version", "page_id", "ir_sha256", "adapter", "routes"},
        "reconstruction plan",
    )
    _schema_version(
        document["schema_version"], PLAN_SCHEMA_VERSION, "reconstruction plan"
    )
    if document["page_id"] != validated_ir["page_id"]:
        raise ValueError("reconstruction plan page_id is invalid")
    _sha256(document["ir_sha256"], "reconstruction plan ir_sha256")
    if type(document["adapter"]) is not str or not document["adapter"]:
        raise ValueError("reconstruction plan adapter is invalid")
    routes = document["routes"]
    if not isinstance(routes, list):
        raise ValueError("reconstruction plan routes are invalid")

    objects = {item["id"]: item for item in validated_ir["objects"]}
    route_ids: list[str] = []
    for route in routes:
        route = _exact_fields(
            route,
            {
                "object_id",
                "selected_route",
                "fallback_route",
                "candidate_confidence",
                "evidence_refs",
                "qa_requirements",
            },
            "reconstruction route",
        )
        object_id = _identifier(route["object_id"], "route object_id")
        route_ids.append(object_id)
        selected = route["selected_route"]
        fallback = route["fallback_route"]
        if selected not in REPRESENTATION_KINDS:
            raise ValueError("selected route is invalid")
        if fallback is not None and fallback not in REPRESENTATION_KINDS:
            raise ValueError("fallback route is invalid")
        _confidence(route["candidate_confidence"])
        _artifact_refs(route["evidence_refs"])
        _strings(route["qa_requirements"], "route QA requirements")
        if object_id not in objects:
            continue
        candidate_kinds = {
            candidate["kind"]
            for candidate in objects[object_id]["candidate_representations"]
        }
        if selected not in candidate_kinds:
            raise ValueError("selected route has no candidate")
        if fallback is not None and fallback not in candidate_kinds:
            raise ValueError("fallback route has no candidate")
        if selected == "native_shape" and fallback != "raster_component":
            raise ValueError("native_shape route requires raster fallback")

    if len(route_ids) != len(set(route_ids)) or set(route_ids) != set(objects):
        raise ValueError("reconstruction plan routes do not match IR objects")
    return deepcopy(document)


def validate_route_result(value: object) -> dict:
    """Validate and copy the hashes and artifacts produced by route execution."""

    document = _exact_fields(
        value,
        {
            "schema_version",
            "page_id",
            "status",
            "component_result_sha256",
            "ir_ref",
            "plan_ref",
            "qa_ref",
            "reason",
        },
        "route result",
    )
    _schema_version(
        document["schema_version"], ROUTE_RESULT_SCHEMA_VERSION, "route result"
    )
    _identifier(document["page_id"], "route result page_id")
    if document["status"] not in ROUTE_STATUSES:
        raise ValueError("route result status is invalid")
    _sha256(document["component_result_sha256"], "component result sha256")
    _artifact_ref(document["ir_ref"])
    _artifact_ref(document["plan_ref"])
    if document["qa_ref"] is not None:
        _artifact_ref(document["qa_ref"])
    if document["reason"] is not None and (
        type(document["reason"]) is not str or not document["reason"]
    ):
        raise ValueError("route result reason is invalid")
    return deepcopy(document)
