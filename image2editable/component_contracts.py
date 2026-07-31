from __future__ import annotations

from pathlib import PurePosixPath
import math


AGENT_PROVIDERS = frozenset({"host", "local"})
MAX_REPAIR_ROUNDS = 5
COMPONENT_STATES = frozenset({"pending", "failed", "frozen", "inactive"})
COMPONENT_KINDS = frozenset({"parent", "child", "text"})
COMPONENT_EVIDENCE_NAMES = frozenset(
    {
        "source.png",
        "numbered-masks.png",
        "ocr-overlay.png",
        "ownership.png",
        "reconstructed.png",
        "difference.png",
        "component-graph.json",
        "quality-report.json",
    }
)

_COMPONENT_AGENT_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "page_id",
        "provider",
        "repair_round",
        "source_sha256",
        "graph_sha256",
        "candidate_ids",
        "frozen_ids",
        "evidence",
    }
)

_COMPONENT_NODE_FIELDS = frozenset(
    {
        "id",
        "kind",
        "parent_id",
        "state",
        "mask",
        "mask_sha256",
        "bbox",
        "z_index",
        "text_ids",
    }
)
_FROZEN_FIELDS = (
    "state",
    "kind",
    "mask",
    "mask_sha256",
    "bbox",
    "z_index",
    "parent_id",
    "text_ids",
)
_RENDER_STATES = frozenset({"pending", "frozen"})


def validate_agent_provider(value: object) -> str:
    if type(value) is not str or value not in AGENT_PROVIDERS:
        raise ValueError(
            "Invalid agent_provider; expected one of: host, local"
        )
    return value


def _validate_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is invalid")
    return value


def validate_repair_round(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_REPAIR_ROUNDS:
        raise ValueError(
            f"repair_round must be between 1 and {MAX_REPAIR_ROUNDS}"
        )
    return value


_COMPONENT_PLAN_FIELDS = frozenset(
    {"schema_version", "kind", "page_id", "provider", "repair_round", "request_sha256", "actions"}
)
_COMPONENT_ACTION_FIELDS = frozenset(
    {"action", "object_ids", "parameters", "confidence", "evidence"}
)
_ACTION_PARAMETERS = {
    "accept": frozenset(),
    "merge": frozenset(),
    "split": frozenset({"parts"}),
    "expand": frozenset({"margin_ratio"}),
    "shrink": frozenset({"margin_ratio"}),
    "retry_with_box": frozenset({"box"}),
    "retry_with_points": frozenset({"positive", "negative"}),
    "attach_text": frozenset(),
    "collapse_to_parent": frozenset(),
}


def _validate_normalized_point(value: object, field: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) not in {int, float} or not math.isfinite(item) or not 0 <= item <= 1 for item in value)
    ):
        raise ValueError(f"component action {field} coordinates are invalid")


def validate_component_plan(plan: object, *, request: dict) -> dict:
    validate_component_agent_request(request)
    if not isinstance(plan, dict) or set(plan) != _COMPONENT_PLAN_FIELDS:
        raise ValueError("component plan fields are invalid")
    if plan["schema_version"] != 1 or type(plan["schema_version"]) is not int:
        raise ValueError("component plan schema_version is invalid")
    if plan["kind"] != "component_plan":
        raise ValueError("component plan kind is invalid")
    for field in ("page_id", "provider", "repair_round"):
        if plan[field] != request[field]:
            raise ValueError(f"component plan {field} does not match current request")
    validate_agent_provider(plan["provider"])
    validate_repair_round(plan["repair_round"])
    _validate_sha256(plan["request_sha256"], "request_sha256")
    actions = plan["actions"]
    if not isinstance(actions, list):
        raise ValueError("component plan actions must be a list")
    known_ids = set(request["candidate_ids"]) | set(request["frozen_ids"])
    touched = set()
    for action in actions:
        if not isinstance(action, dict) or set(action) != _COMPONENT_ACTION_FIELDS:
            raise ValueError("component action fields are invalid")
        name = action["action"]
        if type(name) is not str or name not in _ACTION_PARAMETERS:
            raise ValueError("component action is invalid")
        object_ids = action["object_ids"]
        if (
            not isinstance(object_ids, list) or not object_ids
            or any(type(value) is not str for value in object_ids)
            or len(object_ids) != len(set(object_ids))
            or any(value not in known_ids for value in object_ids)
        ):
            raise ValueError("component action object_ids are invalid")
        if set(object_ids) & set(request["frozen_ids"]):
            raise ValueError("component action object is frozen")
        if touched & set(object_ids):
            raise ValueError("component plan has conflicting object actions")
        touched.update(object_ids)
        parameters = action["parameters"]
        if not isinstance(parameters, dict) or set(parameters) != _ACTION_PARAMETERS[name]:
            raise ValueError("component action parameters are invalid")
        confidence = action["confidence"]
        if type(confidence) not in {int, float} or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("component action confidence is invalid")
        evidence = action["evidence"]
        if not isinstance(evidence, list) or not evidence or any(type(item) is not str or not item.strip() for item in evidence):
            raise ValueError("component action evidence is invalid")
        if name == "split" and (type(parameters["parts"]) is not int or parameters["parts"] < 2):
            raise ValueError("component action split parts are invalid")
        if name in {"expand", "shrink"} and (
            type(parameters["margin_ratio"]) not in {int, float}
            or not math.isfinite(parameters["margin_ratio"])
            or not 0 < parameters["margin_ratio"] <= 1
        ):
            raise ValueError("component action margin_ratio is invalid")
        if name == "retry_with_box":
            box = parameters["box"]
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("component action box coordinates are invalid")
            _validate_normalized_point(box[:2], "box")
            _validate_normalized_point(box[2:], "box")
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("component action box coordinates are invalid")
        if name == "retry_with_points":
            for field in ("positive", "negative"):
                points = parameters[field]
                if not isinstance(points, list):
                    raise ValueError(f"component action {field} coordinates are invalid")
                for point in points:
                    _validate_normalized_point(point, field)
            if not parameters["positive"]:
                raise ValueError("component action positive coordinates are invalid")
    return plan


def validate_component_agent_request(request: object) -> dict:
    if not isinstance(request, dict) or set(request) != _COMPONENT_AGENT_REQUEST_FIELDS:
        raise ValueError("component agent request fields are invalid")
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("component agent request schema_version is invalid")
    page_id = request["page_id"]
    if (
        type(page_id) is not str
        or not page_id
        or "/" in page_id
        or "\\" in page_id
        or page_id in {".", ".."}
    ):
        raise ValueError("component agent request page_id is invalid")
    validate_agent_provider(request["provider"])
    validate_repair_round(request["repair_round"])
    _validate_sha256(request["source_sha256"], "source_sha256")
    _validate_sha256(request["graph_sha256"], "graph_sha256")
    for field in ("candidate_ids", "frozen_ids"):
        values = request[field]
        if (
            not isinstance(values, list)
            or any(type(value) is not str or not value for value in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"component agent request {field} is invalid")
    if set(request["candidate_ids"]) & set(request["frozen_ids"]):
        raise ValueError("candidate_ids and frozen_ids must be disjoint")
    evidence = request["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != COMPONENT_EVIDENCE_NAMES:
        raise ValueError("component agent request evidence fields are invalid")
    for name, record in evidence.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"component evidence record is invalid: {name}")
        path = record["path"]
        if type(path) is not str or not path or "\\" in path or ":" in path:
            raise ValueError(f"component evidence path is invalid: {name}")
        pure_path = PurePosixPath(path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or pure_path != PurePosixPath(name)
        ):
            raise ValueError(f"component evidence path is invalid: {name}")
        _validate_sha256(record["sha256"], f"component evidence sha256: {name}")
    return request


def _validate_component_node(node: object) -> dict:
    if not isinstance(node, dict) or set(node) != _COMPONENT_NODE_FIELDS:
        raise ValueError("component node fields are invalid")
    component_id = node["id"]
    if type(component_id) is not str or not component_id.strip():
        raise ValueError("component id must be a non-empty string")
    if type(node["kind"]) is not str or node["kind"] not in COMPONENT_KINDS:
        raise ValueError("component kind is invalid")
    if type(node["state"]) is not str or node["state"] not in COMPONENT_STATES:
        raise ValueError("component state is invalid")
    parent_id = node["parent_id"]
    if parent_id is not None and (
        type(parent_id) is not str or not parent_id.strip()
    ):
        raise ValueError("component parent_id is invalid")
    mask = node["mask"]
    if type(mask) is not str or not mask or "\\" in mask or ":" in mask:
        raise ValueError("component mask path is invalid")
    mask_path = PurePosixPath(mask)
    if (
        mask_path.is_absolute()
        or ".." in mask_path.parts
        or not mask_path.parts
        or mask_path.parts[0] != "masks"
    ):
        raise ValueError("component mask path is invalid")
    digest = node["mask_sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("component mask_sha256 is invalid")
    bbox = node["bbox"]
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(type(value) is int for value in bbox)
        or min(bbox) < 0
        or bbox[0] >= bbox[2]
        or bbox[1] >= bbox[3]
    ):
        raise ValueError("component bbox is invalid")
    if type(node["z_index"]) is not int or node["z_index"] < 0:
        raise ValueError("component z_index is invalid")
    text_ids = node["text_ids"]
    if (
        not isinstance(text_ids, list)
        or any(type(text_id) is not str or not text_id for text_id in text_ids)
        or len(text_ids) != len(set(text_ids))
    ):
        raise ValueError("component text_ids are invalid")
    return node


def is_render_active_component(node: object) -> bool:
    validated = _validate_component_node(node)
    return (
        validated["kind"] != "text"
        and validated["state"] in _RENDER_STATES
    )


def validate_component_graph(graph: object) -> dict:
    if not isinstance(graph, dict) or set(graph) != {"nodes"}:
        raise ValueError("component graph fields are invalid")
    if not isinstance(graph["nodes"], list):
        raise ValueError("component graph nodes must be a list")
    nodes = [_validate_component_node(node) for node in graph["nodes"]]
    by_id = {node["id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("component ids must be unique")
    if len({node["mask"] for node in nodes}) != len(nodes):
        raise ValueError("component mask paths must be unique")

    for node in nodes:
        parent_id = node["parent_id"]
        if node["kind"] in {"parent", "text"} and parent_id is not None:
            raise ValueError(f"{node['kind']} component cannot have a parent")
        if node["kind"] == "child":
            if parent_id is None or parent_id not in by_id:
                raise ValueError("child component parent is missing")
            if by_id[parent_id]["kind"] == "text":
                raise ValueError("child component parent cannot be text")
        for text_id in node["text_ids"]:
            if text_id not in by_id or by_id[text_id]["kind"] != "text":
                raise ValueError("component text ownership references unknown text")
            if node["state"] == "frozen" and by_id[text_id]["state"] != "frozen":
                raise ValueError("frozen component requires frozen linked text")

    for node in nodes:
        ancestors = set()
        parent_id = node["parent_id"]
        while parent_id is not None:
            if parent_id in ancestors:
                raise ValueError("component graph contains a parent cycle")
            ancestors.add(parent_id)
            parent = by_id[parent_id]
            if is_render_active_component(node) and parent["state"] != "inactive":
                raise ValueError("parent and child cannot render together")
            if is_render_active_component(parent) and node["state"] != "inactive":
                raise ValueError("parent and child cannot render together")
            parent_id = parent["parent_id"]

    text_owners: dict[str, str] = {}
    active_z_indexes = set()
    for node in nodes:
        if not is_render_active_component(node):
            continue
        if node["z_index"] in active_z_indexes:
            raise ValueError("active component z_index values must be unique")
        active_z_indexes.add(node["z_index"])
        for text_id in node["text_ids"]:
            previous = text_owners.setdefault(text_id, node["id"])
            if previous != node["id"]:
                raise ValueError("text component has multiple active owners")
    return graph


def validate_graph_transition(*, before: object, after: object) -> dict:
    before_graph = validate_component_graph(before)
    if not isinstance(after, dict) or set(after) != {"nodes"}:
        raise ValueError("component graph fields are invalid")
    if not isinstance(after["nodes"], list):
        raise ValueError("component graph nodes must be a list")
    after_nodes = {
        node["id"]: node
        for node in after["nodes"]
        if isinstance(node, dict) and type(node.get("id")) is str
    }
    for node in before_graph["nodes"]:
        if node["state"] != "frozen":
            continue
        replacement = after_nodes.get(node["id"])
        if replacement is None or any(
            replacement.get(field) != node[field] for field in _FROZEN_FIELDS
        ):
            raise ValueError(f"frozen component {node['id']} cannot change")
    return validate_component_graph(after)
