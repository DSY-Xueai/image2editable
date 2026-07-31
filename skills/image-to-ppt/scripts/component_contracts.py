from __future__ import annotations

from pathlib import PurePosixPath


AGENT_PROVIDERS = frozenset({"host", "local"})
MAX_REPAIR_ROUNDS = 5
COMPONENT_STATES = frozenset({"pending", "failed", "frozen", "inactive"})
COMPONENT_KINDS = frozenset({"parent", "child", "text"})

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
