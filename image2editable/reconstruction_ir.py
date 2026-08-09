from __future__ import annotations

from copy import deepcopy

from image2editable.reconstruction_contracts import validate_reconstruction_ir


def _mask_ref(node: dict) -> dict | None:
    if "mask" not in node or "mask_sha256" not in node:
        return None
    return {"path": node["mask"], "sha256": node["mask_sha256"]}


def _add_relation(item: dict, kind: str, target_id: str) -> None:
    relation = {"kind": kind, "target_id": target_id}
    if relation not in item["relations"]:
        item["relations"].append(relation)


def _add_bbox_relations(objects: list[dict]) -> None:
    for index, left_item in enumerate(objects):
        left = left_item["bbox"]
        for right_item in objects[index + 1 :]:
            right = right_item["bbox"]
            left_contains = (
                left[0] <= right[0]
                and left[1] <= right[1]
                and left[2] >= right[2]
                and left[3] >= right[3]
                and left != right
            )
            right_contains = (
                right[0] <= left[0]
                and right[1] <= left[1]
                and right[2] >= left[2]
                and right[3] >= left[3]
                and left != right
            )
            if left_contains:
                _add_relation(left_item, "contains", right_item["id"])
            elif right_contains:
                _add_relation(right_item, "contains", left_item["id"])
            elif (
                max(left[0], right[0]) < min(left[2], right[2])
                and max(left[1], right[1]) < min(left[3], right[3])
            ):
                _add_relation(left_item, "overlaps", right_item["id"])
                _add_relation(right_item, "overlaps", left_item["id"])


def build_reconstruction_ir(
    *,
    page_id: str,
    canvas: tuple[int, int],
    graph: dict,
    component_assets: dict[str, dict],
    text_items: list[dict],
    shape_candidates: dict[str, list[dict]],
) -> dict:
    """Build a target-independent IR from accepted component assets."""

    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, list):
        raise ValueError("component graph nodes are invalid")
    indexed_nodes = list(enumerate(nodes))
    frozen_visuals = [
        (index, node)
        for index, node in indexed_nodes
        if isinstance(node, dict)
        and node.get("kind") in {"parent", "child"}
        and node.get("state") == "frozen"
    ]
    frozen_text = {
        node.get("id"): (index, node)
        for index, node in indexed_nodes
        if isinstance(node, dict)
        and node.get("kind") == "text"
        and node.get("state") == "frozen"
    }
    text_by_id = {
        item.get("id"): item for item in text_items if isinstance(item, dict)
    }
    objects: list[tuple[int, dict]] = []

    for index, node in frozen_visuals:
        object_id = node.get("id")
        asset_ref = component_assets.get(object_id)
        if not isinstance(asset_ref, dict):
            raise ValueError(f"accepted component asset is missing: {object_id}")
        mask_ref = _mask_ref(node)
        evidence_refs = [mask_ref] if mask_ref is not None else [asset_ref]
        candidates = [
            {
                "kind": "raster_component",
                "confidence": 1.0,
                "payload": {"asset_ref": deepcopy(asset_ref)},
                "evidence_refs": deepcopy(evidence_refs),
                "required_qa_checks": ["ownership", "render_difference"],
            }
        ]
        candidates.extend(deepcopy(shape_candidates.get(object_id, [])))
        item = {
            "id": object_id,
            "bbox": deepcopy(node.get("bbox")),
            "z_index": node.get("z_index"),
            "source_refs": [deepcopy(asset_ref)],
            "mask_ref": deepcopy(mask_ref),
            "relations": [],
            "candidate_representations": candidates,
        }
        parent_id = node.get("parent_id")
        if isinstance(parent_id, str) and parent_id:
            _add_relation(item, "parent", parent_id)
        for text_id in node.get("text_ids", []):
            if isinstance(text_id, str) and text_id in frozen_text:
                _add_relation(item, "text_owner", text_id)
        objects.append((index, item))

    for text_id, (index, node) in frozen_text.items():
        text = text_by_id.get(text_id)
        if not isinstance(text, dict):
            continue
        source_refs = deepcopy(text.get("source_refs", []))
        evidence_refs = deepcopy(text.get("evidence_refs", source_refs))
        mask_ref = _mask_ref(node)
        if mask_ref is not None and mask_ref not in evidence_refs:
            evidence_refs.append(deepcopy(mask_ref))
        item = {
            "id": text_id,
            "bbox": deepcopy(node.get("bbox", text.get("box"))),
            "z_index": node.get("z_index"),
            "source_refs": source_refs,
            "mask_ref": deepcopy(mask_ref),
            "relations": [],
            "candidate_representations": [
                {
                    "kind": "editable_text",
                    "confidence": 1.0,
                    "payload": {
                        "text": text.get("text"),
                        "box": deepcopy(text.get("box")),
                    },
                    "evidence_refs": evidence_refs,
                    "required_qa_checks": [
                        "editable_text",
                        "render_difference",
                    ],
                }
            ],
        }
        parent_id = node.get("parent_id")
        if isinstance(parent_id, str) and parent_id:
            _add_relation(item, "parent", parent_id)
        objects.append((index, item))

    ordered = [
        item
        for _, item in sorted(
            objects, key=lambda pair: (pair[1]["z_index"], pair[0])
        )
    ]
    _add_bbox_relations(ordered)
    return validate_reconstruction_ir(
        {
            "schema_version": 1,
            "page_id": page_id,
            "canvas": {"width": canvas[0], "height": canvas[1]},
            "objects": ordered,
        }
    )
