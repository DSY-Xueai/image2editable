from __future__ import annotations

import pytest

from image2editable.component_contracts import validate_component_graph


def _node(node_id: str, kind: str, state: str, parent_id=None):
    return {
        "id": node_id,
        "kind": kind,
        "state": state,
        "parent_id": parent_id,
        "mask": f"masks/{node_id}.png",
        "mask_sha256": ("a" if node_id == "p" else "b") * 64,
        "z_index": 1 if node_id == "p" else 2,
        "text_ids": [],
        "bbox": [0, 0, 1, 1],
    }


def test_task10_parent_and_child_active_are_rejected():
    with pytest.raises(ValueError, match="parent and child cannot render together"):
        validate_component_graph(
            {
                "nodes": [
                    _node("p", "parent", "pending"),
                    _node("c", "child", "pending", "p"),
                ]
            }
        )


def test_task10_inactive_parent_allows_active_child():
    graph = validate_component_graph(
        {
            "nodes": [
                _node("p", "parent", "inactive"),
                _node("c", "child", "pending", "p"),
            ]
        }
    )
    assert graph["nodes"][1]["state"] == "pending"
