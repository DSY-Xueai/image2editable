from __future__ import annotations

import pytest

from image2editable import component_contracts


AGENT_PROVIDERS = component_contracts.AGENT_PROVIDERS
MAX_REPAIR_ROUNDS = component_contracts.MAX_REPAIR_ROUNDS
validate_agent_provider = component_contracts.validate_agent_provider


def test_component_agent_provider_contract_is_frozen() -> None:
    assert AGENT_PROVIDERS == frozenset({"host", "local"})
    assert MAX_REPAIR_ROUNDS == 5


@pytest.mark.parametrize("value", ["host", "local"])
def test_validate_agent_provider_accepts_supported_lowercase_values(value: str) -> None:
    assert validate_agent_provider(value) == value


@pytest.mark.parametrize("value", ["", "HOST", "remote", None])
def test_validate_agent_provider_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(ValueError, match="agent_provider"):
        validate_agent_provider(value)


def _node(
    component_id: str,
    *,
    kind: str,
    parent_id: str | None,
    state: str,
    mask_sha256: str = "a" * 64,
    text_ids: list[str] | None = None,
) -> dict:
    return {
        "id": component_id,
        "kind": kind,
        "parent_id": parent_id,
        "state": state,
        "mask": f"masks/{component_id}.png",
        "mask_sha256": mask_sha256,
        "bbox": [1, 2, 9, 10],
        "z_index": 0,
        "text_ids": [] if text_ids is None else text_ids,
    }


def test_component_graph_contract_is_category_independent_and_strict() -> None:
    assert component_contracts.COMPONENT_STATES == frozenset(
        {"pending", "failed", "frozen", "inactive"}
    )
    assert component_contracts.COMPONENT_KINDS == frozenset(
        {"parent", "child", "text"}
    )
    node = _node(
        "person_outline",
        kind="child",
        parent_id="parent_0001",
        state="pending",
    )
    node["category"] = "person"

    with pytest.raises(ValueError, match="fields"):
        component_contracts.validate_component_graph(
            {
                "nodes": [
                    _node(
                        "parent_0001",
                        kind="parent",
                        parent_id=None,
                        state="inactive",
                    ),
                    node,
                ]
            }
        )


def test_parent_and_child_cannot_render_together() -> None:
    graph = {
        "nodes": [
            _node(
                "parent_0001",
                kind="parent",
                parent_id=None,
                state="pending",
            ),
            _node(
                "component_0001",
                kind="child",
                parent_id="parent_0001",
                state="pending",
            ),
        ]
    }

    with pytest.raises(ValueError, match="parent and child"):
        component_contracts.validate_component_graph(graph)


def test_active_child_requires_metadata_only_parent() -> None:
    graph = {
        "nodes": [
            _node(
                "parent_0001",
                kind="parent",
                parent_id=None,
                state="inactive",
            ),
            _node(
                "component_0001",
                kind="child",
                parent_id="parent_0001",
                state="pending",
            ),
        ]
    }

    assert component_contracts.validate_component_graph(graph) is graph


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mask", "masks/replaced.png"),
        ("mask_sha256", "b" * 64),
        ("bbox", [1, 2, 8, 10]),
        ("z_index", 1),
        ("parent_id", "parent_0002"),
        ("text_ids", ["text_0002"]),
        ("state", "inactive"),
        ("kind", "text"),
    ],
)
def test_frozen_component_cannot_change(field: str, value: object) -> None:
    parent = _node(
        "parent_0001",
        kind="parent",
        parent_id=None,
        state="inactive",
    )
    frozen = _node(
        "component_0001",
        kind="child",
        parent_id="parent_0001",
        state="frozen",
        text_ids=["text_0001"],
    )
    text = _node(
        "text_0001",
        kind="text",
        parent_id=None,
        state="inactive",
    )
    before = {"nodes": [parent, frozen, text]}
    mutated = [dict(node) for node in before["nodes"]]
    mutated[1][field] = value

    with pytest.raises(ValueError, match="frozen"):
        component_contracts.validate_graph_transition(
            before=before,
            after={"nodes": mutated},
        )


def test_frozen_component_cannot_be_removed() -> None:
    before = {
        "nodes": [
            _node(
                "component_0001",
                kind="parent",
                parent_id=None,
                state="frozen",
            )
        ]
    }

    with pytest.raises(ValueError, match="frozen"):
        component_contracts.validate_graph_transition(
            before=before,
            after={"nodes": []},
        )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("pending", True),
        ("frozen", True),
        ("failed", False),
        ("inactive", False),
    ],
)
def test_render_activity_uses_state_not_content_category(
    state: str,
    expected: bool,
) -> None:
    node = _node(
        "dense_research_drawing",
        kind="child",
        parent_id="parent_0001",
        state=state,
    )

    assert component_contracts.is_render_active_component(node) is expected


@pytest.mark.parametrize(("field", "value"), [("kind", []), ("state", {})])
def test_component_node_rejects_non_scalar_enum_values(
    field: str,
    value: object,
) -> None:
    node = _node(
        "component_0001",
        kind="parent",
        parent_id=None,
        state="pending",
    )
    node[field] = value

    with pytest.raises(ValueError, match=f"component {field}"):
        component_contracts.validate_component_graph({"nodes": [node]})


def test_component_mask_path_must_be_relative_to_graph_directory() -> None:
    node = _node(
        "component_0001",
        kind="parent",
        parent_id=None,
        state="pending",
    )
    node["mask"] = "C:/outside/component.png"

    with pytest.raises(ValueError, match="mask path"):
        component_contracts.validate_component_graph({"nodes": [node]})
