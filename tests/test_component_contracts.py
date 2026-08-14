from __future__ import annotations

import pytest

from image2editable import component_contracts


AGENT_PROVIDERS = component_contracts.AGENT_PROVIDERS
MAX_REPAIR_ROUNDS = component_contracts.MAX_REPAIR_ROUNDS
validate_agent_provider = component_contracts.validate_agent_provider


def test_component_agent_provider_contract_is_frozen() -> None:
    assert AGENT_PROVIDERS == frozenset({"host", "local"})
    assert MAX_REPAIR_ROUNDS == 5
    assert "pending_gate" in component_contracts.COMPONENT_STATES
    assert "component-isolation.png" in component_contracts.COMPONENT_EVIDENCE_NAMES
    assert "presentation-manifest.json" in component_contracts.COMPONENT_EVIDENCE_NAMES
    assert "unexplained-mask.png" in component_contracts.COMPONENT_EVIDENCE_NAMES


def test_quality_input_contract_requires_presentation_manifest_ref() -> None:
    refs = {
        name: {"path": f"quality/{name}", "sha256": "a" * 64}
        for name in (
            "background", "reconstructed", "text_mask", "native_check",
            "presentation_manifest",
        )
    }

    assert component_contracts._validate_quality_input_refs(refs) is refs
    refs["foreground_evidence"] = {
        "path": "quality/foreground_evidence",
        "sha256": "b" * 64,
    }
    assert component_contracts._validate_quality_input_refs(refs) is refs
    refs.pop("foreground_evidence")
    refs.pop("presentation_manifest")
    with pytest.raises(ValueError, match="quality input refs"):
        component_contracts._validate_quality_input_refs(refs)


@pytest.mark.parametrize(
    ("action", "object_ids", "parameters"),
    [
        ("accept", ["component_0001"], {}),
        ("discard", ["component_0001"], {}),
        ("rebuild_background", ["component_0001"], {"margin_ratio": 0.01}),
        ("absorb_residual", ["component_0001"], {}),
        ("absorb_into_parent", ["parent_0001", "component_0001"], {}),
        ("merge", ["component_0001", "component_0002"], {}),
        ("split", ["component_0001"], {"parts": 2}),
        ("expand", ["component_0001"], {"margin_ratio": 0.01}),
        ("shrink", ["component_0001"], {"margin_ratio": 0.01}),
        ("retry_with_box", ["component_0001"], {"box": [0.1, 0.1, 0.5, 0.5]}),
        ("retry_with_points", ["component_0001"], {"positive": [[0.2, 0.2]], "negative": []}),
        ("attach_text", ["component_0001", "text_0001"], {}),
        ("suppress_text", ["text_0001"], {}),
        ("collapse_to_parent", ["parent_0001"], {}),
    ],
)
def test_component_actions_have_strict_shapes(action: str, object_ids: list[str], parameters: dict) -> None:
    value = {"action": action, "object_ids": object_ids, "parameters": parameters,
             "confidence": 0.95, "evidence": ["visible relationship"]}
    assert component_contracts.validate_component_action(value) is value


def test_component_action_rejects_non_string_object_id_with_value_error() -> None:
    value = {"action": "accept", "object_ids": [{}], "parameters": {},
             "confidence": 0.95, "evidence": ["visible relationship"]}
    with pytest.raises(ValueError, match="object_ids"):
        component_contracts.validate_component_action(value)


@pytest.mark.parametrize(
    "graph",
    [
        {"nodes": [{"state": "frozen"}]},
        {"nodes": [{"id": [], "state": "pending"}]},
    ],
)
def test_component_action_rejects_malformed_graph_with_value_error(graph: dict) -> None:
    value = {"action": "accept", "object_ids": ["component_0001"], "parameters": {},
             "confidence": 0.95, "evidence": ["visible relationship"]}
    with pytest.raises(ValueError):
        component_contracts.validate_component_action(value, graph=graph)


def test_component_agent_request_contract_is_strict() -> None:
    request = {
        "schema_version": 1,
        "page_id": "page_001",
        "provider": "host",
        "repair_round": 1,
        "source_sha256": "a" * 64,
        "graph_sha256": "b" * 64,
        "candidate_ids": ["component_0001"],
        "frozen_ids": ["component_0002"],
        "evidence": {
            name: {"path": name, "sha256": "c" * 64}
            for name in component_contracts.COMPONENT_EVIDENCE_NAMES
        },
        "review_evidence": [
            *component_contracts.FULL_COMPONENT_REVIEW_EVIDENCE,
        ],
    }

    assert component_contracts.validate_component_agent_request(request) is request
    request["extra"] = None
    with pytest.raises(ValueError, match="request fields"):
        component_contracts.validate_component_agent_request(request)


def test_component_agent_request_rejects_mislabeled_evidence_path() -> None:
    request = {
        "schema_version": 1,
        "page_id": "page_001",
        "provider": "host",
        "repair_round": 1,
        "source_sha256": "a" * 64,
        "graph_sha256": "b" * 64,
        "candidate_ids": [],
        "frozen_ids": [],
        "evidence": {
            name: {"path": name, "sha256": "c" * 64}
            for name in component_contracts.COMPONENT_EVIDENCE_NAMES
        },
        "review_evidence": [
            *component_contracts.FULL_COMPONENT_REVIEW_EVIDENCE,
        ],
    }
    request["evidence"]["source.png"]["path"] = "ownership.png"

    with pytest.raises(ValueError, match="evidence path"):
        component_contracts.validate_component_agent_request(request)


def test_component_agent_request_rejects_boolean_schema_version() -> None:
    request = {
        "schema_version": True,
        "page_id": "page_001",
        "provider": "host",
        "repair_round": 1,
        "source_sha256": "a" * 64,
        "graph_sha256": "b" * 64,
        "candidate_ids": [],
        "frozen_ids": [],
        "evidence": {
            name: {"path": name, "sha256": "c" * 64}
            for name in component_contracts.COMPONENT_EVIDENCE_NAMES
        },
        "review_evidence": [
            *component_contracts.FULL_COMPONENT_REVIEW_EVIDENCE,
        ],
    }

    with pytest.raises(ValueError, match="schema_version"):
        component_contracts.validate_component_agent_request(request)


@pytest.mark.parametrize(
    "review_evidence",
    [
        ["source.png", "source.png"],
        ["source.png", "unknown.png"],
        ["reconstructed.png", "source.png"],
    ],
)
def test_component_agent_request_rejects_invalid_review_evidence(
    review_evidence: list[str],
) -> None:
    request = {
        "schema_version": 1,
        "page_id": "page_001",
        "provider": "host",
        "repair_round": 1,
        "source_sha256": "a" * 64,
        "graph_sha256": "b" * 64,
        "candidate_ids": [],
        "frozen_ids": [],
        "evidence": {
            name: {"path": name, "sha256": "c" * 64}
            for name in component_contracts.COMPONENT_EVIDENCE_NAMES
        },
        "review_evidence": review_evidence,
    }

    with pytest.raises(ValueError, match="review_evidence"):
        component_contracts.validate_component_agent_request(request)


@pytest.mark.parametrize("value", ["host", "local"])
def test_validate_agent_provider_accepts_supported_lowercase_values(value: str) -> None:
    assert validate_agent_provider(value) == value


@pytest.mark.parametrize("value", ["", "HOST", "remote", None])
def test_validate_agent_provider_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(ValueError, match="agent_provider"):
        validate_agent_provider(value)


@pytest.mark.parametrize(
    ("action", "object_ids"),
    [
        ("accept", ["visual", "visual2"]),
        ("merge", ["visual"]),
        ("split", ["visual", "visual2"]),
        ("attach_text", ["visual"]),
        ("suppress_text", ["text", "visual"]),
        ("collapse_to_parent", ["visual", "visual2"]),
    ],
)
def test_component_plan_rejects_invalid_action_object_count(action: str, object_ids: list[str]) -> None:
    request, graph = _plan_contract_fixture()
    plan = _plan(request, action, object_ids)
    with pytest.raises(ValueError, match="object count"):
        component_contracts.validate_component_plan(plan, request=request, graph=graph)


@pytest.mark.parametrize(
    ("action", "object_ids"),
    [
        ("attach_text", ["text", "visual"]),
        ("suppress_text", ["visual"]),
        ("collapse_to_parent", ["child_a",]),
        ("accept", ["text"]),
        ("merge", ["child_a", "child_b"]),
        ("merge", ["visual", "child_a"]),
    ],
)
def test_component_plan_rejects_graph_role_and_parent_mismatch(action: str, object_ids: list[str]) -> None:
    request, graph = _plan_contract_fixture()
    with pytest.raises(ValueError, match="kind|role|parent|text|frozen"):
        component_contracts.validate_component_plan(_plan(request, action, object_ids), request=request, graph=graph)


def test_component_plan_rejects_attach_text_to_pending_text() -> None:
    request, graph = _plan_contract_fixture()
    request["candidate_ids"].append("text")
    request["candidate_ids"].sort()
    request["frozen_ids"] = []
    next(node for node in graph["nodes"] if node["id"] == "text")["state"] = "pending"

    with pytest.raises(ValueError, match="frozen"):
        component_contracts.validate_component_plan(
            _plan(request, "attach_text", ["visual", "text"]),
            request=request,
            graph=graph,
        )


def test_component_plan_allows_suppressing_linked_frozen_text() -> None:
    request, graph = _plan_contract_fixture()
    plan = _plan(request, "suppress_text", ["text"])

    assert component_contracts.validate_component_plan(
        plan, request=request, graph=graph,
    ) is plan

    next(node for node in graph["nodes"] if node["id"] == "visual")[
        "text_ids"
    ] = ["text"]
    assert component_contracts.validate_component_plan(
        plan, request=request, graph=graph,
    ) is plan


def test_component_plan_allows_collapse_to_parent_of_requested_child() -> None:
    request, graph = _plan_contract_fixture()
    request["candidate_ids"].remove("parent_a")

    plan = _plan(request, "collapse_to_parent", ["parent_a"])

    assert component_contracts.validate_component_plan(
        plan, request=request, graph=graph,
    ) is plan


def test_component_plan_allows_absorb_into_authenticated_inactive_parent() -> None:
    request, graph = _plan_contract_fixture()
    request["candidate_ids"].remove("parent_b")
    request["candidate_ids"].remove("child_b")
    next(node for node in graph["nodes"] if node["id"] == "parent_b")["state"] = "inactive"
    plan = _plan(request, "absorb_into_parent", ["parent_b", "visual"])

    assert component_contracts.validate_component_plan(
        plan, request=request, graph=graph,
    ) is plan


def test_component_plan_allows_retrying_authenticated_inactive_visual() -> None:
    request, graph = _plan_contract_fixture()
    request["candidate_ids"].remove("visual")
    next(node for node in graph["nodes"] if node["id"] == "visual")[
        "state"
    ] = "inactive"
    plan = _plan(request, "retry_with_box", ["visual"])
    plan["actions"][0]["parameters"] = {"box": [0.1, 0.1, 0.5, 0.5]}

    assert component_contracts.validate_component_plan(
        plan, request=request, graph=graph,
    ) is plan

    plan["actions"][0]["action"] = "accept"
    plan["actions"][0]["parameters"] = {}
    with pytest.raises(ValueError, match="object_ids"):
        component_contracts.validate_component_plan(
            plan, request=request, graph=graph,
        )


def test_component_plan_rebuilds_background_with_retried_inactive_visual() -> None:
    request, graph = _plan_contract_fixture()
    request["candidate_ids"].remove("visual")
    next(node for node in graph["nodes"] if node["id"] == "visual")[
        "state"
    ] = "inactive"
    plan = _plan(request, "retry_with_box", ["visual"])
    plan["actions"][0]["parameters"] = {"box": [0.1, 0.1, 0.5, 0.5]}
    plan["actions"].append({
        "action": "rebuild_background",
        "object_ids": ["visual"],
        "parameters": {"margin_ratio": 0.01},
        "confidence": 0.9,
        "evidence": ["remove the retried visual from the rebuilt background"],
    })

    assert component_contracts.validate_component_plan(
        plan, request=request, graph=graph,
    ) is plan


def test_component_plan_rejects_inactive_secondary_parent_absorption() -> None:
    request, graph = _plan_contract_fixture()
    for parent_id in ("parent_a", "parent_b"):
        request["candidate_ids"].remove(parent_id)
        next(node for node in graph["nodes"] if node["id"] == parent_id)[
            "state"
        ] = "inactive"
    request["candidate_ids"].remove("child_b")
    plan = _plan(
        request,
        "absorb_into_parent",
        ["parent_b", "parent_a", "visual"],
    )

    with pytest.raises(ValueError, match="pending"):
        component_contracts.validate_component_plan(
            plan, request=request, graph=graph,
        )


def _plan_contract_fixture() -> tuple[dict, dict]:
    ids = ["visual", "visual2", "text", "parent_a", "parent_b", "child_a", "child_b"]
    request = {"schema_version": 1, "page_id": "page_001", "provider": "host",
               "repair_round": 1, "source_sha256": "a" * 64, "graph_sha256": "b" * 64,
               "candidate_ids": sorted(set(ids) - {"text"}), "frozen_ids": ["text"],
               "evidence": {name: {"path": name, "sha256": "c" * 64}
                            for name in component_contracts.COMPONENT_EVIDENCE_NAMES},
               "review_evidence": list(
                   component_contracts.FULL_COMPONENT_REVIEW_EVIDENCE
               )}
    def node(value: str, kind: str, parent: str | None, z: int) -> dict:
        return {"id": value, "kind": kind, "parent_id": parent, "state": "pending",
                "mask": f"masks/{value}.png", "mask_sha256": "d" * 64,
                "bbox": [0, 0, 2, 2], "z_index": z, "text_ids": []}
    text = node("text", "text", None, 2)
    text["state"] = "frozen"
    graph = {"nodes": [node("visual", "parent", None, 0), node("visual2", "parent", None, 1),
                       text, node("parent_a", "parent", None, 3),
                       node("parent_b", "parent", None, 4), node("child_a", "child", "parent_a", 5),
                       node("child_b", "child", "parent_b", 6)]}
    return request, graph


def _plan(request: dict, action: str, object_ids: list[str]) -> dict:
    parameters = {"split": {"parts": 2}}.get(action, {})
    return {"schema_version": 1, "kind": "component_plan", "page_id": request["page_id"],
            "provider": "host", "repair_round": 1, "request_sha256": "e" * 64,
            "actions": [{"action": action, "object_ids": object_ids,
                         "parameters": parameters, "confidence": 0.9,
                         "evidence": ["visible"]}]}


def test_component_plan_allows_accept_then_absorb_residual_for_same_object() -> None:
    request, graph = _plan_contract_fixture()
    plan = _plan(request, "accept", ["visual"])
    plan["actions"].append({
        "action": "absorb_residual",
        "object_ids": ["visual"],
        "parameters": {},
        "confidence": 0.9,
        "evidence": ["bind the unexplained residual"],
    })

    assert component_contracts.validate_component_plan(
        plan, request=request, graph=graph,
    ) is plan


@pytest.mark.parametrize(
    "action_names",
    [
        ["absorb_residual", "accept"],
        ["accept", "accept"],
        ["absorb_residual", "absorb_residual"],
        ["accept", "absorb_residual", "absorb_residual"],
        ["accept", "expand"],
    ],
)
def test_component_plan_rejects_other_repeated_object_action_sequences(
    action_names: list[str],
) -> None:
    request, graph = _plan_contract_fixture()
    plan = _plan(request, action_names[0], ["visual"])
    plan["actions"] = [
        {
            "action": action_name,
            "object_ids": ["visual"],
            "parameters": {"margin_ratio": 0.01} if action_name == "expand" else {},
            "confidence": 0.9,
            "evidence": ["visible"],
        }
        for action_name in action_names
    ]

    with pytest.raises(ValueError, match="conflicting object actions"):
        component_contracts.validate_component_plan(
            plan, request=request, graph=graph,
        )


def test_component_plan_accepts_multiple_background_rebuilds() -> None:
    request, graph = _plan_contract_fixture()
    plan = _plan(request, "rebuild_background", ["visual"])
    plan["actions"][0]["parameters"] = {"margin_ratio": 0.01}
    plan["actions"].append({
        **plan["actions"][0],
        "object_ids": ["visual2"],
        "parameters": {"margin_ratio": 0.02},
    })

    assert component_contracts.validate_component_plan(
        plan, request=request, graph=graph,
    ) is plan


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
        {"pending", "pending_gate", "failed", "frozen", "inactive"}
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
        state="frozen",
    )
    before = {"nodes": [parent, frozen, text]}
    mutated = [dict(node) for node in before["nodes"]]
    mutated[1][field] = value

    with pytest.raises(ValueError, match="frozen"):
        component_contracts.validate_graph_transition(
            before=before,
            after={"nodes": mutated},
        )


def test_frozen_text_suppression_requires_explicit_transition_authorization() -> None:
    text = _node(
        "text_0001", kind="text", parent_id=None, state="frozen",
    )
    before = {"nodes": [text]}
    suppressed = {"nodes": [{**text, "state": "inactive"}]}

    with pytest.raises(ValueError, match="frozen"):
        component_contracts.validate_graph_transition(
            before=before, after=suppressed,
        )

    assert component_contracts.validate_graph_transition(
        before=before,
        after=suppressed,
        allowed_suppressed_text_ids={"text_0001"},
    ) is suppressed


def test_inactive_visual_reactivation_requires_exact_authorization() -> None:
    visual = _node(
        "visual_0001", kind="parent", parent_id=None, state="inactive",
    )
    before = {"nodes": [visual]}
    reactivated = {"nodes": [{**visual, "state": "pending"}]}

    with pytest.raises(ValueError, match="inactive"):
        component_contracts.validate_graph_transition(
            before=before,
            after=reactivated,
        )

    assert component_contracts.validate_graph_transition(
        before=before,
        after=reactivated,
        allowed_reactivated_ids={"visual_0001"},
    ) is reactivated

    with pytest.raises(ValueError, match="reactivation"):
        component_contracts.validate_graph_transition(
            before=before,
            after=reactivated,
            allowed_reactivated_ids={"other"},
        )


def test_frozen_visual_repair_reactivation_requires_exact_authorization() -> None:
    visual = _node(
        "visual_0001", kind="parent", parent_id=None, state="frozen",
    )
    before = {"nodes": [visual]}
    reactivated = {"nodes": [{**visual, "state": "pending"}]}

    with pytest.raises(ValueError, match="frozen"):
        component_contracts.validate_graph_transition(
            before=before, after=reactivated,
        )

    assert component_contracts.validate_graph_transition(
        before=before,
        after=reactivated,
        allowed_reactivated_ids={"visual_0001"},
    ) is reactivated


def test_authorized_text_suppression_cannot_change_any_other_frozen_field() -> None:
    text = _node(
        "text_0001", kind="text", parent_id=None, state="frozen",
    )
    mutated = {"nodes": [{
        **text,
        "state": "inactive",
        "bbox": [1, 2, 8, 10],
    }]}

    with pytest.raises(ValueError, match="frozen"):
        component_contracts.validate_graph_transition(
            before={"nodes": [text]},
            after=mutated,
            allowed_suppressed_text_ids={"text_0001"},
        )


def test_authorized_text_suppression_may_only_unlink_it_from_frozen_visuals() -> None:
    text = _node(
        "text_0001", kind="text", parent_id=None, state="frozen",
    )
    visual = _node(
        "visual_0001", kind="parent", parent_id=None, state="frozen",
        text_ids=["text_0001"],
    )
    after = {"nodes": [
        {**text, "state": "inactive"},
        {**visual, "text_ids": []},
    ]}

    assert component_contracts.validate_graph_transition(
        before={"nodes": [text, visual]},
        after=after,
        allowed_suppressed_text_ids={"text_0001"},
    ) is after

    after["nodes"][1]["bbox"] = [1, 2, 8, 10]
    with pytest.raises(ValueError, match="frozen"):
        component_contracts.validate_graph_transition(
            before={"nodes": [text, visual]},
            after=after,
            allowed_suppressed_text_ids={"text_0001"},
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


def test_active_component_z_indexes_must_be_unique() -> None:
    first = _node(
        "component_0001",
        kind="parent",
        parent_id=None,
        state="pending",
    )
    second = _node(
        "component_0002",
        kind="parent",
        parent_id=None,
        state="frozen",
    )

    with pytest.raises(ValueError, match="z_index"):
        component_contracts.validate_component_graph({"nodes": [first, second]})


def test_frozen_visual_component_requires_frozen_linked_text() -> None:
    visual = _node(
        "component_0001",
        kind="parent",
        parent_id=None,
        state="frozen",
        text_ids=["text_0001"],
    )
    text = _node(
        "text_0001",
        kind="text",
        parent_id=None,
        state="pending",
    )

    with pytest.raises(ValueError, match="frozen.*text"):
        component_contracts.validate_component_graph({"nodes": [visual, text]})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mask", "masks/replaced-text.png"),
        ("mask_sha256", "b" * 64),
        ("bbox", [1, 2, 8, 10]),
        ("z_index", 1),
        ("state", "inactive"),
    ],
)
def test_text_linked_to_frozen_component_cannot_change(
    field: str,
    value: object,
) -> None:
    visual = _node(
        "component_0001",
        kind="parent",
        parent_id=None,
        state="frozen",
        text_ids=["text_0001"],
    )
    text = _node(
        "text_0001",
        kind="text",
        parent_id=None,
        state="frozen",
    )
    before = {"nodes": [visual, text]}
    replacement = dict(text)
    replacement[field] = value

    with pytest.raises(ValueError, match="frozen"):
        component_contracts.validate_graph_transition(
            before=before,
            after={"nodes": [visual, replacement]},
        )


def test_text_linked_to_frozen_component_cannot_be_deleted() -> None:
    visual = _node(
        "component_0001",
        kind="parent",
        parent_id=None,
        state="frozen",
        text_ids=["text_0001"],
    )
    text = _node(
        "text_0001",
        kind="text",
        parent_id=None,
        state="frozen",
    )

    with pytest.raises(ValueError, match="frozen"):
        component_contracts.validate_graph_transition(
            before={"nodes": [visual, text]},
            after={"nodes": [visual]},
        )


def test_frozen_text_cannot_have_two_active_visual_owners() -> None:
    first = _node(
        "component_0001",
        kind="parent",
        parent_id=None,
        state="frozen",
        text_ids=["text_0001"],
    )
    second = _node(
        "component_0002",
        kind="parent",
        parent_id=None,
        state="frozen",
        text_ids=["text_0001"],
    )
    second["z_index"] = 1
    text = _node(
        "text_0001",
        kind="text",
        parent_id=None,
        state="frozen",
    )

    with pytest.raises(ValueError, match="multiple active owners"):
        component_contracts.validate_component_graph(
            {"nodes": [first, second, text]}
        )
