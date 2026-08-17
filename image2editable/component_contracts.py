from __future__ import annotations

from pathlib import PurePosixPath
import math


AGENT_PROVIDERS = frozenset({"host", "local", "local-service"})
MAX_REPAIR_ROUNDS = 5
MAX_COMPONENT_PROMPT_POINTS = 256
COMPONENT_STATES = frozenset(
    {"pending", "pending_gate", "failed", "frozen", "inactive"}
)
COMPONENT_KINDS = frozenset({"parent", "child", "text"})
LEGACY_COMPONENT_EVIDENCE_NAMES = frozenset(
    {
        "source.png",
        "numbered-masks.png",
        "ocr-overlay.png",
        "component-isolation.png",
        "ownership.png",
        "reconstructed.png",
        "difference.png",
        "component-graph.json",
        "quality-report.json",
        "presentation-manifest.json",
    }
)
COMPONENT_EVIDENCE_NAMES = LEGACY_COMPONENT_EVIDENCE_NAMES | {
    "unexplained-mask.png"
}
ROUND_REVIEW_EVIDENCE_NAME = "round-review.png"
FULL_COMPONENT_REVIEW_EVIDENCE = (
    "source.png",
    "numbered-masks.png",
    "ocr-overlay.png",
    "component-isolation.png",
    "ownership.png",
    "reconstructed.png",
    "difference.png",
    "unexplained-mask.png",
    "quality-report.json",
)
INCREMENTAL_COMPONENT_REVIEW_EVIDENCE = (
    "source.png",
    "reconstructed.png",
    "difference.png",
    "unexplained-mask.png",
    "quality-report.json",
    ROUND_REVIEW_EVIDENCE_NAME,
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
        "review_evidence",
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
_RENDER_STATES = frozenset({"pending", "pending_gate", "frozen"})
COMPONENT_REPAIR_PHASES = frozenset({
    "request_published", "awaiting_plan", "plan_recorded", "actions_executed",
    "quality_recorded", "freeze_committed", "fallback_required",
    "fallback_executed", "fallback_quality_recorded", "ready_for_assembly",
    "preserved_with_warning",
})


def validate_component_repair_state(state: object) -> dict:
    fields = {
        "schema_version", "page_id", "provider", "source_sha256",
        "initial_component_count", "quality_gate_version", "revision", "phase",
        "status", "repair_round", "plan_count", "stop_reason", "graph_ref",
        "current_round", "frozen", "candidate_ids", "failed_ids", "fallback",
        "last_normalized_plan_sha256", "result_ref", "delivery_checks", "updated_at",
        "round_history", "parent_assets", "fallback_graph_ref", "fallback_quality_ref",
        "fallback_input_refs",
    }
    if not isinstance(state, dict) or set(state) != fields:
        raise ValueError("component repair state fields are invalid")
    if state["schema_version"] != 1 or type(state["schema_version"]) is not int:
        raise ValueError("component repair state schema_version is invalid")
    page_id = state["page_id"]
    if type(page_id) is not str or not page_id or "/" in page_id or "\\" in page_id:
        raise ValueError("component repair state page_id is invalid")
    validate_agent_provider(state["provider"])
    _validate_sha256(state["source_sha256"], "source_sha256")
    if type(state["initial_component_count"]) is not int or state["initial_component_count"] < 0:
        raise ValueError("component repair initial count is invalid")
    for name in ("quality_gate_version", "revision"):
        if type(state[name]) is not int or state[name] < 1:
            raise ValueError(f"component repair {name} is invalid")
    if state["phase"] not in COMPONENT_REPAIR_PHASES:
        raise ValueError("component repair phase is invalid")
    if state["status"] not in {"active", "ready_for_assembly", "preserved_with_warning"}:
        raise ValueError("component repair status is invalid")
    validate_repair_round(state["repair_round"])
    if type(state["plan_count"]) is not int or not 0 <= state["plan_count"] <= MAX_REPAIR_ROUNDS:
        raise ValueError("component repair plan_count is invalid")
    if state["stop_reason"] not in {
        None, "empty_plan", "repeated_plan", "no_executable_actions",
        "round_limit", "no_quality_improvement", "unowned_raster_text",
        "page_quality_failed",
    }:
        raise ValueError("component repair stop_reason is invalid")
    _validate_artifact_ref(state["graph_ref"], "graph_ref")
    current = state["current_round"]
    if not isinstance(current, dict) or set(current) != {
        "round", "request_ref", "plan_ref", "execution_ref", "quality_ref"
    }:
        raise ValueError("component repair current_round is invalid")
    if current["round"] != state["repair_round"]:
        raise ValueError("component repair current round is inconsistent")
    _validate_artifact_ref(current["request_ref"], "request_ref")
    for name in ("plan_ref", "execution_ref", "quality_ref"):
        if current[name] is not None:
            _validate_artifact_ref(current[name], name)
    for name in ("candidate_ids", "failed_ids"):
        values = state[name]
        if not isinstance(values, list) or values != sorted(set(values)) or any(type(value) is not str or not value for value in values):
            raise ValueError(f"component repair {name} is invalid")
    frozen = state["frozen"]
    if not isinstance(frozen, dict) or any(type(key) is not str for key in frozen):
        raise ValueError("component repair frozen map is invalid")
    for digest in frozen.values():
        _validate_sha256(digest, "frozen mask sha256")
    if set(state["candidate_ids"]) & set(frozen):
        raise ValueError("component repair candidate cannot be frozen")
    parent_assets = state["parent_assets"]
    if not isinstance(parent_assets, dict) or any(type(key) is not str for key in parent_assets):
        raise ValueError("component repair parent assets are invalid")
    for reference in parent_assets.values():
        _validate_artifact_ref(reference, "parent asset ref")
    history = state["round_history"]
    if not isinstance(history, list) or len(history) > MAX_REPAIR_ROUNDS:
        raise ValueError("component repair round history is invalid")
    for entry in history:
        if not isinstance(entry, dict) or set(entry) != {
            "round", "plan_sha256", "normalized_plan_sha256", "execution_sha256",
            "quality_sha256", "frozen_ids", "failed_ids",
        }:
            raise ValueError("component repair round history entry is invalid")
        validate_repair_round(entry["round"])
        for name in ("plan_sha256", "normalized_plan_sha256", "execution_sha256", "quality_sha256"):
            if entry[name] is not None:
                _validate_sha256(entry[name], f"component repair history {name}")
        for name in ("frozen_ids", "failed_ids"):
            if not isinstance(entry[name], list) or entry[name] != sorted(set(entry[name])):
                raise ValueError("component repair history component ids are invalid")
    fallback = state["fallback"]
    if not isinstance(fallback, dict) or set(fallback) != {"status", "parent_ids"}:
        raise ValueError("component repair fallback is invalid")
    if fallback["status"] not in {"none", "required", "parent_pending", "parent_preserved", "warning"}:
        raise ValueError("component repair fallback status is invalid")
    if not isinstance(fallback["parent_ids"], list) or fallback["parent_ids"] != sorted(set(fallback["parent_ids"])):
        raise ValueError("component repair fallback parents are invalid")
    if state["last_normalized_plan_sha256"] is not None:
        _validate_sha256(state["last_normalized_plan_sha256"], "last plan sha256")
    if state["result_ref"] is not None:
        _validate_artifact_ref(state["result_ref"], "result_ref")
    for name in ("fallback_graph_ref", "fallback_quality_ref"):
        if state[name] is not None:
            _validate_artifact_ref(state[name], name)
    if state["fallback_input_refs"] is not None:
        _validate_quality_input_refs(state["fallback_input_refs"])
    if state["delivery_checks"] != {"pptx_reopen": "unknown"}:
        raise ValueError("component repair delivery checks are invalid")
    if type(state["updated_at"]) is not str or not state["updated_at"]:
        raise ValueError("component repair updated_at is invalid")
    if state["candidate_ids"] != state["failed_ids"]:
        raise ValueError("component repair candidates and failures are inconsistent")
    plan_ref = current["plan_ref"]
    execution_ref = current["execution_ref"]
    quality_ref = current["quality_ref"]
    phase = state["phase"]
    if phase in {"request_published", "awaiting_plan"} and any(
        value is not None for value in (plan_ref, execution_ref, quality_ref)
    ):
        raise ValueError("component repair awaiting phase has premature references")
    if phase == "plan_recorded" and (
        plan_ref is None or execution_ref is not None or quality_ref is not None
    ):
        raise ValueError("component repair plan phase references are invalid")
    if phase == "actions_executed" and (
        plan_ref is None or execution_ref is None or quality_ref is not None
    ):
        raise ValueError("component repair execution phase references are invalid")
    if phase in {"quality_recorded", "freeze_committed"} and any(
        value is None for value in (plan_ref, execution_ref, quality_ref)
    ):
        raise ValueError("component repair quality phase references are invalid")
    terminal = phase in {"ready_for_assembly", "preserved_with_warning"}
    expected_status = phase if terminal else "active"
    if state["status"] != expected_status:
        raise ValueError("component repair status and phase are inconsistent")
    if (phase == "ready_for_assembly") != (state["result_ref"] is not None):
        raise ValueError("component repair result reference is inconsistent")
    if state["plan_count"] > state["repair_round"]:
        raise ValueError("component repair plan count exceeds page rounds")
    history_rounds = [entry["round"] for entry in history]
    if history_rounds != sorted(set(history_rounds)) or any(
        value > state["repair_round"] for value in history_rounds
    ):
        raise ValueError("component repair round history order is invalid")
    if len(history) > state["plan_count"] or state["plan_count"] - len(history) > 1:
        raise ValueError("component repair plan count and history are inconsistent")
    fallback_status = fallback["status"]
    if phase == "fallback_required" and fallback_status != "required":
        raise ValueError("component repair fallback requirement is inconsistent")
    if phase in {"fallback_executed", "fallback_quality_recorded"} and fallback_status != "parent_pending":
        raise ValueError("component repair parent fallback phase is inconsistent")
    if phase == "ready_for_assembly" and fallback_status not in {"none", "parent_preserved"}:
        raise ValueError("component repair ready fallback is inconsistent")
    if phase == "preserved_with_warning" and fallback_status != "warning":
        raise ValueError("component repair warning fallback is inconsistent")
    fallback_phase = phase in {
        "fallback_required", "fallback_executed", "fallback_quality_recorded",
        "preserved_with_warning",
    } or fallback_status == "parent_preserved"
    resumed_progress_override = (
        phase == "freeze_committed"
        and fallback_status == "none"
        and state["stop_reason"] in {
            "round_limit", "no_quality_improvement",
        }
    )
    if (fallback_phase or resumed_progress_override) != (
        state["stop_reason"] is not None
    ):
        raise ValueError("component repair fallback stop reason is inconsistent")
    if phase == "fallback_required" and any(
        state[name] is not None for name in (
            "fallback_graph_ref", "fallback_quality_ref", "fallback_input_refs"
        )
    ):
        raise ValueError("component repair fallback has premature references")
    if phase == "fallback_executed" and (
        state["fallback_graph_ref"] != state["graph_ref"]
        or state["fallback_quality_ref"] is not None
        or state["fallback_input_refs"] is None
    ):
        raise ValueError("component repair fallback execution references are invalid")
    if phase == "fallback_quality_recorded" and (
        state["fallback_graph_ref"] != state["graph_ref"]
        or state["fallback_quality_ref"] is None
        or state["fallback_input_refs"] is None
    ):
        raise ValueError("component repair fallback quality references are invalid")
    if phase == "preserved_with_warning" and state["result_ref"] is not None:
        raise ValueError("component repair warning cannot have a result reference")
    return state


def _validate_quality_input_refs(value: object) -> dict:
    legacy_fields = {
        "background", "reconstructed", "text_mask", "native_check",
        "presentation_manifest",
    }
    if not isinstance(value, dict) or frozenset(value) not in {
        frozenset(legacy_fields),
        frozenset({*legacy_fields, "foreground_evidence"}),
    }:
        raise ValueError("component quality input refs are invalid")
    for reference in value.values():
        _validate_artifact_ref(reference, "quality input ref")
    return value


def _validate_artifact_ref(value: object, field: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"component repair {field} is invalid")
    path = value["path"]
    if type(path) is not str or not path or "\\" in path or ":" in path:
        raise ValueError(f"component repair {field} path is invalid")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"component repair {field} path is invalid")
    _validate_sha256(value["sha256"], f"component repair {field} sha256")
    return value


def validate_agent_provider(value: object) -> str:
    if type(value) is not str or value not in AGENT_PROVIDERS:
        raise ValueError(
            "Invalid agent_provider; expected one of: host, local, local-service"
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
    "discard": frozenset(),
    "merge": frozenset(),
    "split": frozenset({"parts"}),
    "expand": frozenset({"margin_ratio"}),
    "shrink": frozenset({"margin_ratio"}),
    "retry_with_box": frozenset({"box"}),
    "retry_with_points": frozenset({"positive", "negative"}),
    "attach_text": frozenset(),
    "suppress_text": frozenset(),
    "collapse_to_parent": frozenset(),
    "rebuild_background": frozenset({"margin_ratio"}),
    "absorb_residual": frozenset(),
    "absorb_into_parent": frozenset(),
}
_OPTIONAL_ACTION_PARAMETERS = {
    "accept": frozenset({"independent"}),
    "retry_with_box": frozenset({"independent"}),
    "retry_with_points": frozenset({"independent"}),
}
_SINGLE_OBJECT_ACTIONS = frozenset(
    {"accept", "discard", "split", "expand", "shrink", "retry_with_box", "retry_with_points", "suppress_text", "collapse_to_parent", "absorb_residual"}
)


def _validate_normalized_point(value: object, field: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) not in {int, float} or not math.isfinite(item) or not 0 <= item <= 1 for item in value)
    ):
        raise ValueError(f"component action {field} coordinates are invalid")


def validate_component_action(action: object, *, graph: dict | None = None) -> dict:
    object_ids = action.get("object_ids", []) if isinstance(action, dict) else []
    if (
        not isinstance(object_ids, list)
        or not object_ids
        or any(type(value) is not str for value in object_ids)
    ):
        raise ValueError("component action object_ids are invalid")
    validated_graph = validate_component_graph(graph) if graph is not None else None
    graph_nodes = validated_graph["nodes"] if validated_graph is not None else []
    frozen_ids = sorted(
        node["id"] for node in graph_nodes
        if isinstance(node, dict) and node.get("state") == "frozen"
    )
    if (
        validated_graph is None
        and isinstance(action, dict)
        and action.get("action") in {"attach_text", "suppress_text"}
        and len(object_ids) == (2 if action.get("action") == "attach_text" else 1)
    ):
        frozen_ids = [object_ids[-1]]
    candidate_ids = sorted(
        (set(object_ids) | {
            node["id"] for node in graph_nodes
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }) - set(frozen_ids)
    )
    request = {
        "schema_version": 1,
        "page_id": "action_validation",
        "provider": "host",
        "repair_round": 1,
        "source_sha256": "0" * 64,
        "graph_sha256": "0" * 64,
        "candidate_ids": candidate_ids,
        "frozen_ids": frozen_ids,
        "evidence": {
            name: {"path": name, "sha256": "0" * 64}
            for name in COMPONENT_EVIDENCE_NAMES
        },
        "review_evidence": list(FULL_COMPONENT_REVIEW_EVIDENCE),
    }
    validate_component_plan(
        {
            "schema_version": 1,
            "kind": "component_plan",
            "page_id": "action_validation",
            "provider": "host",
            "repair_round": 1,
            "request_sha256": "0" * 64,
            "actions": [action],
        },
        request=request,
        graph=validated_graph,
    )
    return action


def validate_component_plan(plan: object, *, request: dict, graph: dict | None = None) -> dict:
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
    collapsible_parent_ids = set()
    recoverable_parent_ids = set()
    recoverable_retry_ids = set()
    if graph is not None:
        candidate_ids = set(request["candidate_ids"])
        collapsible_parent_ids = {
            node["parent_id"]
            for node in graph["nodes"]
            if node.get("id") in candidate_ids and node.get("parent_id") is not None
        }
        recoverable_parent_ids = {
            node["id"] for node in graph["nodes"]
            if node.get("kind") == "parent" and node.get("state") == "inactive"
        }
        recoverable_retry_ids = {
            node["id"] for node in graph["nodes"]
            if node.get("kind") != "text" and node.get("state") == "inactive"
        }
    touched = {}
    retried_ids = set()
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
            or any(
                value not in known_ids
                and not (
                    name in {"collapse_to_parent", "absorb_into_parent"}
                    and value in collapsible_parent_ids
                )
                and not (
                    name == "absorb_into_parent"
                    and value == object_ids[0]
                    and value in recoverable_parent_ids
                )
                and not (
                    name in {"retry_with_box", "retry_with_points"}
                    and value in recoverable_retry_ids
                )
                and not (
                    name == "rebuild_background"
                    and value in retried_ids
                )
                for value in object_ids
            )
        ):
            raise ValueError("component action object_ids are invalid")
        if (
            (name in _SINGLE_OBJECT_ACTIONS and len(object_ids) != 1)
            or (name == "merge" and len(object_ids) < 2)
            or (name == "absorb_into_parent" and len(object_ids) < 2)
            or (name == "attach_text" and len(object_ids) != 2)
        ):
            raise ValueError("component action object count is invalid")
        if graph is not None:
            _validate_action_graph_roles(name, object_ids, graph)
        if name == "attach_text" and object_ids[1] not in request["frozen_ids"]:
            raise ValueError("attach_text requires a frozen text object")
        frozen_targets = set(object_ids) & set(request["frozen_ids"])
        if frozen_targets and not (
            (
                name == "attach_text"
                and frozen_targets == {object_ids[1]}
            )
            or (
                name == "suppress_text"
                and frozen_targets == {object_ids[0]}
            )
            or name == "rebuild_background"
        ):
            raise ValueError("component action object is frozen")
        if name != "rebuild_background":
            if any(
                value in touched
                and not (touched[value] == "accept" and name == "absorb_residual")
                for value in object_ids
            ):
                raise ValueError("component plan has conflicting object actions")
            touched.update({value: name for value in object_ids})
        parameters = action["parameters"]
        optional_parameters = _OPTIONAL_ACTION_PARAMETERS.get(name, frozenset())
        if (
            not isinstance(parameters, dict)
            or not _ACTION_PARAMETERS[name] <= set(parameters)
            or not set(parameters) <= _ACTION_PARAMETERS[name] | optional_parameters
        ):
            raise ValueError("component action parameters are invalid")
        if "independent" in parameters and type(parameters["independent"]) is not bool:
            raise ValueError("component action independent parameter is invalid")
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
        if name == "rebuild_background" and (
            type(parameters["margin_ratio"]) not in {int, float}
            or not math.isfinite(parameters["margin_ratio"])
            or not 0 < parameters["margin_ratio"] <= 0.1
        ):
            raise ValueError("component action background margin_ratio is invalid")
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
            if (
                len(parameters["positive"]) + len(parameters["negative"])
                > MAX_COMPONENT_PROMPT_POINTS
            ):
                raise ValueError("component action has too many prompt points")
            if not parameters["positive"]:
                raise ValueError("component action positive coordinates are invalid")
        if name in {"retry_with_box", "retry_with_points"}:
            retried_ids.update(object_ids)
    return plan


def _validate_action_graph_roles(action: str, object_ids: list[str], graph: dict) -> None:
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ValueError("component plan graph is invalid")
    nodes = {node.get("id"): node for node in graph["nodes"] if isinstance(node, dict)}
    try:
        selected = [nodes[object_id] for object_id in object_ids]
    except KeyError as error:
        raise ValueError("component action object is missing from graph") from error
    if action == "attach_text":
        if selected[0].get("kind") == "text" or selected[1].get("kind") != "text":
            raise ValueError("attach_text requires visual then text roles")
        if selected[1].get("state") != "frozen":
            raise ValueError("attach_text requires a frozen text object")
        return
    if action == "suppress_text":
        if selected[0].get("kind") != "text" or selected[0].get("state") != "frozen":
            raise ValueError("suppress_text requires a frozen text object")
        return
    if action == "collapse_to_parent":
        if selected[0].get("kind") != "parent":
            raise ValueError("collapse_to_parent requires parent kind")
        return
    if action == "absorb_into_parent":
        if selected[0].get("kind") != "parent":
            raise ValueError("absorb_into_parent requires parent first")
        if any(node.get("kind") == "text" for node in selected[1:]):
            raise ValueError("absorb_into_parent cannot absorb text kind")
        if any(node.get("state") != "pending" for node in selected[1:]):
            raise ValueError("absorb_into_parent requires pending absorbed components")
        return
    if any(node.get("kind") == "text" for node in selected):
        raise ValueError("component action cannot target text kind")
    if action == "merge":
        kinds = {node.get("kind") for node in selected}
        if len(kinds) != 1:
            raise ValueError("merge requires the same component kind")
        if kinds == {"child"} and len({node.get("parent_id") for node in selected}) != 1:
            raise ValueError("merge child components must share one parent")


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
    evidence_names = frozenset(evidence) if isinstance(evidence, dict) else frozenset()
    if evidence_names not in {
        LEGACY_COMPONENT_EVIDENCE_NAMES,
        COMPONENT_EVIDENCE_NAMES,
        LEGACY_COMPONENT_EVIDENCE_NAMES | {ROUND_REVIEW_EVIDENCE_NAME},
        COMPONENT_EVIDENCE_NAMES | {ROUND_REVIEW_EVIDENCE_NAME},
    }:
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
    review_evidence = request["review_evidence"]
    if (
        not isinstance(review_evidence, list)
        or any(type(name) is not str for name in review_evidence)
        or len(review_evidence) != len(set(review_evidence))
        or any(name not in evidence for name in review_evidence)
    ):
        raise ValueError("component agent request review_evidence is invalid")
    canonical = [
        name for name in (*FULL_COMPONENT_REVIEW_EVIDENCE, ROUND_REVIEW_EVIDENCE_NAME)
        if name in review_evidence
    ]
    full = [name for name in FULL_COMPONENT_REVIEW_EVIDENCE if name in evidence]
    if review_evidence != canonical:
        raise ValueError("component agent request review_evidence order is invalid")
    if request["repair_round"] == 1:
        if review_evidence != full or ROUND_REVIEW_EVIDENCE_NAME in evidence:
            raise ValueError("component agent request review_evidence is invalid")
    elif ROUND_REVIEW_EVIDENCE_NAME not in evidence:
        if review_evidence != full:
            raise ValueError("component agent request review_evidence fallback is invalid")
    else:
        required = {
            "source.png", "reconstructed.png", "difference.png",
            "quality-report.json", ROUND_REVIEW_EVIDENCE_NAME,
        }
        if not required <= set(review_evidence):
            raise ValueError("component agent request review_evidence is incomplete")
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


def validate_graph_transition(
    *,
    before: object,
    after: object,
    allowed_suppressed_text_ids: set[str] | frozenset[str] | None = None,
    allowed_reactivated_ids: set[str] | frozenset[str] | None = None,
) -> dict:
    before_graph = validate_component_graph(before)
    allowed = (
        frozenset()
        if allowed_suppressed_text_ids is None
        else frozenset(allowed_suppressed_text_ids)
    )
    reactivated = (
        frozenset()
        if allowed_reactivated_ids is None
        else frozenset(allowed_reactivated_ids)
    )
    if (
        allowed_suppressed_text_ids is not None
        and not isinstance(allowed_suppressed_text_ids, (set, frozenset))
    ) or any(type(value) is not str or not value for value in allowed):
        raise ValueError("suppressed text authorization is invalid")
    if (
        allowed_reactivated_ids is not None
        and not isinstance(allowed_reactivated_ids, (set, frozenset))
    ) or any(type(value) is not str or not value for value in reactivated):
        raise ValueError("component reactivation authorization is invalid")
    before_nodes = {node["id"]: node for node in before_graph["nodes"]}
    if any(
        component_id not in before_nodes
        or before_nodes[component_id]["kind"] != "text"
        or before_nodes[component_id]["state"] != "frozen"
        for component_id in allowed
    ):
        raise ValueError("suppressed text authorization is invalid")
    if any(
        component_id not in before_nodes
        or before_nodes[component_id]["kind"] == "text"
        or before_nodes[component_id]["state"] not in {"inactive", "frozen"}
        for component_id in reactivated
    ):
        raise ValueError("component reactivation authorization is invalid")
    if not isinstance(after, dict) or set(after) != {"nodes"}:
        raise ValueError("component graph fields are invalid")
    if not isinstance(after["nodes"], list):
        raise ValueError("component graph nodes must be a list")
    after_nodes = {
        node["id"]: node
        for node in after["nodes"]
        if isinstance(node, dict) and type(node.get("id")) is str
    }
    actual_reactivated = {
        component_id
        for component_id, node in before_nodes.items()
        if node["state"] in {"inactive", "frozen"}
        and after_nodes.get(component_id, {}).get("state") == "pending"
    }
    if actual_reactivated != set(reactivated):
        if any(
            before_nodes[component_id]["state"] == "frozen"
            for component_id in actual_reactivated - set(reactivated)
        ):
            raise ValueError("frozen component reactivation is not authorized")
        raise ValueError("inactive component reactivation is not authorized")
    if any(
        after_nodes[component_id].get("state") != "pending"
        for component_id in actual_reactivated
    ):
        raise ValueError("inactive component reactivation is not authorized")
    for node in before_graph["nodes"]:
        if node["state"] != "frozen":
            continue
        replacement = after_nodes.get(node["id"])
        fields = _FROZEN_FIELDS
        if node["id"] in reactivated:
            fields = tuple(field for field in fields if field != "state")
            valid = replacement is not None and replacement.get("state") == "pending"
        elif node["id"] in allowed:
            fields = tuple(field for field in fields if field != "state")
            valid = replacement is not None and replacement.get("state") == "inactive"
        elif node["kind"] != "text" and set(node["text_ids"]) & allowed:
            fields = tuple(field for field in fields if field != "text_ids")
            valid = replacement is not None and replacement.get("text_ids") == [
                text_id for text_id in node["text_ids"] if text_id not in allowed
            ]
        else:
            valid = replacement is not None
        if not valid or any(
            replacement.get(field) != node[field] for field in fields
        ):
            raise ValueError(f"frozen component {node['id']} cannot change")
    return validate_component_graph(after)
