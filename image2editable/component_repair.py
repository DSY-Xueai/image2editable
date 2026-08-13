from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
import time
import uuid

from scripts.initial_diagnostics import validate_initial_diagnostics

from image2editable.component_contracts import (
    COMPONENT_EVIDENCE_NAMES,
    LEGACY_COMPONENT_EVIDENCE_NAMES,
    MAX_REPAIR_ROUNDS,
    validate_agent_provider,
    validate_component_agent_request,
    validate_component_graph,
    validate_component_plan,
    validate_component_repair_state,
    validate_repair_round,
)
from image2editable.execution import ExecutionLease


EVIDENCE_NAMES = tuple(sorted(COMPONENT_EVIDENCE_NAMES))
LEGACY_EVIDENCE_NAMES = tuple(sorted(LEGACY_COMPONENT_EVIDENCE_NAMES))
REQUEST_NAME = "component_agent_request.json"
MARKER_NAME = "publication-marker.json"
INTEGRITY_DIRECTORY = ".component-agent-integrity"
INTEGRITY_KEY_NAME = "key.bin"
PUBLICATION_LOCK_NAME = ".component-agent-publication.lock"
IO_CHUNK_SIZE = 1024 * 1024
GRAPH_JSON_LIMIT = 16 * 1024 * 1024
REQUEST_JSON_LIMIT = 4 * 1024 * 1024
MARKER_JSON_LIMIT = 64 * 1024
PRESENTATION_ASSET_LIMIT = 256 * 1024 * 1024
COMPONENT_STATE_NAME = "component_state.json"
_REPAIRABLE_PAGE_VIOLATIONS = frozenset({
    "background_text_residual",
    "unexplained_visual_residual",
})
_LEGACY_QUALITY_INPUT_NAMES = frozenset({
    "background", "reconstructed", "text_mask", "native_check",
    "presentation_manifest",
})
_QUALITY_INPUT_NAMES = _LEGACY_QUALITY_INPUT_NAMES | {"foreground_evidence"}


def advance_component_repair(
    store, page_id: str, *, _lease: ExecutionLease | None = None
) -> dict:
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    state_path = reconstruction / COMPONENT_STATE_NAME
    if not state_path.is_file():
        return {"status": "needs_initialization", "page_id": page_id}
    if _lease is not None:
        _require_held_execution_lease(store, _lease)
    lease = nullcontext() if _lease is not None else ExecutionLease(
        store.root / "execution.lock", run_root=store.root
    )
    with lease:
        state = validate_component_repair_state(store.read_json(
            f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}"
        ))
        _validate_repair_state_identity(store, state, page_id)
        _load_state_artifact(store.root, state["current_round"]["request_ref"])
        if state["phase"] == "request_published":
            updated = dict(state)
            updated["phase"] = "awaiting_plan"
            updated["revision"] += 1
            updated["updated_at"] = _utc_now()
            validate_component_repair_state(updated)
            store.write_json(
                f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}", updated
            )
            return {"status": "awaiting_agent", "page_id": page_id,
                    "repair_round": updated["repair_round"]}
        if state["phase"] == "awaiting_plan":
            return {"status": "awaiting_agent", "page_id": page_id,
                    "repair_round": state["repair_round"]}
        if state["phase"] == "plan_recorded":
            request_payload = _load_state_artifact(
                store.root, state["current_round"]["request_ref"]
            )
            plan_payload = _load_state_artifact(
                store.root, state["current_round"]["plan_ref"]
            )
            graph_payload = _load_state_artifact(
                store.root, state["graph_ref"], max_bytes=GRAPH_JSON_LIMIT
            )
            request = json.loads(request_payload.decode("utf-8"))
            plan = json.loads(plan_payload.decode("utf-8"))
            graph = json.loads(graph_payload.decode("utf-8"))
            validate_component_plan(plan, request=request, graph=graph)
            normalized = _normalized_plan_sha256(plan)
            if not plan["actions"] or normalized == state["last_normalized_plan_sha256"]:
                updated = dict(state)
                updated["last_normalized_plan_sha256"] = normalized
                return _commit_fallback_required(
                    store,
                    updated,
                    page_id,
                    "empty_plan" if not plan["actions"] else "repeated_plan",
                )
            return {"status": "needs_execution", "page_id": page_id,
                    "repair_round": state["repair_round"]}
        if state["phase"] == "actions_executed":
            execution = json.loads(_load_state_artifact(
                store.root, state["current_round"]["execution_ref"]
            ).decode("utf-8"))
            if execution["executable_action_count"] == 0:
                return _commit_fallback_required(
                    store, state, page_id, "no_executable_actions"
                )
            return {"status": "needs_quality", "page_id": page_id,
                    "repair_round": state["repair_round"]}
        if state["phase"] == "quality_recorded":
            return _commit_component_freeze(store, state, page_id)
        if state["phase"] == "freeze_committed":
            page_violations = _repairable_page_quality_violations(store, state)
            if state["failed_ids"] or page_violations:
                if not _next_round_progress_allowed(store, state):
                    return _commit_fallback_required(
                        store, state, page_id, "no_quality_improvement"
                    )
                if state["repair_round"] >= MAX_REPAIR_ROUNDS:
                    return _commit_fallback_required(
                        store, state, page_id, "round_limit"
                    )
                return {"status": "needs_next_round", "page_id": page_id,
                        "repair_round": state["repair_round"] + 1,
                        "candidate_ids": list(state["failed_ids"]),
                        "page_violations": page_violations}
            blocking_violations = _blocking_page_quality_violations(store, state)
            if blocking_violations:
                updated = dict(state)
                updated["stop_reason"] = (
                    "unowned_raster_text"
                    if "unowned_raster_text" in blocking_violations
                    else "page_quality_failed"
                )
                return _commit_preserved_warning(store, updated, page_id)
            if state["initial_component_count"] and not state["frozen"]:
                return _commit_preserved_warning(store, state, page_id)
            return _commit_ready_result(store, state, page_id)
        if state["phase"] == "fallback_required":
            return {"status": "needs_parent_fallback", "page_id": page_id,
                    "parent_ids": list(state["fallback"]["parent_ids"])}
        if state["phase"] == "fallback_executed":
            return {"status": "needs_parent_quality", "page_id": page_id,
                    "parent_ids": list(state["fallback"]["parent_ids"])}
        if state["phase"] == "fallback_quality_recorded":
            return _commit_parent_fallback_result(store, state, page_id)
        return {"status": state["phase"], "page_id": page_id,
                "repair_round": state["repair_round"]}


def resume_round_limited_component_repair(store, page_id: str) -> bool:
    state_path = f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}"
    try:
        state = validate_component_repair_state(store.read_json(state_path))
    except (FileNotFoundError, ValueError):
        return False
    resumable_warning = (
        state["phase"] == "preserved_with_warning"
        and state["stop_reason"] in {"round_limit", "no_quality_improvement"}
    )
    interrupted_resume = (
        state["phase"] == "freeze_committed"
        and state["status"] == "active"
        and state["stop_reason"] is None
    )
    if not (
        (resumable_warning or interrupted_resume)
        and state["repair_round"] < MAX_REPAIR_ROUNDS
        and state["failed_ids"]
    ):
        return False
    graph_ref = state["graph_ref"]
    try:
        graph = validate_component_graph(json.loads(
            _load_state_artifact(
                store.root, graph_ref, max_bytes=GRAPH_JSON_LIMIT
            ).decode("utf-8")
        ))
    except (FileNotFoundError, ValueError, RuntimeError):
        graph = {"nodes": []}
    pending_ids = {
        node["id"] for node in graph["nodes"]
        if node["kind"] != "text" and node["state"] in {
            "pending", "pending_gate",
        }
    }
    if not set(state["failed_ids"]) <= pending_ids:
        execution_path = store.root / Path(
            *PurePosixPath(state["current_round"]["execution_ref"]["path"]).parts
        )
        frozen_graph_path = execution_path.parent / "component-graph-frozen.json"
        try:
            graph_ref, graph_payload = _artifact_reference(
                store.root, frozen_graph_path, "round-limit frozen graph"
            )
            graph = validate_component_graph(json.loads(
                graph_payload.decode("utf-8")
            ))
        except (FileNotFoundError, ValueError, RuntimeError):
            return False
        pending_ids = {
            node["id"] for node in graph["nodes"]
            if node["kind"] != "text" and node["state"] in {
                "pending", "pending_gate",
            }
        }
        if not set(state["failed_ids"]) <= pending_ids:
            return False
    frozen = {
        node["id"]: node["mask_sha256"]
        for node in graph["nodes"]
        if node["kind"] != "text" and node["state"] == "frozen"
    }
    updated = dict(state)
    updated.update({
        "phase": "freeze_committed",
        "status": "active",
        "graph_ref": graph_ref,
        "frozen": frozen,
        "fallback": {"status": "none", "parent_ids": []},
        "fallback_graph_ref": None,
        "fallback_quality_ref": None,
        "fallback_input_refs": None,
        "revision": state["revision"] + 1,
        "updated_at": _utc_now(),
    })
    validate_component_repair_state(updated)
    store.write_json(state_path, updated)
    return True


def initialize_component_repair_state(
    store,
    page_id: str,
    *,
    request_path: str | Path,
    initial_component_count: int,
    _lease: ExecutionLease | None = None,
) -> dict:
    if _lease is not None:
        _require_held_execution_lease(store, _lease)
    lease = nullcontext() if _lease is not None else ExecutionLease(
        store.root / "execution.lock", run_root=store.root
    )
    with lease:
        return _initialize_component_repair_state_locked(
            store, page_id, request_path=request_path,
            initial_component_count=initial_component_count,
        )


def record_local_component_plan(
    store,
    page_id: str,
    *,
    plan: dict,
    _lease: ExecutionLease | None = None,
) -> dict:
    if _lease is not None:
        _require_held_execution_lease(store, _lease)
    lease = nullcontext() if _lease is not None else ExecutionLease(
        store.root / "execution.lock", run_root=store.root
    )
    with lease:
        relative_state = (
            f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}"
        )
        state = validate_component_repair_state(store.read_json(relative_state))
        _validate_repair_state_identity(store, state, page_id)
        if state["provider"] != "local" or state["phase"] not in {
            "awaiting_plan",
            "plan_recorded",
        }:
            raise RuntimeError("Local component plan does not match repair state")
        request_payload = _load_state_artifact(
            store.root, state["current_round"]["request_ref"]
        )
        request = json.loads(request_payload.decode("utf-8"))
        graph = json.loads(
            _load_state_artifact(
                store.root,
                state["graph_ref"],
                max_bytes=GRAPH_JSON_LIMIT,
            ).decode("utf-8")
        )
        validate_component_plan(plan, request=request, graph=graph)
        if (
            plan["request_sha256"]
            != state["current_round"]["request_ref"]["sha256"]
        ):
            raise ValueError(
                "component plan request_sha256 does not match current request"
            )
        payload = json.dumps(
            plan,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        reconstruction = store.root / "pages" / page_id / "reconstruction"
        destination = reconstruction / (
            f"local-component-plan-{state['repair_round']:02d}-"
            f"{state['current_round']['request_ref']['sha256']}.json"
        )
        if destination.exists() or destination.is_symlink():
            if (
                _read_bound_file(
                    destination,
                    store.root,
                    max_bytes=REQUEST_JSON_LIMIT,
                    label="local component plan",
                )
                != payload
            ):
                raise RuntimeError(
                    "A different local component plan is already recorded"
                )
        else:
            _write_exclusive(destination, payload, reconstruction)
        plan_ref = {
            "path": destination.resolve()
            .relative_to(store.root.resolve())
            .as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if state["phase"] == "plan_recorded":
            if state["current_round"]["plan_ref"] != plan_ref:
                raise RuntimeError(
                    "A different local component plan is already recorded"
                )
            return {
                "status": "recorded",
                "plan_ref": plan_ref,
                "recovered": True,
            }
        updated = dict(state)
        updated["current_round"] = dict(state["current_round"])
        updated["current_round"]["plan_ref"] = plan_ref
        updated["phase"] = "plan_recorded"
        updated["plan_count"] += 1
        updated["revision"] += 1
        updated["updated_at"] = _utc_now()
        validate_component_repair_state(updated)
        store.write_json(relative_state, updated)
        return {
            "status": "recorded",
            "plan_ref": plan_ref,
            "recovered": False,
        }


def _require_held_execution_lease(store, lease: ExecutionLease) -> None:
    if (
        not isinstance(lease, ExecutionLease)
        or not lease._file_locked
        or lease._file is None
        or lease.path != (store.root / "execution.lock").resolve()
        or lease.run_root != store.root.resolve()
    ):
        raise RuntimeError("component repair requires the held Run execution lease")


def _initialize_component_repair_state_locked(
    store,
    page_id: str,
    *,
    request_path: str | Path,
    initial_component_count: int,
) -> dict:
    if type(initial_component_count) is not int or initial_component_count < 0:
        raise ValueError("initial_component_count is invalid")
    request_path = Path(request_path)
    request = load_component_agent_request(request_path)
    graph = load_component_agent_graph(request_path)
    manifest = store.read_json("job_manifest.json")
    provider = manifest.get("options", {}).get("agent_provider")
    if request["page_id"] != page_id or request["provider"] != provider:
        raise ValueError("component repair initialization identity mismatch")
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    relative_request = request_path.resolve().relative_to(store.root.resolve()).as_posix()
    graph_path = request_path.parent / request["evidence"]["component-graph.json"]["path"]
    relative_graph = graph_path.resolve().relative_to(store.root.resolve()).as_posix()
    frozen = {
        node["id"]: node["mask_sha256"] for node in graph["nodes"]
        if node["state"] == "frozen" and node["kind"] != "text"
    }
    parent_assets = {}
    for node in graph["nodes"]:
        if node["kind"] != "parent":
            continue
        mask_path = request_path.parent / Path(*PurePosixPath(node["mask"]).parts)
        reference, _ = _artifact_reference(
            store.root, mask_path, "initial parent mask"
        )
        if reference["sha256"] != node["mask_sha256"]:
            raise RuntimeError("initial parent mask hash mismatch")
        parent_assets[node["id"]] = reference
    state = {
        "schema_version": 1, "page_id": page_id, "provider": provider,
        "source_sha256": request["source_sha256"],
        "initial_component_count": initial_component_count,
        "quality_gate_version": (
            2 if "unexplained-mask.png" in request["evidence"] else 1
        ),
        "revision": 1,
        "phase": "request_published", "status": "active",
        "repair_round": request["repair_round"], "plan_count": 0,
        "stop_reason": None,
        "graph_ref": {"path": relative_graph, "sha256": request["graph_sha256"]},
        "current_round": {
            "round": request["repair_round"],
            "request_ref": {"path": relative_request,
                            "sha256": hashlib.sha256(request_path.read_bytes()).hexdigest()},
            "plan_ref": None, "execution_ref": None, "quality_ref": None,
        },
        "frozen": frozen,
        "parent_assets": parent_assets,
        "round_history": [],
        "fallback_graph_ref": None, "fallback_quality_ref": None,
        "fallback_input_refs": None,
        "candidate_ids": list(request["candidate_ids"]),
        "failed_ids": list(request["candidate_ids"]),
        "fallback": {"status": "none", "parent_ids": []},
        "last_normalized_plan_sha256": None, "result_ref": None,
        "delivery_checks": {"pptx_reopen": "unknown"},
        "updated_at": _utc_now(),
    }
    validate_component_repair_state(state)
    relative_state = f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}"
    state_path = reconstruction / COMPONENT_STATE_NAME
    if state_path.exists() or state_path.is_symlink():
        existing = validate_component_repair_state(store.read_json(relative_state))
        if (
            existing["page_id"] == page_id
            and existing["provider"] == provider
            and existing["source_sha256"] == request["source_sha256"]
            and existing["initial_component_count"] == initial_component_count
            and existing["current_round"]["request_ref"] == state["current_round"]["request_ref"]
        ):
            return existing
        raise RuntimeError("component repair state already exists with another identity")
    store.write_json(relative_state, state)
    return state


def record_component_execution(
    store,
    page_id: str,
    *,
    execution_path: str | Path,
    output_graph_path: str | Path,
    _lease: ExecutionLease | None = None,
) -> dict:
    if _lease is not None:
        _require_held_execution_lease(store, _lease)
    lease = nullcontext() if _lease is not None else ExecutionLease(
        store.root / "execution.lock", run_root=store.root
    )
    with lease:
        relative = f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}"
        state = validate_component_repair_state(store.read_json(relative))
        _validate_repair_state_identity(store, state, page_id)
        if state["phase"] != "plan_recorded":
            raise RuntimeError("component repair is not ready for execution")
        execution_ref, execution_payload = _artifact_reference(
            store.root, Path(execution_path), "component execution JSON"
        )
        graph_ref, graph_payload = _artifact_reference(
            store.root, Path(output_graph_path), "component execution graph"
        )
        execution = json.loads(execution_payload.decode("utf-8"))
        if not isinstance(execution, dict) or set(execution) != {
            "schema_version", "page_id", "provider", "repair_round",
            "request_sha256", "input_graph_sha256", "output_graph_sha256",
            "executable_action_count", "quality_input_refs",
        }:
            raise ValueError("component execution record fields are invalid")
        if (
            execution["schema_version"] != 1
            or execution["page_id"] != page_id
            or execution["provider"] != state["provider"]
            or execution["repair_round"] != state["repair_round"]
            or execution["request_sha256"] != state["current_round"]["request_ref"]["sha256"]
            or execution["input_graph_sha256"] != state["graph_ref"]["sha256"]
            or execution["output_graph_sha256"] != graph_ref["sha256"]
            or type(execution["executable_action_count"]) is not int
            or execution["executable_action_count"] < 0
        ):
            raise ValueError("component execution record identity is invalid")
        request_path = store.root / Path(
            *PurePosixPath(state["current_round"]["request_ref"]["path"]).parts
        )
        request = load_component_agent_request(request_path)
        before = json.loads(_load_state_artifact(
            store.root, state["graph_ref"], max_bytes=GRAPH_JSON_LIMIT
        ).decode("utf-8"))
        after = json.loads(graph_payload.decode("utf-8"))
        plan = json.loads(_load_state_artifact(
            store.root, state["current_round"]["plan_ref"]
        ).decode("utf-8"))
        suppressed_text_ids = {
            action["object_ids"][0] for action in plan["actions"]
            if action["action"] == "suppress_text"
        }
        before_by_id = {node["id"]: node for node in before["nodes"]}
        after_by_id = {node["id"]: node for node in after["nodes"]}
        reactivated_ids = {
            action["object_ids"][0]
            for action in plan["actions"]
            if action["action"] in {
                "retry_with_box", "retry_with_points",
                "collapse_to_parent", "absorb_into_parent",
            }
            and before_by_id[action["object_ids"][0]]["state"] == "inactive"
            and after_by_id.get(action["object_ids"][0], {}).get("state") == "pending"
        }
        _, _, presentation_manifest, _ = _verify_quality_input_refs(
            store, state, execution["quality_input_refs"],
            request=request,
            request_path=request_path,
            expected_graph_sha256=graph_ref["sha256"],
            expected_component_ids=_presentation_component_ids(after),
            return_bound_inputs=True,
        )
        request_manifest_record = request["evidence"][
            "presentation-manifest.json"
        ]
        request_manifest_path = request_path.parent / Path(
            *PurePosixPath(request_manifest_record["path"]).parts
        )
        request_manifest_payload = _read_bound_file(
            request_manifest_path,
            store.root / "pages" / page_id / "reconstruction",
            max_bytes=GRAPH_JSON_LIMIT,
            label="request presentation manifest",
        )
        if (
            hashlib.sha256(request_manifest_payload).hexdigest()
            != request_manifest_record["sha256"]
        ):
            raise RuntimeError("request presentation manifest hash mismatch")
        request_manifest = _validate_presentation_manifest_payload(
            request_manifest_payload,
            store.root / "pages" / page_id / "reconstruction",
            source_sha256=state["source_sha256"],
            graph_sha256=request["graph_sha256"],
            expected_component_ids=_presentation_component_ids(before),
        )
        _validate_frozen_presentation_assets(
            request_manifest,
            presentation_manifest,
            frozen_ids=set(state["frozen"]),
        )
        from image2editable.component_contracts import validate_graph_transition
        validate_graph_transition(
            before=before,
            after=after,
            allowed_suppressed_text_ids=suppressed_text_ids,
            allowed_reactivated_ids=reactivated_ids,
        )
        for node in after["nodes"]:
            mask_path = Path(output_graph_path).parent / Path(
                *PurePosixPath(node["mask"]).parts
            )
            if _hash_bound_file(mask_path, store.root) != node["mask_sha256"]:
                raise ValueError(f"component execution mask hash mismatch: {node['id']}")
        _validate_inactive_source_provenance(
            store.root,
            before=before,
            after=after,
            before_graph_path=store.root / Path(
                *PurePosixPath(state["graph_ref"]["path"]).parts
            ),
            after_graph_path=store.root / Path(
                *PurePosixPath(graph_ref["path"]).parts
            ),
        )
        normalized = _normalized_plan_sha256(plan)
        action_count = len(plan["actions"])
        if (
            execution["executable_action_count"] not in {0, action_count}
            or (execution["executable_action_count"] == 0 and after != before)
            or (execution["executable_action_count"] > 0 and action_count == 0)
        ):
            raise ValueError("component execution count does not match plan and graph")
        updated = dict(state)
        updated["current_round"] = dict(state["current_round"])
        updated["current_round"]["execution_ref"] = execution_ref
        updated["graph_ref"] = graph_ref
        updated["candidate_ids"] = sorted(
            node["id"] for node in after["nodes"]
            if node["kind"] != "text"
            and node["state"] in {"pending", "pending_gate"}
        )
        updated["failed_ids"] = list(updated["candidate_ids"])
        updated["phase"] = "actions_executed"
        updated["last_normalized_plan_sha256"] = normalized
        updated["round_history"] = list(state["round_history"]) + [{
            "round": state["repair_round"],
            "plan_sha256": state["current_round"]["plan_ref"]["sha256"],
            "normalized_plan_sha256": normalized,
            "execution_sha256": execution_ref["sha256"],
            "quality_sha256": None, "frozen_ids": [], "failed_ids": [],
        }]
        updated["revision"] += 1
        updated["updated_at"] = _utc_now()
        validate_component_repair_state(updated)
        store.write_json(relative, updated)
        return updated


def _validate_inactive_source_provenance(
    root: Path,
    *,
    before: dict,
    after: dict,
    before_graph_path: Path,
    after_graph_path: Path,
) -> None:
    before_by_id = {node["id"]: node for node in before["nodes"]}
    after_ids = {node["id"] for node in after["nodes"]}
    if set(before_by_id) - after_ids:
        raise ValueError("component source identity changed")
    for node in after["nodes"]:
        original = before_by_id.get(node["id"])
        if node["state"] != "inactive" or original is None:
            continue
        if any(
            node[field] != original[field]
            for field in original
            if field != "state"
        ):
            raise ValueError(
                f"inactive source provenance changed: {node['id']}"
            )
        before_mask = before_graph_path.parent / Path(
            *PurePosixPath(original["mask"]).parts
        )
        after_mask = after_graph_path.parent / Path(
            *PurePosixPath(node["mask"]).parts
        )
        before_digest = _hash_bound_file(before_mask, root)
        after_digest = _hash_bound_file(after_mask, root)
        if (
            before_digest != original["mask_sha256"]
            or after_digest != before_digest
        ):
            raise ValueError(
                f"inactive source provenance changed: {node['id']}"
            )


def record_component_quality(
    store, page_id: str, *, _lease: ExecutionLease | None = None
) -> dict:
    if _lease is not None:
        _require_held_execution_lease(store, _lease)
    lease = nullcontext() if _lease is not None else ExecutionLease(
        store.root / "execution.lock", run_root=store.root
    )
    with lease:
        relative = f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}"
        state = validate_component_repair_state(store.read_json(relative))
        _validate_repair_state_identity(store, state, page_id)
        if state["phase"] != "actions_executed":
            raise RuntimeError("component repair is not ready for quality")
        execution = json.loads(_load_state_artifact(
            store.root, state["current_round"]["execution_ref"]
        ).decode("utf-8"))
        quality_ref = _recompute_quality_artifact(
            store, state, expected_component_ids=state["candidate_ids"],
            quality_input_refs=execution["quality_input_refs"],
            filename="component-quality.json",
        )
        updated = dict(state)
        updated["current_round"] = dict(state["current_round"])
        updated["current_round"]["quality_ref"] = quality_ref
        updated["phase"] = "quality_recorded"
        updated["revision"] += 1
        updated["updated_at"] = _utc_now()
        validate_component_repair_state(updated)
        store.write_json(relative, updated)
        return updated


def record_next_component_request(
    store,
    page_id: str,
    *,
    request_path: str | Path,
    _lease: ExecutionLease | None = None,
) -> dict:
    if _lease is not None:
        _require_held_execution_lease(store, _lease)
    lease = nullcontext() if _lease is not None else ExecutionLease(
        store.root / "execution.lock", run_root=store.root
    )
    with lease:
        relative = f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}"
        state = validate_component_repair_state(store.read_json(relative))
        _validate_repair_state_identity(store, state, page_id)
        if (
            state["phase"] != "freeze_committed"
            or not (
                state["failed_ids"]
                or _repairable_page_quality_violations(store, state)
            )
        ):
            raise RuntimeError("component repair is not ready for a next round")
        if state["repair_round"] >= MAX_REPAIR_ROUNDS:
            raise RuntimeError(
                f"component repair cannot publish round {MAX_REPAIR_ROUNDS + 1} "
                f"after {MAX_REPAIR_ROUNDS} rounds"
            )
        if not _next_round_progress_allowed(store, state):
            raise RuntimeError("component quality did not improve")
        request_path = Path(request_path)
        request = load_component_agent_request(request_path)
        graph = load_component_agent_graph(request_path)
        if (
            request["evidence"]["quality-report.json"]["sha256"]
            != state["current_round"]["quality_ref"]["sha256"]
        ):
            raise ValueError(
                "next component request does not reference previous quality"
            )
        previous_quality = json.loads(_load_state_artifact(
            store.root,
            state["current_round"]["quality_ref"],
            max_bytes=GRAPH_JSON_LIMIT,
        ).decode("utf-8"))
        if (
            not isinstance(previous_quality, dict)
            or previous_quality.get("schema_version") != 1
            or type(previous_quality.get("schema_version")) is not int
            or previous_quality.get("page_id") != state["page_id"]
            or previous_quality.get("provider") != state["provider"]
            or previous_quality.get("repair_round") != state["repair_round"]
            or previous_quality.get("request_sha256")
            != state["current_round"]["request_ref"]["sha256"]
            or previous_quality.get("quality_gate_version")
            != state["quality_gate_version"]
            or previous_quality.get("initial_component_count")
            != state["initial_component_count"]
        ):
            raise ValueError("previous component quality artifact identity is invalid")
        validate_initial_diagnostics(
            previous_quality.get("initial_diagnostics"),
            source_sha256=state["source_sha256"],
        )
        _previous_component_reports(
            previous_quality,
            state={**state, "repair_round": state["repair_round"] + 1},
            request=request,
            active_component_ids=_presentation_component_ids(graph),
        )
        previous_manifest = _quality_presentation_manifest(
            store, state, previous_quality
        )
        next_manifest = _request_presentation_manifest(
            store, state, request_path, request
        )
        _validate_frozen_presentation_assets(
            previous_manifest,
            next_manifest,
            frozen_ids=set(state["frozen"]),
        )
        if (
            request["page_id"] != page_id
            or request["provider"] != state["provider"]
            or request["repair_round"] != state["repair_round"] + 1
            or request["source_sha256"] != state["source_sha256"]
            or request["candidate_ids"] != state["failed_ids"]
            or request["frozen_ids"] != sorted(
                set(state["frozen"])
                | {
                    node["id"] for node in graph["nodes"]
                    if node["kind"] == "text" and node["state"] == "frozen"
                }
            )
        ):
            raise ValueError("next component request does not match repair state")
        before = json.loads(_load_state_artifact(
            store.root, state["graph_ref"], max_bytes=GRAPH_JSON_LIMIT
        ).decode("utf-8"))
        from image2editable.component_contracts import validate_graph_transition
        validate_graph_transition(before=before, after=graph)
        graph_path = request_path.parent / request["evidence"]["component-graph.json"]["path"]
        request_ref, _ = _artifact_reference(
            store.root, request_path, "next component request"
        )
        graph_ref, _ = _artifact_reference(
            store.root, graph_path, "next component graph"
        )
        updated = dict(state)
        updated["repair_round"] = request["repair_round"]
        updated["phase"] = "request_published"
        updated["stop_reason"] = None
        updated["graph_ref"] = graph_ref
        updated["candidate_ids"] = list(request["candidate_ids"])
        updated["current_round"] = {
            "round": request["repair_round"], "request_ref": request_ref,
            "plan_ref": None, "execution_ref": None, "quality_ref": None,
        }
        updated["revision"] += 1
        updated["updated_at"] = _utc_now()
        validate_component_repair_state(updated)
        store.write_json(relative, updated)
        return updated


def record_parent_fallback_execution(
    store, page_id: str, *, graph_path: str | Path, quality_input_refs: dict,
    _lease: ExecutionLease | None = None,
) -> dict:
    if _lease is not None:
        _require_held_execution_lease(store, _lease)
    lease = nullcontext() if _lease is not None else ExecutionLease(
        store.root / "execution.lock", run_root=store.root
    )
    with lease:
        relative = f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}"
        state = validate_component_repair_state(store.read_json(relative))
        _validate_repair_state_identity(store, state, page_id)
        if state["phase"] != "fallback_required":
            raise RuntimeError("component repair does not require parent fallback")
        graph_ref, payload = _artifact_reference(
            store.root, Path(graph_path), "parent fallback graph"
        )
        graph = validate_component_graph(json.loads(payload.decode("utf-8")))
        request_path = store.root / Path(
            *PurePosixPath(state["current_round"]["request_ref"]["path"]).parts
        )
        request = load_component_agent_request(request_path)
        (
            quality_input_refs,
            _,
            fallback_manifest,
            _,
        ) = _verify_quality_input_refs(
            store, state, quality_input_refs,
            request=request,
            request_path=request_path,
            expected_graph_sha256=graph_ref["sha256"],
            expected_component_ids=_presentation_component_ids(graph),
            return_bound_inputs=True,
        )
        if state["current_round"]["quality_ref"] is None:
            previous_manifest = _request_presentation_manifest(
                store, state, request_path, request
            )
        else:
            previous_quality = json.loads(_load_state_artifact(
                store.root,
                state["current_round"]["quality_ref"],
                max_bytes=GRAPH_JSON_LIMIT,
            ).decode("utf-8"))
            previous_manifest = _quality_presentation_manifest(
                store, state, previous_quality
            )
        _validate_frozen_presentation_assets(
            previous_manifest,
            fallback_manifest,
            frozen_ids=set(state["frozen"]),
        )
        before = json.loads(_load_state_artifact(
            store.root, state["graph_ref"], max_bytes=GRAPH_JSON_LIMIT
        ).decode("utf-8"))
        from image2editable.component_contracts import validate_graph_transition
        validate_graph_transition(before=before, after=graph)
        for node in graph["nodes"]:
            mask_path = Path(graph_path).parent / Path(
                *PurePosixPath(node["mask"]).parts
            )
            if _hash_bound_file(mask_path, store.root) != node["mask_sha256"]:
                raise ValueError(f"parent fallback mask hash mismatch: {node['id']}")
        by_id = {node["id"]: node for node in graph["nodes"]}
        for parent_id in state["fallback"]["parent_ids"]:
            parent = by_id.get(parent_id)
            initial_ref = state["parent_assets"][parent_id]
            if (
                parent is None or parent["kind"] != "parent"
                or parent["state"] != "pending_gate"
                or parent["mask_sha256"] != initial_ref["sha256"]
            ):
                raise ValueError("parent fallback did not preserve intact parent")
            initial_mask = _load_state_artifact(store.root, initial_ref)
            output_mask_path = Path(graph_path).parent / Path(
                *PurePosixPath(parent["mask"]).parts
            )
            output_ref, output_mask = _artifact_reference(
                store.root, output_mask_path, "parent fallback mask"
            )
            if output_ref["sha256"] != initial_ref["sha256"] or output_mask != initial_mask:
                raise ValueError("parent fallback mask is not the intact parent asset")
            if any(
                node["parent_id"] == parent_id and node["state"] != "inactive"
                for node in graph["nodes"]
            ):
                raise ValueError("parent fallback descendants must be inactive")
        updated = dict(state)
        updated["graph_ref"] = graph_ref
        updated["fallback_graph_ref"] = graph_ref
        updated["fallback_input_refs"] = quality_input_refs
        updated["phase"] = "fallback_executed"
        updated["fallback"] = {
            "status": "parent_pending",
            "parent_ids": list(state["fallback"]["parent_ids"]),
        }
        updated["revision"] += 1
        updated["updated_at"] = _utc_now()
        validate_component_repair_state(updated)
        store.write_json(relative, updated)
        return updated


def record_parent_fallback_quality(
    store, page_id: str, *, _lease: ExecutionLease | None = None
) -> dict:
    if _lease is not None:
        _require_held_execution_lease(store, _lease)
    lease = nullcontext() if _lease is not None else ExecutionLease(
        store.root / "execution.lock", run_root=store.root
    )
    with lease:
        relative = f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}"
        state = validate_component_repair_state(store.read_json(relative))
        _validate_repair_state_identity(store, state, page_id)
        if state["phase"] != "fallback_executed":
            raise RuntimeError("parent fallback is not ready for quality")
        quality_ref = _recompute_quality_artifact(
            store, state, expected_component_ids=state["fallback"]["parent_ids"],
            quality_input_refs=state["fallback_input_refs"],
            filename="parent-quality.json",
        )
        updated = dict(state)
        updated["fallback_quality_ref"] = quality_ref
        updated["phase"] = "fallback_quality_recorded"
        updated["revision"] += 1
        updated["updated_at"] = _utc_now()
        validate_component_repair_state(updated)
        store.write_json(relative, updated)
        return updated


def _validate_repair_state_identity(store, state: dict, page_id: str) -> None:
    provider = store.read_json("job_manifest.json").get("options", {}).get("agent_provider")
    if state["page_id"] != page_id or state["provider"] != provider:
        raise RuntimeError("component repair state identity mismatch")


def _artifact_reference(root: Path, path: Path, label: str) -> tuple[dict, bytes]:
    contained = _contained_path(path, root)
    payload = _read_bound_file(
        contained, root, max_bytes=GRAPH_JSON_LIMIT, label=label
    )
    return {
        "path": contained.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }, payload


def _validate_presentation_manifest_payload(
    payload: bytes,
    reconstruction: Path,
    *,
    source_sha256: str,
    graph_sha256: str,
    run_root: Path | None = None,
    expected_component_ids: list[str] | None = None,
) -> dict:
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("presentation manifest JSON is invalid") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "source_sha256", "graph_sha256", "components"
    }:
        raise ValueError("presentation manifest fields are invalid")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("presentation manifest schema_version is invalid")
    if manifest["source_sha256"] != source_sha256:
        raise RuntimeError("presentation manifest source hash mismatch")
    if manifest["graph_sha256"] != graph_sha256:
        raise RuntimeError("presentation manifest graph hash mismatch")
    components = manifest["components"]
    if not isinstance(components, list):
        raise ValueError("presentation manifest components are invalid")
    component_ids = []
    run_root = reconstruction.parents[2] if run_root is None else run_root
    asset_fields = (
        "rgba", "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask",
    )
    for component in components:
        if not isinstance(component, dict) or set(component) != {
            "component_id", *asset_fields, "metrics"
        }:
            raise ValueError("presentation manifest component fields are invalid")
        component_id = component["component_id"]
        if type(component_id) is not str or not component_id:
            raise ValueError("presentation manifest component_id is invalid")
        component_ids.append(component_id)
        metrics = component["metrics"]
        if (
            not isinstance(metrics, dict)
            or any(type(name) is not str or not name for name in metrics)
            or any(
                type(value) not in {int, float} or not math.isfinite(value)
                for value in metrics.values()
            )
        ):
            raise ValueError("presentation manifest metrics are invalid")
        for field in asset_fields:
            reference = component[field]
            if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
                raise ValueError("presentation asset reference is invalid")
            path = reference["path"]
            if type(path) is not str or not path or "\\" in path or ":" in path:
                raise ValueError("presentation asset path is invalid")
            pure = PurePosixPath(path)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError("presentation asset path is invalid")
            asset_path = run_root / Path(*pure.parts)
            try:
                asset_path.relative_to(reconstruction)
            except ValueError as error:
                raise ValueError("presentation asset path is outside current page") from error
            digest = reference["sha256"]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("presentation asset sha256 is invalid")
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("presentation manifest component ids are duplicated")
    if expected_component_ids is not None and component_ids != expected_component_ids:
        raise ValueError("presentation manifest components do not match graph")
    return manifest


def _verify_presentation_manifest_assets(
    manifest: dict,
    reconstruction: Path,
    *,
    run_root: Path | None = None,
) -> None:
    run_root = reconstruction.parents[2] if run_root is None else run_root
    for component in manifest["components"]:
        for field in (
            "rgba", "ownership_mask", "presentation_alpha_mask",
            "generated_underlay_mask",
        ):
            reference = component[field]
            asset_path = run_root / Path(
                *PurePosixPath(reference["path"]).parts
            )
            payload = _read_bound_file(
                asset_path,
                reconstruction,
                max_bytes=PRESENTATION_ASSET_LIMIT,
                label=f"presentation asset {component['component_id']}/{field}",
            )
            if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
                raise RuntimeError(
                    f"presentation asset hash mismatch: "
                    f"{component['component_id']}/{field}"
                )


def _validate_presentation_manifest(
    manifest_path: Path,
    reconstruction: Path,
    *,
    source_sha256: str,
    graph_sha256: str,
    run_root: Path | None = None,
    expected_component_ids: list[str] | None = None,
    expected_sha256: str | None = None,
) -> dict:
    payload = _read_bound_file(
        manifest_path,
        reconstruction,
        max_bytes=GRAPH_JSON_LIMIT,
        label="presentation manifest JSON",
    )
    if (
        expected_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise RuntimeError("presentation manifest sha256 mismatch")
    manifest = _validate_presentation_manifest_payload(
        payload,
        reconstruction,
        source_sha256=source_sha256,
        graph_sha256=graph_sha256,
        run_root=run_root,
        expected_component_ids=expected_component_ids,
    )
    _verify_presentation_manifest_assets(
        manifest,
        reconstruction,
        run_root=run_root,
    )
    return manifest


def _decode_quality_presentation_image(
    payload: bytes,
    *,
    page_shape: tuple[int, int],
    channels: int,
    label: str,
):
    import cv2
    import numpy as np

    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{label} is not a PNG image")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    expected_shape = page_shape if channels == 1 else (*page_shape, channels)
    if image is None or image.dtype != np.uint8 or image.shape != expected_shape:
        raise ValueError(f"{label} dimensions or dtype are invalid")
    return image


def _iter_quality_presentation_layers(
    *,
    run_root: Path,
    reconstruction: Path,
    manifest: dict,
    page_shape: tuple[int, int],
):
    import numpy as np
    from image2editable.component_quality import _validate_underlay_metrics

    max_bytes = max(1024 * 1024, page_shape[0] * page_shape[1] * 8)
    for component in manifest["components"]:
        arrays = {}
        for name in (
            "ownership_mask",
            "presentation_alpha_mask",
            "generated_underlay_mask",
        ):
            reference = component[name]
            path = run_root / Path(*PurePosixPath(reference["path"]).parts)
            payload = _read_bound_file(
                path,
                reconstruction,
                max_bytes=max_bytes,
                label=f"component quality {name}",
            )
            if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
                raise RuntimeError(
                    f"presentation asset hash mismatch: "
                    f"{component['component_id']}/{name}"
                )
            mask = _decode_quality_presentation_image(
                payload,
                page_shape=page_shape,
                channels=1,
                label=f"component quality {name}",
            )
            if not np.all((mask == 0) | (mask == 255)):
                raise ValueError("presentation asset masks must be binary")
            arrays[name] = mask == 255
            del payload, mask

        rgba_reference = component["rgba"]
        rgba_path = run_root / Path(*PurePosixPath(rgba_reference["path"]).parts)
        rgba_payload = _read_bound_file(
            rgba_path,
            reconstruction,
            max_bytes=max_bytes,
            label="component quality RGBA",
        )
        if hashlib.sha256(rgba_payload).hexdigest() != rgba_reference["sha256"]:
            raise RuntimeError(
                f"presentation asset hash mismatch: "
                f"{component['component_id']}/rgba"
            )
        rgba = _decode_quality_presentation_image(
            rgba_payload,
            page_shape=page_shape,
            channels=4,
            label="component quality RGBA",
        )
        ownership = arrays["ownership_mask"]
        alpha = arrays["presentation_alpha_mask"]
        generated = arrays["generated_underlay_mask"]
        if not np.all((rgba[:, :, 3] == 0) | (rgba[:, :, 3] == 255)):
            raise ValueError("presentation RGBA alpha must be binary")
        if np.any(ownership & generated):
            raise ValueError(
                "presentation ownership and generated underlay masks overlap"
            )
        if not np.array_equal(alpha, ownership | generated):
            raise ValueError("presentation asset mask alpha union is invalid")
        if not np.array_equal(rgba[:, :, 3] == 255, alpha):
            raise ValueError("presentation RGBA alpha does not match alpha mask")
        if np.any(rgba[~alpha, :3]):
            raise ValueError("presentation transparent RGB must be zero")
        yield {
            "component_id": component["component_id"],
            **arrays,
            "metrics": _validate_underlay_metrics(component["metrics"]),
        }
        del rgba_payload, rgba, arrays


def _presentation_component_ids(graph: dict) -> list[str]:
    validated = validate_component_graph(graph)
    return [
        node["id"] for node in validated["nodes"]
        if node["kind"] != "text"
        and node["state"] in {"pending", "pending_gate", "frozen"}
    ]


def _quality_history_component_ids(
    graph: dict, *, parent_fallback: bool,
) -> list[str]:
    if not parent_fallback:
        return _presentation_component_ids(graph)
    validated = validate_component_graph(graph)
    return [
        node["id"] for node in validated["nodes"]
        if node["kind"] != "text"
    ]


def _validate_frozen_presentation_assets(
    previous_manifest: dict,
    current_manifest: dict,
    *,
    frozen_ids: set[str],
) -> None:
    if not frozen_ids:
        return
    previous = {
        component["component_id"]: component
        for component in previous_manifest["components"]
    }
    current = {
        component["component_id"]: component
        for component in current_manifest["components"]
    }
    for component_id in sorted(frozen_ids):
        if component_id not in previous or component_id not in current:
            raise ValueError(
                f"frozen presentation component is missing: {component_id}"
            )
        for field in (
            "rgba", "ownership_mask", "presentation_alpha_mask",
            "generated_underlay_mask",
        ):
            if (
                previous[component_id][field]["sha256"]
                != current[component_id][field]["sha256"]
            ):
                raise ValueError(
                    f"frozen presentation asset changed: "
                    f"{component_id}/{field}"
                )


def _quality_presentation_manifest(store, state: dict, quality: dict) -> dict:
    reference = quality.get("input_refs", {}).get("presentation_manifest")
    graph_sha256 = quality.get("input_graph_sha256")
    if not _is_artifact_reference(reference) or not _is_sha256(graph_sha256):
        raise ValueError("previous presentation manifest binding is invalid")
    payload = _load_state_artifact(
        store.root, reference, max_bytes=GRAPH_JSON_LIMIT
    )
    return _validate_presentation_manifest_payload(
        payload,
        store.root / "pages" / state["page_id"] / "reconstruction",
        source_sha256=state["source_sha256"],
        graph_sha256=graph_sha256,
    )


def _request_presentation_manifest(
    store,
    state: dict,
    request_path: Path,
    request: dict,
) -> dict:
    reference = request["evidence"]["presentation-manifest.json"]
    path = request_path.parent / Path(*PurePosixPath(reference["path"]).parts)
    payload = _read_bound_file(
        path,
        store.root / "pages" / state["page_id"] / "reconstruction",
        max_bytes=GRAPH_JSON_LIMIT,
        label="request presentation manifest",
    )
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise RuntimeError("request presentation manifest hash mismatch")
    return _validate_presentation_manifest_payload(
        payload,
        store.root / "pages" / state["page_id"] / "reconstruction",
        source_sha256=state["source_sha256"],
        graph_sha256=request["graph_sha256"],
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_artifact_reference(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        return False
    path = value["path"]
    if type(path) is not str or not path or "\\" in path or ":" in path:
        return False
    pure = PurePosixPath(path)
    return not pure.is_absolute() and ".." not in pure.parts and _is_sha256(
        value["sha256"]
    )


def _is_component_pair_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and all(
            isinstance(pair, list)
            and len(pair) == 2
            and pair == sorted(set(pair))
            and all(type(component_id) is str for component_id in pair)
            for pair in value
        )
        and len(value) == len({tuple(pair) for pair in value})
    )


def _previous_component_reports(
    quality_evidence: dict,
    *,
    state: dict,
    request: dict,
    active_component_ids: list[str],
) -> dict[str, dict]:
    active_visual_count = len(active_component_ids)
    if state["repair_round"] == 1:
        if not isinstance(quality_evidence, dict) or set(quality_evidence) != {
            "schema_version", "phase", "text_items",
            "initial_diagnostics", "violations",
        }:
            raise ValueError("initial component quality evidence is invalid")
        if (
            quality_evidence["schema_version"] != 1
            or type(quality_evidence["schema_version"]) is not int
            or quality_evidence["phase"] != "initial_layers"
            or not isinstance(quality_evidence["text_items"], list)
            or not isinstance(quality_evidence["violations"], list)
            or any(type(value) is not str for value in quality_evidence["violations"])
        ):
            raise ValueError("initial component quality evidence is invalid")
        validate_initial_diagnostics(
            quality_evidence["initial_diagnostics"],
            source_sha256=state["source_sha256"],
        )
        return {}
    required_fields = {
        "schema_version", "page_id", "provider", "repair_round",
        "request_sha256", "input_graph_sha256", "quality_gate_version",
        "expected_component_ids", "initial_component_count",
        "initial_diagnostics", "contained_parent_pairs",
        "approved_contained_parent_pairs", "input_refs", "report",
    }
    if (
        not isinstance(quality_evidence, dict)
        or not required_fields <= set(quality_evidence)
        or not set(quality_evidence) <= required_fields | {
            "text_items", "unexplained_mask_ref",
        }
    ):
        raise ValueError("previous component quality artifact fields are invalid")
    expected_component_ids = quality_evidence.get("expected_component_ids")
    initial_component_count = quality_evidence.get("initial_component_count")
    state_failed_ids = state.get("failed_ids", [])
    valid_state_failed_ids = (
        isinstance(state_failed_ids, list)
        and all(type(value) is str for value in state_failed_ids)
    )
    state_failed_id_set = set(state_failed_ids) if valid_state_failed_ids else set()
    reopened_pair_ids = (
        _unapproved_contained_parent_ids(quality_evidence)
        if _is_component_pair_list(quality_evidence["contained_parent_pairs"])
        and _is_component_pair_list(
            quality_evidence["approved_contained_parent_pairs"]
        )
        else set()
    )
    if (
        type(quality_evidence["schema_version"]) is not int
        or quality_evidence["schema_version"] != 1
        or quality_evidence["page_id"] != state["page_id"]
        or quality_evidence["provider"] != state["provider"]
        or quality_evidence["repair_round"] != state["repair_round"] - 1
        or quality_evidence["quality_gate_version"] != state["quality_gate_version"]
        or type(expected_component_ids) is not list
        or len(expected_component_ids) != len(set(expected_component_ids))
        or not (
            set(expected_component_ids) - set(request["candidate_ids"])
        ) <= set(active_component_ids)
        or not set(request["candidate_ids"]) <= (
            set(expected_component_ids)
            | reopened_pair_ids
            | state_failed_id_set
        )
        or not valid_state_failed_ids
        or initial_component_count != state["initial_component_count"]
        or not _is_sha256(quality_evidence["request_sha256"])
        or not _is_sha256(quality_evidence["input_graph_sha256"])
        or not _is_component_pair_list(
            quality_evidence["contained_parent_pairs"]
        )
        or not _is_component_pair_list(
            quality_evidence["approved_contained_parent_pairs"]
        )
        or not isinstance(quality_evidence["input_refs"], dict)
        or frozenset(quality_evidence["input_refs"]) not in {
            _LEGACY_QUALITY_INPUT_NAMES | {"source"},
            _QUALITY_INPUT_NAMES | {"source"},
        }
        or any(
            not _is_artifact_reference(reference)
            for reference in quality_evidence["input_refs"].values()
        )
        or (
            "unexplained_mask_ref" in quality_evidence
            and not _is_artifact_reference(
                quality_evidence["unexplained_mask_ref"]
            )
        )
        or quality_evidence["input_refs"]["source"]["sha256"]
        != request["source_sha256"]
        or (
            "text_items" in quality_evidence
            and not isinstance(quality_evidence["text_items"], list)
        )
    ):
        raise ValueError("previous component quality artifact identity is invalid")
    validate_initial_diagnostics(
        quality_evidence["initial_diagnostics"],
        source_sha256=state["source_sha256"],
    )
    from image2editable.component_quality import validate_component_quality_report

    report = validate_component_quality_report(
        quality_evidence["report"],
        expected_component_ids=expected_component_ids,
        initial_component_count=initial_component_count,
        active_visual_count=max(active_visual_count, len(expected_component_ids)),
    )
    return {
        component["component_id"]: component
        for component in report["component_reports"]
    }


def _verify_quality_input_refs(
    store, state: dict, refs: object, *, request: dict, request_path: Path,
    expected_graph_sha256: str | None = None,
    expected_component_ids: list[str] | None = None,
    return_bound_inputs: bool = False,
):
    expected_names = (
        _QUALITY_INPUT_NAMES
        if state.get("quality_gate_version", 1) >= 2
        else _LEGACY_QUALITY_INPUT_NAMES
    )
    if not isinstance(refs, dict) or frozenset(refs) != expected_names:
        missing = expected_names - frozenset(refs) if isinstance(refs, dict) else set()
        if "foreground_evidence" in missing:
            raise ValueError("component quality input refs require foreground_evidence")
        raise ValueError("component quality input refs are invalid")
    bound_payloads = {}
    for name, reference in refs.items():
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("component quality input ref is invalid")
        bound_payloads[name] = _load_state_artifact(
            store.root, reference, max_bytes=GRAPH_JSON_LIMIT
        )
    native = json.loads(bound_payloads["native_check"].decode("utf-8"))
    native_fields = {
        "schema_version", "page_id", "source_sha256", "protected_native_overlap",
        "initial_diagnostics",
    }
    optional_native_fields = {"text_items", "contained_parent_pairs"}
    contained_pairs = native.get("contained_parent_pairs", []) if isinstance(native, dict) else []
    if (
        not isinstance(native, dict)
        or not (
            set(native_fields) <= set(native)
            and set(native) <= native_fields | optional_native_fields
        )
        or native["schema_version"] != 1
        or native["page_id"] != state["page_id"]
        or native["source_sha256"] != state["source_sha256"]
        or native["protected_native_overlap"] not in {"pass", "fail", "unknown"}
        or not isinstance(native.get("text_items", []), list)
        or not isinstance(contained_pairs, list)
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(component_id) is not str for component_id in pair)
            or pair != sorted(set(pair))
            for pair in contained_pairs
        )
        or len(contained_pairs) != len({tuple(pair) for pair in contained_pairs})
    ):
        raise ValueError("component native overlap check is invalid")
    evidence = request["evidence"]["quality-report.json"]
    quality_path = request_path.parent / evidence["path"]
    quality_ref, quality_payload = _artifact_reference(
        store.root, quality_path, "initial quality evidence"
    )
    if quality_ref["sha256"] != evidence["sha256"]:
        raise RuntimeError("initial quality evidence hash mismatch")
    initial_quality = json.loads(quality_payload.decode("utf-8"))
    expected = initial_quality.get("initial_diagnostics", [])
    validate_initial_diagnostics(expected, source_sha256=state["source_sha256"])
    validate_initial_diagnostics(
        native["initial_diagnostics"], source_sha256=state["source_sha256"]
    )
    if native["initial_diagnostics"] != expected:
        raise ValueError("component native diagnostics do not match request evidence")
    if expected_component_ids is None:
        graph = json.loads(_load_state_artifact(
            store.root, state["graph_ref"], max_bytes=GRAPH_JSON_LIMIT
        ).decode("utf-8"))
        expected_component_ids = _presentation_component_ids(graph)
    manifest = _validate_presentation_manifest_payload(
        bound_payloads["presentation_manifest"],
        store.root / "pages" / state["page_id"] / "reconstruction",
        source_sha256=state["source_sha256"],
        graph_sha256=(
            expected_graph_sha256
            or state.get("graph_ref", {}).get("sha256")
            or request["graph_sha256"]
        ),
        expected_component_ids=expected_component_ids,
    )
    if return_bound_inputs:
        return refs, bound_payloads, manifest, initial_quality
    return refs


def _approved_contained_parent_pairs(
    plan: dict, contained_parent_pairs: set[tuple[str, str]]
) -> set[tuple[str, str]]:
    accepted = {
        action["object_ids"][0]: action
        for action in plan["actions"]
        if action["action"] == "accept"
    }
    approved = set()
    for pair in contained_parent_pairs:
        actions = [accepted.get(component_id) for component_id in pair]
        if all(
            action is not None
            and action["confidence"] >= 0.92
            and all(
                component_id in action["evidence"]
                for component_id in pair
            )
            for action in actions
        ):
            approved.add(pair)
    return approved


def _carried_contained_parent_pairs(
    previous_quality: dict, contained_parent_pairs: set[tuple[str, str]]
) -> set[tuple[str, str]]:
    return set()


def _unapproved_contained_parent_ids(quality: dict) -> set[str]:
    contained = {
        tuple(pair) for pair in quality.get("contained_parent_pairs", [])
    }
    approved = {
        tuple(pair)
        for pair in quality.get("approved_contained_parent_pairs", [])
    }
    return {
        component_id
        for pair in contained - approved
        for component_id in pair
    }


def _failed_overlap_dependency_ids(report: dict, graph: dict) -> set[str]:
    frozen_ids = {
        node["id"] for node in graph["nodes"]
        if node.get("kind") != "text" and node.get("state") == "frozen"
    }
    return {
        component_id
        for item in report["component_reports"]
        if "component_overlap" in item.get("violations", [])
        for component_id in item.get("overlap_component_ids", [])
        if component_id in frozen_ids
    }


def _unowned_raster_text_check(
    diagnostics: list[dict], text_items: list[dict]
) -> str:
    for diagnostic in diagnostics:
        x1, y1, x2, y2 = diagnostic["bbox"]
        diagnostic_area = max(1, (x2 - x1) * (y2 - y1))
        covered = False
        for item in text_items:
            box = item.get("box") if isinstance(item, dict) else None
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            tx, ty, width, height = (int(value) for value in box)
            intersection = max(0, min(x2, tx + width) - max(x1, tx)) * max(
                0, min(y2, ty + height) - max(y1, ty)
            )
            if intersection / diagnostic_area >= 0.80:
                covered = True
                break
        if not covered:
            return "fail"
    return "pass"


def _recompute_quality_artifact(
    store, state: dict, *, expected_component_ids: list[str],
    quality_input_refs: dict, filename: str,
) -> dict:
    import cv2
    import numpy as np

    _load_state_artifact(
        store.root, state["current_round"]["request_ref"]
    )
    request_path = store.root / Path(
        *PurePosixPath(state["current_round"]["request_ref"]["path"]).parts
    )
    request = load_component_agent_request(request_path)
    request_dir = request_path.parent
    source_path = request_dir / request["evidence"]["source.png"]["path"]
    source_ref, source_payload = _artifact_reference(
        store.root, source_path, "component quality source"
    )
    if source_ref["sha256"] != state["source_sha256"]:
        raise RuntimeError("component quality source hash mismatch")
    (
        quality_input_refs,
        bound_quality_payloads,
        presentation_manifest,
        quality_evidence,
    ) = _verify_quality_input_refs(
        store, state, quality_input_refs, request=request, request_path=request_path,
        return_bound_inputs=True,
    )
    input_refs = {"source": source_ref, **quality_input_refs}
    payloads = {"source": source_payload}
    for name in ("background", "reconstructed", "text_mask"):
        payloads[name] = bound_quality_payloads[name]
    if "foreground_evidence" in bound_quality_payloads:
        payloads["foreground_evidence"] = bound_quality_payloads[
            "foreground_evidence"
        ]

    def decode(name: str, flags: int):
        image = cv2.imdecode(np.frombuffer(payloads[name], dtype=np.uint8), flags)
        if image is None:
            raise ValueError(f"component quality {name} is not a valid image")
        return image

    source = cv2.cvtColor(decode("source", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    background = cv2.cvtColor(
        decode("background", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )
    reconstructed = cv2.cvtColor(
        decode("reconstructed", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )
    text_mask = decode("text_mask", cv2.IMREAD_GRAYSCALE)
    material_foreground = (
        decode("foreground_evidence", cv2.IMREAD_GRAYSCALE)
        if "foreground_evidence" in payloads
        else None
    )
    plan = None
    if filename == "component-quality.json":
        input_graph = load_component_agent_graph(request_path)
        plan = json.loads(_load_state_artifact(
            store.root, state["current_round"]["plan_ref"]
        ).decode("utf-8"))
        validate_component_plan(plan, request=request, graph=input_graph)
    graph_payload = _load_state_artifact(
        store.root, state["graph_ref"], max_bytes=GRAPH_JSON_LIMIT
    )
    graph = json.loads(graph_payload.decode("utf-8"))
    graph_path = store.root / Path(*PurePosixPath(state["graph_ref"]["path"]).parts)
    presentation_layers = _iter_quality_presentation_layers(
        run_root=store.root,
        reconstruction=store.root / "pages" / state["page_id"] / "reconstruction",
        manifest=presentation_manifest,
        page_shape=source.shape[:2],
    )
    native = json.loads(bound_quality_payloads["native_check"].decode("utf-8"))
    previous_reports = _previous_component_reports(
        quality_evidence,
        state=state,
        request=request,
        active_component_ids=_quality_history_component_ids(
            graph, parent_fallback=filename == "parent-quality.json"
        ),
    )
    contained_parent_pairs = {
        tuple(pair) for pair in native.get("contained_parent_pairs", [])
    }
    approved_contained_parent_pairs = _carried_contained_parent_pairs(
        quality_evidence, contained_parent_pairs
    )
    if plan is not None:
        approved_contained_parent_pairs.update(_approved_contained_parent_pairs(
            plan, contained_parent_pairs
        ))
    checks = {
        "protected_native_overlap": native["protected_native_overlap"],
        "pptx_reopen": "unknown",
        "unowned_raster_text": _unowned_raster_text_check(
            native.get("initial_diagnostics", []),
            native.get("text_items", []),
        ),
    }
    from scripts.visual_segment import visual_difference
    visual_metrics = visual_difference(source, reconstructed, text_mask)
    report = evaluate_component_quality_round(
        source, background, reconstructed, graph,
        graph_dir=graph_path.parent, trusted_root=store.root,
        text_mask=text_mask, visual_metrics=visual_metrics, page_checks=checks,
        initial_component_count=state["initial_component_count"],
        expected_component_ids=expected_component_ids,
        previous_reports=previous_reports,
        presentation_layers=presentation_layers,
        text_items=native.get("text_items", []),
        contained_parent_pairs=contained_parent_pairs,
        approved_contained_parent_pairs=approved_contained_parent_pairs,
        material_foreground=material_foreground,
        unexplained_output_path=(
            graph_path.parent / "unexplained-mask.png"
            if material_foreground is not None
            else None
        ),
    )
    unexplained_mask_ref = None
    if material_foreground is not None:
        unexplained_mask_path = graph_path.parent / "unexplained-mask.png"
        unexplained_mask_ref, _ = _artifact_reference(
            store.root, unexplained_mask_path, "unexplained visual mask"
        )
    quality = {
        "schema_version": 1, "page_id": state["page_id"],
        "provider": state["provider"], "repair_round": state["repair_round"],
        "request_sha256": state["current_round"]["request_ref"]["sha256"],
        "input_graph_sha256": state["graph_ref"]["sha256"],
        "quality_gate_version": state["quality_gate_version"],
        "expected_component_ids": expected_component_ids,
        "initial_component_count": state["initial_component_count"],
        "initial_diagnostics": native["initial_diagnostics"],
        "contained_parent_pairs": [
            list(pair) for pair in sorted(contained_parent_pairs)
        ],
        "approved_contained_parent_pairs": [
            list(pair) for pair in sorted(approved_contained_parent_pairs)
        ],
        "input_refs": input_refs, "report": report,
    }
    if unexplained_mask_ref is not None:
        quality["unexplained_mask_ref"] = unexplained_mask_ref
    if native.get("text_items"):
        quality["text_items"] = native["text_items"]
    payload = json.dumps(
        quality, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    target = graph_path.parent / filename
    if target.exists():
        existing = _read_bound_file(
            target, store.root, max_bytes=REQUEST_JSON_LIMIT,
            label="component quality artifact",
        )
        if existing != payload:
            raise RuntimeError("component quality artifact already differs")
    else:
        _write_exclusive(target, payload, store.root)
    return {
        "path": target.resolve().relative_to(store.root.resolve()).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _fallback_parent_id(node: dict, parent_assets: dict) -> str | None:
    if node["kind"] != "parent":
        parent_id = node["parent_id"]
    else:
        parent_id = node["id"]
        paired_id = node["id"].replace("component_", "parent_", 1)
        if parent_id not in parent_assets and paired_id in parent_assets:
            parent_id = paired_id
    return parent_id if parent_id in parent_assets else None


def _commit_fallback_required(store, state: dict, page_id: str, reason: str) -> dict:
    graph = json.loads(_load_state_artifact(
        store.root, state["graph_ref"], max_bytes=GRAPH_JSON_LIMIT
    ).decode("utf-8"))
    by_id = {node["id"]: node for node in graph["nodes"]}
    parent_ids = set()
    for component_id in state["failed_ids"]:
        node = by_id[component_id]
        parent_id = _fallback_parent_id(node, state["parent_assets"])
        if parent_id is not None:
            parent_ids.add(parent_id)
    updated = dict(state)
    updated["phase"] = "fallback_required"
    updated["stop_reason"] = reason
    updated["fallback"] = {
        "status": "required", "parent_ids": sorted(parent_ids)
    }
    updated["revision"] += 1
    updated["updated_at"] = _utc_now()
    validate_component_repair_state(updated)
    store.write_json(
        f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}", updated
    )
    return {"status": "fallback_required", "page_id": page_id,
            "repair_round": state["repair_round"], "stop_reason": reason}


def _commit_component_freeze(store, state: dict, page_id: str) -> dict:
    quality = json.loads(_load_state_artifact(
        store.root, state["current_round"]["quality_ref"]
    ).decode("utf-8"))
    report = quality["report"]
    page_violations = set(report.get("violations", [])) - {"pptx_reopen_unknown"}
    accepted = {
        item["component_id"] for item in report["component_reports"]
        if item.get("accepted") is True and not item.get("violations")
    }
    contained_review_ids = _unapproved_contained_parent_ids(quality)
    accepted.difference_update(contained_review_ids)
    graph_payload = _load_state_artifact(
        store.root, state["graph_ref"], max_bytes=GRAPH_JSON_LIMIT
    )
    graph = json.loads(graph_payload.decode("utf-8"))
    failed = sorted(
        (set(state["candidate_ids"]) - accepted) | contained_review_ids
    )
    failed = sorted(
        set(failed) | _failed_overlap_dependency_ids(report, graph)
    )
    fixable_page_violations = page_violations - {"unowned_raster_text"}
    if fixable_page_violations:
        residual_owner_ids = _page_residual_owner_ids(
            store,
            quality=quality,
            graph=graph,
            graph_root=(
                store.root
                / Path(*PurePosixPath(state["graph_ref"]["path"]).parts)
            ).parent,
        )
        if residual_owner_ids:
            failed = sorted(set(failed) | residual_owner_ids)
        elif not failed:
            accepted.clear()
    failed = sorted(
        set(failed)
        | (set(state["candidate_ids"]) - accepted)
        | contained_review_ids
    )
    accepted.difference_update(failed)
    reopened_frozen_ids = {
        node["id"] for node in graph["nodes"]
        if node["id"] in failed and node["state"] == "frozen"
    }
    for node in graph["nodes"]:
        if node["id"] in accepted:
            node["state"] = "frozen"
        elif node["id"] in failed and node["state"] in {
            "pending", "pending_gate", "frozen",
        }:
            node["state"] = "pending"
    from image2editable.component_contracts import validate_graph_transition
    validate_graph_transition(
        before=json.loads(graph_payload.decode("utf-8")),
        after=graph,
        allowed_reactivated_ids=reopened_frozen_ids,
    )
    frozen_payload = json.dumps(
        graph, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    graph_path = store.root / Path(*PurePosixPath(state["graph_ref"]["path"]).parts)
    frozen_path = graph_path.with_name("component-graph-frozen.json")
    if frozen_path.exists():
        existing = _read_bound_file(
            frozen_path, store.root, max_bytes=GRAPH_JSON_LIMIT,
            label="frozen component graph",
        )
        if existing != frozen_payload:
            raise RuntimeError("frozen component graph already differs")
    else:
        _write_exclusive(frozen_path, frozen_payload, store.root)
    graph_ref = {
        "path": frozen_path.resolve().relative_to(store.root.resolve()).as_posix(),
        "sha256": hashlib.sha256(frozen_payload).hexdigest(),
    }
    frozen = dict(state["frozen"])
    for component_id in reopened_frozen_ids:
        frozen.pop(component_id, None)
    for node in graph["nodes"]:
        if node["state"] == "frozen" and node["kind"] != "text":
            frozen[node["id"]] = node["mask_sha256"]
    history = list(state["round_history"])
    history[-1] = dict(history[-1])
    history[-1]["quality_sha256"] = state["current_round"]["quality_ref"]["sha256"]
    history[-1]["frozen_ids"] = sorted(accepted)
    history[-1]["failed_ids"] = failed
    updated = dict(state)
    updated["graph_ref"] = graph_ref
    updated["frozen"] = frozen
    updated["candidate_ids"] = failed
    updated["failed_ids"] = failed
    updated["round_history"] = history
    updated["phase"] = "freeze_committed"
    updated["revision"] += 1
    updated["updated_at"] = _utc_now()
    validate_component_repair_state(updated)
    store.write_json(
        f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}", updated
    )
    return {"status": "freeze_committed", "page_id": page_id,
            "repair_round": state["repair_round"], "frozen_ids": sorted(accepted),
            "failed_ids": failed}


def _page_residual_owner_ids(
    store, *, quality: dict, graph: dict, graph_root: Path
) -> set[str]:
    reference = quality.get("unexplained_mask_ref")
    if reference is None:
        return set()
    import cv2
    import numpy as np

    payload = _load_state_artifact(
        store.root, reference, max_bytes=PRESENTATION_ASSET_LIMIT
    )
    residual = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
    )
    if residual is None or residual.dtype != np.uint8 or residual.ndim != 2:
        raise ValueError("unexplained visual mask is invalid")
    residual = residual > 0
    if not np.any(residual):
        return set()
    trusted_chain = _snapshot_quality_directory_chain(graph_root, store.root)
    owners = set()
    active_nodes = []
    for node in graph["nodes"]:
        if node["kind"] == "text" or node["state"] not in {
            "pending", "pending_gate", "frozen",
        }:
            continue
        mask = _load_quality_graph_mask(
            node,
            graph_root=graph_root,
            trusted_chain=trusted_chain,
            shape=residual.shape,
        )
        active_nodes.append((node, mask))
    region_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8), 8
    )
    for label in range(1, region_count):
        region = labels == label
        nearby = cv2.dilate(
            region.astype(np.uint8), np.ones((9, 9), dtype=np.uint8)
        ) > 0
        adjacent = []
        for node, mask in active_nodes:
            overlap = int(np.count_nonzero(mask & nearby))
            minimum_overlap = max(1, round(np.count_nonzero(mask) * 0.0002))
            if overlap >= minimum_overlap:
                adjacent.append(node["id"])
        if adjacent:
            owners.update(adjacent)
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        containing = [
            node for node, _ in active_nodes
            if node["bbox"][0] <= x
            and node["bbox"][1] <= y
            and node["bbox"][2] >= x + width
            and node["bbox"][3] >= y + height
        ]
        if containing:
            owner = min(
                containing,
                key=lambda node: (
                    (node["bbox"][2] - node["bbox"][0])
                    * (node["bbox"][3] - node["bbox"][1]),
                    -node["z_index"],
                    node["id"],
                ),
            )
            owners.add(owner["id"])
    return owners


def _blocking_page_quality_violations(store, state: dict) -> set[str]:
    quality_ref = state["current_round"].get("quality_ref")
    if quality_ref is None:
        return set()
    quality = json.loads(_load_state_artifact(
        store.root, quality_ref
    ).decode("utf-8"))
    return set(quality.get("report", {}).get("violations", [])) - {
        "pptx_reopen_unknown"
    }


def _repairable_page_quality_violations(store, state: dict) -> list[str]:
    return sorted(
        _blocking_page_quality_violations(store, state)
        & _REPAIRABLE_PAGE_VIOLATIONS
    )


def _page_progress_key(quality: dict) -> tuple[float, ...]:
    visual = quality.get("visual_metrics")
    violation_items = quality.get("violations")
    if not isinstance(visual, dict) or not isinstance(violation_items, list):
        raise ValueError("component quality report is invalid for progress check")
    violations = set(violation_items)
    return (
        float(visual.get("largest_unexplained_region_pixels", 0)),
        float(visual.get("unexplained_visual_pixels", 0)),
        float(len(violations - {"pptx_reopen_unknown"})),
        float("background_text_residual" in violations),
        float("component_text_residual" in violations),
        float("duplicate_pixels" in violations),
        float("over_merged_component" in violations),
        float(visual["mae"]),
        float(visual["p95"]),
    )


def _page_quality_progressed(store, state: dict) -> bool:
    if state["repair_round"] == 1:
        return True
    if (
        state["round_history"]
        and state["round_history"][-1]["round"] == state["repair_round"]
        and state["round_history"][-1]["frozen_ids"]
    ):
        return True
    request_path = store.root / Path(
        *PurePosixPath(state["current_round"]["request_ref"]["path"]).parts
    )
    request = load_component_agent_request(request_path)
    if set(state["failed_ids"]) != set(request["candidate_ids"]):
        return True
    reference = request["evidence"]["quality-report.json"]
    previous_path = request_path.parent / Path(
        *PurePosixPath(reference["path"]).parts
    )
    previous_payload = _read_bound_file(
        previous_path,
        store.root,
        max_bytes=REQUEST_JSON_LIMIT,
        label="previous component quality",
    )
    if hashlib.sha256(previous_payload).hexdigest() != reference["sha256"]:
        raise ValueError("previous component quality sha256 mismatch")
    current = json.loads(_load_state_artifact(
        store.root, state["current_round"]["quality_ref"]
    ).decode("utf-8"))
    previous = json.loads(previous_payload.decode("utf-8"))
    previous_report = previous.get("report")
    current_report = current.get("report")
    if not isinstance(previous_report, dict) or not isinstance(current_report, dict):
        raise ValueError("component quality report is invalid for progress check")
    return _page_progress_key(current_report) < _page_progress_key(previous_report)


def _next_round_progress_allowed(store, state: dict) -> bool:
    return state.get("stop_reason") in {
        "round_limit", "no_quality_improvement",
    } or _page_quality_progressed(store, state)


def _commit_ready_result(store, state: dict, page_id: str) -> dict:
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    quality_ref = (
        state["fallback_quality_ref"]
        if state["fallback"]["status"] == "parent_preserved"
        else state["current_round"]["quality_ref"]
    )
    quality = json.loads(_load_state_artifact(
        store.root, quality_ref
    ).decode("utf-8"))
    result = {
        "schema_version": 1, "page_id": page_id, "status": "ready_for_assembly",
        "provider": state["provider"], "repair_rounds": state["plan_count"],
        "initial_component_count": state["initial_component_count"],
        "final_component_ids": sorted(state["frozen"]),
        "graph_ref": state["graph_ref"], "round_history": state["round_history"],
        "accepted_graph_sha256": quality["input_graph_sha256"],
        "fallback": state["fallback"],
        "accepted_asset_refs": quality["input_refs"],
        "text_items": quality.get("text_items", []),
        "raster_text_preserved": False,
        "warning": None,
        "delivery_checks": {"pptx_reopen": "unknown"},
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path = reconstruction / "component_result.json"
    if path.exists():
        if _read_bound_file(path, store.root, max_bytes=GRAPH_JSON_LIMIT,
                            label="component result") != payload:
            raise RuntimeError("component result already differs")
    else:
        _write_exclusive(path, payload, store.root)
    updated = dict(state)
    updated["phase"] = "ready_for_assembly"
    updated["status"] = "ready_for_assembly"
    updated["result_ref"] = {
        "path": path.resolve().relative_to(store.root.resolve()).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    updated["revision"] += 1
    updated["updated_at"] = _utc_now()
    validate_component_repair_state(updated)
    store.write_json(
        f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}", updated
    )
    return {"status": "ready_for_assembly", "page_id": page_id,
            "result_path": str(path.resolve())}


def _commit_parent_fallback_result(store, state: dict, page_id: str) -> dict:
    quality = json.loads(_load_state_artifact(
        store.root, state["fallback_quality_ref"]
    ).decode("utf-8"))
    report = quality["report"]
    allowed_page_violations = {"pptx_reopen_unknown"}
    passed = (
        set(report.get("violations", [])) <= allowed_page_violations
        and all(
            item.get("accepted") is True and not item.get("violations")
            for item in report["component_reports"]
        )
        and len(report["component_reports"]) == len(state["fallback"]["parent_ids"])
    )
    if not passed or not state["fallback"]["parent_ids"]:
        return _commit_preserved_warning(store, state, page_id)
    graph = json.loads(_load_state_artifact(
        store.root, state["graph_ref"], max_bytes=GRAPH_JSON_LIMIT
    ).decode("utf-8"))
    frozen = dict(state["frozen"])
    for node in graph["nodes"]:
        if node["id"] in state["fallback"]["parent_ids"]:
            node["state"] = "frozen"
            frozen[node["id"]] = node["mask_sha256"]
    payload = json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    current_graph = store.root / Path(*PurePosixPath(state["graph_ref"]["path"]).parts)
    final_graph = current_graph.with_name("component-graph-parent-preserved.json")
    if final_graph.exists():
        if _read_bound_file(final_graph, store.root, max_bytes=GRAPH_JSON_LIMIT,
                            label="parent preserved graph") != payload:
            raise RuntimeError("parent preserved graph already differs")
    else:
        _write_exclusive(final_graph, payload, store.root)
    updated = dict(state)
    updated["graph_ref"] = {
        "path": final_graph.resolve().relative_to(store.root.resolve()).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    updated["frozen"] = frozen
    updated["candidate_ids"] = []
    updated["failed_ids"] = []
    updated["fallback"] = {
        "status": "parent_preserved",
        "parent_ids": list(state["fallback"]["parent_ids"]),
    }
    return _commit_ready_result(store, updated, page_id)


def _commit_preserved_warning(store, state: dict, page_id: str) -> dict:
    updated = dict(state)
    updated["phase"] = "preserved_with_warning"
    updated["status"] = "preserved_with_warning"
    updated["fallback"] = {"status": "warning", "parent_ids": []}
    updated["revision"] += 1
    updated["updated_at"] = _utc_now()
    validate_component_repair_state(updated)
    store.write_json(
        f"pages/{page_id}/reconstruction/{COMPONENT_STATE_NAME}", updated
    )
    return {"status": "preserved_with_warning", "page_id": page_id}


def _load_state_artifact(
    root: Path, reference: dict, *, max_bytes: int = REQUEST_JSON_LIMIT
) -> bytes:
    path = _contained_path(root / Path(*PurePosixPath(reference["path"]).parts), root)
    payload = _read_bound_file(path, root, max_bytes=max_bytes,
                               label="component repair artifact")
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise RuntimeError("component repair artifact hash mismatch")
    return payload


def _utc_now() -> str:
    from image2editable.contracts import utc_now
    return utc_now()


def _normalized_plan_sha256(plan: dict) -> str:
    planned_ids = {
        component_id
        for action in plan["actions"]
        for component_id in action["object_ids"]
    }
    actions = [
        {
            "action": action["action"],
            "object_ids": sorted(action["object_ids"]),
            "parameters": action["parameters"],
            "approval_evidence_ids": (
                sorted(planned_ids & set(action["evidence"]))
                if action["action"] == "accept" and action["confidence"] >= 0.92
                else []
            ),
        }
        for action in plan["actions"]
    ]
    actions.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
    payload = json.dumps(actions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def execute_component_action_round(
    image,
    graph: dict,
    actions: list[dict],
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    sam_runner=None,
    sam_batch_runner=None,
) -> dict:
    from scripts.visual_segment import execute_component_actions

    return execute_component_actions(
        image,
        graph,
        actions,
        input_dir=input_dir,
        output_dir=output_dir,
        sam_runner=sam_runner,
        sam_batch_runner=sam_batch_runner,
    )


def evaluate_component_quality_round(
    source,
    background,
    reconstructed,
    graph: dict,
    *,
    graph_dir: str | Path,
    trusted_root: str | Path,
    text_mask,
    visual_metrics: dict,
    page_checks: dict,
    initial_component_count: int,
    expected_component_ids: list[str],
    previous_reports: dict | None = None,
    agent_confidence_by_id: dict | None = None,
    over_merged_component_ids: set[str] | None = None,
    contained_parent_pairs: set[tuple[str, str]] | None = None,
    approved_contained_parent_pairs: set[tuple[str, str]] | None = None,
    presentation_layers=None,
    text_items: list[dict] | None = None,
    material_foreground=None,
    unexplained_output_path: str | Path | None = None,
) -> dict:
    import cv2
    import numpy as np

    from image2editable.component_quality import (
        absorbed_leaf_cluster_count,
        calibrate_page,
        contained_active_parent_pairs,
        evaluate_component,
        evaluate_page_quality,
        material_ownership_metrics,
        refine_material_foreground,
        resolve_visual_mask_ownership,
        _strict_binary_mask,
        _validate_presentation_mask_union,
        _validate_underlay_metrics,
        _prepare_page_quality_context,
    )
    validated = validate_component_graph(graph)
    calibration = calibrate_page(source, text_mask)
    previous_reports = previous_reports or {}
    agent_confidence_by_id = agent_confidence_by_id or {}
    over_merged_component_ids = over_merged_component_ids or set()
    contained_parent_pairs = {
        tuple(sorted(pair)) for pair in (contained_parent_pairs or set())
    }
    approved_contained_parent_pairs = {
        tuple(sorted(pair))
        for pair in (approved_contained_parent_pairs or set())
    }
    reports = []
    active_visual = [
        node for node in validated["nodes"]
        if node["kind"] != "text"
        and node["state"] in {"pending", "pending_gate", "frozen"}
    ]
    candidates = [
        node for node in active_visual
        if node["state"] in {"pending", "pending_gate"}
    ]
    candidate_ids = [node["id"] for node in candidates]
    if sorted(candidate_ids) != sorted(expected_component_ids):
        raise ValueError("component quality graph IDs do not match expected IDs")
    graph_root = Path(graph_dir)
    directory_chain = _snapshot_quality_directory_chain(
        graph_root,
        Path(trusted_root),
    )
    nodes_by_id = {node["id"]: node for node in validated["nodes"]}
    shape = source.shape[:2]
    packed_layers = None
    masks = None
    if presentation_layers is None:
        mask_nodes = list(active_visual)
        loaded_ids = {node["id"] for node in mask_nodes}
        for node in candidates:
            parent_id = node.get("parent_id")
            if parent_id is not None and parent_id not in loaded_ids:
                mask_nodes.append(nodes_by_id[parent_id])
                loaded_ids.add(parent_id)
        masks = {}
        for node in mask_nodes:
            masks[node["id"]] = _load_quality_graph_mask(
                node, graph_root=graph_root, trusted_chain=directory_chain,
                shape=shape,
            )
        contained_parent_pairs.update(contained_active_parent_pairs(
            active_visual, [masks[node["id"]] for node in active_visual]
        ))
        effective_masks = resolve_visual_mask_ownership(
            active_visual, [masks[node["id"]] for node in active_visual]
        )
        for node, mask in zip(active_visual, effective_masks, strict=True):
            masks[node["id"]] = mask
    else:
        packed_layers = {}
        layer_iterator = iter(presentation_layers)
        for node in active_visual:
            try:
                layer = next(layer_iterator)
            except StopIteration as error:
                raise ValueError("presentation layer stream ended early") from error
            if not isinstance(layer, dict) or set(layer) != {
                "component_id", "ownership_mask", "presentation_alpha_mask",
                "generated_underlay_mask", "metrics",
            }:
                raise ValueError("presentation quality layer fields are invalid")
            if layer["component_id"] != node["id"]:
                raise ValueError("presentation layer stream order does not match graph")
            ownership = _strict_binary_mask(
                layer["ownership_mask"], shape, "component ownership mask"
            )
            alpha = _strict_binary_mask(
                layer["presentation_alpha_mask"], shape,
                "presentation alpha mask",
            )
            generated = _strict_binary_mask(
                layer["generated_underlay_mask"], shape,
                "generated underlay mask",
            )
            _validate_presentation_mask_union(
                ownership, alpha, generated, label="component presentation"
            )
            packed_layers[node["id"]] = {
                "ownership_mask": np.packbits(ownership, axis=None).tobytes(),
                "presentation_alpha_mask": np.packbits(alpha, axis=None).tobytes(),
                "generated_underlay_mask": np.packbits(generated, axis=None).tobytes(),
                "metrics": _validate_underlay_metrics(layer["metrics"]),
                "ownership_pixels": int(np.count_nonzero(ownership)),
            }
            del layer, ownership, alpha, generated
        try:
            next(layer_iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("presentation layer stream has extra components")

        def unpack(component_id: str, name: str):
            packed = np.frombuffer(packed_layers[component_id][name], dtype=np.uint8)
            return np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)

        parents = [
            node for node in active_visual
            if node["kind"] == "parent"
            and packed_layers[node["id"]]["ownership_pixels"]
        ]
        for left_index, left in enumerate(parents):
            left_mask = unpack(left["id"], "ownership_mask")
            left_area = packed_layers[left["id"]]["ownership_pixels"]
            for right in parents[left_index + 1:]:
                ix1 = max(left["bbox"][0], right["bbox"][0])
                iy1 = max(left["bbox"][1], right["bbox"][1])
                ix2 = min(left["bbox"][2], right["bbox"][2])
                iy2 = min(left["bbox"][3], right["bbox"][3])
                if ix1 >= ix2 or iy1 >= iy2:
                    continue
                right_area = packed_layers[right["id"]]["ownership_pixels"]
                right_mask = unpack(right["id"], "ownership_mask")
                overlap = int(np.count_nonzero(
                    left_mask[iy1:iy2, ix1:ix2]
                    & right_mask[iy1:iy2, ix1:ix2]
                ))
                del right_mask
                if overlap / min(left_area, right_area) >= 0.95:
                    contained_parent_pairs.add(tuple(sorted((
                        left["id"], right["id"]
                    ))))
            del left_mask

    contained_parent_review_ids = {
        component_id
        for pair in contained_parent_pairs - approved_contained_parent_pairs
        for component_id in pair
    }

    def component_ownership(component_id: str):
        if packed_layers is None:
            return masks[component_id]
        return unpack(component_id, "ownership_mask")

    def related_inactive_sources(target: dict, target_mask):
        tx1, ty1, tx2, ty2 = target["bbox"]
        for source_node in validated["nodes"]:
            if source_node["kind"] == "text" or source_node["state"] != "inactive":
                continue
            if not _is_related_inactive_source(target, source_node, nodes_by_id):
                continue
            sx1, sy1, sx2, sy2 = source_node["bbox"]
            ix1, iy1 = max(tx1, sx1), max(ty1, sy1)
            ix2, iy2 = min(tx2, sx2), min(ty2, sy2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            source_mask = _load_quality_graph_mask(
                source_node, graph_root=graph_root, trusted_chain=directory_chain,
                shape=source.shape[:2],
            )
            source_area = int(np.count_nonzero(source_mask))
            covered = int(np.count_nonzero(
                source_mask[iy1:iy2, ix1:ix2]
                & target_mask[iy1:iy2, ix1:ix2]
            ))
            if covered / max(source_area, 1) >= 0.8:
                yield source_mask & target_mask

    for node in candidates:
        target_mask = component_ownership(node["id"])
        if absorbed_leaf_cluster_count(
            related_inactive_sources(node, target_mask), calibration
        ) > 1:
            over_merged_component_ids.add(node["id"])
        if packed_layers is not None:
            del target_mask
    page_context = _prepare_page_quality_context(
        source, background, reconstructed, text_mask,
        calibration=calibration,
        component_masks=(
            component_ownership(node["id"])
            for node in active_visual
        ),
        text_items=text_items,
    )
    page_checks = dict(page_checks)
    text_pixels = int(np.count_nonzero(page_context.text_ink))
    residual_pixels = page_context.background_text_residual_ratio * text_pixels
    if residual_pixels >= max(
        calibration.min_component_pixels,
        calibration.text_halo_px * 2,
    ):
        page_checks["background_text_clean"] = "fail"
    else:
        page_checks["background_text_clean"] = "pass"
    if text_items is not None:
        text_node_count = sum(
            node["kind"] == "text" and node["state"] == "frozen"
            for node in validated["nodes"]
        )
        page_checks["editable_text_once"] = (
            "pass"
            if (
                (not np.any(page_context.text) and not text_items)
                or (
                    np.any(page_context.text)
                    and len(text_items) == text_node_count
                    and text_node_count > 0
                )
            )
            else "fail"
        )
    if material_foreground is not None:
        material_foreground = refine_material_foreground(
            material_foreground, source, background, calibration
        )
        ownership_metrics, unexplained = material_ownership_metrics(
            material_foreground,
            (
                component_ownership(node["id"])
                for node in active_visual
            ),
            text_mask,
            calibration,
            generated_underlay_masks=(
                unpack(node["id"], "generated_underlay_mask")
                for node in active_visual
            ) if packed_layers is not None else (),
        )
        visual_metrics = {**visual_metrics, **ownership_metrics}
        page_checks["visual_ownership"] = (
            "pass"
            if ownership_metrics["unexplained_visual_pixels"] == 0
            else "fail"
        )
        if unexplained_output_path is not None:
            if not cv2.imwrite(
                str(unexplained_output_path),
                unexplained.astype(np.uint8) * 255,
            ):
                raise RuntimeError("could not write unexplained visual mask")
    for node in candidates:
        component_id = node["id"]
        component_mask = component_ownership(component_id)
        overlap_component_ids = []
        for other in active_visual:
            if other["id"] == component_id:
                continue
            ix1 = max(node["bbox"][0], other["bbox"][0])
            iy1 = max(node["bbox"][1], other["bbox"][1])
            ix2 = min(node["bbox"][2], other["bbox"][2])
            iy2 = min(node["bbox"][3], other["bbox"][3])
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            other_mask = component_ownership(other["id"])
            if np.any(
                component_mask[iy1:iy2, ix1:ix2]
                & other_mask[iy1:iy2, ix1:ix2]
            ):
                overlap_component_ids.append(other["id"])
            if packed_layers is not None:
                del other_mask
        previous = previous_reports.get(component_id, {})
        presentation_kwargs = {}
        if packed_layers is None:
            parent_mask = masks.get(node.get("parent_id"))
        else:
            semantic_id = node.get("parent_id") or component_id
            parent_mask = _load_quality_graph_mask(
                nodes_by_id[semantic_id],
                graph_root=graph_root,
                trusted_chain=directory_chain,
                shape=shape,
            )
            presentation_kwargs = {
                "presentation_alpha_mask": unpack(
                    component_id, "presentation_alpha_mask"
                ),
                "generated_underlay_mask": unpack(
                    component_id, "generated_underlay_mask"
                ),
                "underlay_metrics": packed_layers[component_id]["metrics"],
            }
        reports.append(evaluate_component(
            source,
            background,
            reconstructed,
            node,
            validated,
            calibration,
            component_mask=component_mask,
            parent_mask=parent_mask,
            **presentation_kwargs,
            text_mask=text_mask,
            page_checks=page_checks,
            agent_confidence=agent_confidence_by_id.get(component_id),
            previous_metrics=previous.get("metrics"),
            over_merged_component=component_id in over_merged_component_ids,
            contained_parent_review=component_id in contained_parent_review_ids,
            overlap_component_ids=overlap_component_ids,
            _page_context=page_context,
        ))
    return evaluate_page_quality(
        reports,
        visual_metrics=visual_metrics,
        page_checks=page_checks,
        expected_component_ids=expected_component_ids,
        initial_component_count=initial_component_count,
        active_visual_count=len(active_visual),
    )


def _is_related_inactive_source(
    target: dict, source_node: dict, nodes_by_id: dict[str, dict]
) -> bool:
    ancestor_id = source_node.get("parent_id")
    while ancestor_id is not None:
        if ancestor_id == target["id"]:
            return True
        ancestor_id = nodes_by_id[ancestor_id].get("parent_id")
    target_parent_id = target.get("parent_id")
    return (
        target_parent_id is not None
        and source_node.get("parent_id") == target_parent_id
    )


def _load_quality_graph_mask(
    node: dict,
    *,
    graph_root: Path,
    trusted_chain: list[tuple[Path, tuple[int, int, int, int]]],
    shape: tuple[int, int],
):
    from scripts.visual_segment import _read_action_mask

    mask_path = graph_root / node["mask"]
    current = graph_root
    mask_parent_chain = []
    for part in PurePosixPath(node["mask"]).parts[:-1]:
        current = current / part
        parent_status = current.lstat()
        if _is_link_or_reparse(parent_status) or not stat.S_ISDIR(parent_status.st_mode):
            raise ValueError("component quality mask parent must be a plain directory")
        mask_parent_chain.append((current, _directory_identity(parent_status)))
    component_mask, _ = _read_action_mask(
        mask_path, shape, node["mask_sha256"]
    )
    _require_directory_chain_identity(trusted_chain + mask_parent_chain)
    return component_mask


def _snapshot_quality_directory_chain(
    directory: Path,
    trusted_root: Path,
) -> list[tuple[Path, tuple[int, int, int, int]]]:
    if ".." in directory.parts or ".." in trusted_root.parts:
        raise ValueError("component quality paths contain unsafe semantic path segments")
    root = trusted_root if trusted_root.is_absolute() else Path.cwd() / trusted_root
    target = directory if directory.is_absolute() else Path.cwd() / directory
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ValueError("component quality graph_dir is outside trusted root") from error
    current = root
    identities = []
    for part in (None, *relative.parts):
        if part is not None:
            current /= part
        status = current.lstat()
        if _is_link_or_reparse(status) or not stat.S_ISDIR(status.st_mode):
            raise ValueError("component quality directory chain is unsafe")
        identities.append((current, _directory_identity(status)))
    return identities


def build_component_agent_request(
    page_session: dict,
    *,
    repair_round: int,
) -> Path:
    repair_round = validate_repair_round(repair_round)
    validated = _validate_page_session(page_session)
    reconstruction = validated[2]
    with _run_publication_lease(reconstruction):
        return _build_component_agent_request_locked(validated, repair_round)


def _build_component_agent_request_locked(
    validated: tuple[str, str, Path, dict],
    repair_round: int,
) -> Path:
    page_id, provider, reconstruction, sources = validated
    integrity_key = _load_or_create_integrity_key(reconstruction)
    agent_dir = reconstruction / "agent"
    _ensure_owned_directory(agent_dir, reconstruction)
    round_dir = agent_dir / f"round-{repair_round:02d}"
    if round_dir.exists() or round_dir.is_symlink():
        raise RuntimeError(f"Agent evidence round is already published: {round_dir}")
    staging = agent_dir / f".{round_dir.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    staging_identity = _directory_identity(staging.lstat())
    try:
        records: dict[str, dict[str, str]] = {}
        graph_payload = b""
        for name in sorted(sources):
            _require_single_directory_identity(staging, staging_identity)
            source = _contained_path(Path(sources[name]), reconstruction)
            digest, captured = _copy_bound_file(
                source,
                staging / name,
                reconstruction,
                capture_limit=(
                    GRAPH_JSON_LIMIT if name == "component-graph.json" else None
                ),
            )
            records[name] = {
                "path": name,
                "sha256": digest,
            }
            if captured is not None:
                graph_payload = captured
            _require_single_directory_identity(staging, staging_identity)
        graph = json.loads(graph_payload.decode("utf-8"))
        validate_component_graph(graph)
        graph_source = _contained_path(
            Path(sources["component-graph.json"]), reconstruction
        )
        _validate_presentation_manifest(
            staging / "presentation-manifest.json",
            reconstruction,
            source_sha256=records["source.png"]["sha256"],
            graph_sha256=records["component-graph.json"]["sha256"],
            expected_component_ids=_presentation_component_ids(graph),
        )
        request = {
            "schema_version": 1,
            "page_id": page_id,
            "provider": provider,
            "repair_round": repair_round,
            "source_sha256": records["source.png"]["sha256"],
            "graph_sha256": records["component-graph.json"]["sha256"],
            "candidate_ids": sorted(
                node["id"] for node in graph["nodes"] if node["state"] == "pending"
            ),
            "frozen_ids": sorted(
                node["id"] for node in graph["nodes"] if node["state"] == "frozen"
            ),
            "evidence": records,
        }
        validate_component_agent_request(request)
        request_bytes = json.dumps(
            request,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        if len(request_bytes) > REQUEST_JSON_LIMIT:
            raise RuntimeError("Component agent request JSON size limit exceeded")
        _require_single_directory_identity(staging, staging_identity)
        _write_exclusive(staging / REQUEST_NAME, request_bytes, reconstruction)
        _require_single_directory_identity(staging, staging_identity)
        marker_fields = {
            "schema_version": 1,
            "page_id": page_id,
            "provider": provider,
            "repair_round": repair_round,
            "request_path": f"{round_dir.name}/{REQUEST_NAME}",
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        }
        marker = {
            **marker_fields,
            "hmac_sha256": hmac.new(
                integrity_key,
                _canonical_marker_fields(marker_fields),
                hashlib.sha256,
            ).hexdigest(),
        }
        marker_bytes = json.dumps(
            marker,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        if len(marker_bytes) > MARKER_JSON_LIMIT:
            raise RuntimeError("Component agent marker JSON size limit exceeded")
        _write_exclusive(staging / MARKER_NAME, marker_bytes, reconstruction)
        _require_single_directory_identity(staging, staging_identity)
        for node in graph["nodes"]:
            mask_relative = Path(*PurePosixPath(node["mask"]).parts)
            source_mask = _contained_path(
                graph_source.parent / mask_relative, reconstruction
            )
            destination_mask = staging / mask_relative
            destination_mask.parent.mkdir(parents=True, exist_ok=True)
            digest, _ = _copy_bound_file(
                source_mask, destination_mask, reconstruction,
                capture_limit=None,
            )
            if digest != node["mask_sha256"]:
                raise RuntimeError("Component graph mask hash mismatch")
        _verify_staged_bundle(
            staging,
            reconstruction,
            staging_identity,
            records,
            request_bytes,
            marker_bytes,
        )
        agent_identity = _snapshot_directory_chain(agent_dir, reconstruction)
        _require_single_directory_identity(staging, staging_identity)
        try:
            staging.rename(round_dir)
        except OSError as error:
            raise RuntimeError(
                f"Agent evidence round is already published: {round_dir}"
            ) from error
        _require_directory_chain_identity(agent_identity)
        try:
            round_status = round_dir.lstat()
        except FileNotFoundError as error:
            raise RuntimeError("Agent evidence staging identity changed") from error
        if (
            _is_link_or_reparse(round_status)
            or not stat.S_ISDIR(round_status.st_mode)
            or _directory_identity(round_status) != staging_identity
        ):
            _remove_rejected_round(round_dir, agent_dir)
            raise RuntimeError("Agent evidence staging identity changed")
    except BaseException:
        _cleanup_owned_staging(staging, staging_identity)
        raise
    return round_dir / REQUEST_NAME


def load_component_agent_request(request_path: str | Path) -> dict:
    request_path = Path(request_path)
    if request_path.name != REQUEST_NAME:
        raise RuntimeError("Component agent request path is invalid")
    round_dir = request_path.parent
    agent_dir = round_dir.parent
    reconstruction = agent_dir.parent
    if (
        reconstruction.name != "reconstruction"
        or reconstruction.parent.parent.name != "pages"
        or agent_dir.name != "agent"
    ):
        raise RuntimeError(
            "Component agent request path must be pages/<page_id>/"
            "reconstruction/agent/round-XX"
        )
    _validate_directory_chain(round_dir, reconstruction)
    integrity_key = _load_integrity_key(reconstruction)
    marker_path = round_dir / MARKER_NAME
    try:
        marker_bytes = _read_bound_file(
            marker_path,
            reconstruction,
            max_bytes=MARKER_JSON_LIMIT,
            label="marker JSON",
        )
    except RuntimeError as error:
        if "size limit" in str(error):
            raise
        raise RuntimeError("Component agent request marker is missing or invalid") from error
    try:
        marker = json.loads(marker_bytes.decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Component agent request marker is missing or invalid") from error
    _validate_request_marker(marker)
    marker_fields = {
        key: value for key, value in marker.items() if key != "hmac_sha256"
    }
    expected_signature = hmac.new(
        integrity_key,
        _canonical_marker_fields(marker_fields),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(marker["hmac_sha256"], expected_signature):
        raise RuntimeError("Component agent publication signature mismatch")
    request_bytes = _read_bound_file(
        request_path,
        reconstruction,
        max_bytes=REQUEST_JSON_LIMIT,
        label="request JSON",
    )
    if hashlib.sha256(request_bytes).hexdigest() != marker["request_sha256"]:
        raise RuntimeError("Component agent request hash mismatch")
    request = json.loads(request_bytes.decode("utf-8"))
    validate_component_agent_request(request)
    expected_round = f"round-{request['repair_round']:02d}"
    if (
        round_dir.name != expected_round
        or reconstruction.parent.name != request["page_id"]
        or marker["page_id"] != request["page_id"]
        or marker["provider"] != request["provider"]
        or marker["repair_round"] != request["repair_round"]
        or marker["request_path"] != f"{round_dir.name}/{REQUEST_NAME}"
    ):
        raise RuntimeError("Component agent request belongs to another page or round")
    graph_payload = b""
    for name, record in request["evidence"].items():
        evidence_path = _resolve_evidence_path(record["path"], round_dir, reconstruction)
        if name == "component-graph.json":
            graph_payload = _read_bound_file(
                evidence_path,
                reconstruction,
                max_bytes=GRAPH_JSON_LIMIT,
                label="component graph JSON",
            )
            digest = hashlib.sha256(graph_payload).hexdigest()
        else:
            digest = _hash_bound_file(evidence_path, reconstruction)
        if digest != record["sha256"]:
            raise RuntimeError(f"Component evidence hash mismatch: {name}")
    if request["source_sha256"] != request["evidence"]["source.png"]["sha256"]:
        raise RuntimeError("Component source evidence hash mismatch")
    if request["graph_sha256"] != request["evidence"]["component-graph.json"]["sha256"]:
        raise RuntimeError("Component graph evidence hash mismatch")
    graph = json.loads(graph_payload.decode("utf-8"))
    validate_component_graph(graph)
    _validate_presentation_manifest(
        round_dir / request["evidence"]["presentation-manifest.json"]["path"],
        reconstruction,
        source_sha256=request["source_sha256"],
        graph_sha256=request["graph_sha256"],
        expected_component_ids=_presentation_component_ids(graph),
    )
    for node in graph["nodes"]:
        mask_path = round_dir / Path(*PurePosixPath(node["mask"]).parts)
        if _hash_bound_file(mask_path, reconstruction) != node["mask_sha256"]:
            raise RuntimeError(
                f"Component graph mask hash mismatch: {node['id']}"
            )
    candidate_ids = sorted(
        node["id"] for node in graph["nodes"] if node["state"] == "pending"
    )
    frozen_ids = sorted(
        node["id"] for node in graph["nodes"] if node["state"] == "frozen"
    )
    if request["candidate_ids"] != candidate_ids or request["frozen_ids"] != frozen_ids:
        raise RuntimeError("Component agent request component ids do not match graph")
    return request


def load_component_agent_graph(request_path: str | Path) -> dict:
    request_path = Path(request_path)
    request = load_component_agent_request(request_path)
    reconstruction = request_path.parent.parent.parent
    graph_path = request_path.parent / request["evidence"]["component-graph.json"]["path"]
    payload = _read_bound_file(
        graph_path,
        reconstruction,
        max_bytes=GRAPH_JSON_LIMIT,
        label="component graph JSON",
    )
    if hashlib.sha256(payload).hexdigest() != request["graph_sha256"]:
        raise RuntimeError("Component graph evidence hash mismatch")
    graph = json.loads(payload.decode("utf-8"))
    return validate_component_graph(graph)


def _validate_page_session(session: object) -> tuple[str, str, Path, dict]:
    fields = {"page_id", "provider", "reconstruction_dir", "evidence"}
    if not isinstance(session, dict) or set(session) != fields:
        raise ValueError("page_session fields are invalid")
    page_id = session["page_id"]
    if type(page_id) is not str or not page_id or session["reconstruction_dir"] is None:
        raise ValueError("page_session page_id is invalid")
    provider = validate_agent_provider(session["provider"])
    reconstruction = Path(session["reconstruction_dir"])
    if (
        reconstruction.name != "reconstruction"
        or reconstruction.parent.name != page_id
        or reconstruction.parent.parent.name != "pages"
    ):
        raise RuntimeError(
            "page_session path must be pages/<page_id>/reconstruction"
        )
    _validate_directory_chain(reconstruction, reconstruction)
    evidence = session["evidence"]
    if not isinstance(evidence, dict) or frozenset(evidence) not in {
        LEGACY_COMPONENT_EVIDENCE_NAMES,
        COMPONENT_EVIDENCE_NAMES,
    }:
        raise ValueError("page_session evidence fields are invalid")
    return page_id, provider, reconstruction, evidence


def _resolve_evidence_path(path: str, round_dir: Path, reconstruction: Path) -> Path:
    relative = Path(*PurePosixPath(path).parts)
    candidate = round_dir / relative
    if not candidate.exists():
        candidate = reconstruction / relative
    return _contained_path(candidate, reconstruction)


def _contained_path(path: Path, root: Path) -> Path:
    lexical = path if path.is_absolute() else Path.cwd() / path
    root_absolute = root if root.is_absolute() else Path.cwd() / root
    try:
        lexical.relative_to(root_absolute)
    except ValueError as error:
        raise RuntimeError(f"Evidence path is outside page reconstruction: {path}") from error
    _validate_directory_chain(lexical.parent, root_absolute)
    return lexical


def _validate_directory_chain(directory: Path, root: Path) -> None:
    _snapshot_directory_chain(directory, root)


def _snapshot_directory_chain(
    directory: Path,
    root: Path,
) -> list[tuple[Path, tuple[int, int, int, int]]]:
    root = root if root.is_absolute() else Path.cwd() / root
    directory = directory if directory.is_absolute() else Path.cwd() / directory
    try:
        relative = directory.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Path is outside page reconstruction: {directory}") from error
    trust_root = root.parent.parent
    current = trust_root
    relative = directory.relative_to(trust_root)
    identities = []
    for part in (Path(), *relative.parts):
        if part != Path():
            current /= part
        status = current.lstat()
        if _is_link_or_reparse(status):
            raise RuntimeError(f"Evidence directory is a link or reparse point: {current}")
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(f"Evidence parent is not a directory: {current}")
        identities.append((current, _directory_identity(status)))
    return identities


def _require_directory_chain_identity(
    identities: list[tuple[Path, tuple[int, int, int, int]]],
) -> None:
    for path, expected in identities:
        try:
            status = path.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(f"Evidence directory identity changed: {path}") from error
        if _is_link_or_reparse(status) or _directory_identity(status) != expected:
            raise RuntimeError(f"Evidence directory identity changed: {path}")


def _directory_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        getattr(status, "st_file_attributes", 0),
    )


def _require_single_directory_identity(
    directory: Path,
    expected: tuple[int, int, int, int],
) -> None:
    try:
        status = directory.lstat()
    except FileNotFoundError as error:
        raise RuntimeError("Agent evidence staging identity changed") from error
    if (
        _is_link_or_reparse(status)
        or not stat.S_ISDIR(status.st_mode)
        or _directory_identity(status) != expected
    ):
        raise RuntimeError("Agent evidence staging identity changed")


def _cleanup_owned_staging(
    staging: Path,
    expected: tuple[int, int, int, int],
) -> None:
    quarantine = staging.with_name(f".quarantine-{uuid.uuid4().hex}")
    try:
        staging.rename(quarantine)
    except OSError:
        return
    try:
        status = quarantine.lstat()
    except FileNotFoundError:
        return
    if (
        _is_link_or_reparse(status)
        or not stat.S_ISDIR(status.st_mode)
        or _directory_identity(status) != expected
    ):
        return
    try:
        _delete_owned_flat_quarantine(quarantine, expected)
    except (OSError, RuntimeError):
        return


def _delete_owned_flat_quarantine(
    quarantine: Path,
    expected: tuple[int, int, int, int],
) -> None:
    allowed = set(EVIDENCE_NAMES) | {REQUEST_NAME, MARKER_NAME}
    entries = list(quarantine.iterdir())
    if any(entry.name not in allowed for entry in entries):
        return
    tombstones = []
    for entry in entries:
        _require_single_directory_identity(quarantine, expected)
        status = entry.lstat()
        if (
            _is_link_or_reparse(status)
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
        ):
            return
        identity = (status.st_dev, status.st_ino)
        tombstone = quarantine.parent / f".delete-{uuid.uuid4().hex}"
        try:
            entry.rename(tombstone)
        except OSError:
            return
        moved = tombstone.lstat()
        if (
            _is_link_or_reparse(moved)
            or not stat.S_ISREG(moved.st_mode)
            or moved.st_nlink != 1
            or (moved.st_dev, moved.st_ino) != identity
        ):
            return
        tombstones.append((tombstone, identity))
    _require_single_directory_identity(quarantine, expected)
    try:
        quarantine.rmdir()
    except OSError:
        return
    for tombstone, identity in tombstones:
        try:
            status = tombstone.lstat()
        except FileNotFoundError:
            continue
        if (
            _is_link_or_reparse(status)
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or (status.st_dev, status.st_ino) != identity
        ):
            continue
        tombstone.unlink()


def _remove_rejected_round(round_dir: Path, agent_dir: Path) -> None:
    if round_dir.parent != agent_dir or not _is_round_name(round_dir.name):
        raise RuntimeError("Refusing to clean an invalid Agent round path")
    quarantine = agent_dir / f".quarantine-round-{uuid.uuid4().hex}"
    try:
        round_dir.rename(quarantine)
    except OSError as error:
        raise RuntimeError("Failed to quarantine a rejected Agent round") from error


@contextmanager
def _run_publication_lease(reconstruction: Path):
    run_root = reconstruction.parent.parent.parent
    deadline = time.monotonic() + 30.0
    lease = ExecutionLease(
        run_root / PUBLICATION_LOCK_NAME,
        run_root=run_root,
    )
    while True:
        try:
            lease.__enter__()
            break
        except RuntimeError as error:
            if "already executing" not in str(error) or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
    try:
        yield
    except BaseException as error:
        lease.__exit__(type(error), error, error.__traceback__)
        raise
    else:
        lease.__exit__(None, None, None)


def _verify_staged_bundle(
    staging: Path,
    reconstruction: Path,
    staging_identity: tuple[int, int, int, int],
    records: dict[str, dict[str, str]],
    request_bytes: bytes,
    marker_bytes: bytes,
) -> None:
    _require_single_directory_identity(staging, staging_identity)
    for name, record in records.items():
        if _hash_bound_file(staging / name, reconstruction) != record["sha256"]:
            raise RuntimeError(f"Staged component evidence hash mismatch: {name}")
        _require_single_directory_identity(staging, staging_identity)
    if _read_bound_file(
        staging / REQUEST_NAME,
        reconstruction,
        max_bytes=REQUEST_JSON_LIMIT,
        label="request JSON",
    ) != request_bytes:
        raise RuntimeError("Staged component request changed before publication")
    if _read_bound_file(
        staging / MARKER_NAME,
        reconstruction,
        max_bytes=MARKER_JSON_LIMIT,
        label="marker JSON",
    ) != marker_bytes:
        raise RuntimeError("Staged component marker changed before publication")
    _require_single_directory_identity(staging, staging_identity)


def _ensure_owned_directory(directory: Path, root: Path) -> None:
    try:
        directory.mkdir()
    except FileExistsError:
        pass
    _validate_directory_chain(directory, root)


def _read_bound_file(
    path: Path,
    reconstruction: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    _contained_path(path, reconstruction)
    directory_identity = _snapshot_directory_chain(path.parent, reconstruction)
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"Evidence file cannot be opened safely: {path}") from error
    with os.fdopen(descriptor, "rb") as source:
        _require_directory_chain_identity(directory_identity)
        opened = os.fstat(source.fileno())
        path_status = path.lstat()
        if _is_link_or_reparse(path_status):
            raise RuntimeError(f"Evidence file is a link or reparse point: {path}")
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(path_status.st_mode):
            raise RuntimeError(f"Evidence is not a regular file: {path}")
        if opened.st_nlink != 1 or path_status.st_nlink != 1:
            raise RuntimeError(f"Evidence file is an unsafe hard link: {path}")
        if (opened.st_dev, opened.st_ino) != (path_status.st_dev, path_status.st_ino):
            raise RuntimeError(f"Evidence file identity changed: {path}")
        chunks = []
        total = 0
        while True:
            chunk = source.read(IO_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(f"Component agent {label} size limit exceeded")
            chunks.append(chunk)
        stable = os.fstat(source.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            stable.st_dev,
            stable.st_ino,
            stable.st_size,
        ):
            raise RuntimeError(f"Evidence file changed while reading: {path}")
        _require_directory_chain_identity(directory_identity)
        return b"".join(chunks)


def _hash_bound_file(path: Path, reconstruction: Path) -> str:
    _contained_path(path, reconstruction)
    directory_identity = _snapshot_directory_chain(path.parent, reconstruction)
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"Evidence file cannot be opened safely: {path}") from error
    with os.fdopen(descriptor, "rb") as source:
        opened = _validate_open_regular_file(path, source.fileno(), directory_identity)
        digest = hashlib.sha256()
        while True:
            chunk = source.read(IO_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        _validate_stable_open_file(path, source.fileno(), opened, directory_identity)
        return digest.hexdigest()


def _copy_bound_file(
    source_path: Path,
    target_path: Path,
    reconstruction: Path,
    *,
    capture_limit: int | None,
) -> tuple[str, bytes | None]:
    _contained_path(source_path, reconstruction)
    _contained_path(target_path, reconstruction)
    source_directories = _snapshot_directory_chain(
        source_path.parent, reconstruction
    )
    target_directories = _snapshot_directory_chain(
        target_path.parent, reconstruction
    )
    read_flags = os.O_RDONLY
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        read_flags |= getattr(os, name, 0)
        write_flags |= getattr(os, name, 0)
    source_descriptor = os.open(source_path, read_flags)
    try:
        target_descriptor = os.open(target_path, write_flags, 0o600)
    except BaseException:
        os.close(source_descriptor)
        raise
    with os.fdopen(source_descriptor, "rb") as source, os.fdopen(
        target_descriptor, "wb"
    ) as target:
        source_status = _validate_open_regular_file(
            source_path, source.fileno(), source_directories
        )
        target_status = _validate_open_regular_file(
            target_path, target.fileno(), target_directories
        )
        digest = hashlib.sha256()
        captured = bytearray() if capture_limit is not None else None
        while True:
            chunk = source.read(IO_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            target.write(chunk)
            if captured is not None:
                if len(captured) + len(chunk) > capture_limit:
                    raise RuntimeError(
                        "Component agent component graph JSON size limit exceeded"
                    )
                captured.extend(chunk)
        target.flush()
        os.fsync(target.fileno())
        _validate_stable_open_file(
            source_path, source.fileno(), source_status, source_directories
        )
        _validate_stable_open_file(
            target_path,
            target.fileno(),
            target_status,
            target_directories,
            allow_size_change=True,
        )
        return digest.hexdigest(), None if captured is None else bytes(captured)


def _validate_open_regular_file(
    path: Path,
    descriptor: int,
    directory_identity: list[tuple[Path, tuple[int, int, int, int]]],
) -> os.stat_result:
    _require_directory_chain_identity(directory_identity)
    opened = os.fstat(descriptor)
    path_status = path.lstat()
    if _is_link_or_reparse(path_status):
        raise RuntimeError(f"Evidence file is a link or reparse point: {path}")
    if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(path_status.st_mode):
        raise RuntimeError(f"Evidence is not a regular file: {path}")
    if opened.st_nlink != 1 or path_status.st_nlink != 1:
        raise RuntimeError(f"Evidence file is an unsafe hard link: {path}")
    if (opened.st_dev, opened.st_ino) != (path_status.st_dev, path_status.st_ino):
        raise RuntimeError(f"Evidence file identity changed: {path}")
    return opened


def _validate_stable_open_file(
    path: Path,
    descriptor: int,
    opened: os.stat_result,
    directory_identity: list[tuple[Path, tuple[int, int, int, int]]],
    *,
    allow_size_change: bool = False,
) -> None:
    stable = os.fstat(descriptor)
    if (
        (opened.st_dev, opened.st_ino) != (stable.st_dev, stable.st_ino)
        or (not allow_size_change and opened.st_size != stable.st_size)
    ):
        raise RuntimeError(f"Evidence file changed while reading: {path}")
    _require_directory_chain_identity(directory_identity)


def _write_exclusive(path: Path, payload: bytes, reconstruction: Path) -> None:
    _contained_path(path, reconstruction)
    directory_identity = _snapshot_directory_chain(path.parent, reconstruction)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        _require_directory_chain_identity(directory_identity)
        raise RuntimeError(f"Evidence file cannot be created safely: {path}") from error
    with os.fdopen(descriptor, "wb") as target:
        _require_directory_chain_identity(directory_identity)
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
        _require_directory_chain_identity(directory_identity)


def _load_or_create_integrity_key(reconstruction: Path) -> bytes:
    run_root = reconstruction.parent.parent.parent
    return load_or_create_run_integrity_key(run_root)


def load_or_create_run_integrity_key(run_root: str | Path) -> bytes:
    run_root = Path(run_root).resolve()
    anchor = run_root / INTEGRITY_DIRECTORY
    if anchor.exists() or anchor.is_symlink():
        return _read_integrity_key(anchor, run_root)
    host_challenge = run_root / "host-challenge"
    if _run_has_published_rounds(run_root) or host_challenge.exists() or host_challenge.is_symlink():
        raise RuntimeError(
            "Run has published Agent rounds but its integrity key is missing"
        )
    staging = run_root / f".{INTEGRITY_DIRECTORY}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir(mode=0o700)
        key_path = staging / INTEGRITY_KEY_NAME
        key = secrets.token_bytes(32)
        _write_new_integrity_key(key_path, key, staging, run_root)
        try:
            staging.rename(anchor)
        except OSError:
            if staging.exists():
                shutil.rmtree(staging)
            if not anchor.exists() and not anchor.is_symlink():
                raise RuntimeError("Component agent integrity key creation failed")
        return _read_integrity_key(anchor, run_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _load_integrity_key(reconstruction: Path) -> bytes:
    run_root = reconstruction.parent.parent.parent
    return load_run_integrity_key(run_root)


def load_run_integrity_key(run_root: str | Path) -> bytes:
    run_root = Path(run_root).resolve()
    anchor = run_root / INTEGRITY_DIRECTORY
    if not anchor.exists() and not anchor.is_symlink():
        raise RuntimeError("Component agent integrity key is missing")
    return _read_integrity_key(anchor, run_root)


def _run_has_published_rounds(run_root: Path) -> bool:
    pages = run_root / "pages"
    if not pages.exists() and not pages.is_symlink():
        return False
    _require_safe_directory(pages, "pages")
    found = False
    for page in pages.iterdir():
        page_status = page.lstat()
        if _is_link_or_reparse(page_status):
            raise RuntimeError(f"Run pages entry is a link or reparse point: {page}")
        if not stat.S_ISDIR(page_status.st_mode):
            continue
        reconstruction = page / "reconstruction"
        if not reconstruction.exists() and not reconstruction.is_symlink():
            continue
        _require_safe_directory(reconstruction, "reconstruction")
        agent = reconstruction / "agent"
        if not agent.exists() and not agent.is_symlink():
            continue
        _require_safe_directory(agent, "agent")
        for child in agent.iterdir():
            if not _is_round_name(child.name):
                continue
            status = child.lstat()
            if _is_link_or_reparse(status):
                raise RuntimeError(
                    f"Published Agent round is a link or reparse point: {child}"
                )
            if not stat.S_ISDIR(status.st_mode):
                raise RuntimeError(f"Published Agent round is not a directory: {child}")
            found = True
    return found


def _require_safe_directory(path: Path, label: str) -> None:
    status = path.lstat()
    if _is_link_or_reparse(status):
        raise RuntimeError(f"Run {label} is a link or reparse point: {path}")
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"Run {label} is not a directory: {path}")


def _is_round_name(name: str) -> bool:
    return len(name) == 8 and name.startswith("round-") and name[6:] in {
        "01",
        "02",
        "03",
        "04",
        "05",
    }


def _write_new_integrity_key(
    path: Path,
    key: bytes,
    staging: Path,
    run_root: Path,
) -> None:
    directory_identity = _snapshot_key_directory_chain(staging, run_root)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        _require_directory_chain_identity(directory_identity)
        target.write(key)
        target.flush()
        os.fsync(target.fileno())
        os.chmod(path, 0o600)
        _require_directory_chain_identity(directory_identity)


def _read_integrity_key(anchor: Path, run_root: Path) -> bytes:
    directory_identity = _snapshot_key_directory_chain(anchor, run_root)
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    key_path = anchor / INTEGRITY_KEY_NAME
    try:
        descriptor = os.open(key_path, flags)
    except OSError as error:
        raise RuntimeError("Component agent integrity key cannot be opened safely") from error
    with os.fdopen(descriptor, "rb") as source:
        _require_directory_chain_identity(directory_identity)
        opened = os.fstat(source.fileno())
        path_status = key_path.lstat()
        if _is_link_or_reparse(path_status):
            raise RuntimeError("Component agent integrity key is a link or reparse point")
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(path_status.st_mode)
            or opened.st_nlink != 1
            or path_status.st_nlink != 1
        ):
            raise RuntimeError("Component agent integrity key is an unsafe hard link")
        if (opened.st_dev, opened.st_ino) != (path_status.st_dev, path_status.st_ino):
            raise RuntimeError("Component agent integrity key identity changed")
        key = source.read(33)
        stable = os.fstat(source.fileno())
        if (
            len(key) != 32
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (stable.st_dev, stable.st_ino, stable.st_size)
        ):
            raise RuntimeError("Component agent integrity key is damaged")
        if os.name != "nt" and stat.S_IMODE(path_status.st_mode) & 0o077:
            raise RuntimeError("Component agent integrity key permissions are unsafe")
        _require_directory_chain_identity(directory_identity)
        return key


def _snapshot_key_directory_chain(
    directory: Path,
    run_root: Path,
) -> list[tuple[Path, tuple[int, int, int, int]]]:
    try:
        relative = directory.relative_to(run_root)
    except ValueError as error:
        raise RuntimeError("Component agent integrity key is outside run root") from error
    current = run_root
    identities = []
    for part in (Path(), *relative.parts):
        if part != Path():
            current /= part
        status = current.lstat()
        if _is_link_or_reparse(status):
            raise RuntimeError(
                "Component agent integrity key directory is a link or reparse point"
            )
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError("Component agent integrity key parent is not a directory")
        identities.append((current, _directory_identity(status)))
    return identities


def _validate_request_marker(marker: object) -> dict:
    fields = {
        "schema_version",
        "page_id",
        "provider",
        "repair_round",
        "request_path",
        "request_sha256",
        "hmac_sha256",
    }
    if not isinstance(marker, dict) or set(marker) != fields:
        raise ValueError("Component agent request marker fields are invalid")
    if type(marker["schema_version"]) is not int or marker["schema_version"] != 1:
        raise ValueError("Component agent request marker schema_version is invalid")
    validate_agent_provider(marker["provider"])
    validate_repair_round(marker["repair_round"])
    if type(marker["page_id"]) is not str or not marker["page_id"]:
        raise ValueError("Component agent request marker page_id is invalid")
    expected_path = f"round-{marker['repair_round']:02d}/{REQUEST_NAME}"
    if marker["request_path"] != expected_path:
        raise ValueError("Component agent request marker path is invalid")
    for field in ("request_sha256", "hmac_sha256"):
        digest = marker[field]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Component agent request marker {field} is invalid")
    return marker


def _canonical_marker_fields(fields: dict) -> bytes:
    return json.dumps(
        fields,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )
