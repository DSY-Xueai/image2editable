from __future__ import annotations

import json
import hashlib
import hmac
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import stat

import pytest
import cv2

from image2editable.component_repair import (
    EVIDENCE_NAMES,
    advance_component_repair,
    build_component_agent_request,
    initialize_component_repair_state,
    load_component_agent_graph,
    load_component_agent_request,
    record_component_execution,
    record_next_component_request,
    record_parent_fallback_execution,
    record_parent_fallback_quality,
    record_component_quality,
    record_local_component_plan,
)
import image2editable.component_repair as component_repair
from scripts.visual_segment import VisualSegmentationError, _publish_action_directory, execute_component_actions
from scripts.sam_worker import component_prompt_mask, run_component_prompt_worker
from PIL import Image
import numpy as np


def test_advance_without_state_only_reports_needs_initialization(tmp_path: Path) -> None:
    from image2editable.inputs import prepare_image_job
    from image2editable.store import RunStore

    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)

    outcome = advance_component_repair(store, "page_001")

    assert outcome == {"status": "needs_initialization", "page_id": "page_001"}
    reconstruction = run_dir / "pages/page_001/reconstruction"
    assert not (reconstruction / "component_state.json").exists()
    assert not (reconstruction / "agent").exists()


def test_component_repair_rejects_unheld_execution_lease(page_session: dict) -> None:
    from image2editable.execution import ExecutionLease
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    lease = ExecutionLease(store.root / "execution.lock", run_root=store.root)

    with pytest.raises(RuntimeError, match="held Run execution lease"):
        initialize_component_repair_state(
            store, "page_001", request_path=request_path,
            initial_component_count=2, _lease=lease,
        )


def test_initialized_state_points_to_hash_bound_current_request(page_session: dict) -> None:
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    run_root = request_path.parents[5]
    store = RunStore(run_root)
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {"schema_version": 1, "status": "prepared"})
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "processing"}
    }})
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )

    assert state["phase"] == "request_published"
    assert state["repair_round"] == 1
    assert state["plan_count"] == 0
    assert state["delivery_checks"] == {"pptx_reopen": "unknown"}
    assert state["current_round"]["request_ref"]["sha256"] == hashlib.sha256(
        request_path.read_bytes()
    ).hexdigest()
    assert state["current_round"]["request_ref"]["path"] == (
        "pages/page_001/reconstruction/agent/round-01/component_agent_request.json"
    )
    assert advance_component_repair(store, "page_001")["status"] == "awaiting_agent"
    persisted = store.read_json("pages/page_001/reconstruction/component_state.json")
    assert persisted["phase"] == "awaiting_plan"


def test_local_plan_is_hash_bound_and_recorded_without_host_state(
    page_session: dict,
) -> None:
    from image2editable.store import RunStore

    local_session = {**page_session, "provider": "local"}
    request_path = build_component_agent_request(local_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json(
        "job_manifest.json",
        {
            "schema_version": 1,
            "pages": ["page_001"],
            "options": {"agent_provider": "local"},
        },
    )
    initialize_component_repair_state(
        store,
        "page_001",
        request_path=request_path,
        initial_component_count=2,
    )
    assert advance_component_repair(store, "page_001")["status"] == "awaiting_agent"
    plan = {
        "schema_version": 1,
        "kind": "component_plan",
        "page_id": "page_001",
        "provider": "local",
        "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("accept", ["candidate_b"])],
    }

    recorded = record_local_component_plan(store, "page_001", plan=plan)

    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    assert state["phase"] == "plan_recorded"
    assert state["plan_count"] == 1
    assert state["current_round"]["plan_ref"] == recorded["plan_ref"]
    assert not (store.root / "host_capabilities.json").exists()


def test_execution_refreshes_candidates_after_discard(
    page_session: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image2editable.store import RunStore

    evidence_root = Path(page_session["reconstruction_dir"]) / "evidence-source"
    graph_path = evidence_root / "component-graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    for component_id, z_index in (("candidate_c", 2), ("candidate_d", 3)):
        candidate = dict(next(
            node for node in graph["nodes"] if node["id"] == "candidate_b"
        ))
        candidate.update({
            "id": component_id, "mask": f"masks/{component_id}.png",
            "z_index": z_index,
        })
        mask_path = evidence_root / candidate["mask"]
        shutil.copyfile(evidence_root / "masks/candidate_b.png", mask_path)
        candidate["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        graph["nodes"].append(candidate)
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    page_session["provider"] = "local"

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=3,
    )
    advance_component_repair(store, "page_001")
    plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "local", "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("discard", ["candidate_b"])],
    }
    record_local_component_plan(store, "page_001", plan=plan)
    output_dir = request_path.parents[2] / "execution-01"
    next_graph = execute_component_actions(
        np.zeros((2, 2, 3), dtype=np.uint8), graph, plan["actions"],
        sam_runner=None, input_dir=request_path.parent, output_dir=output_dir,
    )
    next_graph_path = output_dir / "component-graph.json"
    execution = {
        "schema_version": 1, "page_id": "page_001", "provider": "local",
        "repair_round": 1, "request_sha256": plan["request_sha256"],
        "input_graph_sha256": hashlib.sha256(
            (request_path.parent / "component-graph.json").read_bytes()
        ).hexdigest(),
        "output_graph_sha256": hashlib.sha256(next_graph_path.read_bytes()).hexdigest(),
        "executable_action_count": 1,
        "quality_input_refs": _quality_input_refs(output_dir, store),
    }
    execution_path = output_dir / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    state = record_component_execution(
        store, "page_001", execution_path=execution_path,
        output_graph_path=next_graph_path,
    )

    assert state["candidate_ids"] == ["candidate_c", "candidate_d"]
    assert next(
        node for node in next_graph["nodes"] if node["id"] == "candidate_b"
    )["state"] == "inactive"

    passed = _strict_quality_report("candidate_c", True)["component_reports"][0]
    failed = _strict_quality_report("candidate_d", False)["component_reports"][0]
    monkeypatch.setattr(
        component_repair, "evaluate_component_quality_round",
        lambda *args, **kwargs: {
            "accepted": False,
            "violations": [
                "missing_edge", "pptx_reopen_unknown", "visual_difference",
            ],
            "component_reports": [passed, failed],
            "visual_metrics": {"mae": 30.0, "p95": 60.0, "changed_ratio": 0.2},
            "checks": {"pptx_reopen": "unknown"},
        },
    )
    record_component_quality(store, "page_001")
    advance_component_repair(store, "page_001")
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    assert sorted(state["frozen"]) == ["candidate_c", "frozen_a"]
    assert state["failed_ids"] == ["candidate_d"]


def test_execution_quality_freeze_reaches_ready_for_assembly(
    page_session: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from image2editable.host_agent import record_host_plan
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {
        "schema_version": 1, "status": "awaiting_agent", "updated_at": "now"
    })
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "awaiting_agent", "updated_at": "now"}
    }})
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    advance_component_repair(store, "page_001")
    request = load_component_agent_request(request_path)
    plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "host", "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("accept", ["candidate_b"])],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    record_host_plan(store.root, plan_path)
    assert advance_component_repair(store, "page_001")["status"] == "needs_execution"

    graph = json.loads((request_path.parent / "component-graph.json").read_text(encoding="utf-8"))
    next_graph = json.loads(json.dumps(graph))
    next(node for node in next_graph["nodes"] if node["id"] == "candidate_b")["state"] = "pending_gate"
    execution_dir = request_path.parents[2] / "execution-01"
    execution_dir.mkdir()
    shutil.copytree(request_path.parent / "masks", execution_dir / "masks")
    quality_input_refs = _quality_input_refs(execution_dir, store)
    graph_path = execution_dir / "component-graph.json"
    graph_path.write_text(json.dumps(next_graph), encoding="utf-8")
    execution = {
        "schema_version": 1, "page_id": "page_001", "provider": "host",
        "repair_round": 1, "request_sha256": plan["request_sha256"],
        "input_graph_sha256": request["graph_sha256"],
        "output_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "executable_action_count": 1,
        "quality_input_refs": quality_input_refs,
    }
    execution_path = execution_dir / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    record_component_execution(
        store, "page_001", execution_path=execution_path, output_graph_path=graph_path,
    )
    assert advance_component_repair(store, "page_001")["status"] == "needs_quality"

    observed_visual = {}

    def quality_evaluator(*args, **kwargs):
        observed_visual.update(kwargs["visual_metrics"])
        return _strict_quality_report("candidate_b", True)

    monkeypatch.setattr(
        component_repair, "evaluate_component_quality_round", quality_evaluator,
    )
    record_component_quality(store, "page_001")
    assert observed_visual["mae"] > 0
    quality_state = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    quality_artifact = json.loads(
        (store.root / quality_state["current_round"]["quality_ref"]["path"])
        .read_text(encoding="utf-8")
    )
    assert set(quality_artifact["input_refs"]) == {
        "source", "background", "reconstructed", "text_mask", "native_check"
    }
    assert advance_component_repair(store, "page_001")["status"] == "freeze_committed"
    ready = advance_component_repair(store, "page_001")
    assert ready["status"] == "ready_for_assembly"
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    assert state["frozen"]["candidate_b"] == state["parent_assets"]["candidate_b"]["sha256"]
    assert state["delivery_checks"] == {"pptx_reopen": "unknown"}
    assert state["result_ref"] is not None
    result = store.read_json("pages/page_001/reconstruction/component_result.json")
    assert set(result["accepted_asset_refs"]) == {
        "source", "background", "reconstructed", "text_mask", "native_check"
    }


def test_next_request_is_page_batch_and_round_six_is_impossible(page_session: dict) -> None:
    from image2editable.store import RunStore

    first = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(first.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {"schema_version": 1, "status": "prepared"})
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "processing"}
    }})
    state = initialize_component_repair_state(
        store, "page_001", request_path=first, initial_component_count=2,
    )
    state["phase"] = "freeze_committed"
    state["candidate_ids"] = state["failed_ids"] = ["candidate_b"]
    state["current_round"]["plan_ref"] = state["current_round"]["request_ref"]
    state["current_round"]["execution_ref"] = state["current_round"]["request_ref"]
    state["current_round"]["quality_ref"] = state["current_round"]["request_ref"]
    store.write_json("pages/page_001/reconstruction/component_state.json", state)
    updated = state
    request_path = first
    for expected_round in range(2, 6):
        request_path = build_component_agent_request(
            page_session, repair_round=expected_round
        )
        updated = record_next_component_request(
            store, "page_001", request_path=request_path
        )
        assert updated["repair_round"] == expected_round
        assert updated["candidate_ids"] == ["candidate_b"]
        assert updated["plan_count"] == 0
        assert updated["phase"] == "request_published"
        updated["phase"] = "freeze_committed"
        updated["current_round"]["plan_ref"] = updated["current_round"]["request_ref"]
        updated["current_round"]["execution_ref"] = updated["current_round"]["request_ref"]
        updated["current_round"]["quality_ref"] = updated["current_round"]["request_ref"]
        store.write_json("pages/page_001/reconstruction/component_state.json", updated)
    assert not (first.parent.parent / "round-06").exists()
    with pytest.raises(RuntimeError, match="round 6|five"):
        record_next_component_request(store, "page_001", request_path=request_path)


def test_agent_request_rejects_tampered_component_mask(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    graph_path = request_path.parent / request["evidence"]["component-graph.json"]["path"]
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    mask_path = request_path.parent / graph["nodes"][0]["mask"]
    mask_path.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="mask hash mismatch"):
        load_component_agent_request(request_path)


def test_quality_recorder_rejects_external_self_report(page_session: dict) -> None:
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    with pytest.raises(TypeError, match="quality_path"):
        record_component_quality(
            store, "page_001", quality_path=request_path.parent / "quality-report.json"
        )


def test_fallback_state_requires_dedicated_refs(page_session: dict) -> None:
    from image2editable.component_contracts import validate_component_repair_state
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    state["phase"] = "fallback_executed"
    state["stop_reason"] = "round_limit"
    state["fallback"] = {"status": "parent_pending", "parent_ids": ["candidate_b"]}

    with pytest.raises(ValueError, match="fallback execution references"):
        validate_component_repair_state(state)


def test_same_normalized_plan_twice_stops_before_execution(page_session: dict) -> None:
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {"schema_version": 1, "status": "prepared"})
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "processing"}
    }})
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "host", "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("accept", ["candidate_b"])],
    }
    plan_path = store.root / "same-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    state["phase"] = "plan_recorded"
    state["plan_count"] = 1
    state["current_round"]["plan_ref"] = {
        "path": "same-plan.json",
        "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    }
    state["last_normalized_plan_sha256"] = component_repair._normalized_plan_sha256(plan)
    store.write_json("pages/page_001/reconstruction/component_state.json", state)

    outcome = advance_component_repair(store, "page_001")

    assert outcome["status"] == "fallback_required"
    assert outcome["stop_reason"] == "repeated_plan"
    assert not (request_path.parents[1] / "execution-01").exists()


def test_zero_executable_actions_stops_without_quality(page_session: dict) -> None:
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {"schema_version": 1, "status": "prepared"})
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "processing"}
    }})
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    execution = {"executable_action_count": 0}
    execution_path = store.root / "execution-zero.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    state["phase"] = "actions_executed"
    state["current_round"]["plan_ref"] = state["current_round"]["request_ref"]
    state["current_round"]["execution_ref"] = {
        "path": "execution-zero.json",
        "sha256": hashlib.sha256(execution_path.read_bytes()).hexdigest(),
    }
    store.write_json("pages/page_001/reconstruction/component_state.json", state)

    outcome = advance_component_repair(store, "page_001")

    assert outcome["status"] == "fallback_required"
    assert outcome["stop_reason"] == "no_executable_actions"
    assert not (request_path.parents[2] / "component_result.json").exists()


@pytest.mark.parametrize(
    ("accepted", "expected"),
    [(True, "ready_for_assembly"), (False, "preserved_with_warning")],
)
def test_intact_parent_gate_controls_fallback_result(
    page_session: dict, accepted: bool, expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {"schema_version": 1, "status": "prepared"})
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "processing"}
    }})
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    state["phase"] = "fallback_required"
    state["stop_reason"] = "round_limit"
    state["fallback"] = {"status": "required", "parent_ids": ["candidate_b"]}
    store.write_json("pages/page_001/reconstruction/component_state.json", state)
    graph = load_component_agent_graph(request_path)
    next(node for node in graph["nodes"] if node["id"] == "candidate_b")["state"] = "pending_gate"
    fallback_dir = request_path.parents[2] / "fallback"
    fallback_dir.mkdir()
    (fallback_dir / "masks").mkdir()
    shutil.copy2(
        request_path.parent / "masks/candidate_b.png",
        fallback_dir / "masks/candidate_b.png",
    )
    shutil.copy2(
        request_path.parent / "masks/frozen_a.png",
        fallback_dir / "masks/frozen_a.png",
    )
    graph_path = fallback_dir / "component-graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    quality_input_refs = _quality_input_refs(fallback_dir, store)
    record_parent_fallback_execution(
        store, "page_001", graph_path=graph_path,
        quality_input_refs=quality_input_refs,
    )
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    monkeypatch.setattr(
        component_repair, "evaluate_component_quality_round",
        lambda *args, **kwargs: _strict_quality_report("candidate_b", accepted),
    )
    record_parent_fallback_quality(store, "page_001")

    result = advance_component_repair(store, "page_001")

    assert result["status"] == expected
    final_state = store.read_json("pages/page_001/reconstruction/component_state.json")
    if accepted:
        assert final_state["fallback"]["status"] == "parent_preserved"
        assert final_state["frozen"]["candidate_b"] == final_state["parent_assets"]["candidate_b"]["sha256"]
    else:
        assert final_state["fallback"]["status"] == "warning"


def _node(component_id: str, state: str, z_index: int) -> dict:
    return {
        "id": component_id,
        "kind": "parent",
        "parent_id": None,
        "state": state,
        "mask": f"masks/{component_id}.png",
        "mask_sha256": "a" * 64,
        "bbox": [0, 0, 2, 2],
        "z_index": z_index,
        "text_ids": [],
    }


def _action(action: str, object_ids: list[str], parameters: dict | None = None) -> dict:
    return {"action": action, "object_ids": object_ids, "parameters": parameters or {},
            "confidence": 0.95, "evidence": ["visible relationship"]}


def _strict_quality_report(component_id: str, accepted: bool) -> dict:
    metrics = {
        "component_pixels": 4, "missing_pixels": 0, "missing_ratio": 0.0,
        "duplicate_pixels": 0, "duplicate_ratio": 0.0, "edge_missing_ratio": 0.0,
        "shadow_duplicate_ratio": 0.0, "alpha_duplicate_ratio": 0.0,
        "exterior_shadow_pixels": 0, "exterior_alpha_pixels": 0,
        "orphan_residual_pixels": 0, "text_support_pixels": 0,
        "text_duplicate_ratio": 0.0, "ownership_out_of_bounds_pixels": 0,
        "parent_coverage_ratio": 1.0, "component_overlap_pixels": 0,
        "parent_child_double": False, "noise_l1": 0.0, "local_contrast": 1.0,
        "edge_width_px": 1, "text_halo_px": 1,
        "adaptive_pixel_tolerance": 3.0, "hard_pixel_tolerance": 3.0,
    }
    component_violations = [] if accepted else ["missing_edge"]
    page_violations = sorted(component_violations + ["pptx_reopen_unknown"])
    return {
        "accepted": False, "violations": page_violations,
        "component_reports": [{
            "component_id": component_id, "accepted": accepted, "metrics": metrics,
            "improvement": {}, "violations": component_violations,
            "checks": {"protected_native_overlap": "pass"},
            "agent_confidence": None,
        }],
        "visual_metrics": {"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        "checks": {"pptx_reopen": "unknown"},
    }


def _quality_input_refs(directory: Path, store) -> dict:
    paths = {}
    for name in ("background", "reconstructed", "text-mask"):
        path = directory / f"{name}.png"
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)
        paths[name] = path
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    native = directory / "native-check.json"
    native.write_text(json.dumps({
        "schema_version": 1, "page_id": "page_001",
        "source_sha256": state["source_sha256"],
        "protected_native_overlap": "pass",
    }), encoding="utf-8")
    paths["native-check"] = native
    return {
        name.replace("-", "_"): {
            "path": path.resolve().relative_to(store.root.resolve()).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
    }


def _action_case(tmp_path: Path) -> tuple[np.ndarray, dict, Path]:
    root = tmp_path / "round-01"
    masks = root / "masks"
    masks.mkdir(parents=True)
    values = {
        "parent": np.ones((12, 16), dtype=bool),
        "left": np.pad(np.ones((4, 4), dtype=bool), ((2, 6), (2, 10))),
        "right": np.pad(np.ones((4, 4), dtype=bool), ((2, 6), (10, 2))),
        "frozen": np.pad(np.ones((2, 2), dtype=bool), ((9, 1), (7, 7))),
        "text": np.pad(np.ones((1, 1), dtype=bool), ((0, 11), (0, 15))),
    }
    nodes = []
    specs = [
        ("parent", "parent", None, "inactive", 0),
        ("left", "child", "parent", "pending", 1),
        ("right", "child", "parent", "pending", 2),
        ("frozen", "parent", None, "frozen", 3),
        ("text", "text", None, "frozen", 4),
    ]
    for component_id, kind, parent_id, state, z_index in specs:
        path = masks / f"{component_id}.png"
        Image.fromarray(values[component_id].astype(np.uint8) * 255).save(path)
        ys, xs = np.where(values[component_id])
        nodes.append({"id": component_id, "kind": kind, "parent_id": parent_id,
                      "state": state, "mask": f"masks/{component_id}.png",
                      "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                      "z_index": z_index, "text_ids": []})
    return np.zeros((12, 16, 3), dtype=np.uint8), {"nodes": nodes}, root


def test_execute_accept_is_pending_gate_and_preserves_frozen_hash(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    output = tmp_path / "round-02"
    result = execute_component_actions(
        image, graph, [_action("accept", ["left"])], sam_runner=None,
        input_dir=input_dir, output_dir=output,
    )
    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["left"]["state"] == "pending_gate"
    assert by_id["frozen"] == next(node for node in graph["nodes"] if node["id"] == "frozen")
    assert (output / "component-graph.json").is_file()


def test_execute_discard_inactivates_redundant_candidate(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)

    result = execute_component_actions(
        image, graph, [_action("discard", ["left"])], sam_runner=None,
        input_dir=input_dir, output_dir=tmp_path / "round-discard",
    )

    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["left"]["state"] == "inactive"
    assert by_id["right"]["state"] == "pending"


def test_execute_background_rebuild_action_preserves_component_graph(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)

    result = execute_component_actions(
        image, graph,
        [_action("rebuild_background", ["left", "right"], {"margin_ratio": 0.01})],
        sam_runner=None, input_dir=input_dir,
        output_dir=tmp_path / "round-background",
    )

    assert result == graph


def test_execute_absorb_unions_visuals_into_one_parent(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    parent = next(node for node in graph["nodes"] if node["id"] == "parent")
    parent_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    parent_mask[0, 0] = 255
    parent_path = input_dir / parent["mask"]
    Image.fromarray(parent_mask).save(parent_path)
    parent["mask_sha256"] = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    parent["bbox"] = [0, 0, 1, 1]

    result = execute_component_actions(
        image, graph,
        [
            _action("absorb_into_parent", ["parent", "left", "right"]),
            _action("rebuild_background", ["left"], {"margin_ratio": 0.01}),
        ],
        sam_runner=None, input_dir=input_dir,
        output_dir=tmp_path / "round-absorb",
    )

    by_id = {node["id"]: node for node in result["nodes"]}
    absorbed = np.asarray(Image.open(
        tmp_path / "round-absorb" / by_id["parent"]["mask"]
    )) > 0
    assert int(absorbed.sum()) == 33
    assert by_id["parent"]["state"] == "pending"
    assert by_id["left"]["state"] == by_id["right"]["state"] == "inactive"


def test_frozen_mask_keeps_nonstandard_relative_path(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    frozen = next(node for node in graph["nodes"] if node["id"] == "frozen")
    custom = input_dir / "masks/archive/frozen-original.png"
    custom.parent.mkdir()
    (input_dir / frozen["mask"]).replace(custom)
    frozen["mask"] = "masks/archive/frozen-original.png"
    output = tmp_path / "round-custom"
    result = execute_component_actions(
        image, graph, [_action("accept", ["left"])], sam_runner=None,
        input_dir=input_dir, output_dir=output,
    )
    published = next(node for node in result["nodes"] if node["id"] == "frozen")
    assert published == frozen
    assert (output / frozen["mask"]).read_bytes() == custom.read_bytes()


def test_execute_merge_unions_masks_and_inactivates_sources(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    output = tmp_path / "round-02"
    result = execute_component_actions(
        image, graph, [_action("merge", ["left", "right"])], sam_runner=None,
        input_dir=input_dir, output_dir=output,
    )
    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["left"]["state"] == by_id["right"]["state"] == "inactive"
    merged = np.asarray(Image.open(output / by_id["merge_0001"]["mask"])) > 0
    assert int(merged.sum()) == 32
    assert by_id["merge_0001"]["kind"] == "child"
    assert by_id["merge_0001"]["parent_id"] == "parent"


def test_split_without_connected_proposals_fails_without_output(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    output = tmp_path / "round-02"
    with pytest.raises(VisualSegmentationError, match="connected proposals"):
        execute_component_actions(
            image, graph, [_action("split", ["left"], {"parts": 2})], sam_runner=None,
            input_dir=input_dir, output_dir=output,
        )
    assert not output.exists()


def test_split_rejects_extra_connected_proposals_instead_of_losing_pixels(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    left = next(node for node in graph["nodes"] if node["id"] == "left")
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[1:3, 1:3] = mask[5:7, 5:7] = mask[9:11, 9:11] = 255
    path = input_dir / left["mask"]
    Image.fromarray(mask).save(path)
    left["mask_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    left["bbox"] = [1, 1, 11, 11]
    output = tmp_path / "round-extra-parts"
    with pytest.raises(VisualSegmentationError, match="exact connected proposals"):
        execute_component_actions(
            image, graph, [_action("split", ["left"], {"parts": 2})], sam_runner=None,
            input_dir=input_dir, output_dir=output,
        )
    assert not output.exists()


def test_split_two_connected_proposals_preserves_pixels_and_layer(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    left = next(node for node in graph["nodes"] if node["id"] == "left")
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[1:3, 1:3] = mask[7:10, 8:11] = 255
    path = input_dir / left["mask"]
    Image.fromarray(mask).save(path)
    left["mask_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    left["bbox"] = [1, 1, 11, 10]
    output = tmp_path / "round-split"
    result = execute_component_actions(
        image, graph, [_action("split", ["left"], {"parts": 2})], sam_runner=None,
        input_dir=input_dir, output_dir=output,
    )
    by_id = {node["id"]: node for node in result["nodes"]}
    children = [node for node in result["nodes"] if node["id"].startswith("split_")]
    union = np.zeros(image.shape[:2], dtype=bool)
    for child in children:
        payload = (output / child["mask"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == child["mask_sha256"]
        union |= np.asarray(Image.open(output / child["mask"])) > 0
        assert child["kind"] == "child" and child["parent_id"] == "parent"
    assert by_id["left"]["state"] == "inactive"
    assert len(children) == 2
    assert np.array_equal(union, mask > 0)


def test_action_failure_does_not_delete_replacement_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    output = tmp_path / "round-replaced"
    real_save = Image.Image.save
    replacement = None

    def replace_staging_then_fail(self: Image.Image, path: object, *args: object, **kwargs: object) -> None:
        nonlocal replacement
        staging = Path(path).parent.parent
        owned = staging.with_name(staging.name + "-owned")
        staging.rename(owned)
        staging.mkdir()
        replacement = staging / "attacker.txt"
        replacement.write_text("keep", encoding="utf-8")
        raise OSError("simulated save failure")

    monkeypatch.setattr(Image.Image, "save", replace_staging_then_fail)
    with pytest.raises(OSError, match="save failure"):
        execute_component_actions(
            image, graph, [_action("accept", ["left"])], sam_runner=None,
            input_dir=input_dir, output_dir=output,
        )
    assert replacement is not None and replacement.read_text(encoding="utf-8") == "keep"
    monkeypatch.setattr(Image.Image, "save", real_save)


def test_sam_prompt_coordinates_and_attach_text_do_not_merge_pixels(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    calls = []
    def runner(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.asarray(Image.open(input_dir / "masks/left.png")) > 0
    output = tmp_path / "round-02"
    result = execute_component_actions(
        image, graph,
        [_action("retry_with_points", ["left"], {"positive": [[1.0, 1.0]], "negative": [[0.0, 0.0]]}),
         _action("attach_text", ["right", "text"])],
        sam_runner=runner, input_dir=input_dir, output_dir=output,
    )
    assert calls[0]["positive"] == [[15.0, 11.0]]
    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["right"]["text_ids"] == ["text"]
    assert int((np.asarray(Image.open(output / by_id["right"]["mask"])) > 0).sum()) == 16


def test_expand_stays_inside_parent_and_collapse_activates_parent(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    expanded_dir = tmp_path / "round-02"
    expanded = execute_component_actions(
        image, graph, [_action("expand", ["left"], {"margin_ratio": 0.2})],
        sam_runner=None, input_dir=input_dir, output_dir=expanded_dir,
    )
    by_id = {node["id"]: node for node in expanded["nodes"]}
    expanded_mask = np.asarray(Image.open(expanded_dir / by_id["left"]["mask"])) > 0
    parent_mask = np.asarray(Image.open(input_dir / "masks/parent.png")) > 0
    assert not np.any(expanded_mask & ~parent_mask)
    collapsed_dir = tmp_path / "round-03"
    collapsed = execute_component_actions(
        image, expanded, [_action("collapse_to_parent", ["parent"])],
        sam_runner=None, input_dir=expanded_dir, output_dir=collapsed_dir,
    )
    states = {node["id"]: node["state"] for node in collapsed["nodes"]}
    assert states["parent"] == "pending"
    assert states["left"] == states["right"] == "inactive"


def test_action_margin_uses_page_short_edge_for_different_component_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    right = next(node for node in graph["nodes"] if node["id"] == "right")
    right_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    right_mask[1:9, 7:15] = 255
    right_path = input_dir / right["mask"]
    Image.fromarray(right_mask).save(right_path)
    right["mask_sha256"] = hashlib.sha256(right_path.read_bytes()).hexdigest()
    right["bbox"] = [7, 1, 15, 9]
    kernel_sizes = []
    real_kernel = cv2.getStructuringElement

    def record_kernel(shape: int, size: tuple[int, int]) -> np.ndarray:
        kernel_sizes.append(size)
        return real_kernel(shape, size)

    monkeypatch.setattr(cv2, "getStructuringElement", record_kernel)
    execute_component_actions(
        image, graph, [_action("expand", ["left"], {"margin_ratio": 0.25})],
        sam_runner=None, input_dir=input_dir, output_dir=tmp_path / "round-margin-left",
    )
    execute_component_actions(
        image, graph, [_action("expand", ["right"], {"margin_ratio": 0.25})],
        sam_runner=None, input_dir=input_dir, output_dir=tmp_path / "round-margin-right",
    )
    assert kernel_sizes == [(7, 7), (7, 7)]


def test_shrink_uses_page_margin_and_publishes_nonempty_mask(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    result = execute_component_actions(
        image, graph, [_action("shrink", ["left"], {"margin_ratio": 0.1})],
        sam_runner=None, input_dir=input_dir, output_dir=tmp_path / "round-shrink",
    )
    left = next(node for node in result["nodes"] if node["id"] == "left")
    mask = np.asarray(Image.open(tmp_path / "round-shrink" / left["mask"])) > 0
    assert 0 < int(mask.sum()) < 16


def test_publish_action_directory_never_replaces_existing_empty_target(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "target"
    staging.mkdir()
    (staging / "component-graph.json").write_text("new", encoding="utf-8")
    target.mkdir()
    with pytest.raises(FileExistsError):
        _publish_action_directory(staging, target)
    assert target.is_dir() and not list(target.iterdir())
    assert (staging / "component-graph.json").read_text(encoding="utf-8") == "new"


def test_multi_action_round_is_validated_before_mutation(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    result = execute_component_actions(
        image,
        graph,
        [_action("merge", ["left", "right"]), _action("collapse_to_parent", ["parent"])],
        sam_runner=None,
        input_dir=input_dir,
        output_dir=tmp_path / "round-batch",
    )
    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["parent"]["state"] == "pending"


def test_invalid_later_action_is_rejected_before_sam_side_effect(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    calls = []

    def runner(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.ones(image.shape[:2], dtype=bool)

    invalid = _action("accept", ["text"])
    with pytest.raises(ValueError, match="text kind"):
        execute_component_actions(
            image, graph,
            [_action("retry_with_box", ["left"], {"box": [0.1, 0.1, 0.9, 0.9]}), invalid],
            sam_runner=runner, input_dir=input_dir, output_dir=tmp_path / "round-invalid",
        )
    assert calls == []


def test_retry_rejects_inactive_non_candidate(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    with pytest.raises(ValueError, match="pending component"):
        execute_component_actions(
            image, graph,
            [_action("retry_with_box", ["parent"], {"box": [0.1, 0.1, 0.9, 0.9]})],
            sam_runner=lambda **_: np.ones(image.shape[:2], dtype=bool),
            input_dir=input_dir, output_dir=tmp_path / "round-inactive",
        )


def test_retry_box_maps_normalized_page_coordinates(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    calls = []
    def runner(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.asarray(Image.open(input_dir / "masks/left.png")) > 0
    execute_component_actions(
        image, graph,
        [_action("retry_with_box", ["left"], {"box": [0.25, 0.25, 0.75, 0.75]})],
        sam_runner=runner, input_dir=input_dir, output_dir=tmp_path / "round-02",
    )
    assert calls[0]["box"] == [4.0, 3.0, 12.0, 9.0]


def test_sam_worker_component_prompt_selects_best_mask_and_can_run_twice() -> None:
    class Predictor:
        def set_image(self, image: np.ndarray) -> None:
            self.image = image
        def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
            masks = np.zeros((2, 4, 5), dtype=bool)
            masks[1, 1:3, 1:4] = True
            return masks, np.asarray([0.1, 0.9]), None
    generator = type("Generator", (), {"predictor": Predictor()})()
    prompt = {"box": [1, 1, 4, 3], "positive": [], "negative": []}
    first = component_prompt_mask(generator, np.zeros((4, 5, 3), dtype=np.uint8), prompt)
    second = component_prompt_mask(generator, np.zeros((4, 5, 3), dtype=np.uint8), prompt)
    assert np.array_equal(first, second)
    assert int(first.sum()) == 6


def test_component_sam_subprocess_runner_reads_result_and_cleans_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def run(command: list[str], **kwargs: object) -> None:
        calls.append((command, kwargs))
        result = Path(command[command.index("--result") + 1])
        mask = np.zeros((4, 5), dtype=bool)
        mask[1:3, 1:4] = True
        packed = np.packbits(mask, axis=None).tobytes()
        import base64
        result.write_text(json.dumps([{
            "mask": base64.b64encode(packed).decode("ascii"),
            "mask_shape": [4, 5],
        }]), encoding="utf-8")

    monkeypatch.setattr("scripts.sam_worker.subprocess.run", run)
    mask = run_component_prompt_worker(
        np.zeros((4, 5, 3), dtype=np.uint8),
        box=[1, 1, 4, 3], positive=[], negative=[], work_dir=tmp_path,
    )
    assert int(mask.sum()) == 6
    assert calls[0][1]["check"] is True
    assert calls[0][1]["timeout"] == 600
    assert not list(tmp_path.glob("component-sam-*"))


@pytest.fixture
def page_session(tmp_path: Path) -> dict:
    return _make_page_session(tmp_path, "page_001")


def _make_page_session(run_root: Path, page_id: str) -> dict:
    reconstruction = run_root / "pages" / page_id / "reconstruction"
    evidence_root = reconstruction / "evidence-source"
    evidence_root.mkdir(parents=True)
    graph = {
        "nodes": [
            _node("candidate_b", "pending", 1),
            _node("frozen_a", "frozen", 0),
        ]
    }
    masks = evidence_root / "masks"
    masks.mkdir()
    for index, node in enumerate(graph["nodes"]):
        mask_path = masks / f"{node['id']}.png"
        Image.fromarray(np.full((2, 2), 255 - index, dtype=np.uint8)).save(mask_path)
        node["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    sources = {}
    for name in EVIDENCE_NAMES:
        path = evidence_root / name
        if name == "component-graph.json":
            path.write_text(json.dumps(graph), encoding="utf-8")
        elif name == "quality-report.json":
            path.write_text('{"valid": false}', encoding="utf-8")
        elif path.suffix == ".png":
            value = 255 if name == "source.png" else 0
            Image.fromarray(np.full((2, 2), value, dtype=np.uint8)).save(path)
        else:
            path.write_bytes((name + " data").encode())
        sources[name] = path
    return {
        "page_id": page_id,
        "provider": "host",
        "reconstruction_dir": reconstruction,
        "evidence": sources,
    }


def test_build_request_hash_binds_every_evidence_file(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert set(request["evidence"]) == set(EVIDENCE_NAMES)
    assert all(len(record["sha256"]) == 64 for record in request["evidence"].values())
    assert request["candidate_ids"] == ["candidate_b"]
    assert request["frozen_ids"] == ["frozen_a"]
    assert request_path.parent.name == "round-01"


def test_validate_request_rejects_changed_overlay(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    overlay = request_path.parent / "ocr-overlay.png"
    overlay.write_bytes(overlay.read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match="evidence hash"):
        load_component_agent_request(request_path)


@pytest.mark.parametrize("repair_round", [0, 6, True, 1.0])
def test_build_request_rejects_round_outside_fixed_limit(
    page_session: dict,
    repair_round: object,
) -> None:
    with pytest.raises(ValueError, match="repair_round"):
        build_component_agent_request(page_session, repair_round=repair_round)


def test_published_request_cannot_be_overwritten(page_session: dict) -> None:
    first = build_component_agent_request(page_session, repair_round=1)
    before = first.read_bytes()

    with pytest.raises(RuntimeError, match="already published"):
        build_component_agent_request(page_session, repair_round=1)

    assert first.read_bytes() == before


def _publish(session: dict, ready: object, result: object) -> None:
    ready.wait(10)
    try:
        build_component_agent_request(session, repair_round=1)
    except RuntimeError:
        result.put("rejected")
    else:
        result.put("published")


def _publish_round(session: dict, repair_round: int, started: object, result: object) -> None:
    started.set()
    try:
        build_component_agent_request(session, repair_round=repair_round)
    except RuntimeError:
        result.put("rejected")
    else:
        result.put("published")


def _hold_publication_lease(reconstruction: str, ready: object, release: object) -> None:
    with component_repair._run_publication_lease(Path(reconstruction)):
        component_repair._load_integrity_key(Path(reconstruction))
        ready.set()
        release.wait(10)


def _refresh_marker_for_contract_test(request_path: Path) -> None:
    marker_path = request_path.parent / "publication-marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["request_sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    key_path = (
        request_path.parents[5]
        / ".component-agent-integrity"
        / "key.bin"
    )
    fields = {key: value for key, value in marker.items() if key != "hmac_sha256"}
    canonical = json.dumps(
        fields,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    marker["hmac_sha256"] = hmac.new(
        key_path.read_bytes(), canonical, hashlib.sha256
    ).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def test_same_page_round_has_one_concurrent_publisher(page_session: dict) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    result = context.Queue()
    processes = [
        context.Process(target=_publish, args=(page_session, ready, result))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0

    assert sorted(result.get(timeout=2) for _ in processes) == [
        "published",
        "rejected",
    ]


def test_build_request_rejects_cross_page_evidence(page_session: dict, tmp_path: Path) -> None:
    outside = tmp_path / "pages" / "page_002" / "reconstruction" / "source.png"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"other page")
    page_session["evidence"]["source.png"] = outside

    with pytest.raises(RuntimeError, match="outside page reconstruction"):
        build_component_agent_request(page_session, repair_round=1)


def test_load_rejects_hard_linked_evidence(page_session: dict, tmp_path: Path) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    evidence = request_path.parent / "ownership.png"
    outside = tmp_path / "outside.png"
    evidence.replace(outside)
    try:
        os.link(outside, evidence)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="hard link"):
        load_component_agent_request(request_path)


def test_load_rejects_request_unknown_field(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["unexpected"] = True
    request_path.write_text(json.dumps(request), encoding="utf-8")
    _refresh_marker_for_contract_test(request_path)

    with pytest.raises(ValueError, match="request fields"):
        load_component_agent_request(request_path)


def test_load_rejects_evidence_path_traversal(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["evidence"]["source.png"]["path"] = "../source.png"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    _refresh_marker_for_contract_test(request_path)

    with pytest.raises(ValueError, match="evidence path"):
        load_component_agent_request(request_path)


def test_build_rejects_symlinked_evidence(page_session: dict, tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    source = page_session["evidence"]["source.png"]
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="link|safely"):
        build_component_agent_request(page_session, repair_round=1)


def test_load_rejects_candidate_ids_not_bound_to_graph(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["candidate_ids"] = []
    request_path.write_text(json.dumps(request), encoding="utf-8")
    _refresh_marker_for_contract_test(request_path)

    with pytest.raises(RuntimeError, match="component ids"):
        load_component_agent_request(request_path)


def test_build_rejects_similar_but_non_pages_directory(tmp_path: Path) -> None:
    reconstruction = tmp_path / "not-pages" / "page_001" / "reconstruction"
    evidence_root = reconstruction / "evidence-source"
    evidence_root.mkdir(parents=True)
    graph = {"nodes": [_node("candidate", "pending", 0)]}
    evidence = {}
    for name in EVIDENCE_NAMES:
        path = evidence_root / name
        if name == "component-graph.json":
            path.write_text(json.dumps(graph), encoding="utf-8")
        elif name == "quality-report.json":
            path.write_text('{"valid": false}', encoding="utf-8")
        else:
            path.write_bytes(name.encode())
        evidence[name] = path

    with pytest.raises(RuntimeError, match="pages/.+reconstruction"):
        build_component_agent_request(
            {
                "page_id": "page_001",
                "provider": "host",
                "reconstruction_dir": reconstruction,
                "evidence": evidence,
            },
            repair_round=1,
        )


def test_load_rejects_request_moved_under_similar_fake_directory(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    pages_dir = Path(page_session["reconstruction_dir"]).parent.parent
    fake_dir = pages_dir.with_name("not-pages")
    pages_dir.rename(fake_dir)
    moved_request = fake_dir / request_path.relative_to(pages_dir)

    with pytest.raises(RuntimeError, match="pages/.+reconstruction"):
        load_component_agent_request(moved_request)


def test_load_requires_external_request_digest_marker(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    marker = request_path.parent / "publication-marker.json"
    marker.unlink()

    with pytest.raises(RuntimeError, match="marker"):
        load_component_agent_request(request_path)


def test_load_rejects_provider_changed_after_publish(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["provider"] = "local"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(RuntimeError, match="request hash"):
        load_component_agent_request(request_path)


def test_load_rejects_evidence_and_request_hash_changed_together(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    evidence_path = request_path.parent / "ocr-overlay.png"
    evidence_path.write_bytes(b"coordinated replacement")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["evidence"]["ocr-overlay.png"]["sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(RuntimeError, match="request hash"):
        load_component_agent_request(request_path)


def test_load_rejects_synchronized_request_evidence_and_marker_without_key(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    evidence_path = request_path.parent / "ocr-overlay.png"
    evidence_path.write_bytes(b"coordinated replacement")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["evidence"]["ocr-overlay.png"]["sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    request_path.write_text(json.dumps(request), encoding="utf-8")
    marker_path = request_path.parent / "publication-marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["request_sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    marker["hmac_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="signature"):
        load_component_agent_request(request_path)


def test_marker_write_failure_does_not_publish_round_and_can_retry(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = component_repair._write_exclusive
    failed = False

    def fail_marker(path: Path, payload: bytes, reconstruction: Path) -> None:
        nonlocal failed
        if path.name == "publication-marker.json" and not failed:
            failed = True
            raise OSError("simulated marker failure")
        real_write(path, payload, reconstruction)

    monkeypatch.setattr(component_repair, "_write_exclusive", fail_marker)
    with pytest.raises(OSError, match="marker failure"):
        build_component_agent_request(page_session, repair_round=1)

    round_dir = Path(page_session["reconstruction_dir"]) / "agent" / "round-01"
    assert not round_dir.exists()
    monkeypatch.setattr(component_repair, "_write_exclusive", real_write)
    assert build_component_agent_request(page_session, repair_round=1).is_file()


def test_round_rename_failure_leaves_no_publication_and_can_retry(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    component_repair._load_or_create_integrity_key(reconstruction)
    real_rename = Path.rename
    failed = False

    def fail_round_rename(path: Path, target: Path) -> Path:
        nonlocal failed
        if path.name.startswith(".round-01.tmp-") and not failed:
            failed = True
            raise OSError("simulated pre-publication crash")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_round_rename)
    with pytest.raises(RuntimeError, match="already published"):
        build_component_agent_request(page_session, repair_round=1)

    round_dir = reconstruction / "agent" / "round-01"
    assert not round_dir.exists()
    monkeypatch.setattr(Path, "rename", real_rename)
    assert build_component_agent_request(page_session, repair_round=1).is_file()


def test_damaged_integrity_key_fails_closed_without_rotation(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    key_path = request_path.parents[5] / ".component-agent-integrity" / "key.bin"
    key_path.write_bytes(b"damaged")

    with pytest.raises(RuntimeError, match="integrity key"):
        load_component_agent_request(request_path)

    assert key_path.read_bytes() == b"damaged"


def test_two_pages_concurrently_reuse_one_complete_integrity_key(
    tmp_path: Path,
) -> None:
    first = _make_page_session(tmp_path, "page_001")
    second = _make_page_session(tmp_path, "page_002")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    result = context.Queue()
    processes = [
        context.Process(target=_publish, args=(session, ready, result))
        for session in (first, second)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0

    assert sorted(result.get(timeout=2) for _ in processes) == [
        "published",
        "published",
    ]
    key_path = tmp_path / ".component-agent-integrity" / "key.bin"
    assert len(key_path.read_bytes()) == 32
    assert key_path.stat().st_nlink == 1
    assert load_component_agent_request(
        tmp_path / "pages/page_001/reconstruction/agent/round-01/component_agent_request.json"
    )["page_id"] == "page_001"
    assert load_component_agent_request(
        tmp_path / "pages/page_002/reconstruction/agent/round-01/component_agent_request.json"
    )["page_id"] == "page_002"


def test_integrity_key_hard_link_fails_closed(page_session: dict, tmp_path: Path) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    key_path = request_path.parents[5] / ".component-agent-integrity" / "key.bin"
    outside = tmp_path / "stolen-key"
    key_path.replace(outside)
    try:
        os.link(outside, key_path)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="integrity key|hard link"):
        load_component_agent_request(request_path)


def test_integrity_key_directory_symlink_fails_closed(
    page_session: dict,
    tmp_path: Path,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    anchor = request_path.parents[5] / ".component-agent-integrity"
    outside = tmp_path / "outside-key-anchor"
    anchor.rename(outside)
    try:
        anchor.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        outside.rename(anchor)
        pytest.skip(f"directory symbolic links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="integrity key|link|reparse"):
        load_component_agent_request(request_path)


def test_missing_key_after_published_round_is_not_rotated(page_session: dict) -> None:
    first = build_component_agent_request(page_session, repair_round=1)
    anchor = first.parents[5] / ".component-agent-integrity"
    shutil.rmtree(anchor)

    with pytest.raises(RuntimeError, match="published.+integrity key"):
        build_component_agent_request(page_session, repair_round=2)

    assert not anchor.exists()


def test_build_streams_large_evidence_in_bounded_reads(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    large = Path(page_session["evidence"]["source.png"])
    large.write_bytes(b"x" * (3 * 1024 * 1024 + 17))
    real_fdopen = os.fdopen
    read_sizes: list[int] = []

    class TrackingFile:
        def __init__(self, wrapped: object) -> None:
            self._wrapped = wrapped

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

        def __enter__(self) -> object:
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> object:
            return self._wrapped.__exit__(exc_type, exc, traceback)

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self._wrapped.read(size)

    def tracking_fdopen(descriptor: int, mode: str) -> object:
        wrapped = real_fdopen(descriptor, mode)
        return TrackingFile(wrapped) if "r" in mode else wrapped

    monkeypatch.setattr(component_repair.os, "fdopen", tracking_fdopen)
    build_component_agent_request(page_session, repair_round=1)

    assert read_sizes
    assert -1 not in read_sizes
    assert max(read_sizes) <= 1024 * 1024


def test_build_rejects_component_graph_over_json_limit(page_session: dict) -> None:
    graph = Path(page_session["evidence"]["component-graph.json"])
    graph.write_bytes(b" " * (16 * 1024 * 1024 + 1))

    with pytest.raises(RuntimeError, match="JSON size limit"):
        build_component_agent_request(page_session, repair_round=1)


def test_load_rejects_marker_over_json_limit(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    marker = request_path.parent / "publication-marker.json"
    marker.write_bytes(marker.read_bytes() + b" " * (64 * 1024))

    with pytest.raises(RuntimeError, match="JSON size limit"):
        load_component_agent_request(request_path)


def test_load_rejects_request_over_json_limit(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request_path.write_bytes(request_path.read_bytes() + b" " * (4 * 1024 * 1024))

    with pytest.raises(RuntimeError, match="JSON size limit"):
        load_component_agent_request(request_path)


def test_load_rejects_component_graph_over_json_limit(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    graph = request_path.parent / "component-graph.json"
    graph.write_bytes(b" " * (16 * 1024 * 1024 + 1))

    with pytest.raises(RuntimeError, match="JSON size limit"):
        load_component_agent_request(request_path)


def test_staging_replacement_before_rename_is_removed_and_retryable(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    component_repair._load_or_create_integrity_key(reconstruction)
    real_rename = Path.rename
    replaced = False

    def replace_staging(path: Path, target: Path) -> Path:
        nonlocal replaced
        if path.name.startswith(".round-01.tmp-") and not replaced:
            original = path.with_name(path.name + ".original")
            attacker = path.with_name(path.name + ".attacker")
            real_rename(path, original)
            attacker.mkdir()
            (attacker / "forged").write_bytes(b"forged")
            real_rename(attacker, path)
            replaced = True
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", replace_staging)
    with pytest.raises(RuntimeError, match="staging identity"):
        build_component_agent_request(page_session, repair_round=1)

    round_dir = reconstruction / "agent" / "round-01"
    assert not round_dir.exists()
    quarantines = list((reconstruction / "agent").glob(".quarantine-round-*"))
    assert any((path / "forged").read_bytes() == b"forged" for path in quarantines)
    monkeypatch.setattr(Path, "rename", real_rename)
    assert build_component_agent_request(page_session, repair_round=1).is_file()


def test_simulated_windows_reparse_flag_on_key_anchor_fails_closed(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    anchor = request_path.parents[5] / ".component-agent-integrity"
    real_lstat = Path.lstat
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    class ReparseStatus:
        def __init__(self, wrapped: os.stat_result) -> None:
            self._wrapped = wrapped
            self.st_file_attributes = (
                getattr(wrapped, "st_file_attributes", 0) | reparse_flag
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    def flagged_lstat(path: Path) -> object:
        status = real_lstat(path)
        return ReparseStatus(status) if path == anchor else status

    monkeypatch.setattr(Path, "lstat", flagged_lstat)

    with pytest.raises(RuntimeError, match="reparse"):
        load_component_agent_request(request_path)


def test_missing_key_scan_rejects_simulated_reparse_agent_directory(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    anchor = request_path.parents[5] / ".component-agent-integrity"
    shutil.rmtree(anchor)
    agent = request_path.parent.parent
    real_lstat = Path.lstat
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    class ReparseStatus:
        def __init__(self, wrapped: os.stat_result) -> None:
            self._wrapped = wrapped
            self.st_file_attributes = (
                getattr(wrapped, "st_file_attributes", 0) | reparse_flag
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    def flagged_lstat(path: Path) -> object:
        status = real_lstat(path)
        return ReparseStatus(status) if path == agent else status

    monkeypatch.setattr(Path, "lstat", flagged_lstat)

    with pytest.raises(RuntimeError, match="agent.+reparse"):
        build_component_agent_request(page_session, repair_round=2)
    assert not anchor.exists()


def test_cleanup_quarantines_replaced_staging_without_deleting_unknown_content(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    component_repair._load_or_create_integrity_key(reconstruction)
    real_write = component_repair._write_exclusive
    real_rename = Path.rename
    swapped = False

    def fail_marker(path: Path, payload: bytes, reconstruction: Path) -> None:
        if path.name == "publication-marker.json":
            raise OSError("marker failure")
        real_write(path, payload, reconstruction)

    def swap_before_quarantine(path: Path, target: Path) -> Path:
        nonlocal swapped
        if (
            not swapped
            and path.name.startswith(".round-01.tmp-")
            and ".quarantine-" in target.name
        ):
            original = path.with_name(path.name + ".original")
            attacker = path.with_name(path.name + ".attacker")
            real_rename(path, original)
            attacker.mkdir()
            (attacker / "unknown.txt").write_text("do not delete", encoding="utf-8")
            real_rename(attacker, path)
            swapped = True
        return real_rename(path, target)

    monkeypatch.setattr(component_repair, "_write_exclusive", fail_marker)
    monkeypatch.setattr(Path, "rename", swap_before_quarantine)

    with pytest.raises(OSError, match="marker failure"):
        build_component_agent_request(page_session, repair_round=1)

    quarantines = list((reconstruction / "agent").glob(".quarantine-*"))
    assert swapped is True
    assert any((path / "unknown.txt").read_text(encoding="utf-8") == "do not delete" for path in quarantines if (path / "unknown.txt").is_file())
    assert not (reconstruction / "agent" / "round-01").exists()


def test_normal_marker_failure_cleans_owned_quarantine_without_residue(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    real_write = component_repair._write_exclusive

    def fail_marker(path: Path, payload: bytes, reconstruction: Path) -> None:
        if path.name == "publication-marker.json":
            raise OSError("marker failure")
        real_write(path, payload, reconstruction)

    monkeypatch.setattr(component_repair, "_write_exclusive", fail_marker)
    with pytest.raises(OSError, match="marker failure"):
        build_component_agent_request(page_session, repair_round=1)

    agent = reconstruction / "agent"
    assert not list(agent.glob(".quarantine-*"))
    assert not list(agent.glob(".delete-*"))
    assert not (agent / "round-01").exists()


def test_run_publication_lease_blocks_key_rotation_during_inflight_publish(
    page_session: dict,
) -> None:
    first = build_component_agent_request(page_session, repair_round=1)
    reconstruction = Path(page_session["reconstruction_dir"])
    anchor = first.parents[5] / ".component-agent-integrity"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_publication_lease,
        args=(str(reconstruction), ready, release),
    )
    holder.start()
    assert ready.wait(10)
    shutil.rmtree(anchor)

    started = context.Event()
    result = context.Queue()
    contender = context.Process(
        target=_publish_round,
        args=(page_session, 2, started, result),
    )
    contender.start()
    assert started.wait(10)
    contender.join(0.5)
    assert contender.is_alive()
    with pytest.raises(queue.Empty):
        result.get_nowait()

    release.set()
    holder.join(10)
    contender.join(10)
    assert holder.exitcode == 0
    assert contender.exitcode == 0
    assert result.get(timeout=2) == "rejected"
    assert not anchor.exists()


def test_build_detects_parent_replaced_between_check_and_open(
    page_session: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = Path(page_session["evidence"]["component-graph.json"]).parent
    original_root = evidence_root.with_name("evidence-original")
    attacker_root = evidence_root.with_name("evidence-attacker")
    shutil.copytree(evidence_root, attacker_root)
    attacker_graph = attacker_root / "component-graph.json"
    attacker_graph.write_text(
        json.dumps({"nodes": [_node("attacker", "pending", 0)]}),
        encoding="utf-8",
    )
    real_open = os.open
    replaced = False

    def replace_parent_before_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if not replaced and Path(path) == evidence_root / "component-graph.json":
            evidence_root.rename(original_root)
            attacker_root.rename(evidence_root)
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(component_repair.os, "open", replace_parent_before_open)

    with pytest.raises(RuntimeError, match="directory identity"):
        build_component_agent_request(page_session, repair_round=1)

    assert replaced is True
    assert json.loads(
        (evidence_root / "component-graph.json").read_text(encoding="utf-8")
    )["nodes"][0]["id"] == "attacker"


def test_build_detects_parent_changed_to_symlink_before_open(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = Path(page_session["evidence"]["component-graph.json"]).parent
    original_root = evidence_root.with_name("evidence-original")
    attacker_root = evidence_root.with_name("evidence-attacker")
    shutil.copytree(evidence_root, attacker_root)
    try:
        probe = evidence_root.with_name("symlink-probe")
        probe.symlink_to(attacker_root, target_is_directory=True)
        probe.unlink()
    except OSError as error:
        pytest.skip(f"directory symbolic links are unavailable: {error}")
    real_open = os.open
    replaced = False

    def replace_with_symlink(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if not replaced and Path(path) == evidence_root / "component-graph.json":
            evidence_root.rename(original_root)
            evidence_root.symlink_to(attacker_root, target_is_directory=True)
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(component_repair.os, "open", replace_with_symlink)

    with pytest.raises(RuntimeError, match="directory identity"):
        build_component_agent_request(page_session, repair_round=1)


def test_marker_write_detects_agent_directory_replacement(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    agent_dir = reconstruction / "agent"
    original_agent = reconstruction / "agent-original"
    attacker_agent = reconstruction / "agent-attacker"
    real_open = os.open
    replaced = False

    def replace_agent_before_marker(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if not replaced and Path(path).name == "publication-marker.json":
            agent_dir.rename(original_agent)
            attacker_agent.mkdir()
            attacker_agent.rename(agent_dir)
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(component_repair.os, "open", replace_agent_before_marker)

    with pytest.raises(RuntimeError, match="directory identity"):
        build_component_agent_request(page_session, repair_round=1)

    assert replaced is True
