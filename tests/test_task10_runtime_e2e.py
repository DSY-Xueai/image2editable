from __future__ import annotations

import json
import hashlib
import copy
import shutil
from pathlib import Path

import pytest
from pptx import Presentation
from image2editable.agent import record_decision
from image2editable.pptx_input import prepare_pptx_job
from image2editable.pptx_input import scan_pptx
from image2editable.execution import ExecutionLease
from image2editable.legacy import initialize_legacy_page
from image2editable.store import RunStore
from image2editable import runtime, legacy, component_repair

from test_agent_decision import _candidate_pptx


def _write_json(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_task10_pptx_approval_is_the_only_page_request_gate(tmp_path: Path) -> None:
    source, _ = _candidate_pptx(tmp_path)
    run_dir = prepare_pptx_job(source, run_dir=tmp_path / "run")
    page = run_dir / "pages/page_001"
    assert not (page / "page_request.json").exists()

    rejected = record_decision(
        run_dir,
        page_id="page_001",
        object_id="2",
        decision="preserve",
        confidence=0.99,
        category="full_slide_screenshot",
        evidence=["native object retained"],
    )
    assert rejected["eligible_for_shadow_run"] is False
    assert not (page / "page_request.json").exists()

    # A fresh run models the approved path without hand-written hashes.
    approved_run = prepare_pptx_job(source, run_dir=tmp_path / "approved")
    approved = record_decision(
        approved_run,
        page_id="page_001",
        object_id="2",
        decision="replace",
        confidence=0.99,
        category="full_slide_screenshot",
        evidence=["full-slide screenshot"],
    )
    request = json.loads(
        (approved_run / "pages/page_001/page_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert approved["eligible_for_shadow_run"] is True
    assert request["sha256"] == approved["image_sha256"]
    assert Path(approved_run / request["source"]).is_file()


def test_full_page_candidate_uses_shared_cv_component_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).parents[1] / "test1.pptx"
    if not source.is_file():
        pytest.skip("real test1.pptx fixture is not present")
    run_dir = prepare_pptx_job(source, run_dir=tmp_path / "run")
    page = run_dir / "pages/page_001"
    candidate = json.loads(
        (page / "agent_request.json").read_text(encoding="utf-8")
    )["candidates"][0]
    record_decision(
        run_dir,
        page_id="page_001",
        object_id=candidate["source_shape_id"],
        decision="replace",
        confidence=0.99,
        category="full_slide_screenshot",
        evidence=["full-slide screenshot"],
    )

    from test_runtime_execution import _install_component_e2e_boundaries
    _, initial_calls, _ = _install_component_e2e_boundaries(
        monkeypatch, component_count=2
    )
    store = RunStore.open(run_dir)
    with ExecutionLease(run_dir / "execution.lock", run_root=run_dir) as lease:
        result = initialize_legacy_page(store, "page_001", _lease=lease)

    assert result["status"] == "initialized"
    assert initial_calls == ["page_001"]
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    assert state["phase"] == "request_published"
    request_ref = state["current_round"]["request_ref"]
    request_path = run_dir / request_ref["path"]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["candidate_ids"] == ["component_0001", "component_0002"]
    graph = json.loads(
        (request_path.parent / request["evidence"]["component-graph.json"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert [node["kind"] for node in graph["nodes"]] == ["parent", "parent"]
    for node in graph["nodes"]:
        mask = request_path.parent / node["mask"]
        assert hashlib.sha256(mask.read_bytes()).hexdigest() == node["mask_sha256"]


def test_test1_two_page_component_state_to_shadow_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).parents[1] / "test1.pptx"
    if not source.is_file():
        pytest.skip("real test1.pptx fixture is not present")
    run_dir = prepare_pptx_job(source, run_dir=tmp_path / "run")
    from test_runtime_execution import _install_component_e2e_boundaries
    _install_component_e2e_boundaries(monkeypatch)
    for page_id in ("page_001", "page_002"):
        page = run_dir / "pages" / page_id
        candidate = json.loads((page / "agent_request.json").read_text())[
            "candidates"
        ][0]
        record_decision(
            run_dir, page_id=page_id,
            object_id=candidate["source_shape_id"], decision="replace",
            confidence=0.99, category="full_slide_screenshot",
            evidence=["full-slide screenshot"],
        )
    store = RunStore.open(run_dir)

    def deterministic_execute(image, graph, actions, *, sam_runner, input_dir, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(input_dir) / "masks", output_dir / "masks")
        after = copy.deepcopy(graph)
        for action in actions:
            if action["action"] == "accept":
                after["nodes"] = [
                    {**node, "state": "pending_gate"}
                    if node["id"] in action["object_ids"] else node
                    for node in after["nodes"]
                ]
        (output_dir / "component-graph.json").write_text(
            json.dumps(after, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return after

    def deterministic_quality(*args, expected_component_ids, initial_component_count, **kwargs):
        from image2editable.component_quality import evaluate_page_quality
        fields = {
            "component_pixels": 0, "missing_pixels": 0, "missing_ratio": 0.0,
            "duplicate_pixels": 0, "duplicate_ratio": 0.0,
            "edge_missing_ratio": 0.0, "shadow_duplicate_ratio": 0.0,
            "alpha_duplicate_ratio": 0.0, "exterior_shadow_pixels": 0,
            "exterior_alpha_pixels": 0, "orphan_residual_pixels": 0,
            "text_support_pixels": 0, "text_duplicate_ratio": 0.0,
            "ownership_out_of_bounds_pixels": 0, "parent_coverage_ratio": 1.0,
            "component_overlap_pixels": 0,
            "parent_child_double": False,
            "noise_l1": 0.0, "local_contrast": 0.0, "edge_width_px": 0,
            "text_halo_px": 0, "adaptive_pixel_tolerance": 0.0,
            "hard_pixel_tolerance": 0.0,
        }
        components = [{
            "component_id": component_id, "accepted": True,
            "metrics": fields, "improvement": {}, "violations": [],
            "checks": {"protected_native_overlap": "pass"},
            "agent_confidence": 1.0,
        } for component_id in expected_component_ids]
        return evaluate_page_quality(
            components, visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
            page_checks={"pptx_reopen": "unknown"},
            expected_component_ids=expected_component_ids,
            initial_component_count=initial_component_count,
        )

    monkeypatch.setattr(legacy, "execute_component_action_round", deterministic_execute)
    monkeypatch.setattr(component_repair, "evaluate_component_quality_round", deterministic_quality)

    waiting = runtime.run_job(run_dir)
    assert waiting["status"] == "awaiting_agent"
    assert not (run_dir / "final" / "output.pptx").exists()

    from test_host_agent import _capability_response
    handshake = runtime.next_host_agent_item(run_dir)
    runtime.record_host_agent_plan(
        run_dir,
        _write_json(tmp_path / "capability.json", _capability_response(handshake)),
    )
    for index, page_id in enumerate(("page_001", "page_002"), start=1):
        request = runtime.next_host_agent_item(run_dir)
        plan = {
            "schema_version": 1, "kind": "component_plan", "page_id": page_id,
            "provider": "host", "repair_round": request["repair_round"],
            "request_sha256": request["request_sha256"], "actions": [{
                "action": "accept", "object_ids": ["component_0001"],
                "parameters": {}, "confidence": 1.0,
                "evidence": ["deterministic acceptance boundary"],
            }],
        }
        runtime.record_host_agent_plan(
            run_dir, _write_json(tmp_path / f"plan-{index}.json", plan)
        )
        result = runtime.run_job(run_dir)
        if index == 1:
            assert result["status"] == "awaiting_agent"
    summary = runtime.run_job(run_dir)
    assert summary["status"] == "completed"
    assert {item["status"] for item in summary["page_results"]} == {"replaced"}
    assert all(
        store.read_json(
            f"pages/{page_id}/reconstruction/component_state.json"
        )["phase"] == "ready_for_assembly"
        for page_id in ("page_001", "page_002")
    )
    output = Path(summary["outputs"]["pptx"])
    assert output.is_file()
    assert len(Presentation(output).slides) == 2


def test_only_approved_page_is_initialized_before_host_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).parents[1] / "test1.pptx"
    if not source.is_file():
        pytest.skip("real test1.pptx fixture is not present")
    run_dir = prepare_pptx_job(source, run_dir=tmp_path / "run")
    from test_runtime_execution import _install_component_e2e_boundaries
    _install_component_e2e_boundaries(monkeypatch)
    page = run_dir / "pages" / "page_001"
    candidate = json.loads((page / "agent_request.json").read_text())["candidates"][0]
    record_decision(
        run_dir, page_id="page_001", object_id=candidate["source_shape_id"],
        decision="replace", confidence=0.99,
        category="full_slide_screenshot", evidence=["approved page"],
    )
    summary = runtime.run_job(run_dir)
    assert summary["status"] == "awaiting_agent"
    assert (run_dir / "pages/page_001/reconstruction/component_state.json").is_file()
    assert not (run_dir / "pages/page_002/reconstruction/component_state.json").exists()
    assert not (run_dir / "final/output.pptx").exists()


def test_pptx_initialization_failure_records_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _candidate_pptx(tmp_path)
    run_dir = prepare_pptx_job(source, run_dir=tmp_path / "run")
    record_decision(
        run_dir, page_id="page_001", object_id="2",
        decision="replace", confidence=0.99,
        category="full_slide_screenshot", evidence=["approved page"],
    )
    monkeypatch.setattr(
        runtime,
        "initialize_legacy_page",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("initialization failed")
        ),
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "failed"
    assert (
        store.read_json("page_jobs.json")["pages"]["page_001"]["status"]
        == "failed"
    )


def test_full_page_second_round_reuses_bound_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_host_agent import _capability_response

    source = Path(__file__).parents[1] / "test1.pptx"
    if not source.is_file():
        pytest.skip("real test1.pptx fixture is not present")
    run_dir = prepare_pptx_job(source, run_dir=tmp_path / "run")
    from test_runtime_execution import _install_component_e2e_boundaries
    _install_component_e2e_boundaries(monkeypatch)
    candidate = json.loads(
        (run_dir / "pages/page_001/agent_request.json").read_text(
            encoding="utf-8"
        )
    )["candidates"][0]
    record_decision(
        run_dir, page_id="page_001", object_id=candidate["source_shape_id"],
        decision="replace", confidence=0.99,
        category="full_slide_screenshot", evidence=["approved page"],
    )
    assert runtime.run_job(run_dir)["status"] == "awaiting_agent"
    handshake = runtime.next_host_agent_item(run_dir)
    runtime.record_host_agent_plan(
        run_dir,
        _write_json(tmp_path / "capability.json", _capability_response(handshake)),
    )
    item = runtime.next_host_agent_item(run_dir)
    request = json.loads(Path(item["request_path"]).read_text(encoding="utf-8"))
    runtime.record_host_agent_plan(
        run_dir,
        _write_json(tmp_path / "plan.json", {
            "schema_version": 1, "kind": "component_plan",
            "page_id": item["page_id"], "provider": "host",
            "repair_round": item["repair_round"],
            "request_sha256": item["request_sha256"],
            "actions": [{
                "action": "accept", "object_ids": request["candidate_ids"],
                "parameters": {}, "confidence": 1.0,
                "evidence": ["exercise real quality boundary"],
            }],
        }),
    )

    result = runtime.run_job(run_dir)

    assert result["status"] == "awaiting_agent"
    store = RunStore.open(run_dir)
    state = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    assert state["repair_round"] == 2
    next_request = json.loads(
        (run_dir / state["current_round"]["request_ref"]["path"])
        .read_text(encoding="utf-8")
    )
    assert next_request["source_sha256"] == state["source_sha256"]

    item = runtime.next_host_agent_item(run_dir)
    repeated_request = json.loads(
        Path(item["request_path"]).read_text(encoding="utf-8")
    )
    runtime.record_host_agent_plan(
        run_dir,
        _write_json(tmp_path / "plan-02.json", {
            "schema_version": 1, "kind": "component_plan",
            "page_id": item["page_id"], "provider": "host",
            "repair_round": item["repair_round"],
            "request_sha256": item["request_sha256"],
            "actions": [{
                "action": "accept",
                "object_ids": repeated_request["candidate_ids"],
                "parameters": {}, "confidence": 1.0,
                "evidence": ["exercise repeated-plan fallback"],
            }],
        }),
    )
    from image2editable import pptx_shadow_run
    cv_calls = []

    def unexpected_cv(*args, **kwargs):
        cv_calls.append((args, kwargs))
        raise AssertionError("warning fallback must not rebuild with CV")

    monkeypatch.setattr(
        pptx_shadow_run, "build_reconstruction_donor", unexpected_cv
    )
    summary = runtime.run_job(run_dir)

    assert summary["status"] == "completed"
    assert summary["page_results"][0]["status"] == "preserved_with_warning"
    assert cv_calls == []


def test_mixed_pptx_preserves_native_inventory_without_eligible_candidate(
    tmp_path: Path,
) -> None:
    sources = [
        path for path in Path(__file__).parents[1].glob("*.pptx")
        if path.name != "test1.pptx"
    ]
    if not sources:
        pytest.skip("real mixed PPTX fixture is not present")
    source = sources[0]
    run_dir = prepare_pptx_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    before = [item["native_objects_sha256"] for item in manifest["input"]["inventories"]]
    assert manifest["input"]["candidate_count"] == 0
    assert not list(run_dir.glob("pages/*/page_request.json"))
    assert not list(run_dir.glob("pages/*/reconstruction/component_state.json"))
    summary = runtime.run_job(run_dir)
    assert summary["status"] == "completed"
    after = store.read_json("job_manifest.json")["input"]["inventories"]
    assert before == [item["native_objects_sha256"] for item in after]
    original = scan_pptx(source)
    for output_value in summary["outputs"].values():
        output = Path(output_value)
        reopened = scan_pptx(output)
        assert (reopened["slide_count"], reopened["slide_width"], reopened["slide_height"]) == (
            original["slide_count"], original["slide_width"], original["slide_height"]
        )
        for before_slide, after_slide in zip(original["slides"], reopened["slides"]):
            assert before_slide["notes_sha256"] == after_slide["notes_sha256"]
            assert [
                (item["shape_id"], item["xml_c14n_sha256"], item["z_order"])
                for item in before_slide["objects"]
            ] == [
                (item["shape_id"], item["xml_c14n_sha256"], item["z_order"])
                for item in after_slide["objects"]
            ]
