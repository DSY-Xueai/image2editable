from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from image2editable import host_agent
from image2editable.component_repair import EVIDENCE_NAMES, build_component_agent_request
from image2editable.contracts import RunStatus
from image2editable.host_agent import next_host_agent_item, record_host_plan
from image2editable.inputs import prepare_image_job
from image2editable.store import RunStore


def _node(component_id: str, state: str, z_index: int) -> dict:
    return {"id": component_id, "kind": "parent", "parent_id": None,
            "state": state, "mask": f"masks/{component_id}.png",
            "mask_sha256": "a" * 64, "bbox": [0, 0, 2, 2],
            "z_index": z_index, "text_ids": []}


@pytest.fixture
def host_run(tmp_path: Path) -> Path:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_run(RunStatus.AWAITING_AGENT)
    return run_dir


def _publish_request(run_dir: Path) -> Path:
    reconstruction = run_dir / "pages/page_001/reconstruction"
    evidence_root = reconstruction / "evidence-source"
    evidence_root.mkdir(parents=True)
    graph = {"nodes": [_node("candidate_1", "pending", 0)]}
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
    return build_component_agent_request({
        "page_id": "page_001", "provider": "host",
        "reconstruction_dir": reconstruction, "evidence": evidence,
    }, repair_round=1)


def _capability_response(item: dict) -> dict:
    return {"schema_version": 1, "kind": "host_capability_response",
            "challenge_id": item["challenge_id"],
            "observed": {"shape": "triangle", "color": "#2f6fed", "count": 3}}


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_host_next_returns_random_hash_bound_visual_handshake_before_page(host_run: Path) -> None:
    item = next_host_agent_item(host_run)
    metadata = json.loads((host_run / "host_challenge.json").read_text(encoding="utf-8"))
    assert item["kind"] == "capability_handshake"
    assert Path(item["image_path"]).is_absolute()
    assert item["required_capabilities"] == [
        "vision", "local_file_read", "tool_use", "structured_json"]
    assert metadata["challenge_id"] == item["challenge_id"]
    assert metadata["image_sha256"] == hashlib.sha256(Path(item["image_path"]).read_bytes()).hexdigest()


def test_host_module_does_not_import_or_download_local_models() -> None:
    source = Path(host_agent.__file__).read_text(encoding="utf-8")
    for forbidden in ("local_agent", "transformers", "torch", "huggingface"):
        assert forbidden not in source


def test_capability_must_match_exact_visual_answer(host_run: Path, tmp_path: Path) -> None:
    item = next_host_agent_item(host_run)
    response = _capability_response(item)
    response["observed"]["count"] = 2
    with pytest.raises(ValueError, match="capability"):
        record_host_plan(host_run, _write_json(tmp_path / "bad.json", response))
    assert not (host_run / "host_capabilities.json").exists()


def test_capability_success_exposes_bound_component_request(host_run: Path, tmp_path: Path) -> None:
    request_path = _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    item = next_host_agent_item(host_run)
    assert item["kind"] == "component_request"
    assert item["provider"] == "host"
    assert item["request_path"] == str(request_path.resolve())
    assert all(Path(path).is_absolute() for path in item["evidence_paths"])
    assert "untrusted" in item["instructions"].lower()


def test_record_plan_rejects_provider_round_hash_and_unknown_id(host_run: Path, tmp_path: Path) -> None:
    _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    request = next_host_agent_item(host_run)
    base = {"schema_version": 1, "kind": "component_plan", "page_id": request["page_id"],
            "provider": "host", "repair_round": request["repair_round"],
            "request_sha256": request["request_sha256"], "actions": []}
    variants = [({**base, "provider": "local"}, "provider"),
                ({**base, "repair_round": 2}, "repair_round"),
                ({**base, "request_sha256": "0" * 64}, "request_sha256"),
                ({**base, "actions": [{"action": "accept", "object_ids": ["unknown"],
                  "parameters": {}, "confidence": 0.9, "evidence": ["visible boundary"]}]}, "object")]
    for index, (plan, message) in enumerate(variants):
        with pytest.raises(ValueError, match=message):
            record_host_plan(host_run, _write_json(tmp_path / f"plan-{index}.json", plan))


@pytest.mark.parametrize(
    "action",
    [
        {"action": "retry_with_box", "object_ids": ["candidate_1"],
         "parameters": {"box": [-0.1, 0.1, 0.5, 0.5]},
         "confidence": 0.9, "evidence": ["edge"]},
        {"action": "accept", "object_ids": ["candidate_1"],
         "parameters": {}, "confidence": float("nan"), "evidence": ["edge"]},
        {"action": "accept", "object_ids": ["candidate_1"],
         "parameters": {}, "confidence": 0.9, "evidence": []},
    ],
)
def test_record_plan_rejects_invalid_coordinates_confidence_and_evidence(
    host_run: Path, tmp_path: Path, action: dict
) -> None:
    _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    request = next_host_agent_item(host_run)
    plan = {"schema_version": 1, "kind": "component_plan", "page_id": "page_001",
            "provider": "host", "repair_round": 1,
            "request_sha256": request["request_sha256"], "actions": [action]}
    with pytest.raises(ValueError):
        record_host_plan(host_run, _write_json(tmp_path / "plan.json", plan))


def test_record_valid_plan_is_atomic_resumable_and_not_repeatable(host_run: Path, tmp_path: Path) -> None:
    _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    request = next_host_agent_item(host_run)
    plan_path = _write_json(tmp_path / "plan.json", {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "host", "repair_round": 1,
        "request_sha256": request["request_sha256"], "actions": []})
    result = record_host_plan(host_run, plan_path)
    assert result["status"] == "recorded"
    assert RunStore.open(host_run).read_json("run_state.json")["status"] == "prepared"
    assert Path(result["plan_path"]).is_file()
    with pytest.raises(RuntimeError, match="already recorded"):
        record_host_plan(host_run, plan_path)
