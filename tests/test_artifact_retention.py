from __future__ import annotations

import json
from pathlib import Path

import pytest

from image2editable import runtime
from image2editable.inputs import prepare_image_job
from image2editable.store import RunStore


def test_performance_summary_exposes_content_free_run_totals(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    trace = runtime._page_performance_trace(store, "page_001")
    assert trace is not None
    trace.event("model_load_start", page_id="page_001", model="ocr")
    trace.event(
        "model_load_finish",
        page_id="page_001",
        model="ocr",
        duration_ms=11,
        status="success",
    )
    reconstruction = run_dir / "pages/page_001/reconstruction"
    reconstruction.mkdir()
    (reconstruction / "component_state.json").write_text(
        json.dumps({"route": "pdf_native"}), encoding="utf-8"
    )

    summary = runtime._performance_summary(
        store, ["page_001"], total_duration_ms=321
    )

    assert summary["agent_runs"] == 0
    assert summary["agent_image_count"] == 0
    assert summary["agent_total_bytes"] == 0
    assert summary["model_loads"] == {"ocr": 1}
    assert summary["route_counts"] == {"pdf_native": 1}
    assert summary["local_fidelity_pages"] == []
    assert summary["total_duration_ms"] == 321
    serialized = json.dumps(summary).casefold()
    assert str(source).casefold() not in serialized


def test_completed_legacy_cleanup_removes_disposable_artifacts(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run")
    reconstruction = store.root / "pages/page_001/reconstruction"
    strict = reconstruction / "strict-escalation-02/graph"
    strict.mkdir(parents=True)
    (strict / "component.mask.png").write_bytes(b"mask")
    (reconstruction / "worker-ocr.json").write_text("{}", encoding="utf-8")
    (reconstruction / "leftover.mask.png").write_bytes(b"mask")
    for name in (
        "component_state.json",
        "component_result.json",
        "component_delivery.json",
        "performance-summary.json",
    ):
        (reconstruction / name).write_text("{}", encoding="utf-8")

    runtime._prune_completed_legacy_artifacts(store, ["page_001"])

    assert not strict.parent.exists()
    assert not (reconstruction / "worker-ocr.json").exists()
    assert not (reconstruction / "leftover.mask.png").exists()
    assert sorted(path.name for path in reconstruction.iterdir()) == [
        "component_delivery.json",
        "component_result.json",
        "component_state.json",
        "performance-summary.json",
    ]


@pytest.mark.parametrize("repair_round", [1, 3])
@pytest.mark.parametrize("quality_state", ["accepted", "wrong_graph", "parent_preserved"])
def test_completed_cleanup_preserves_release_quality_evidence(
    tmp_path: Path, repair_round: int, quality_state: str,
) -> None:
    from scripts.release_benchmark import (
        BenchmarkCaseResult, BenchmarkFailure, _validate_batch_case,
    )

    store = RunStore(tmp_path / "run")
    reconstruction = store.root / "pages/page_001/reconstruction"
    reconstruction.mkdir(parents=True)
    graph_sha256 = "a" * 64
    component_result = {
        "final_component_ids": ["visual_0001"], "text_items": [],
        "repair_rounds": repair_round, "accepted_graph_sha256": graph_sha256,
        "warning": None, "fallback": {"status": "none", "parent_ids": []},
    }
    if quality_state == "parent_preserved":
        component_result["accepted_graph_sha256"] = "b" * 64
        component_result["fallback"] = {
            "status": "parent_preserved", "parent_ids": ["visual_0001"],
        }
    (reconstruction / "component_result.json").write_text(
        json.dumps(component_result), encoding="utf-8",
    )
    quality_bytes = json.dumps({
        "page_id": "page_001", "repair_round": repair_round,
        "input_graph_sha256": graph_sha256,
        "report": {
            "visual_metrics": {"unexplained_visual_pixels": 0},
            "violations": [], "component_reports": [],
        },
    }).encode("utf-8")
    quality_path = reconstruction / f"execution-{repair_round:02d}/component-quality.json"
    for number in range(1, repair_round + 2):
        execution = reconstruction / f"execution-{number:02d}"
        (execution / "layers").mkdir(parents=True)
        (execution / "layers/visual.png").write_bytes(b"disposable pixels")
        (execution / "preview.png").write_bytes(b"disposable preview")
        (execution / "component-quality.json").write_bytes(quality_bytes)
    case = {"expected_pages": [{
        "expected_status": "validated", "min_visual_components": 1,
        "min_text_boxes": 0, "max_unexplained_pixels": 0,
        "max_quality_violations": 0,
    }]}
    result = BenchmarkCaseResult(
        "image-one", str(store.root),
        [{"page_id": "page_001", "status": "validated"}], 1,
    )
    if quality_state == "parent_preserved":
        with pytest.raises(BenchmarkFailure, match="warning_fallback"):
            _validate_batch_case(case, result)
    else:
        _validate_batch_case(case, result)

    if quality_state == "wrong_graph":
        quality = json.loads(quality_bytes)
        quality["input_graph_sha256"] = "b" * 64
        quality_path.write_text(json.dumps(quality), encoding="utf-8")
        with pytest.raises(RuntimeError, match="quality.*binding"):
            runtime._prune_completed_legacy_artifacts(store, ["page_001"])
        assert (quality_path.parent / "preview.png").is_file()
        assert (reconstruction / f"execution-{repair_round + 1:02d}").is_dir()
        return

    runtime._prune_completed_legacy_artifacts(store, ["page_001"])

    if quality_state == "parent_preserved":
        with pytest.raises(BenchmarkFailure, match="warning_fallback"):
            _validate_batch_case(case, result)
    else:
        _validate_batch_case(case, result)
    assert quality_path.read_bytes() == quality_bytes
    assert sorted(path.relative_to(reconstruction).as_posix()
                  for path in reconstruction.rglob("*") if path.is_file()) == [
        "component_result.json",
        f"execution-{repair_round:02d}/component-quality.json",
    ]
