from __future__ import annotations

import json
from pathlib import Path

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
