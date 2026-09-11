import json
from types import SimpleNamespace

import pytest

from scripts import real_corpus_benchmark as benchmark
from scripts.real_corpus_benchmark import summarize_attempt


def test_failed_output_is_not_a_success():
    result = summarize_attempt(
        returncode=1, summary={"status": "failed"}, pptx_exists=False,
        events=[],
    )
    assert result["status"] == "failed"
    assert result["sam_calls"] is None
    assert result["lama_calls"] is None
    assert result["host_token_count"] is None
    assert result["observed_worker_starts"] is None
    assert result["observed_worker_requests"] is None


def test_worker_reuse_is_not_a_process_start():
    result = summarize_attempt(
        returncode=0, summary={"status": "completed"}, pptx_exists=True,
        events=[
            {"event": "worker_start", "stage": "task_worker", "model": "visual"},
            {"event": "worker_start", "stage": "worker_reuse", "model": "visual"},
            {"event": "worker_finish", "model": "visual"},
            {"event": "worker_finish", "model": "visual"},
        ],
    )
    assert result["status"] == "completed_unreviewed"
    assert result["observed_worker_starts"] == 1
    assert result["observed_worker_requests"] == {"visual": 2}
    assert result["local_fast_token_count"] == 0


def test_completed_without_pptx_is_failed():
    result = summarize_attempt(
        returncode=0, summary={"status": "completed"}, pptx_exists=False,
        events=[],
    )
    assert result["status"] == "failed"


def test_measurement_records_missing_page_count_and_continues(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    sources = [tmp_path / name for name in ("one.png", "two.pdf")]
    for source in sources:
        source.write_bytes(b"input")
    output = tmp_path / "measurements"

    def failed_conversion(command, **kwargs):
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(benchmark.subprocess, "run", failed_conversion)
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *a, **k: "head" if k.get("text") else b"diff")
    benchmark.main([*(str(p) for p in sources), "--output-dir", str(output)])
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "measured"
    assert len(report["files"]) == 2
    for row in report["files"]:
        assert row["pages"] is None
        assert row["started_at"] <= row["finished_at"]
        assert row["duration_s"] >= 0
    assert report["total_conversion_s"] == sum(r["duration_s"] for r in report["files"])
    assert (output / "performance.png").is_file()
