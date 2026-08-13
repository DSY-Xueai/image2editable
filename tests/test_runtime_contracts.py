from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json

import pytest

from image2editable import runtime
from image2editable.contracts import (
    PageStatus,
    RunStatus,
    SCHEMA_VERSION,
    transition_page_document,
    transition_run_document,
    validate_schema_version,
)
from image2editable.inputs import prepare_image_job


EXPECTED_RUN_TRANSITIONS = {
    RunStatus.CREATED: {RunStatus.PREPARED, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.PREPARED: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.RUNNING: {
        RunStatus.AWAITING_AGENT,
        RunStatus.FINALIZING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.AWAITING_AGENT: {RunStatus.PREPARED},
    RunStatus.FINALIZING: {RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: {RunStatus.PREPARED},
    RunStatus.CANCELLED: set(),
}

EXPECTED_PAGE_TRANSITIONS = {
    PageStatus.PENDING: {PageStatus.ANALYZED, PageStatus.PROCESSING, PageStatus.FAILED},
    PageStatus.ANALYZED: {PageStatus.PRESERVED, PageStatus.PROCESSING, PageStatus.FAILED},
    PageStatus.PRESERVED: set(),
    PageStatus.PROCESSING: {
        PageStatus.AWAITING_AGENT,
        PageStatus.VALIDATED,
        PageStatus.PRESERVED_WITH_WARNING,
        PageStatus.FAILED,
    },
    PageStatus.AWAITING_AGENT: {PageStatus.PROCESSING},
    PageStatus.VALIDATED: {
        PageStatus.REPLACED,
        PageStatus.PRESERVED_WITH_WARNING,
        PageStatus.FAILED,
    },
    PageStatus.REPLACED: set(),
    PageStatus.PRESERVED_WITH_WARNING: set(),
    PageStatus.FAILED: {PageStatus.PENDING},
}


@pytest.mark.parametrize("provider", [None, "remote"])
def test_runtime_rejects_missing_or_invalid_manifest_agent_provider(
    tmp_path, provider: object
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    manifest_path = run_dir / "job_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if provider is None:
        del manifest["options"]["agent_provider"]
    else:
        manifest["options"]["agent_provider"] = provider
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest.*agent_provider"):
        runtime.run_job(run_dir)


@pytest.mark.parametrize("provider", [None, "remote"])
def test_runtime_rejects_invalid_provider_before_agent_delegation(
    tmp_path, provider: object
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    manifest_path = run_dir / "job_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if provider is None:
        del manifest["options"]["agent_provider"]
    else:
        manifest["options"]["agent_provider"] = provider
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest.*agent_provider"):
        runtime.next_candidate(run_dir)


@pytest.mark.parametrize("provider", [None, "remote"])
def test_get_status_rejects_missing_or_invalid_manifest_agent_provider(
    tmp_path, provider: object
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    manifest_path = run_dir / "job_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if provider is None:
        del manifest["options"]["agent_provider"]
    else:
        manifest["options"]["agent_provider"] = provider
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest.*agent_provider"):
        runtime.get_status(run_dir)


@pytest.mark.parametrize(
    "entry_name",
    ["record_decision", "rerender_pdf_page", "recover_job", "retry_page"],
)
@pytest.mark.parametrize("provider", [None, "remote"])
def test_public_runtime_entries_reject_missing_or_invalid_agent_provider(
    tmp_path, entry_name: str, provider: object
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = runtime.RunStore.open(run_dir)
    if entry_name == "recover_job":
        store.transition_run(RunStatus.RUNNING)
    manifest_path = run_dir / "job_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if provider is None:
        del manifest["options"]["agent_provider"]
    else:
        manifest["options"]["agent_provider"] = provider
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    calls = {
        "record_decision": lambda: runtime.record_decision(
            run_dir,
            page_id="page_001",
            object_id="candidate_001",
            decision="preserve",
            confidence=1.0,
            category="unknown",
            evidence=["test"],
        ),
        "rerender_pdf_page": lambda: runtime.rerender_pdf_page(
            run_dir, "page_001"
        ),
        "recover_job": lambda: runtime.recover_job(run_dir),
        "retry_page": lambda: runtime.retry_page(run_dir, "page_001"),
    }
    with pytest.raises(RuntimeError, match="manifest.*agent_provider"):
        calls[entry_name]()
    if entry_name == "recover_job":
        assert store.read_json("run_state.json")["status"] == "running"


def test_run_state_rejects_skipped_transition() -> None:
    state = {
        "schema_version": SCHEMA_VERSION,
        "status": RunStatus.CREATED.value,
        "updated_at": "2026-07-28T00:00:00Z",
    }

    with pytest.raises(ValueError, match="created -> running"):
        transition_run_document(state, RunStatus.RUNNING)


def test_page_failure_can_return_to_pending() -> None:
    page = {
        "schema_version": SCHEMA_VERSION,
        "status": PageStatus.FAILED.value,
        "updated_at": "2026-07-28T00:00:00Z",
        "metadata": {"attempt": 1},
    }
    original = deepcopy(page)

    updated = transition_page_document(page, PageStatus.PENDING)

    assert updated["status"] == PageStatus.PENDING.value
    assert updated["updated_at"].endswith("Z")
    assert updated["metadata"] is not page["metadata"]
    assert page == original


def test_run_document_transition_is_immutable_and_uses_utc_timestamp() -> None:
    run = {
        "schema_version": SCHEMA_VERSION,
        "status": RunStatus.CREATED.value,
        "updated_at": "2026-07-28T00:00:00Z",
        "metadata": {"attempt": 1},
    }
    original = deepcopy(run)

    updated = transition_run_document(run, RunStatus.PREPARED)
    updated_at = datetime.fromisoformat(updated["updated_at"].replace("Z", "+00:00"))

    assert updated["status"] == RunStatus.PREPARED.value
    assert updated_at.utcoffset() == timedelta(0)
    assert updated["metadata"] is not run["metadata"]
    assert run == original


@pytest.mark.parametrize("version", [2, True, 1.0, "1"])
def test_schema_version_requires_exact_integer_one(version: object) -> None:
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        validate_schema_version({"schema_version": version})


@pytest.mark.parametrize(
    ("transition", "document", "target", "enum_name"),
    [
        (
            transition_run_document,
            {"schema_version": SCHEMA_VERSION, "status": RunStatus.CREATED.value},
            PageStatus.FAILED,
            "RunStatus",
        ),
        (
            transition_page_document,
            {"schema_version": SCHEMA_VERSION, "status": PageStatus.PENDING.value},
            RunStatus.FAILED,
            "PageStatus",
        ),
        (
            transition_run_document,
            {"schema_version": SCHEMA_VERSION, "status": RunStatus.CREATED.value},
            "prepared",
            "RunStatus",
        ),
        (
            transition_page_document,
            {"schema_version": SCHEMA_VERSION, "status": PageStatus.PENDING.value},
            "processing",
            "PageStatus",
        ),
    ],
)
def test_transitions_require_their_own_status_enum(
    transition: object, document: dict[str, object], target: object, enum_name: str
) -> None:
    with pytest.raises(TypeError, match=enum_name):
        transition(document, target)  # type: ignore[operator]


def test_bound_performance_trace_refuses_file_limit_growth(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = runtime.RunStore.open(run_dir)
    trace = runtime._page_performance_trace(store, "page_001")
    assert trace is not None
    assert trace.path == run_dir / "performance-page_001.jsonl"
    monkeypatch.setattr(runtime, "_PERFORMANCE_TRACE_LIMIT", 32)

    with pytest.raises(RuntimeError, match="size limit"):
        trace.event(
            "span", page_id="page_001", stage="visual_prepare", duration_ms=1
        )

    assert trace.path.stat().st_size == 0


def test_runtime_performance_summary_contract_rejects_content_fields() -> None:
    valid = {
        "pages": {
            "page_001": {
                "model_loads": {"sam": 1},
                "stage_runs": {"visual_prepare": 1},
                "stage_duration_ms": {"visual_prepare": 4},
                "worker_runs": {},
                "worker_duration_ms": {},
                "inference_runs": {"sam": 1},
                "inference_operations": {"sam": 2},
                "inference_duration_ms": {"sam": 3},
                "agent_runs": 1,
                "agent_image_count": 4,
                "agent_total_bytes": 20,
                "agent_duration_ms": 5,
            }
        }
    }

    runtime._validate_performance_summary(valid, ["page_001"])

    invalid = deepcopy(valid)
    invalid["pages"]["page_001"]["source_path"] = "secret.png"
    with pytest.raises(RuntimeError, match="performance summary"):
        runtime._validate_performance_summary(invalid, ["page_001"])


def test_invalid_performance_event_is_skipped_atomically(monkeypatch) -> None:
    summary = runtime._empty_page_performance()
    first = {
        "event": "local_agent", "image_count": 2, "total_bytes": 1,
        "duration_ms": 1, "status": "success",
    }
    runtime._aggregate_performance_event(summary, first)
    monkeypatch.setattr(runtime, "_PERFORMANCE_MAX_INTEGER", 2)

    candidate = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in summary.items()
    }
    with pytest.raises(ValueError, match="integer limit"):
        runtime._aggregate_performance_event(candidate, first)

    assert summary["agent_runs"] == 1
    assert summary["agent_image_count"] == 2
    assert candidate == summary


def test_unpaired_model_load_finish_does_not_increment_summary() -> None:
    summary = runtime._empty_page_performance()
    pending = set()

    with pytest.raises(ValueError, match="model load"):
        runtime._aggregate_performance_event(
            summary,
            {
                "event": "model_load_finish",
                "page_id": "page_001",
                "model": "sam",
                "duration_ms": 1,
                "status": "success",
            },
            pending_model_loads=pending,
            page_id="page_001",
        )

    assert summary["model_loads"] == {}


def test_empty_page_trace_still_produces_empty_page_summary(tmp_path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = runtime.RunStore.open(run_dir)

    summary = runtime._performance_summary(store, ["page_001"])

    assert summary == {"pages": {"page_001": runtime._empty_page_performance()}}
    assert (run_dir / "performance-page_001.jsonl").is_file()


def test_performance_summary_write_failure_is_isolated(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = runtime.RunStore.open(run_dir)
    trace = runtime._page_performance_trace(store, "page_001")
    assert trace is not None
    trace.event(
        "span", page_id="page_001", stage="visual_prepare", duration_ms=1
    )

    monkeypatch.setattr(
        runtime,
        "_run_owned_directory",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unsafe")),
    )

    summary = runtime._read_page_performance(store, "page_001")
    runtime._write_performance_summaries(
        store, {"pages": {"page_001": summary}}
    )

    assert summary is not None
    assert summary["stage_runs"] == {"visual_prepare": 1}
    assert "Performance summary could not be written" in caplog.text


def test_performance_trace_concurrent_appends_never_exceed_limit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = runtime.RunStore.open(run_dir)
    traces = [runtime._page_performance_trace(store, "page_001") for _ in range(16)]
    assert all(trace is not None for trace in traces)
    monkeypatch.setattr(runtime, "_PERFORMANCE_TRACE_LIMIT", 512)

    def append(index: int) -> None:
        try:
            traces[index].event(
                "span",
                page_id="page_001",
                stage="visual_prepare",
                duration_ms=index,
            )
        except RuntimeError:
            pass

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(append, range(16)))

    payload = traces[0].path.read_bytes()
    assert len(payload) <= runtime._PERFORMANCE_TRACE_LIMIT
    assert all(json.loads(line)["event"] == "span" for line in payload.splitlines())


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (current, target)
        for current, targets in EXPECTED_RUN_TRANSITIONS.items()
        for target in targets
    ],
)
def test_run_state_allows_expected_transitions(
    current: RunStatus, target: RunStatus
) -> None:
    updated = transition_run_document(
        {"schema_version": SCHEMA_VERSION, "status": current.value}, target
    )

    assert updated["status"] == target.value


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (current, target)
        for current in RunStatus
        for target in RunStatus
        if target not in EXPECTED_RUN_TRANSITIONS[current]
    ],
)
def test_run_state_rejects_unexpected_transitions(
    current: RunStatus, target: RunStatus
) -> None:
    with pytest.raises(ValueError, match=f"{current.value} -> {target.value}"):
        transition_run_document(
            {"schema_version": SCHEMA_VERSION, "status": current.value}, target
        )


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (current, target)
        for current, targets in EXPECTED_PAGE_TRANSITIONS.items()
        for target in targets
    ],
)
def test_page_state_allows_expected_transitions(
    current: PageStatus, target: PageStatus
) -> None:
    updated = transition_page_document(
        {"schema_version": SCHEMA_VERSION, "status": current.value}, target
    )

    assert updated["status"] == target.value


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (current, target)
        for current in PageStatus
        for target in PageStatus
        if target not in EXPECTED_PAGE_TRANSITIONS[current]
    ],
)
def test_page_state_rejects_unexpected_transitions(
    current: PageStatus, target: PageStatus
) -> None:
    with pytest.raises(ValueError, match=f"{current.value} -> {target.value}"):
        transition_page_document(
            {"schema_version": SCHEMA_VERSION, "status": current.value}, target
        )
