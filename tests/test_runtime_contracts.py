from __future__ import annotations

from copy import deepcopy
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
    PageStatus.VALIDATED: {PageStatus.REPLACED, PageStatus.FAILED},
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
