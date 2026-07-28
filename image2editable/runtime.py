from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from image2editable.contracts import (
    PageStatus,
    RunStatus,
    SCHEMA_VERSION,
    transition_page_document,
    validate_schema_version,
)
from image2editable.inputs import prepare_image_job
from image2editable.legacy import execute_legacy
from image2editable.store import RunStore


def prepare_job(
    inputs: str | Path | Iterable[str | Path],
    *,
    run_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    slide_size: str = "both",
    lang: str = "ch",
) -> Path:
    return prepare_image_job(
        inputs,
        run_dir=run_dir,
        output_path=output_path,
        slide_size=slide_size,
        lang=lang,
    )


def get_status(run_dir: str | Path) -> dict[str, Any]:
    store = RunStore.open(run_dir)
    return {
        "run": store.read_json("run_state.json"),
        "pages": store.read_json("page_jobs.json"),
    }


def _transition_pages(
    store: RunStore, page_ids: Sequence[str], target: PageStatus
) -> None:
    if not page_ids:
        return
    page_jobs = store.read_json("page_jobs.json")
    pages = page_jobs["pages"]
    updates = {
        page_id: transition_page_document(pages[page_id], target)
        for page_id in page_ids
    }
    pages.update(updates)
    store.write_json("page_jobs.json", page_jobs)


def _record_failure(
    store: RunStore, page_ids: Sequence[str], error: Exception
) -> Exception | None:
    cleanup_errors = []
    try:
        pages = store.read_json("page_jobs.json")["pages"]
        failed_page_ids = [
            page_id
            for page_id in page_ids
            if pages[page_id]["status"]
            in {
                PageStatus.PENDING.value,
                PageStatus.PROCESSING.value,
                PageStatus.VALIDATED.value,
            }
        ]
        _transition_pages(store, failed_page_ids, PageStatus.FAILED)
    except Exception as cleanup_error:
        cleanup_errors.append(cleanup_error)

    try:
        status = store.read_json("run_state.json")["status"]
        if status in {RunStatus.RUNNING.value, RunStatus.FINALIZING.value}:
            store.transition_run(RunStatus.FAILED)
    except Exception as cleanup_error:
        cleanup_errors.append(cleanup_error)

    try:
        store.write_json(
            "run_summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": RunStatus.FAILED.value,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "outputs": {},
            },
        )
    except Exception as cleanup_error:
        cleanup_errors.append(cleanup_error)

    return cleanup_errors[0] if cleanup_errors else None


def run_job(run_dir: str | Path) -> dict[str, Any]:
    store = RunStore.open(run_dir)
    state = store.read_json("run_state.json")
    if state["status"] == RunStatus.COMPLETED.value:
        summary = store.read_json("run_summary.json")
        validate_schema_version(summary)
        return summary
    if state["status"] != RunStatus.PREPARED.value:
        raise RuntimeError(
            f"Run must be prepared before execution; current status is {state['status']}"
        )

    page_ids = list(store.read_json("page_jobs.json")["pages"])
    store.transition_run(RunStatus.RUNNING)

    try:
        _transition_pages(store, page_ids, PageStatus.PROCESSING)
        outputs = execute_legacy(store)
        for page_id in page_ids:
            store.write_json(
                Path("pages") / page_id / "page_result.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "page_id": page_id,
                    "status": PageStatus.VALIDATED.value,
                    "outputs": outputs,
                },
            )
        _transition_pages(store, page_ids, PageStatus.VALIDATED)
        store.transition_run(RunStatus.FINALIZING)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": RunStatus.COMPLETED.value,
            "pages": len(page_ids),
            "outputs": outputs,
        }
        store.write_json("run_summary.json", summary)
        store.transition_run(RunStatus.COMPLETED)
        return summary
    except Exception as error:
        cleanup_error = _record_failure(store, page_ids, error)
        if cleanup_error is not None:
            raise error from cleanup_error
        raise


def _has_failed_summary(store: RunStore) -> bool:
    try:
        summary = store.read_json("run_summary.json")
    except FileNotFoundError:
        return False
    validate_schema_version(summary)
    return summary.get("status") == RunStatus.FAILED.value


def _reset_pages_for_retry(
    store: RunStore, page_jobs: dict[str, Any]
) -> None:
    pages = page_jobs["pages"]
    updates = {}
    for page_id, page in pages.items():
        status = PageStatus(page["status"])
        if status is PageStatus.PENDING:
            continue
        if status in {PageStatus.PROCESSING, PageStatus.VALIDATED}:
            page = transition_page_document(page, PageStatus.FAILED)
        elif status is not PageStatus.FAILED:
            raise RuntimeError(
                f"Page cannot be reset for P0 retry: {page_id} ({status.value})"
            )
        updates[page_id] = transition_page_document(page, PageStatus.PENDING)
    if updates:
        pages.update(updates)
        store.write_json("page_jobs.json", page_jobs)


def retry_page(run_dir: str | Path, page_id: str) -> dict[str, Any]:
    store = RunStore.open(run_dir)
    page_jobs = store.read_json("page_jobs.json")
    if page_id not in page_jobs["pages"]:
        raise KeyError(f"Unknown page_id: {page_id}")

    run_status = store.read_json("run_state.json")["status"]
    failed_summary = _has_failed_summary(store)
    orphaned_failed_batch = (
        run_status in {RunStatus.RUNNING.value, RunStatus.FINALIZING.value}
        and failed_summary
        and all(
            page["status"] == PageStatus.FAILED.value
            for page in page_jobs["pages"].values()
        )
    )
    if orphaned_failed_batch:
        store.transition_run(RunStatus.FAILED)
        run_status = RunStatus.FAILED.value
    retrying_failed_run = run_status == RunStatus.FAILED.value
    continuing_retry = (
        run_status == RunStatus.PREPARED.value and failed_summary
    )
    if not retrying_failed_run and not continuing_retry:
        raise RuntimeError(f"Run is not failed or continuing a retry: {page_id}")
    if run_status == RunStatus.FAILED.value:
        store.transition_run(RunStatus.PREPARED)
    _reset_pages_for_retry(store, page_jobs)
    return get_status(store.root)


def convert(
    inputs: str | Path | Iterable[str | Path],
    *,
    run_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    slide_size: str = "both",
    lang: str = "ch",
) -> dict[str, Any]:
    prepared = prepare_job(
        inputs,
        run_dir=run_dir,
        output_path=output_path,
        slide_size=slide_size,
        lang=lang,
    )
    return run_job(prepared)
