from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterable, Sequence

from image2editable.contracts import (
    PageStatus,
    RunStatus,
    SCHEMA_VERSION,
    transition_page_document,
    utc_now,
    validate_schema_version,
)
from image2editable.inputs import classify_inputs, prepare_image_job, sha256_file
from image2editable.legacy import execute_legacy
from image2editable.store import RunStore


def _pdf_function(name: str) -> Any:
    try:
        from image2editable import pdf_input
    except ModuleNotFoundError as error:
        if error.name == "pypdfium2":
            raise ModuleNotFoundError(
                "PDF support requires pypdfium2>=5.7.1,<6"
            ) from error
        raise
    return getattr(pdf_input, name)


def prepare_pdf_job(*args: Any, **kwargs: Any) -> Path:
    return _pdf_function("prepare_pdf_job")(*args, **kwargs)


def rerender_pdf_page(*args: Any, **kwargs: Any) -> dict[str, bool]:
    return _pdf_function("rerender_pdf_page")(*args, **kwargs)


def prepare_pptx_job(*args: Any, **kwargs: Any) -> Path:
    from image2editable.pptx_input import prepare_pptx_job as prepare

    return prepare(*args, **kwargs)


def execute_pptx_preserve(store: RunStore) -> dict[str, object]:
    from image2editable.pptx_input import execute_pptx_preserve as execute

    return execute(store)


def prepare_job(
    inputs: str | Path | Iterable[str | Path],
    *,
    run_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    slide_size: str = "both",
    lang: str = "ch",
) -> Path:
    input_type, paths = classify_inputs(inputs)
    prepare = {
        "images": prepare_image_job,
        "pdf": prepare_pdf_job,
        "pptx": prepare_pptx_job,
    }[input_type]
    source: Path | list[Path] = paths if input_type == "images" else paths[0]
    return prepare(
        source,
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
    store: RunStore,
    page_ids: Sequence[str],
    error: Exception,
    *,
    recover_completed: bool = False,
    retry_blocked: bool = False,
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
                PageStatus.ANALYZED.value,
                PageStatus.PROCESSING.value,
                PageStatus.VALIDATED.value,
            }
        ]
        _transition_pages(store, failed_page_ids, PageStatus.FAILED)
    except Exception as cleanup_error:
        cleanup_errors.append(cleanup_error)

    try:
        run_state = store.read_json("run_state.json")
        status = run_state["status"]
        if status in {RunStatus.RUNNING.value, RunStatus.FINALIZING.value}:
            store.transition_run(RunStatus.FAILED)
        elif status == RunStatus.COMPLETED.value and recover_completed:
            run_state["status"] = RunStatus.FAILED.value
            run_state["updated_at"] = utc_now()
            store.write_json("run_state.json", run_state)
    except Exception as cleanup_error:
        cleanup_errors.append(cleanup_error)

    try:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": RunStatus.FAILED.value,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "outputs": {},
        }
        if retry_blocked:
            summary["retry_blocked"] = True
        store.write_json(
            "run_summary.json",
            summary,
        )
    except Exception as cleanup_error:
        cleanup_errors.append(cleanup_error)

    return cleanup_errors[0] if cleanup_errors else None


def _manifest_input(
    store: RunStore,
) -> tuple[dict[str, Any], str]:
    manifest = store.read_json("job_manifest.json")
    validate_schema_version(manifest)
    input_record = manifest.get("input")
    if not isinstance(input_record, dict):
        raise RuntimeError("Run manifest input must be an object")
    input_type = input_record.get("type")
    if input_type not in {"images", "pdf", "pptx"}:
        raise RuntimeError(f"Unsupported input type: {input_type}")
    return manifest, input_type


def _pptx_page_ids(
    manifest: dict[str, Any],
    page_jobs: dict[str, Any],
    expected_status: PageStatus,
) -> list[str]:
    manifest_pages = manifest.get("pages")
    pages = page_jobs["pages"]
    if (
        not isinstance(manifest_pages, list)
        or any(not isinstance(page_id, str) for page_id in manifest_pages)
        or len(pages) != len(manifest_pages)
        or set(pages) != set(manifest_pages)
    ):
        raise RuntimeError("PPTX manifest pages do not match page jobs")
    invalid = [
        page_id
        for page_id in manifest_pages
        if pages[page_id]["status"] != expected_status.value
    ]
    if invalid:
        raise RuntimeError(
            f"PPTX pages must be {expected_status.value}: {', '.join(invalid)}"
        )
    return manifest_pages


def _pptx_output_path(store: RunStore, manifest: dict[str, Any]) -> Path:
    options = manifest.get("options")
    if not isinstance(options, dict):
        raise RuntimeError("PPTX manifest options must be an object")
    output_value = options.get("output_path")
    if output_value is None:
        return store.root / "final" / "output.pptx"
    if not isinstance(output_value, str):
        raise RuntimeError("PPTX manifest output_path must be a string or null")
    output = Path(output_value)
    if not output.is_absolute():
        raise RuntimeError("PPTX manifest output_path must be absolute")
    return output


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _pptx_output_identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode):
        raise RuntimeError(f"PPTX output is not a regular file: {path}")
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _record_pptx_output(
    summary: dict[str, Any],
) -> tuple[Path, tuple[int, int, int, int, int], str]:
    outputs = summary.get("outputs")
    output_value = outputs.get("pptx") if isinstance(outputs, dict) else None
    expected_hash = summary.get("output_sha256")
    if not isinstance(output_value, str) or not isinstance(expected_hash, str):
        raise RuntimeError("PPTX execution summary is missing output identity")
    output = Path(output_value)
    if not output.is_absolute():
        raise RuntimeError("PPTX execution output path must be absolute")
    identity = _pptx_output_identity(output)
    if sha256_file(output) != expected_hash:
        raise RuntimeError("PPTX execution output hash does not match summary")
    if _pptx_output_identity(output) != identity:
        raise RuntimeError("PPTX execution output changed while recording identity")
    return output, identity, expected_hash


def _restore_isolated_pptx_output(
    isolated: Path,
    output: Path,
) -> None:
    try:
        if stat.S_ISREG(isolated.lstat().st_mode):
            from image2editable.pptx_input import _publish_pptx_no_clobber

            _publish_pptx_no_clobber(isolated, output)
        else:
            os.link(isolated, output, follow_symlinks=False)
        isolated.unlink()
    except Exception as error:
        raise RuntimeError(
            f"Concurrent PPTX output was preserved at {isolated}"
        ) from error


def _isolate_recorded_pptx_output(
    record: tuple[Path, tuple[int, int, int, int, int], str],
) -> None:
    output, expected_identity, expected_hash = record
    descriptor, isolated_value = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.recovery-",
        suffix=".tmp",
    )
    os.close(descriptor)
    isolated = Path(isolated_value)
    try:
        os.replace(output, isolated)
    except FileNotFoundError:
        isolated.unlink(missing_ok=True)
        return
    except Exception:
        isolated.unlink(missing_ok=True)
        raise

    try:
        identity = _pptx_output_identity(isolated)
        digest = sha256_file(isolated)
        stable_identity = _pptx_output_identity(isolated)
    except Exception as error:
        _restore_isolated_pptx_output(isolated, output)
        raise RuntimeError(
            "PPTX output cannot be safely verified for removal"
        ) from error
    if (
        identity != expected_identity
        or stable_identity != expected_identity
        or digest != expected_hash
    ):
        _restore_isolated_pptx_output(isolated, output)
        raise RuntimeError(
            "PPTX output changed and cannot be safely removed"
        )
    isolated.unlink()


def run_job(run_dir: str | Path) -> dict[str, Any]:
    store = RunStore.open(run_dir)
    manifest, input_type = _manifest_input(store)
    state = store.read_json("run_state.json")
    page_jobs = store.read_json("page_jobs.json")
    if state["status"] == RunStatus.COMPLETED.value:
        if input_type == "pptx":
            _pptx_page_ids(manifest, page_jobs, PageStatus.PRESERVED)
        summary = store.read_json("run_summary.json")
        validate_schema_version(summary)
        return summary
    if state["status"] != RunStatus.PREPARED.value:
        raise RuntimeError(
            f"Run must be prepared before execution; current status is {state['status']}"
        )

    if input_type == "pptx":
        page_ids = _pptx_page_ids(
            manifest, page_jobs, PageStatus.ANALYZED
        )
        pptx_expected_output = _pptx_output_path(store, manifest)
        pptx_output_existed = _path_entry_exists(pptx_expected_output)
    else:
        page_ids = list(page_jobs["pages"])
        pptx_expected_output = None
        pptx_output_existed = False
    store.transition_run(RunStatus.RUNNING)
    pptx_output_published = False
    pptx_output_record = None

    try:
        if input_type == "pptx":
            summary = execute_pptx_preserve(store)
            pptx_output_published = True
            pptx_output_record = _record_pptx_output(summary)
            validate_schema_version(summary)
            _transition_pages(store, page_ids, PageStatus.PRESERVED)
            store.transition_run(RunStatus.FINALIZING)
        else:
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
        compensation_error = None
        pptx_output_removed = False
        pages_restored = False
        if (
            input_type == "pptx"
            and not pptx_output_published
            and not pptx_output_existed
            and pptx_expected_output is not None
            and _path_entry_exists(pptx_expected_output)
        ):
            pptx_output_published = True
        retry_blocked = (
            input_type == "pptx"
            and pptx_output_published
            and pptx_output_record is None
        )
        if input_type == "pptx" and pptx_output_published:
            try:
                if pptx_output_record is None:
                    _transition_pages(
                        store, page_ids, PageStatus.PRESERVED
                    )
                else:
                    _isolate_recorded_pptx_output(pptx_output_record)
                    pptx_output_removed = True
                    store.write_json("page_jobs.json", page_jobs)
                    pages_restored = True
            except Exception as caught:
                compensation_error = caught
                if pptx_output_removed:
                    try:
                        store.write_json("page_jobs.json", page_jobs)
                        pages_restored = True
                    except Exception as retry_error:
                        compensation_error.__cause__ = retry_error
                if not (pptx_output_removed and pages_restored):
                    retry_blocked = True
        cleanup_error = _record_failure(
            store,
            page_ids,
            error,
            recover_completed=(
                input_type == "pptx" and pptx_output_published
            ),
            retry_blocked=retry_blocked,
        )
        if compensation_error is not None:
            raise error from compensation_error
        if cleanup_error is not None:
            raise error from cleanup_error
        raise


def _failed_summary(store: RunStore) -> dict[str, Any] | None:
    try:
        summary = store.read_json("run_summary.json")
    except FileNotFoundError:
        return None
    validate_schema_version(summary)
    if summary.get("status") != RunStatus.FAILED.value:
        return None
    return summary


def _reset_pages_for_retry(
    store: RunStore,
    page_jobs: dict[str, Any],
    *,
    analyzed: bool = False,
) -> None:
    pages = page_jobs["pages"]
    updates = {}
    for page_id, page in pages.items():
        status = PageStatus(page["status"])
        if status is PageStatus.PENDING:
            if analyzed:
                updates[page_id] = transition_page_document(
                    page, PageStatus.ANALYZED
                )
            continue
        if status is PageStatus.ANALYZED and analyzed:
            continue
        if status is PageStatus.PRESERVED and analyzed:
            raise RuntimeError(
                f"PPTX retry is blocked because its output could not be "
                f"safely recovered: {page_id}"
            )
        if status in {PageStatus.PROCESSING, PageStatus.VALIDATED}:
            page = transition_page_document(page, PageStatus.FAILED)
        elif status is not PageStatus.FAILED:
            raise RuntimeError(
                f"Page cannot be reset for P0 retry: {page_id} ({status.value})"
            )
        page = transition_page_document(page, PageStatus.PENDING)
        if analyzed:
            page = transition_page_document(page, PageStatus.ANALYZED)
        updates[page_id] = page
    if updates:
        pages.update(updates)
        store.write_json("page_jobs.json", page_jobs)


def retry_page(run_dir: str | Path, page_id: str) -> dict[str, Any]:
    store = RunStore.open(run_dir)
    manifest, input_type = _manifest_input(store)
    page_jobs = store.read_json("page_jobs.json")
    if page_id not in page_jobs["pages"]:
        raise KeyError(f"Unknown page_id: {page_id}")

    run_status = store.read_json("run_state.json")["status"]
    failed_summary = _failed_summary(store)
    has_failed_summary = failed_summary is not None
    orphaned_failed_batch = (
        run_status in {RunStatus.RUNNING.value, RunStatus.FINALIZING.value}
        and has_failed_summary
        and all(
            page["status"] == PageStatus.FAILED.value
            for page in page_jobs["pages"].values()
        )
    )
    retrying_failed_run = (
        run_status == RunStatus.FAILED.value or orphaned_failed_batch
    )
    continuing_retry = (
        run_status == RunStatus.PREPARED.value and has_failed_summary
    )
    if not retrying_failed_run and not continuing_retry:
        raise RuntimeError(f"Run is not failed or continuing a retry: {page_id}")
    if input_type == "pptx" and (
        (
            failed_summary is not None
            and failed_summary.get("retry_blocked") is True
        )
        or _path_entry_exists(_pptx_output_path(store, manifest))
    ):
        raise RuntimeError(
            f"PPTX retry is blocked while an output entry may be owned "
            f"by another process: {page_id}"
        )
    if orphaned_failed_batch:
        store.transition_run(RunStatus.FAILED)
        run_status = RunStatus.FAILED.value
    _reset_pages_for_retry(
        store, page_jobs, analyzed=input_type == "pptx"
    )
    if run_status == RunStatus.FAILED.value:
        store.transition_run(RunStatus.PREPARED)
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
