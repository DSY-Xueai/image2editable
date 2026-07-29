from __future__ import annotations

from contextvars import ContextVar
import os
from pathlib import Path
import secrets
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
from image2editable.execution import ExecutionLease
from image2editable.legacy import _safe_rmtree, execute_legacy
from image2editable.resources import (
    apply_resource_policy,
    validate_resource_policy,
)
from image2editable.store import RunStore


_PPTX_EXECUTION_MANIFEST: ContextVar[dict[str, Any] | None] = ContextVar(
    "_PPTX_EXECUTION_MANIFEST", default=None
)


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

    return execute(store, _PPTX_EXECUTION_MANIFEST.get())


def validate_pptx_inventories(
    store: RunStore, manifest: dict[str, Any]
) -> tuple[int, int]:
    from image2editable.pptx_input import validate_pptx_inventories as validate

    return validate(store, manifest)


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
        try:
            work = _run_work_directory(store)
        except (OSError, RuntimeError):
            work = None
        if work is not None:
            summary["diagnostics"] = str(work[0])
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


def _manifest_resource_policy(manifest: dict[str, Any]) -> dict[str, object]:
    options = manifest.get("options")
    if type(options) is not dict:
        raise ValueError("Run manifest resource policy requires options")
    return validate_resource_policy(options.get("resource_policy"))


def _validate_completed_resource_policy(
    summary: dict[str, Any],
    resource_policy: dict[str, object],
) -> None:
    try:
        summary_policy = validate_resource_policy(
            summary.get("resource_policy")
        )
    except ValueError as error:
        raise RuntimeError(
            "Run completion summary resource policy is invalid"
        ) from error
    if summary_policy != resource_policy:
        raise RuntimeError(
            "Run completion summary resource policy does not match manifest"
        )


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


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )


def _run_work_directory(
    store: RunStore,
) -> tuple[Path, tuple[int, int]] | None:
    return _run_owned_directory(store, "work")


def _run_owned_directory(
    store: RunStore,
    name: str,
) -> tuple[Path, tuple[int, int]] | None:
    path = store.root / name
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    if _is_link_or_reparse(status):
        raise RuntimeError(
            f"Run {name} directory is a link or reparse point: {path}"
        )
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"Run {name} path is not a directory: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(store.root):
        raise RuntimeError(
            f"Run {name} directory is outside run directory: {path}"
        )
    return resolved, (status.st_dev, status.st_ino)


def _pptx_output_identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode):
        raise RuntimeError(f"PPTX output is not a regular file: {path}")
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _validate_completed_pptx_output(path: Path, expected_sha256: str) -> None:
    try:
        identity = _pptx_output_identity(path)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"PPTX completed output is not a regular file: {path}"
        ) from error
    try:
        digest = sha256_file(path)
        stable_identity = _pptx_output_identity(path)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"PPTX completed output cannot be verified: {path}"
        ) from error
    if stable_identity != identity:
        raise RuntimeError(
            f"PPTX completed output changed during verification: {path}"
        )
    if digest != expected_sha256:
        raise RuntimeError(
            f"PPTX completed output hash does not match manifest: {path}"
        )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _pptx_manifest_expectations(
    manifest: dict[str, Any],
    page_jobs: dict[str, Any],
) -> tuple[int, int, int, str]:
    input_record = manifest["input"]
    slide_count = input_record.get("slide_count")
    preserved_objects = input_record.get("object_count")
    pending_candidates = input_record.get("candidate_count")
    input_sha256 = input_record.get("sha256")
    manifest_pages = manifest.get("pages")
    job_pages = page_jobs.get("pages")
    if (
        type(slide_count) is not int
        or slide_count < 0
        or not isinstance(manifest_pages, list)
        or not isinstance(job_pages, dict)
        or slide_count != len(manifest_pages)
        or slide_count != len(job_pages)
    ):
        raise RuntimeError("PPTX manifest slide_count is invalid")
    if (
        type(preserved_objects) is not int
        or preserved_objects < 0
        or type(pending_candidates) is not int
        or pending_candidates < 0
        or pending_candidates > preserved_objects
    ):
        raise RuntimeError("PPTX manifest input counts are invalid")
    if not _is_sha256(input_sha256):
        raise RuntimeError("PPTX manifest input sha256 is invalid")
    return slide_count, preserved_objects, pending_candidates, input_sha256


def _validate_pptx_public_summary(
    summary: object,
    expected_output: Path,
    slide_count: int,
    preserved_objects: int,
    pending_candidates: int,
    input_sha256: str,
    resource_policy: dict[str, object],
) -> None:
    if type(summary) is not dict:
        raise RuntimeError("PPTX execution summary must be an object")
    outputs = summary.get("outputs")
    if (
        type(outputs) is not dict
        or outputs != {"pptx": str(expected_output)}
    ):
        raise RuntimeError(
            "PPTX execution summary did not return the expected output path"
        )
    expected_public_keys = {
        "schema_version",
        "status",
        "pages",
        "preserved_objects",
        "pending_candidates",
        "warnings",
        "outputs",
        "input_sha256",
        "output_sha256",
        "resource_policy",
    }
    if set(summary) != expected_public_keys:
        raise RuntimeError("PPTX execution summary fields are invalid")
    try:
        summary_resource_policy = validate_resource_policy(
            summary.get("resource_policy")
        )
    except ValueError as error:
        raise RuntimeError(
            "PPTX execution summary resource policy is invalid"
        ) from error
    warnings = summary.get("warnings")
    expected_warnings = (
        ["P1 preserved screenshot candidates without replacement"]
        if pending_candidates
        else []
    )
    if (
        type(summary.get("schema_version")) is not int
        or summary["schema_version"] != SCHEMA_VERSION
        or type(summary.get("status")) is not str
        or summary["status"] != RunStatus.COMPLETED.value
        or type(summary.get("pages")) is not int
        or summary["pages"] != slide_count
        or type(summary.get("preserved_objects")) is not int
        or summary["preserved_objects"] != preserved_objects
        or type(summary.get("pending_candidates")) is not int
        or summary["pending_candidates"] != pending_candidates
        or type(warnings) is not list
        or any(type(warning) is not str for warning in warnings)
        or warnings != expected_warnings
        or summary_resource_policy != resource_policy
    ):
        raise RuntimeError("PPTX execution summary values are invalid")
    if (
        not _is_sha256(summary.get("input_sha256"))
        or summary["input_sha256"] != input_sha256
        or not _is_sha256(summary.get("output_sha256"))
        or summary["output_sha256"] != input_sha256
    ):
        raise RuntimeError("PPTX execution summary hash does not match manifest")


def _validate_pptx_execution_summary(
    summary: object,
    expected_output: Path,
    slide_count: int,
    preserved_objects: int,
    pending_candidates: int,
    input_sha256: str,
    resource_policy: dict[str, object],
) -> None:
    if type(summary) is not dict:
        raise RuntimeError("PPTX execution summary must be an object")
    public_summary = dict(summary)
    token = public_summary.pop("_output_identity", None)
    _validate_pptx_public_summary(
        public_summary,
        expected_output,
        slide_count,
        preserved_objects,
        pending_candidates,
        input_sha256,
        resource_policy,
    )
    if (
        not isinstance(token, dict)
        or not _is_sha256(token.get("sha256"))
        or token["sha256"] != input_sha256
    ):
        raise RuntimeError(
            "PPTX execution summary or identity token hash does not match manifest"
        )


def _claim_pptx_output(
    summary: dict[str, Any],
    expected_output: Path,
    output_existed: bool,
) -> tuple[Path, tuple[int, int, int, int, int], str]:
    token = summary.get("_output_identity")
    if output_existed:
        raise RuntimeError("PPTX expected output already existed before execution")
    token_keys = {
        "version",
        "path",
        "dev",
        "ino",
        "mode",
        "size",
        "mtime_ns",
        "sha256",
    }
    if (
        not isinstance(token, dict)
        or set(token) != token_keys
        or type(token.get("version")) is not int
        or token["version"] != 1
        or token.get("path") != str(expected_output)
        or any(
            type(token.get(name)) is not int
            for name in ("dev", "ino", "mode", "size", "mtime_ns")
        )
        or not _is_sha256(token.get("sha256"))
    ):
        raise RuntimeError("PPTX execution output identity token is invalid")
    expected_hash = token["sha256"]
    identity = (
        token["dev"],
        token["ino"],
        token["mode"],
        token["size"],
        token["mtime_ns"],
    )
    if _pptx_output_identity(expected_output) != identity:
        raise RuntimeError("PPTX execution output identity token does not match")
    if sha256_file(expected_output) != expected_hash:
        raise RuntimeError("PPTX execution output hash does not match identity token")
    if _pptx_output_identity(expected_output) != identity:
        raise RuntimeError("PPTX execution output changed during token verification")
    return expected_output, identity, expected_hash


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
    return _run_job(run_dir, lease_acquired=False)


def _run_job(
    run_dir: str | Path,
    *,
    lease_acquired: bool,
) -> dict[str, Any]:
    store = RunStore.open(run_dir)
    manifest, input_type = _manifest_input(store)
    resource_policy = _manifest_resource_policy(manifest)
    state = store.read_json("run_state.json")
    page_jobs = store.read_json("page_jobs.json")
    if state["status"] == RunStatus.COMPLETED.value:
        if input_type == "pptx":
            _pptx_page_ids(manifest, page_jobs, PageStatus.PRESERVED)
            (
                pptx_slide_count,
                pptx_preserved_objects,
                pptx_pending_candidates,
                pptx_input_sha256,
            ) = _pptx_manifest_expectations(manifest, page_jobs)
            validate_pptx_inventories(store, manifest)
            summary = store.read_json("run_summary.json")
            _validate_pptx_public_summary(
                summary,
                _pptx_output_path(store, manifest),
                pptx_slide_count,
                pptx_preserved_objects,
                pptx_pending_candidates,
                pptx_input_sha256,
                resource_policy,
            )
            _validate_completed_pptx_output(
                _pptx_output_path(store, manifest),
                pptx_input_sha256,
            )
            return summary
        summary = store.read_json("run_summary.json")
        validate_schema_version(summary)
        _validate_completed_resource_policy(summary, resource_policy)
        return summary
    if state["status"] != RunStatus.PREPARED.value:
        raise RuntimeError(
            f"Run must be prepared before execution; current status is {state['status']}"
        )
    if not lease_acquired:
        with ExecutionLease(
            store.root / "execution.lock",
            run_root=store.root,
        ):
            return _run_job(store.root, lease_acquired=True)

    store.write_json(
        "execution.json",
        {
            "schema_version": SCHEMA_VERSION,
            "token": secrets.token_hex(16),
            "pid": os.getpid(),
            "started_at": utc_now(),
            "input_type": input_type,
        },
    )
    apply_resource_policy(resource_policy)
    if input_type == "pptx":
        page_ids = _pptx_page_ids(
            manifest, page_jobs, PageStatus.ANALYZED
        )
        (
            pptx_slide_count,
            pptx_preserved_objects,
            pptx_pending_candidates,
            pptx_input_sha256,
        ) = _pptx_manifest_expectations(manifest, page_jobs)
        validate_pptx_inventories(store, manifest)
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
            manifest_token = _PPTX_EXECUTION_MANIFEST.set(manifest)
            try:
                summary = execute_pptx_preserve(store)
            finally:
                _PPTX_EXECUTION_MANIFEST.reset(manifest_token)
            pptx_output_published = True
            if isinstance(summary, dict):
                summary["resource_policy"] = resource_policy
            try:
                if pptx_output_existed:
                    raise RuntimeError(
                        "PPTX expected output already existed before execution"
                    )
                _validate_pptx_execution_summary(
                    summary,
                    pptx_expected_output,
                    pptx_slide_count,
                    pptx_preserved_objects,
                    pptx_pending_candidates,
                    pptx_input_sha256,
                    resource_policy,
                )
            except Exception:
                try:
                    pptx_output_record = _claim_pptx_output(
                        summary,
                        pptx_expected_output,
                        pptx_output_existed,
                    )
                except Exception:
                    pass
                raise
            else:
                pptx_output_record = _claim_pptx_output(
                    summary,
                    pptx_expected_output,
                    pptx_output_existed,
                )
            finally:
                if isinstance(summary, dict):
                    summary.pop("_output_identity", None)
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
                "resource_policy": resource_policy,
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
    if _reset_page_jobs(page_jobs, analyzed=analyzed):
        store.write_json("page_jobs.json", page_jobs)


def _reset_page_jobs(
    page_jobs: dict[str, Any],
    *,
    analyzed: bool,
) -> bool:
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
    return bool(updates)


def _manifest_output_path(manifest: dict[str, Any]) -> Path | None:
    options = manifest.get("options")
    if not isinstance(options, dict):
        raise RuntimeError("Run manifest options must be an object")
    value = options.get("output_path")
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError("Run manifest output_path must be a string or null")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError("Run manifest output_path must be absolute")
    return path


def _expected_legacy_output_entries(
    manifest: dict[str, Any],
    input_type: str,
) -> list[Path]:
    output = _manifest_output_path(manifest)
    if output is None:
        return []
    entries = [output]
    options = manifest["options"]
    slide_size = options.get("slide_size")
    if slide_size == "16:9":
        return entries
    pages = manifest.get("pages")
    if not isinstance(pages, list):
        raise RuntimeError("Run manifest pages must be an array")
    base = output.with_suffix("")
    if len(pages) == 1:
        if slide_size == "both":
            entries.extend(
                (
                    Path(f"{base}_16x9.pptx"),
                    Path(f"{base}_original.pptx"),
                )
            )
        return entries
    if slide_size == "both":
        entries.append(Path(f"{base}_16x9.pptx"))
    if slide_size not in {"both", "original"}:
        return entries
    combine_original = (
        input_type == "pdf"
        and manifest["input"].get("page_ratios_equal") is True
    )
    if combine_original:
        entries.append(Path(f"{base}_original.pptx"))
    else:
        entries.append(Path(f"{base}_original"))
    return entries


def _is_owned_final_output(store: RunStore, output: Path) -> bool:
    parent = Path(os.path.abspath(output.parent))
    final = Path(os.path.abspath(store.root / "final"))
    if not parent.is_relative_to(final):
        return False
    current = final
    for part in parent.relative_to(final).parts:
        try:
            status = current.lstat()
        except FileNotFoundError:
            return False
        if _is_link_or_reparse(status) or not stat.S_ISDIR(status.st_mode):
            return False
        current /= part
    try:
        status = current.lstat()
    except FileNotFoundError:
        return False
    return not _is_link_or_reparse(status) and stat.S_ISDIR(status.st_mode)


def recover_job(run_dir: str | Path) -> dict[str, Any]:
    store = RunStore.open(run_dir)
    with ExecutionLease(
        store.root / "execution.lock",
        run_root=store.root,
    ):
        store = RunStore.open(store.root)
        state = store.read_json("run_state.json")
        if state["status"] not in {
            RunStatus.RUNNING.value,
            RunStatus.FINALIZING.value,
        }:
            raise RuntimeError(
                "Run must be running or finalizing before recovery; "
                f"current status is {state['status']}"
            )

        manifest, input_type = _manifest_input(store)
        if input_type == "pptx":
            expected_output = _pptx_output_path(store, manifest)
            if _path_entry_exists(expected_output):
                raise RuntimeError(
                    f"PPTX recovery is blocked by an existing output: "
                    f"{expected_output}"
                )
        else:
            for output in _expected_legacy_output_entries(
                manifest, input_type
            ):
                if _path_entry_exists(
                    output
                ) and not _is_owned_final_output(store, output):
                    raise RuntimeError(
                        "Run recovery is blocked by an existing external "
                        f"output: {output}"
                    )

        page_jobs = store.read_json("page_jobs.json")
        pages_changed = _reset_page_jobs(
            page_jobs,
            analyzed=input_type == "pptx",
        )
        cleanup = [
            directory
            for directory in (
                _run_owned_directory(store, "final"),
                _run_owned_directory(store, "work"),
            )
            if directory is not None
        ]

        for directory in cleanup:
            _safe_rmtree(*directory)
        if pages_changed:
            store.write_json("page_jobs.json", page_jobs)
        store.transition_run(RunStatus.FAILED)
        store.transition_run(RunStatus.PREPARED)
        return get_status(store.root)


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
    work = _run_work_directory(store)
    if work is not None:
        _safe_rmtree(*work)
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
