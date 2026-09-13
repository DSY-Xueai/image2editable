from __future__ import annotations

from contextvars import ContextVar
import importlib
import json
import logging
import os
from pathlib import Path
import secrets
import stat
import tempfile
import time
from typing import Any, Iterable, Sequence

from image2editable.component_contracts import (
    MAX_REPAIR_ROUNDS,
    validate_agent_provider,
    validate_component_repair_state,
)
from image2editable.component_repair import (
    _read_bound_file,
    resume_round_limited_component_repair,
)
from image2editable.contracts import (
    PageStatus,
    RunStatus,
    SCHEMA_VERSION,
    transition_page_document,
    utc_now,
    validate_schema_version,
)
from image2editable.inputs import (
    classify_inputs,
    prepare_image_job,
    sha256_file,
    validate_pipeline_mode,
)
from image2editable.execution import ExecutionLease, _lock, _unlock
from image2editable.legacy import (
    _safe_rmtree,
    _clear_legacy_output_records,
    _clear_legacy_staging_records,
    _cleanup_native_pdf_sources,
    _legacy_output_targets,
    _legacy_psd_output_targets,
    _legacy_output_identity,
    _load_legacy_output_records,
    _load_legacy_staging_records,
    _write_legacy_output_records,
    _load_legacy_ref,
    _native_pdf_analysis,
    _page_ocr_rotation,
    _remove_legacy_outputs,
    _verify_legacy_output_record,
    _source_path,
    advance_legacy_page,
    assemble_route_candidate,
    assemble_legacy_results,
    execute_legacy,  # noqa: F401 - retained as the legacy-boundary test seam
    initialize_legacy_page,
)
from image2editable.resources import (
    apply_resource_policy,
    validate_resource_policy,
)
from image2editable.store import RunStore
from scripts.performance_trace import PerformanceTrace, _validate_event, _validate_field


_PPTX_EXECUTION_MANIFEST: ContextVar[dict[str, Any] | None] = ContextVar(
    "_PPTX_EXECUTION_MANIFEST", default=None
)
COMPONENT_QUALITY_GATE_VERSION = "component-quality-v2"
_PERFORMANCE_TRACE_PREFIX = "performance-"
_PERFORMANCE_SUMMARY_NAME = "performance-summary.json"
_PERFORMANCE_TRACE_LIMIT = 16 * 1024 * 1024
_PERFORMANCE_LINE_LIMIT = 16 * 1024
_PERFORMANCE_MAX_LINES = 100_000
_PERFORMANCE_MAX_FIELDS = 16
_PERFORMANCE_MAX_INTEGER = 1_000_000_000_000
_PERFORMANCE_MAPS = {
    "model_loads": "model",
    "stage_runs": "stage",
    "stage_duration_ms": "stage",
    "worker_runs": "model",
    "worker_duration_ms": "model",
    "inference_runs": "model",
    "inference_operations": "model",
    "inference_duration_ms": "model",
}
_LEGACY_HOST_PERFORMANCE_FIELDS = {
    "agent_runs",
    "agent_image_count",
    "agent_total_bytes",
    "agent_duration_ms",
}
_LOGGER = logging.getLogger(__name__)


class _BoundPerformanceTrace(PerformanceTrace):
    def __init__(
        self,
        path: Path,
        identity: tuple[int, int],
        root: Path,
        root_identity: tuple[int, int],
    ) -> None:
        super().__init__(path)
        self._identity = identity
        self._root = root
        self._root_identity = root_identity

    def _require_root(self) -> None:
        status = self._root.lstat()
        if (
            _is_link_or_reparse(status)
            or not stat.S_ISDIR(status.st_mode)
            or (status.st_dev, status.st_ino) != self._root_identity
        ):
            raise RuntimeError("Performance trace root identity changed")

    def event(self, event: str, **fields) -> None:
        _validate_event(event, fields)
        if any(
            type(value) is int and value > _PERFORMANCE_MAX_INTEGER
            for value in fields.values()
        ):
            raise RuntimeError("Performance trace integer limit exceeded")
        encoded = json.dumps(
            {"schema_version": 1, "event": event, **fields},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        flags = os.O_RDWR
        for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= getattr(os, name, 0)
        self._require_root()
        try:
            descriptor = os.open(self.path, flags)
            with os.fdopen(descriptor, "r+b") as stream:
                stream.seek(0)
                _lock(stream)
                try:
                    opened = os.fstat(stream.fileno())
                    current = self.path.lstat()
                    self._require_root()
                    if (
                        len(encoded) > _PERFORMANCE_LINE_LIMIT
                        or opened.st_size + len(encoded) > _PERFORMANCE_TRACE_LIMIT
                        or _is_link_or_reparse(current)
                        or not stat.S_ISREG(opened.st_mode)
                        or not stat.S_ISREG(current.st_mode)
                        or opened.st_nlink != 1
                        or current.st_nlink != 1
                        or (opened.st_dev, opened.st_ino) != self._identity
                        or (current.st_dev, current.st_ino) != self._identity
                    ):
                        raise RuntimeError(
                            "Performance trace identity or size limit changed"
                        )
                    stream.seek(0, os.SEEK_END)
                    stream.write(encoded)
                    stream.flush()
                finally:
                    stream.seek(0)
                    _unlock(stream)
        except OSError:
            raise RuntimeError("Performance trace append failed") from None


def _page_performance_trace(
    store: RunStore, page_id: str
) -> PerformanceTrace | None:
    try:
        if page_id not in store.read_json("page_jobs.json")["pages"]:
            raise RuntimeError("Performance trace page is not part of the Run")
        _validate_field("page_id", page_id)
        root_status = store.root.lstat()
        if _is_link_or_reparse(root_status) or not stat.S_ISDIR(root_status.st_mode):
            raise RuntimeError("Performance trace root is unsafe")
        path = store.root / f"{_PERFORMANCE_TRACE_PREFIX}{page_id}.jsonl"
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOINHERIT", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        status = path.lstat()
        if (
            _is_link_or_reparse(status)
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or status.st_size > _PERFORMANCE_TRACE_LIMIT
        ):
            raise RuntimeError("Performance trace is not a safe regular file")
        return _BoundPerformanceTrace(
            path,
            (status.st_dev, status.st_ino),
            store.root,
            (root_status.st_dev, root_status.st_ino),
        )
    except Exception:
        _LOGGER.warning("Performance trace could not be bound for page %s", page_id)
        return None


def _empty_page_performance() -> dict[str, Any]:
    return {
        "model_loads": {},
        "stage_runs": {},
        "stage_duration_ms": {},
        "worker_runs": {},
        "worker_duration_ms": {},
        "inference_runs": {},
        "inference_operations": {},
        "inference_duration_ms": {},
    }


def _metric_update(
    summary: dict[str, Any], field: str, key: str | None, value: int
) -> tuple[str, str | None, int]:
    current = summary[field] if key is None else summary[field].get(key, 0)
    updated = current + value
    if updated > _PERFORMANCE_MAX_INTEGER:
        raise ValueError("performance aggregate integer limit exceeded")
    return field, key, updated


def _aggregate_performance_event(
    summary: dict[str, Any],
    event: dict[str, Any],
    *,
    pending_model_loads: set[tuple[str, str]] | None = None,
    page_id: str | None = None,
) -> None:
    kind = event["event"]
    model = event.get("model")
    stage = event.get("stage")
    updates = []
    if kind in {"model_load_start", "model_load_finish"}:
        if pending_model_loads is None or event.get("page_id") != page_id:
            raise ValueError("performance model load page identity is invalid")
        key = (event["page_id"], model)
        if kind == "model_load_start":
            if key in pending_model_loads:
                raise ValueError("performance model load start is duplicated")
            pending_model_loads.add(key)
        else:
            if key not in pending_model_loads:
                raise ValueError("performance model load finish is unpaired")
            pending_model_loads.remove(key)
            if event["status"] == "success":
                updates.append(_metric_update(summary, "model_loads", model, 1))
    elif kind == "span":
        updates.extend((
            _metric_update(summary, "stage_runs", stage, 1),
            _metric_update(
                summary, "stage_duration_ms", stage, event["duration_ms"]
            ),
        ))
    elif kind == "worker" and model is not None:
        updates.extend((
            _metric_update(summary, "worker_runs", model, 1),
            _metric_update(
                summary, "worker_duration_ms", model, event["duration_ms"]
            ),
        ))
    elif kind == "inference_finish" and model is not None:
        updates.extend((
            _metric_update(summary, "inference_runs", model, 1),
            _metric_update(
                summary, "inference_operations", model,
                event.get("operation_count", 0),
            ),
            _metric_update(
                summary, "inference_duration_ms", model, event["duration_ms"]
            ),
        ))
    for field, key, value in updates:
        if key is None:
            summary[field] = value
        else:
            summary[field][key] = value


def _read_page_performance(
    store: RunStore, page_id: str
) -> dict[str, Any]:
    path = store.root / f"{_PERFORMANCE_TRACE_PREFIX}{page_id}.jsonl"
    try:
        payload = _read_bound_file(
            path,
            store.root,
            max_bytes=_PERFORMANCE_TRACE_LIMIT,
            label="performance trace",
        )
    except Exception:
        _LOGGER.warning("Performance trace could not be read for page %s", page_id)
        return _empty_page_performance()
    summary = _empty_page_performance()
    lines = payload.splitlines()
    pending_model_loads: set[tuple[str, str]] = set()
    if len(lines) > _PERFORMANCE_MAX_LINES:
        _LOGGER.warning("Performance trace line limit exceeded for page %s", page_id)
        return summary
    for line_number, line in enumerate(lines, start=1):
        try:
            if not line or len(line) > _PERFORMANCE_LINE_LIMIT:
                raise ValueError("performance trace line size is invalid")
            event = json.loads(line)
            if not isinstance(event, dict) or len(event) > _PERFORMANCE_MAX_FIELDS:
                raise ValueError("performance trace event is invalid")
            if event.get("schema_version") != 1 or type(
                event.get("schema_version")
            ) is not int:
                raise ValueError("performance trace schema version is invalid")
            kind = event.get("event")
            fields = {
                name: value for name, value in event.items()
                if name not in {"schema_version", "event"}
            }
            _validate_event(kind, fields)
            if "page_id" in fields and fields["page_id"] != page_id:
                raise ValueError("performance trace page identity is invalid")
            if any(
                type(value) is int and value > _PERFORMANCE_MAX_INTEGER
                for value in fields.values()
            ):
                raise ValueError("performance trace integer limit exceeded")
            _aggregate_performance_event(
                summary,
                event,
                pending_model_loads=pending_model_loads,
                page_id=page_id,
            )
        except Exception:
            _LOGGER.warning(
                "Skipping invalid performance trace event for page %s at line %d",
                page_id, line_number,
            )
    return summary


def _page_route(store: RunStore, page_id: str) -> tuple[str, bool]:
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    try:
        state = store.read_json(
            Path("pages") / page_id / "reconstruction" / "component_state.json"
        )
    except (FileNotFoundError, ValueError):
        state = {}
    if state.get("route") == "pdf_native":
        return "pdf_native", False
    result_ref = state.get("result_ref")
    if isinstance(result_ref, dict):
        try:
            result = json.loads(_load_legacy_ref(store, result_ref)[1])
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            result = {}
        if result.get("route") == "local_fidelity":
            return "local_fidelity", True
    try:
        prepared = json.loads(
            (reconstruction / "initial" / "prepared_page.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return "unknown", False
    policy = prepared.get("page_policy")
    route = policy.get("route") if isinstance(policy, dict) else None
    return (route, False) if route in {"direct", "local_refine", "strict"} else (
        "unknown", False
    )


def _performance_summary(
    store: RunStore,
    page_ids: Sequence[str],
    *,
    total_duration_ms: int = 0,
) -> dict[str, Any]:
    if type(total_duration_ms) is not int or total_duration_ms < 0:
        raise ValueError("performance total duration is invalid")
    pages = {}
    model_loads: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    local_fidelity_pages = []
    for page_id in page_ids:
        _page_performance_trace(store, page_id)
        page = _read_page_performance(store, page_id)
        pages[page_id] = page
        for model, count in page["model_loads"].items():
            model_loads[model] = model_loads.get(model, 0) + count
        route, local_fidelity = _page_route(store, page_id)
        route_counts[route] = route_counts.get(route, 0) + 1
        if local_fidelity:
            local_fidelity_pages.append(page_id)
    return {
        "pages": pages,
        "agent_runs": 0,
        "agent_image_count": 0,
        "agent_total_bytes": 0,
        "model_loads": model_loads,
        "route_counts": route_counts,
        "local_fidelity_pages": local_fidelity_pages,
        "total_duration_ms": total_duration_ms,
    }


def _write_performance_summaries(
    store: RunStore, performance: dict[str, Any]
) -> None:
    for page_id, summary in performance["pages"].items():
        try:
            page = _run_owned_directory(store, Path("pages") / page_id)
            if page is None:
                raise RuntimeError("Performance summary page directory is missing")
            reconstruction_path = page[0] / "reconstruction"
            try:
                reconstruction_path.mkdir(mode=0o700)
            except FileExistsError:
                pass
            current_page = _run_owned_directory(store, Path("pages") / page_id)
            if current_page is None or current_page[1] != page[1]:
                raise RuntimeError("Performance summary page identity changed")
            reconstruction = _run_owned_directory(
                store, Path("pages") / page_id / "reconstruction"
            )
            if reconstruction is not None:
                store.write_json(
                    Path("pages") / page_id / "reconstruction"
                    / _PERFORMANCE_SUMMARY_NAME,
                    summary,
                )
        except Exception:
            _LOGGER.warning(
                "Performance summary could not be written for page %s", page_id
            )


def _validate_performance_summary(
    performance: object, expected_page_ids: Sequence[str]
) -> None:
    fields = set(_PERFORMANCE_MAPS)
    aggregate_fields = {
        "agent_runs", "agent_image_count", "agent_total_bytes",
        "model_loads", "route_counts", "local_fidelity_pages",
        "total_duration_ms",
    }
    if not isinstance(performance, dict) or frozenset(performance) not in {
        frozenset({"pages"}), frozenset({"pages", *aggregate_fields})
    }:
        raise RuntimeError("Run performance summary fields are invalid")
    pages = performance["pages"]
    if not isinstance(pages, dict) or set(pages) != set(expected_page_ids):
        raise RuntimeError("Run performance summary pages are invalid")
    for page_id, page in pages.items():
        try:
            _validate_field("page_id", page_id)
        except ValueError as error:
            raise RuntimeError("Run performance summary page id is invalid") from error
        if not isinstance(page, dict):
            raise RuntimeError("Run performance summary page fields are invalid")
        page_fields = set(page)
        if page_fields == fields | _LEGACY_HOST_PERFORMANCE_FIELDS:
            if any(
                type(page[field]) is not int or page[field] != 0
                for field in _LEGACY_HOST_PERFORMANCE_FIELDS
            ):
                raise RuntimeError("Run performance summary page fields are invalid")
            for field in _LEGACY_HOST_PERFORMANCE_FIELDS:
                page.pop(field)
        elif page_fields != fields:
            raise RuntimeError("Run performance summary page fields are invalid")
        for field, identifier_kind in _PERFORMANCE_MAPS.items():
            metrics = page[field]
            if not isinstance(metrics, dict):
                raise RuntimeError("Run performance summary metric is invalid")
            for name, value in metrics.items():
                try:
                    _validate_field(identifier_kind, name)
                except ValueError as error:
                    raise RuntimeError(
                        "Run performance summary metric name is invalid"
                    ) from error
                if (
                    type(value) is not int or value < 0
                    or value > _PERFORMANCE_MAX_INTEGER
                ):
                    raise RuntimeError("Run performance summary metric is invalid")
    if set(performance) == {"pages"}:
        return
    if any(
        type(performance[field]) is not int or performance[field] != 0
        for field in ("agent_runs", "agent_image_count", "agent_total_bytes")
    ):
        raise RuntimeError("Run performance Agent metrics are invalid")
    if (
        type(performance["total_duration_ms"]) is not int
        or not 0 <= performance["total_duration_ms"] <= _PERFORMANCE_MAX_INTEGER
    ):
        raise RuntimeError("Run performance duration is invalid")
    for field, identifier_kind in (
        ("model_loads", "model"), ("route_counts", "route")
    ):
        metrics = performance[field]
        if not isinstance(metrics, dict):
            raise RuntimeError("Run performance aggregate is invalid")
        for name, value in metrics.items():
            try:
                _validate_field(identifier_kind, name)
            except ValueError as error:
                raise RuntimeError("Run performance aggregate is invalid") from error
            if (
                type(value) is not int or value < 0
                or value > _PERFORMANCE_MAX_INTEGER
            ):
                raise RuntimeError("Run performance aggregate is invalid")
    local_pages = performance["local_fidelity_pages"]
    if (
        not isinstance(local_pages, list)
        or local_pages != list(dict.fromkeys(local_pages))
        or any(page_id not in expected_page_ids for page_id in local_pages)
    ):
        raise RuntimeError("Run performance local fidelity pages are invalid")


def _discover_powerpoint_renderer():
    from image2editable.powerpoint_renderer import PowerPointRenderer

    return PowerPointRenderer.discover()


def _finalize_reconstruction_route(context, *, renderer, policy):
    from image2editable.route_execution import finalize_page_route

    return finalize_page_route(context, renderer=renderer, policy=policy)


def _finalize_reconstruction_routes(
    store: RunStore,
    manifest: dict[str, Any],
    page_ids: list[str],
) -> None:
    from image2editable.powerpoint_renderer import PowerPointRenderer
    from image2editable.route_execution import RouteContext

    output_format = manifest.get("output_format", "pptx")
    is_psd = output_format == "psd"
    capabilities = frozenset(
        {"editable_text", "raster_component"}
        if is_psd
        else {"editable_text", "native_shape", "raster_component"}
    )
    renderer = PowerPointRenderer(None) if is_psd else _discover_powerpoint_renderer()
    policy = {
        "schema_version": 1,
        "native_shape_enabled": False,
        "allowed_shapes": ["rectangle", "rounded_rectangle", "ellipse", "line"],
        "min_geometry_score": 0.99,
        "max_color_mad": 3.0,
    }
    for page_id in page_ids:
        state_path = (
            store.root / "pages" / page_id / "reconstruction"
            / "component_state.json"
        )
        if not state_path.is_file():
            continue
        state = store.read_json(
            f"pages/{page_id}/reconstruction/component_state.json"
        )
        if state.get("route") == "pdf_native":
            continue
        if state.get("status") != "ready_for_assembly":
            continue
        component_result_path, _ = _load_legacy_ref(store, state["result_ref"])
        context = RouteContext(
            store=store,
            page_id=page_id,
            component_result_path=component_result_path,
            adapter="psd" if is_psd else "pptx",
            capabilities=capabilities,
            source_image_path=_source_path(store, page_id),
            assemble_page=lambda plan, output, result_path=component_result_path: (
                assemble_route_candidate(store, result_path, plan, output)
            ),
        )
        _finalize_reconstruction_route(context, renderer=renderer, policy=policy)


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


def rerender_pdf_page(
    run_dir: str | Path, page_id: str
) -> dict[str, bool]:
    _manifest_input(RunStore.open(run_dir))
    return _pdf_function("rerender_pdf_page")(run_dir, page_id)


def prepare_pptx_job(*args: Any, **kwargs: Any) -> Path:
    from image2editable.pptx_input import prepare_pptx_job as prepare

    return prepare(*args, **kwargs)


def execute_pptx_preserve(store: RunStore) -> dict[str, object]:
    from image2editable.pptx_input import execute_pptx_preserve as execute

    return execute(store, _PPTX_EXECUTION_MANIFEST.get())


def execute_pptx_shadow(
    store: RunStore,
    plans: list[dict[str, object]],
) -> dict[str, object]:
    from image2editable.pptx_input import execute_pptx_shadow as execute

    return execute(store, plans, _PPTX_EXECUTION_MANIFEST.get())


def shadow_replacement_plans(
    store: RunStore,
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    from image2editable.agent import shadow_replacement_plans as build

    return build(store, manifest)


def next_candidate(run_dir: str | Path) -> dict[str, object]:
    _manifest_input(RunStore.open(run_dir))
    from image2editable.agent import next_candidate as find_next

    return find_next(run_dir)


def record_decision(
    run_dir: str | Path,
    **kwargs: object,
) -> dict[str, object]:
    _manifest_input(RunStore.open(run_dir))
    from image2editable.agent import record_decision as record

    return record(run_dir, **kwargs)


def next_host_agent_item(run_dir: str | Path) -> dict[str, object]:
    from image2editable.host_agent import next_host_agent_item as find_next

    return find_next(run_dir)


def record_host_agent_plan(
    run_dir: str | Path, plan_path: str | Path
) -> dict[str, object]:
    from image2editable.host_agent import record_host_plan

    return record_host_plan(run_dir, plan_path)


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
    agent_provider: str = "host",
    output_format: str = "pptx",
    pipeline_mode: str = "strict",
) -> Path:
    input_type, paths = classify_inputs(inputs)
    if output_format not in {"pptx", "psd"}:
        raise ValueError(f"Unsupported output_format: {output_format}")
    if output_format == "psd" and input_type != "images":
        raise ValueError("PSD output only supports image input")
    prepare = {
        "images": prepare_image_job,
        "pdf": prepare_pdf_job,
        "pptx": prepare_pptx_job,
    }[input_type]
    source: Path | list[Path] = paths if input_type == "images" else paths[0]
    prepare_kwargs = {
        "run_dir": run_dir,
        "output_path": output_path,
        "slide_size": slide_size,
        "lang": lang,
        "agent_provider": agent_provider,
        "pipeline_mode": pipeline_mode,
    }
    if input_type == "images" and output_format != "pptx":
        prepare_kwargs["output_format"] = output_format
    return prepare(
        source,
        **prepare_kwargs,
    )


def get_status(run_dir: str | Path) -> dict[str, Any]:
    store = RunStore.open(run_dir)
    manifest, _ = _manifest_input(store)
    run = store.read_json("run_state.json")
    pages = store.read_json("page_jobs.json")
    status = {"run": run, "pages": pages}
    if run["status"] == RunStatus.AWAITING_AGENT.value:
        awaiting = [
            page_id for page_id in manifest["pages"]
            if pages["pages"][page_id]["status"] == PageStatus.AWAITING_AGENT.value
        ]
        if len(awaiting) != 1:
            raise RuntimeError("Awaiting Agent Run must have one current page")
        status.update(_legacy_component_status(store, awaiting[0]))
    return status


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


def _legacy_waiting_summary(
    store: RunStore, manifest: dict[str, Any], page_id: str, outcome: dict
) -> dict[str, Any]:
    component = _legacy_component_status(store, page_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": RunStatus.AWAITING_AGENT.value,
        "provider": manifest["options"]["agent_provider"],
        "quality_gate_version": COMPONENT_QUALITY_GATE_VERSION,
        **component,
        "updated_at": utc_now(),
    }


def _legacy_component_status(store: RunStore, page_id: str) -> dict[str, Any]:
    state = validate_component_repair_state(store.read_json(
        f"pages/{page_id}/reconstruction/component_state.json"
    ))
    if state["page_id"] != page_id:
        raise RuntimeError("Component repair status page identity mismatch")
    reconstruction = store.root / "pages" / page_id / "reconstruction"
    return {
        "current_page": page_id,
        "repair_round": state["repair_round"],
        "frozen_components": len(state["frozen"]),
        "pending_components": len(state["candidate_ids"]),
        "provider": state["provider"],
        "diagnostics": str((reconstruction / "agent").resolve()),
    }


def _batch_legacy_ocr(
    store: RunStore,
    manifest: dict[str, Any],
    page_ids: list[str],
    *,
    ocr_worker_pool=None,
) -> dict[str, Any]:
    """Run one isolated OCR batch for eligible, not-yet-initialized pages."""
    if manifest.get("options", {}).get("pipeline_mode", "strict") != "fast":
        return {}
    page_jobs = store.read_json("page_jobs.json")["pages"]
    completed = {
        PageStatus.VALIDATED.value,
    }
    eligible = []
    for page_id in page_ids:
        if page_jobs[page_id]["status"] in completed:
            continue
        if _native_pdf_analysis(store, page_id) is not None:
            continue
        if (
            store.root / "pages" / page_id / "reconstruction"
            / "component_state.json"
        ).is_file():
            continue
        if _page_ocr_rotation(store, page_id) != 0:
            continue
        eligible.append((page_id, _source_path(store, page_id)))
    if len(eligible) < 2:
        return {}

    try:
        module = importlib.import_module("image_to_ppt")
        batch_kwargs = {
            "lang": manifest["options"]["lang"],
            "isolated": True,
            "worker_root": store.root,
        }
        if ocr_worker_pool is not None:
            batch_kwargs["worker_pool"] = ocr_worker_pool
        results = module.detect_text_batch(
            [source for _, source in eligible],
            **batch_kwargs,
        )
        if not isinstance(results, list) or len(results) != len(eligible):
            _LOGGER.warning("Fast OCR batch returned the wrong page count")
            return {}
    except Exception as error:
        _LOGGER.warning("Fast OCR batch failed; falling back to page OCR: %s", error)
        return {}
    return {
        page_id: result
        for (page_id, _), result in zip(eligible, results, strict=True)
    }


def _advance_legacy_pages(
    store: RunStore, manifest: dict[str, Any], page_ids: list[str],
    lease: ExecutionLease, *, ocr_worker_pool=None, visual_worker_pool=None,
    visual_worker_pool_factory=None,
) -> dict[str, Any] | None:
    completed = {
        PageStatus.VALIDATED.value,
    }
    visual_page_ids = [
        page_id for page_id in page_ids
        if _native_pdf_analysis(store, page_id) is None
    ]
    module = (
        importlib.import_module("image_to_ppt") if visual_page_ids else None
    )
    owns_ocr_worker_pool = ocr_worker_pool is None
    owns_visual_worker_pool = (
        visual_worker_pool is None and visual_worker_pool_factory is None
    )
    if ocr_worker_pool is None and module is not None:
        ocr_worker_pool = module.create_ocr_worker_pool()
    pipeline_mode = manifest.get("options", {}).get("pipeline_mode", "strict")
    if (
        module is not None
        and pipeline_mode != "fast"
        and visual_worker_pool is None
    ):
        visual_worker_pool = module.create_visual_worker_pool()

    def ensure_visual_worker_pool():
        nonlocal visual_worker_pool
        if visual_worker_pool is None:
            if module is None:
                raise RuntimeError("visual worker requested for native PDF page")
            visual_worker_pool = (
                visual_worker_pool_factory()
                if visual_worker_pool_factory is not None
                else module.create_visual_worker_pool()
            )
        return visual_worker_pool

    try:
        batch_ocr = _batch_legacy_ocr(
            store, manifest, page_ids, ocr_worker_pool=ocr_worker_pool,
        )
        for page_id in page_ids:
            performance_trace = _page_performance_trace(store, page_id)
            if store.read_json("page_jobs.json")["pages"][page_id]["status"] in completed:
                continue
            reconstruction = store.root / "pages" / page_id / "reconstruction"
            page_status = store.read_json("page_jobs.json")["pages"][page_id]["status"]
            if page_status == PageStatus.PRESERVED_WITH_WARNING.value:
                # Warning is an internal repair boundary, not a user-visible
                # terminal result. Reopen the next bounded repair round before
                # attempting assembly.
                if not resume_round_limited_component_repair(store, page_id):
                    raise RuntimeError(
                        f"Automatic repair could not resume page {page_id}"
                    )
                page_jobs = store.read_json("page_jobs.json")
                page_jobs["pages"][page_id].update({
                    "status": PageStatus.PROCESSING.value,
                    "updated_at": utc_now(),
                })
                store.write_json("page_jobs.json", page_jobs)
            if not (reconstruction / "component_state.json").is_file():
                initialize_kwargs = {
                    "_lease": lease,
                    "performance_trace": performance_trace,
                    "ocr_worker_pool": ocr_worker_pool,
                    "visual_worker_pool": visual_worker_pool,
                }
                if pipeline_mode == "fast":
                    initialize_kwargs["visual_worker_pool_factory"] = (
                        ensure_visual_worker_pool
                    )
                if page_id in batch_ocr:
                    initialize_kwargs["ocr_result"] = batch_ocr[page_id]
                initialize_legacy_page(store, page_id, **initialize_kwargs)
            for _ in range(MAX_REPAIR_ROUNDS * 6 + 4):
                advance_kwargs = {
                    "_lease": lease,
                    "performance_trace": performance_trace,
                    "visual_worker_pool": visual_worker_pool,
                }
                if pipeline_mode == "fast":
                    advance_kwargs["visual_worker_pool_factory"] = (
                        ensure_visual_worker_pool
                    )
                outcome = advance_legacy_page(
                    store,
                    page_id,
                    **advance_kwargs,
                )
                if outcome["status"] != "processing":
                    break
            else:
                raise RuntimeError(
                    "Legacy component page exceeded durable boundary limit"
                )
            if outcome["status"] == "awaiting_agent":
                _transition_pages(store, [page_id], PageStatus.AWAITING_AGENT)
                store.transition_run(RunStatus.AWAITING_AGENT)
                summary = _legacy_waiting_summary(store, manifest, page_id, outcome)
                store.write_json("run_summary.json", summary)
                return summary
            if outcome["status"] == "preserved_with_warning":
                _transition_pages(store, [page_id], PageStatus.PRESERVED_WITH_WARNING)
            elif outcome["status"] == "ready_for_assembly":
                _transition_pages(store, [page_id], PageStatus.VALIDATED)
            else:
                raise RuntimeError(
                    "Legacy component page did not reach a terminal boundary: "
                    f"{outcome['status']}"
                )
        return None
    finally:
        if owns_visual_worker_pool and visual_worker_pool is not None:
            visual_worker_pool.close()
        if owns_ocr_worker_pool and ocr_worker_pool is not None:
            ocr_worker_pool.close()


def _ensure_legacy_pages_processing(
    store: RunStore, page_ids: list[str]
) -> None:
    pages = store.read_json("page_jobs.json")["pages"]
    completed = {
        PageStatus.VALIDATED.value,
        PageStatus.PRESERVED_WITH_WARNING.value,
    }
    for page_id in page_ids:
        if pages[page_id]["status"] not in {
            PageStatus.PROCESSING.value,
            *completed,
        }:
            _transition_pages(store, [page_id], PageStatus.PROCESSING)


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
        elif page_ids:
            reconstruction = (
                store.root / "pages" / page_ids[0] / "reconstruction"
            ).resolve()
            if reconstruction.is_dir() and any(
                child.name != "component_state.json"
                for child in reconstruction.iterdir()
            ):
                summary["diagnostics"] = str(reconstruction)
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
    _manifest_agent_provider(manifest)
    return manifest, input_type


def _manifest_agent_provider(manifest: dict[str, Any]) -> str:
    options = manifest.get("options")
    if type(options) is not dict:
        raise RuntimeError("Run manifest agent_provider requires options")
    try:
        return validate_agent_provider(options.get("agent_provider"))
    except ValueError as error:
        raise RuntimeError("Run manifest agent_provider is invalid") from error


def _manifest_pipeline_mode(manifest: dict[str, Any]) -> str:
    options = manifest.get("options")
    if type(options) is not dict:
        raise RuntimeError("Run manifest pipeline_mode requires options")
    try:
        return validate_pipeline_mode(options.get("pipeline_mode", "strict"))
    except ValueError as error:
        raise RuntimeError("Run manifest pipeline_mode is invalid") from error


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


def _clear_host_plan_records(store: RunStore, page_id: str) -> None:
    prefix = f"host-component-plan-{page_id}-"
    records = []
    for entry in store.root.iterdir():
        if not (entry.name.startswith(prefix) and entry.name.endswith(".json")):
            continue
        status = entry.lstat()
        if (
            _is_link_or_reparse(status)
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
        ):
            raise RuntimeError(f"Run host plan is not a regular file: {entry}")
        records.append(entry)
    for record in records:
        record.unlink()


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
    name: str | Path,
) -> tuple[Path, tuple[int, int]] | None:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Run directory path is invalid: {name}")
    current = store.root
    status = None
    for part in relative.parts:
        current /= part
        try:
            status = current.lstat()
        except FileNotFoundError:
            return None
        if _is_link_or_reparse(status):
            raise RuntimeError(
                f"Run {name} directory is a link or reparse point: {current}"
            )
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(f"Run {name} path is not a directory: {current}")
    if status is None:
        raise RuntimeError(f"Run directory path is invalid: {name}")
    resolved = current.resolve()
    if not resolved.is_relative_to(store.root):
        raise RuntimeError(
            f"Run {name} directory is outside run directory: {current}"
        )
    return resolved, (status.st_dev, status.st_ino)


def _prune_completed_legacy_artifacts(
    store: RunStore, page_ids: Sequence[str]
) -> None:
    retained = {
        "component_state.json",
        "component_result.json",
        "component_delivery.json",
        "native_page.json",
        "native-quality.json",
        "native-source.png",
        "native-render-original.png",
        "native-render-16x9.png",
        _PERFORMANCE_SUMMARY_NAME,
    }
    for page_id in page_ids:
        owned = _run_owned_directory(
            store, Path("pages") / page_id / "reconstruction"
        )
        if owned is None:
            continue
        reconstruction = owned[0]
        native_path = reconstruction / "native_page.json"
        if native_path.is_file():
            state = store.read_json(
                f"pages/{page_id}/reconstruction/component_state.json"
            )
            bound_path, payload = _load_legacy_ref(store, state.get("native_page_ref"))
            native = json.loads(payload.decode("utf-8"))
            if (
                state.get("route") != "pdf_native" or bound_path != native_path
                or native.get("page_id") != page_id
            ):
                raise RuntimeError("Completed native PDF analysis binding is invalid")
            delivery = store.read_json(f"pages/{page_id}/reconstruction/component_delivery.json")
            report_path, report_payload = _load_legacy_ref(store, delivery.get("native_quality_ref"))
            reports = json.loads(report_payload.decode("utf-8"))
            if (
                report_path != reconstruction / "native-quality.json"
                or reports.get("page_id") != page_id
                or set(reports.get("variants", {})) != set(delivery["outputs"])
            ):
                raise RuntimeError("Completed native PDF quality binding is invalid")
            for variant, report in reports["variants"].items():
                if (
                    report["analysis_sha256"] != state["native_page_ref"]["sha256"]
                    or report["pptx_sha256"] != delivery["outputs"][variant]["sha256"]
                    or report["report"].get("accepted") is not True
                    or report["report"].get("violations") != []
                    or sha256_file(reconstruction / "native-source.png") != report["source_sha256"]
                    or sha256_file(reconstruction / ("native-render-" + variant.replace(":", "x") + ".png")) != report["rendered_sha256"]
                ):
                    raise RuntimeError("Completed native PDF quality binding is invalid")
        quality_path = None
        result_path = reconstruction / "component_result.json"
        if result_path.is_file():
            result = json.loads(_read_bound_file(
                result_path, reconstruction, max_bytes=16 * 1024 * 1024,
                label="completed component result",
            ))
            repair_round = result.get("repair_rounds")
            if repair_round is not None:
                if type(repair_round) is not int or repair_round < 1:
                    raise RuntimeError("Completed quality report binding is invalid")
                quality_path = (
                    reconstruction / f"execution-{repair_round:02d}"
                    / "component-quality.json"
                )
                quality = json.loads(_read_bound_file(
                    quality_path, reconstruction, max_bytes=16 * 1024 * 1024,
                    label="completed quality report",
                ))
                # A parent fallback accepts a new graph; retain the original
                # round's report as failure evidence, not proof of acceptance.
                parent_fallback = (
                    result.get("fallback", {}).get("status") == "parent_preserved"
                )
                if (
                    quality.get("page_id") != page_id
                    or quality.get("repair_round") != repair_round
                    or not result.get("accepted_graph_sha256")
                    or (
                        not parent_fallback
                        and quality.get("input_graph_sha256")
                        != result["accepted_graph_sha256"]
                    )
                ):
                    raise RuntimeError("Completed quality report binding is invalid")
        preserved_directories = [owned]
        directories = []
        files = []
        entries = list(reconstruction.iterdir())
        for child in entries:
            status = child.lstat()
            if _is_link_or_reparse(status):
                raise RuntimeError(
                    f"Completed reconstruction contains an unsafe link: {child}"
                )
            if stat.S_ISDIR(status.st_mode):
                directory = (child, (status.st_dev, status.st_ino))
                if quality_path is not None and child == quality_path.parent:
                    preserved_directories.append(directory)
                    entries.extend(child.iterdir())
                else:
                    directories.append(directory)
            elif stat.S_ISREG(status.st_mode):
                if child != quality_path and not (
                    child.parent == reconstruction and child.name in retained
                ):
                    files.append((
                        child,
                        _legacy_output_identity(child),
                        sha256_file(child),
                    ))
            else:
                raise RuntimeError(
                    f"Completed reconstruction contains an unsafe entry: {child}"
                )
        for directory in directories:
            _safe_rmtree(*directory)
        if files:
            _remove_legacy_outputs(files)
        for directory, expected_identity in preserved_directories:
            current = directory.lstat()
            if (
                _is_link_or_reparse(current)
                or not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != expected_identity
            ):
                raise RuntimeError("Completed reconstruction identity changed")


def _quarantine_run_owned_directory(
    directory: tuple[Path, tuple[int, int]],
) -> tuple[Path, Path, tuple[int, int]]:
    path, expected_identity = directory
    status = path.lstat()
    if (
        _is_link_or_reparse(status)
        or not stat.S_ISDIR(status.st_mode)
        or (status.st_dev, status.st_ino) != expected_identity
    ):
        raise RuntimeError(f"Run directory identity changed before quarantine: {path}")
    quarantine = path.with_name(
        f".{path.name}.retry-quarantine-{secrets.token_hex(8)}"
    )
    os.replace(path, quarantine)
    moved = quarantine.lstat()
    if (moved.st_dev, moved.st_ino) != expected_identity:
        raise RuntimeError(f"Run directory identity changed during quarantine: {path}")
    return quarantine, path, expected_identity


def _restore_quarantined_run_directory(
    quarantine: tuple[Path, Path, tuple[int, int]],
) -> None:
    quarantined, original, expected_identity = quarantine
    status = quarantined.lstat()
    if (
        _is_link_or_reparse(status)
        or not stat.S_ISDIR(status.st_mode)
        or (status.st_dev, status.st_ino) != expected_identity
        or _path_entry_exists(original)
    ):
        raise RuntimeError(
            f"Quarantined run directory cannot be safely restored: {original}"
        )
    os.replace(quarantined, original)


def _discard_quarantined_run_directory(
    quarantine: tuple[Path, Path, tuple[int, int]],
) -> None:
    quarantined, _, expected_identity = quarantine
    try:
        _safe_rmtree(quarantined, expected_identity)
    except OSError:
        pass


def _pptx_reconstruction_directories(
    store: RunStore,
    page_jobs: dict[str, Any],
) -> list[tuple[Path, tuple[int, int]]]:
    directories = []
    for page_id in page_jobs["pages"]:
        directory = _run_owned_directory(
            store,
            Path("pages") / page_id / "reconstruction",
        )
        if directory is not None:
            directories.append(directory)
    return directories


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


def _legacy_output_hashes(
    outputs: object,
    records: list[tuple[Path, tuple[int, int, int, int, int], str]],
) -> dict[str, str]:
    if not isinstance(outputs, dict) or any(
        type(name) is not str or type(path) is not str
        for name, path in outputs.items()
    ):
        raise RuntimeError("Legacy outputs are invalid")
    hashes = {str(path): digest for path, _, digest in records}
    if len(hashes) != len(records) or set(hashes) != set(outputs.values()):
        raise RuntimeError("Legacy output records do not match outputs")
    return {name: hashes[path] for name, path in outputs.items()}


def _validate_completed_legacy_summary(
    store: RunStore,
    manifest: dict[str, Any],
    page_jobs: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    if (
        summary.get("status") != RunStatus.COMPLETED.value
        or type(summary.get("pages")) is not int
        or summary["pages"] != len(manifest["pages"])
        or summary.get("quality_gate_version")
        != COMPONENT_QUALITY_GATE_VERSION
    ):
        raise RuntimeError("Completed legacy summary values are invalid")
    if summary.get("outputs") != _expected_legacy_outputs(store, manifest):
        raise RuntimeError(
            "Completed legacy summary outputs do not match manifest"
        )
    pages = page_jobs.get("pages")
    if (
        not isinstance(pages, dict)
        or set(pages) != set(manifest["pages"])
        or any(
            not isinstance(page, dict)
            or page.get("status") not in {
                PageStatus.VALIDATED.value,
                PageStatus.PRESERVED_WITH_WARNING.value,
            }
            for page in pages.values()
        )
    ):
        raise RuntimeError("Completed legacy page states are invalid")


def _legacy_assembly_records(
    store: RunStore,
    outputs: dict[str, str],
    returned_records: list[tuple[Path, tuple[int, int, int, int, int], str]],
) -> list[tuple[Path, tuple[int, int, int, int, int], str]]:
    try:
        persisted = _load_legacy_output_records(store, expected_outputs=outputs)
    except RuntimeError as error:
        if str(error) != "Legacy output records are missing":
            raise
        persisted = []
    if persisted:
        if returned_records and set(returned_records) != set(persisted):
            raise RuntimeError(
                "Legacy returned output records do not match persisted records"
            )
        return persisted
    if returned_records:
        raise RuntimeError("Legacy output records are missing")
    if not outputs:
        return []
    records = {}
    for name, value in outputs.items():
        if type(name) is not str or type(value) is not str:
            raise RuntimeError("Legacy outputs are invalid")
        path = Path(value)
        identity = _legacy_output_identity(path)
        digest = sha256_file(path)
        if _legacy_output_identity(path) != identity:
            raise RuntimeError("Legacy output changed before record creation")
        records[name] = (path, identity, digest)
    _write_legacy_output_records(store, records)
    return list(records.values())


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
    if "performance" in summary:
        expected_public_keys.add("performance")
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


def _validate_pptx_shadow_public_summary(
    summary: object,
    expected_output: Path,
    slide_count: int,
    original_objects: int,
    original_candidates: int,
    input_sha256: str,
    resource_policy: dict[str, object],
) -> None:
    if type(summary) is not dict:
        raise RuntimeError("PPTX shadow summary must be an object")
    expected_keys = {
        "schema_version",
        "status",
        "pages",
        "preserved_objects",
        "pending_candidates",
        "replaced_pages",
        "preserved_with_warning_pages",
        "page_results",
        "warnings",
        "outputs",
        "input_sha256",
        "output_sha256",
        "resource_policy",
    }
    if "performance" in summary:
        expected_keys.add("performance")
    if set(summary) != expected_keys:
        raise RuntimeError("PPTX shadow summary fields are invalid")
    page_results = summary.get("page_results")
    if (
        not isinstance(page_results, list)
        or len(page_results) != slide_count
    ):
        raise RuntimeError("PPTX shadow page results are invalid")
    result_pages = []
    statuses = []
    for item in page_results:
        if (
            not isinstance(item, dict)
            or item.get("schema_version") != SCHEMA_VERSION
            or not isinstance(item.get("page_id"), str)
            or item.get("status")
            not in {
                PageStatus.PRESERVED.value,
                PageStatus.REPLACED.value,
                PageStatus.PRESERVED_WITH_WARNING.value,
            }
        ):
            raise RuntimeError("PPTX shadow page result is invalid")
        result_pages.append(item["page_id"])
        statuses.append(item["status"])
    replaced_pages = statuses.count(PageStatus.REPLACED.value)
    warning_pages = statuses.count(
        PageStatus.PRESERVED_WITH_WARNING.value
    )
    warnings = summary.get("warnings")
    try:
        summary_policy = validate_resource_policy(
            summary.get("resource_policy")
        )
    except ValueError as error:
        raise RuntimeError("PPTX shadow resource policy is invalid") from error
    if (
        len(set(result_pages)) != slide_count
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != RunStatus.COMPLETED.value
        or summary.get("pages") != slide_count
        or summary.get("preserved_objects")
        != original_objects - replaced_pages
        or summary.get("pending_candidates")
        != original_candidates - replaced_pages
        or summary.get("replaced_pages") != replaced_pages
        or summary.get("preserved_with_warning_pages") != warning_pages
        or not isinstance(warnings, list)
        or any(not isinstance(item, str) for item in warnings)
        or summary.get("outputs") != {"pptx": str(expected_output)}
        or summary.get("input_sha256") != input_sha256
        or not _is_sha256(summary.get("output_sha256"))
        or summary_policy != resource_policy
    ):
        raise RuntimeError("PPTX shadow summary values are invalid")


def _validate_pptx_shadow_execution_summary(
    summary: object,
    expected_output: Path,
    slide_count: int,
    original_objects: int,
    original_candidates: int,
    input_sha256: str,
    resource_policy: dict[str, object],
) -> None:
    if type(summary) is not dict:
        raise RuntimeError("PPTX shadow summary must be an object")
    public_summary = dict(summary)
    token = public_summary.pop("_output_identity", None)
    _validate_pptx_shadow_public_summary(
        public_summary,
        expected_output,
        slide_count,
        original_objects,
        original_candidates,
        input_sha256,
        resource_policy,
    )
    if (
        not isinstance(token, dict)
        or token.get("sha256") != public_summary["output_sha256"]
    ):
        raise RuntimeError(
            "PPTX shadow summary identity hash does not match output"
        )


def _validate_completed_shadow_pages(
    store: RunStore,
    manifest: dict[str, Any],
    page_jobs: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    manifest_pages = _pptx_page_ids_manifest(manifest, page_jobs)
    by_page = {
        item["page_id"]: item
        for item in summary["page_results"]
        if isinstance(item, dict) and isinstance(item.get("page_id"), str)
    }
    if set(by_page) != set(manifest_pages):
        raise RuntimeError("PPTX shadow page results do not match manifest")
    for page_id in manifest_pages:
        result = by_page[page_id]
        if page_jobs["pages"][page_id]["status"] != result.get("status"):
            raise RuntimeError("PPTX shadow page status does not match result")
        stored = store.read_json(
            Path("pages") / page_id / "page_result.json"
        )
        if stored != result:
            raise RuntimeError("PPTX shadow stored page result changed")


def _pptx_page_ids_manifest(
    manifest: dict[str, Any],
    page_jobs: dict[str, Any],
) -> list[str]:
    manifest_pages = manifest.get("pages")
    pages = page_jobs.get("pages")
    if (
        not isinstance(manifest_pages, list)
        or not isinstance(pages, dict)
        or any(not isinstance(page_id, str) for page_id in manifest_pages)
        or len(pages) != len(manifest_pages)
        or set(pages) != set(manifest_pages)
    ):
        raise RuntimeError("PPTX manifest pages do not match page jobs")
    return manifest_pages


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
    *,
    keep_isolated: bool = False,
) -> Path | None:
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
        return None
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
    if keep_isolated:
        return isolated
    isolated.unlink()
    return None


def run_job(run_dir: str | Path) -> dict[str, Any]:
    return _run_job(run_dir, _lease=None)


def _run_job(
    run_dir: str | Path,
    *,
    _lease: ExecutionLease | None,
) -> dict[str, Any]:
    store = RunStore.open(run_dir)
    manifest, input_type = _manifest_input(store)
    resource_policy = _manifest_resource_policy(manifest)
    state = store.read_json("run_state.json")
    page_jobs = store.read_json("page_jobs.json")
    if state["status"] == RunStatus.COMPLETED.value:
        if input_type == "pptx":
            (
                pptx_slide_count,
                pptx_preserved_objects,
                pptx_pending_candidates,
                pptx_input_sha256,
            ) = _pptx_manifest_expectations(manifest, page_jobs)
            validate_pptx_inventories(store, manifest)
            summary = store.read_json("run_summary.json")
            if "performance" in summary:
                _validate_performance_summary(summary["performance"], manifest["pages"])
            if "page_results" in summary:
                _validate_completed_shadow_pages(
                    store,
                    manifest,
                    page_jobs,
                    summary,
                )
                _validate_pptx_shadow_public_summary(
                    summary,
                    _pptx_output_path(store, manifest),
                    pptx_slide_count,
                    pptx_preserved_objects,
                    pptx_pending_candidates,
                    pptx_input_sha256,
                    resource_policy,
                )
                completed_output_sha256 = summary["output_sha256"]
            else:
                _pptx_page_ids(
                    manifest,
                    page_jobs,
                    PageStatus.PRESERVED,
                )
                _validate_pptx_public_summary(
                    summary,
                    _pptx_output_path(store, manifest),
                    pptx_slide_count,
                    pptx_preserved_objects,
                    pptx_pending_candidates,
                    pptx_input_sha256,
                    resource_policy,
                )
                completed_output_sha256 = pptx_input_sha256
            _validate_completed_pptx_output(
                _pptx_output_path(store, manifest),
                completed_output_sha256,
            )
            return summary
        summary = store.read_json("run_summary.json")
        validate_schema_version(summary)
        _validate_completed_resource_policy(summary, resource_policy)
        _validate_completed_legacy_summary(
            store, manifest, page_jobs, summary
        )
        if "performance" in summary:
            _validate_performance_summary(summary["performance"], manifest["pages"])
        output_records = _load_legacy_output_records(
            store,
            expected_outputs=summary.get("outputs"),
        )
        expected_hashes = _legacy_output_hashes(
            summary.get("outputs"), output_records
        )
        if summary.get("output_sha256") != expected_hashes:
            raise RuntimeError(
                "Completed legacy output hashes do not match records"
            )
        for record in output_records:
            _verify_legacy_output_record(record)
        return summary
    if state["status"] != RunStatus.PREPARED.value:
        raise RuntimeError(
            f"Run must be prepared before execution; current status is {state['status']}"
        )
    if _lease is None:
        with ExecutionLease(
            store.root / "execution.lock",
            run_root=store.root,
        ) as lease:
            return _run_job(store.root, _lease=lease)

    execution_started = time.perf_counter()
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
        try:
            page_ids = _pptx_page_ids(manifest, page_jobs, PageStatus.ANALYZED)
        except RuntimeError:
            # A resumed Host run has durable page states (processing,
            # awaiting_agent or validated) rather than the prepare-time
            # ``analyzed`` marker.  Continue only with known page states.
            allowed = {
                PageStatus.ANALYZED.value, PageStatus.PROCESSING.value,
                PageStatus.AWAITING_AGENT.value, PageStatus.VALIDATED.value,
                PageStatus.PRESERVED_WITH_WARNING.value,
            }
            pages = page_jobs.get("pages", {})
            if set(pages) != set(manifest.get("pages", [])) or any(
                page.get("status") not in allowed
                for page in pages.values()
            ):
                raise
            page_ids = list(manifest["pages"])
        (
            pptx_slide_count,
            pptx_preserved_objects,
            pptx_pending_candidates,
            pptx_input_sha256,
        ) = _pptx_manifest_expectations(manifest, page_jobs)
        validate_pptx_inventories(store, manifest)
        # Component repair may pause at the Agent boundary.  Enter RUNNING
        # only after all immutable PPTX input/inventory checks pass so a
        # rejected prepare remains recoverable in PREPARED.
        store.transition_run(RunStatus.RUNNING)
        ocr_worker_pool = None
        visual_worker_pool = None
        try:
            visual_page_ids = [
                page_id for page_id in page_ids
                if _native_pdf_analysis(store, page_id) is None
            ]
            module = (
                importlib.import_module("image_to_ppt")
                if visual_page_ids else None
            )
            if module is not None:
                ocr_worker_pool = module.create_ocr_worker_pool()
            pipeline_mode = manifest.get("options", {}).get(
                "pipeline_mode", "strict"
            )
            if module is not None and pipeline_mode != "fast":
                visual_worker_pool = module.create_visual_worker_pool()

            def ensure_visual_worker_pool():
                nonlocal visual_worker_pool
                if visual_worker_pool is None:
                    if module is None:
                        raise RuntimeError(
                            "visual worker requested for native PDF page"
                        )
                    visual_worker_pool = module.create_visual_worker_pool()
                return visual_worker_pool

            # Advance any durable component-reconstruction state before asking
            # the shadow planner for donor/OOXML work.  Host rounds therefore
            # remain at the durable boundary until assembly-ready.
            for page_id in page_ids:
                state_path = (
                    store.root / "pages" / page_id / "reconstruction"
                    / "component_state.json"
                )
                request_path = (
                    store.root / "pages" / page_id / "page_request.json"
                )
                if state_path.is_file() or not request_path.is_file():
                    continue
                # An approved page request is the only entry point for
                # component extraction.  Full-page candidates use the
                # deterministic layer builder; other approved candidates use
                # the existing isolated CV initializer.
                initialize_kwargs = {
                    "_lease": _lease,
                    "performance_trace": _page_performance_trace(store, page_id),
                    "ocr_worker_pool": ocr_worker_pool,
                    "visual_worker_pool": visual_worker_pool,
                }
                if pipeline_mode == "fast":
                    initialize_kwargs["visual_worker_pool_factory"] = (
                        ensure_visual_worker_pool
                    )
                initialize_legacy_page(store, page_id, **initialize_kwargs)
            existing_component_pages = [
                page_id
                for page_id in page_ids
                if (
                    store.root / "pages" / page_id / "reconstruction"
                    / "component_state.json"
                ).is_file()
            ]
            if existing_component_pages:
                _ensure_legacy_pages_processing(store, existing_component_pages)
                advance_kwargs = {
                    "ocr_worker_pool": ocr_worker_pool,
                    "visual_worker_pool": visual_worker_pool,
                }
                if pipeline_mode == "fast":
                    advance_kwargs["visual_worker_pool_factory"] = (
                        ensure_visual_worker_pool
                    )
                waiting = _advance_legacy_pages(
                    store,
                    manifest,
                    existing_component_pages,
                    _lease,
                    **advance_kwargs,
                )
                if waiting is not None:
                    return waiting
            # Do not enter the PPTX shadow pipeline while a Host component
            # round is awaiting its plan.
            awaiting_components = []
            for page_id in page_ids:
                state_path = (
                    store.root / "pages" / page_id / "reconstruction"
                    / "component_state.json"
                )
                if state_path.is_file():
                    component_state = store.read_json(
                        f"pages/{page_id}/reconstruction/component_state.json"
                    )
                    validate_component_repair_state(component_state)
                    if (
                        component_state.get("provider") == "host"
                        and component_state.get("phase") not in {
                            "ready_for_assembly", "preserved_with_warning"
                        }
                    ):
                        awaiting_components.append(page_id)
            if awaiting_components:
                _transition_pages(
                    store, awaiting_components, PageStatus.AWAITING_AGENT
                )
                store.transition_run(RunStatus.AWAITING_AGENT)
                summary = _legacy_waiting_summary(
                    store, manifest, awaiting_components[0],
                    {"status": "awaiting_agent"},
                )
                store.write_json("run_summary.json", summary)
                return summary
            _finalize_reconstruction_routes(
                store, manifest, existing_component_pages
            )
            pptx_shadow_plans = shadow_replacement_plans(store, manifest)
            pptx_expected_output = _pptx_output_path(store, manifest)
            pptx_output_existed = _path_entry_exists(pptx_expected_output)
        except Exception as error:
            cleanup_error = _record_failure(store, page_ids, error)
            if cleanup_error is not None:
                raise error from cleanup_error
            raise
        finally:
            if visual_worker_pool is not None:
                visual_worker_pool.close()
            if ocr_worker_pool is not None:
                ocr_worker_pool.close()
    else:
        page_ids = list(page_jobs["pages"])
        pptx_shadow_plans = []
        pptx_expected_output = None
        pptx_output_existed = False
        ocr_worker_pool = None
        visual_worker_pool = None
        try:
            visual_page_ids = [
                page_id for page_id in page_ids
                if _native_pdf_analysis(store, page_id) is None
            ]
            module = (
                importlib.import_module("image_to_ppt")
                if visual_page_ids else None
            )
            if module is not None:
                ocr_worker_pool = module.create_ocr_worker_pool()
            pipeline_mode = manifest.get("options", {}).get(
                "pipeline_mode", "strict"
            )
            if module is not None and pipeline_mode != "fast":
                visual_worker_pool = module.create_visual_worker_pool()

            def ensure_visual_worker_pool():
                nonlocal visual_worker_pool
                if visual_worker_pool is None:
                    if module is None:
                        raise RuntimeError(
                            "visual worker requested for native PDF page"
                        )
                    visual_worker_pool = module.create_visual_worker_pool()
                return visual_worker_pool

            store.transition_run(RunStatus.RUNNING)
            _ensure_legacy_pages_processing(store, page_ids)
            advance_kwargs = {
                "ocr_worker_pool": ocr_worker_pool,
                "visual_worker_pool": visual_worker_pool,
            }
            if pipeline_mode == "fast":
                advance_kwargs["visual_worker_pool_factory"] = (
                    ensure_visual_worker_pool
                )
            waiting = _advance_legacy_pages(
                store, manifest, page_ids, _lease, **advance_kwargs
            )
            if waiting is not None:
                return waiting
        except Exception as error:
            cleanup_error = _record_failure(store, page_ids, error)
            if cleanup_error is not None:
                raise error from cleanup_error
            raise
        finally:
            if visual_worker_pool is not None:
                visual_worker_pool.close()
            if ocr_worker_pool is not None:
                ocr_worker_pool.close()
    if store.read_json("run_state.json")["status"] != RunStatus.RUNNING.value:
        store.transition_run(RunStatus.RUNNING)
    pptx_output_published = False
    pptx_output_record = None
    legacy_output_records = []
    legacy_staging_records = []

    try:
        if input_type == "pptx":
            if pptx_shadow_plans:
                current_pages = store.read_json("page_jobs.json")["pages"]
                _transition_pages(
                    store,
                    [
                        plan["page_id"] for plan in pptx_shadow_plans
                        if current_pages[plan["page_id"]]["status"]
                        not in {
                            PageStatus.VALIDATED.value,
                            PageStatus.PRESERVED_WITH_WARNING.value,
                        }
                    ],
                    PageStatus.PROCESSING,
                )
            manifest_token = _PPTX_EXECUTION_MANIFEST.set(manifest)
            try:
                if pptx_shadow_plans:
                    summary = execute_pptx_shadow(
                        store,
                        pptx_shadow_plans,
                    )
                else:
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
                if pptx_shadow_plans:
                    _validate_pptx_shadow_execution_summary(
                        summary,
                        pptx_expected_output,
                        pptx_slide_count,
                        pptx_preserved_objects,
                        pptx_pending_candidates,
                        pptx_input_sha256,
                        resource_policy,
                    )
                else:
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
            if pptx_shadow_plans:
                for result in summary["page_results"]:
                    page_id = result["page_id"]
                    status = PageStatus(result["status"])
                    if status is PageStatus.REPLACED:
                        current_status = PageStatus(
                            store.read_json("page_jobs.json")["pages"][page_id]["status"]
                        )
                        if current_status is not PageStatus.VALIDATED:
                            _transition_pages(store, [page_id], PageStatus.VALIDATED)
                        if current_status is not PageStatus.REPLACED:
                            _transition_pages(store, [page_id], PageStatus.REPLACED)
                    elif status is PageStatus.PRESERVED_WITH_WARNING:
                        current_status = PageStatus(
                            store.read_json("page_jobs.json")["pages"][page_id]["status"]
                        )
                        if current_status is not PageStatus.PRESERVED_WITH_WARNING:
                            _transition_pages(
                                store,
                                [page_id],
                                PageStatus.PRESERVED_WITH_WARNING,
                            )
                    else:
                        _transition_pages(
                            store,
                            [page_id],
                            PageStatus.PRESERVED,
                        )
                    store.write_json(
                        Path("pages") / page_id / "page_result.json",
                        result,
                    )
            else:
                _transition_pages(store, page_ids, PageStatus.PRESERVED)
            store.transition_run(RunStatus.FINALIZING)
        else:
            _finalize_reconstruction_routes(store, manifest, page_ids)
            assembled_outputs = assemble_legacy_results(store)
            returned_records = list(
                getattr(assembled_outputs, "records", [])
            )
            outputs = dict(assembled_outputs)
            persisted_records = _legacy_assembly_records(
                store, outputs, returned_records
            )
            output_sha256 = _legacy_output_hashes(outputs, persisted_records)
            legacy_output_records = persisted_records
            for record in legacy_output_records:
                _verify_legacy_output_record(record)
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
            store.transition_run(RunStatus.FINALIZING)
            summary = {
                "schema_version": SCHEMA_VERSION,
                "status": RunStatus.COMPLETED.value,
                "pages": len(page_ids),
                "outputs": outputs,
                "output_sha256": output_sha256,
                "resource_policy": resource_policy,
                "quality_gate_version": COMPONENT_QUALITY_GATE_VERSION,
            }
        performance = _performance_summary(
            store,
            page_ids,
            total_duration_ms=round(
                (time.perf_counter() - execution_started) * 1000
            ),
        )
        summary["performance"] = performance
        if input_type != "pptx":
            for record in legacy_output_records:
                _verify_legacy_output_record(record)
        store.write_json("run_summary.json", summary)
        if input_type != "pptx":
            for record in legacy_output_records:
                _verify_legacy_output_record(record)
        store.transition_run(RunStatus.COMPLETED)
        if input_type != "pptx":
            for record in legacy_output_records:
                _verify_legacy_output_record(record)
        _write_performance_summaries(store, performance)
        if input_type != "pptx":
            for record in legacy_output_records:
                _verify_legacy_output_record(record)
        if input_type == "pdf":
            _cleanup_native_pdf_sources(store, page_ids)
        if input_type != "pptx":
            try:
                _prune_completed_legacy_artifacts(store, page_ids)
            except Exception:
                _LOGGER.warning(
                    "Completed legacy artifact cleanup failed", exc_info=True
                )
        return summary
    except Exception as error:
        compensation_error = None
        pptx_output_removed = False
        pages_restored = False
        legacy_compensation_failed = False
        if input_type != "pptx" and not legacy_output_records:
            try:
                legacy_output_records = _load_legacy_output_records(store)
            except Exception as caught:
                legacy_compensation_failed = True
                compensation_error = caught
        if input_type != "pptx":
            try:
                legacy_staging_records = _load_legacy_staging_records(store)
            except Exception as caught:
                legacy_compensation_failed = True
                if compensation_error is None:
                    compensation_error = caught
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
        if input_type != "pptx" and legacy_output_records:
            try:
                _remove_legacy_outputs(list(reversed(legacy_output_records)))
            except Exception as caught:
                legacy_compensation_failed = True
                if compensation_error is None:
                    compensation_error = caught
            if not legacy_compensation_failed:
                try:
                    _clear_legacy_output_records(store)
                except Exception as caught:
                    legacy_compensation_failed = True
                    if compensation_error is None:
                        compensation_error = caught
            if legacy_compensation_failed:
                retry_blocked = True
        if input_type != "pptx" and legacy_staging_records:
            try:
                existing_staging = [
                    record for record in legacy_staging_records
                    if _path_entry_exists(record[0])
                ]
                if existing_staging:
                    _remove_legacy_outputs(list(reversed(existing_staging)))
                _clear_legacy_staging_records(store)
            except Exception as caught:
                legacy_compensation_failed = True
                retry_blocked = True
                if compensation_error is None:
                    compensation_error = caught
        cleanup_error = _record_failure(
            store,
            page_ids,
            error,
            recover_completed=(
                (input_type == "pptx" and pptx_output_published)
                or bool(legacy_output_records)
                or bool(legacy_staging_records)
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
    preserve_page_ids: frozenset[str] = frozenset(),
) -> None:
    if _reset_page_jobs(
        page_jobs,
        analyzed=analyzed,
        preserve_page_ids=preserve_page_ids,
    ):
        store.write_json("page_jobs.json", page_jobs)


def _reset_page_jobs(
    page_jobs: dict[str, Any],
    *,
    analyzed: bool,
    preserve_page_ids: frozenset[str] = frozenset(),
) -> bool:
    pages = page_jobs["pages"]
    updates = {}
    for page_id, page in pages.items():
        if page_id in preserve_page_ids:
            continue
        status = PageStatus(page["status"])
        if status is PageStatus.PENDING:
            if analyzed:
                updates[page_id] = transition_page_document(
                    page, PageStatus.ANALYZED
                )
            continue
        if status is PageStatus.ANALYZED and analyzed:
            continue
        if status is PageStatus.PRESERVED_WITH_WARNING:
            page = {
                **page,
                "status": PageStatus.PENDING.value,
                "updated_at": utc_now(),
            }
            if analyzed:
                page = transition_page_document(page, PageStatus.ANALYZED)
            updates[page_id] = page
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


def _expected_legacy_outputs(
    store: RunStore,
    manifest: dict[str, Any],
) -> dict[str, str]:
    output = _manifest_output_path(manifest)
    if output is None:
        output_format = manifest.get("output_format", "pptx")
        pages = manifest.get("pages")
        if not isinstance(pages, list):
            raise RuntimeError("Run manifest pages must be an array")
        if output_format == "psd" and len(pages) == 1:
            output = store.root / "final" / "output.psd"
        elif output_format == "psd":
            output = store.root / "final"
        else:
            output = store.root / "final" / "output.pptx"
    output = output.absolute()
    if manifest.get("output_format", "pptx") == "psd":
        targets = _legacy_psd_output_targets(manifest, output)
    else:
        targets = _legacy_output_targets(
            output, manifest["options"]["slide_size"]
        )
    return {name: str(path) for name, path in targets.items()}


def _legacy_output_candidates(
    store: RunStore,
    manifest: dict[str, Any],
) -> set[Path]:
    candidates = {
        Path(path) for path in _expected_legacy_outputs(store, manifest).values()
    }
    declared = _manifest_output_path(manifest)
    if declared is not None:
        candidates.add(declared.absolute())
        pages = manifest.get("pages")
        options = manifest.get("options", {})
        if (
            isinstance(pages, list)
            and len(pages) > 1
            and isinstance(options, dict)
            and options.get("slide_size") in {"both", "original"}
            and manifest.get("output_format", "pptx") == "pptx"
        ):
            candidates.add(Path(f"{declared.absolute().with_suffix('')}_original"))
    return candidates


def _expected_legacy_output_entries(
    store: RunStore,
    manifest: dict[str, Any],
) -> list[Path]:
    return [Path(path) for path in _expected_legacy_outputs(store, manifest).values()]


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

        failed_summary = _failed_summary(store)
        if (
            failed_summary is not None
            and failed_summary.get("retry_blocked") is True
        ):
            raise RuntimeError(
                "Run recovery is blocked while an output entry may be owned "
                "by another process"
            )

        manifest, input_type = _manifest_input(store)
        page_jobs = store.read_json("page_jobs.json")
        pages_changed = _reset_page_jobs(
            page_jobs,
            analyzed=input_type == "pptx",
        )
        cleanup_candidates = [_run_owned_directory(store, "work")]
        if input_type == "pptx":
            cleanup_candidates.append(_run_owned_directory(store, "final"))
            cleanup_candidates.extend(
                _pptx_reconstruction_directories(store, page_jobs)
            )
        cleanup = [
            directory
            for directory in cleanup_candidates
            if directory is not None
        ]

        if input_type == "pptx":
            expected_output = _pptx_output_path(store, manifest)
            if _path_entry_exists(expected_output):
                raise RuntimeError(
                    f"PPTX recovery is blocked by an existing output: "
                    f"{expected_output}"
                )
        else:
            expected_outputs = _expected_legacy_outputs(store, manifest)
            output_records = _load_legacy_output_records(store)
            staging_records = _load_legacy_staging_records(store)
            if output_records:
                _legacy_output_hashes(expected_outputs, output_records)
                published_records = [
                    record for record in output_records
                    if _path_entry_exists(record[0])
                ]
                for record in published_records:
                    _verify_legacy_output_record(record)
            else:
                published_records = []
            recorded_paths = {record[0] for record in output_records}
            for output in _legacy_output_candidates(store, manifest):
                if _path_entry_exists(output) and output not in recorded_paths:
                    label = (
                        "an unbound legacy output"
                        if _is_owned_final_output(store, output)
                        else "an existing external output"
                    )
                    raise RuntimeError(
                        f"Run recovery is blocked by {label}: {output}"
                    )
            if published_records:
                _remove_legacy_outputs(list(reversed(published_records)))
            existing_staging = [
                record for record in staging_records
                if _path_entry_exists(record[0])
            ]
            for record in existing_staging:
                _verify_legacy_output_record(record)
            if existing_staging:
                _remove_legacy_outputs(list(reversed(existing_staging)))
            if output_records:
                _clear_legacy_output_records(store)
            if staging_records:
                _clear_legacy_staging_records(store)

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
    retrying_warning_wait = (
        run_status == RunStatus.AWAITING_AGENT.value
        and page_jobs["pages"][page_id]["status"]
        == PageStatus.PRESERVED_WITH_WARNING.value
    )
    retrying_completed_warning = (
        run_status == RunStatus.COMPLETED.value
        and page_jobs["pages"][page_id]["status"]
        == PageStatus.PRESERVED_WITH_WARNING.value
    )
    if not (
        retrying_failed_run
        or continuing_retry
        or retrying_warning_wait
        or retrying_completed_warning
    ):
        raise RuntimeError(f"Run is not failed or continuing a retry: {page_id}")
    if (
        not retrying_completed_warning
        and failed_summary is not None
        and failed_summary.get("retry_blocked") is True
    ):
        raise RuntimeError(
            f"Retry is blocked while an output entry may be owned "
            f"by another process: {page_id}"
        )
    try:
        component_state = validate_component_repair_state(store.read_json(
            f"pages/{page_id}/reconstruction/component_state.json"
        ))
    except (FileNotFoundError, ValueError):
        component_state = None
    if (
        component_state is not None
        and component_state["phase"] == "preserved_with_warning"
        and (
            component_state["stop_reason"] not in {
                "round_limit", "no_quality_improvement",
            }
            or component_state["repair_round"] >= MAX_REPAIR_ROUNDS
        )
    ):
        raise RuntimeError(
            "Component warning reached a non-resumable repair boundary; "
            "the existing output and reconstruction were preserved"
        )
    if retrying_failed_run:
        resumed_warning = resume_round_limited_component_repair(
            store, page_id
        ) if component_state is not None else False
        if (
            component_state is not None
            and (
                resumed_warning
                or (
                    component_state["status"] == "active"
                    and component_state["phase"] in {
                        "request_published", "awaiting_plan", "plan_recorded",
                        "actions_executed", "quality_recorded", "freeze_committed",
                        "fallback_required", "fallback_executed",
                        "fallback_quality_recorded",
                    }
                )
            )
        ):
            for retry_page_id, page in page_jobs["pages"].items():
                if retry_page_id == page_id:
                    page["status"] = PageStatus.ANALYZED.value
                else:
                    try:
                        other_state = validate_component_repair_state(
                            store.read_json(
                                f"pages/{retry_page_id}/reconstruction/"
                                "component_state.json"
                            )
                        )
                    except (FileNotFoundError, ValueError):
                        continue
                    if other_state["phase"] == "ready_for_assembly":
                        page["status"] = PageStatus.VALIDATED.value
                page["updated_at"] = utc_now()
            store.write_json("page_jobs.json", page_jobs)
            store.write_json("run_state.json", {
                **store.read_json("run_state.json"),
                "status": RunStatus.PREPARED.value,
                "updated_at": utc_now(),
            })
            return get_status(store.root)
    if (
        input_type == "pptx"
        and not retrying_completed_warning
        and _path_entry_exists(_pptx_output_path(store, manifest))
    ):
        raise RuntimeError(
            f"PPTX retry is blocked while an output entry may be owned "
            f"by another process: {page_id}"
        )
    if (
        input_type != "pptx"
        and not retrying_completed_warning
        and (
            any(
                _path_entry_exists(output)
                for output in _expected_legacy_output_entries(store, manifest)
            )
            or (
                (final_directory := _run_owned_directory(store, "final"))
                is not None
                and next(final_directory[0].iterdir(), None) is not None
            )
        )
    ):
        raise RuntimeError(
            f"Legacy retry is blocked while an output entry may be owned "
            f"by another process: {page_id}"
        )
    if retrying_completed_warning:
        if input_type != "pptx":
            raise RuntimeError(
                "Completed warning retry is only supported for PPTX inputs"
            )
        summary = store.read_json("run_summary.json")
        validate_schema_version(summary)
        output = _pptx_output_path(store, manifest)
        page_results = summary.get("page_results")
        if (
            summary.get("status") != RunStatus.COMPLETED.value
            or summary.get("outputs") != {"pptx": str(output)}
            or summary.get("input_sha256") != manifest["input"]["sha256"]
            or not _is_sha256(summary.get("output_sha256"))
            or not isinstance(page_results, list)
            or not any(
                result.get("page_id") == page_id
                and result.get("status")
                == PageStatus.PRESERVED_WITH_WARNING.value
                for result in page_results
                if isinstance(result, dict)
            )
        ):
            raise RuntimeError(
                "Completed PPTX warning retry summary is invalid"
            )
        expected_hash = summary["output_sha256"]
        identity = _pptx_output_identity(output)
        digest = sha256_file(output)
        if (
            digest != expected_hash
            or _pptx_output_identity(output) != identity
        ):
            raise RuntimeError(
                "Completed PPTX output changed and cannot be safely retried"
            )
        isolated = _isolate_recorded_pptx_output(
            (output, identity, expected_hash), keep_isolated=True
        )
        if isolated is None:
            raise RuntimeError("Completed PPTX output disappeared during retry")
        previous_pages = store.read_json("page_jobs.json")
        previous_run = store.read_json("run_state.json")
        previous_component_state = None
        resumed_component_state = False
        quarantined_reconstruction = None
        try:
            for other_id, other_page in page_jobs["pages"].items():
                if other_id == page_id:
                    other_page["status"] = PageStatus.ANALYZED.value
                    other_page["updated_at"] = utc_now()
                elif other_page["status"] == PageStatus.REPLACED.value:
                    other_page["status"] = PageStatus.VALIDATED.value
                    other_page["updated_at"] = utc_now()
                elif (
                    other_page["status"]
                    == PageStatus.PRESERVED_WITH_WARNING.value
                ):
                    try:
                        component_state = validate_component_repair_state(
                            store.read_json(
                                f"pages/{other_id}/reconstruction/"
                                "component_state.json"
                            )
                        )
                    except (FileNotFoundError, ValueError):
                        continue
                    if (
                        component_state["page_id"] == other_id
                        and component_state["phase"] == "ready_for_assembly"
                        and component_state["status"] == "ready_for_assembly"
                    ):
                        other_page["status"] = PageStatus.VALIDATED.value
                        other_page["updated_at"] = utc_now()
            store.write_json("page_jobs.json", page_jobs)
            store.write_json("run_state.json", {
                **previous_run,
                "status": RunStatus.PREPARED.value,
                "updated_at": utc_now(),
            })
            component_state_path = (
                f"pages/{page_id}/reconstruction/component_state.json"
            )
            try:
                previous_component_state = store.read_json(component_state_path)
            except FileNotFoundError:
                previous_component_state = None
            resumed_component_state = resume_round_limited_component_repair(
                store, page_id
            )
            reconstruction = _run_owned_directory(
                store, Path("pages") / page_id / "reconstruction"
            )
            if not resumed_component_state:
                _clear_host_plan_records(store, page_id)
            if reconstruction is not None and not resumed_component_state:
                quarantined_reconstruction = _quarantine_run_owned_directory(
                    reconstruction
                )
            isolated.unlink()
        except Exception:
            if quarantined_reconstruction is not None:
                _restore_quarantined_run_directory(quarantined_reconstruction)
            store.write_json("page_jobs.json", previous_pages)
            store.write_json("run_state.json", previous_run)
            if previous_component_state is not None:
                store.write_json(
                    f"pages/{page_id}/reconstruction/component_state.json",
                    previous_component_state,
                )
            _restore_isolated_pptx_output(isolated, output)
            raise
        if quarantined_reconstruction is not None:
            _discard_quarantined_run_directory(quarantined_reconstruction)
        return get_status(store.root)
    if retrying_warning_wait:
        if resume_round_limited_component_repair(store, page_id):
            page = {
                **page_jobs["pages"][page_id],
                "status": PageStatus.ANALYZED.value,
                "updated_at": utc_now(),
            }
            page_jobs["pages"][page_id] = page
            store.write_json("page_jobs.json", page_jobs)
            store.write_json("run_state.json", {
                **store.read_json("run_state.json"),
                "status": RunStatus.PREPARED.value,
                "updated_at": utc_now(),
            })
            return get_status(store.root)
        reconstruction = _run_owned_directory(
            store, Path("pages") / page_id / "reconstruction"
        )
        _clear_host_plan_records(store, page_id)
        if reconstruction is not None:
            _safe_rmtree(*reconstruction)
        page = {
            **page_jobs["pages"][page_id],
            "status": PageStatus.PENDING.value,
            "updated_at": utc_now(),
        }
        if input_type == "pptx":
            page = transition_page_document(page, PageStatus.ANALYZED)
        page_jobs["pages"][page_id] = page
        store.write_json("page_jobs.json", page_jobs)
        store.transition_run(RunStatus.PREPARED)
        return get_status(store.root)
    reusable_page_ids = set()
    for retry_page_id, page in page_jobs["pages"].items():
        if retry_page_id == page_id:
            continue
        try:
            other_state = validate_component_repair_state(store.read_json(
                f"pages/{retry_page_id}/reconstruction/component_state.json"
            ))
        except (FileNotFoundError, ValueError):
            continue
        if (
            other_state["page_id"] == retry_page_id
            and other_state["phase"] == "ready_for_assembly"
            and other_state["status"] == "ready_for_assembly"
        ):
            reusable_page_ids.add(retry_page_id)
            page["status"] = PageStatus.VALIDATED.value
            page["updated_at"] = utc_now()
    cleanup = []
    work = _run_work_directory(store)
    if work is not None:
        cleanup.append(work)
    cleanup.extend(
        directory
        for retry_page_id, page in page_jobs["pages"].items()
        if retry_page_id not in reusable_page_ids
        for directory in [
            _run_owned_directory(
                store, Path("pages") / retry_page_id / "reconstruction"
            )
        ]
        if directory is not None
    )
    for directory in cleanup:
        _safe_rmtree(*directory)
    for retry_page_id in page_jobs["pages"]:
        if retry_page_id in reusable_page_ids:
            continue
        _clear_host_plan_records(store, retry_page_id)
    if orphaned_failed_batch:
        store.transition_run(RunStatus.FAILED)
        run_status = RunStatus.FAILED.value
    _reset_pages_for_retry(
        store,
        page_jobs,
        analyzed=input_type == "pptx",
        preserve_page_ids=frozenset(reusable_page_ids),
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
    agent_provider: str = "host",
    output_format: str = "pptx",
    pipeline_mode: str = "strict",
) -> dict[str, Any]:
    prepare_kwargs: dict[str, Any] = {
        "run_dir": run_dir,
        "output_path": output_path,
        "slide_size": slide_size,
        "lang": lang,
        "agent_provider": agent_provider,
    }
    if pipeline_mode != "strict":
        prepare_kwargs["pipeline_mode"] = pipeline_mode
    if output_format != "pptx":
        prepare_kwargs["output_format"] = output_format
    prepared = prepare_job(inputs, **prepare_kwargs)
    return run_job(prepared)
