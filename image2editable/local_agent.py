from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time

from image2editable.component_contracts import (
    MAX_REPAIR_ROUNDS,
    validate_component_plan,
)


_PLAN_LIMIT = 4 * 1024 * 1024
_DIAGNOSTIC_TEXT_LIMIT = 32 * 1024
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "FLAGS_paddle_num_threads",
)
_LOGGER = logging.getLogger(__name__)


def run_local_agent(
    request_path: str | Path,
    *,
    model_receipt: dict,
    resource_policy: dict | None = None,
    timeout_seconds: int = 600,
    performance_trace=None,
) -> dict:
    from image2editable.component_repair import (
        _ensure_owned_directory,
        _write_exclusive,
        load_component_agent_graph,
        load_component_agent_request,
    )

    request_path = Path(request_path).resolve()
    request = load_component_agent_request(request_path)
    if request["provider"] != "local":
        raise RuntimeError("Local Agent requires provider local")
    graph = load_component_agent_graph(request_path)
    _ensure_page_disk_budget(request_path, request, graph)
    snapshot = _model_snapshot(model_receipt)
    environment = _worker_environment(resource_policy)
    reconstruction = request_path.parents[2]
    with tempfile.TemporaryDirectory(
        prefix=".local-agent-worker-",
        dir=reconstruction,
    ) as temporary:
        output_path = Path(temporary) / "component-plan.json"
        command = [
            sys.executable,
            "-m",
            "image2editable.local_agent_worker",
            "--request",
            str(request_path),
            "--model-snapshot",
            str(snapshot),
            "--output",
            str(output_path),
        ]
        started = time.perf_counter() if performance_trace is not None else None
        try:
            completed = _invoke_worker(
                command,
                environment=environment,
                timeout_seconds=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            _record_local_agent_performance(
                performance_trace, started, request_path, request, status="error"
            )
            diagnostic_error = _write_diagnostic(
                request_path,
                request,
                {
                    "status": "worker_timeout",
                    "timeout_seconds": timeout_seconds,
                    "stdout": _diagnostic_text(error.stdout),
                    "stderr": _diagnostic_text(error.stderr),
                },
                ensure_directory=_ensure_owned_directory,
                write_exclusive=_write_exclusive,
            )
            message = f"Local Agent worker timed out after {timeout_seconds} seconds"
            if diagnostic_error is not None:
                message += "; diagnostic could not be written safely"
            raise RuntimeError(message) from error
        except BaseException:
            _record_local_agent_performance(
                performance_trace, started, request_path, request, status="error"
            )
            raise
        if completed.returncode != 0:
            _record_local_agent_performance(
                performance_trace, started, request_path, request, status="failed"
            )
            diagnostic_error = _write_diagnostic(
                request_path,
                request,
                {
                    "status": "worker_failed",
                    "returncode": completed.returncode,
                    "stdout": _diagnostic_text(completed.stdout),
                    "stderr": _diagnostic_text(completed.stderr),
                },
                ensure_directory=_ensure_owned_directory,
                write_exclusive=_write_exclusive,
            )
            message = (
                f"Local Agent worker exited with exit code {completed.returncode}"
            )
            if diagnostic_error is not None:
                message += "; diagnostic could not be written safely"
            raise RuntimeError(message)
        try:
            plan = _read_plan(output_path)
            request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
            if plan.get("request_sha256") != request_sha256:
                raise ValueError(
                    "component plan request_sha256 does not match current request"
                )
            validate_component_plan(plan, request=request, graph=graph)
        except Exception as error:
            _record_local_agent_performance(
                performance_trace, started, request_path, request, status="error"
            )
            _write_diagnostic(
                request_path,
                request,
                {
                    "status": "invalid_plan",
                    "error": str(error),
                    "stdout": _diagnostic_text(completed.stdout),
                    "stderr": _diagnostic_text(completed.stderr),
                },
                ensure_directory=_ensure_owned_directory,
                write_exclusive=_write_exclusive,
            )
            raise
        _record_local_agent_performance(
            performance_trace, started, request_path, request, status="success"
        )
        return plan


def run_local_service_agent(
    request_path: str | Path,
    *,
    service_config: object,
    timeout_seconds: int = 600,
    performance_trace=None,
) -> dict:
    from image2editable.component_repair import (
        load_component_agent_graph,
        load_component_agent_request,
    )
    from image2editable.local_agent_worker import _messages
    from image2editable.local_service import complete

    request_path = Path(request_path).resolve()
    request = load_component_agent_request(request_path)
    if request["provider"] != "local":
        raise RuntimeError("Local Agent requires provider local")
    graph = load_component_agent_graph(request_path)
    _ensure_page_disk_budget(request_path, request, graph)
    evidence = {
        name: request_path.parent / Path(*record["path"].split("/"))
        for name, record in request["evidence"].items()
    }
    quality_text = evidence["quality-report.json"].read_text(
        encoding="utf-8", errors="replace"
    )
    request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
    messages = _service_messages(
        _messages(request, graph, quality_text, evidence, request_sha256)
    )
    started = time.perf_counter() if performance_trace is not None else None
    try:
        plan = json.loads(
            complete(service_config, messages=messages, timeout_seconds=timeout_seconds)
        )
        validate_component_plan(plan, request=request, graph=graph)
        if plan.get("request_sha256") != request_sha256:
            raise ValueError("component plan request_sha256 does not match current request")
    except BaseException:
        _record_local_agent_performance(
            performance_trace, started, request_path, request, status="error"
        )
        raise
    _record_local_agent_performance(
        performance_trace, started, request_path, request, status="success"
    )
    return plan


def _service_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            converted.append(message)
            continue
        parts = []
        for part in content:
            if part.get("type") != "image":
                parts.append(part)
                continue
            path = Path(part["image"])
            media_type = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                }
            )
        converted.append({**message, "content": parts})
    return converted


def _model_snapshot(receipt: object) -> Path:
    if not isinstance(receipt, dict):
        raise ValueError("Local Agent model receipt must be an object")
    resolved_revision = receipt.get("resolved_revision")
    snapshot_value = receipt.get("snapshot_path")
    if not isinstance(resolved_revision, str) or not isinstance(snapshot_value, str):
        raise ValueError("Local Agent model receipt is invalid")
    snapshot = Path(snapshot_value).resolve()
    if not snapshot.is_dir() or snapshot.name.lower() != resolved_revision:
        raise RuntimeError("Local Agent model snapshot does not match its receipt")
    return snapshot


def _ensure_page_disk_budget(
    request_path: Path,
    request: dict,
    graph: dict,
) -> None:
    paths = {request_path}
    for record in request["evidence"].values():
        paths.add((request_path.parent / Path(*record["path"].split("/"))).resolve())
    for node in graph["nodes"]:
        paths.add((request_path.parent / Path(*node["mask"].split("/"))).resolve())
    current_round_bytes = sum(path.lstat().st_size for path in paths)
    remaining_rounds = MAX_REPAIR_ROUNDS - request["repair_round"] + 1
    required_bytes = max(
        current_round_bytes * 2 * remaining_rounds,
        256 * 1024 * 1024,
    )
    free_bytes = shutil.disk_usage(request_path.parents[2]).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Local Agent page repair disk budget is insufficient: "
            f"requires {required_bytes} bytes, available {free_bytes} bytes"
        )


def _worker_environment(resource_policy: dict | None) -> dict[str, str]:
    from image2editable.resources import safe_default_policy, validate_resource_policy

    policy = validate_resource_policy(
        safe_default_policy() if resource_policy is None else resource_policy
    )
    environment = os.environ.copy()
    thread_budget = str(policy["cpu_threads"])
    for name in _THREAD_ENVIRONMENT:
        environment[name] = thread_budget
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return environment


def _invoke_worker(
    command: list[str],
    *,
    environment: dict[str, str],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )


def _record_local_agent_performance(
    performance_trace,
    started: float | None,
    request_path: Path,
    request: dict,
    *,
    status: str,
) -> None:
    if performance_trace is None or started is None:
        return
    try:
        image_paths = [
            request_path.parent / Path(*request["evidence"][name]["path"].split("/"))
            for name in request["review_evidence"]
            if name.endswith(".png")
        ]
        performance_trace.event(
            "local_agent",
            image_count=len(image_paths),
            total_bytes=sum(path.stat().st_size for path in image_paths),
            duration_ms=round((time.perf_counter() - started) * 1000),
            status=status,
        )
    except Exception:
        _LOGGER.warning("Performance trace recording failed", exc_info=True)


def _read_plan(path: Path) -> dict:
    status = path.lstat()
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or status.st_size > _PLAN_LIMIT
    ):
        raise RuntimeError("Local Agent plan output is not a bounded regular file")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("Local Agent plan output must be a JSON object")
    return plan


def _diagnostic_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-_DIAGNOSTIC_TEXT_LIMIT:]


def _write_diagnostic(
    request_path: Path,
    request: dict,
    details: dict[str, object],
    *,
    ensure_directory,
    write_exclusive,
) -> str | None:
    directory = request_path.parents[2] / "local-agent-diagnostics"
    reconstruction = request_path.parents[2]
    try:
        ensure_directory(directory, reconstruction)
        target = directory / f"round-{request['repair_round']:02d}.json"
        document = {
            "schema_version": 1,
            "page_id": request["page_id"],
            "provider": "local",
            "repair_round": request["repair_round"],
            **details,
        }
        payload = (
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        write_exclusive(target, payload, reconstruction)
    except Exception as error:
        return str(error)
    return None
