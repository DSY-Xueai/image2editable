"""Resource handling for blocking heavyweight worker processes."""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import subprocess
import time


_LOGGER = logging.getLogger(__name__)


def _empty_current_process_working_set_windows() -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    empty_working_set = psapi.EmptyWorkingSet
    empty_working_set.argtypes = [ctypes.c_void_p]
    empty_working_set.restype = ctypes.c_int
    if not empty_working_set(get_current_process()):
        raise OSError(ctypes.get_last_error(), "EmptyWorkingSet failed")


def trim_parent_working_set_before_worker() -> None:
    gc.collect()
    if os.name != "nt":
        return
    try:
        _empty_current_process_working_set_windows()
    except OSError:
        return


def run_isolated_worker(
    command: list[str],
    *,
    performance_trace=None,
    stage: str | None = None,
    model: str | None = None,
    operation_count: int | None = None,
    **kwargs,
):
    trim_parent_working_set_before_worker()
    env = os.environ.copy()
    env.update(kwargs.pop("env", {}))
    env["PYTHONUTF8"] = "1"
    kwargs["env"] = env
    if kwargs.get("text") and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    if kwargs.get("text") and "errors" not in kwargs:
        kwargs["errors"] = "replace"
    if performance_trace is None:
        return subprocess.run(command, **kwargs)
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, **kwargs)
    except BaseException:
        _record_worker_performance(
            performance_trace,
            started,
            stage=stage,
            model=model,
            operation_count=operation_count,
            status="error",
        )
        raise
    _record_worker_performance(
        performance_trace,
        started,
        stage=stage,
        model=model,
        operation_count=operation_count,
        status="success" if completed.returncode == 0 else "failed",
    )
    return completed


def _record_worker_performance(
    performance_trace,
    started: float,
    *,
    stage: str | None,
    model: str | None,
    operation_count: int | None,
    status: str,
) -> None:
    fields = {
        "duration_ms": round((time.perf_counter() - started) * 1000),
        "status": status,
    }
    if stage is not None:
        fields["stage"] = stage
    if model is not None:
        fields["model"] = model
    if operation_count is not None:
        fields["operation_count"] = operation_count
    try:
        performance_trace.event("worker", **fields)
    except Exception:
        _LOGGER.warning("Performance trace recording failed", exc_info=True)
