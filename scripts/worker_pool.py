"""Task-scoped, sequential workers for heavyweight model processes."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from typing import Any, Callable, Protocol


_DEFAULT_REQUEST_TIMEOUT_SECONDS = 5 * 60


class WorkerPoolError(RuntimeError):
    """Base error for task worker lifecycle failures."""


class WorkerCancelled(WorkerPoolError):
    """The caller cancelled before its request reached a worker."""


class WorkerDeadlineExceeded(WorkerPoolError):
    """The worker did not complete before the request deadline."""


class WorkerPoolClosed(WorkerPoolError):
    """The task worker pool is no longer available."""


class WorkerQueueFull(WorkerPoolError):
    """The task worker has reached its bounded request capacity."""


class WorkerRemoteError(WorkerPoolError):
    """A worker reported a request-local error and remains usable."""


class _Worker(Protocol):
    def request(self, envelope: dict[str, Any], *, deadline: float | None) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...

    def terminate(self) -> None:
        ...


def _deadline_remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise WorkerDeadlineExceeded("worker request deadline exceeded")
    return remaining


class JsonLineWorker:
    """One ordered JSON-line subprocess request channel."""

    def __init__(self, command: list[str]) -> None:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._lock = threading.Lock()

    def request(self, envelope: dict[str, Any], *, deadline: float | None) -> dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None:
                raise WorkerPoolError("worker process exited before its request")
            if self._process.stdin is None or self._process.stdout is None:
                raise WorkerPoolError("worker process streams are unavailable")
            self._process.stdin.write(json.dumps(envelope, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
            received: queue.Queue[str] = queue.Queue(maxsize=1)

            def read_response() -> None:
                line = self._process.stdout.readline()
                received.put(line)

            reader = threading.Thread(target=read_response, daemon=True)
            reader.start()
            reader.join(_deadline_remaining(deadline))
            if reader.is_alive():
                raise WorkerDeadlineExceeded("worker request deadline exceeded")
            try:
                line = received.get_nowait()
            except queue.Empty as exc:
                raise WorkerPoolError("worker did not return a response") from exc
            if not line:
                raise WorkerPoolError("worker process exited without a response")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkerPoolError("worker returned invalid JSON") from exc
            if not isinstance(response, dict):
                raise WorkerPoolError("worker returned an invalid response")
            return response

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.write('{"control":"close"}\n')
                self._process.stdin.flush()
                self._process.stdin.close()
            self._process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.terminate()

    def terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)


class TaskWorkerPool:
    """Reuse one task worker with a bounded active-and-waiting request count."""

    def __init__(
        self,
        worker_factory: Callable[[], _Worker],
        *,
        queue_limit: int = 1,
        worker_name: str,
    ) -> None:
        if queue_limit < 1:
            raise ValueError("worker queue_limit must be positive")
        if not worker_name:
            raise ValueError("worker_name is required")
        self._worker_factory = worker_factory
        self._worker_name = worker_name
        self._slots = threading.BoundedSemaphore(queue_limit)
        self._lock = threading.Lock()
        self._worker: _Worker | None = None
        self._closed = False
        self._next_request = 1

    def request(
        self,
        payload: dict[str, Any],
        *,
        deadline: float | None = None,
        cancel: threading.Event | None = None,
        performance_trace=None,
        page_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("worker payload must be a mapping")
        if deadline is None:
            deadline = time.monotonic() + _DEFAULT_REQUEST_TIMEOUT_SECONDS
        queued_at = time.monotonic()
        slot_acquired = False
        lock_acquired = False
        try:
            self._acquire_slot(deadline, cancel)
            slot_acquired = True
            queue_wait_ms = round((time.monotonic() - queued_at) * 1000)
            self._trace(
                performance_trace,
                "span",
                stage="worker_queue",
                model=self._worker_name,
                page_id=page_id,
                duration_ms=queue_wait_ms,
            )
            self._acquire_worker_lock(deadline, cancel)
            lock_acquired = True
            if cancel is not None and cancel.is_set():
                raise WorkerCancelled("worker request was cancelled")
            if self._closed:
                raise WorkerPoolClosed("worker pool is closed")
            request_id = f"{self._worker_name}-{self._next_request:08d}"
            self._next_request += 1
            if self._worker is None:
                model_load_started = time.monotonic()
                self._trace(
                    performance_trace,
                    "model_load_start",
                    model=self._worker_name,
                    page_id=page_id,
                )
                try:
                    self._worker = self._worker_factory()
                except Exception:
                    self._trace(
                        performance_trace,
                        "model_load_finish",
                        model=self._worker_name,
                        page_id=page_id,
                        duration_ms=round(
                            (time.monotonic() - model_load_started) * 1000
                        ),
                        status="error",
                    )
                    raise
                self._trace(
                    performance_trace,
                    "model_load_finish",
                    model=self._worker_name,
                    page_id=page_id,
                    duration_ms=round(
                        (time.monotonic() - model_load_started) * 1000
                    ),
                    status="success",
                )
                self._trace(
                    performance_trace,
                    "worker_start",
                    stage="task_worker",
                    model=self._worker_name,
                    page_id=page_id,
                    operation_count=1,
                )
            else:
                self._trace(
                    performance_trace,
                    "worker_start",
                    stage="worker_reuse",
                    model=self._worker_name,
                    page_id=page_id,
                    operation_count=1,
                )
            started = time.monotonic()
            try:
                response = self._request_worker(
                    self._worker,
                    {"request_id": request_id, "payload": payload},
                    deadline=deadline,
                    cancel=cancel,
                )
            except WorkerCancelled:
                self._discard_worker()
                self._trace_finish(performance_trace, page_id, started, "error")
                raise
            except WorkerDeadlineExceeded:
                self._discard_worker()
                self._trace_finish(performance_trace, page_id, started, "error")
                raise
            except Exception as exc:
                self._discard_worker()
                self._trace_finish(performance_trace, page_id, started, "error")
                raise WorkerPoolError("worker request failed") from exc
            try:
                result = self._decode_response(response, request_id)
            except WorkerRemoteError:
                self._trace_finish(performance_trace, page_id, started, "error")
                raise
            except WorkerPoolError:
                self._discard_worker()
                self._trace_finish(performance_trace, page_id, started, "error")
                raise
            self._trace_finish(performance_trace, page_id, started, "success")
            return result
        except WorkerCancelled:
            self._trace_cancel(performance_trace, page_id)
            raise
        finally:
            if lock_acquired:
                self._lock.release()
            if slot_acquired:
                self._slots.release()

    def close(self, *, performance_trace=None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker
            self._worker = None
            if worker is not None:
                try:
                    worker.close()
                except Exception:
                    worker.terminate()
            self._trace(
                performance_trace,
                "worker",
                stage="worker_close",
                model=self._worker_name,
                operation_count=0,
                duration_ms=0,
                status="success",
            )

    def terminate(self, *, performance_trace=None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker
            self._worker = None
            if worker is not None:
                worker.terminate()
            self._trace(
                performance_trace,
                "worker",
                stage="worker_cancel",
                model=self._worker_name,
                operation_count=0,
                duration_ms=0,
                status="error",
            )

    def _acquire_slot(
        self,
        deadline: float | None,
        cancel: threading.Event | None,
    ) -> None:
        if cancel is not None and cancel.is_set():
            raise WorkerCancelled("worker request was cancelled")
        _deadline_remaining(deadline)
        if not self._slots.acquire(blocking=False):
            raise WorkerQueueFull("worker request queue is full")

    def _acquire_worker_lock(
        self,
        deadline: float | None,
        cancel: threading.Event | None,
    ) -> None:
        while True:
            if cancel is not None and cancel.is_set():
                raise WorkerCancelled("worker request was cancelled")
            remaining = _deadline_remaining(deadline)
            timeout = 0.05 if remaining is None else min(remaining, 0.05)
            if self._lock.acquire(timeout=timeout):
                return

    @staticmethod
    def _request_worker(
        worker: _Worker,
        envelope: dict[str, Any],
        *,
        deadline: float | None,
        cancel: threading.Event | None,
    ) -> dict[str, Any]:
        completed = threading.Event()
        outcome: dict[str, Any] = {}

        def execute() -> None:
            try:
                outcome["response"] = worker.request(envelope, deadline=deadline)
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                completed.set()

        threading.Thread(target=execute, daemon=True).start()
        while not completed.wait(0.05):
            if cancel is not None and cancel.is_set():
                raise WorkerCancelled("worker request was cancelled")
            _deadline_remaining(deadline)
        if cancel is not None and cancel.is_set():
            raise WorkerCancelled("worker request was cancelled")
        error = outcome.get("error")
        if error is not None:
            raise error
        response = outcome.get("response")
        if not isinstance(response, dict):
            raise WorkerPoolError("worker returned an invalid response")
        return response

    def _discard_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.terminate()

    @staticmethod
    def _decode_response(response: dict[str, Any], request_id: str) -> dict[str, Any]:
        if response.get("request_id") != request_id:
            raise WorkerPoolError("worker response request id mismatch")
        if "error" in response:
            error = response["error"]
            if not isinstance(error, dict):
                raise WorkerPoolError("worker returned an invalid error envelope")
            message = error.get("message")
            if not isinstance(message, str) or not message:
                raise WorkerPoolError("worker returned an invalid error envelope")
            raise WorkerRemoteError(message)
        result = response.get("result")
        if not isinstance(result, dict):
            raise WorkerPoolError("worker returned an invalid result envelope")
        return result

    def _trace_cancel(self, performance_trace, page_id: str | None) -> None:
        self._trace(
            performance_trace,
            "worker",
            stage="worker_cancel",
            model=self._worker_name,
            page_id=page_id,
            operation_count=0,
            duration_ms=0,
            status="error",
        )

    def _trace_finish(
        self,
        performance_trace,
        page_id: str | None,
        started: float,
        status: str,
    ) -> None:
        self._trace(
            performance_trace,
            "worker_finish",
            stage="task_worker",
            model=self._worker_name,
            page_id=page_id,
            operation_count=1,
            duration_ms=round((time.monotonic() - started) * 1000),
            status=status,
        )

    @staticmethod
    def _trace(performance_trace, event: str, **fields) -> None:
        if performance_trace is None:
            return
        fields = {name: value for name, value in fields.items() if value is not None}
        try:
            performance_trace.event(event, **fields)
        except Exception:
            pass
