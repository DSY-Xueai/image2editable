"""Content-free performance trace records for conversion workers."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import platform
import re
import time


_LOGGER = logging.getLogger(__name__)
_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PLATFORM = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
_EVENT_FIELDS = {
    "span": ({"stage", "page_id", "model", "operation_count", "duration_ms"}, {"stage", "duration_ms"}),
    "worker": ({"stage", "page_id", "model", "operation_count", "duration_ms", "status"}, {"duration_ms", "status"}),
    "local_agent": ({"image_count", "total_bytes", "duration_ms", "status"}, {"image_count", "total_bytes", "duration_ms", "status"}),
    "worker_start": ({"stage", "page_id", "model", "operation_count"}, {"stage"}),
    "worker_finish": ({"stage", "page_id", "model", "operation_count", "duration_ms", "status"}, {"stage", "duration_ms", "status"}),
    "model_load_start": ({"page_id", "model"}, {"model"}),
    "model_load_finish": ({"page_id", "model", "duration_ms", "status"}, {"model", "duration_ms", "status"}),
    "inference_start": ({"stage", "page_id", "model", "operation_count"}, {"stage"}),
    "inference_finish": ({"stage", "page_id", "model", "operation_count", "duration_ms", "status"}, {"stage", "duration_ms", "status"}),
    "agent_request_published": ({"page_id", "operation_count"}, {"page_id"}),
    "agent_plan_recorded": ({"page_id", "operation_count", "duration_ms", "status"}, {"page_id", "duration_ms", "status"}),
    "device_summary": ({"platform", "device", "cuda_available", "mps_available"}, {"platform", "device", "cuda_available", "mps_available"}),
}


class PerformanceTrace:
    def __init__(self, path: str | Path, *, clock=time.perf_counter) -> None:
        self.path = Path(path)
        self.clock = clock

    def event(self, event: str, **fields) -> None:
        _validate_event(event, fields)
        document = {"schema_version": _SCHEMA_VERSION, "event": event, **fields}
        if not self.path.parent.is_dir():
            raise FileNotFoundError(self.path.parent)
        with self.path.open("a", encoding="utf-8", newline="\n") as target:
            target.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
            target.write("\n")
            target.flush()

    def span(self, stage: str, **fields):
        if "duration_ms" in fields:
            raise ValueError("span duration is recorded internally")
        _validate_event("span", {"stage": stage, **fields, "duration_ms": 0})

        @contextmanager
        def timer():
            started = self.clock()
            try:
                yield
            finally:
                try:
                    self.event(
                        "span",
                        stage=stage,
                        **fields,
                        duration_ms=round((self.clock() - started) * 1000),
                    )
                except Exception:
                    _LOGGER.warning("Performance trace recording failed", exc_info=True)

        return timer()


def _validate_event(event: object, fields: dict) -> None:
    if not isinstance(event, str) or event not in _EVENT_FIELDS:
        raise ValueError("unknown performance event")
    allowed, required = _EVENT_FIELDS[event]
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"unknown performance field: {sorted(unknown)[0]}")
    missing = required - set(fields)
    if missing:
        raise ValueError(f"missing performance field: {sorted(missing)[0]}")
    for name, value in fields.items():
        _validate_field(name, value)


def _validate_field(name: str, value: object) -> None:
    if isinstance(value, (list, dict, tuple, set)):
        raise ValueError(f"invalid performance field: {name}")
    if name in {"page_id", "stage", "model"}:
        if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"invalid performance field: {name}")
    elif name == "platform":
        if not isinstance(value, str) or not _PLATFORM.fullmatch(value):
            raise ValueError("invalid performance field: platform")
    elif name == "status":
        if value not in {"success", "failed", "error"}:
            raise ValueError("invalid performance field: status")
    elif name == "device":
        if value not in {"cuda", "cpu", "unknown"}:
            raise ValueError("invalid performance field: device")
    elif name in {"operation_count", "duration_ms", "image_count", "total_bytes"}:
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid performance field: {name}")
    elif name in {"cuda_available", "mps_available"} and type(value) is not bool:
        raise ValueError(f"invalid performance field: {name}")


def device_summary(torch_module=None, *, platform_name=platform.system()) -> dict:
    summary = {
        "platform": platform_name,
        "device": "unknown",
        "cuda_available": False,
        "mps_available": False,
    }
    try:
        if torch_module is None:
            import torch as torch_module
        summary["cuda_available"] = bool(torch_module.cuda.is_available())
        summary["mps_available"] = bool(torch_module.backends.mps.is_available())
    except Exception:
        return summary
    summary["device"] = "cuda" if summary["cuda_available"] else "cpu"
    return summary
