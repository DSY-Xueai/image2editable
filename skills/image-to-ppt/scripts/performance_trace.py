"""Content-free performance trace records for conversion workers."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import platform
import time


_SCHEMA_VERSION = 1
_FIELDS = frozenset(
    {
        "stage",
        "page_id",
        "model",
        "operation_count",
        "duration_ms",
        "status",
        "image_count",
        "total_bytes",
        "platform",
        "device",
        "cuda_available",
        "mps_available",
    }
)


class PerformanceTrace:
    def __init__(self, path: str | Path, *, clock=time.perf_counter) -> None:
        self.path = Path(path)
        self.clock = clock

    def event(self, event: str, **fields) -> None:
        unknown = set(fields) - _FIELDS
        if unknown:
            raise ValueError(f"unknown performance field: {sorted(unknown)[0]}")
        document = {"schema_version": _SCHEMA_VERSION, "event": event, **fields}
        if not self.path.parent.is_dir():
            raise FileNotFoundError(self.path.parent)
        with self.path.open("a", encoding="utf-8", newline="\n") as target:
            target.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
            target.write("\n")
            target.flush()

    def span(self, stage: str, **fields):
        self._check_fields(fields)
        @contextmanager
        def timer():
            started = self.clock()
            try:
                yield
            finally:
                self.event(
                    "span",
                    stage=stage,
                    **fields,
                    duration_ms=round((self.clock() - started) * 1000),
                )

        return timer()

    @staticmethod
    def _check_fields(fields: dict) -> None:
        unknown = set(fields) - _FIELDS
        if unknown:
            raise ValueError(f"unknown performance field: {sorted(unknown)[0]}")


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
