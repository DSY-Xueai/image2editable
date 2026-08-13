from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_performance_trace():
    path = ROOT / "scripts" / "performance_trace.py"
    spec = importlib.util.spec_from_file_location("tested_performance_trace", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clock(values: list[float]):
    return lambda: values.pop(0)


def test_span_writes_allowlisted_utf8_jsonl_event(tmp_path: Path) -> None:
    performance_trace = _load_performance_trace()
    target = tmp_path / "performance.jsonl"

    with performance_trace.PerformanceTrace(target, clock=_clock([10.0, 12.5])).span(
        "inference",
        page_id="page_001",
        model="sam",
        operation_count=2,
    ):
        pass

    event = json.loads(target.read_text(encoding="utf-8"))
    assert event == {
        "schema_version": 1,
        "event": "span",
        "stage": "inference",
        "page_id": "page_001",
        "model": "sam",
        "operation_count": 2,
        "duration_ms": 2500,
    }
    serialized = target.read_text(encoding="utf-8")
    for forbidden in ("source", "path", "ocr", "prompt", "response", "extra"):
        assert forbidden not in serialized.casefold()


def test_event_and_span_reject_unknown_fields(tmp_path: Path) -> None:
    trace = _load_performance_trace().PerformanceTrace(tmp_path / "performance.jsonl")

    with pytest.raises(ValueError, match="unknown performance field"):
        trace.event("worker", source_path="secret.png")
    with pytest.raises(ValueError, match="unknown performance field"):
        trace.span("inference", prompt="secret")


class _Torch:
    def __init__(self, cuda: bool, mps: bool) -> None:
        self.cuda = type("Cuda", (), {"is_available": lambda _: cuda})()
        self.backends = type(
            "Backends",
            (), {"mps": type("Mps", (), {"is_available": lambda _: mps})()},
        )()


@pytest.mark.parametrize("platform_name", ["Windows", "Linux", "Darwin"])
def test_device_summary_keeps_cpu_when_mps_is_available(platform_name: str) -> None:
    result = _load_performance_trace().device_summary(
        _Torch(cuda=False, mps=True), platform_name=platform_name
    )

    assert result == {
        "platform": platform_name,
        "device": "cpu",
        "cuda_available": False,
        "mps_available": True,
    }


def test_device_summary_selects_cuda_and_handles_probe_failure() -> None:
    performance_trace = _load_performance_trace()

    assert performance_trace.device_summary(_Torch(cuda=True, mps=True)) ["device"] == "cuda"

    class BrokenTorch:
        cuda = type("Cuda", (), {"is_available": lambda _: (_ for _ in ()).throw(RuntimeError())})()
        backends = object()

    assert performance_trace.device_summary(BrokenTorch())["device"] == "unknown"


def test_product_and_skill_performance_scripts_are_identical() -> None:
    assert (ROOT / "scripts" / "performance_trace.py").read_bytes() == (
        ROOT / "skills" / "image-to-ppt" / "scripts" / "performance_trace.py"
    ).read_bytes()
    assert (ROOT / "scripts" / "worker_resources.py").read_bytes() == (
        ROOT / "skills" / "image-to-ppt" / "scripts" / "worker_resources.py"
    ).read_bytes()
