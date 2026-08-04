from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import types

import pytest


def _load_worker_resources():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "worker_resources.py"
    assert module_path.is_file(), "shared worker resource helper is missing"
    spec = importlib.util.spec_from_file_location("tested_worker_resources", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeFunction:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


def _fake_ctypes(*, empty_result: int, last_error: int = 0):
    get_current_process = _FakeFunction(123)
    empty_working_set = _FakeFunction(empty_result)
    libraries = {
        "kernel32": types.SimpleNamespace(GetCurrentProcess=get_current_process),
        "psapi": types.SimpleNamespace(EmptyWorkingSet=empty_working_set),
    }
    fake = types.SimpleNamespace(
        c_void_p=object(),
        c_int=object(),
        get_last_error=lambda: last_error,
        WinDLL=lambda name, *, use_last_error: libraries[name],
    )
    return fake, get_current_process, empty_working_set


def test_windows_working_set_call_configures_exact_abi(monkeypatch) -> None:
    worker_resources = _load_worker_resources()
    fake, get_current_process, empty_working_set = _fake_ctypes(empty_result=1)
    monkeypatch.setattr(worker_resources, "ctypes", fake)

    worker_resources._empty_current_process_working_set_windows()

    assert get_current_process.argtypes == []
    assert get_current_process.restype is fake.c_void_p
    assert empty_working_set.argtypes == [fake.c_void_p]
    assert empty_working_set.restype is fake.c_int
    assert get_current_process.calls == [()]
    assert empty_working_set.calls == [(123,)]


def test_windows_working_set_false_result_raises_oserror(monkeypatch) -> None:
    worker_resources = _load_worker_resources()
    fake, _, _ = _fake_ctypes(empty_result=0, last_error=1450)
    monkeypatch.setattr(worker_resources, "ctypes", fake)

    with pytest.raises(OSError) as caught:
        worker_resources._empty_current_process_working_set_windows()

    assert caught.value.errno == 1450
    assert "EmptyWorkingSet failed" in str(caught.value)


def test_non_windows_trim_only_collects_garbage(monkeypatch) -> None:
    worker_resources = _load_worker_resources()
    events = []
    monkeypatch.setattr(worker_resources.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(worker_resources.os, "name", "posix")
    monkeypatch.setattr(
        worker_resources,
        "_empty_current_process_working_set_windows",
        lambda: events.append("windows"),
    )

    worker_resources.trim_parent_working_set_before_worker()

    assert events == ["gc"]


def test_runner_spawns_after_best_effort_windows_oserror(monkeypatch) -> None:
    worker_resources = _load_worker_resources()
    events = []
    expected = subprocess.CompletedProcess(["worker"], 0, "out", "err")
    monkeypatch.setattr(worker_resources.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(worker_resources.os, "name", "nt")
    monkeypatch.setattr(
        worker_resources,
        "_empty_current_process_working_set_windows",
        lambda: (_ for _ in ()).throw(OSError("working set unavailable")),
    )

    def fake_run(command, **kwargs):
        events.append(("spawn", command, kwargs))
        return expected

    monkeypatch.setattr(worker_resources.subprocess, "run", fake_run)

    actual = worker_resources.run_isolated_worker(
        ["worker"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert actual is expected
    assert events[0] == "gc"
    _, command, kwargs = events[1]
    assert command == ["worker"]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["env"]["PYTHONUTF8"] == "1"


def test_runner_preserves_custom_environment_and_forces_utf8(monkeypatch) -> None:
    worker_resources = _load_worker_resources()
    monkeypatch.setattr(worker_resources, "trim_parent_working_set_before_worker", lambda: None)
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(worker_resources.subprocess, "run", fake_run)

    worker_resources.run_isolated_worker(
        ["worker"], env={"CUSTOM_WORKER_SETTING": "kept", "PYTHONUTF8": "0"}
    )

    assert captured["env"]["CUSTOM_WORKER_SETTING"] == "kept"
    assert captured["env"]["PYTHONUTF8"] == "1"


def test_trim_does_not_hide_windows_programming_error(monkeypatch) -> None:
    worker_resources = _load_worker_resources()
    monkeypatch.setattr(worker_resources.gc, "collect", lambda: None)
    monkeypatch.setattr(worker_resources.os, "name", "nt")
    monkeypatch.setattr(
        worker_resources,
        "_empty_current_process_working_set_windows",
        lambda: (_ for _ in ()).throw(TypeError("bad ABI")),
    )
    monkeypatch.setattr(
        worker_resources.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("worker must not spawn"),
    )

    with pytest.raises(TypeError, match="bad ABI"):
        worker_resources.run_isolated_worker(["worker"])


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(7, ["worker"]),
        subprocess.TimeoutExpired(["worker"], 600),
    ],
)
def test_runner_preserves_subprocess_failures(monkeypatch, failure) -> None:
    worker_resources = _load_worker_resources()
    monkeypatch.setattr(worker_resources.gc, "collect", lambda: None)
    monkeypatch.setattr(worker_resources.os, "name", "posix")
    monkeypatch.setattr(
        worker_resources.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(type(failure)) as caught:
        worker_resources.run_isolated_worker(["worker"], check=True, timeout=600)

    assert caught.value is failure
