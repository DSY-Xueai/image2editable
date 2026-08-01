from __future__ import annotations

import logging
import types
import ctypes
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pptx import Presentation
from pypdf import PdfWriter

from image2editable import legacy, resources, runtime
from image2editable.resources import (
    _THREAD_ENVIRONMENT,
    apply_resource_policy,
    safe_default_policy,
    validate_resource_policy,
)
from image2editable.store import RunStore


SAFE_POLICY = {
    "name": "safe-default",
    "cpu_threads": 8,
    "heavy_page_concurrency": 1,
    "sam_points_per_batch": 1,
}


@pytest.mark.parametrize(
    ("cpu_count", "cpu_threads"),
    [(32, 8), (8, 4), (1, 1)],
)
def test_safe_default_policy_uses_bounded_half_cpu_budget(
    cpu_count: int, cpu_threads: int
) -> None:
    assert safe_default_policy(cpu_count=cpu_count) == {
        "name": "safe-default",
        "cpu_threads": cpu_threads,
        "heavy_page_concurrency": 1,
        "sam_points_per_batch": 1,
    }


def test_thread_environment_names_are_fixed() -> None:
    assert _THREAD_ENVIRONMENT == (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "FLAGS_paddle_num_threads",
    )


def test_apply_resource_policy_preserves_explicit_thread_environment() -> None:
    environment = {
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "2",
    }
    priority_calls = 0

    def set_priority() -> None:
        nonlocal priority_calls
        priority_calls += 1

    warnings = apply_resource_policy(
        SAFE_POLICY,
        environ=environment,
        priority_setter=set_priority,
    )

    assert warnings == []
    assert environment == {
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "FLAGS_paddle_num_threads": "8",
    }
    assert priority_calls == 1


def test_apply_resource_policy_warns_and_continues_when_priority_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def denied() -> None:
        raise OSError("denied")

    with caplog.at_level(logging.WARNING, logger="image2editable.resources"):
        warnings = apply_resource_policy(
            SAFE_POLICY,
            environ={},
            priority_setter=denied,
        )

    assert warnings == ["Could not lower process priority: denied"]
    assert warnings[0] in caplog.messages


def test_windows_priority_uses_pointer_sized_process_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFunction:
        def __init__(self, result: object) -> None:
            self.result = result
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args: object) -> object:
            self.calls.append(args)
            return self.result

    get_current_process = FakeFunction(2**40)
    set_priority_class = FakeFunction(1)
    kernel32 = types.SimpleNamespace(
        GetCurrentProcess=get_current_process,
        SetPriorityClass=set_priority_class,
    )
    monkeypatch.setattr(resources.os, "name", "nt")
    monkeypatch.setattr(
        resources.ctypes,
        "windll",
        types.SimpleNamespace(kernel32=kernel32),
    )

    resources._lower_process_priority()

    assert get_current_process.restype is ctypes.c_void_p
    assert set_priority_class.argtypes == [ctypes.c_void_p, ctypes.c_uint32]
    assert set_priority_class.calls == [(2**40, 0x00004000)]


def test_posix_priority_lowering_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_priority = 0
    set_calls: list[int] = []

    def nice(increment: int) -> None:
        nonlocal current_priority
        current_priority += increment

    def getpriority(which: int, who: int) -> int:
        return current_priority

    def setpriority(which: int, who: int, priority: int) -> None:
        nonlocal current_priority
        current_priority = priority
        set_calls.append(priority)

    monkeypatch.setattr(resources.os, "name", "posix")
    monkeypatch.setattr(resources.os, "nice", nice, raising=False)
    monkeypatch.setattr(resources.os, "getpriority", getpriority, raising=False)
    monkeypatch.setattr(resources.os, "setpriority", setpriority, raising=False)
    monkeypatch.setattr(resources.os, "PRIO_PROCESS", 0, raising=False)

    resources._lower_process_priority()
    resources._lower_process_priority()

    assert current_priority == 5
    assert set_calls == [5]


def test_posix_priority_does_not_raise_an_already_lower_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_priority = 10
    set_calls: list[int] = []

    def nice(increment: int) -> None:
        nonlocal current_priority
        current_priority += increment

    monkeypatch.setattr(resources.os, "name", "posix")
    monkeypatch.setattr(resources.os, "nice", nice, raising=False)
    monkeypatch.setattr(
        resources.os,
        "getpriority",
        lambda which, who: current_priority,
        raising=False,
    )
    monkeypatch.setattr(
        resources.os,
        "setpriority",
        lambda which, who, priority: set_calls.append(priority),
        raising=False,
    )
    monkeypatch.setattr(resources.os, "PRIO_PROCESS", 0, raising=False)

    resources._lower_process_priority()

    assert current_priority == 10
    assert set_calls == []


@pytest.mark.parametrize(
    "policy",
    [
        None,
        [],
        {},
        {key: value for key, value in SAFE_POLICY.items() if key != "name"},
        {**SAFE_POLICY, "extra": True},
        {**SAFE_POLICY, "name": "custom"},
        {**SAFE_POLICY, "cpu_threads": True},
        {**SAFE_POLICY, "cpu_threads": 0},
        {**SAFE_POLICY, "cpu_threads": 9},
        {**SAFE_POLICY, "heavy_page_concurrency": 2},
        {**SAFE_POLICY, "sam_points_per_batch": 3},
    ],
)
def test_validate_resource_policy_rejects_noncanonical_values(policy: object) -> None:
    with pytest.raises(ValueError, match="resource policy"):
        validate_resource_policy(policy)


def test_validate_resource_policy_returns_a_copy() -> None:
    policy = dict(SAFE_POLICY)

    validated = validate_resource_policy(policy)

    assert validated == policy
    assert validated is not policy


def test_validate_resource_policy_accepts_legacy_sam_batch_four() -> None:
    legacy_policy = {
        **SAFE_POLICY,
        "sam_points_per_batch": 4,
    }

    assert validate_resource_policy(legacy_policy) == legacy_policy


def _image(path: Path) -> None:
    Image.new("RGB", (12, 8), (1, 2, 3)).save(path)


def _pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=36)
    with path.open("wb") as stream:
        writer.write(stream)


def _mock_legacy_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    def initialize(store: RunStore, page_id: str, **kwargs: object) -> dict:
        reconstruction = store.root / "pages" / page_id / "reconstruction"
        reconstruction.mkdir(parents=True, exist_ok=True)
        (reconstruction / "component_state.json").write_text("{}", encoding="utf-8")
        return {"status": "initialized", "page_id": page_id}

    monkeypatch.setattr(runtime, "initialize_legacy_page", initialize)
    monkeypatch.setattr(
        runtime, "advance_legacy_page",
        lambda store, page_id, **kwargs: {
            "status": "ready_for_assembly", "page_id": page_id,
        },
    )
    monkeypatch.setattr(runtime, "assemble_legacy_results", lambda store: {})


@pytest.mark.parametrize("suffix", [".png", ".pdf"])
def test_image_and_pdf_completion_summaries_include_manifest_resource_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    source = tmp_path / f"source{suffix}"
    (_image if suffix == ".png" else _pdf)(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    policy = store.read_json("job_manifest.json")["options"]["resource_policy"]
    monkeypatch.setattr(runtime, "apply_resource_policy", lambda value: [])
    _mock_legacy_completion(monkeypatch)

    summary = runtime.run_job(run_dir)

    assert summary["resource_policy"] == policy


@pytest.mark.parametrize("suffix", [".png", ".pdf"])
@pytest.mark.parametrize("mutation", ["missing", "invalid", "mismatch"])
def test_completed_image_and_pdf_reject_tampered_summary_resource_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    mutation: str,
) -> None:
    source = tmp_path / f"source{suffix}"
    (_image if suffix == ".png" else _pdf)(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    monkeypatch.setattr(runtime, "apply_resource_policy", lambda value: [])
    _mock_legacy_completion(monkeypatch)
    runtime.run_job(run_dir)
    manifest_policy = store.read_json("job_manifest.json")["options"][
        "resource_policy"
    ]
    summary = store.read_json("run_summary.json")
    if mutation == "missing":
        summary.pop("resource_policy")
    elif mutation == "invalid":
        summary["resource_policy"] = {
            **manifest_policy,
            "heavy_page_concurrency": True,
        }
    else:
        summary["resource_policy"] = {
            **manifest_policy,
            "cpu_threads": (
                7 if manifest_policy["cpu_threads"] == 8 else 8
            ),
        }
    store.write_json("run_summary.json", summary)
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    before_summary = store.read_json("run_summary.json")

    with pytest.raises(RuntimeError, match="completion summary resource policy"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages
    assert store.read_json("run_summary.json") == before_summary


def test_runtime_pptx_summary_adds_policy_without_changing_direct_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_source = tmp_path / "direct.pptx"
    Presentation().save(direct_source)
    direct_run = runtime.prepare_job(direct_source, run_dir=tmp_path / "direct-run")
    monkeypatch.setattr(runtime, "apply_resource_policy", lambda value: [])

    direct = runtime.execute_pptx_preserve(RunStore.open(direct_run))

    assert "resource_policy" not in direct

    source = tmp_path / "runtime.pptx"
    Presentation().save(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "runtime-run")
    store = RunStore.open(run_dir)
    policy = store.read_json("job_manifest.json")["options"]["resource_policy"]

    summary = runtime.run_job(run_dir)

    assert summary["resource_policy"] == policy


def test_runtime_applies_policy_before_importing_heavy_legacy_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    policy = RunStore.open(run_dir).read_json("job_manifest.json")["options"][
        "resource_policy"
    ]
    events: list[tuple[str, object]] = []

    def apply(value: dict[str, object]) -> list[str]:
        events.append(("apply", value))
        return []

    def import_module(name: str) -> Any:
        events.append(("import", name))
        return types.SimpleNamespace(convert_variants=lambda *args, **kwargs: {})

    monkeypatch.setattr(runtime, "apply_resource_policy", apply)
    monkeypatch.setattr(legacy.importlib, "import_module", import_module)
    _mock_legacy_completion(monkeypatch)
    original_initialize = runtime.initialize_legacy_page

    def initialize(*args: object, **kwargs: object) -> dict:
        import_module("image_to_ppt")
        return original_initialize(*args, **kwargs)

    monkeypatch.setattr(runtime, "initialize_legacy_page", initialize)

    runtime.run_job(run_dir)

    assert events[:2] == [
        ("apply", policy),
        ("import", "image_to_ppt"),
    ]


@pytest.mark.parametrize("mutation", ["missing", "invalid"])
def test_runtime_rejects_missing_or_invalid_policy_before_heavy_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    if mutation == "missing":
        manifest["options"].pop("resource_policy")
    else:
        manifest["options"]["resource_policy"]["cpu_threads"] = True
    store.write_json("job_manifest.json", manifest)
    imported = False

    def unexpected_import(name: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("heavy module imported before policy validation")

    monkeypatch.setattr(legacy.importlib, "import_module", unexpected_import)

    with pytest.raises(ValueError, match="resource policy"):
        runtime.run_job(run_dir)

    assert imported is False
    assert store.read_json("run_state.json")["status"] == "prepared"


def test_local_preflight_rejects_low_resources_before_hashing_model_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image2editable import models

    events = []

    def recommend(store):
        events.append("recommend")
        assert store.root == tmp_path
        return {"compatible": False, "reason": "显存不足"}

    def unexpected_status():
        events.append("status")
        raise AssertionError("model snapshot must not be hashed after failed preflight")

    monkeypatch.setattr(runtime, "_local_hardware_recommendation", recommend)
    monkeypatch.setattr(models, "model_status", unexpected_status)

    with pytest.raises(RuntimeError, match="preflight.*显存不足"):
        runtime._local_model_receipt(types.SimpleNamespace(root=tmp_path))

    assert events == ["recommend"]


def test_local_hardware_preflight_runs_in_disposable_models_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "main",
        "compatible": True,
        "reason": "supported",
    }
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(expected).encode("utf-8"),
            b"",
        )

    monkeypatch.setattr(runtime.subprocess, "run", run)

    result = runtime._local_hardware_recommendation(
        types.SimpleNamespace(root=tmp_path)
    )

    assert result == expected
    command, kwargs = calls[0]
    assert command[1:] == [
        "-m",
        "image2editable",
        "models",
        "recommend",
        "--json",
    ]
    assert kwargs["timeout"] == 60
    assert kwargs["check"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["env"]["PYTHONUTF8"] == "1"
    assert kwargs["env"]["IMAGE2EDITABLE_MODEL_CACHE"] == str(tmp_path.resolve())


def test_local_model_provenance_is_frozen_for_the_run(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(
        source,
        run_dir=tmp_path / "run",
        agent_provider="local",
    )
    store = RunStore.open(run_dir)
    receipt = {
        "schema_version": 1,
        "model_id": "test/model",
        "requested_revision": "main",
        "resolved_revision": "a" * 40,
        "stability": "experimental",
        "snapshot_path": str((tmp_path / "snapshot").resolve()),
        "files": [
            {"path": "config.json", "size": 2, "sha256": "b" * 64}
        ],
        "installed_at": "first install",
    }

    assert runtime._bind_local_model_receipt(store, receipt) is receipt
    assert runtime._bind_local_model_receipt(
        store,
        {**receipt, "installed_at": "same snapshot installed again"},
    )["resolved_revision"] == "a" * 40
    with pytest.raises(RuntimeError, match="different model snapshot"):
        runtime._bind_local_model_receipt(
            store,
            {**receipt, "resolved_revision": "c" * 40},
        )
