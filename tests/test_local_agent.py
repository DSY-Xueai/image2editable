from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import pytest

from image2editable import local_agent, local_agent_worker
from image2editable.component_repair import (
    EVIDENCE_NAMES,
    build_component_agent_request,
    load_component_agent_request,
)


ROOT = Path(__file__).resolve().parents[1]


def test_local_prompt_requires_independently_movable_leaf_components() -> None:
    prompt = local_agent_worker.SYSTEM_PROMPT

    assert "semantic relationship does not justify merging" in prompt
    assert "independently moved" in prompt
    assert (
        "same physical entity: duplicate masks, edge fragments, shadows, "
        "or segmentation gaps"
    ) in prompt
    assert "semantic parent is grouping-only and non-rendering" in prompt
    assert "Prefer preserving one complete parent" not in prompt


def test_host_skill_limits_absorb_to_one_physical_entity() -> None:
    text = (ROOT / "skills/image-to-ppt/SKILL.md").read_text(encoding="utf-8")

    assert "同一物理实体" in text
    assert "重复掩码" in text
    assert "碎边" in text
    assert "阴影" in text
    assert "分割缺口" in text
    assert "语义父级只用于分组，不参与最终像素渲染" in text


def _request_path(tmp_path: Path) -> Path:
    reconstruction = tmp_path / "pages" / "page_001" / "reconstruction"
    evidence_root = reconstruction / "evidence-source"
    masks = evidence_root / "masks"
    masks.mkdir(parents=True)
    mask = masks / "component_0001.png"
    Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(mask)
    graph = {
        "nodes": [
            {
                "id": "component_0001",
                "kind": "parent",
                "parent_id": None,
                "state": "pending",
                "mask": "masks/component_0001.png",
                "mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
                "bbox": [0, 0, 4, 4],
                "z_index": 0,
                "text_ids": [],
            }
        ]
    }
    evidence = {}
    for name in EVIDENCE_NAMES:
        path = evidence_root / name
        if name == "component-graph.json":
            path.write_text(json.dumps(graph), encoding="utf-8")
        elif name == "quality-report.json":
            path.write_text('{"violations":[]}', encoding="utf-8")
        else:
            Image.fromarray(np.full((4, 4), 127, dtype=np.uint8)).save(path)
        evidence[name] = path
    return build_component_agent_request(
        {
            "page_id": "page_001",
            "provider": "local",
            "reconstruction_dir": reconstruction,
            "evidence": evidence,
        },
        repair_round=1,
    )


def _receipt(tmp_path: Path) -> dict[str, object]:
    snapshot = tmp_path / "model-cache" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    return {
        "schema_version": 1,
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "requested_revision": "main",
        "resolved_revision": "a" * 40,
        "stability": "experimental",
        "snapshot_path": str(snapshot.resolve()),
        "files": [],
        "installed_at": "now",
    }


def _plan(request_path: Path, *, action: str = "accept") -> dict[str, object]:
    request = load_component_agent_request(request_path)
    return {
        "schema_version": 1,
        "kind": "component_plan",
        "page_id": request["page_id"],
        "provider": "local",
        "repair_round": request["repair_round"],
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [
            {
                "action": action,
                "object_ids": ["component_0001"],
                "parameters": {},
                "confidence": 0.95,
                "evidence": ["complete visible component boundary"],
            }
        ],
    }


def test_local_agent_starts_one_worker_with_offline_bounded_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    observed: dict[str, object] = {}

    def invoke(command, *, environment, timeout_seconds):
        observed["command"] = command
        observed["environment"] = environment
        observed["timeout_seconds"] = timeout_seconds
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_plan(request_path)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    result = local_agent.run_local_agent(
        request_path,
        model_receipt=_receipt(tmp_path),
        resource_policy={
            "name": "safe-default",
            "cpu_threads": 3,
            "heavy_page_concurrency": 1,
            "sam_points_per_batch": 1,
        },
    )

    assert result == _plan(request_path)
    command = observed["command"]
    assert command[:3] == [sys.executable, "-m", "image2editable.local_agent_worker"]
    assert "--model-snapshot" in command
    environment = observed["environment"]
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["OMP_NUM_THREADS"] == "3"
    assert observed["timeout_seconds"] == 600


def test_local_plan_passes_the_same_strict_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)

    def invoke(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(_plan(request_path, action="unknown")), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    with pytest.raises(ValueError, match="component action"):
        local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))


def test_local_worker_nonzero_exit_preserves_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    monkeypatch.setattr(
        local_agent,
        "_invoke_worker",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 7, "worker output", "CUDA out of memory"
        ),
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))

    diagnostic = request_path.parents[2] / "local-agent-diagnostics" / "round-01.json"
    saved = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert saved["status"] == "worker_failed"
    assert saved["returncode"] == 7
    assert "out of memory" in saved["stderr"]


def test_local_worker_process_boundary_releases_after_every_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    live = 0
    peak = 0

    def invoke(command, **kwargs):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_plan(request_path)), encoding="utf-8")
        live -= 1
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))
    local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))

    assert peak == 1
    assert live == 0


def test_local_agent_rejects_insufficient_page_round_disk_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    invoked = False

    def unexpected_invoke(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("worker must not start with insufficient page budget")

    monkeypatch.setattr(local_agent, "_invoke_worker", unexpected_invoke)
    monkeypatch.setattr(
        local_agent.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(free=1),
    )

    with pytest.raises(RuntimeError, match="page repair disk budget"):
        local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))

    assert invoked is False


def test_unsafe_diagnostic_directory_does_not_mask_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    reconstruction = request_path.parents[2]
    outside = tmp_path / "outside"
    outside.mkdir()
    diagnostics = reconstruction / "local-agent-diagnostics"
    try:
        os.symlink(outside, diagnostics, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink is unavailable: {error}")
    monkeypatch.setattr(
        local_agent,
        "_invoke_worker",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 7, "worker output", "CUDA out of memory"
        ),
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))

    assert list(outside.iterdir()) == []


def test_local_modules_do_not_import_model_runtime_at_parent_import() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import image2editable.local_agent; "
                "import image2editable.local_agent_worker; "
                "print(any(name in sys.modules for name in "
                "('torch', 'transformers')))"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_worker_loads_only_the_confirmed_local_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    receipt = _receipt(tmp_path)
    calls: list[tuple[str, object, dict[str, object]]] = []

    class Inputs(dict):
        input_ids = [[1, 2, 3]]

        def to(self, device):
            calls.append(("inputs", str(device), {}))
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor", str(path), kwargs))
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls.append(("prompt", messages, kwargs))
            return Inputs(input_ids=[[1, 2, 3]])

        def batch_decode(self, values, **kwargs):
            return [json.dumps(_plan(request_path))]

    class Model:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", str(path), kwargs))
            return cls()

        def generate(self, **kwargs):
            return [[1, 2, 3, 4]]

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=Processor,
        AutoModelForImageTextToText=Model,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = local_agent_worker.generate_plan(
        request_path,
        Path(receipt["snapshot_path"]),
    )

    assert result == _plan(request_path)
    snapshot = receipt["snapshot_path"]
    assert ("processor", snapshot, {"local_files_only": True}) in calls
    assert (
        "model",
        snapshot,
        {
            "local_files_only": True,
            "device_map": "auto",
            "torch_dtype": "auto",
        },
    ) in calls
    assert set(local_agent_worker.ALLOWED_ACTIONS) == {
        "accept",
        "discard",
        "merge",
        "split",
        "expand",
        "shrink",
        "retry_with_box",
        "retry_with_points",
        "attach_text",
        "collapse_to_parent",
        "rebuild_background",
        "absorb_into_parent",
    }
    assert "untrusted" in local_agent_worker.SYSTEM_PROMPT.casefold()
    assert "JSON" in local_agent_worker.SYSTEM_PROMPT
    for required_rule in (
        'split parameters: {"parts"',
        'expand/shrink parameters: {"margin_ratio"',
        'rebuild_background parameters: {"margin_ratio"',
        'absorb_into_parent parameters: {}',
        'retry_with_box parameters: {"box"',
        'retry_with_points parameters: {"positive"',
        '"negative"',
        "normalized to 0..1",
    ):
        assert required_rule in local_agent_worker.SYSTEM_PROMPT
    prompt_messages = next(value for kind, value, _ in calls if kind == "prompt")
    prompt_text = "\n".join(
        item["text"]
        for item in prompt_messages[1]["content"]
        if item["type"] == "text"
    )
    for evidence_name in local_agent_worker._IMAGE_EVIDENCE:
        assert evidence_name in prompt_text
