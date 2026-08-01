from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from pptx import Presentation

from image2editable import models, runtime
from image2editable.models import HardwareProfile


def _catalog() -> dict[str, object]:
    return {
        "catalog_version": 1,
        "models": [
            {
                "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                "revision": "main",
                "stability": "experimental",
                "minimum_vram_gib": 8,
                "minimum_ram_gib": 16,
                "required_free_disk_gib": 8,
                "priority": 100,
            }
        ],
    }


def _compatible_packages() -> dict[str, str | None]:
    return {
        "torch": "2.9.0+cu130",
        "transformers": "4.57.3",
        "accelerate": "1.12.0",
        "huggingface-hub": "0.35.3",
    }


def test_versioned_model_catalog_matches_the_initial_experimental_entry() -> None:
    assert models.load_model_catalog() == _catalog()


def test_recommend_uses_hardware_and_catalog_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unexpected_download(**kwargs: object) -> str:
        raise AssertionError(f"recommendation attempted a download: {kwargs}")

    monkeypatch.setattr(models, "snapshot_download", unexpected_download)

    result = models.recommend_agent_model(
        HardwareProfile(vram_gib=8, ram_gib=16, free_disk_gib=20, cuda=True),
        catalog=_catalog(),
        cache_dir=tmp_path,
        package_versions=_compatible_packages(),
    )

    assert result == {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "main",
        "stability": "experimental",
        "compatible": True,
        "reason": "CUDA、显存、内存、磁盘和本地依赖均满足目录要求",
        "minimum_vram_gib": 8,
        "minimum_ram_gib": 16,
        "required_free_disk_gib": 8,
        "cache_dir": str(tmp_path.resolve()),
        "hardware": {
            "cuda": True,
            "vram_gib": 8,
            "ram_gib": 16,
            "free_disk_gib": 20,
        },
        "dependencies": {
            "accelerate": {
                "installed": "1.12.0",
                "minimum": "1.8.0",
                "compatible": True,
            },
            "huggingface-hub": {
                "installed": "0.35.3",
                "minimum": "0.34.0",
                "compatible": True,
            },
            "torch": {
                "installed": "2.9.0+cu130",
                "minimum": "2.5.0",
                "compatible": True,
            },
            "transformers": {
                "installed": "4.57.3",
                "minimum": "4.57.0",
                "compatible": True,
            },
        },
    }


def test_recommend_explains_incompatible_hardware(tmp_path: Path) -> None:
    result = models.recommend_agent_model(
        HardwareProfile(vram_gib=0, ram_gib=8, free_disk_gib=4, cuda=False),
        catalog=_catalog(),
        cache_dir=tmp_path,
        package_versions=_compatible_packages(),
    )

    assert result["compatible"] is False
    assert result["reason"] == (
        "未检测到 CUDA；显存 0 GiB < 8 GiB；内存 8 GiB < 16 GiB；可用磁盘 4 GiB < 8 GiB"
    )


@pytest.mark.parametrize(
    ("package_versions", "expected_reason"),
    [
        (
            {**_compatible_packages(), "transformers": None},
            "未安装 transformers>=4.57.0",
        ),
        (
            {**_compatible_packages(), "transformers": "4.56.2"},
            "transformers 4.56.2 < 4.57.0",
        ),
    ],
)
def test_recommend_includes_missing_or_old_local_dependencies(
    tmp_path: Path,
    package_versions: dict[str, str | None],
    expected_reason: str,
) -> None:
    result = models.recommend_agent_model(
        HardwareProfile(vram_gib=8, ram_gib=16, free_disk_gib=20, cuda=True),
        catalog=_catalog(),
        cache_dir=tmp_path,
        package_versions=package_versions,
    )

    assert result["compatible"] is False
    assert result["reason"] == expected_reason


def test_cuda_profile_uses_the_largest_available_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_properties(index: int) -> object:
            gib = 4 if index == 0 else 12
            return type("Properties", (), {"total_memory": gib * 1024**3})()

    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        models.importlib,
        "import_module",
        lambda name: type("Torch", (), {"cuda": FakeCuda})(),
    )

    assert models._cuda_profile() == (True, 12.0)


def test_install_stops_before_network_when_disk_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def unexpected_download(**kwargs: object) -> str:
        nonlocal called
        called = True
        return ""

    monkeypatch.setattr(models, "snapshot_download", unexpected_download)

    with pytest.raises(RuntimeError, match="free disk"):
        models.install_agent_model(cache_dir=tmp_path, free_disk_gib=2)

    assert called is False


def test_install_requires_explicit_confirmation_before_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def unexpected_download(**kwargs: object) -> str:
        nonlocal called
        called = True
        return ""

    monkeypatch.setattr(models, "snapshot_download", unexpected_download)

    with pytest.raises(PermissionError, match="explicit confirmation"):
        models.install_agent_model(cache_dir=tmp_path, free_disk_gib=20)

    assert called is False


def test_install_writes_resolved_revision_file_manifest_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_revision = "a" * 40
    snapshot = tmp_path / "hub" / "snapshots" / resolved_revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
    (snapshot / "weights.safetensors").write_bytes(b"weights")
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(models, "snapshot_download", fake_download)

    receipt = models.install_agent_model(
        cache_dir=tmp_path,
        free_disk_gib=20,
        confirmed=True,
    )

    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "main",
            "cache_dir": str(tmp_path.resolve()),
        }
    ]
    assert receipt["model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert receipt["requested_revision"] == "main"
    assert receipt["resolved_revision"] == resolved_revision
    assert receipt["stability"] == "experimental"
    assert receipt["snapshot_path"] == str(snapshot.resolve())
    assert receipt["files"] == [
        {
            "path": "config.json",
            "size": 25,
            "sha256": "c30faf048f789902e4ec71cf25adee61136b7fec1d440cca214f2184a16d874b",
        },
        {
            "path": "weights.safetensors",
            "size": 7,
            "sha256": "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
        },
    ]
    saved = json.loads((tmp_path / "agent-receipt.json").read_text(encoding="utf-8"))
    assert saved == receipt
    assert models.model_status(cache_dir=tmp_path) == {
        "installed": True,
        "valid": True,
        "install_command": "image2editable models install agent",
        "receipt": receipt,
    }


def test_install_downloads_the_exact_model_and_revision_that_were_confirmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    catalog["models"].append(
        {
            "model_id": "example/larger-model",
            "revision": "large-main",
            "stability": "experimental",
            "minimum_vram_gib": 24,
            "minimum_ram_gib": 32,
            "required_free_disk_gib": 20,
            "priority": 200,
        }
    )
    snapshot = tmp_path / "snapshots" / ("e" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("config", encoding="utf-8")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(models, "load_model_catalog", lambda: catalog)

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(models, "snapshot_download", fake_download)

    models.install_agent_model(
        cache_dir=tmp_path,
        free_disk_gib=20,
        confirmed=True,
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        revision="main",
    )

    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "main",
            "cache_dir": str(tmp_path.resolve()),
        }
    ]


def test_model_status_detects_changed_snapshot_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_revision = "b" * 40
    snapshot = tmp_path / "snapshots" / resolved_revision
    snapshot.mkdir(parents=True)
    target = snapshot / "config.json"
    target.write_text("original", encoding="utf-8")
    monkeypatch.setattr(models, "snapshot_download", lambda **kwargs: str(snapshot))
    models.install_agent_model(
        cache_dir=tmp_path,
        free_disk_gib=20,
        confirmed=True,
    )

    target.write_text("changed", encoding="utf-8")

    status = models.model_status(cache_dir=tmp_path)
    assert status["installed"] is True
    assert status["valid"] is False
    assert status["reason"] == "snapshot file checksum mismatch: config.json"


def test_model_status_rejects_receipt_with_wrong_resolved_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("config", encoding="utf-8")
    monkeypatch.setattr(models, "snapshot_download", lambda **kwargs: str(snapshot))
    models.install_agent_model(
        cache_dir=tmp_path,
        free_disk_gib=20,
        confirmed=True,
    )
    receipt_path = tmp_path / "agent-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["resolved_revision"] = "c" * 40
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    status = models.model_status(cache_dir=tmp_path)

    assert status["installed"] is True
    assert status["valid"] is False
    assert status["reason"] == "receipt commit does not match snapshot path"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("schema_version", 2, "receipt schema version is unsupported"),
        ("model_id", "unknown/model", "receipt model is not in the current catalog"),
        (
            "resolved_revision",
            "not-a-sha",
            "receipt resolved revision is not a commit SHA",
        ),
    ],
)
def test_model_status_rejects_invalid_receipt_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    reason: str,
) -> None:
    snapshot = tmp_path / "snapshots" / ("f" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("config", encoding="utf-8")
    monkeypatch.setattr(models, "snapshot_download", lambda **kwargs: str(snapshot))
    models.install_agent_model(
        cache_dir=tmp_path,
        free_disk_gib=20,
        confirmed=True,
    )
    receipt_path = tmp_path / "agent-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    status = models.model_status(cache_dir=tmp_path)

    assert status["valid"] is False
    assert status["reason"] == reason


def test_model_status_rejects_files_added_after_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshots" / ("1" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("config", encoding="utf-8")
    monkeypatch.setattr(models, "snapshot_download", lambda **kwargs: str(snapshot))
    models.install_agent_model(
        cache_dir=tmp_path,
        free_disk_gib=20,
        confirmed=True,
    )

    (snapshot / "unexpected.py").write_text("raise RuntimeError", encoding="utf-8")

    status = models.model_status(cache_dir=tmp_path)
    assert status["valid"] is False
    assert status["reason"] == "snapshot file set does not match receipt"


def test_model_status_accepts_hugging_face_blob_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "repo" / "snapshots" / ("d" * 40)
    blob = tmp_path / "repo" / "blobs" / "model-blob"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"model")
    try:
        (snapshot / "model.safetensors").symlink_to(blob)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    monkeypatch.setattr(models, "snapshot_download", lambda **kwargs: str(snapshot))

    models.install_agent_model(
        cache_dir=tmp_path,
        free_disk_gib=20,
        confirmed=True,
    )

    assert models.model_status(cache_dir=tmp_path)["valid"] is True


def test_missing_model_status_is_read_only(tmp_path: Path) -> None:
    cache = tmp_path / "model-cache"

    assert models.model_status(cache_dir=cache) == {
        "installed": False,
        "valid": False,
        "install_command": "image2editable models install agent",
    }
    assert not cache.exists()


@pytest.mark.parametrize("receipt_text", ["[]", '"value"', "1", "null"])
def test_model_status_rejects_non_object_json_receipt(
    tmp_path: Path,
    receipt_text: str,
) -> None:
    (tmp_path / "agent-receipt.json").write_text(receipt_text, encoding="utf-8")

    assert models.model_status(cache_dir=tmp_path) == {
        "installed": True,
        "valid": False,
        "install_command": "image2editable models install agent",
        "reason": "model receipt must be an object",
    }


def test_cli_import_does_not_import_local_models_module() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import image2editable.cli; "
                "print('image2editable.models' in sys.modules)"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_host_run_never_probes_or_modifies_local_model_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "native.pptx"
    Presentation().save(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")

    def unexpected_call(*args: object, **kwargs: object) -> object:
        raise AssertionError("host run touched local model management")

    monkeypatch.setattr(models, "detect_hardware", unexpected_call)
    monkeypatch.setattr(models, "model_status", unexpected_call)

    summary = runtime.run_job(run_dir)

    assert summary["status"] == "completed"
    assert not (tmp_path / "model-cache").exists()
