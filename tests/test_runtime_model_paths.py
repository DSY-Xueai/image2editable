from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from image2editable import runtime_models
from scripts import runtime_model_paths
from scripts import visual_segment


def test_file_model_identities_are_fixed_release_values() -> None:
    assert runtime_model_paths.FILE_MODELS == {
        "sam2_large": (
            "SAM2_MODEL",
            898083611,
            "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
        ),
        "big_lama": (
            "LAMA_MODEL",
            205803670,
            "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c",
        ),
    }


def _small_file_model(monkeypatch: pytest.MonkeyPatch, name: str, payload: bytes) -> None:
    env_name, _, _ = runtime_model_paths.FILE_MODELS[name]
    monkeypatch.setitem(
        runtime_model_paths.FILE_MODELS,
        name,
        (env_name, len(payload), hashlib.sha256(payload).hexdigest()),
    )


@pytest.mark.parametrize(
    ("name", "env_name"),
    [("sam2_large", "SAM2_MODEL"), ("big_lama", "LAMA_MODEL")],
)
def test_explicit_file_model_is_verified_without_product_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    env_name: str,
) -> None:
    payload = f"fixed-{name}".encode()
    model = tmp_path / f"{name}.pt"
    model.write_bytes(payload)
    _small_file_model(monkeypatch, name, payload)
    monkeypatch.setenv(env_name, str(model.resolve()))
    monkeypatch.setattr(
        runtime_model_paths,
        "_product_runtime_model_path",
        lambda _name: pytest.fail("explicit standalone path must not import product"),
    )

    assert runtime_model_paths.resolve_runtime_model_path(name) == model.resolve()


@pytest.mark.parametrize("name", ["sam2_large", "big_lama"])
def test_explicit_file_model_rejects_wrong_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    env_name, _, _ = runtime_model_paths.FILE_MODELS[name]
    model = tmp_path / f"{name}.pt"
    model.write_bytes(b"tampered")
    monkeypatch.setenv(env_name, str(model.resolve()))

    with pytest.raises(runtime_model_paths.RuntimeModelPathError, match="integrity"):
        runtime_model_paths.resolve_runtime_model_path(name)


def test_explicit_file_model_rejects_same_size_wrong_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "sam.pt"
    model.write_bytes(b"tampered")
    monkeypatch.setitem(
        runtime_model_paths.FILE_MODELS,
        "sam2_large",
        ("SAM2_MODEL", len(b"tampered"), hashlib.sha256(b"expected").hexdigest()),
    )
    monkeypatch.setenv("SAM2_MODEL", str(model.resolve()))

    with pytest.raises(runtime_model_paths.RuntimeModelPathError, match="integrity"):
        runtime_model_paths.resolve_runtime_model_path("sam2_large")


def test_explicit_grounding_dino_directory_is_absolute_local_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "dino"
    snapshot.mkdir()
    monkeypatch.setenv("GROUNDING_DINO_MODEL", str(snapshot.resolve()))

    assert runtime_model_paths.resolve_runtime_model_path("grounding_dino") == snapshot.resolve()

    monkeypatch.setenv("GROUNDING_DINO_MODEL", "relative/dino")
    with pytest.raises(runtime_model_paths.RuntimeModelPathError, match="absolute"):
        runtime_model_paths.resolve_runtime_model_path("grounding_dino")


def test_explicit_grounding_dino_rejects_linked_snapshot_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(snapshot, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    monkeypatch.setenv("GROUNDING_DINO_MODEL", str(linked.absolute()))

    with pytest.raises(runtime_model_paths.RuntimeModelPathError, match="non-link"):
        runtime_model_paths.resolve_runtime_model_path("grounding_dino")


def test_default_model_path_delegates_to_product_receipt_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "verified"
    calls: list[str] = []
    for env_name in ("SAM2_MODEL", "LAMA_MODEL", "GROUNDING_DINO_MODEL"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(
        runtime_models,
        "runtime_model_path",
        lambda name: calls.append(name) or expected,
    )

    assert runtime_model_paths.resolve_runtime_model_path("grounding_dino") == expected
    assert calls == ["grounding_dino"]


def test_sam_consumer_uses_runtime_model_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "sam.pt"
    calls: list[str] = []
    monkeypatch.setattr(
        visual_segment,
        "resolve_runtime_model_path",
        lambda name: calls.append(name) or checkpoint,
    )

    assert visual_segment.resolve_sam_checkpoint() == checkpoint
    assert calls == ["sam2_large"]


def test_standalone_requires_explicit_path_when_product_package_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_name in ("SAM2_MODEL", "LAMA_MODEL", "GROUNDING_DINO_MODEL"):
        monkeypatch.delenv(env_name, raising=False)

    def unavailable(_name: str) -> Path:
        raise ModuleNotFoundError("No module named 'image2editable'", name="image2editable")

    monkeypatch.setattr(runtime_model_paths, "_product_runtime_model_path", unavailable)

    with pytest.raises(
        runtime_model_paths.RuntimeModelPathError,
        match=r"GROUNDING_DINO_MODEL.*absolute",
    ):
        runtime_model_paths.resolve_runtime_model_path("grounding_dino")


def test_runtime_model_bridge_and_consumers_match_skill_mirror() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "runtime_model_paths.py",
        "visual_segment.py",
        "lama_inpaint.py",
        "object_detect.py",
    ):
        assert (root / "scripts" / name).read_bytes() == (
            root / "skills" / "image-to-ppt" / "scripts" / name
        ).read_bytes()


def test_inference_scripts_contain_no_model_downloader() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("visual_segment.py", "lama_inpaint.py", "object_detect.py"):
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "urlretrieve" not in source
        assert "snapshot_download" not in source


def test_skill_documents_offline_product_and_standalone_model_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    skill = (root / "skills" / "image-to-ppt" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "推理不会下载模型或回退 Hugging Face cache" in skill
    assert "独立 skill 不假设该包存在" in skill
    for env_name in ("SAM2_MODEL", "LAMA_MODEL", "GROUNDING_DINO_MODEL"):
        assert env_name in skill
    assert skill.index("image2editable models install runtime") < skill.index(
        "image2editable doctor"
    )
    assert skill.index('python -m pip install ".[agent-local]"') < skill.index(
        "image2editable models install agent"
    ) < skill.index("image2editable doctor --agent-local")
    assert "不运行 `image2editable doctor`" in skill
    assert "产品环境须通过 `doctor`，所有环境须通过下列设备预检" in skill
    assert "SAM2_MODEL`、`LAMA_MODEL` 必须指向文件" in skill
    assert "GROUNDING_DINO_MODEL` 必须指向目录" in skill
    assert "runtime model paths: ok" in skill
