from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "image-to-psd"
PSD_COMMON_ENGINE_FILES = {
    "__init__.py",
    "bg_model.py",
    "component_contracts.py",
    "component_quality.py",
    "component_underlay.py",
    "fg_extract.py",
    "image_to_ppt.py",
    "initial_diagnostics.py",
    "lama_inpaint.py",
    "lama_worker.py",
    "object_detect.py",
    "object_worker.py",
    "ocr_worker.py",
    "performance_trace.py",
    "runtime_model_paths.py",
    "sam_worker.py",
    "text_detect.py",
    "visual_compare_qa.py",
    "visual_segment.py",
    "visual_worker.py",
    "worker_resources.py",
}


def test_psd_skill_documents_shared_agent_quality_workflow() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    for required in (
        "--format psd",
        "python -m scripts.image_to_psd",
        "--agent-provider host",
        "--agent-provider local",
        "--agent-provider local-service",
        "SAM2_MODEL",
        "LAMA_MODEL",
        "GROUNDING_DINO_MODEL",
        "ASPOSE_PSD_LICENSE",
        "推理不会下载模型",
        "最多 5",
        "仅支持图片",
        "Aspose.PSD",
        "preserved_with_warning",
    ):
        assert required in text
    for stale in (
        "--diff-threshold",
        "--min-area",
        "背景建模瓦片周期",
    ):
        assert stale not in text


def test_psd_skill_requirements_cover_standalone_runtime_without_ppt_or_pdf() -> None:
    requirements = (SKILL / "references" / "requirements.txt").read_text(
        encoding="utf-8"
    )

    for required in (
        "opencv-python>=4.10.0.84,<5",
        "Pillow>=10.4,<12",
        "numpy>=1.26.4,<2",
        "torch>=2.5.1,<3",
        "torchvision>=0.20.1,<1",
        "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "transformers>=4.57,<5",
        "accelerate>=1.8,<2",
        "aspose-psd>=26.5.0",
    ):
        assert required in requirements.splitlines()
    assert "python-pptx" not in requirements
    assert "pypdfium2" not in requirements


def test_psd_skill_launcher_does_not_embed_stale_cv_pipeline() -> None:
    launcher = (SKILL / "scripts" / "image_to_psd.py").read_text(encoding="utf-8")

    assert "image2editable.cli" not in launcher
    assert "from scripts.image_to_ppt import" in launcher
    assert "from scripts.psd_assemble import" in launcher
    for stale_import in (
        "build_background",
        "extract_foreground_mask",
        "split_components",
    ):
        assert stale_import not in launcher


def test_repository_launcher_delegates_to_shared_runtime() -> None:
    source = (ROOT / "image_to_psd.py").read_text(encoding="utf-8")

    assert "image2editable.cli" in source
    assert '"--format", "psd"' in source
    assert "scripts.bg_model" not in source
    assert "scripts.fg_extract" not in source


def test_psd_skill_contains_current_standalone_engine() -> None:
    script_names = {
        path.name
        for path in (SKILL / "scripts").iterdir()
        if path.is_file() and path.suffix == ".py"
    }

    assert script_names == PSD_COMMON_ENGINE_FILES | {
        "image_to_psd.py",
        "psd_assemble.py",
    }
    assert "ppt_assemble.py" not in script_names


def test_psd_common_engine_matches_ppt_skill() -> None:
    ppt_scripts = ROOT / "skills" / "image-to-ppt" / "scripts"
    psd_scripts = SKILL / "scripts"

    for name in PSD_COMMON_ENGINE_FILES:
        assert (psd_scripts / name).read_bytes() == (ppt_scripts / name).read_bytes()


def test_ppt_writer_is_loaded_only_for_ppt_assembly() -> None:
    for path in (
        ROOT / "image_to_ppt.py",
        ROOT / "skills" / "image-to-ppt" / "scripts" / "image_to_ppt.py",
    ):
        source = path.read_text(encoding="utf-8")
        prefix = source.split("logger =", 1)[0]

        assert "from scripts.ppt_assemble import" not in prefix
        assert "def assemble_pptx(*args, **kwargs):" in source
        assert "def assemble_pptx_multi(*args, **kwargs):" in source
        assert source.count(
            "from scripts.ppt_assemble import assemble_pptx as writer"
        ) == 1
        assert source.count(
            "from scripts.ppt_assemble import assemble_pptx_multi as writer"
        ) == 1


def test_psd_skill_assembler_matches_runtime_assembler() -> None:
    root = ROOT / "scripts" / "psd_assemble.py"
    bundled = SKILL / "scripts" / "psd_assemble.py"

    assert hashlib.sha256(root.read_bytes()).digest() == hashlib.sha256(
        bundled.read_bytes()
    ).digest()


def _load_psd_skill_launcher():
    script = SKILL / "scripts" / "image_to_psd.py"
    spec = importlib.util.spec_from_file_location("standalone_image_to_psd", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_path = list(sys.path)
    previous_scripts = {
        name: loaded
        for name, loaded in sys.modules.items()
        if name == "scripts" or name.startswith("scripts.")
    }
    for name in previous_scripts:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(SKILL))
    try:
        spec.loader.exec_module(module)
    finally:
        for name in list(sys.modules):
            if name == "scripts" or name.startswith("scripts."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_scripts)
        sys.path[:] = previous_path
    return module


def _slide_data(tmp_path: Path, stem: str = "page") -> dict:
    background = tmp_path / f"{stem}-background.png"
    background.write_bytes(b"background")
    return {
        "background_original_path": str(background),
        "components": [{"path": "component.png", "x": 1, "y": 2, "w": 3, "h": 4}],
        "text_items": [{"text": "hello", "box": [1, 2, 30, 10]}],
        "img_width": 100,
        "img_height": 80,
    }


def test_psd_standalone_convert_preflights_and_assembles_original_canvas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_psd_skill_launcher()
    source = tmp_path / "input.png"
    source.write_bytes(b"input")
    slide = _slide_data(tmp_path)
    events = []
    calls = []

    monkeypatch.setattr(
        module,
        "_preflight_standalone_runtime",
        lambda: events.append("preflight"),
    )
    monkeypatch.setattr(
        module,
        "_prepare_single_image",
        lambda *args, **kwargs: (
            events.append("prepare") or (slide, tmp_path / "work")
        ),
    )

    def fake_assemble(**kwargs):
        events.append("assemble")
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"psd")
        return str(kwargs["output_path"])

    monkeypatch.setattr(module, "assemble_psd", fake_assemble)

    output = module.convert(source, tmp_path / "output.psd")

    assert events == ["preflight", "prepare", "assemble"]
    assert Path(output).read_bytes() == b"psd"
    assert calls[0]["background_path"] == slide["background_original_path"]
    assert calls[0]["components"] == slide["components"]
    assert calls[0]["text_items"] == slide["text_items"]
    assert calls[0]["img_width"] == slide["img_width"]
    assert calls[0]["img_height"] == slide["img_height"]


def test_psd_standalone_batch_disambiguates_duplicate_stems(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_psd_skill_launcher()
    first = tmp_path / "first" / "page.png"
    second = tmp_path / "second" / "page.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    slides = [_slide_data(tmp_path, "first"), _slide_data(tmp_path, "second")]

    monkeypatch.setattr(module, "_preflight_standalone_runtime", lambda: None)
    monkeypatch.setattr(
        module,
        "_prepare_multiple_images",
        lambda *args, **kwargs: slides,
    )

    def fake_assemble(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"psd")
        return str(kwargs["output_path"])

    monkeypatch.setattr(module, "assemble_psd", fake_assemble)

    outputs = module.convert_batch([first, second], tmp_path / "out")

    assert [Path(path).name for path in outputs] == ["page.psd", "page_2.psd"]
    assert all(Path(path).read_bytes() == b"psd" for path in outputs)


def test_psd_standalone_rejects_existing_output_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_psd_skill_launcher()
    source = tmp_path / "input.png"
    output = tmp_path / "output.psd"
    source.write_bytes(b"input")
    output.write_bytes(b"existing")
    preflight_calls = []
    monkeypatch.setattr(
        module,
        "_preflight_standalone_runtime",
        lambda: preflight_calls.append(True),
    )

    with pytest.raises(FileExistsError, match="output.psd"):
        module.convert(source, output)

    assert preflight_calls == []
    assert output.read_bytes() == b"existing"


def test_psd_standalone_does_not_assemble_after_prepare_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_psd_skill_launcher()
    source = tmp_path / "input.png"
    source.write_bytes(b"input")
    assemble_calls = []
    monkeypatch.setattr(module, "_preflight_standalone_runtime", lambda: None)

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("quality gate failed")

    monkeypatch.setattr(module, "_prepare_single_image", fail_prepare)
    monkeypatch.setattr(
        module,
        "assemble_psd",
        lambda **kwargs: assemble_calls.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="quality gate failed"):
        module.convert(source, tmp_path / "output.psd")

    assert assemble_calls == []
    assert not (tmp_path / "output.psd").exists()


@pytest.mark.parametrize("batched", [False, True])
def test_psd_standalone_rejects_file_used_as_output_directory_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    batched: bool,
) -> None:
    module = _load_psd_skill_launcher()
    source = tmp_path / "input.png"
    output_container = tmp_path / "not-a-directory"
    source.write_bytes(b"input")
    output_container.write_bytes(b"file")
    preflight_calls = []
    monkeypatch.setattr(
        module,
        "_preflight_standalone_runtime",
        lambda: preflight_calls.append(True),
        raising=False,
    )

    with pytest.raises(NotADirectoryError, match="not-a-directory"):
        if batched:
            module.convert_batch([source], output_container)
        else:
            module.convert(source, output_container)

    assert preflight_calls == []


def test_psd_standalone_model_path_preflight_runs_before_ocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_psd_skill_launcher()
    sam = tmp_path / "sam.pt"
    lama = tmp_path / "lama.pt"
    dino = tmp_path / "dino"
    sam.write_bytes(b"sam")
    lama.write_bytes(b"lama")
    dino.mkdir()
    monkeypatch.setenv("SAM2_MODEL", str(sam))
    monkeypatch.setenv("LAMA_MODEL", str(lama))
    monkeypatch.setenv("GROUNDING_DINO_MODEL", str(dino))
    events = []
    monkeypatch.setattr(
        module,
        "preflight_psd_runtime",
        lambda: events.append("license"),
    )

    module._preflight_standalone_runtime()

    assert events == ["license"]


def test_psd_standalone_model_path_preflight_rejects_relative_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_psd_skill_launcher()
    lama = tmp_path / "lama.pt"
    dino = tmp_path / "dino"
    lama.write_bytes(b"lama")
    dino.mkdir()
    monkeypatch.setenv("SAM2_MODEL", "relative/sam.pt")
    monkeypatch.setenv("LAMA_MODEL", str(lama))
    monkeypatch.setenv("GROUNDING_DINO_MODEL", str(dino))
    monkeypatch.setattr(module, "preflight_psd_runtime", lambda: None)

    with pytest.raises(RuntimeError, match="SAM2_MODEL must be an absolute"):
        module._preflight_standalone_runtime()


def test_psd_standalone_batch_does_not_publish_partial_assembly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_psd_skill_launcher()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    slides = [_slide_data(tmp_path, "first"), _slide_data(tmp_path, "second")]
    calls = []
    monkeypatch.setattr(module, "_preflight_standalone_runtime", lambda: None)
    monkeypatch.setattr(
        module,
        "_prepare_multiple_images",
        lambda *args, **kwargs: slides,
    )

    def fail_second_assembly(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("second assembly failed")
        Path(kwargs["output_path"]).write_bytes(b"first psd")
        return str(kwargs["output_path"])

    monkeypatch.setattr(module, "assemble_psd", fail_second_assembly)

    with pytest.raises(RuntimeError, match="second assembly failed"):
        module.convert_batch([first, second], tmp_path / "out")

    assert not (tmp_path / "out" / "first.psd").exists()
    assert not (tmp_path / "out" / "second.psd").exists()


def test_psd_standalone_batch_rolls_back_after_publish_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_psd_skill_launcher()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    slides = [_slide_data(tmp_path, "first"), _slide_data(tmp_path, "second")]
    monkeypatch.setattr(module, "_preflight_standalone_runtime", lambda: None)
    monkeypatch.setattr(
        module,
        "_prepare_multiple_images",
        lambda *args, **kwargs: slides,
    )

    def assemble(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"staged psd")
        return str(kwargs["output_path"])

    real_link = os.link
    link_calls = []

    def conflict_on_second_link(source, destination):
        link_calls.append(Path(destination))
        if len(link_calls) == 2:
            Path(destination).write_bytes(b"concurrent output")
            raise FileExistsError(destination)
        real_link(source, destination)

    monkeypatch.setattr(module, "assemble_psd", assemble)
    monkeypatch.setattr(module.os, "link", conflict_on_second_link)

    output_dir = tmp_path / "out"
    with pytest.raises(FileExistsError):
        module.convert_batch([first, second], output_dir)

    assert not (output_dir / "first.psd").exists()
    assert (output_dir / "second.psd").read_bytes() == b"concurrent output"


def test_psd_standalone_help_does_not_import_product_runtime(tmp_path: Path) -> None:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import importlib.abc
import sys

class BlockImage2Editable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "image2editable" or fullname.startswith("image2editable."):
            raise ModuleNotFoundError("blocked product runtime", name=fullname)
        return None

sys.meta_path.insert(0, BlockImage2Editable())
""".lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "scripts.image_to_psd", "--help"],
        cwd=SKILL,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout
