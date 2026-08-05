from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "image-to-psd"


def test_psd_skill_documents_shared_agent_quality_workflow() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    for required in (
        "--format psd",
        "--agent-provider host",
        "--agent-provider local",
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


def test_psd_skill_launcher_does_not_embed_stale_cv_pipeline() -> None:
    launcher = (SKILL / "scripts" / "image_to_psd.py").read_text(encoding="utf-8")

    assert "image2editable.cli" in launcher
    assert '"--format", "psd"' in launcher
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


def test_psd_skill_contains_only_psd_specific_scripts() -> None:
    script_names = {
        path.name
        for path in (SKILL / "scripts").iterdir()
        if path.is_file() and path.suffix == ".py"
    }

    assert script_names == {"image_to_psd.py", "psd_assemble.py"}


def test_psd_skill_assembler_matches_runtime_assembler() -> None:
    root = ROOT / "scripts" / "psd_assemble.py"
    bundled = SKILL / "scripts" / "psd_assemble.py"

    assert hashlib.sha256(root.read_bytes()).digest() == hashlib.sha256(
        bundled.read_bytes()
    ).digest()
