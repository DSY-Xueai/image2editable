from pathlib import Path


def test_skill_instructions_cover_core_guarantees():
    text = Path("skills/pdf-image-to-editable-ppt/SKILL.md").read_text(encoding="utf-8")
    assert "multi-page PDF" in text
    assert "visual fidelity" in text
    assert "background layer" in text
    assert "editable layer" in text
    assert "do not force extraction" in text
