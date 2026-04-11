from pathlib import Path


def test_skill_instructions_cover_core_guarantees():
    text = Path("skills/pdf-image-to-editable-ppt/SKILL.md").read_text(encoding="utf-8")
    assert "multi-page PDF" in text
    assert "visual fidelity" in text
    assert "background layer" in text
    assert "editable layer" in text
    assert "do not force extraction" in text


def test_skill_mentions_runtime_pipeline():
    text = Path("skills/pdf-image-to-editable-ppt/SKILL.md").read_text(encoding="utf-8")
    assert "PyMuPDF" in text
    assert "PaddleOCR" in text
    assert "convert_to_ppt.py" in text


def test_skill_mentions_stage2_rules():
    text = Path("skills/pdf-image-to-editable-ppt/SKILL.md").read_text(encoding="utf-8")
    assert "fully identical" in text
    assert "Stage 2" in text
    assert "effects" in text


def test_skill_mentions_2b_layering_and_vector_boundaries():
    text = Path("skills/pdf-image-to-editable-ppt/SKILL.md").read_text(encoding="utf-8")
    assert "2B" in text
    assert "layering" in text
    assert "vector" in text
