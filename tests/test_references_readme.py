from pathlib import Path


def test_references_document_known_limits():
    text = Path("skills/pdf-image-to-editable-ppt/references/README.md").read_text(
        encoding="utf-8"
    )
    assert "multi-page PDF" in text
    assert "fallback" in text
    assert "font" in text
    assert "OCR" in text


def test_references_document_2b_scope():
    text = Path("skills/pdf-image-to-editable-ppt/references/README.md").read_text(
        encoding="utf-8"
    )
    assert "2B" in text
    assert "layering" in text
    assert "vector boundaries" in text
    assert "blend-heavy" in text
