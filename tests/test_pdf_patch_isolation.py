from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from reportlab.pdfgen import canvas

from image2editable.pdf_input import prepare_pdf_job
from image2editable.store import RunStore


@pytest.mark.parametrize("text_first", [False, True])
def test_pdf_patch_has_no_text_or_overlapping_neighbours(tmp_path: Path, text_first):
    source = tmp_path / "overlap.pdf"
    pdf = canvas.Canvas(str(source), pagesize=(240, 160))
    pdf.setFillColorRGB(0, 0, 1)
    pdf.rect(0, 0, 240, 160, fill=1, stroke=0)

    def text():
        pdf.setFillColorRGB(1, 1, 1)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(45, 90, "EDITABLE")

    if text_first:
        text()
    pdf.setFillColorRGB(1, 0, 0)
    pdf.roundRect(30, 65, 140, 55, 12, fill=1, stroke=0)
    if not text_first:
        text()
    pdf.setFillColorRGB(0, 1, 0)
    pdf.circle(165, 90, 20, fill=1, stroke=0)
    pdf.save()
    run = prepare_pdf_job(source, run_dir=tmp_path / "run")
    request = RunStore.open(run).read_json("pages/page_001/page_request.json")
    analysis = request["pdf_analysis"]
    assert analysis["classification"] == "hybrid"
    assert any(obj.get("text") == "EDITABLE" for obj in analysis["objects"])
    patches = [obj for obj in analysis["objects"] if obj["type"] == "patch"]
    assert len(patches) == 2
    for patch, expected in zip(patches, [(255, 0, 0), (0, 255, 0)], strict=True):
        with Image.open(run / patch["asset_path"]) as image:
            assert image.mode == "RGBA"
            pixels = np.asarray(image)
        assert np.any(pixels[:, :, 3] == 0), "corners must preserve underlying layers"
        opaque = pixels[:, :, 3] == 255
        assert np.count_nonzero(opaque) > 100
        assert np.all(pixels[:, :, :3][opaque] == expected)


def test_pdf_patch_keeps_thick_stroke_outside_path_bounds(tmp_path: Path):
    source = tmp_path / "stroke.pdf"
    pdf = canvas.Canvas(str(source), pagesize=(240, 160))
    pdf.setStrokeColorRGB(1, 0, 0)
    pdf.setLineWidth(12)
    pdf.circle(100, 80, 20, fill=0, stroke=1)
    pdf.save()
    run = prepare_pdf_job(source, run_dir=tmp_path / "run")
    analysis = RunStore.open(run).read_json("pages/page_001/page_request.json")["pdf_analysis"]
    patch = next(obj for obj in analysis["objects"] if obj["type"] == "patch")
    assert patch["bbox_pt"][0] < 74
    assert patch["bbox_pt"][2] > 126
    with Image.open(run / patch["asset_path"]) as image:
        alpha = np.asarray(image)[:, :, 3]
    assert np.any(alpha == 255)
    assert not alpha[[0, -1], :].any()
    assert not alpha[:, [0, -1]].any()
