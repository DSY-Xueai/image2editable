from pathlib import Path

from PIL import Image

from conftest import load_skill_module


models = load_skill_module("models")
build_ppt = load_skill_module("build_ppt")

PagePlan = models.PagePlan
build_presentation = build_ppt.build_presentation


def test_build_presentation_creates_a_pptx(tmp_path):
    background_path = tmp_path / "page.png"
    Image.new("RGB", (100, 200), "white").save(background_path)
    page = PagePlan(
        page_number=1,
        width_px=100,
        height_px=200,
        background_path=str(background_path),
    )
    output_path = tmp_path / "result.pptx"
    build_presentation([page], output_path)
    assert output_path.exists()
