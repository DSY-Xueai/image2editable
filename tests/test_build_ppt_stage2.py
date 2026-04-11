from pathlib import Path

from PIL import Image

from conftest import load_skill_module


models = load_skill_module("models")
build_ppt = load_skill_module("build_ppt")

PagePlan = models.PagePlan
TextBlock = models.TextBlock


def test_build_presentation_accepts_fit_verified_text_blocks(tmp_path):
    background = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(background)
    plan = PagePlan(page_number=1, width_px=100, height_px=100, background_path=str(background))
    plan.text_blocks.append(
        TextBlock(
            text="Title",
            left=10,
            top=10,
            width=50,
            height=20,
            font_size=16,
            color="#000000",
            alignment="left",
            confidence=0.95,
            font_name="Arial",
            fit_verified=True,
        )
    )
    output = tmp_path / "stage2.pptx"
    build_ppt.build_presentation([plan], output)
    assert output.exists()
