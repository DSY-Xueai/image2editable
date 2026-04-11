from types import SimpleNamespace

from conftest import load_skill_module


stage2_enhance = load_skill_module("stage2_enhance")


def test_apply_stage2_enhancements_keeps_only_verified_text_blocks():
    verified_text = SimpleNamespace(text="kept", fit_verified=True)
    rejected_text = SimpleNamespace(text="removed", fit_verified=False)
    image_block = SimpleNamespace(path="image.png")
    effect_block = SimpleNamespace(kind="shadow")
    plan = SimpleNamespace(
        page_number=7,
        width_px=1200,
        height_px=900,
        background_path="page.png",
        source_type="pdf",
        page_width_points=612.0,
        page_height_points=792.0,
        text_blocks=[verified_text, rejected_text],
        image_blocks=[image_block],
        effect_blocks=[effect_block],
    )

    enhanced = stage2_enhance.apply_stage2_enhancements(plan)

    assert enhanced is not plan
    assert enhanced.page_number == plan.page_number
    assert enhanced.width_px == plan.width_px
    assert enhanced.height_px == plan.height_px
    assert enhanced.background_path == plan.background_path
    assert enhanced.source_type == plan.source_type
    assert enhanced.page_width_points == plan.page_width_points
    assert enhanced.page_height_points == plan.page_height_points
    assert enhanced.text_blocks == [verified_text]
    assert enhanced.image_blocks == [image_block]
    assert enhanced.effect_blocks == [effect_block]
