from conftest import load_skill_module


models = load_skill_module("models")
EffectBlock = models.EffectBlock
PagePlan = models.PagePlan
TextBlock = models.TextBlock


def test_text_block_stage2_fields_default_cleanly():
    block = TextBlock(
        text="Title",
        left=10.0,
        top=20.0,
        width=100.0,
        height=30.0,
        font_size=24.0,
        color="#112233",
        alignment="center",
        confidence=0.95,
    )

    assert block.font_name is None
    assert block.line_height is None
    assert block.fit_verified is False


def test_effect_block_defaults_payload_to_empty_dict():
    effect = EffectBlock(
        effect_type="shadow",
        left=1.0,
        top=2.0,
        width=3.0,
        height=4.0,
        confidence=0.91,
    )

    assert effect.effect_type == "shadow"
    assert effect.payload == {}


def test_page_plan_includes_effect_blocks_list():
    plan = PagePlan(page_number=1, width_px=1000, height_px=1500, background_path="page-1.png")

    assert plan.effect_blocks == []
