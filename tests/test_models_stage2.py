from conftest import load_skill_module


models = load_skill_module("models")
TextBlock = models.TextBlock
EffectBlock = models.EffectBlock


def test_text_block_supports_fit_validation_flags():
    block = TextBlock(
        text="Title",
        left=10.0,
        top=20.0,
        width=200.0,
        height=40.0,
        font_size=24.0,
        color="#112233",
        alignment="center",
        confidence=0.95,
        font_name="Arial",
        line_height=28.0,
        fit_verified=True,
    )
    assert block.font_name == "Arial"
    assert block.line_height == 28.0
    assert block.fit_verified is True


def test_effect_block_tracks_effect_kind_and_confidence():
    block = EffectBlock(
        effect_type="shadow",
        left=0.0,
        top=0.0,
        width=100.0,
        height=40.0,
        confidence=0.9,
        payload={"blur": 2.0},
    )
    assert block.effect_type == "shadow"
    assert block.payload["blur"] == 2.0
