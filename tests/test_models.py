from conftest import load_skill_module


models = load_skill_module("models")
ImageBlock = models.ImageBlock
PagePlan = models.PagePlan
TextBlock = models.TextBlock


def test_page_plan_defaults_to_background_only():
    plan = PagePlan(page_number=1, width_px=1000, height_px=1500, background_path="page-1.png")
    assert plan.page_number == 1
    assert plan.text_blocks == []
    assert plan.image_blocks == []


def test_text_block_tracks_confidence_and_layout():
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
    assert block.alignment == "center"
    assert block.confidence == 0.95


def test_image_block_can_fall_back_to_background():
    block = ImageBlock(
        path="img.png",
        left=1.0,
        top=2.0,
        width=3.0,
        height=4.0,
        confidence=0.4,
        extractable=False,
    )
    assert block.extractable is False
