from conftest import load_skill_module


models = load_skill_module("models")
filtering = load_skill_module("filtering")

ImageBlock = models.ImageBlock
PagePlan = models.PagePlan
TextBlock = models.TextBlock
select_editable_blocks = filtering.select_editable_blocks


def test_low_confidence_text_is_removed_from_editable_layer():
    plan = PagePlan(page_number=1, width_px=100, height_px=100, background_path="page.png")
    plan.text_blocks.append(TextBlock("x", 0, 0, 10, 10, 12, "#000000", "left", 0.49))
    filtered = select_editable_blocks(
        plan, min_text_confidence=0.8, min_image_confidence=0.8
    )
    assert filtered.text_blocks == []


def test_high_confidence_text_and_images_are_preserved():
    plan = PagePlan(page_number=1, width_px=100, height_px=100, background_path="page.png")
    plan.text_blocks.append(TextBlock("x", 0, 0, 10, 10, 12, "#000000", "left", 0.95))
    plan.image_blocks.append(ImageBlock("img.png", 0, 0, 10, 10, 0.96, True))
    filtered = select_editable_blocks(
        plan, min_text_confidence=0.8, min_image_confidence=0.8
    )
    assert len(filtered.text_blocks) == 1
    assert len(filtered.image_blocks) == 1
