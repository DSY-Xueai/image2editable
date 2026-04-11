from conftest import load_skill_module


models = load_skill_module("models")
stage2 = load_skill_module("stage2_enhance")

PagePlan = models.PagePlan
TextBlock = models.TextBlock


def test_stage2_enhance_removes_unverified_text_blocks():
    plan = PagePlan(page_number=1, width_px=100, height_px=100, background_path="page.png")
    plan.text_blocks.append(
        TextBlock(
            text="Title",
            left=0,
            top=0,
            width=10,
            height=10,
            font_size=12,
            color="#000000",
            alignment="left",
            confidence=0.95,
            font_name="Arial",
            fit_verified=False,
        )
    )
    enhanced = stage2.apply_stage2_enhancements(plan)
    assert enhanced.text_blocks == []
