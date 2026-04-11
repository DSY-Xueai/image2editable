from conftest import load_skill_module


models = load_skill_module("models")
page_planner = load_skill_module("page_planner")

PagePlan = models.PagePlan
TextBlock = models.TextBlock
ImageBlock = models.ImageBlock
build_page_plan = page_planner.build_page_plan


def test_build_page_plan_normalizes_nested_dicts():
    plan = build_page_plan(
        {
            "page_number": 3,
            "width_px": 1280,
            "height_px": 720,
            "background_path": "page-3.png",
            "text_blocks": [
                {
                    "text": "Title",
                    "left": 10,
                    "top": 20,
                    "width": 30,
                    "height": 40,
                    "font_size": 24,
                    "color": "#112233",
                    "alignment": "center",
                    "confidence": 0.97,
                }
            ],
            "image_blocks": [
                {
                    "path": "logo.png",
                    "left": 50,
                    "top": 60,
                    "width": 70,
                    "height": 80,
                    "confidence": 0.88,
                    "extractable": False,
                }
            ],
        }
    )

    assert isinstance(plan, PagePlan)
    assert plan.page_number == 3
    assert plan.background_path == "page-3.png"
    assert isinstance(plan.text_blocks[0], TextBlock)
    assert isinstance(plan.image_blocks[0], ImageBlock)
    assert plan.text_blocks[0].alignment == "center"
    assert plan.image_blocks[0].extractable is False
