from conftest import load_skill_module


models = load_skill_module("models")
PagePlan = models.PagePlan


def test_page_plan_accepts_pdf_page_dimensions_in_points():
    plan = PagePlan(
        page_number=1,
        width_px=1000,
        height_px=1500,
        background_path="page-1.png",
        source_type="pdf",
        page_width_points=595.0,
        page_height_points=842.0,
    )

    assert plan.source_type == "pdf"
    assert plan.page_width_points == 595.0
    assert plan.page_height_points == 842.0
