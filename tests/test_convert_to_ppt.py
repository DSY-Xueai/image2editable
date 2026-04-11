from conftest import load_skill_module


models = load_skill_module("models")
convert_module = load_skill_module("convert_to_ppt")

PagePlan = models.PagePlan
TextBlock = models.TextBlock
ImageBlock = models.ImageBlock
convert_to_ppt = convert_module.convert_to_ppt


def test_convert_to_ppt_rejects_missing_input(tmp_path):
    output_path = tmp_path / "out.pptx"

    try:
        convert_to_ppt(tmp_path / "missing.pdf", output_path)
    except FileNotFoundError as exc:
        assert "missing.pdf" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for a missing input file")


def test_convert_to_ppt_wires_render_extract_plan_filter_and_build(monkeypatch, tmp_path):
    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"fake pdf")
    output_path = tmp_path / "out.pptx"

    page_plan = PagePlan(
        page_number=1,
        width_px=640,
        height_px=480,
        background_path="page-1.png",
    )

    calls = []

    def fake_render_pdf_pages(path, output_dir):
        calls.append(("render", path, output_dir))
        return ["page-1.png"]

    def fake_extract_text_layout(path, *, page_number):
        calls.append(("text", path, page_number))
        return [
            {
                "text": "Hello",
                "left": 1,
                "top": 2,
                "width": 3,
                "height": 4,
                "font_size": 12,
                "color": "#000000",
                "alignment": "left",
                "confidence": 0.9,
            }
        ]

    def fake_extract_images(path, *, page_number, output_dir):
        calls.append(("image", path, page_number, output_dir))
        return [
            {
                "path": "logo.png",
                "left": 5,
                "top": 6,
                "width": 7,
                "height": 8,
                "confidence": 0.95,
                "extractable": True,
            }
        ]

    def fake_build_page_plan(page_dict):
        calls.append(("plan", page_dict["page_number"]))
        return page_plan

    def fake_select_editable_blocks(plan, *, min_text_confidence, min_image_confidence):
        calls.append(("filter", min_text_confidence, min_image_confidence))
        return plan

    def fake_build_presentation(page_plans, output_path_arg):
        calls.append(("build", len(page_plans), output_path_arg))

    monkeypatch.setattr(convert_module, "render_pdf_pages", fake_render_pdf_pages)
    monkeypatch.setattr(convert_module, "extract_text_layout", fake_extract_text_layout)
    monkeypatch.setattr(convert_module, "extract_images", fake_extract_images)
    monkeypatch.setattr(convert_module, "build_page_plan", fake_build_page_plan)
    monkeypatch.setattr(convert_module, "select_editable_blocks", fake_select_editable_blocks)
    monkeypatch.setattr(convert_module, "build_presentation", fake_build_presentation)

    result = convert_to_ppt(input_path, output_path)

    assert result == output_path
    assert [item[0] for item in calls] == ["render", "text", "image", "plan", "filter", "build"]
