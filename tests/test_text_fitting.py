from conftest import load_skill_module


text_fitting = load_skill_module("text_fitting")


def test_validate_text_block_fit_rejects_unverified_wrap():
    candidate = {
        "text": "A long title",
        "font_name": "Arial",
        "font_size": 24.0,
        "line_height": 28.0,
        "left": 10.0,
        "top": 20.0,
        "width": 200.0,
        "height": 40.0,
        "alignment": "center",
        "expected_lines": 1,
        "predicted_lines": 2,
    }
    result = text_fitting.validate_text_block_fit(candidate)
    assert result is False


def test_validate_text_block_fit_accepts_exact_layout_match():
    candidate = {
        "text": "Title",
        "font_name": "Arial",
        "font_size": 24.0,
        "line_height": 28.0,
        "left": 10.0,
        "top": 20.0,
        "width": 200.0,
        "height": 40.0,
        "alignment": "center",
        "expected_lines": 1,
        "predicted_lines": 1,
        "position_delta": 0.0,
    }
    result = text_fitting.validate_text_block_fit(candidate)
    assert result is True
