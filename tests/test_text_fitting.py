from types import SimpleNamespace

from conftest import load_skill_module


text_fitting = load_skill_module("text_fitting")
models = load_skill_module("models")
TextBlock = models.TextBlock


def test_validate_text_block_fit_rejects_line_mismatch():
    candidate = SimpleNamespace(
        expected_lines=2,
        predicted_lines=1,
        position_delta=0.0,
        font_name="Arial",
    )

    assert text_fitting.validate_text_block_fit(candidate) is False


def test_validate_text_block_fit_rejects_position_delta():
    candidate = SimpleNamespace(
        expected_lines=1,
        predicted_lines=1,
        position_delta=0.5,
        font_name="Arial",
    )

    assert text_fitting.validate_text_block_fit(candidate) is False


def test_validate_text_block_fit_rejects_missing_font_name():
    candidate = SimpleNamespace(
        expected_lines=1,
        predicted_lines=1,
        position_delta=0.0,
        font_name=None,
    )

    assert text_fitting.validate_text_block_fit(candidate) is False


def test_mark_fit_verified_sets_flag_and_returns_block():
    block = TextBlock(
        text="Hello",
        left=0.0,
        top=0.0,
        width=10.0,
        height=10.0,
        font_size=12.0,
        color="#000000",
        alignment="left",
        confidence=0.99,
        font_name="Arial",
    )

    result = text_fitting.mark_fit_verified(block)

    assert result is block
    assert block.fit_verified is True
