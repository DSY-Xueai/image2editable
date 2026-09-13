from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from scripts import font_match, text_detect


@pytest.mark.parametrize('bold', [False, True])
@pytest.mark.parametrize('adjacent_fragment', [False, True])
def test_plain_text_matches_native_variable_weight_and_reuses_evidence(monkeypatch, bold, adjacent_fragment):
    path = Path(__file__).resolve().parents[1] / 'benchmarks/release/fonts/NotoSansSC[wght].ttf'
    monkeypatch.setattr(font_match, 'installed_faces', lambda: tuple(
        ('Noto Sans SC', weight, False, str(path), 0) for weight in (False, True)
    ))
    font_match.match_text_face.cache_clear()
    font = ImageFont.truetype(str(path), 48)
    font.set_variation_by_name('Bold' if bold else 'Regular')
    image = Image.new('RGB', (480, 110), '#182838')
    ImageDraw.Draw(image).text((15, 6), 'MOVE  IT', font=font, fill='#e8eef4')
    if adjacent_fragment:
        ImageDraw.Draw(image).rectangle((20, 1, 50, 2), fill='#e8eef4')
    item = {'text': 'MOVE IT', 'box': [8, 0, 420, 100], 'font': 'Arial', 'bold': not bold}
    pixels = np.asarray(image)
    actual = text_detect.refine_plain_text_fonts(pixels, [item])[0]
    assert actual['text'] == item['text']
    assert actual['font'] == 'Noto Sans SC'
    assert actual['bold'] is bold
    assert actual['box_kind'] == 'ink'
    assert actual['box'][1] > 5
    assert abs(actual['font_size'] * image.width / (13.333 * 72) - 48) < 3
    before = font_match.match_text_face.cache_info().hits
    assert text_detect.refine_plain_text_fonts(pixels, [item])[0] == actual
    assert font_match.match_text_face.cache_info().hits == before + 1
    font_match.match_text_face.cache_clear()


@pytest.mark.parametrize('size', [9, 14, 32])
def test_plain_font_matching_excludes_cell_border(monkeypatch, size):
    path = Path(__file__).resolve().parents[1] / 'benchmarks/release/fonts/NotoSansSC[wght].ttf'
    monkeypatch.setattr(font_match, 'installed_faces', lambda: (
        ('Noto Sans SC', True, False, str(path), 0),
    ))
    font_match.match_text_face.cache_clear()
    font = ImageFont.truetype(str(path), size)
    font.set_variation_by_name('Bold')
    image = Image.new('RGB', (100, 60), '#182838')
    draw = ImageDraw.Draw(image)
    draw.text((12, 5), 'DUE', font=font, fill='white')
    draw.line((0, 0, 0, 59), fill='white')
    result = font_match.match_text_face(image.tobytes(), 100, 60, 'DUE')
    assert result is not None
    assert abs(result['font_size_px'] - size) < 3
    assert result['ink_box'][0] > 1
    font_match.match_text_face.cache_clear()


def test_plain_font_refinement_preserves_styled_and_rotated_text(monkeypatch):
    def unexpected(*args):
        raise AssertionError('Art text must retain its own layout')

    monkeypatch.setattr(font_match, 'match_text_face', unexpected)
    items = [
        {'text': 'Editable', 'box': [0, 0, 100, 50], 'runs': [{'text': 'Editable'}]},
        {'text': 'Rotated', 'box': [0, 0, 100, 50], 'rotation': 90},
        {'text': 'Italic', 'box': [0, 0, 100, 50], 'italic': True},
        {'text': 'Outlined', 'box': [0, 0, 100, 50], 'outline_width': 2},
        {'text': 'Native', 'box': [0, 0, 100, 50], 'font_size_pt': 18},
    ]
    assert text_detect.refine_plain_text_fonts(np.full((50, 100, 3), 240, np.uint8), items) == items
