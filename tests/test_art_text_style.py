from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest


def _outlined_line():
    font_path = Path("C:/Windows/Fonts/arialbd.ttf")
    if not font_path.exists():
        pytest.skip("Requires a known outline font")
    image = Image.new("RGB", (250, 120), (235, 201, 164))
    font = ImageFont.truetype(str(font_path), 66)
    draw = ImageDraw.Draw(image)
    draw.text((20, 25), "A", font=font, fill=(95, 200, 210), stroke_width=3, stroke_fill=(40, 30, 20))
    draw.text((150, 5), "B", font=font, fill=(230, 130, 160), stroke_width=3, stroke_fill=(40, 30, 20))
    item = {"text": "AB", "box": [0, 0, 250, 120], "font": "Arial", "font_size": 50,
        "words": [{"text": "A", "box": [0, 0, 0.45, 1]},
                  {"text": "B", "box": [0.5, 0, 0.5, 1]}]}
    return np.asarray(image).copy(), item


def test_extracts_fill_and_outline_instead_of_using_outline_as_text_color():
    from scripts.art_text import estimate_art_text_runs
    pixels, item = _outlined_line()
    runs = estimate_art_text_runs(pixels, item, reference_width=960)
    assert runs is not None
    assert [run["text"] for run in runs] == ["A", "B"]
    for run, expected in zip(runs, ((95, 200, 210), (230, 130, 160))):
        assert run["box_kind"] == "ink"
        color = [int(run["color"][i:i + 2], 16) for i in (1, 3, 5)]
        assert np.max(np.abs(np.array(color) - expected)) < 8
        assert run["outline_color"] == "#281e14"
        # The fitted glyph lies at the center of the visible outline, so
        # DrawingML's full line width equals the measured dark band.
        assert 2 <= run["outline_width"] <= 4
    assert runs[0]["box"][1] > runs[1]["box"][1]


def test_plain_text_does_not_require_positioned_runs():
    from scripts.art_text import estimate_art_text_runs
    pixels, item = _outlined_line()
    pixels[:] = (255, 255, 255)
    assert estimate_art_text_runs(pixels, item, reference_width=960) is None
    assert estimate_art_text_runs(pixels, {"box": [0, 0, 250, 120], "text": "AB"}, reference_width=960) is None


def test_cleanup_removes_light_fill_inside_dark_letter_outline():
    import image_to_ppt
    from scripts.text_detect import _build_text_mask
    image = Image.new("RGB", (220, 150), (232, 164, 161))
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 110)
    ImageDraw.Draw(image).text((25, 5), "HI", font=font, fill=(245, 225, 146),
                              stroke_width=5, stroke_fill=(90, 48, 27))
    pixels = np.asarray(image)
    item = {"text": "HI", "box": [15, 15, 170, 120], "color": "#5a301b", "font_size": 60}
    mask = image_to_ppt._build_text_cleanup_mask(
        pixels, _build_text_mask(pixels.shape[:2], [item]), [item])
    fill = np.all(pixels == (245, 225, 146), axis=2)
    assert np.count_nonzero(fill) > 100
    assert np.all(mask[fill] > 0)


def test_cleanup_includes_attached_letter_shadow():
    import image_to_ppt
    from scripts.text_detect import _build_text_mask
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 100)
    image = Image.new("RGB", (240, 180), (235, 170, 165))
    draw = ImageDraw.Draw(image)
    draw.text((40, 39), "H", font=font, fill=(232, 161, 155), stroke_width=4,
              stroke_fill=(232, 161, 155))
    shadow = np.all(np.asarray(image) == (232, 161, 155), axis=2)
    draw.text((36, 27), "H", font=font, fill=(250, 225, 150), stroke_width=4,
              stroke_fill=(60, 30, 20))
    pixels = np.asarray(image)
    shadow &= np.all(pixels == (232, 161, 155), axis=2)
    item = {"text": "H", "box": [28, 35, 110, 115], "color": "#3c1e14", "font_size": 80}
    mask = image_to_ppt._build_text_cleanup_mask(pixels, _build_text_mask(pixels.shape[:2], [item]), [item])
    assert np.count_nonzero(shadow) > 100
    assert np.mean(mask[shadow] > 0) > .98


def test_narrow_ocr_word_boxes_do_not_clip_letter_outlines():
    from scripts.art_text import estimate_art_text_runs
    pixels, item = _outlined_line()
    item["words"] = [{"text": "A", "box": [.12, .3, .08, .55]},
                     {"text": "B", "box": [.64, .15, .08, .55]}]
    runs = estimate_art_text_runs(pixels, item, reference_width=960)
    assert runs is not None
    assert runs[0]["box"][0] < .10
    assert runs[0]["box"][2] > .16
    assert runs[1]["box"][2] > .16
    assert runs[0]["color"] == "#5fc8d2"
    assert runs[1]["color"] == "#e682a0"


def test_colored_image_border_does_not_become_text_fill():
    from scripts.art_text import estimate_art_text_runs
    pixels, item = _outlined_line()
    pixels[:5] = (180, 220, 250)
    pixels[-5:] = (180, 220, 250)
    runs = estimate_art_text_runs(pixels, item, reference_width=960)
    assert runs is not None
    assert [run["color"] for run in runs] == ["#5fc8d2", "#e682a0"]


def test_enclosed_counter_is_background_not_glyph_fill():
    from scripts.art_text import estimate_art_text_runs
    image = Image.new("RGB", (140, 120), (235, 201, 164))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 20, 108, 100), fill=(40, 30, 20))
    draw.rectangle((34, 24, 104, 96), fill=(230, 130, 160))
    draw.rectangle((42, 32, 96, 88), fill=(40, 30, 20))
    draw.rectangle((46, 36, 92, 84), fill=(235, 201, 164))
    item = {"text": "O", "box": [0, 0, 140, 120], "words": [{"text": "O", "box": [0, 0, 1, 1]}]}
    runs = estimate_art_text_runs(np.asarray(image), item, reference_width=960)
    assert runs is not None
    assert runs[0]["color"] == "#e682a0"


def test_nested_letter_strokes_compare_to_external_background():
    from scripts.art_text import _line_ink
    image = Image.new("RGB", (150, 140), (235, 170, 165))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 15, 130, 125), fill=(60, 30, 20))
    draw.rectangle((24, 19, 126, 121), fill=(250, 225, 150))
    draw.rectangle((46, 41, 104, 99), fill=(60, 30, 20))
    draw.rectangle((50, 45, 100, 95), fill=(235, 170, 165))
    draw.rectangle((55, 50, 95, 80), fill=(60, 30, 20))
    draw.rectangle((59, 54, 91, 76), fill=(250, 225, 150))
    pixels = np.asarray(image)
    fill, _, _ = _line_ink(pixels)
    assert np.all(fill[56:74, 61:89])
    assert not np.any(fill[85:94, 52:98])


def test_pastel_fill_components_are_not_dropped_by_global_threshold():
    from scripts.art_text import _line_ink
    image = Image.new("RGB", (600, 150), (252, 253, 248))
    font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 100)
    draw = ImageDraw.Draw(image)
    draw.text((15, 5), "哪些玩具是", font=font, fill=(243, 169, 176),
              stroke_width=4, stroke_fill=(136, 89, 78))
    pixels = np.asarray(image)
    expected = np.all(pixels == (243, 169, 176), axis=2).astype(np.uint8)
    import cv2
    expected = cv2.erode(expected, np.ones((3, 3), np.uint8)) > 0
    measured = _line_ink(pixels)
    assert measured is not None
    assert np.mean(measured[0][expected]) > .98


def test_faint_enclosed_decoration_does_not_become_glyph_fill():
    from scripts.art_text import _line_ink
    image = Image.new("RGB", (300, 150), (250, 200, 170))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 75, 100), fill=(136, 89, 78))
    draw.rectangle((25, 25, 70, 95), fill=(243, 169, 176))
    draw.rectangle((170, 90, 230, 140), fill=(211, 135, 138))
    draw.rectangle((175, 95, 225, 135), fill=(234, 153, 158))
    fill, _, _ = _line_ink(np.asarray(image))
    assert np.all(fill[30:90, 30:65])
    assert not np.any(fill[100:130, 180:220])


def test_word_crops_cutting_outline_use_full_line_context():
    from scripts.art_text import estimate_art_text_runs
    pixels, item = _outlined_line()
    item["words"][0]["box"] = [0.12, 0, 0.11, 1]
    item["words"][1]["box"] = [0.65, 0, 0.12, 1]
    runs = estimate_art_text_runs(pixels, item, reference_width=960)
    assert runs is not None
    assert runs[0]["color"] == "#5fc8d2"
    assert runs[1]["color"] == "#e682a0"


def test_neighbor_outline_does_not_inflate_short_glyph_box():
    from scripts.art_text import _measure_ink
    region = np.full((100, 50, 3), (235, 201, 164), dtype=np.uint8)
    fill = np.zeros((100, 50), bool)
    fill[10:25, 10:25] = True
    dark = np.zeros_like(fill)
    dark[7:28, 7:28] = True
    dark[fill] = False
    dark[40:95, 48:] = True
    region[dark] = (40, 30, 20)
    region[fill] = (230, 130, 160)
    bounds, *_ = _measure_ink(region, fill, fill | dark, dark)
    assert bounds[1] + bounds[3] < 35


def test_grouped_punctuation_gets_individual_pixel_positions():
    from scripts.art_text import estimate_art_text_runs
    image = Image.new("RGB", (140, 120), (235, 201, 164))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 66)
    draw.text((15, 10), "!", font=font, fill=(230, 130, 160), stroke_width=3, stroke_fill=(40, 30, 20))
    draw.text((90, 30), "!", font=font, fill=(230, 130, 160), stroke_width=3, stroke_fill=(40, 30, 20))
    item = {"text": "!!", "box": [0, 0, 140, 120], "words": [{"text": "!!", "box": [0, 0, 1, 1]}]}
    runs = estimate_art_text_runs(np.asarray(image), item, reference_width=960)
    assert runs is not None
    assert [run["text"] for run in runs] == ["!", "!"]
    assert runs[0]["box"][1] < runs[1]["box"][1]


def test_cjk_horizontal_stroke_uses_font_metrics_not_ink_height():
    from scripts.art_text import estimate_art_text_runs
    image = Image.new("RGB", (160, 100), (235, 201, 164))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 80)
    draw.text((25, 0), "\u4e00", font=font, fill=(95, 200, 210), stroke_width=3, stroke_fill=(40, 30, 20))
    item = {"text": "\u4e00", "font": "Microsoft YaHei", "box": [0, 0, 160, 100],
            "words": [{"text": "\u4e00", "box": [0, 0, 1, 1]}]}
    runs = estimate_art_text_runs(np.asarray(image), item, reference_width=960)
    assert runs is not None
    assert 65 <= runs[0]["font_size"] <= 95


def test_solid_text_with_counters_is_not_mistaken_for_outlined_text():
    from scripts.art_text import estimate_art_text_runs
    pixels, item = _outlined_line()
    image = Image.fromarray(pixels)
    ImageDraw.Draw(image).rectangle((0, 0, 250, 120), fill=(235, 201, 164))
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 66)
    ImageDraw.Draw(image).text((20, 25), "AB", font=font, fill=(40, 30, 20))
    assert estimate_art_text_runs(np.asarray(image), item, reference_width=960) is None
