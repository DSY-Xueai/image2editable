from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest


def test_art_metrics_resolve_installed_cjk_family():
    from scripts.art_text import _font_metrics
    if not Path("C:/Windows/Fonts/simhei.ttf").exists():
        pytest.skip("Requires SimHei")
    font = _font_metrics("SimHei")
    assert font is not None
    assert font.getname()[0] == "SimHei"


@pytest.mark.parametrize("angle", [-20, 15])
def test_rotated_known_font_preserves_native_angle_and_size(angle):
    from scripts.art_text import estimate_art_text_runs
    font_path = Path("C:/Windows/Fonts/arialbd.ttf")
    if not font_path.exists():
        pytest.skip("Requires Arial")
    glyph = Image.new("RGBA", (160, 170))
    ImageDraw.Draw(glyph).text((35, 15), "R", font=ImageFont.truetype(str(font_path), 100),
                              fill=(95, 200, 210), stroke_width=3, stroke_fill=(40, 30, 20))
    glyph = glyph.rotate(-angle, Image.Resampling.BICUBIC, expand=True)
    image = Image.new("RGB", glyph.size, (235, 201, 164))
    image.paste(glyph, (0, 0), glyph)
    item = {"text": "R", "font": "Arial", "box": [0, 0, *image.size],
            "words": [{"text": "R", "box": [0, 0, 1, 1]}]}
    runs = estimate_art_text_runs(np.asarray(image), item, reference_width=960)
    assert runs is not None
    assert abs(runs[0].get("rotation", 0) - angle) <= 2
    assert abs(runs[0]["font_size"] - 100) <= 8
    assert runs[0]["font"] == "Arial"


def test_matching_uses_installed_cjk_face_instead_of_default_font():
    from scripts.font_match import match_glyph, resolve_font
    font = resolve_font("KaiTi", size=100)
    if font is None:
        pytest.skip("Requires KaiTi")
    image = Image.new("L", (160, 160))
    ImageDraw.Draw(image).text((25, 10), "图", font=font, fill=255)
    image = image.rotate(-10, Image.Resampling.BICUBIC, expand=True)
    result = match_glyph(np.asarray(image) > 127, "图", "Arial")
    assert result is not None
    assert result["font"] == "KaiTi"
    assert result["bold"] is False
    assert abs(result["rotation"] - 10) <= 2
    assert result["fit_iou"] > .85


def test_grouped_letters_with_different_angles_get_individual_native_runs(art_font):
    from scripts.art_text import estimate_art_text_runs
    image = Image.new("RGB", (350, 220), (235, 201, 164))
    font = art_font(100)
    for text, angle, left in (("R", -20, 0), ("B", 15, 160)):
        glyph = Image.new("RGBA", (150, 150))
        ImageDraw.Draw(glyph).text((35, 10), text, font=font, fill=(95, 200, 210),
                                  stroke_width=3, stroke_fill=(40, 30, 20))
        glyph = glyph.rotate(-angle, Image.Resampling.BICUBIC, expand=True)
        image.paste(glyph, (left, 0), glyph)
    item = {"text": "RB", "font": font.getname()[0], "box": [0, 0, *image.size],
            "words": [{"text": "RB", "box": [0, 0, 1, 1]}]}
    runs = estimate_art_text_runs(np.asarray(image), item, reference_width=960)
    assert runs is not None
    assert [run["text"] for run in runs] == ["R", "B"]
    assert abs(runs[0]["rotation"] + 20) <= 2
    assert abs(runs[1]["rotation"] - 15) <= 2


def test_centered_outline_recovers_font_before_stroke_occlusion(art_font):
    import cv2
    from scripts.art_text import estimate_art_text_runs
    glyph = Image.new("L", (180, 170))
    font = art_font(110)
    ImageDraw.Draw(glyph).text((35, 10), "H", font=font, fill=255)
    base = np.asarray(glyph) > 127
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fill = cv2.erode(base.astype(np.uint8), kernel) > 0
    ink = cv2.dilate(base.astype(np.uint8), kernel) > 0
    pixels = np.full((170, 180, 3), (235, 201, 164), dtype=np.uint8)
    pixels[ink] = (40, 30, 20)
    pixels[fill] = (95, 200, 210)
    item = {"text": "H", "font": font.getname()[0], "box": [0, 0, 180, 170],
            "words": [{"text": "H", "box": [0, 0, 1, 1]}]}
    runs = estimate_art_text_runs(pixels, item, reference_width=960)
    assert runs is not None
    run = runs[0]
    assert run["font"] == font.getname()[0] and run["bold"]
    assert abs(run["font_size"] - 110) < 4
    assert 3 <= run["outline_width"] <= 5
