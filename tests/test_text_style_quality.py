"""Raster font-weight checks using known regular and bold font files."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from scripts.text_detect import _build_text_result, _estimate_bold, _estimate_style, _sample_text_color


def _font_path(family: str, bold: bool) -> Path:
    if family == "latin":
        candidates = [
            Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
            Path("/usr/share/fonts/truetype/liberation2")
            / ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"),
        ]
    else:
        candidates = [
            Path("C:/Windows/Fonts") / ("msyhbd.ttc" if bold else "msyh.ttc"),
            Path("/usr/share/fonts/opentype/noto")
            / ("NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"),
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip(f"No known {family} font fixture installed")


def _render_text(family, text, size, padding, background, foreground, bold,
                 resampling=Image.Resampling.LANCZOS):
    scale = 4
    font = ImageFont.truetype(str(_font_path(family, bold)), size * scale)
    left, top, right, bottom = font.getbbox(text)
    margin = padding * scale
    image = Image.new(
        "RGB", (right - left + 2 * margin, bottom - top + 2 * margin), background,
    )
    ImageDraw.Draw(image).text(
        (margin - left, margin - top), text, font=font, fill=foreground,
    )
    return image.resize(
        (image.width // scale, image.height // scale), resampling,
    )


@pytest.mark.parametrize("family,text", [
    ("latin", "Consistent body text"),
    ("latin", "Accuracy"),
    ("latin", "Regular and bold fonts"),
    ("cjk", "\u80bf\u7624\u6cbb\u7597\u7684\u51c6\u786e\u6027\u8fde\u7eed\u6027\u548c\u53ef\u8ffd\u6eaf\u6027"),
    ("cjk", "\u6d4b\u8bd5"),
    ("cjk", "\u6587\u5b57\u6392\u7248\u4e0e\u8bc6\u522b"),
])
@pytest.mark.parametrize("size", [12, 16, 20, 24, 36, 60])
@pytest.mark.parametrize("padding", [2, 6])
@pytest.mark.parametrize("background,foreground", [
    ((255, 255, 255), (0, 0, 0)),
    ((240, 244, 238), (80, 100, 90)),
    ((35, 45, 40), (235, 245, 240)),
])
@pytest.mark.parametrize("bold", [False, True])
def test_known_font_weight_survives_antialiasing_scale_and_background(
    family, text, size, padding, background, foreground, bold,
):
    image = _render_text(
        family, text, size, padding, background, foreground, bold,
    )

    assert _estimate_bold(np.asarray(image), text=text) is bold


@pytest.mark.parametrize("family,text,size,bold", [
    ("latin", "Consistent body text", 16, False),
    ("latin", "Accuracy", 12, True),
    ("cjk", "\u6d4b\u8bd5", 36, True),
])
def test_ocr_result_and_recovery_style_use_recognized_text_for_weight(
    family, text, size, bold,
):
    image = _render_text(family, text, size, 2, "white", "black", bold)
    pixels = np.asarray(image)
    box = (0, 0, image.width, image.height)
    style = _estimate_style(pixels, box, text=text)
    items, _ = _build_text_result(
        pixels, [{"text": text, "box": box, "confidence": 1.0}], 0.7, 2,
    )
    assert style["bold"] is bold
    assert len(items) == 1
    assert items[0]["bold"] is bold


def test_unavailable_reference_fonts_preserve_existing_pixel_estimate(monkeypatch):
    from scripts import text_detect

    image = np.full((20, 100, 3), 255, dtype=np.uint8)
    image[5:15, 10:90] = 0
    expected = _estimate_bold(image)
    monkeypatch.setattr(text_detect, "_weight_reference_font", lambda *args: None)
    assert _estimate_bold(image, text="Example") is expected


@pytest.mark.parametrize("background", [0, 128, 255])
def test_blank_region_is_not_bold(background):
    pixels = np.full((20, 100, 3), background, dtype=np.uint8)
    assert not _estimate_bold(pixels, text="Example")


@pytest.mark.parametrize("size", [12, 16, 24])
@pytest.mark.parametrize("resampling,tolerance", [
    (Image.Resampling.BOX, 12),
    # Sharpening introduces overshoot beyond the original ink color.
    (Image.Resampling.LANCZOS, 25),
])
@pytest.mark.parametrize("background,foreground", [
    ((255, 255, 255), (0, 0, 0)),
    ((240, 244, 238), (80, 100, 90)),
    ((35, 45, 40), (235, 245, 240)),
    ((255, 255, 255), (180, 180, 180)),
])
def test_text_color_uses_ink_not_antialiased_edges(
    size, background, foreground, resampling, tolerance,
):
    image = _render_text(
        "latin", "Quarterly report", size, 4, background, foreground, False,
        resampling=resampling,
    )
    color = _sample_text_color(np.asarray(image))
    actual = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    assert actual == pytest.approx(foreground, abs=tolerance)


@pytest.mark.parametrize("foreground", [(80, 100, 90), (20, 80, 160)])
@pytest.mark.parametrize("inset", [0, 1, 3])
@pytest.mark.parametrize("vertical", [False, True])
def test_text_color_ignores_crossing_black_table_line(foreground, inset, vertical):
    image = _render_text("latin", "Accuracy", 16, 10, "white", foreground, False)
    line = ((2, inset, 2, image.height - 1 - inset) if vertical else
            (inset, 2, image.width - 1 - inset, 2))
    ImageDraw.Draw(image).line(line, fill="black")
    color = _sample_text_color(np.asarray(image))
    actual = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    assert actual == pytest.approx(foreground, abs=25)


def test_text_color_retains_single_vertical_glyph():
    image = _render_text("latin", "I", 24, 2, "white", (20, 80, 160), False,
                         resampling=Image.Resampling.BOX)
    color = _sample_text_color(np.asarray(image))
    actual = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    assert actual == pytest.approx((20, 80, 160), abs=12)


@pytest.mark.parametrize("family,text", [("latin", "Quarterly report"), ("cjk", "\u6587\u5b57\u6392\u7248")])
@pytest.mark.parametrize("size", [10, 16, 32])
@pytest.mark.parametrize("padding", [2, 10])
def test_font_size_tracks_glyphs_not_ocr_padding(family, text, size, padding):
    image = _render_text(family, text, size, padding, "white", "black", False)
    style = _estimate_style(
        np.asarray(image), (0, 0, image.width, image.height),
        reference_width=960, text=text,
    )
    assert style["font_size"] == pytest.approx(size, rel=0.12, abs=1.0)


@pytest.mark.parametrize("foreground", ["black", (180, 180, 180)])
@pytest.mark.parametrize("inset", [0, 3])
def test_font_height_ignores_crossing_table_line(foreground, inset):
    text = "Accuracy"
    image = _render_text("latin", text, 16, 10, "white", foreground, False)
    ImageDraw.Draw(image).line((inset, 2, image.width - 1 - inset, 2), fill="black")
    pixels = np.asarray(image)
    box = (0, 0, image.width, image.height)
    style = _estimate_style(pixels, box, reference_width=960, text=text)
    assert style["font_size"] == pytest.approx(16, rel=0.12)
    items, _ = _build_text_result(
        pixels, [{"text": text, "box": box, "confidence": 1.0}], 0.7, 2,
        style_reference_width=960,
    )
    assert len(items) == 1


def test_line_only_crop_keeps_legacy_font_size_fallback():
    pixels = np.full((35, 100, 3), 255, dtype=np.uint8)
    pixels[2, 5:95] = 0
    box = (0, 0, 100, 35)
    fallback = _estimate_style(pixels, box, reference_width=960)
    measured = _estimate_style(pixels, box, reference_width=960, text="Accuracy")
    assert measured["font_size"] == fallback["font_size"]


@pytest.mark.parametrize("confidence,expected_count", [(0.99, 1), (0.75, 0)])
def test_small_label_retention_depends_on_ocr_confidence(confidence, expected_count):
    text = "Source: annual report"
    image = _render_text("latin", text, 6, 2, "white", "black", False)
    items, _ = _build_text_result(
        np.asarray(image), [{"text": text, "box": (0, 0, image.width, image.height),
                            "confidence": confidence}], 0.7, 2,
        style_reference_width=960,
    )
    assert len(items) == expected_count


def test_targeted_recovery_retains_known_font_weight(tmp_path, monkeypatch):
    import image_to_ppt

    text = "Consistent body text"
    image = _render_text("latin", text, 16, 2, "white", "black", False)
    page = Image.new("RGB", (500, 150), "white")
    page.paste(image, (20, 20))
    source = tmp_path / "source.png"
    page.save(source)

    def fake_batch(paths, **kwargs):
        results = []
        for path in paths:
            with Image.open(path) as crop:
                results.append(([{
                    "text": text, "confidence": 1.0,
                    "box": [0, 0, crop.width, crop.height],
                }], np.zeros((crop.height, crop.width), dtype=np.uint8)))
        return results

    monkeypatch.setattr(image_to_ppt, "detect_text_batch", fake_batch)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)
    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source, [{
            "x": 20, "y": 20, "w": image.width, "h": image.height,
            "area": image.width * image.height,
        }], [], np.zeros((page.height, page.width), dtype=np.uint8),
        tmp_path, lang="en", isolated=True,
    )
    assert len(result["recovered_items"]) == 1
    assert result["recovered_items"][0]["bold"] is False
