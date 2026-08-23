from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import numpy as np
from PIL import Image, ImageDraw
import pytest

import image_to_ppt
from image2editable import legacy
from image2editable.component_quality import evaluate_page_quality
from image2editable.store import RunStore
from scripts.initial_diagnostics import validate_initial_diagnostics


@pytest.fixture(autouse=True)
def _batch_ocr_uses_the_test_single_image_detector(monkeypatch):
    monkeypatch.setattr(
        image_to_ppt,
        "detect_text_batch",
        lambda paths, **kwargs: [
            image_to_ppt.detect_text(path, **kwargs) for path in paths
        ],
    )


def _label_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "generic-technical-label.png"
    image = Image.new("RGB", (120, 70), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 20, 90, 50), radius=5, fill="#1769aa")
    draw.text((55, 29), "NX", fill="white")
    image.save(path)
    return path


def _component(index: int = 0, *, x: int = 40, y: int = 20,
               w: int = 50, h: int = 30) -> dict:
    return {
        "path": f"component_{index:04d}.png",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": w * h,
        "z_index": index,
    }


@pytest.mark.parametrize(
    ("rotation", "box", "expected_box", "inverse_turns"),
    [
        (90, [1, 2, 3, 4], [2, 1, 4, 3], 3),
        (180, [1, 1, 3, 2], [4, 3, 3, 2], 2),
        (270, [1, 2, 3, 4], [2, 2, 4, 3], 1),
    ],
)
def test_restore_rotated_ocr_analysis_maps_items_and_mask_to_display(
    rotation: int,
    box: list[int],
    expected_box: list[int],
    inverse_turns: int,
) -> None:
    source_size = (8, 6)
    upright_size = source_size[::-1] if rotation in {90, 270} else source_size
    upright_mask = np.zeros((upright_size[1], upright_size[0]), dtype=np.uint8)
    upright_mask[1:3, 1:4] = 255

    items, mask = image_to_ppt._restore_rotated_ocr_analysis(
        [{"text": "vertical", "box": box}],
        upright_mask,
        rotation=rotation,
        source_size=source_size,
    )

    assert items == [{
        "text": "vertical",
        "box": expected_box,
        "rotation": rotation,
    }]
    assert np.array_equal(mask, np.rot90(upright_mask, k=inverse_turns))


def _ocr_item(text: str, confidence: float, scale: int) -> dict:
    return {
        "box": [5 * scale, 4 * scale, 35 * scale, 9 * scale],
        "text": text,
        "font_size": 18.0,
        "color": "#ffffff",
        "bold": False,
        "font": "Arial",
        "align": 1,
        "confidence": confidence,
    }


def test_targeted_ocr_recovers_consistent_candidate_with_source_style_and_mask(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _label_fixture(tmp_path)
    calls = []
    style_calls = []

    def fake_detect(path, **kwargs):
        with Image.open(path) as crop:
            scale = crop.width // 50
            calls.append((crop.size, kwargs))
            return [_ocr_item("NX", 0.89, scale)], np.zeros(
                (crop.height, crop.width), dtype=np.uint8
            )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)
    monkeypatch.setattr(
        image_to_ppt, "_load_rgb",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("recovery must not load the full page RGB")
        ),
    )
    monkeypatch.setattr(
        image_to_ppt.text_detection,
        "_estimate_style",
        lambda pixels, box, *, reference_width: (
            style_calls.append((pixels.shape, box, reference_width))
            or {"font_size": 22.0, "color": "#fefefe", "bold": True}
        ),
    )

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        [_component()],
        [],
        np.zeros((70, 120), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
    )

    assert [size for size, _ in calls] == [(100, 60), (150, 90)]
    assert all(call["isolated"] is True for _, call in calls)
    assert result["diagnostics"] == []
    assert len(result["recovered_items"]) == 1
    recovered = result["recovered_items"][0]
    assert recovered["text"] == "NX"
    assert recovered["box"] == [45, 24, 35, 9]
    assert recovered["color"].lower() == "#fefefe"
    assert recovered["bold"] is True
    assert recovered["font"] == "Arial"
    assert recovered["align"] == 1
    assert style_calls == [((21, 47, 3), (6, 6, 35, 9), 120)]
    assert np.all(result["text_mask"][24:33, 45:80] == 255)


def test_targeted_ocr_rotates_views_and_matches_existing_vertical_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _label_fixture(tmp_path)
    crop_sizes = []
    known = [{
        **_ocr_item("NX", 0.99, 1),
        "box": [77, 25, 9, 35],
        "rotation": 90,
    }]

    def fake_detect(path, **kwargs):
        with Image.open(path) as crop:
            crop_sizes.append(crop.size)
            scale = crop.width // 30
            return [_ocr_item("NX", 0.96, scale)], np.zeros(
                (crop.height, crop.width), dtype=np.uint8
            )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        [_component()],
        known,
        np.zeros((70, 120), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
        ocr_rotation=90,
    )

    assert crop_sizes == [(60, 100), (90, 150)]
    assert result["items"] == known
    assert result["recovered_items"] == []
    assert result["diagnostics"] == []


def test_targeted_ocr_sizes_rotated_text_against_display_page_width(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _label_fixture(tmp_path)
    style_calls = []

    def fake_detect(path, **kwargs):
        with Image.open(path) as crop:
            scale = crop.width // 30
            return [_ocr_item("NX", 0.96, scale)], np.zeros(
                (crop.height, crop.width), dtype=np.uint8
            )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)
    monkeypatch.setattr(
        image_to_ppt.text_detection,
        "_estimate_style",
        lambda pixels, box, *, reference_width: (
            style_calls.append(reference_width)
            or {"font_size": 18.0, "color": "#ffffff", "bold": False}
        ),
    )

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        [_component()],
        [],
        np.zeros((70, 120), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
        ocr_rotation=90,
    )

    assert len(result["recovered_items"]) == 1
    assert style_calls == [120]


def test_targeted_ocr_sizes_recovered_text_against_page_width(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _label_fixture(tmp_path)

    def fake_detect(path, **kwargs):
        with Image.open(path) as crop:
            scale = crop.width // 50
            return [_ocr_item("NX", 0.89, scale)], np.zeros(
                (crop.height, crop.width), dtype=np.uint8
            )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        [_component()],
        [],
        np.zeros((70, 120), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
    )

    expected = image_to_ppt.text_detection._estimate_style(
        np.zeros((21, 120, 3), dtype=np.uint8),
        (6, 6, 35, 9),
    )["font_size"]
    assert result["recovered_items"][0]["font_size"] == expected


def test_estimate_style_default_reference_width_preserves_full_image_behavior() -> None:
    pixels = np.zeros((12, 120, 3), dtype=np.uint8)

    default = image_to_ppt.text_detection._estimate_style(pixels, (1, 1, 5, 5))
    explicit = image_to_ppt.text_detection._estimate_style(
        pixels,
        (1, 1, 5, 5),
        reference_width=120,
    )

    assert default == explicit


@pytest.mark.parametrize("terminal", [".", "!", "?"])
def test_build_text_result_preserves_terminal_semantic_punctuation(
    terminal: str,
) -> None:
    items, _ = image_to_ppt.text_detection._build_text_result(
        np.zeros((120, 400, 3), dtype=np.uint8),
        [{
            "box": (20, 20, 180, 50),
            "text": f"EDITABLE{terminal}",
            "confidence": 0.99,
        }],
        0.7,
        6,
    )

    assert items[0]["text"] == f"EDITABLE{terminal}"


def test_build_text_result_recovers_adjacent_large_heading_period() -> None:
    pixels = np.full((120, 300, 3), (8, 13, 22), dtype=np.uint8)
    pixels[30:60, 30:150] = (240, 244, 248)
    pixels[56:68, 184:196] = (240, 244, 248)

    items, mask = image_to_ppt.text_detection._build_text_result(
        pixels,
        [{
            "box": (20, 20, 160, 50),
            "text": "EDITABLE",
            "confidence": 0.99,
        }],
        0.7,
        6,
    )

    assert items[0]["text"] == "EDITABLE."
    assert items[0]["box"] == [20, 20, 176, 50]
    assert mask[60, 190] == 255


@pytest.mark.parametrize(
    ("text", "dot_top"),
    [("Editable", 56), ("EDITABLE", 30)],
)
def test_build_text_result_does_not_invent_heading_period(
    text: str,
    dot_top: int,
) -> None:
    pixels = np.full((120, 300, 3), (8, 13, 22), dtype=np.uint8)
    pixels[30:60, 30:150] = (240, 244, 248)
    pixels[dot_top:dot_top + 12, 184:196] = (240, 244, 248)

    items, _ = image_to_ppt.text_detection._build_text_result(
        pixels,
        [{
            "box": (20, 20, 160, 50),
            "text": text,
            "confidence": 0.99,
        }],
        0.7,
        6,
    )

    assert items[0]["text"] == text
    assert items[0]["box"] == [20, 20, 160, 50]


def test_adjust_font_size_uses_bbox_for_large_uppercase_latin_heading() -> None:
    adjusted = image_to_ppt.text_detection._adjust_font_size(
        "MAKE IT",
        34.9,
        bbox_height=83,
        reference_width=1600,
    )

    assert adjusted == 53.8
    assert image_to_ppt.text_detection._adjust_font_size(
        "Release Notes",
        34.9,
        bbox_height=83,
        reference_width=1600,
    ) == 34.9
    assert image_to_ppt.text_detection._adjust_font_size(
        "RELEASE",
        20.0,
        bbox_height=25,
        reference_width=1600,
    ) == 20.0


def test_detect_text_uses_explicit_display_width_for_style_estimation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "upright.png"
    Image.new("RGB", (200, 400), "white").save(source)
    style_calls = []
    monkeypatch.setattr(
        image_to_ppt.text_detection,
        "_ocr_detect",
        lambda *args, **kwargs: [{
            "box": (20, 30, 100, 20),
            "text": "Vertical text",
            "confidence": 0.99,
        }],
    )
    monkeypatch.setattr(
        image_to_ppt.text_detection,
        "_estimate_style",
        lambda pixels, box, *, reference_width=None: (
            style_calls.append(reference_width)
            or {"font_size": 18.0, "color": "#000000", "bold": False}
        ),
    )

    items, _ = image_to_ppt.text_detection.detect_text(
        source,
        lang="en",
        style_reference_width=400,
    )

    assert len(items) == 1
    assert style_calls == [400]


@pytest.mark.parametrize("reference_width", [0, -1, True, 120.0])
def test_build_text_result_rejects_invalid_style_width_without_ocr_items(
    reference_width,
) -> None:
    with pytest.raises(ValueError, match="reference_width must be a positive integer"):
        image_to_ppt.text_detection._build_text_result(
            np.zeros((12, 120, 3), dtype=np.uint8),
            [],
            0.7,
            6,
            style_reference_width=reference_width,
        )


@pytest.mark.parametrize("reference_width", [1, 10**9])
def test_estimate_style_accepts_positive_integer_reference_width(
    reference_width: int,
) -> None:
    style = image_to_ppt.text_detection._estimate_style(
        np.zeros((12, 120, 3), dtype=np.uint8),
        (1, 1, 5, 5),
        reference_width=reference_width,
    )

    assert 6.0 <= style["font_size"] <= 200.0


@pytest.mark.parametrize("reference_width", [0, -1, True, 120.0])
def test_estimate_style_rejects_invalid_reference_width(reference_width) -> None:
    with pytest.raises(ValueError, match="reference_width must be a positive integer"):
        image_to_ppt.text_detection._estimate_style(
            np.zeros((12, 120, 3), dtype=np.uint8),
            (1, 1, 5, 5),
            reference_width=reference_width,
        )


def test_targeted_ocr_large_candidate_uses_two_distinct_bounded_views(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "large-candidate.png"
    image = Image.new("RGB", (2400, 800), "white")
    draw = ImageDraw.Draw(image)
    for offset in range(0, 518, 8):
        draw.line((100 + offset, 100, 100 + offset, 219), fill=(offset % 255, 80, 160))
    image.save(source)
    views = []

    def fake_detect(path, **kwargs):
        with Image.open(path) as crop:
            pixels = np.asarray(crop.convert("RGB")).copy()
            views.append((crop.size, pixels, Path(path).read_bytes()))
            return [{
                **_ocr_item("NX", 0.96, 1),
                "box": [0, 0, crop.width, crop.height],
            }], np.zeros((crop.height, crop.width), dtype=np.uint8)

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)
    monkeypatch.setattr(
        image_to_ppt.text_detection,
        "_estimate_style",
        lambda pixels, box, *, reference_width: {
            "font_size": 18.0, "color": "#ffffff", "bold": False,
        },
    )

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        [_component(x=100, y=100, w=518, h=120)],
        [],
        np.zeros((800, 2400), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
    )

    assert [view[0][0] for view in views] == [512, 448]
    assert views[0][0] != views[1][0]
    assert not np.array_equal(views[0][1], views[1][1])
    assert views[0][2] != views[1][2]
    assert len(result["recovered_items"]) == 1


def test_targeted_ocr_matches_multiple_items_by_space_and_filters_known_line(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "multi-item.png"
    Image.new("RGB", (400, 200), "white").save(source)
    known = [{**_ocr_item("A-B", 0.99, 1), "box": [30, 25, 60, 10]}]
    known_mask = np.zeros((200, 400), dtype=np.uint8)
    known_mask[20:45, 20:140] = 255
    calls = 0

    def item(text: str, confidence: float, box: list[int], scale: int) -> dict:
        return {**_ocr_item(text, confidence, 1),
                "box": [value * scale for value in box]}

    def fake_detect(path, **kwargs):
        nonlocal calls
        scale = 2 + calls
        first = [
            item("A-B", 0.97, [10, 5, 60, 10], scale),
            item("1.5", 0.96, [10, 30, 50, 10], scale),
            item("x+y", 0.95, [70, 55, 40, 10], scale),
        ]
        second = [
            item("x-y", 0.94, [70, 55, 40, 10], scale),
            item("A-B", 0.98, [10, 5, 60, 10], scale),
            item("1.5", 0.97, [10, 30, 50, 10], scale),
        ]
        calls += 1
        return (first if calls == 1 else second), np.zeros(
            (70 * scale, 120 * scale), dtype=np.uint8
        )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)
    monkeypatch.setattr(
        image_to_ppt.text_detection, "_estimate_style",
        lambda pixels, box, *, reference_width: {
            "font_size": 18.0, "color": "#000000", "bold": False,
        },
    )

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source, [_component(x=20, y=20, w=120, h=70)], known, known_mask,
        tmp_path, lang="en", isolated=True,
    )

    assert calls == 2
    assert [item["text"] for item in result["recovered_items"]] == ["1.5"]
    assert result["recovered_items"][0]["box"] == [30, 50, 50, 10]
    assert result["diagnostics"][0]["candidate_id"] == "candidate_0001_03"
    assert result["diagnostics"][0]["bbox"] == [90, 75, 130, 85]
    assert result["diagnostics"][0]["views"] == [
        {"normalized_text": "x+y", "confidence": 0.95},
        {"normalized_text": "x-y", "confidence": 0.94},
    ]


def test_targeted_ocr_streams_source_hash_and_only_converts_crop_to_rgb(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _label_fixture(tmp_path)
    original_read_bytes = Path.read_bytes
    original_convert = Image.Image.convert

    def guarded_read_bytes(path):
        if Path(path).resolve() == source.resolve():
            raise AssertionError("source hashing must be streamed")
        return original_read_bytes(path)

    def guarded_convert(image, mode=None, *args, **kwargs):
        if image.size == (120, 70) and mode == "RGB":
            raise AssertionError("full source must not be converted to RGB")
        return original_convert(image, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Image.Image, "convert", guarded_convert)
    monkeypatch.setattr(
        image_to_ppt, "detect_text",
        lambda path, **kwargs: ([], np.zeros((1, 1), dtype=np.uint8)),
    )
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)

    image_to_ppt._targeted_candidate_ocr_sweep(
        source, [_component()], [], np.zeros((70, 120), dtype=np.uint8),
        tmp_path, lang="en", isolated=True,
    )


@pytest.mark.parametrize(
    ("left", "right", "expected_equal"),
    [
        ("1.5", "15", False),
        ("A-B", "AB", False),
        ("x+y", "xy", False),
        (" A - B ", "a-b", True),
        ("Ａ－Ｂ", "a-b", True),
    ],
)
def test_candidate_text_normalization_preserves_semantic_punctuation(
    left: str,
    right: str,
    expected_equal: bool,
) -> None:
    assert (
        image_to_ppt._normalized_candidate_text(left)
        == image_to_ppt._normalized_candidate_text(right)
    ) is expected_equal


@pytest.mark.parametrize(
    ("views", "expected_diagnostics"),
    [
        (("NX", 0.96, "NY", 0.95), 1),
        (("NX", 0.96, "NY", 0.84), 0),
        ((".", 0.99, "/", 0.99), 1),
        (("IXED SIZE", 0.99, "IIXED SIZE", 0.99), 1),
    ],
)
def test_targeted_ocr_only_diagnoses_high_confidence_text_conflicts(
    tmp_path: Path,
    monkeypatch,
    views,
    expected_diagnostics: int,
) -> None:
    source = _label_fixture(tmp_path)
    first_text, first_confidence, second_text, second_confidence = views
    call_index = 0

    def fake_detect(path, **kwargs):
        nonlocal call_index
        scale = 2 + call_index
        text = first_text if call_index == 0 else second_text
        confidence = first_confidence if call_index == 0 else second_confidence
        call_index += 1
        return [_ocr_item(text, confidence, scale)], np.zeros(
            (30 * scale, 50 * scale), dtype=np.uint8
        )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        [_component()],
        [],
        np.zeros((70, 120), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
    )

    assert result["recovered_items"] == []
    assert len(result["diagnostics"]) == expected_diagnostics
    if expected_diagnostics:
        diagnostic = result["diagnostics"][0]
        assert diagnostic["kind"] == "unowned_raster_text"
        assert diagnostic["candidate_id"] == "candidate_0001_01"
        assert diagnostic["bbox"] == [45, 24, 80, 33]
        assert diagnostic["source_sha256"] == image_to_ppt.hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        assert diagnostic["views"] == [
            {
                "normalized_text": image_to_ppt._normalized_candidate_text(
                    first_text
                ),
                "confidence": first_confidence,
            },
            {
                "normalized_text": image_to_ppt._normalized_candidate_text(
                    second_text
                ),
                "confidence": second_confidence,
            },
        ]


def test_targeted_ocr_recovers_more_complete_contained_view(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _label_fixture(tmp_path)
    readings = iter(("LETTER", "LETTER PORTRAIT"))

    def fake_detect(path, **kwargs):
        scale = 2 if Image.open(path).width == 100 else 3
        return [_ocr_item(next(readings), 0.99, scale)], np.zeros(
            (30 * scale, 50 * scale), dtype=np.uint8
        )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        [_component()],
        [{**_ocr_item("RAIT", 0.99, 1), "box": [68, 24, 12, 9]}],
        np.zeros((70, 120), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
    )

    assert [item["text"] for item in result["recovered_items"]] == [
        "LETTER PORTRAIT"
    ]
    assert result["diagnostics"] == []


def test_targeted_ocr_deduplicates_known_overlapping_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _label_fixture(tmp_path)
    known = [{
        **_ocr_item("NX", 0.99, 1),
        "box": [44, 23, 37, 11],
    }]
    known_mask = np.zeros((70, 120), dtype=np.uint8)

    def fake_detect(path, **kwargs):
        scale = 2 if Image.open(path).width == 100 else 3
        return [_ocr_item("N X", 0.96, scale)], np.zeros(
            (30 * scale, 50 * scale), dtype=np.uint8
        )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        [_component()],
        known,
        known_mask,
        tmp_path,
        lang="en",
        isolated=True,
    )

    assert result["recovered_items"] == []
    assert result["diagnostics"] == []


def test_targeted_ocr_upgrades_known_text_with_terminal_punctuation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _label_fixture(tmp_path)
    known = [{
        **_ocr_item("EDITABLE", 0.99, 1),
        "box": [44, 23, 37, 11],
    }]

    def fake_detect(path, **kwargs):
        scale = 2 if Image.open(path).width == 100 else 3
        return [_ocr_item("EDITABLE.", 0.99, scale)], np.zeros(
            (30 * scale, 50 * scale), dtype=np.uint8
        )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        [_component()],
        known,
        np.zeros((70, 120), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
    )

    assert [item["text"] for item in result["recovered_items"]] == ["EDITABLE."]
    assert [item["text"] for item in result["items"]] == ["EDITABLE."]
    assert result["diagnostics"] == []


def test_targeted_ocr_treats_highly_overlapping_ocr_variant_as_known() -> None:
    item = {
        "normalized_text": "planningmaybetoofine",
        "box": [1045, 335, 416, 31],
    }
    known = [{
        "text": "Planning may be too detailed for flexible execution",
        "box": [1047, 337, 436, 30],
    }]

    assert image_to_ppt._matches_known_text(item, known)


def test_targeted_ocr_does_not_silently_match_rotated_ocr_typo() -> None:
    item = {
        "text": "Fixed geometi",
        "normalized_text": "fixedgeometi",
        "box": [1532, 640, 33, 176],
        "rotation": 90,
    }
    known = [{
        "text": "Fixed geometry and metadata",
        "box": [1530, 640, 37, 377],
        "rotation": 90,
    }]

    assert not image_to_ppt._matches_known_text(item, known)


@pytest.mark.parametrize(
    ("candidate", "known_text"),
    [
        ("Fixed geometi", "Fixed geometry and metadata"),
        ("target servera", "Target serverb"),
        ("api version v1", "API version v2"),
        ("retention period 30", "Retention period 90"),
        ("enable archived records", "Disable archived records"),
    ],
)
def test_rotated_text_never_uses_generic_fuzzy_matching(
    candidate: str,
    known_text: str,
) -> None:
    item = {
        "text": candidate,
        "normalized_text": image_to_ppt._normalized_candidate_text(candidate),
        "box": [10, 10, 100, 30],
        "rotation": 90,
    }
    known = [{
        "text": known_text,
        "box": [10, 10, 100, 30],
        "rotation": 90,
    }]

    assert not image_to_ppt._matches_known_text(item, known)


@pytest.mark.parametrize(
    ("candidate", "known_text"),
    [
        ("planning beta execution", "Planning alpha execution details"),
        ("risk management owner", "Risk management framework and owner"),
        ("include archived records", "Exclude archived records from this export"),
        ("enable archived records", "Disable archived records for this export"),
        ("enable archived records", "Policy: disable archived records"),
        ("retention period 30", "Retention period 90 days for archived records"),
        ("api version v1", "API version v2 migration guidance"),
        ("target servera", "Target serverb and backups"),
        ("risk higher", "Risk highest priority items"),
        ("access public", "Access publish settings"),
    ],
)
def test_targeted_ocr_keeps_semantically_different_rotated_long_text(
    candidate: str,
    known_text: str,
    ) -> None:
    item = {
        "text": candidate,
        "normalized_text": image_to_ppt._normalized_candidate_text(candidate),
        "box": [10, 10, 30, 100],
        "rotation": 90,
    }
    known = [{
        "text": known_text,
        "box": [9, 0, 32, 200],
        "rotation": 90,
    }]

    assert not image_to_ppt._matches_known_text(item, known)


def test_targeted_ocr_keeps_different_text_inside_larger_known_box() -> None:
    item = {"normalized_text": "newlabel", "box": [40, 10, 30, 10]}
    known = [{"text": "existing heading", "box": [0, 0, 200, 40]}]

    assert not image_to_ppt._matches_known_text(item, known)


@pytest.mark.parametrize(
    ("candidate", "known_text"),
    [
        ("existingfooter", "existingheading"),
        ("planningbeta", "planningalpha"),
        ("sectionone", "sectiontwo"),
        ("risklow", "riskhigh"),
    ],
)
def test_targeted_ocr_keeps_similar_but_semantically_different_short_labels(
    candidate: str,
    known_text: str,
) -> None:
    item = {"normalized_text": candidate, "box": [10, 10, 120, 20]}
    known = [{"text": known_text, "box": [11, 10, 118, 20]}]

    assert not image_to_ppt._matches_known_text(item, known)


def test_targeted_ocr_keeps_recovery_signal_when_new_text_merges_with_known_line(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _label_fixture(tmp_path)
    known = [{
        **_ocr_item("AB", 0.99, 1),
        "box": [5, 24, 35, 9],
    }]

    def fake_detect(path, **kwargs):
        with Image.open(path) as crop:
            scale = crop.width // 50
        return [_ocr_item("NX", 0.96, scale)], np.zeros(
            (30 * scale, 50 * scale), dtype=np.uint8
        )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)
    monkeypatch.setattr(
        image_to_ppt.text_detection,
        "_estimate_style",
        lambda pixels, box, *, reference_width: {
            "font_size": 18.0, "color": "#ffffff", "bold": False,
        },
    )

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        [_component()],
        known,
        np.zeros((70, 120), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
    )

    assert [item["text"] for item in result["items"]] == ["AB NX"]
    assert result["recovered_items"]


def test_targeted_ocr_enforces_candidate_and_pixel_budgets_without_page_copies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "large.png"
    Image.new("RGB", (1600, 900), "white").save(source)
    components = [
        _component(index, x=(index % 12) * 100, y=(index // 12) * 100,
                   w=80, h=50)
        for index in range(48)
    ]
    crop_pixels = []
    close_calls = []

    def fake_detect(path, **kwargs):
        with Image.open(path) as crop:
            crop_pixels.append(crop.width * crop.height)
            return [], np.zeros((crop.height, crop.width), dtype=np.uint8)

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(
        image_to_ppt,
        "_load_rgb",
        lambda path: pytest.fail("no full-page RGB array is needed without recovery"),
    )
    monkeypatch.setattr(
        image_to_ppt, "close_ocr_engines", lambda: close_calls.append("closed")
    )

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        components,
        [],
        np.zeros((900, 1600), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
    )

    stats = result["resource_stats"]
    assert stats["selected_candidates"] <= stats["candidate_limit"]
    assert len(crop_pixels) <= stats["candidate_limit"] * 2
    assert max(crop_pixels) <= stats["single_crop_pixel_limit"]
    assert sum(crop_pixels) <= stats["total_crop_pixel_limit"]
    assert close_calls == ["closed"]


def test_targeted_ocr_caps_each_view_at_32_and_page_diagnostics_at_96(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "many-conflicts.png"
    Image.new("RGB", (1200, 800), "white").save(source)
    components = [
        _component(index, x=50 + index * 250, y=100, w=200, h=200)
        for index in range(4)
    ]

    def fake_detect(path, **kwargs):
        stem = Path(path).stem.split("-")
        component_index = int(stem[1])
        view_index = int(stem[3])
        with Image.open(path) as crop:
            scale = crop.width / 200
            items = [
                {
                    **_ocr_item(
                        f"{'a' if view_index == 1 else 'b'}{component_index}_{index}",
                        0.96,
                        1,
                    ),
                    "box": [
                        int(round(10 * scale)),
                        int(round((2 + index * 4) * scale)),
                        int(round(30 * scale)),
                        int(round(3 * scale)),
                    ],
                }
                for index in reversed(range(40))
            ]
            return items, np.zeros((crop.height, crop.width), dtype=np.uint8)

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source, components, [], np.zeros((800, 1200), dtype=np.uint8),
        tmp_path, lang="en", isolated=True,
    )

    diagnostics = result["diagnostics"]
    assert len(diagnostics) == 96
    assert diagnostics[0]["candidate_id"] == "candidate_0001_01"
    assert diagnostics[31]["candidate_id"] == "candidate_0001_32"
    assert diagnostics[32]["candidate_id"] == "candidate_0002_01"
    assert diagnostics[-1]["candidate_id"] == "candidate_0003_32"
    assert not any(
        item["candidate_id"].startswith("candidate_0004_")
        for item in diagnostics
    )
    source_sha256 = image_to_ppt.hashlib.sha256(source.read_bytes()).hexdigest()
    assert validate_initial_diagnostics(
        diagnostics, source_sha256=source_sha256, image_size=(1200, 800)
    ) == diagnostics
    report = evaluate_page_quality(
        [],
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={"pptx_reopen": "pass", "unowned_raster_text": "fail"},
        expected_component_ids=[], initial_component_count=0,
        active_visual_count=0,
    )
    assert report["accepted"] is False
    assert report["violations"] == ["unowned_raster_text"]


def test_targeted_ocr_budget_reaches_late_small_candidate_after_large_summaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (1200, 675), "white").save(source)
    components = [
        _component(index, x=10, y=10, w=500, h=220)
        for index in range(3)
    ] + [
        _component(index, x=20 + index * 40, y=300, w=30, h=20)
        for index in range(3, 12)
    ]
    seen = []

    def fake_detect(path, **kwargs):
        candidate = int(Path(path).stem.split("-")[1])
        seen.append(candidate)
        with Image.open(path) as crop:
            scale = round(crop.width / 30)
            shape = (crop.height, crop.width)
        items = [_ocr_item("NX", 0.96, scale)] if candidate == 12 else []
        return items, np.zeros(shape, dtype=np.uint8)

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)

    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source,
        components,
        [],
        np.zeros((675, 1200), dtype=np.uint8),
        tmp_path,
        lang="en",
        isolated=True,
    )

    assert 12 in seen
    assert [item["text"] for item in result["recovered_items"]] == ["NX"]


def test_targeted_ocr_closes_worker_lifecycle_when_detection_raises(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _label_fixture(tmp_path)
    closed = []
    monkeypatch.setattr(
        image_to_ppt,
        "detect_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ocr failed")),
    )
    monkeypatch.setattr(
        image_to_ppt, "close_ocr_engines", lambda: closed.append(True)
    )

    with pytest.raises(RuntimeError, match="ocr failed"):
        image_to_ppt._targeted_candidate_ocr_sweep(
            source,
            [_component()],
            [],
            np.zeros((70, 120), dtype=np.uint8),
            tmp_path,
            lang="en",
            isolated=True,
        )

    assert closed == [True]


def _rewrite_prepared_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = image_to_ppt.hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name("prepared_page.sha256").write_bytes(
        (digest + "\n").encode("ascii")
    )


def _write_text_delta_mask(root: Path, name: str, mask: np.ndarray) -> dict:
    path = root / "masks" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": image_to_ppt.hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _text_delta_graph(root: Path, old_mask: np.ndarray) -> tuple[dict, dict]:
    child = np.zeros_like(old_mask)
    child[20:28, 30:40] = 255
    parent = np.zeros_like(old_mask)
    parent[18:30, 28:42] = 255
    neighbor = np.zeros_like(old_mask)
    neighbor[20:28, 44:50] = 255
    distant = np.zeros_like(old_mask)
    distant[55:65, 75:85] = 255
    source_sha256 = "a" * 64
    identity = {
        "schema_version": 1,
        "cache_key": "",
        "source_sha256": source_sha256,
        "old_cleanup_mask_sha256": image_to_ppt.hashlib.sha256(
            np.ascontiguousarray(old_mask, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "sam_protocol_sha256": image_to_ppt._TEXT_DELTA_SAM_PROTOCOL_SHA256,
        "dino_protocol_sha256": image_to_ppt._TEXT_DELTA_DINO_PROTOCOL_SHA256,
        "prepared_manifest_sha256": "d" * 64,
    }
    graph = {
        "schema_version": 1,
        "cache_key": "",
        "nodes": [
            {
                "id": "child",
                "mask": _write_text_delta_mask(root, "child", child),
                "bbox": [30, 20, 40, 28],
                "parents": ["parent"],
                "children": [],
            },
            {
                "id": "parent",
                "mask": _write_text_delta_mask(root, "parent", parent),
                "bbox": [28, 18, 42, 30],
                "parents": [],
                "children": ["child"],
            },
            {
                "id": "touching_neighbor",
                "mask": _write_text_delta_mask(root, "neighbor", neighbor),
                "bbox": [44, 20, 50, 28],
                "parents": [],
                "children": [],
            },
            {
                "id": "distant",
                "mask": _write_text_delta_mask(root, "distant", distant),
                "bbox": [75, 55, 85, 65],
                "parents": [],
                "children": [],
            },
        ],
    }
    cache_key = image_to_ppt.hashlib.sha256(json.dumps({
        "schema_version": identity["schema_version"],
        "source_sha256": identity["source_sha256"],
        "old_cleanup_mask_sha256": identity["old_cleanup_mask_sha256"],
        "sam_protocol_sha256": identity["sam_protocol_sha256"],
        "dino_protocol_sha256": identity["dino_protocol_sha256"],
        "prepared_manifest_sha256": identity["prepared_manifest_sha256"],
        "nodes": graph["nodes"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    identity["cache_key"] = cache_key
    graph["cache_key"] = cache_key
    return graph, identity


def test_text_delta_reopens_intersections_parents_children_and_neighbors(
    tmp_path: Path,
) -> None:
    old_mask = np.zeros((80, 100), dtype=np.uint8)
    new_mask = old_mask.copy()
    new_mask[21:27, 32:38] = 255
    graph, identity = _text_delta_graph(tmp_path, old_mask)

    scope = image_to_ppt._text_delta_recompute_scope(
        old_mask=old_mask,
        new_mask=new_mask,
        graph=graph,
        graph_dir=tmp_path,
        source_sha256="a" * 64,
        cache_identity=identity,
    )

    assert scope == {"child", "parent", "touching_neighbor"}


@pytest.mark.parametrize(
    "mutation",
    [
        "source_hash",
        "old_mask_hash",
        "sam_protocol_hash",
        "dino_protocol_hash",
        "prepared_manifest_hash",
        "cache_identity",
        "missing_relation",
        "unreadable_mask",
        "wrong_shape",
        "path_escape",
        "graph_content",
    ],
)
def test_text_delta_incomplete_or_mismatched_cache_requires_full_recompute(
    tmp_path: Path,
    mutation: str,
) -> None:
    old_mask = np.zeros((80, 100), dtype=np.uint8)
    new_mask = old_mask.copy()
    new_mask[21:27, 32:38] = 255
    graph, identity = _text_delta_graph(tmp_path, old_mask)
    source_sha256 = "a" * 64
    if mutation == "source_hash":
        source_sha256 = "c" * 64
    elif mutation == "old_mask_hash":
        identity["old_cleanup_mask_sha256"] = "c" * 64
    elif mutation == "sam_protocol_hash":
        identity["sam_protocol_sha256"] = "c" * 64
    elif mutation == "dino_protocol_hash":
        identity["dino_protocol_sha256"] = "c" * 64
    elif mutation == "prepared_manifest_hash":
        identity["prepared_manifest_sha256"] = "c" * 64
    elif mutation == "cache_identity":
        graph["cache_key"] = "c" * 64
    elif mutation == "missing_relation":
        graph["nodes"][1]["children"] = []
    elif mutation == "unreadable_mask":
        Path(tmp_path, graph["nodes"][0]["mask"]["path"]).write_bytes(b"bad")
    elif mutation == "wrong_shape":
        path = Path(tmp_path, graph["nodes"][0]["mask"]["path"])
        Image.new("L", (10, 10), 255).save(path)
        graph["nodes"][0]["mask"]["sha256"] = image_to_ppt.hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    elif mutation == "path_escape":
        outside = tmp_path.parent / "outside-mask.png"
        Image.new("L", (100, 80), 255).save(outside)
        graph["nodes"][0]["mask"] = {
            "path": "../outside-mask.png",
            "sha256": image_to_ppt.hashlib.sha256(outside.read_bytes()).hexdigest(),
        }
    elif mutation == "graph_content":
        graph["nodes"][0]["mask"] = dict(graph["nodes"][3]["mask"])

    assert image_to_ppt._text_delta_recompute_scope(
        old_mask=old_mask,
        new_mask=new_mask,
        graph=graph,
        graph_dir=tmp_path,
        source_sha256=source_sha256,
        cache_identity=identity,
    ) is None


def test_text_delta_changed_ocr_with_empty_mask_delta_requires_full_recompute(
    tmp_path: Path,
) -> None:
    old_mask = np.zeros((80, 100), dtype=np.uint8)
    graph, identity = _text_delta_graph(tmp_path, old_mask)

    assert image_to_ppt._text_delta_recompute_scope(
        old_mask=old_mask,
        new_mask=old_mask.copy(),
        graph=graph,
        graph_dir=tmp_path,
        source_sha256="a" * 64,
        cache_identity=identity,
    ) is None


def test_text_delta_cleanup_mask_shrink_requires_full_recompute(tmp_path: Path) -> None:
    old_mask = np.zeros((80, 100), dtype=np.uint8)
    old_mask[5:10, 5:10] = 255
    new_mask = old_mask.copy()
    new_mask[5:10, 5:10] = 0
    new_mask[70:75, 90:95] = 255
    graph, identity = _text_delta_graph(tmp_path, old_mask)

    assert image_to_ppt._text_delta_recompute_scope(
        old_mask=old_mask,
        new_mask=new_mask,
        graph=graph,
        graph_dir=tmp_path,
        source_sha256="a" * 64,
        cache_identity=identity,
    ) is None


def test_text_delta_three_pixel_safety_margin_reopens_nearby_node(
    tmp_path: Path,
) -> None:
    old_mask = np.zeros((80, 100), dtype=np.uint8)
    new_mask = old_mask.copy()
    new_mask[20:24, 26:28] = 255
    graph, identity = _text_delta_graph(tmp_path, old_mask)

    scope = image_to_ppt._text_delta_recompute_scope(
        old_mask=old_mask,
        new_mask=new_mask,
        graph=graph,
        graph_dir=tmp_path,
        source_sha256="a" * 64,
        cache_identity=identity,
    )

    assert scope == {"child", "parent", "touching_neighbor"}


def test_text_delta_component_bbox_reopens_transparent_mask_region(
    tmp_path: Path,
) -> None:
    old_mask = np.zeros((80, 100), dtype=np.uint8)
    new_mask = old_mask.copy()
    new_mask[20:24, 20:24] = 255
    graph, identity = _text_delta_graph(tmp_path, old_mask)
    ring = np.zeros_like(old_mask)
    ring[5:35, 5:40] = 255
    ring[8:32, 8:37] = 0
    child_path = Path(tmp_path, graph["nodes"][0]["mask"]["path"])
    Image.fromarray(ring).save(child_path)
    graph["nodes"][0]["mask"]["sha256"] = image_to_ppt.hashlib.sha256(
        child_path.read_bytes()
    ).hexdigest()
    graph["nodes"][0]["bbox"] = [5, 5, 40, 35]
    identity["cache_key"] = image_to_ppt.hashlib.sha256(json.dumps({
        "schema_version": identity["schema_version"],
        "source_sha256": identity["source_sha256"],
        "old_cleanup_mask_sha256": identity["old_cleanup_mask_sha256"],
        "sam_protocol_sha256": identity["sam_protocol_sha256"],
        "dino_protocol_sha256": identity["dino_protocol_sha256"],
        "prepared_manifest_sha256": identity["prepared_manifest_sha256"],
        "nodes": graph["nodes"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    graph["cache_key"] = identity["cache_key"]

    assert image_to_ppt._text_delta_recompute_scope(
        old_mask=old_mask,
        new_mask=new_mask,
        graph=graph,
        graph_dir=tmp_path,
        source_sha256="a" * 64,
        cache_identity=identity,
    ) == {"child", "parent", "touching_neighbor"}


def test_text_delta_pairwise_budget_requires_full_recompute(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old_mask = np.zeros((80, 100), dtype=np.uint8)
    new_mask = old_mask.copy()
    new_mask[70:75, 90:95] = 255
    graph, identity = _text_delta_graph(tmp_path, old_mask)
    monkeypatch.setattr(image_to_ppt, "_TEXT_DELTA_MAX_PAIRWISE_PIXELS", 1)

    assert image_to_ppt._text_delta_recompute_scope(
        old_mask=old_mask,
        new_mask=new_mask,
        graph=graph,
        graph_dir=tmp_path,
        source_sha256="a" * 64,
        cache_identity=identity,
    ) is None


def test_text_delta_pair_candidate_budget_counts_disjoint_nodes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old_mask = np.zeros((80, 100), dtype=np.uint8)
    new_mask = old_mask.copy()
    new_mask[70:75, 90:95] = 255
    graph, identity = _text_delta_graph(tmp_path, old_mask)
    graph["nodes"] = [graph["nodes"][0], graph["nodes"][3]]
    graph["nodes"][0]["parents"] = []
    identity["cache_key"] = image_to_ppt.hashlib.sha256(json.dumps({
        "schema_version": identity["schema_version"],
        "source_sha256": identity["source_sha256"],
        "old_cleanup_mask_sha256": identity["old_cleanup_mask_sha256"],
        "sam_protocol_sha256": identity["sam_protocol_sha256"],
        "dino_protocol_sha256": identity["dino_protocol_sha256"],
        "prepared_manifest_sha256": identity["prepared_manifest_sha256"],
        "nodes": graph["nodes"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    graph["cache_key"] = identity["cache_key"]
    monkeypatch.setattr(image_to_ppt, "_TEXT_DELTA_MAX_PAIRWISE_CANDIDATES", 0)

    assert image_to_ppt._text_delta_recompute_scope(
        old_mask=old_mask,
        new_mask=new_mask,
        graph=graph,
        graph_dir=tmp_path,
        source_sha256="a" * 64,
        cache_identity=identity,
    ) is None


def test_text_delta_only_dilates_one_full_page_array(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old_mask = np.zeros((720, 1280), dtype=np.uint8)
    new_mask = old_mask.copy()
    new_mask[600:605, 1100:1105] = 255
    graph, identity = _text_delta_graph(tmp_path, old_mask)
    dilated_shapes = []
    original_dilate = image_to_ppt.cv2.dilate

    def recording_dilate(value, *args, **kwargs):
        dilated_shapes.append(np.asarray(value).shape)
        return original_dilate(value, *args, **kwargs)

    monkeypatch.setattr(image_to_ppt.cv2, "dilate", recording_dilate)

    assert image_to_ppt._text_delta_recompute_scope(
        old_mask=old_mask,
        new_mask=new_mask,
        graph=graph,
        graph_dir=tmp_path,
        source_sha256="a" * 64,
        cache_identity=identity,
    ) == set()
    assert dilated_shapes.count(old_mask.shape) == 1
    assert all(shape == old_mask.shape or np.prod(shape) < 1000
               for shape in dilated_shapes)


def _prepare_rerun_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    include_diagnostic: bool = False,
    diagnostics_count: int | None = None,
    check_first_pass_cleanup: bool = False,
    safe_text_delta: bool = False,
    affected_text_delta: bool = False,
    corrupt_cache: bool = False,
    stable_visual_output: bool = False,
    background_error: bool = False,
    second_pass_error: bool = False,
    cache_error: bool = False,
    missing_cache: bool = False,
    source_change_before_manifest: bool = False,
    wrong_component_shape: bool = False,
    forbid_cache_cleanup: bool = False,
    replace_mask_after_first_worker: bool = False,
    unchanged_text_mask: bool = False,
) -> tuple[dict, list[int]]:
    source = _label_fixture(tmp_path)
    work_dir = tmp_path / "prepared"
    process_text_counts = []
    stale_component = work_dir / "components/stale.png"
    stale_child = work_dir / "element-masks/stale.png"
    stale_parent = work_dir / "semantic-masks/stale.png"
    outside_component = tmp_path / "outside-component.png"

    initial_text_mask = np.zeros((70, 120), dtype=np.uint8)
    if unchanged_text_mask:
        initial_text_mask[18:39, 39:86] = 255
    monkeypatch.setattr(
        image_to_ppt,
        "detect_text",
        lambda *args, **kwargs: (
            [], initial_text_mask.copy()
        ),
    )
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)
    monkeypatch.setattr(image_to_ppt, "_release_visual_resources", lambda: None)
    cleanup_calls = []

    def fake_cleanup(image, text_mask, text_items):
        cleanup = np.zeros((70, 120), dtype=np.uint8)
        if safe_text_delta and text_items:
            cleanup[18:39, 39:86] = 255
        if affected_text_delta and text_items:
            cleanup[12:20, 12:20] = 255
        cleanup_calls.append(cleanup.copy())
        return cleanup

    monkeypatch.setattr(image_to_ppt, "_build_text_cleanup_mask", fake_cleanup)
    monkeypatch.setattr(
        image_to_ppt,
        "_repair_text_background",
        lambda image, *args, **kwargs: image.copy(),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "_isolated_large_inpainter",
        lambda work_dir: object(),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "build_clean_background",
        lambda image, *args, **kwargs: (
            (_ for _ in ()).throw(RuntimeError("background failed"))
            if background_error else image.copy()
        ),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "build_widescreen_background",
        lambda image, **kwargs: (image.copy(), 0, 0, "identity"),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "build_removal_mask",
        lambda masks, text_mask: np.logical_or.reduce(
            [np.asarray(mask) > 0 for mask in masks] + [np.asarray(text_mask) > 0]
        ).astype(np.uint8) * 255,
    )
    if corrupt_cache:
        original_write_cache = image_to_ppt._write_first_visual_cache

        def corrupting_write_cache(*args, **kwargs):
            path = original_write_cache(*args, **kwargs)
            path.write_text("{}", encoding="utf-8")
            return path

        monkeypatch.setattr(
            image_to_ppt, "_write_first_visual_cache", corrupting_write_cache
        )
    if cache_error:
        monkeypatch.setattr(
            image_to_ppt,
            "_write_first_visual_cache",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("first visual cache exceeds its size limit")
            ),
        )
    if missing_cache or wrong_component_shape:
        original_write_cache = image_to_ppt._write_first_visual_cache

        def changing_write_cache(manifest, target, cleanup):
            if wrong_component_shape:
                component = manifest["components"][0]
                component_path = Path(target, component["asset"]["path"])
                Image.new("RGBA", (1, 1), "red").save(component_path)
                component["asset"]["sha256"] = image_to_ppt.hashlib.sha256(
                    component_path.read_bytes()
                ).hexdigest()
            path = original_write_cache(manifest, target, cleanup)
            if missing_cache:
                path.unlink()
            return path

        monkeypatch.setattr(
            image_to_ppt, "_write_first_visual_cache", changing_write_cache
        )
    if source_change_before_manifest:
        original_write_prepared = image_to_ppt._write_prepared_page
        prepared_writes = 0

        def changing_write_prepared(slide_data, target):
            nonlocal prepared_writes
            if prepared_writes == 0:
                with Image.open(slide_data["original_image_path"]) as original:
                    size = original.size
                Image.new("RGB", size, "black").save(slide_data["original_image_path"])
            prepared_writes += 1
            return original_write_prepared(slide_data, target)

        monkeypatch.setattr(
            image_to_ppt, "_write_prepared_page", changing_write_prepared
        )
    if forbid_cache_cleanup:
        original_unlink = Path.unlink

        def guarded_unlink(path, *args, **kwargs):
            if Path(path).name in {
                "first-ocr-mask.png",
                "first-text-cleanup-mask.png",
                "first-visual-cache.json",
            }:
                raise AssertionError("text delta cache cleanup must not use path unlink")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", guarded_unlink)

    recovered = _ocr_item("NX", 0.96, 1)
    recovered["box"] = [45, 24, 35, 9]
    recovered_mask = np.zeros((70, 120), dtype=np.uint8)
    recovered_mask[18:39, 39:86] = 255
    diagnostics = []
    if include_diagnostic or diagnostics_count:
        diagnostics = [{
            "kind": "unowned_raster_text",
            "source_sha256": image_to_ppt.hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
            "candidate_id": f"candidate_{index:04d}",
            "bbox": [40, 20, 90, 50],
            "views": [
                {"normalized_text": "nx", "confidence": 0.96},
                {"normalized_text": "ny", "confidence": 0.95},
            ],
        } for index in range(1, (diagnostics_count or 1) + 1)]
    monkeypatch.setattr(
        image_to_ppt,
        "_targeted_candidate_ocr_sweep",
        lambda *args, **kwargs: {
            "items": [recovered],
            "recovered_items": [recovered],
            "text_mask": recovered_mask,
            "diagnostics": diagnostics,
            "resource_stats": {},
        },
    )

    def fake_process(path, target, lang, text_analysis):
        pass_index = len(process_text_counts) + 1
        if check_first_pass_cleanup and pass_index == 2:
            assert not stale_component.exists()
            assert not stale_child.exists()
            assert not stale_parent.exists()
            assert outside_component.exists()
        process_text_counts.append(len(text_analysis["items"]))
        if second_pass_error and pass_index == 2:
            raise RuntimeError("full visual failed")
        components_dir = target / "components"
        child_dir = target / "element-masks"
        parent_dir = target / "semantic-masks"
        components_dir.mkdir(exist_ok=True)
        child_dir.mkdir(exist_ok=True)
        parent_dir.mkdir(exist_ok=True)
        component_path = components_dir / "component_0000.png"
        Image.new(
            "RGBA", (20, 20),
            "red" if stable_visual_output or pass_index == 1 else "green",
        ).save(component_path)
        child = np.zeros((70, 120), dtype=np.uint8)
        child[10:30, 10:30] = 255
        parent = np.zeros((70, 120), dtype=np.uint8)
        parent[8:32, 8:32] = 255
        child_path = child_dir / "0000.png"
        parent_path = parent_dir / "0000.png"
        Image.fromarray(child).save(child_path)
        Image.fromarray(parent).save(parent_path)
        components = [{
            "path": str(component_path), "x": 10, "y": 10,
            "w": 20, "h": 20, "area": 400, "z_index": 0,
        }]
        child_paths = [str(child_path)]
        parent_paths = [str(parent_path)]
        if check_first_pass_cleanup and pass_index == 1:
            Image.new("RGBA", (5, 5), "blue").save(stale_component)
            Image.new("L", (120, 70), 255).save(stale_child)
            Image.new("L", (120, 70), 255).save(stale_parent)
            Image.new("RGBA", (5, 5), "yellow").save(outside_component)
            components.extend([
                {"path": str(stale_component), "x": 40, "y": 10,
                 "w": 5, "h": 5, "area": 25, "z_index": 1},
            ])
            child_paths.append(str(stale_child))
            parent_paths.append(str(stale_parent))
        foreground_evidence = np.zeros((70, 120), dtype=np.uint8)
        for parent_mask_path in parent_paths:
            with Image.open(parent_mask_path) as parent_mask:
                foreground_evidence |= np.asarray(
                    parent_mask.convert("L"), dtype=np.uint8
                )
        foreground_evidence_path = target / "foreground-evidence-mask.png"
        Image.fromarray(foreground_evidence).save(foreground_evidence_path)
        background = target / "background-original.png"
        Image.new("RGB", (120, 70), "white").save(background)
        Image.new("L", (120, 70), 0).save(target / "background-removal-mask.png")
        Image.new("RGB", (120, 70), "black").save(
            target / "background-difference.png"
        )
        with Image.open(text_analysis["mask_path"]) as used_text_mask:
            used_text_mask_array = np.asarray(
                used_text_mask.convert("L"), dtype=np.uint8
            ).copy()
        visual_text_mask_sha256 = image_to_ppt.hashlib.sha256(
            np.ascontiguousarray(used_text_mask_array).tobytes()
        ).hexdigest()
        if replace_mask_after_first_worker and pass_index == 1:
            Image.new("L", (120, 70), 255).save(text_analysis["mask_path"])
        return {
            "background_path": str(background),
            "background_original_path": str(background),
            "background_widescreen_path": str(background),
            "components": components,
            "text_items": text_analysis["items"],
            "img_width": 120,
            "img_height": 70,
            "canvas_width": 120,
            "canvas_height": 70,
            "content_offset_x": 0,
            "content_offset_y": 0,
            "widescreen_background_method": "identity",
            "original_image_path": str(path),
            "_work_dir": str(target),
            "_text_mask_path": text_analysis["mask_path"],
            "_element_mask_paths": child_paths,
            "_semantic_mask_paths": parent_paths,
            "_foreground_evidence_mask_path": str(foreground_evidence_path),
            "_visual_source_sha256": image_to_ppt.hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest(),
            "_visual_text_mask_sha256": visual_text_mask_sha256,
            "_visual_text_clean_sha256": None,
        }

    monkeypatch.setattr(image_to_ppt, "_process_image_isolated", fake_process)
    prepared = image_to_ppt.prepare_component_layers(
        source,
        work_dir,
        lang="en",
        resource_isolation=True,
    )
    return prepared, process_text_counts


def test_prepare_reruns_all_visual_assets_after_targeted_text_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, process_text_counts = _prepare_rerun_fixture(tmp_path, monkeypatch)

    assert process_text_counts == [0, 1]
    assert [item["text"] for item in prepared["text_items"]] == ["NX"]
    assert prepared["initial_diagnostics"] == []
    with Image.open(prepared["components"][0]["path"]) as component:
        assert component.convert("RGB").getpixel((0, 0)) == (0, 128, 0)
    with Image.open(prepared["_text_mask_path"]) as mask:
        assert mask.getpixel((50, 25)) == 255


def test_prepare_reuses_verified_visual_assets_for_disjoint_text_delta(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, process_text_counts = _prepare_rerun_fixture(
        tmp_path,
        monkeypatch,
        safe_text_delta=True,
    )

    assert process_text_counts == [0]
    assert [item["text"] for item in prepared["text_items"]] == ["NX"]
    assert [component["z_index"] for component in prepared["components"]] == [0]
    with Image.open(prepared["components"][0]["path"]) as component:
        assert component.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    with Image.open(prepared["_element_mask_paths"][0]) as child:
        child_mask = np.asarray(child.convert("L")) > 0
    with Image.open(prepared["_semantic_mask_paths"][0]) as parent:
        parent_mask = np.asarray(parent.convert("L")) > 0
    assert int(np.count_nonzero(child_mask)) == 400
    assert int(np.count_nonzero(parent_mask)) == 24 * 24
    with Image.open(prepared["background_removal_mask_path"]) as removal:
        removal_mask = np.asarray(removal.convert("L")) > 0
    assert np.all(removal_mask[18:39, 39:86])
    assert (Path(prepared["_work_dir"]) / "first-visual-cache.json").is_file()


def test_prepare_reuses_visual_assets_when_targeted_ocr_keeps_exact_masks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, process_text_counts = _prepare_rerun_fixture(
        tmp_path,
        monkeypatch,
        unchanged_text_mask=True,
    )

    assert process_text_counts == [0]
    assert [item["text"] for item in prepared["text_items"]] == ["NX"]
    with Image.open(prepared["components"][0]["path"]) as component:
        assert component.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_text_delta_mask_changed_after_first_worker_uses_full_visual_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, calls = _prepare_rerun_fixture(
        tmp_path,
        monkeypatch,
        safe_text_delta=True,
        replace_mask_after_first_worker=True,
    )

    assert calls == [0, 1]


@pytest.mark.parametrize("case", ["affected", "corrupt_cache"])
def test_prepare_uses_full_second_visual_pass_when_reuse_is_not_provable(
    tmp_path: Path,
    monkeypatch,
    case: str,
) -> None:
    prepared, process_text_counts = _prepare_rerun_fixture(
        tmp_path,
        monkeypatch,
        affected_text_delta=case == "affected",
        safe_text_delta=case == "corrupt_cache",
        corrupt_cache=case == "corrupt_cache",
    )

    assert process_text_counts == [0, 1]
    with Image.open(prepared["components"][0]["path"]) as component:
        assert component.convert("RGB").getpixel((0, 0)) == (0, 128, 0)
    assert (Path(prepared["_work_dir"]) / "first-visual-cache.json").is_file()


def test_disjoint_text_delta_matches_full_visual_recompute_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    incremental_root = tmp_path / "incremental"
    incremental_root.mkdir()
    incremental, incremental_calls = _prepare_rerun_fixture(
        incremental_root,
        monkeypatch,
        safe_text_delta=True,
        stable_visual_output=True,
    )
    full_root = tmp_path / "full"
    full_root.mkdir()
    full, full_calls = _prepare_rerun_fixture(
        full_root,
        monkeypatch,
        safe_text_delta=True,
        corrupt_cache=True,
        stable_visual_output=True,
    )

    def evidence(prepared: dict, name: str) -> tuple[set[str], np.ndarray, list, list]:
        store = RunStore(tmp_path / f"run-{name}")
        store.write_json("job_manifest.json", {
            "schema_version": 1,
            "pages": ["page_001"],
            "options": {"agent_provider": "host"},
        })
        reconstruction = store.root / "pages/page_001/reconstruction"
        reconstruction.mkdir(parents=True)
        session = legacy._build_initial_page_session(
            store, "page_001", prepared, reconstruction
        )
        graph = json.loads(
            session["evidence"]["component-graph.json"].read_text(encoding="utf-8")
        )
        quality = json.loads(
            session["evidence"]["quality-report.json"].read_text(encoding="utf-8")
        )
        mask_union = np.zeros((70, 120), dtype=bool)
        for path in prepared["_element_mask_paths"]:
            with Image.open(path) as mask:
                mask_union |= np.asarray(mask.convert("L")) > 0
        visual_ids = {
            node["id"] for node in graph["nodes"] if node["kind"] != "text"
        }
        return visual_ids, mask_union, prepared["text_items"], quality["violations"]

    incremental_evidence = evidence(incremental, "incremental")
    full_evidence = evidence(full, "full")

    assert incremental_calls == [0]
    assert full_calls == [0, 1]
    assert incremental_evidence[0] == full_evidence[0]
    assert np.array_equal(incremental_evidence[1], full_evidence[1])
    assert incremental_evidence[2:] == full_evidence[2:]


def test_text_delta_incremental_failure_does_not_repeat_expensive_inference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with pytest.raises(RuntimeError, match="background failed"):
        _prepare_rerun_fixture(
            tmp_path,
            monkeypatch,
            safe_text_delta=True,
            background_error=True,
        )


def test_text_delta_full_fallback_preserves_original_visual_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with pytest.raises(RuntimeError, match="full visual failed"):
        _prepare_rerun_fixture(
            tmp_path,
            monkeypatch,
            affected_text_delta=True,
            second_pass_error=True,
        )


def test_text_delta_cache_limit_uses_full_visual_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, calls = _prepare_rerun_fixture(
        tmp_path,
        monkeypatch,
        safe_text_delta=True,
        cache_error=True,
    )

    assert calls == [0, 1]
    with Image.open(prepared["components"][0]["path"]) as component:
        assert component.convert("RGB").getpixel((0, 0)) == (0, 128, 0)


@pytest.mark.parametrize(
    "case", ["missing_cache", "source_changed", "wrong_component_shape"]
)
def test_text_delta_untrusted_cache_binding_uses_full_visual_fallback(
    tmp_path: Path,
    monkeypatch,
    case: str,
) -> None:
    prepared, calls = _prepare_rerun_fixture(
        tmp_path,
        monkeypatch,
        safe_text_delta=True,
        missing_cache=case == "missing_cache",
        source_change_before_manifest=case == "source_changed",
        wrong_component_shape=case == "wrong_component_shape",
    )

    assert calls == [0, 1]
    with Image.open(prepared["components"][0]["path"]) as component:
        assert component.convert("RGB").getpixel((0, 0)) == (0, 128, 0)


def test_text_delta_full_fallback_never_path_unlinks_cache_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, calls = _prepare_rerun_fixture(
        tmp_path,
        monkeypatch,
        affected_text_delta=True,
        forbid_cache_cleanup=True,
    )

    assert calls == [0, 1]


def test_prepare_preserves_stable_diagnostic_when_another_candidate_recovers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, process_text_counts = _prepare_rerun_fixture(
        tmp_path, monkeypatch, include_diagnostic=True
    )

    assert process_text_counts == [0, 1]
    assert prepared["initial_diagnostics"] == [{
        "kind": "unowned_raster_text",
        "source_sha256": image_to_ppt.hashlib.sha256(
            Path(prepared["original_image_path"]).read_bytes()
        ).hexdigest(),
        "candidate_id": "candidate_0001",
        "bbox": [40, 20, 90, 50],
        "views": [
            {"normalized_text": "nx", "confidence": 0.96},
            {"normalized_text": "ny", "confidence": 0.95},
        ],
    }]
    assert prepared["components"][0]["x"] == 10


def test_prepare_writes_25_diagnostics_and_initial_quality_hard_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, _ = _prepare_rerun_fixture(
        tmp_path, monkeypatch, diagnostics_count=25
    )
    reloaded = image_to_ppt.load_component_layers(prepared["state_path"])
    assert len(reloaded["initial_diagnostics"]) == 25

    store = RunStore(tmp_path / "run-many")
    store.write_json("job_manifest.json", {
        "schema_version": 1,
        "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    reconstruction = store.root / "pages/page_001/reconstruction"
    reconstruction.mkdir(parents=True)
    session = legacy._build_initial_page_session(
        store, "page_001", reloaded, reconstruction
    )
    quality = json.loads(
        session["evidence"]["quality-report.json"].read_text(encoding="utf-8")
    )
    assert len(quality["initial_diagnostics"]) == 25
    assert quality["violations"] == ["unowned_raster_text"]


def test_prepare_removes_only_owned_first_pass_visual_residuals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, passes = _prepare_rerun_fixture(
        tmp_path, monkeypatch, check_first_pass_cleanup=True
    )

    assert passes == [0, 1]
    assert Path(prepared["components"][0]["path"]).is_file()
    assert (tmp_path / "outside-component.png").is_file()


def _cleanup_assets(tmp_path: Path) -> tuple[Path, dict, list[Path]]:
    work_dir = tmp_path / "cleanup"
    files = []
    for directory, name in (
        ("components", "component.png"),
        ("element-masks", "element.png"),
        ("semantic-masks", "semantic.png"),
    ):
        path = work_dir / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(directory.encode("ascii"))
        files.append(path)
    return work_dir, {
        "components": [{"path": str(files[0])}],
        "_element_mask_paths": [str(files[1])],
        "_semantic_mask_paths": [str(files[2])],
    }, files


def test_first_visual_cleanup_deletes_only_normal_owned_assets(tmp_path: Path) -> None:
    work_dir, slide_data, files = _cleanup_assets(tmp_path)

    image_to_ppt._remove_owned_first_visual_assets(slide_data, work_dir)

    assert not any(path.exists() for path in files)


@pytest.mark.parametrize("location", ["source", "ocr-mask", "other", "outside"])
def test_first_visual_cleanup_rejects_non_visual_asset_paths(
    tmp_path: Path,
    location: str,
) -> None:
    work_dir, _, _ = _cleanup_assets(tmp_path)
    paths = {
        "source": work_dir / "source.png",
        "ocr-mask": work_dir / "source-text-mask.png",
        "other": work_dir / "notes.bin",
        "outside": tmp_path / "outside.png",
    }
    target = paths[location]
    target.write_bytes(b"keep")

    with pytest.raises(ValueError, match="outside|owned"):
        image_to_ppt._remove_owned_first_visual_assets(
            {"components": [{"path": str(target)}]}, work_dir
        )

    assert target.read_bytes() == b"keep"


def test_first_visual_cleanup_rejects_hardlinked_asset(tmp_path: Path) -> None:
    work_dir, slide_data, files = _cleanup_assets(tmp_path)
    external_name = work_dir / "hardlink-copy.png"
    try:
        os.link(files[0], external_name)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="identity"):
        image_to_ppt._remove_owned_first_visual_assets(slide_data, work_dir)

    assert files[0].is_file()
    assert external_name.is_file()


@pytest.mark.parametrize("kind", ["symlink", "reparse"])
def test_first_visual_cleanup_rejects_link_or_reparse_asset(
    tmp_path: Path,
    monkeypatch,
    kind: str,
) -> None:
    work_dir, slide_data, files = _cleanup_assets(tmp_path)
    target = files[0]
    original_lstat = Path.lstat

    class ReparseStatus:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_file_attributes = getattr(
                stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
            )

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def fake_lstat(path):
        result = original_lstat(path)
        if Path(path) != target:
            return result
        if kind == "reparse":
            return ReparseStatus(result)
        values = list(result)
        values[0] = stat.S_IFLNK | 0o777
        return os.stat_result(values)

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(ValueError, match="identity"):
        image_to_ppt._remove_owned_first_visual_assets(slide_data, work_dir)

    assert target.is_file()


def test_first_visual_cleanup_rejects_reparse_owned_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    work_dir, slide_data, files = _cleanup_assets(tmp_path)
    components_dir = work_dir / "components"
    original_lstat = Path.lstat

    def fake_lstat(path):
        result = original_lstat(path)
        if Path(path) == components_dir:
            values = list(result)
            values[0] = stat.S_IFLNK | 0o777
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(ValueError, match="identity|link|reparse"):
        image_to_ppt._remove_owned_first_visual_assets(slide_data, work_dir)

    assert files[0].is_file()


def test_first_visual_cleanup_rejects_owned_directory_identity_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    work_dir, slide_data, files = _cleanup_assets(tmp_path)
    components_dir = work_dir / "components"
    original_lstat = Path.lstat
    calls = 0

    def fake_lstat(path):
        nonlocal calls
        result = original_lstat(path)
        if Path(path) == components_dir:
            calls += 1
            if calls == 2:
                class ChangedDirectoryStatus:
                    st_ino = result.st_ino + 1

                    def __getattr__(self, name):
                        return getattr(result, name)

                return ChangedDirectoryStatus()
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(RuntimeError, match="directory identity changed"):
        image_to_ppt._remove_owned_first_visual_assets(slide_data, work_dir)

    assert files[0].is_file()


def test_first_visual_cleanup_rejects_asset_identity_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    work_dir, slide_data, files = _cleanup_assets(tmp_path)
    target = files[0]
    original_lstat = Path.lstat
    calls = 0

    def fake_lstat(path):
        nonlocal calls
        result = original_lstat(path)
        if Path(path) == target:
            calls += 1
            if calls == 2:
                values = list(result)
                values[1] += 1
                return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(RuntimeError, match="identity changed"):
        image_to_ppt._remove_owned_first_visual_assets(slide_data, work_dir)

    assert target.is_file()


def test_prepared_page_v2_loads_with_empty_initial_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, _ = _prepare_rerun_fixture(tmp_path, monkeypatch)
    state_path = Path(prepared["state_path"])
    manifest = json.loads(state_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    for item in manifest["text_items"]:
        item.pop("rotation")
    manifest.pop("initial_diagnostics")
    manifest["assets"].pop("text_cleanup_mask")
    manifest["assets"].pop("foreground_evidence_mask")
    _rewrite_prepared_manifest(state_path, manifest)

    loaded = image_to_ppt.load_component_layers(state_path)

    assert loaded["initial_diagnostics"] == []


def test_prepared_page_v3_rejects_diagnostic_not_bound_to_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, _ = _prepare_rerun_fixture(tmp_path, monkeypatch)
    state_path = Path(prepared["state_path"])
    manifest = json.loads(state_path.read_text(encoding="utf-8"))
    manifest["initial_diagnostics"] = [{
        "kind": "unowned_raster_text",
        "source_sha256": "0" * 64,
        "candidate_id": "candidate_0001",
        "bbox": [10, 10, 30, 30],
        "views": [
            {"normalized_text": "nx", "confidence": 0.96},
            {"normalized_text": "ny", "confidence": 0.95},
        ],
    }]
    _rewrite_prepared_manifest(state_path, manifest)

    with pytest.raises(ValueError, match="diagnostic"):
        image_to_ppt.load_component_layers(state_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.update({"extra": "not allowed"}),
        lambda item: item.update({"candidate_id": "component_0001"}),
        lambda item: item.update({"bbox": [0, 0, 121, 70]}),
        lambda item: item.update({"bbox": [10, 10, 10, 20]}),
        lambda item: item["views"][0].update({"confidence": 0.5}),
        lambda item: item["views"][0].update({"normalized_text": "x" * 257}),
    ],
)
def test_prepared_page_v3_strictly_validates_diagnostics(
    tmp_path: Path,
    monkeypatch,
    mutation,
) -> None:
    prepared, _ = _prepare_rerun_fixture(tmp_path, monkeypatch)
    state_path = Path(prepared["state_path"])
    manifest = json.loads(state_path.read_text(encoding="utf-8"))
    item = {
        "kind": "unowned_raster_text",
        "source_sha256": manifest["assets"]["source_image"]["sha256"],
        "candidate_id": "candidate_0001",
        "bbox": [40, 20, 90, 50],
        "views": [
            {"normalized_text": "nx", "confidence": 0.96},
            {"normalized_text": "ny", "confidence": 0.95},
        ],
    }
    mutation(item)
    manifest["initial_diagnostics"] = [item]
    _rewrite_prepared_manifest(state_path, manifest)

    with pytest.raises(ValueError, match="diagnostic"):
        image_to_ppt.load_component_layers(state_path)


def test_prepared_page_v3_rejects_duplicate_diagnostic_candidate_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, _ = _prepare_rerun_fixture(tmp_path, monkeypatch)
    state_path = Path(prepared["state_path"])
    manifest = json.loads(state_path.read_text(encoding="utf-8"))
    item = {
        "kind": "unowned_raster_text",
        "source_sha256": manifest["assets"]["source_image"]["sha256"],
        "candidate_id": "candidate_0001",
        "bbox": [40, 20, 90, 50],
        "views": [
            {"normalized_text": "nx", "confidence": 0.96},
            {"normalized_text": "ny", "confidence": 0.95},
        ],
    }
    manifest["initial_diagnostics"] = [item, json.loads(json.dumps(item))]
    _rewrite_prepared_manifest(state_path, manifest)

    with pytest.raises(ValueError, match="diagnostic"):
        image_to_ppt.load_component_layers(state_path)


def test_initial_quality_evidence_contains_hash_bound_unowned_text_violation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared, _ = _prepare_rerun_fixture(tmp_path, monkeypatch)
    prepared["initial_diagnostics"] = [{
        "kind": "unowned_raster_text",
        "source_sha256": image_to_ppt.hashlib.sha256(
            Path(prepared["original_image_path"]).read_bytes()
        ).hexdigest(),
        "candidate_id": "candidate_0001",
        "bbox": [10, 10, 30, 30],
        "views": [
            {"normalized_text": "nx", "confidence": 0.96},
            {"normalized_text": "ny", "confidence": 0.95},
        ],
    }]
    store = RunStore(tmp_path / "run")
    store.write_json("job_manifest.json", {
        "schema_version": 1,
        "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    reconstruction = store.root / "pages/page_001/reconstruction"
    reconstruction.mkdir(parents=True)

    session = legacy._build_initial_page_session(
        store, "page_001", prepared, reconstruction
    )
    quality = json.loads(
        session["evidence"]["quality-report.json"].read_text(encoding="utf-8")
    )

    assert quality["initial_diagnostics"] == prepared["initial_diagnostics"]
    assert quality["violations"] == ["unowned_raster_text"]
