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


def _prepare_rerun_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    include_diagnostic: bool = False,
    diagnostics_count: int | None = None,
    check_first_pass_cleanup: bool = False,
) -> tuple[dict, list[int]]:
    source = _label_fixture(tmp_path)
    work_dir = tmp_path / "prepared"
    process_text_counts = []
    stale_component = work_dir / "components/stale.png"
    stale_child = work_dir / "element-masks/stale.png"
    stale_parent = work_dir / "semantic-masks/stale.png"
    outside_component = tmp_path / "outside-component.png"

    monkeypatch.setattr(
        image_to_ppt,
        "detect_text",
        lambda *args, **kwargs: (
            [], np.zeros((70, 120), dtype=np.uint8)
        ),
    )
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)
    monkeypatch.setattr(image_to_ppt, "_release_visual_resources", lambda: None)
    monkeypatch.setattr(
        image_to_ppt,
        "_build_text_cleanup_mask",
        lambda *args, **kwargs: np.zeros((70, 120), dtype=np.uint8),
    )
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
        components_dir = target / "components"
        child_dir = target / "element-masks"
        parent_dir = target / "semantic-masks"
        components_dir.mkdir(exist_ok=True)
        child_dir.mkdir(exist_ok=True)
        parent_dir.mkdir(exist_ok=True)
        component_path = components_dir / "component_0000.png"
        Image.new(
            "RGBA", (20, 20), "red" if pass_index == 1 else "green"
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
