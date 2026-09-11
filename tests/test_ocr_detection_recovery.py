from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import image_to_ppt
from scripts import ocr_worker, text_detect


POLY = [[12, 8], [252, 8], [252, 48], [12, 48]]


def test_recognition_uses_filtered_polygons(monkeypatch):
    class Engine:
        def predict(self, path, *, return_word_box=False):
            return [{
                "rec_texts": ["Heading"], "rec_scores": [0.99],
                "dt_polys": [[[1, 1], [3, 1], [3, 3], [1, 3]], POLY],
                "rec_polys": [POLY],
            }]

    monkeypatch.setattr(text_detect, "_get_paddleocr", lambda lang: Engine())
    result = text_detect._try_paddleocr(Path("unused.png"), "en", 0.7)
    assert result[0]["box"] == (12, 8, 240, 40)


@pytest.mark.parametrize("score, expected", [(0.99, ["Heading"]), (0.85, [])])
def test_empty_crop_recovery_requires_high_recognition_confidence(
    tmp_path, monkeypatch, score, expected,
):
    source = tmp_path / "crop.png"
    Image.new("RGB", (280, 60), "white").save(source)
    calls = []

    class Engine:
        def predict(self, path, *, return_word_box=False, **kwargs):
            calls.append(kwargs)
            return [{
                "rec_texts": ["Heading"] if kwargs else [],
                "rec_scores": [score] if kwargs else [],
                "rec_polys": [POLY] if kwargs else [],
                "dt_polys": [POLY] if kwargs else [],
            }]

    monkeypatch.setattr(text_detect, "_get_paddleocr", lambda lang: Engine())
    results = text_detect.detect_text_batch([source], recover_empty=True)
    assert [item["text"] for item in results[0][0]] == expected
    assert calls == [{}, {"text_det_thresh": 0.15, "text_det_box_thresh": 0.3}]


def test_default_page_ocr_does_not_repeat_empty_detection(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 60), "white").save(source)
    calls = []

    class Engine:
        def predict(self, path, *, return_word_box=False):
            calls.append(path)
            return []

    monkeypatch.setattr(text_detect, "_get_paddleocr", lambda lang: Engine())
    assert text_detect.detect_text(source)[0] == []
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["split", "batch", "resident"])
@pytest.mark.parametrize("score, expected", [(0.99, ["Heading"]), (0.85, [])])
def test_isolated_empty_crop_recovery(tmp_path, monkeypatch, mode, score, expected):
    source = tmp_path / "crop.png"
    Image.new("RGB", (280, 60), "white").save(source)
    calls = []

    class Detector:
        def __init__(self, **kwargs):
            pass

        def predict(self, path, **kwargs):
            calls.append(kwargs)
            return [{"dt_polys": [POLY] if "thresh" in kwargs else []}]

        def close(self):
            pass

    class Sorter:
        def __call__(self, polys):
            return polys

    class Cropper:
        def __init__(self, **kwargs):
            pass

        def __call__(self, image, polys):
            return [image[8:48, 12:252].copy() for poly in polys]

    class Recognizer(Detector):
        def predict(self, crops, *, return_word_box=False):
            return [{"rec_text": "Heading", "rec_score": score} for crop in crops]

    monkeypatch.setattr(ocr_worker, "_load_detection_tools", lambda: (Detector, Sorter, Cropper))
    monkeypatch.setattr(ocr_worker, "_load_recognition_model", lambda: Recognizer)
    monkeypatch.setattr(ocr_worker, "_resolve_recognition_model_name", lambda lang: "test")
    output = tmp_path / "result.json"
    if mode == "split":
        detection = tmp_path / "detection.json"
        ocr_worker.run_detection(source, tmp_path, detection, recover_empty=True)
        ocr_worker.run_recognition(detection, output)
        items = json.loads(output.read_text(encoding="utf-8"))["items"]
    else:
        processor = ocr_worker._ResidentOcrProcessor() if mode == "resident" else None
        try:
            ocr_worker.run_batch([source], output, processor=processor, recover_empty=True)
        finally:
            if processor:
                processor.close()
        items = json.loads(output.read_text(encoding="utf-8"))["images"][0]["items"]
    assert [item["text"] for item in items] == expected
    assert calls == [
        {"max_side_limit": 4000},
        {"max_side_limit": 4000, "thresh": 0.15, "box_thresh": 0.3},
    ]


def test_targeted_ocr_recovers_wide_title_when_page_has_other_text(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (1600, 900), "white").save(source)
    calls = []

    class Engine:
        def predict(self, path, *, return_word_box=False, **kwargs):
            calls.append(kwargs)
            with Image.open(path) as crop:
                poly = [[0, 0], [crop.width, 0], [crop.width, crop.height], [0, crop.height]]
            return [{
                "rec_texts": ["Curved colored heading"] if kwargs else [],
                "rec_scores": [0.99] if kwargs else [],
                "rec_polys": [poly] if kwargs else [],
                "dt_polys": [poly] if kwargs else [],
            }]

    monkeypatch.setattr(text_detect, "_get_paddleocr", lambda lang: Engine())
    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source, [{"x": 100, "y": 40, "w": 1200, "h": 160, "area": 100000}],
        [{"text": "Existing body", "box": [120, 700, 200, 60], "confidence": 0.99}],
        np.zeros((900, 1600), dtype=np.uint8), tmp_path, lang="en", isolated=False,
    )
    assert [item["text"] for item in result["recovered_items"]] == ["Curved colored heading"]
    assert len(calls) == 4


def test_wide_title_crop_retains_context_outside_component_box(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    pixels = np.full((900, 1600, 3), 255, dtype=np.uint8)
    pixels[80:120, 1305:1320] = 0
    Image.fromarray(pixels).save(source)
    context_visible = []

    def inspect_crops(paths, **kwargs):
        for path in paths:
            with Image.open(path) as crop:
                context_visible.append(np.asarray(crop).min() < 128)
        return [([], np.zeros((1, 1), dtype=np.uint8)) for path in paths]

    monkeypatch.setattr(image_to_ppt, "detect_text_batch", inspect_crops)
    image_to_ppt._targeted_candidate_ocr_sweep(
        source, [{"x": 100, "y": 40, "w": 1200, "h": 160, "area": 100000}],
        [], np.zeros((900, 1600), dtype=np.uint8), tmp_path, lang="en", isolated=False,
    )
    assert context_visible == [True, True]
