from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import ocr_worker, text_detect


POLY = [[10, 10], [210, 10], [210, 50], [10, 50]]
WORDS = [{"text": "Hello", "box": [0.05, 0.0, 0.4, 1.0]},
         {"text": "world", "box": [0.55, 0.0, 0.4, 1.0]}]


def test_direct_ocr_requests_and_normalizes_words(monkeypatch):
    calls = []

    def predict(path, **kwargs):
        calls.append(kwargs)
        return [{"rec_texts": ["Hello world"], "rec_scores": [0.99],
                 "rec_polys": [POLY], "text_word": [["Hello", "world"]],
                 "text_word_region": [[[[20, 10], [100, 10], [100, 50], [20, 50]],
                                       [[120, 10], [200, 10], [200, 50], [120, 50]]]]}]

    monkeypatch.setattr(text_detect, "_get_paddleocr", lambda lang: SimpleNamespace(predict=predict))
    result = text_detect._try_paddleocr(Path("unused.png"), "en", 0.7)
    assert calls == [{"return_word_box": True}]
    assert result[0]["words"] == WORDS
    json.dumps(result)


@pytest.mark.parametrize("mode", ["split", "batch", "resident"])
def test_worker_words_survive_each_execution_mode(tmp_path, monkeypatch, mode):
    source = tmp_path / "source.png"
    Image.new("RGB", (240, 60), "white").save(source)
    calls = []
    created = []

    class Detector:
        def __init__(self, **kwargs):
            pass

        def predict(self, path, **kwargs):
            return [{"dt_polys": [POLY]}]

        def close(self):
            pass

    class Recognizer(Detector):
        def __init__(self, **kwargs):
            created.append(1)

        def predict(self, crops, **kwargs):
            calls.append(kwargs)
            return [{"rec_text": ("\u4e2dA\u6587", [20, [["\u4e2d"], ["A"], ["\u6587"]],
                                                      [[2], [8], [15]], ["cn", "en&num", "cn"]]),
                     "rec_score": 0.99} for crop in crops]

    class Cropper:
        def __init__(self, **kwargs):
            pass

        def __call__(self, image, polys):
            return [image[10:50, 10:210] for poly in polys]

    monkeypatch.setattr(ocr_worker, "_load_detection_tools", lambda: (Detector, lambda: lambda p: p, Cropper))
    monkeypatch.setattr(ocr_worker, "_load_recognition_model", lambda: Recognizer)
    monkeypatch.setattr(ocr_worker, "_resolve_recognition_model_name", lambda lang: "test")
    result_path = tmp_path / "result.json"
    if mode == "split":
        detection = tmp_path / "detect.json"
        ocr_worker.run_detection(source, tmp_path, detection)
        ocr_worker.run_recognition(detection, result_path)
        payload = json.loads(result_path.read_text(encoding="utf-8"))["items"]
    else:
        processor = ocr_worker._ResidentOcrProcessor() if mode == "resident" else None
        for _ in range(2 if processor else 1):
            ocr_worker.run_batch([source], result_path, processor=processor)
        if processor:
            processor.close()
        payload = json.loads(result_path.read_text(encoding="utf-8"))["images"][0]["items"]
    assert created == [1]
    assert all(call == {"return_word_box": True} for call in calls)
    assert payload[0]["text"] == "\u4e2dA\u6587"
    words = payload[0]["words"]
    assert [word["text"] for word in words] == ["\u4e2d", "A", "\u6587"]
    centers = [word["box"][0] + word["box"][2] / 2 for word in words]
    assert centers == sorted(centers)
    assert all(0 <= value <= 1 for word in words for value in word["box"])


@pytest.mark.parametrize("mode", ["split", "batch", "resident"])
def test_isolated_reader_preserves_words(tmp_path, monkeypatch, mode):
    source = tmp_path / "source.png"
    item = {"poly": POLY, "text": "Hello world", "score": 0.99, "words": WORDS}

    def run(command, **kwargs):
        output = Path(command[command.index("--result") + 1])
        payload = {"items": [item]} if "recognize" in command else {"images": [{"items": [item]}]}
        output.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    def request(payload, **kwargs):
        Path(payload["result"]).write_text(json.dumps({"images": [{"items": [item]}]}), encoding="utf-8")

    monkeypatch.setattr(text_detect, "run_isolated_worker", run)
    if mode == "split":
        result = text_detect._try_isolated_paddleocr(source, "en", 0.7, worker_root=tmp_path)
    else:
        result = text_detect._try_isolated_paddleocr_batch(
            [source], "en", 0.7, worker_root=tmp_path,
            worker_pool=SimpleNamespace(request=request) if mode == "resident" else None,
        )[0]
    assert result[0]["words"] == WORDS


def test_style_build_preserves_words_and_discards_changed_text(monkeypatch):
    monkeypatch.setattr(text_detect, "_estimate_style", lambda *a, **k: {"font_size": 18, "color": "#000000", "bold": False})
    image = np.full((80, 250, 3), 255, dtype=np.uint8)
    for text, expected in [(" Hello world ", True), ("|Hello world", False)]:
        raw = [{"box": [10, 10, 200, 40], "text": text, "confidence": 0.99,
                "words": ([{"text": "|", "box": [0, 0, 0.04, 1]}] if text.startswith("|") else []) + WORDS}]
        result, _ = text_detect._build_text_result(image, raw, 0.7, 0)
        assert ("words" in result[0]) is expected


def test_merge_remaps_word_geometry():
    left = {"box": [10, 10, 100, 40], "text": "Hello", "words": [{"text": "Hello", "box": [0, 0, 1, 1]}]}
    right = {"box": [120, 20, 80, 20], "text": "world", "words": [{"text": "world", "box": [0, 0, 1, 1]}]}
    merged = text_detect._merge_text_pair(left, right)
    assert merged["words"][0]["box"] == pytest.approx([0, 0, 100 / 190, 1])
    assert merged["words"][1]["box"] == pytest.approx([110 / 190, .25, 80 / 190, .5])


@pytest.mark.parametrize("words", [
    [{"text": "wrong", "box": [0, 0, 1, 1]}],
    [{"text": "Hello world", "box": [0, 0, float("nan"), 1]}],
    [{"text": "Hello world", "box": [0, 0, 2, 1]}],
])
def test_invalid_word_geometry_is_rejected(words):
    assert text_detect._validated_words("Hello world", words) == []


def test_partial_geometry_does_not_merge_away_valid_words():
    left = {"box": [10, 10, 100, 40], "text": "Hello", "words": [{"text": "Hello", "box": [0, 0, 1, 1]}]}
    right = {"box": [112, 10, 80, 40], "text": "world"}
    assert len(text_detect._merge_adjacent_text_items([left, right])) == 2


@pytest.mark.parametrize("styled_side", ["left", "right", "both"])
def test_positioned_styles_are_not_destroyed_by_plain_ocr_merge(styled_side):
    left = {"box": [10, 10, 100, 40], "text": "Hello"}
    right = {"box": [112, 10, 80, 40], "text": "world"}
    for side, item in (("left", left), ("right", right)):
        if styled_side in {side, "both"}:
            item["runs"] = [{"text": item["text"], "box": [0, 0, 1, 1],
                             "color": "#abcdef", "rotation": 12}]
    assert text_detect._merge_adjacent_text_items([left, right]) == [left, right]


def test_worker_rejects_cross_group_column_reordering():
    result = {"rec_text": ("AB", [20, [["A"], ["B"]], [[15], [2]], ["en&num", "en&num"]]), "rec_score": 0.99}
    assert "words" not in ocr_worker._recognition_item(result, POLY)


def test_word_boxes_clip_only_small_rounding_overflow():
    result = text_detect._validated_words("A", [{"text": "A", "box": [-.001, 0, 1.002, 1]}])
    assert result == [{"text": "A", "box": [0, 0, 1, 1]}]


def test_worker_uses_same_minimum_rectangle_as_cropper():
    from paddlex.inference.pipelines.components import CropByPolys

    poly = [[20, 20], [210, 50], [200, 100], [10, 65]]
    captured = []
    cropper = CropByPolys(det_box_type="quad")
    cropper.get_rotate_crop_image = lambda image, points: captured.append(points)
    cropper.get_minarea_rect_crop(np.zeros((120, 240, 3), dtype=np.uint8), np.asarray(poly))
    result = {"rec_text": ("AB", [20, [["A"], ["B"]], [[5], [12]], ["en&num", "en&num"]]), "rec_score": 0.99}
    words = ocr_worker._recognition_item(result, poly)["words"]
    quad = captured[0]
    expected = quad[0] * .75 + quad[1] * .25
    assert words[0]["box"][1] == pytest.approx((expected[1] - 20) / 80, abs=.002)


def test_direct_rejects_regions_in_reversed_reading_order():
    regions = [np.asarray(POLY) + [100, 0], np.asarray(POLY)]
    assert text_detect._words_from_polys("AB", ["A", "B"], regions, [10, 10, 300, 40]) == []


def test_vertical_worker_word_positions_follow_rotated_crop():
    result = {"rec_text": ("AB", [20, [["A"], ["B"]], [[2], [15]], ["en&num", "en&num"]]), "rec_score": .99}
    words = ocr_worker._recognition_item(result, [[10, 10], [50, 10], [50, 210], [10, 210]])["words"]
    assert words[0]["box"] == pytest.approx([0, .1, 1, .05])
    assert words[1]["box"] == pytest.approx([0, .75, 1, .05])
