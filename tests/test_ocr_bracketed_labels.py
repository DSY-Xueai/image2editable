"""Short bracketed labels remain editable instead of being discarded as noise."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(params=[
    "scripts/text_detect.py",
    "skills/image-to-ppt/scripts/text_detect.py",
    "skills/image-to-psd/scripts/text_detect.py",
])
def detector(request):
    path = Path(__file__).resolve().parents[1] / request.param
    spec = importlib.util.spec_from_file_location("bracketed_label_detector", path)
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


@pytest.mark.parametrize("pair", [
    "[]", "()", "{}", "\uff08\uff09", "\u3010\u3011",
    "\u3014\u3015", "\u3008\u3009", "\u300a\u300b",
])
@pytest.mark.parametrize("label", ["\u7532", "\u7532\u4e59", "A", "AB", "1"])
def test_filter_keeps_short_labels_in_matching_brackets(detector, pair, label):
    box = {"text": pair[0] + label + pair[1],
           "box": (10, 10, 90, 30), "confidence": 0.95}

    assert detector._filter_noise([box]) == [box]


@pytest.mark.parametrize("text", [
    "[]", "()", "\u3010\u3011", "[!!!]", "[A!!!]", "[A)", "[A",
    "A]", "!!A!!", "...", "[ ]",
])
def test_filter_still_rejects_punctuation_noise(detector, text):
    box = {"text": text, "box": (10, 10, 90, 30), "confidence": 0.99}

    assert detector._filter_noise([box]) == []


def test_bracketed_labels_still_require_confidence(detector):
    box = {"text": "[A]", "box": (10, 10, 90, 30), "confidence": 0.6}

    assert detector._filter_noise([box], confidence_threshold=0.7) == []


def test_bracketed_label_reaches_editable_text_and_mask(detector):
    box = {"text": "[A]", "box": (10, 10, 90, 30), "confidence": 0.99}
    image = np.full((60, 120, 3), 255, dtype=np.uint8)

    items, mask = detector._build_text_result(image, [box], 0.7, 2)

    assert [item["text"] for item in items] == ["[A]"]
    assert np.all(mask[10:40, 10:100] == 255)
