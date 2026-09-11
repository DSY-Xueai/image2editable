from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(params=["image_to_ppt.py", "skills/image-to-ppt/scripts/image_to_ppt.py", "skills/image-to-psd/scripts/image_to_ppt.py"])
def pipeline(request):
    path = Path(__file__).resolve().parents[1] / request.param
    spec = importlib.util.spec_from_file_location("page_signal_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_ocr_overlap_does_not_describe_whole_page_as_occluded(pipeline, monkeypatch):
    captured = []
    monkeypatch.setattr(pipeline, "classify_page", lambda signals: captured.append(signals))
    items = [{"box": [20, 20 + row * 35, 160, 20], "confidence": 0.99} for row in range(10)]
    items.append({"box": [20, 25, 160, 20], "confidence": 0.99})
    pipeline._infer_page_policy(np.full((400, 600, 3), 255, np.uint8), items, source_kind="image", pipeline_mode="fast")
    assert captured[0].overlap_ratio == pytest.approx(2 * 2400 / (11 * 3200))
    assert captured[0].overlap_ratio < 0.30


def test_repeated_overlapping_ocr_regions_still_have_high_overlap(pipeline, monkeypatch):
    captured = []
    monkeypatch.setattr(pipeline, "classify_page", lambda signals: captured.append(signals))
    items = [{"box": [20, 20, 160, 20], "confidence": 0.99} for _ in range(3)]
    pipeline._infer_page_policy(np.full((100, 300, 3), 255, np.uint8), items, source_kind="image", pipeline_mode="fast")
    assert captured[0].overlap_ratio == 1.0
