from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import unicodedata

import cv2
import numpy as np


ERROR_METRICS = (
    "normalized_mae",
    "changed_ratio",
    "edge_symmetric_difference_ratio",
    "largest_error_region_ratio",
)


def _image(value: object, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 3
        or array.shape[2] != 3
        or array.dtype.kind not in "biuf"
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{label} image is invalid")
    return array.astype(np.float32, copy=False)


def _ssim(left: np.ndarray, right: np.ndarray) -> float:
    left_values = left.astype(np.float64, copy=False).reshape(-1)
    right_values = right.astype(np.float64, copy=False).reshape(-1)
    left_mean = float(left_values.mean())
    right_mean = float(right_values.mean())
    left_variance = float(left_values.var())
    right_variance = float(right_values.var())
    covariance = float(
        np.mean((left_values - left_mean) * (right_values - right_mean))
    )
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    denominator = (
        (left_mean**2 + right_mean**2 + c1)
        * (left_variance + right_variance + c2)
    )
    return (
        (2 * left_mean * right_mean + c1) * (2 * covariance + c2)
    ) / denominator


def _edges(image: np.ndarray) -> np.ndarray:
    rgb = np.clip(image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 100, 200) > 0


def _metrics(reference: np.ndarray, rendered: np.ndarray) -> dict:
    difference = np.abs(reference - rendered)
    changed = np.max(difference, axis=2) > 8
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        changed.astype(np.uint8), connectivity=8
    )
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    reference_edges = _edges(reference)
    rendered_edges = _edges(rendered)
    edge_union = reference_edges | rendered_edges
    edge_difference = reference_edges ^ rendered_edges
    return {
        "normalized_mae": float(difference.mean() / 255.0),
        "changed_ratio": float(np.count_nonzero(changed) / changed.size),
        "ssim": float(_ssim(reference, rendered)),
        "edge_symmetric_difference_ratio": float(
            np.count_nonzero(edge_difference) / max(np.count_nonzero(edge_union), 1)
        ),
        "largest_error_region_ratio": float(largest / changed.size),
    }


def _within_baseline(candidate: dict, baseline: dict, noise: dict) -> bool:
    epsilon = 1e-12
    errors_pass = all(
        candidate[name] <= baseline[name] + noise[name] + epsilon
        for name in ERROR_METRICS
    )
    ssim_pass = candidate["ssim"] >= (
        baseline["ssim"] - (1.0 - noise["ssim"]) - epsilon
    )
    return errors_pass and ssim_pass


def _region(value: object, shape: tuple[int, int], label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(coordinate) is not int for coordinate in value)
    ):
        raise ValueError(f"{label} region is invalid")
    left, top, right, bottom = value
    if not (0 <= left < right <= shape[1] and 0 <= top < bottom <= shape[0]):
        raise ValueError(f"{label} region is invalid")
    return value


def _normalize_text(value: object) -> str:
    if type(value) is not str:
        raise ValueError("OCR text is invalid")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _text_bbox(item: dict) -> list[float]:
    if "bbox" in item:
        bbox = item["bbox"]
    elif "box" in item:
        x, y, width, height = item["box"]
        bbox = [x, y, x + width, y + height]
    else:
        raise ValueError("OCR bbox is missing")
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(
            type(coordinate) not in {int, float} or not math.isfinite(coordinate)
            for coordinate in bbox
        )
        or bbox[0] >= bbox[2]
        or bbox[1] >= bbox[3]
    ):
        raise ValueError("OCR bbox is invalid")
    return [float(coordinate) for coordinate in bbox]


def _bbox_iou(left: list[float], right: list[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _text_metrics(expected: list[dict], rendered: list[dict]) -> tuple[dict, list[str]]:
    expected_records = [
        (_normalize_text(item.get("text")), _text_bbox(item)) for item in expected
    ]
    rendered_records = [
        (_normalize_text(item.get("text")), _text_bbox(item)) for item in rendered
    ]
    violations = []
    if Counter(text for text, _ in expected_records) != Counter(
        text for text, _ in rendered_records
    ):
        violations.append("text_mismatch")
    minimum_iou = 1.0
    expected_by_text = defaultdict(list)
    rendered_by_text = defaultdict(list)
    for text, bbox in expected_records:
        expected_by_text[text].append(bbox)
    for text, bbox in rendered_records:
        rendered_by_text[text].append(bbox)
    if not violations:
        for text, expected_boxes in expected_by_text.items():
            available = list(rendered_by_text[text])
            for expected_box in expected_boxes:
                scored = [(_bbox_iou(expected_box, box), index) for index, box in enumerate(available)]
                score, index = max(scored)
                minimum_iou = min(minimum_iou, score)
                available.pop(index)
        if minimum_iou < 0.8:
            violations.append("text_alignment_mismatch")
    return {
        "expected": [text for text, _ in expected_records],
        "rendered": [text for text, _ in rendered_records],
        "minimum_bbox_iou": minimum_iou if not violations or "text_mismatch" not in violations else None,
    }, violations


def compare_rendered_page(
    source: np.ndarray,
    raster_baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    object_regions: dict[str, list[int]],
    repeated_baseline: np.ndarray,
    expected_text_items: list[dict] | None = None,
    rendered_text_items: list[dict] | None = None,
) -> dict:
    """Compare a candidate render against source and repeatable Raster quality."""

    source_image = _image(source, "source")
    baseline_image = _image(raster_baseline, "raster baseline")
    candidate_image = _image(candidate, "candidate")
    repeated_image = _image(repeated_baseline, "repeated baseline")
    if not (
        source_image.shape
        == baseline_image.shape
        == candidate_image.shape
        == repeated_image.shape
    ):
        raise ValueError("render QA image dimensions differ")
    if not isinstance(object_regions, dict) or any(
        type(object_id) is not str or not object_id for object_id in object_regions
    ):
        raise ValueError("render QA object regions are invalid")

    baseline_metrics = _metrics(source_image, baseline_image)
    candidate_metrics = _metrics(source_image, candidate_image)
    noise_metrics = _metrics(baseline_image, repeated_image)
    page_passed = _within_baseline(
        candidate_metrics, baseline_metrics, noise_metrics
    )
    object_metrics = {}
    failed_object_ids = []
    for object_id, raw_region in object_regions.items():
        left, top, right, bottom = _region(
            raw_region, source_image.shape[:2], object_id
        )
        slices = (slice(top, bottom), slice(left, right))
        local_baseline = _metrics(source_image[slices], baseline_image[slices])
        local_candidate = _metrics(source_image[slices], candidate_image[slices])
        local_noise = _metrics(baseline_image[slices], repeated_image[slices])
        accepted = _within_baseline(local_candidate, local_baseline, local_noise)
        object_metrics[object_id] = {
            "baseline": local_baseline,
            "candidate": local_candidate,
            "noise": local_noise,
            "accepted": accepted,
        }
        if not accepted:
            failed_object_ids.append(object_id)

    violations = []
    if not page_passed or failed_object_ids:
        violations.append("render_difference")
    metrics = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "objects": object_metrics,
    }
    if (expected_text_items is None) != (rendered_text_items is None):
        raise ValueError("both expected and rendered OCR items are required")
    if expected_text_items is not None and rendered_text_items is not None:
        if not isinstance(expected_text_items, list) or not isinstance(
            rendered_text_items, list
        ):
            raise ValueError("OCR items are invalid")
        text_metrics, text_violations = _text_metrics(
            expected_text_items, rendered_text_items
        )
        metrics["text"] = text_metrics
        violations.extend(text_violations)

    return {
        "schema_version": 1,
        "accepted": not violations,
        "metrics": metrics,
        "failed_object_ids": failed_object_ids,
        "violations": violations,
        "noise_tolerance": {
            **noise_metrics,
            "ssim_drop": 1.0 - noise_metrics["ssim"],
        },
    }
