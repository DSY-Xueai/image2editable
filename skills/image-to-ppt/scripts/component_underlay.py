"""Deterministic presentation-layer underlay reconstruction."""

from __future__ import annotations

import cv2
import numpy as np


def _rgb_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape (height, width, 3)")
    if array.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8")
    return array


def _mask_array(name: str, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype.kind not in "biu":
        raise TypeError(f"{name} must have a boolean or integer dtype")
    return array.astype(bool, copy=False)


def _visual_metrics(
    candidate: np.ndarray, source: np.ndarray, visual_hole: np.ndarray,
) -> dict[str, float]:
    if not np.any(visual_hole):
        return {
            "boundary_color_mae": 0.0,
            "gradient_jump_p95": 0.0,
            "added_high_frequency_pixels": 0.0,
        }

    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = visual_hole & cv2.dilate((~visual_hole).astype(np.uint8), kernel).astype(bool)
    difference = np.abs(candidate.astype(np.int16) - source.astype(np.int16))
    boundary_mae = float(difference[boundary].mean()) if np.any(boundary) else 0.0

    edge_errors: list[np.ndarray] = []
    for axis in (0, 1):
        candidate_gradient = np.diff(candidate.astype(np.int16), axis=axis)
        source_gradient = np.diff(source.astype(np.int16), axis=axis)
        edge_hole = np.logical_or(
            np.take(visual_hole, range(1, visual_hole.shape[axis]), axis=axis),
            np.take(visual_hole, range(0, visual_hole.shape[axis] - 1), axis=axis),
        )
        edge_errors.append(np.abs(candidate_gradient - source_gradient)[edge_hole])
    gradient_values = np.concatenate(edge_errors)
    gradient_jump_p95 = float(np.percentile(gradient_values, 95)) if gradient_values.size else 0.0

    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    candidate_detail = np.abs(cv2.Laplacian(candidate_gray, cv2.CV_32F))
    source_detail = np.abs(cv2.Laplacian(source_gray, cv2.CV_32F))
    high_frequency = float(np.count_nonzero(
        visual_hole & (candidate_detail > source_detail + 12.0)
    ))
    return {
        "boundary_color_mae": boundary_mae,
        "gradient_jump_p95": gradient_jump_p95,
        "added_high_frequency_pixels": high_frequency,
    }


def _choose_visual_fill(
    *, rgb: np.ndarray, source_rgb: np.ndarray, semantic_mask: np.ndarray,
    visual_hole: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    ys, xs = np.nonzero(semantic_mask)
    if not len(ys):
        return rgb.copy(), _visual_metrics(rgb, source_rgb, visual_hole)
    y0, y1 = max(0, int(ys.min()) - 8), min(rgb.shape[0], int(ys.max()) + 9)
    x0, x1 = max(0, int(xs.min()) - 8), min(rgb.shape[1], int(xs.max()) + 9)
    crop = rgb[y0:y1, x0:x1]
    mask = visual_hole[y0:y1, x0:x1].astype(np.uint8) * 255
    candidates = (
        cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA),
        cv2.inpaint(crop, mask, 3, cv2.INPAINT_NS),
    )
    selected = rgb.copy()
    selected_metrics: dict[str, float] | None = None
    selected_key: tuple[float, float, float] | None = None
    for candidate_crop in candidates:
        candidate = rgb.copy()
        candidate[y0:y1, x0:x1][visual_hole[y0:y1, x0:x1]] = candidate_crop[
            visual_hole[y0:y1, x0:x1]
        ]
        metrics = _visual_metrics(candidate, source_rgb, visual_hole)
        key = (
            metrics["boundary_color_mae"],
            metrics["gradient_jump_p95"],
            metrics["added_high_frequency_pixels"],
        )
        if selected_key is None or key < selected_key:
            selected, selected_metrics, selected_key = candidate, metrics, key
    return selected, selected_metrics or _visual_metrics(selected, source_rgb, visual_hole)


def build_presentation_layer(
    *,
    source_rgb: np.ndarray,
    text_clean_rgb: np.ndarray,
    ownership_mask: np.ndarray,
    semantic_mask: np.ndarray,
    higher_layer_mask: np.ndarray,
    text_mask: np.ndarray,
) -> dict:
    """Build a movable component appearance without changing owned pixels."""
    source = _rgb_array("source_rgb", source_rgb)
    text_clean = _rgb_array("text_clean_rgb", text_clean_rgb)
    if source.shape != text_clean.shape:
        raise ValueError("source_rgb and text_clean_rgb must have the same shape")
    shape = source.shape[:2]
    ownership = _mask_array("ownership_mask", ownership_mask, shape)
    semantic = _mask_array("semantic_mask", semantic_mask, shape)
    higher_layer = _mask_array("higher_layer_mask", higher_layer_mask, shape)
    text = _mask_array("text_mask", text_mask, shape)
    if np.any(ownership & ~semantic):
        raise ValueError("ownership_mask must be contained by semantic_mask")

    text_hole = semantic & ~ownership & text
    visual_hole = semantic & ~ownership & higher_layer & ~text_hole
    generated = text_hole | visual_hole
    rgb = np.asarray(text_clean_rgb, dtype=np.uint8).copy()
    if np.any(visual_hole):
        visual_fill, metrics = _choose_visual_fill(
            rgb=rgb, source_rgb=source, semantic_mask=semantic,
            visual_hole=visual_hole,
        )
        rgb[visual_hole] = visual_fill[visual_hole]
    else:
        metrics = _visual_metrics(rgb, source, visual_hole)

    return {
        "rgb": rgb,
        "ownership_mask": ownership,
        "presentation_alpha_mask": ownership | generated,
        "generated_underlay_mask": generated,
        "metrics": metrics,
    }
