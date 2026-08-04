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
    candidate: np.ndarray, source: np.ndarray, donor_mask: np.ndarray,
    visual_hole: np.ndarray,
) -> dict[str, float]:
    empty = {
        "boundary_color_mae": 0.0,
        "gradient_jump_p95": 0.0,
        "added_high_frequency_pixels": 0.0,
    }
    if not np.any(visual_hole):
        return empty

    height, width = visual_hole.shape
    inside_y, inside_x = np.nonzero(visual_hole)
    boundary_errors: list[np.ndarray] = []
    gradient_errors: list[np.ndarray] = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        outside_y, outside_x = inside_y + dy, inside_x + dx
        valid = (
            (outside_y >= 0) & (outside_y < height)
            & (outside_x >= 0) & (outside_x < width)
        )
        valid_indices = np.flatnonzero(valid)
        if not valid_indices.size:
            continue
        oy, ox = outside_y[valid_indices], outside_x[valid_indices]
        visible = donor_mask[oy, ox]
        valid_indices = valid_indices[visible]
        if not valid_indices.size:
            continue
        iy, ix = inside_y[valid_indices], inside_x[valid_indices]
        oy, ox = outside_y[valid_indices], outside_x[valid_indices]
        boundary_errors.append(np.abs(
            candidate[iy, ix].astype(np.int16) - source[oy, ox].astype(np.int16)
        ))

        outer_y, outer_x = oy + dy, ox + dx
        has_outer = (
            (outer_y >= 0) & (outer_y < height)
            & (outer_x >= 0) & (outer_x < width)
        )
        gradient_indices = np.flatnonzero(has_outer)
        if not gradient_indices.size:
            continue
        o2y, o2x = outer_y[gradient_indices], outer_x[gradient_indices]
        visible_outer = donor_mask[o2y, o2x]
        gradient_indices = gradient_indices[visible_outer]
        if not gradient_indices.size:
            continue
        i = candidate[iy[gradient_indices], ix[gradient_indices]].astype(np.int16)
        o = source[oy[gradient_indices], ox[gradient_indices]].astype(np.int16)
        o2 = source[
            outer_y[gradient_indices], outer_x[gradient_indices]
        ].astype(np.int16)
        gradient_errors.append(np.mean(np.abs((i - o) - (o - o2)), axis=1))

    if not boundary_errors:
        return empty
    boundary_mae = float(np.concatenate(boundary_errors).mean())
    gradient_values = np.concatenate(gradient_errors) if gradient_errors else np.array([])
    gradient_jump_p95 = float(np.percentile(gradient_values, 95)) if gradient_values.size else 0.0

    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    candidate_detail = np.abs(cv2.Laplacian(candidate_gray, cv2.CV_32F))
    source_detail = np.abs(cv2.Laplacian(source_gray, cv2.CV_32F))
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    donor_ring = (
        cv2.dilate(visual_hole.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)).astype(bool)
        & ~cv2.dilate(visual_hole.astype(np.uint8), kernel3).astype(bool)
        & cv2.erode(donor_mask.astype(np.uint8), kernel3).astype(bool)
    )
    detail_threshold = (
        float(np.percentile(source_detail[donor_ring], 95)) + 12.0
        if np.any(donor_ring) else 12.0
    )
    interior = cv2.erode(
        visual_hole.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
    ).astype(bool)
    high_frequency = float(np.count_nonzero(
        interior & (candidate_detail > detail_threshold)
    ))
    return {
        "boundary_color_mae": boundary_mae,
        "gradient_jump_p95": gradient_jump_p95,
        "added_high_frequency_pixels": high_frequency,
    }


def _choose_visual_fill(
    *, rgb: np.ndarray, source_rgb: np.ndarray, semantic_mask: np.ndarray,
    donor_mask: np.ndarray, visual_hole: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    ys, xs = np.nonzero(semantic_mask)
    if not len(ys):
        return rgb.copy(), _visual_metrics(rgb, source_rgb, donor_mask, visual_hole)
    y0, y1 = max(0, int(ys.min()) - 8), min(rgb.shape[0], int(ys.max()) + 9)
    x0, x1 = max(0, int(xs.min()) - 8), min(rgb.shape[1], int(xs.max()) + 9)
    crop = rgb[y0:y1, x0:x1]
    mask = visual_hole[y0:y1, x0:x1].astype(np.uint8) * 255
    candidates = [
        cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA),
        cv2.inpaint(crop, mask, 3, cv2.INPAINT_NS),
    ]
    hole_crop = visual_hole[y0:y1, x0:x1]
    donor_crop = donor_mask[y0:y1, x0:x1]
    count, labels = cv2.connectedComponents(hole_crop.astype(np.uint8), 8)
    areas = [int(np.count_nonzero(labels == label)) for label in range(1, count)]
    if any(area >= 25 for area in areas):
        local_fill = candidates[1].copy()
        filled = False
        for label, area in zip(range(1, count), areas):
            if area < 25:
                continue
            component = labels == label
            radius = max(3, min(21, int(np.ceil(np.sqrt(area) * 0.15))))
            ring = (
                cv2.dilate(
                    component.astype(np.uint8),
                    np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8),
                ).astype(bool)
                & donor_crop
            )
            if not np.any(ring):
                continue
            local_fill[component] = np.median(crop[ring], axis=0).astype(np.uint8)
            filled = True
        if filled:
            candidates.append(local_fill)
    selected = rgb.copy()
    selected_metrics: dict[str, float] | None = None
    selected_key: tuple[float, ...] | None = None
    for candidate_crop in candidates:
        candidate = rgb.copy()
        candidate[y0:y1, x0:x1][visual_hole[y0:y1, x0:x1]] = candidate_crop[
            visual_hole[y0:y1, x0:x1]
        ]
        metrics = _visual_metrics(candidate, source_rgb, donor_mask, visual_hole)
        limits = (
            6.0,
            12.0,
            float(max(4, round(np.count_nonzero(visual_hole) * 0.005))),
        )
        values = (
            metrics["boundary_color_mae"],
            metrics["gradient_jump_p95"],
            metrics["added_high_frequency_pixels"],
        )
        ratios = tuple(value / limit for value, limit in zip(values, limits))
        key = (
            float(sum(value > limit for value, limit in zip(values, limits))),
            max(ratios),
            sum(ratios),
            *values,
        )
        if selected_key is None or key < selected_key:
            selected, selected_metrics, selected_key = candidate, metrics, key
    return selected, selected_metrics or _visual_metrics(
        selected, source_rgb, donor_mask, visual_hole,
    )


def _higher_layer_halo(
    ownership: np.ndarray,
    semantic: np.ndarray,
    higher_layer: np.ndarray,
) -> np.ndarray:
    ys, xs = np.nonzero(semantic)
    if not len(ys) or not np.any(higher_layer):
        return np.zeros_like(semantic)
    short_side = min(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
    if short_side < 20:
        return np.zeros_like(semantic)
    radius = max(1, min(4, int(np.ceil(short_side * 0.02))))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return (
        cv2.dilate(higher_layer.astype(np.uint8), kernel).astype(bool)
        & ownership
        & semantic
    )


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

    expanded_higher = higher_layer | _higher_layer_halo(
        ownership, semantic, higher_layer,
    )
    ownership = ownership & ~expanded_higher
    text_hole = semantic & ~ownership & text
    visual_hole = semantic & ~ownership & expanded_higher & ~text_hole
    if np.any(visual_hole) and not np.any(ownership):
        raise ValueError("visual hole requires at least one visible ownership donor")
    generated = text_hole | visual_hole
    rgb = np.asarray(text_clean_rgb, dtype=np.uint8).copy()
    if np.any(visual_hole):
        visual_fill, metrics = _choose_visual_fill(
            rgb=rgb, source_rgb=source, semantic_mask=semantic,
            donor_mask=ownership, visual_hole=visual_hole,
        )
        rgb[visual_hole] = visual_fill[visual_hole]
    else:
        metrics = _visual_metrics(rgb, source, ownership, visual_hole)

    return {
        "rgb": rgb,
        "ownership_mask": ownership,
        "presentation_alpha_mask": ownership | generated,
        "generated_underlay_mask": generated,
        "metrics": metrics,
    }
