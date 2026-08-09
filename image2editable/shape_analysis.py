from __future__ import annotations

import cv2
import numpy as np


MIN_GEOMETRY_SCORE = 0.90


def _iou(mask: np.ndarray, prototype: np.ndarray) -> float:
    intersection = int(np.count_nonzero(mask & prototype))
    union = int(np.count_nonzero(mask | prototype))
    return intersection / union if union else 0.0


def _rectangle(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def _ellipse(height: int, width: int) -> np.ndarray:
    prototype = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        prototype,
        ((width - 1) // 2, (height - 1) // 2),
        ((width - 1) // 2, (height - 1) // 2),
        0,
        0,
        360,
        1,
        -1,
    )
    return prototype.astype(bool)


def _rounded_rectangle(height: int, width: int, radius: int) -> np.ndarray:
    prototype = np.zeros((height, width), dtype=np.uint8)
    right = width - 1
    bottom = height - 1
    cv2.rectangle(prototype, (radius, 0), (right - radius, bottom), 1, -1)
    cv2.rectangle(prototype, (0, radius), (right, bottom - radius), 1, -1)
    for center in (
        (radius, radius),
        (right - radius, radius),
        (radius, bottom - radius),
        (right - radius, bottom - radius),
    ):
        cv2.circle(prototype, center, radius, 1, -1)
    return prototype.astype(bool)


def _best_rounded(mask: np.ndarray) -> tuple[float, np.ndarray] | None:
    height, width = mask.shape
    maximum = min((width - 1) // 2, (height - 1) // 2)
    if maximum < 2:
        return None
    samples = np.unique(
        np.linspace(1, maximum, min(32, maximum), dtype=np.int32)
    ).tolist()
    measured = [
        (_iou(mask, prototype), radius, prototype)
        for radius in samples
        for prototype in [_rounded_rectangle(height, width, radius)]
    ]
    coarse = max(measured, key=lambda item: item[0])
    coarse_index = samples.index(coarse[1])
    lower = samples[max(0, coarse_index - 1)]
    upper = samples[min(len(samples) - 1, coarse_index + 1)]
    refined = [
        (_iou(mask, prototype), radius, prototype)
        for radius in range(lower, upper + 1)
        for prototype in [_rounded_rectangle(height, width, radius)]
    ]
    best = max(refined, key=lambda item: (item[0], -item[1]))
    return best[0], best[2]


def _best_line(mask: np.ndarray) -> tuple[float, np.ndarray] | None:
    height, width = mask.shape
    fill_ratio = np.count_nonzero(mask) / mask.size
    if max(height, width) / max(1, min(height, width)) < 3 and fill_ratio > 0.35:
        return None
    points = np.column_stack(np.nonzero(mask))[:, ::-1].astype(np.float32)
    direction_x, direction_y, _, _ = cv2.fitLine(
        points, cv2.DIST_L2, 0, 0.01, 0.01
    ).reshape(-1)
    direction = np.array([direction_x, direction_y], dtype=np.float32)
    center = points.mean(axis=0)
    projections = (points - center) @ direction
    start = center + direction * projections.min()
    end = center + direction * projections.max()
    length = max(float(projections.max() - projections.min()), 1.0)
    estimated = max(1, int(round(np.count_nonzero(mask) / length)))
    measured = []
    for thickness in range(max(1, estimated - 3), estimated + 4):
        prototype = np.zeros((height, width), dtype=np.uint8)
        cv2.line(
            prototype,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            1,
            thickness,
            cv2.LINE_8,
        )
        measured.append((_iou(mask, prototype.astype(bool)), prototype.astype(bool)))
    return max(measured, key=lambda item: item[0])


def _has_hole(mask: np.ndarray) -> bool:
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return False
    return any(
        hierarchy[0][index][3] >= 0 and cv2.contourArea(contour) >= 1.0
        for index, contour in enumerate(contours)
    )


def _fill_evidence(
    rgba: np.ndarray, mask: np.ndarray
) -> tuple[list[int], float] | None:
    interior = cv2.erode(
        mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=2
    ).astype(bool)
    if not np.any(interior):
        interior = mask
    pixels = rgba[interior]
    if np.any(pixels[:, 3] != 255):
        return None
    colors = pixels[:, :3].astype(np.float32)
    median = np.median(colors, axis=0)
    mad = np.median(np.abs(colors - median), axis=0)
    return [int(round(value)) for value in median], float(np.max(mad))


def analyze_shape_candidate(
    rgba: np.ndarray,
    mask: np.ndarray,
) -> dict | None:
    """Return measured evidence for one simple solid shape, never a route decision."""

    rgba = np.asarray(rgba)
    mask = np.asarray(mask)
    if (
        rgba.ndim != 3
        or rgba.shape[2] != 4
        or mask.ndim != 2
        or rgba.shape[:2] != mask.shape
        or mask.dtype.kind not in "biuf"
        or not np.all(np.isfinite(mask))
    ):
        raise ValueError("shape candidate arrays are invalid")
    binary = mask != 0
    if not np.any(binary):
        return None
    components, _ = cv2.connectedComponents(binary.astype(np.uint8))
    if components != 2 or _has_hole(binary):
        return None

    y_values, x_values = np.nonzero(binary)
    left, right = int(x_values.min()), int(x_values.max()) + 1
    top, bottom = int(y_values.min()), int(y_values.max()) + 1
    local = binary[top:bottom, left:right]
    height, width = local.shape
    candidates: list[tuple[str, float, np.ndarray]] = []
    rectangle = _rectangle(height, width)
    candidates.append(("rectangle", _iou(local, rectangle), rectangle))
    rounded = _best_rounded(local)
    if rounded is not None:
        candidates.append(("rounded_rectangle", rounded[0], rounded[1]))
    ellipse = _ellipse(height, width)
    candidates.append(("ellipse", _iou(local, ellipse), ellipse))
    line = _best_line(local)
    if line is not None:
        candidates.append(("line", line[0], line[1]))
    shape_type, geometry_score, _ = max(candidates, key=lambda item: item[1])
    if geometry_score < MIN_GEOMETRY_SCORE:
        return None
    fill = _fill_evidence(rgba[top:bottom, left:right], local)
    if fill is None:
        return None
    fill_rgb, color_mad = fill
    return {
        "shape_type": shape_type,
        "bbox": [left, top, right, bottom],
        "geometry_score": geometry_score,
        "fill_rgb": fill_rgb,
        "color_mad": color_mad,
    }
