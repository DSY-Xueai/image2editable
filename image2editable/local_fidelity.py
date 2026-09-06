from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


_VISUAL_DIFFERENCE_THRESHOLD = 8.0


def build_local_fidelity_components(
    *,
    source_path: str | Path,
    candidate_path: str | Path,
    reliable_text_mask_path: str | Path,
    output_dir: str | Path,
    max_component_coverage: float = 0.35,
) -> dict[str, object]:
    """Return hash-bound RGBA patches for non-text residual pixels."""

    if (
        type(max_component_coverage) not in {int, float}
        or not math.isfinite(max_component_coverage)
        or not 0 < max_component_coverage <= 1
    ):
        raise ValueError("max_component_coverage must be between zero and one")
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB")).copy()
    with Image.open(candidate_path) as image:
        candidate = np.asarray(image.convert("RGB")).copy()
    with Image.open(reliable_text_mask_path) as image:
        text_mask = np.asarray(image.convert("L")) > 0
    if source.shape != candidate.shape or text_mask.shape != source.shape[:2]:
        raise ValueError("local fidelity input dimensions differ")

    difference = np.mean(
        np.abs(source.astype(np.float32) - candidate.astype(np.float32)),
        axis=2,
    )
    residual = (difference > _VISUAL_DIFFERENCE_THRESHOLD) & ~text_mask
    residual_pixels = int(np.count_nonzero(residual))
    if residual_pixels == 0:
        return {
            "components": [],
            "residual_pixels": 0,
            "uncovered_pixels": 0,
        }

    height, width = residual.shape
    page_area = height * width
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8), 8
    )
    regions: list[tuple[int, int, np.ndarray]] = []
    for label in range(1, count):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        region_width = int(stats[label, cv2.CC_STAT_WIDTH])
        region_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        mask = labels[
            top:top + region_height, left:left + region_width
        ] == label
        pending = [(left, top, mask)]
        while pending:
            region_left, region_top, region = pending.pop()
            ys, xs = np.nonzero(region)
            if not len(xs):
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            region = region[y1:y2, x1:x2]
            region_left += x1
            region_top += y1
            coverage = region.shape[0] * region.shape[1] / page_area
            if coverage <= max_component_coverage:
                regions.append((region_left, region_top, region.copy()))
                continue
            if region.shape[1] >= region.shape[0]:
                split = region.shape[1] // 2
                pending.extend((
                    (region_left + split, region_top, region[:, split:]),
                    (region_left, region_top, region[:, :split]),
                ))
            else:
                split = region.shape[0] // 2
                pending.extend((
                    (region_left, region_top + split, region[split:, :]),
                    (region_left, region_top, region[:split, :]),
                ))

    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=False)
    covered = np.zeros(residual.shape, dtype=bool)
    components = []
    for index, (left, top, mask) in enumerate(regions, start=1):
        region_height, region_width = mask.shape
        right = left + region_width
        bottom = top + region_height
        rgba = np.zeros((region_height, region_width, 4), dtype=np.uint8)
        rgba[:, :, :3] = source[top:bottom, left:right]
        rgba[:, :, 3] = mask.astype(np.uint8) * 255
        path = target / f"component-{index:04d}.png"
        Image.fromarray(rgba, mode="RGBA").save(path)
        covered[top:bottom, left:right] |= mask
        components.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bbox": [left, top, right, bottom],
            "coverage": region_width * region_height / page_area,
        })
    return {
        "components": components,
        "residual_pixels": residual_pixels,
        "uncovered_pixels": int(np.count_nonzero(residual & ~covered)),
    }
