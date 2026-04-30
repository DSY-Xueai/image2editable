#!/usr/bin/env python3
"""Foreground extraction and component splitting module.

Extracts non-background foreground elements from an image by comparing
against a background model, then splits the foreground into independent
transparent PNG components via connected-component analysis.

Usage:
    from fg_extract import extract_foreground_mask, split_components
    mask = extract_foreground_mask(img, bg, text_mask)
    components = split_components(img, mask, output_dir)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_foreground_mask(
    img: np.ndarray,
    bg: np.ndarray,
    text_mask: np.ndarray,
    diff_threshold: float = 20.0,
) -> np.ndarray:
    """Extract foreground binary mask using multiple detection methods.

    Uses three complementary approaches:
    1. Direct color distance from background color (primary, model-independent)
    2. Diff against background model (secondary, benefits from iterative refinement)
    3. Edge detection with color-aware filtering (captures fine details)

    Args:
        img: Original image (H, W, 3) RGB uint8.
        bg: Background model (H, W, 3) RGB uint8.
        text_mask: Binary mask (H, W) where text regions = 255.
        diff_threshold: Base threshold for foreground detection.

    Returns:
        Cleaned foreground mask (H, W) uint8, foreground = 255.
    """
    h, w = img.shape[:2]

    # === Method 1: Direct color distance from background color ===
    # This is the PRIMARY method — it doesn't depend on background model quality.
    # Estimate bg_color from the bg image edges (robust to foreground contamination)
    bg_color = _estimate_bg_color(bg)
    color_dist = np.linalg.norm(
        img.astype(np.float32) - bg_color, axis=2
    )
    # Threshold: slightly above diff_threshold to avoid background noise
    color_threshold = diff_threshold * 1.25
    color_mask = color_dist > color_threshold

    # === Method 2: Diff against background model ===
    # Complements color distance — effective when bg model is accurate (2nd pass)
    diff = np.linalg.norm(
        img.astype(np.float32) - bg.astype(np.float32), axis=2
    )
    diff_mask = diff > diff_threshold

    # HSV saturation boost: catch colored elements with moderate diff
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    bg_hsv = cv2.cvtColor(bg, cv2.COLOR_RGB2HSV)
    bg_sat_mean = float(np.mean(bg_hsv[:, :, 1]))
    sat_threshold = max(30.0, bg_sat_mean + 20.0)
    sat_mask = (diff > diff_threshold * 0.65) & (sat > sat_threshold)

    # Brightness diff
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY).astype(np.float32)
    brightness_diff = np.abs(gray_img - gray_bg)
    bright_mask = brightness_diff > diff_threshold * 1.2

    # === Method 3: Edge-based detection ===
    # Captures fine details (thin lines, icon outlines) using color distance
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Dilate edges slightly to include adjacent pixels
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    # Edge pixels with any color difference from bg are foreground
    edge_fg = (edges_dilated > 0) & (color_dist > diff_threshold * 0.5)

    # === Combine all methods ===
    mask = color_mask | diff_mask | sat_mask | bright_mask | edge_fg

    # Exclude text regions — text will be rebuilt as editable text boxes
    if text_mask is not None:
        mask[text_mask > 0] = False

    mask = mask.astype(np.uint8) * 255

    # Morphological cleanup
    # CLOSE: fill small holes inside foreground regions (beneficial)
    kernel_close = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    # OPEN: remove isolated noise pixels, but use 2x2 kernel to preserve thin lines
    kernel_open = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Remove small noise blobs and edge artifacts
    mask = _remove_noise(mask, img_shape=(h, w))

    logger.info(
        "Foreground mask: %d non-zero pixels (%.1f%%)",
        int(np.count_nonzero(mask)),
        np.count_nonzero(mask) / (h * w) * 100,
    )
    return mask


def split_components(
    img: np.ndarray,
    fg_mask: np.ndarray,
    output_dir: str | Path,
    min_area: int = 20,
    padding: int = 3,
) -> list[dict]:
    """Split foreground mask into independent transparent PNG components.

    Uses distance-transform-based splitting to separate elements connected
    by thin bridges (e.g., connecting lines between icons). This produces
    finer-grained components than simple connected-component analysis.

    Args:
        img: Original image (H, W, 3) RGB uint8.
        fg_mask: Foreground binary mask (H, W) uint8.
        output_dir: Directory to save component PNGs.
        min_area: Minimum component area in pixels.
        padding: Pixels to pad around each component bounding box.

    Returns:
        List of component dicts with keys: path, x, y, w, h, area.
        Sorted by area descending.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_h, img_w = img.shape[:2]

    # Split connected foreground into individual elements
    label_map = _split_by_distance_transform(fg_mask, min_area)

    # Extract each labeled component
    components: list[dict] = []
    num_labels = label_map.max()

    for i in range(1, num_labels + 1):
        comp_mask_full = (label_map == i).astype(np.uint8) * 255
        area = int(np.count_nonzero(comp_mask_full))

        if area < min_area:
            continue

        # Bounding box
        ys, xs = np.where(comp_mask_full > 0)
        if len(ys) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        # Pad bounding box
        x1 = max(0, x_min - padding)
        y1 = max(0, y_min - padding)
        x2 = min(img_w, x_max + 1 + padding)
        y2 = min(img_h, y_max + 1 + padding)

        # Extract component mask and apply alpha feathering
        comp_mask = comp_mask_full[y1:y2, x1:x2]
        comp_alpha = cv2.GaussianBlur(comp_mask, (3, 3), 0)

        # Crop RGB and combine with alpha
        crop_rgb = img[y1:y2, x1:x2]
        rgba = np.dstack([crop_rgb, comp_alpha])

        # Save as transparent PNG
        comp_path = output_dir / f"component_{i:04d}.png"
        Image.fromarray(rgba.astype(np.uint8)).save(str(comp_path))

        components.append({
            "path": str(comp_path),
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "area": area,
        })

    # Sort by area descending (largest first)
    components.sort(key=lambda c: c["area"], reverse=True)

    logger.info("Split into %d foreground components.", len(components))
    return components


def _split_by_distance_transform(
    fg_mask: np.ndarray, min_area: int = 20
) -> np.ndarray:
    """Split connected foreground regions into individual elements.

    Uses a per-component adaptive strategy:
    - Small components (< 0.1% of image area): kept intact, no splitting
    - Large components: split using adaptive threshold based on their
      internal distance distribution (30% of 75th percentile distance)

    This avoids over-splitting small elements while aggressively splitting
    large connected groups where distinct elements are joined by thin bridges.

    Args:
        fg_mask: Binary foreground mask (H, W) uint8.
        min_area: Minimum area for a component.

    Returns:
        Label map (H, W) int32 where each pixel is assigned a component ID.
    """
    h, w = fg_mask.shape

    # Distance transform for the entire mask
    dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)

    # Find connected components in original mask
    num_orig, orig_labels, orig_stats, _ = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8
    )

    # Adaptive size threshold: only split components larger than this
    # Scale with image resolution (~0.1% of image area, min 500px)
    img_area = h * w
    split_size_threshold = max(500, int(img_area * 0.001))

    # Phase 1: Assign seeds per component
    all_seeds = np.zeros((h, w), dtype=np.int32)
    next_label = 1

    for i in range(1, num_orig):
        area = orig_stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        comp_mask = (orig_labels == i)

        if area < split_size_threshold:
            # Small component: keep as single element
            all_seeds[comp_mask] = next_label
            next_label += 1
        else:
            # Large component: split using per-component adaptive threshold
            comp_dists = dist[comp_mask]
            comp_dists = comp_dists[comp_dists > 0]

            if len(comp_dists) == 0:
                all_seeds[comp_mask] = next_label
                next_label += 1
                continue

            # Threshold = 30% of 75th percentile distance
            # This breaks bridges much thinner than the main element bodies
            p75 = float(np.percentile(comp_dists, 75))
            comp_threshold = max(1.5, p75 * 0.3)

            # Find cores within this component
            comp_cores = (dist > comp_threshold) & comp_mask
            cores_uint8 = comp_cores.astype(np.uint8) * 255
            num_seeds, seed_labels = cv2.connectedComponents(
                cores_uint8, connectivity=8
            )

            if num_seeds <= 2:
                # Single core or no split — keep as one element
                all_seeds[comp_mask] = next_label
                next_label += 1
            else:
                # Multiple cores — each becomes a seed
                for s in range(1, num_seeds):
                    seed_pixels = (seed_labels == s)
                    seed_area = np.count_nonzero(seed_pixels)
                    if seed_area >= min_area:
                        all_seeds[seed_pixels] = next_label
                        next_label += 1
                    # Tiny seed fragments are left unassigned (will be claimed in expansion)

    # Phase 2: Expand seeds to claim remaining foreground pixels
    assigned = all_seeds.astype(np.float32)
    fg_bool = fg_mask > 0
    remaining = fg_bool & (assigned == 0)

    kernel = np.ones((3, 3), np.uint8)

    for _ in range(300):
        if not np.any(remaining):
            break
        dilated = cv2.dilate(assigned, kernel, iterations=1)
        new_pixels = remaining & (dilated > 0)
        if not np.any(new_pixels):
            break
        assigned[new_pixels] = dilated[new_pixels]
        remaining = fg_bool & (assigned == 0)

    assigned = assigned.astype(np.int32)

    # Phase 3: Remaining unassigned pixels become their own components
    if np.any(remaining):
        remaining_mask = remaining.astype(np.uint8) * 255
        num_rem, rem_labels = cv2.connectedComponents(
            remaining_mask, connectivity=8
        )
        for i in range(1, num_rem):
            rem_area = np.count_nonzero(rem_labels == i)
            if rem_area >= min_area:
                assigned[rem_labels == i] = next_label
                next_label += 1

    return assigned


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_bg_color(bg: np.ndarray) -> np.ndarray:
    """Estimate background color from edge pixels of the background image.

    Uses the 5% border on each side, which is least likely to contain
    foreground elements. Returns the median color as a (3,) float array.
    """
    h, w = bg.shape[:2]
    margin_y = max(5, int(h * 0.05))
    margin_x = max(5, int(w * 0.05))

    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:margin_y, :] = True
    edge_mask[-margin_y:, :] = True
    edge_mask[:, :margin_x] = True
    edge_mask[:, -margin_x:] = True

    edge_pixels = bg[edge_mask].reshape(-1, 3).astype(np.float32)

    if len(edge_pixels) < 10:
        return np.median(bg.reshape(-1, 3).astype(np.float32), axis=0)

    return np.median(edge_pixels, axis=0)


def _remove_noise(
    mask: np.ndarray, min_area: int = 15, img_shape: tuple | None = None
) -> np.ndarray:
    """Remove small noise blobs and edge artifacts from foreground mask.

    Filters:
    - Components smaller than min_area (noise)
    - Components spanning >=80% of image height/width with low fill ratio (edge artifacts)
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    clean = np.zeros_like(mask)

    img_h, img_w = img_shape if img_shape else (mask.shape[0], mask.shape[1])

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # Skip tiny noise
        if area < min_area:
            continue

        # Skip edge artifacts: components spanning most of the image
        # with low fill ratio (real foreground has higher density)
        bbox_area = max(w * h, 1)
        fill_ratio = area / bbox_area
        is_full_span = (h >= img_h * 0.8) or (w >= img_w * 0.8)
        if is_full_span and fill_ratio < 0.15:
            continue

        clean[labels == i] = 255

    return clean
