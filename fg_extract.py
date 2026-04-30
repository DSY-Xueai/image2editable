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
    """Extract foreground binary mask by comparing image against background.

    Args:
        img: Original image (H, W, 3) RGB uint8.
        bg: Background model (H, W, 3) RGB uint8.
        text_mask: Binary mask (H, W) where text regions = 255.
        diff_threshold: Base threshold for foreground detection.

    Returns:
        Cleaned foreground mask (H, W) uint8, foreground = 255.
    """
    h, w = img.shape[:2]

    # L2 norm difference per pixel
    diff = np.linalg.norm(
        img.astype(np.float32) - bg.astype(np.float32), axis=2
    )

    # HSV for saturation-based detection
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    # Grayscale for brightness-based detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY).astype(np.float32)
    brightness_diff = np.abs(gray_img - gray_bg)

    # Adaptive saturation threshold based on background saturation
    bg_hsv = cv2.cvtColor(bg, cv2.COLOR_RGB2HSV)
    bg_sat_mean = float(np.mean(bg_hsv[:, :, 1]))
    sat_threshold = max(30.0, bg_sat_mean + 20.0)

    # Multi-condition foreground detection
    mask = (
        (diff > diff_threshold)
        | ((diff > diff_threshold * 0.65) & (sat > sat_threshold))
        | (brightness_diff > diff_threshold * 1.2)
    )

    # Exclude text regions — text will be rebuilt as editable text boxes
    if text_mask is not None:
        mask[text_mask > 0] = False

    mask = mask.astype(np.uint8) * 255

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove noise and grid lines
    mask = _remove_noise_and_lines(mask)

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

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8
    )

    components: list[dict] = []
    img_h, img_w = img.shape[:2]

    for i in range(1, num_labels):  # skip background label 0
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        # Pad bounding box
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        # Extract component mask and apply alpha feathering
        comp_mask = (labels[y1:y2, x1:x2] == i).astype(np.uint8) * 255
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
            "area": int(area),
        })

    # Sort by area descending (largest first)
    components.sort(key=lambda c: c["area"], reverse=True)

    logger.info("Split into %d foreground components.", len(components))
    return components


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _remove_noise_and_lines(
    mask: np.ndarray, min_area: int = 15
) -> np.ndarray:
    """Remove small noise blobs and thin grid lines from foreground mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    clean = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # Skip tiny noise
        if area < min_area:
            continue

        # Skip thin horizontal lines (likely grid)
        if w > 500 and h <= 3:
            continue

        # Skip thin vertical lines (likely grid)
        if h > 500 and w <= 3:
            continue

        clean[labels == i] = 255

    return clean
