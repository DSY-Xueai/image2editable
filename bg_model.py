#!/usr/bin/env python3
"""Background modeling and repair module.

Builds a clean background image by:
1. Adaptive background color detection (edge sampling)
2. Periodic tile median modeling
3. Inpainting to remove foreground/text residuals

Usage:
    from bg_model import build_background
    bg = build_background(img_rgb, text_mask=mask)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_background(
    img: np.ndarray,
    text_mask: np.ndarray | None = None,
    fg_hint_mask: np.ndarray | None = None,
    period: int = 32,
) -> np.ndarray:
    """Build a clean background image from the input.

    Args:
        img: Input image (H, W, 3) RGB uint8.
        text_mask: Binary mask (H, W) where text regions = 255.
        fg_hint_mask: Optional binary mask of known foreground regions.
        period: Tile period for periodic median modeling.

    Returns:
        Clean background image (H, W, 3) RGB uint8.
    """
    h, w = img.shape[:2]

    if text_mask is None:
        text_mask = np.zeros((h, w), dtype=np.uint8)
    if fg_hint_mask is None:
        fg_hint_mask = np.zeros((h, w), dtype=np.uint8)

    # Combined exclusion mask
    exclude = ((text_mask > 0) | (fg_hint_mask > 0)).astype(np.uint8) * 255

    # Step 1: Detect background color adaptively
    bg_color, bg_std, candidate_mask = _detect_background(img, exclude)
    logger.info(
        "Background color: RGB(%d,%d,%d), std=%.1f",
        int(bg_color[0]), int(bg_color[1]), int(bg_color[2]), bg_std,
    )

    # Step 2: Periodic tile median modeling
    bg = _tile_median_model(img, candidate_mask, bg_color, period)

    # Step 3: Inpainting — repair areas where foreground/text was removed
    inpaint_mask = _build_inpaint_mask(img, bg, exclude, candidate_mask)
    if np.any(inpaint_mask > 0):
        bg = _inpaint(bg, inpaint_mask)

    return bg


# ---------------------------------------------------------------------------
# Step 1: Adaptive background detection
# ---------------------------------------------------------------------------


def _detect_background(
    img: np.ndarray, exclude_mask: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """Detect the dominant background color by sampling image edges.

    Returns:
        bg_color: (3,) float array — dominant background RGB.
        bg_std: float — standard deviation of background pixels.
        candidate_mask: (H, W) bool — pixels likely belonging to background.
    """
    h, w = img.shape[:2]

    # Sample from edges (5% border on each side)
    margin_y = max(5, int(h * 0.05))
    margin_x = max(5, int(w * 0.05))

    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:margin_y, :] = True   # top
    edge_mask[-margin_y:, :] = True  # bottom
    edge_mask[:, :margin_x] = True   # left
    edge_mask[:, -margin_x:] = True  # right

    # Exclude known text/foreground from edge sampling
    edge_mask &= (exclude_mask == 0)

    edge_pixels = img[edge_mask].reshape(-1, 3).astype(np.float32)

    if len(edge_pixels) < 10:
        # Fallback: use all non-excluded pixels
        valid = exclude_mask == 0
        edge_pixels = img[valid].reshape(-1, 3).astype(np.float32)

    if len(edge_pixels) < 10:
        # Ultimate fallback
        bg_color = np.array([255.0, 255.0, 255.0])
        return bg_color, 30.0, np.ones((h, w), dtype=bool)

    # Find dominant color via histogram peak (faster than KMeans)
    bg_color = np.median(edge_pixels, axis=0)
    bg_std = float(np.mean(np.std(edge_pixels, axis=0)))

    # Adaptive threshold: pixels within N standard deviations of bg_color
    threshold = max(35.0, bg_std * 2.5)

    all_pixels = img.reshape(-1, 3).astype(np.float32)
    dists = np.linalg.norm(all_pixels - bg_color, axis=1)
    candidate_flat = dists < threshold

    candidate_mask = candidate_flat.reshape(h, w)
    # Exclude known foreground/text
    candidate_mask &= (exclude_mask == 0)

    return bg_color, bg_std, candidate_mask


# ---------------------------------------------------------------------------
# Step 2: Periodic tile median modeling
# ---------------------------------------------------------------------------


def _tile_median_model(
    img: np.ndarray,
    candidate_mask: np.ndarray,
    bg_color: np.ndarray,
    period: int,
) -> np.ndarray:
    """Build background via periodic tile median sampling.

    For each (py, px) position within a period×period tile, collect all
    candidate pixels at that periodic position and take their median.
    """
    h, w, c = img.shape
    tile = np.zeros((period, period, c), dtype=np.float32)

    for py in range(period):
        for px in range(period):
            ys = np.arange(py, h, period)
            xs = np.arange(px, w, period)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            pixels = img[yy, xx].reshape(-1, c).astype(np.float32)
            valid = candidate_mask[yy, xx].reshape(-1)

            if np.any(valid):
                tile[py, px] = np.median(pixels[valid], axis=0)
            else:
                tile[py, px] = bg_color

    # Tile the background
    # Use np.tile for efficiency instead of per-pixel loop
    reps_y = (h + period - 1) // period
    reps_x = (w + period - 1) // period
    bg = np.tile(tile, (reps_y, reps_x, 1))[:h, :w, :]

    return np.clip(bg, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 3: Inpainting
# ---------------------------------------------------------------------------


def _build_inpaint_mask(
    img: np.ndarray,
    bg: np.ndarray,
    exclude_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """Build a mask of regions that need inpainting.

    These are areas where the tile model couldn't get clean background
    because they were covered by foreground or text.
    """
    h, w = img.shape[:2]

    # Regions that are excluded (text/fg) need inpainting in the background
    # Also regions where the background model differs significantly from
    # what we'd expect (non-candidate areas)
    inpaint = exclude_mask.copy()

    # Add non-candidate regions that might have foreground residuals
    non_candidate = (~candidate_mask).astype(np.uint8) * 255
    # But only where the tile model had to use fallback (bg_color)
    diff = np.linalg.norm(
        img.astype(np.float32) - bg.astype(np.float32), axis=2
    )
    residual = (diff > 40) & (~candidate_mask)
    inpaint = np.maximum(inpaint, residual.astype(np.uint8) * 255)

    # Dilate slightly to cover edges
    kernel = np.ones((5, 5), np.uint8)
    inpaint = cv2.dilate(inpaint, kernel, iterations=1)

    return inpaint


def _inpaint(bg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint masked regions using Telea algorithm."""
    bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    # Use larger radius for better fill quality
    repaired = cv2.inpaint(bgr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Second pass with NS method for smoother results on large areas
    large_areas = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    repaired = cv2.inpaint(repaired, large_areas, inpaintRadius=5, flags=cv2.INPAINT_NS)

    result = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

    # Gaussian blur at inpainting boundaries to smooth seams
    blurred = cv2.GaussianBlur(result, (5, 5), 0)
    boundary = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - cv2.erode(mask, np.ones((3, 3), np.uint8))
    boundary_mask = (boundary > 0)

    output = result.copy()
    output[boundary_mask] = blurred[boundary_mask]

    return output
