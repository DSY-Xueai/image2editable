#!/usr/bin/env python3
"""Background modeling and repair module.

Builds a clean background image by:
1. Adaptive background color detection (edge sampling)
2. Using the original image as base (preserving real background)
3. Inpainting foreground/text regions from surrounding pixels

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
        period: Tile period (kept for API compatibility, unused in new approach).

    Returns:
        Clean background image (H, W, 3) RGB uint8.
    """
    h, w = img.shape[:2]

    if text_mask is None:
        text_mask = np.zeros((h, w), dtype=np.uint8)
    if fg_hint_mask is None:
        fg_hint_mask = np.zeros((h, w), dtype=np.uint8)
    elif not _should_use_fg_hint(
        nonzero_pixels=int(np.count_nonzero(fg_hint_mask)),
        total_pixels=h * w,
    ):
        logger.warning("Ignoring oversized foreground hint for background refinement.")
        fg_hint_mask = np.zeros((h, w), dtype=np.uint8)

    # Combined exclusion mask
    exclude = ((text_mask > 0) | (fg_hint_mask > 0)).astype(np.uint8) * 255

    # Step 1: Detect background color adaptively
    bg_color, bg_std, candidate_mask = _detect_background(img, exclude)
    logger.info(
        "Background color: RGB(%d,%d,%d), std=%.1f",
        int(bg_color[0]), int(bg_color[1]), int(bg_color[2]), bg_std,
    )

    # Step 2: Build background — strategy depends on whether we have fg hints
    has_fg_hint = np.any(fg_hint_mask > 0)

    if has_fg_hint:
        # Refinement pass: use original image + inpainting for pixel-accurate bg
        bg = _original_based_background(
            img,
            exclude,
            bg_color,
            text_mask=text_mask,
            fg_mask=fg_hint_mask,
        )
    else:
        # Initial pass: smooth background for foreground detection
        bg = _smooth_background(img, bg_color, candidate_mask, text_mask)

    return bg


def _should_use_fg_hint(
    nonzero_pixels: int,
    total_pixels: int,
    max_foreground_ratio: float = 0.45,
) -> bool:
    """Reject failed foreground hints that cover too much of the slide."""
    if total_pixels <= 0:
        return False
    return nonzero_pixels / total_pixels <= max_foreground_ratio


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
# Step 2a: Smooth background for initial foreground detection
# ---------------------------------------------------------------------------


def _smooth_background(
    img: np.ndarray,
    bg_color: np.ndarray,
    candidate_mask: np.ndarray,
    text_mask: np.ndarray,
) -> np.ndarray:
    """Build a smooth background for the initial foreground detection pass.

    Replaces non-background pixels with bg_color and applies smoothing.
    This creates enough contrast for diff-based foreground detection while
    preserving the general background appearance.

    Args:
        img: Original image (H, W, 3) RGB uint8.
        bg_color: Detected background color (3,) float.
        candidate_mask: (H, W) bool — pixels likely belonging to background.
        text_mask: Binary mask (H, W) uint8 where text regions = 255.

    Returns:
        Smooth background (H, W, 3) RGB uint8.
    """
    bg = img.copy()
    fill = np.clip(bg_color, 0, 255).astype(np.uint8)

    # Replace non-candidate pixels (likely foreground) with bg_color
    bg[~candidate_mask] = fill

    # Also replace text regions
    if text_mask is not None:
        bg[text_mask > 0] = fill

    # Smooth to blend transitions and reduce artifacts
    bg = cv2.GaussianBlur(bg, (21, 21), 0)

    return bg


# ---------------------------------------------------------------------------
# Step 2b: Original-based background with inpainting (refinement pass)
# ---------------------------------------------------------------------------


def _original_based_background(
    img: np.ndarray,
    exclude_mask: np.ndarray,
    bg_color: np.ndarray,
    text_mask: np.ndarray | None = None,
    fg_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build background by starting from original image and inpainting excluded regions.

    This preserves the original background pixel-for-pixel in areas without
    foreground/text, and uses inpainting to fill the excluded regions from
    surrounding real background pixels.

    Args:
        img: Original image (H, W, 3) RGB uint8.
        exclude_mask: Binary mask (H, W) uint8, regions to repair = 255.
        bg_color: Detected background color (3,) float.

    Returns:
        Clean background (H, W, 3) RGB uint8.
    """
    bg = img.copy()

    # If nothing to repair, return original
    if not np.any(exclude_mask > 0):
        return bg

    if text_mask is not None:
        bg = _fill_text_regions(bg, text_mask)

    repair_mask = fg_mask if fg_mask is not None else exclude_mask
    if not np.any(repair_mask > 0):
        return bg

    # Pre-fill foreground regions with bg_color for better inpainting seed
    fill_color = np.clip(bg_color, 0, 255).astype(np.uint8)
    bg[repair_mask > 0] = fill_color

    # Build inpaint mask from the exact excluded regions.
    inpaint_mask = _build_inpaint_mask(repair_mask)

    # Inpaint to blend filled regions with surrounding real background
    bg = _inpaint(bg, inpaint_mask)

    return bg


def _build_inpaint_mask(exclude_mask: np.ndarray) -> np.ndarray:
    """Build inpaint mask from the exact exclusion mask."""
    return exclude_mask.copy()


def _fill_text_regions(img: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """Clean OCR text boxes with nearby non-text background color."""
    output = img.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (text_mask > 0).astype(np.uint8), connectivity=8
    )
    h, w = text_mask.shape

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        pad = max(4, min(16, max(bw, bh) // 6))
        sx1 = max(0, x1 - pad)
        sy1 = max(0, y1 - pad)
        sx2 = min(w, x2 + pad)
        sy2 = min(h, y2 + pad)

        box = output[y1:y2, x1:x2]
        ink = _estimate_text_ink(box)
        if not np.any(ink):
            continue

        local_mask = text_mask[sy1:sy2, sx1:sx2] == 0
        local_pixels = output[sy1:sy2, sx1:sx2][local_mask]
        if len(local_pixels) == 0:
            fill = np.median(output.reshape(-1, 3), axis=0)
        else:
            fill = np.median(local_pixels.reshape(-1, 3), axis=0)
        output[y1:y2, x1:x2][ink] = np.clip(fill, 0, 255).astype(np.uint8)

    return output


def _estimate_text_ink(region: np.ndarray) -> np.ndarray:
    """Estimate glyph pixels in an OCR text box."""
    if region.size == 0 or region.shape[0] < 3 or region.shape[1] < 3:
        return np.zeros(region.shape[:2], dtype=bool)

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    if float(np.std(gray)) < 8.0:
        return np.zeros(gray.shape, dtype=bool)

    thresh, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    border = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ]).astype(np.float32)
    border_mean = float(np.mean(border))

    if border_mean > float(thresh):
        cutoff = max(float(thresh), border_mean - 25.0)
        ink = gray <= cutoff
    else:
        cutoff = min(float(thresh), border_mean + 25.0)
        ink = gray >= cutoff

    ink_uint8 = ink.astype(np.uint8) * 255
    ink_uint8 = cv2.dilate(ink_uint8, np.ones((3, 3), np.uint8), iterations=1)
    return ink_uint8 > 0


# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------


def _inpaint(bg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint masked regions using dual-pass approach."""
    original = bg.copy()
    bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    # First pass: Telea algorithm with larger radius for structural fill
    repaired = cv2.inpaint(bgr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Second pass: NS method for smoother blending on the same regions
    repaired = cv2.inpaint(repaired, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)

    result = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

    output = result.copy()
    output[mask == 0] = original[mask == 0]

    return output
