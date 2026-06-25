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
    if fg_mask is not None:
        bg = _replace_unrecoverable_large_regions(bg, fg_mask, bg_color)
        repair_mask = _build_component_repair_mask(fg_mask)
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


def _build_component_repair_mask(fg_mask: np.ndarray) -> np.ndarray:
    """Build a repair mask that also covers small component shadows."""
    repair_mask = np.zeros_like(fg_mask)
    safe_mask = _mask_for_destructive_repair(fg_mask)
    total_area = max(int(fg_mask.shape[0] * fg_mask.shape[1]), 1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (safe_mask > 0).astype(np.uint8), connectivity=8
    )

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if _is_unrecoverable_large_region(area, bw, bh, total_area):
            continue

        repair_mask[labels == i] = 255
        if not _should_expand_shadow_halo(bw, bh, fg_mask.shape):
            continue

        pad = max(2, min(10, max(bw, bh) // 8))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fg_mask.shape[1], x + bw + pad)
        y2 = min(fg_mask.shape[0], y + bh + pad)
        repair_mask[y1:y2, x1:x2] = np.maximum(
            repair_mask[y1:y2, x1:x2],
            cv2.dilate(
                (labels[y1:y2, x1:x2] == i).astype(np.uint8) * 255,
                np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8),
                iterations=1,
            ),
        )

    return repair_mask


def _should_expand_shadow_halo(
    width: int,
    height: int,
    mask_shape: tuple[int, int],
    max_bbox_area_ratio: float = 0.08,
    max_width_ratio: float = 0.25,
    max_height_ratio: float = 0.30,
) -> bool:
    """Only expand repair for compact objects likely to have drop shadows."""
    img_h, img_w = mask_shape
    total_area = max(img_h * img_w, 1)
    return (
        width * height / total_area <= max_bbox_area_ratio
        and width / max(img_w, 1) <= max_width_ratio
        and height / max(img_h, 1) <= max_height_ratio
    )


def _mask_for_destructive_repair(fg_mask: np.ndarray) -> np.ndarray:
    """Keep only foreground regions that are small enough to repair safely."""
    repair_mask = fg_mask.copy()
    total_area = max(int(fg_mask.shape[0] * fg_mask.shape[1]), 1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (fg_mask > 0).astype(np.uint8), connectivity=8
    )

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if _is_unrecoverable_large_region(area, bw, bh, total_area):
            repair_mask[labels == i] = 0

    return repair_mask


def _replace_unrecoverable_large_regions(
    bg: np.ndarray,
    fg_mask: np.ndarray,
    bg_color: np.ndarray,
) -> np.ndarray:
    """Replace foreground bboxes that are unlikely to reveal true hidden pixels."""
    output = bg.copy()
    total_area = max(int(fg_mask.shape[0] * fg_mask.shape[1]), 1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (fg_mask > 0).astype(np.uint8), connectivity=8
    )

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if not _should_replace_region_bbox(area, bw, bh, total_area):
            continue

        pad = max(2, min(10, max(bw, bh) // 16))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fg_mask.shape[1], x + bw + pad)
        y2 = min(fg_mask.shape[0], y + bh + pad)
        fill = _fill_region_from_low_frequency_context(
            output, fg_mask, x1, y1, x2 - x1, y2 - y1, bg_color
        )
        output[y1:y2, x1:x2] = fill

    return output


def _should_replace_region_bbox(
    area: int,
    width: int,
    height: int,
    total_area: int,
    min_dense_fill_ratio: float = 0.18,
    max_dense_bbox_ratio: float = 0.12,
) -> bool:
    """Choose regions where bbox fill is more stable than local inpaint."""
    bbox_area = max(width * height, 1)
    if _is_unrecoverable_large_region(area, width, height, total_area):
        return True
    return (
        bbox_area / max(total_area, 1) <= max_dense_bbox_ratio
        and area / bbox_area >= min_dense_fill_ratio
    )


def _sample_large_region_fill(
    img: np.ndarray,
    fg_mask: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    bg_color: np.ndarray,
) -> np.ndarray:
    """Estimate a clean fill color from the ring around a large region."""
    h, w = fg_mask.shape
    pad = max(12, min(80, max(width, height) // 12))
    sx1 = max(0, x - pad)
    sy1 = max(0, y - pad)
    sx2 = min(w, x + width + pad)
    sy2 = min(h, y + height + pad)

    ring = np.zeros((sy2 - sy1, sx2 - sx1), dtype=bool)
    ring[:, :] = True
    ring[
        y - sy1:y + height - sy1,
        x - sx1:x + width - sx1,
    ] = False
    ring &= fg_mask[sy1:sy2, sx1:sx2] == 0

    pixels = img[sy1:sy2, sx1:sx2][ring]
    if len(pixels) < 20:
        fill = bg_color
    else:
        fill = np.median(pixels.reshape(-1, 3), axis=0)
    return np.clip(fill, 0, 255).astype(np.uint8)


def _fill_region_from_low_frequency_context(
    img: np.ndarray,
    fg_mask: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    bg_color: np.ndarray,
) -> np.ndarray:
    """Fill a hidden region from low-frequency local context."""
    h, w = fg_mask.shape
    context = max(20, min(180, max(width, height) // 2))
    sx1 = max(0, x - context)
    sy1 = max(0, y - context)
    sx2 = min(w, x + width + context)
    sy2 = min(h, y + height + context)

    roi = img[sy1:sy2, sx1:sx2].copy()
    if roi.size == 0:
        fill = np.clip(bg_color, 0, 255).astype(np.uint8)
        return np.tile(fill, (height, width, 1))

    mask = np.zeros((sy2 - sy1, sx2 - sx1), dtype=np.uint8)
    mask[y - sy1:y + height - sy1, x - sx1:x + width - sx1] = 255

    max_dim = max(roi.shape[:2])
    scale = min(1.0, 220.0 / max(max_dim, 1))
    if scale < 1.0:
        small_size = (
            max(1, int(roi.shape[1] * scale)),
            max(1, int(roi.shape[0] * scale)),
        )
        small_roi = cv2.resize(roi, small_size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, small_size, interpolation=cv2.INTER_NEAREST)
    else:
        small_roi = roi
        small_mask = mask

    repaired = cv2.inpaint(
        cv2.cvtColor(small_roi, cv2.COLOR_RGB2BGR),
        small_mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_TELEA,
    )
    repaired = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
    if scale < 1.0:
        repaired = cv2.resize(
            repaired, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC
        )

    return repaired[y - sy1:y + height - sy1, x - sx1:x + width - sx1]


def _is_unrecoverable_large_region(
    area: int,
    width: int,
    height: int,
    total_area: int,
    min_bbox_area_ratio: float = 0.12,
    min_fill_ratio: float = 0.30,
) -> bool:
    """Identify large dense regions where local inpainting leaves visible scars."""
    bbox_area = max(width * height, 1)
    return (
        bbox_area / max(total_area, 1) >= min_bbox_area_ratio
        and area / bbox_area >= min_fill_ratio
    )


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

        fill = _sample_text_background(box, ink)
        if fill is None:
            local_mask = text_mask[sy1:sy2, sx1:sx2] == 0
            local_pixels = output[sy1:sy2, sx1:sx2][local_mask]
            if len(local_pixels) == 0:
                fill = np.median(output.reshape(-1, 3), axis=0)
            else:
                fill = np.median(local_pixels.reshape(-1, 3), axis=0)
        output[y1:y2, x1:x2][ink] = np.clip(fill, 0, 255).astype(np.uint8)

    return output


def _sample_text_background(
    region: np.ndarray, ink: np.ndarray
) -> np.ndarray | None:
    """Sample the text box's own background, avoiding outside-page colors."""
    background = (~ink).astype(np.uint8) * 255
    if background.shape[0] >= 3 and background.shape[1] >= 3:
        background = cv2.erode(background, np.ones((3, 3), np.uint8), iterations=1)
    pixels = region[background > 0]
    if len(pixels) < 10:
        pixels = region[~ink]
    if len(pixels) < 10:
        return None
    return np.median(pixels.reshape(-1, 3), axis=0)


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
    ink = _select_text_ink(gray, float(thresh))

    ink_uint8 = ink.astype(np.uint8) * 255
    ink_uint8 = cv2.dilate(ink_uint8, np.ones((3, 3), np.uint8), iterations=1)
    return ink_uint8 > 0


def _select_text_ink(gray: np.ndarray, thresh: float) -> np.ndarray:
    """Pick the glyph class after dropping background connected to box edges."""
    dark = gray <= thresh
    light = gray > thresh
    dark_inner = _remove_border_connected(dark)
    light_inner = _remove_border_connected(light)

    if np.count_nonzero(dark_inner) or np.count_nonzero(light_inner):
        ink = (
            dark_inner
            if np.count_nonzero(dark_inner) >= np.count_nonzero(light_inner)
            else light_inner
        )
    else:
        ink = dark if np.count_nonzero(dark) <= np.count_nonzero(light) else light

    return _add_antialiased_text_edges(gray, ink)


def _add_antialiased_text_edges(gray: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Include same-direction antialiased text pixels without taking the background."""
    if not np.any(ink):
        return ink
    border = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ]).astype(np.float32)
    border_mean = float(np.mean(border))
    ink_mean = float(np.mean(gray[ink]))

    if ink_mean < border_mean:
        candidate = gray <= max(0.0, border_mean - 25.0)
    else:
        candidate = gray >= min(255.0, border_mean + 25.0)

    candidate_inner = _remove_border_connected(candidate)
    if np.any(candidate_inner):
        return ink | candidate_inner
    return ink


def _remove_border_connected(mask: np.ndarray) -> np.ndarray:
    """Remove mask components touching the OCR box edge."""
    if not np.any(mask):
        return mask.copy()
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    border_labels = set(labels[0, :])
    border_labels.update(labels[-1, :])
    border_labels.update(labels[:, 0])
    border_labels.update(labels[:, -1])
    keep = np.ones(num_labels, dtype=bool)
    keep[list(border_labels)] = False
    keep[0] = False
    return keep[labels]


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
