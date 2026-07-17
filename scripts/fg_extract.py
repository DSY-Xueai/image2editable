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


class ComponentExtractionError(RuntimeError):
    pass


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
    detector_masks = [color_mask, diff_mask, sat_mask, bright_mask, edge_fg]
    detector_masks = [
        m for m in detector_masks
        if _keep_detector_mask(
            nonzero_pixels=int(np.count_nonzero(m)),
            total_pixels=h * w,
        )
    ]
    if detector_masks:
        mask = np.logical_or.reduce(detector_masks)
    else:
        mask = np.zeros((h, w), dtype=bool)
    mask = _limit_combined_mask(mask, edge_fg)

    text_ink_mask = None
    if text_mask is not None:
        text_ink_mask = _build_text_ink_mask(img, text_mask)

    mask = mask.astype(np.uint8) * 255

    # Morphological cleanup
    # CLOSE: fill small holes inside foreground regions (beneficial)
    kernel_close = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Remove small noise blobs and edge artifacts
    mask = _remove_noise(mask, img_shape=(h, w))

    # Exclude only likely text ink, not the whole OCR bbox. Whole-box removal
    # cuts holes into graphics that sit behind editable text.
    if text_ink_mask is not None:
        mask[text_ink_mask > 0] = 0

    logger.info(
        "Foreground mask: %d non-zero pixels (%.1f%%)",
        int(np.count_nonzero(mask)),
        np.count_nonzero(mask) / (h * w) * 100,
    )
    return mask


def _keep_detector_mask(
    nonzero_pixels: int,
    total_pixels: int,
    max_mask_ratio: float = 0.45,
) -> bool:
    """Reject detector masks that classify most of the slide as foreground."""
    if total_pixels <= 0:
        return False
    return nonzero_pixels / total_pixels <= max_mask_ratio


def _limit_combined_mask(
    mask: np.ndarray,
    fallback_mask: np.ndarray,
    max_mask_ratio: float = 0.45,
) -> np.ndarray:
    """Reject combined masks that still cover too much of the slide."""
    total_pixels = int(mask.size)
    if _keep_detector_mask(int(np.count_nonzero(mask)), total_pixels, max_mask_ratio):
        return mask
    if _keep_detector_mask(
        int(np.count_nonzero(fallback_mask)), total_pixels, max_mask_ratio
    ):
        return fallback_mask
    return np.zeros(mask.shape, dtype=bool)


def export_visual_components(
    img: np.ndarray,
    element_masks: list[np.ndarray],
    output_dir: str | Path,
    text_mask: np.ndarray,
    padding: int = 3,
    semantic_masks: list[np.ndarray] | None = None,
    text_items: list[dict] | None = None,
    text_clean_image: np.ndarray | None = None,
) -> list[dict]:
    """Export each visual element as an independent transparent PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_h, img_w = img.shape[:2]
    has_text_item_boxes = bool(text_items) and all(
        "box" in item for item in text_items
    )
    text_ink = (
        _build_text_ink_mask(img, text_mask)
        if not has_text_item_boxes
        else _build_text_ink_mask(img, text_mask, text_items=text_items)
    )
    text_removal = cv2.dilate(
        (
            text_ink
            if has_text_item_boxes
            else (text_mask > 0).astype(np.uint8) * 255
        ),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    if text_clean_image is None:
        text_clean_img = _repair_component_rgb(img, text_removal)
    else:
        text_clean_img = np.asarray(text_clean_image, dtype=np.uint8)
        if text_clean_img.shape != img.shape:
            raise ValueError("text-clean image shape must match image")
    ownership_masks = [np.asarray(mask, dtype=bool) for mask in element_masks]
    if any(mask.shape != (img_h, img_w) for mask in ownership_masks):
        raise ValueError("element mask shape must match image")
    semantic_masks = ownership_masks if semantic_masks is None else [
        np.asarray(mask, dtype=bool) for mask in semantic_masks
    ]
    if len(semantic_masks) != len(ownership_masks):
        raise ValueError("semantic mask count must match element mask count")
    if any(mask.shape != (img_h, img_w) for mask in semantic_masks):
        raise ValueError("semantic mask shape must match image")
    owned_union = (
        np.logical_or.reduce(ownership_masks)
        if ownership_masks
        else np.zeros((img_h, img_w), dtype=bool)
    )
    assigned_hole_repairs = _assign_text_hole_repairs(
        ownership_masks,
        text_ink,
        text_mask,
        semantic_masks=semantic_masks,
    )

    components: list[dict] = []

    for position, ownership_mask in enumerate(ownership_masks):
        refined = _refine_visual_mask(img, ownership_mask)
        occupied_by_other = owned_union & ~ownership_mask
        refined = _restore_supported_holes(
            refined,
            ownership_mask,
            semantic_masks[position],
            occupied_by_other,
        )
        unsupported_holes = _remove_border_connected(~refined) & ownership_mask
        if np.any(unsupported_holes):
            raise ComponentExtractionError(
                f"component {position + 1} still has unsupported internal holes"
            )
        semantic_underlay = (
            _remove_border_connected(~ownership_mask)
            & semantic_masks[position]
        )
        alpha_mask = (
            refined
            | assigned_hole_repairs[position]
            | semantic_underlay
        )
        area = int(np.count_nonzero(alpha_mask))
        if area == 0:
            continue

        ys, xs = np.where(alpha_mask)
        x1 = max(0, int(xs.min()) - padding)
        y1 = max(0, int(ys.min()) - padding)
        x2 = min(img_w, int(xs.max()) + 1 + padding)
        y2 = min(img_h, int(ys.max()) + 1 + padding)

        local_mask = alpha_mask[y1:y2, x1:x2]
        rgb = text_clean_img[y1:y2, x1:x2].copy()
        local_underlay = semantic_underlay[y1:y2, x1:x2]
        if np.any(local_underlay):
            rgb = _fill_component_underlay(
                rgb,
                local_underlay,
                refined[y1:y2, x1:x2],
            )
        alpha = _soft_alpha(local_mask)
        rgba = np.dstack([rgb, alpha])

        component_path = output_dir / f"component_{position + 1:04d}.png"
        Image.fromarray(rgba.astype(np.uint8)).save(str(component_path))
        components.append({
            "path": str(component_path),
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "area": area,
            "z_index": position,
        })

    return components


def _restore_supported_holes(
    refined: np.ndarray,
    ownership_mask: np.ndarray,
    semantic: np.ndarray,
    occupied_by_other: np.ndarray,
) -> np.ndarray:
    restored = np.asarray(refined, dtype=bool).copy()
    new_internal_loss = (
        _remove_border_connected(~restored)
        & ownership_mask
        & ~occupied_by_other
    )
    count, labels = cv2.connectedComponents(
        new_internal_loss.astype(np.uint8),
        connectivity=8,
    )
    for label in range(1, count):
        recoverable = labels == label
        recoverable_area = int(np.count_nonzero(recoverable))
        if recoverable_area == 0:
            continue
        supported = recoverable & semantic
        if np.count_nonzero(supported) / recoverable_area >= 0.90:
            restored |= supported
    return restored


def _assign_text_hole_repairs(
    ownership_masks: list[np.ndarray],
    text_ink: np.ndarray,
    text_mask: np.ndarray,
    semantic_masks: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Assign each removed glyph hole to the nearest underlying visual element."""
    if not ownership_masks:
        return []
    semantic_masks = ownership_masks if semantic_masks is None else semantic_masks
    owned_union = np.logical_or.reduce(ownership_masks)
    unowned_text_ink = (text_ink > 0) & ~owned_union
    unowned_text_region = (text_mask > 0) & ~owned_union
    claimed_holes = np.zeros(text_mask.shape, dtype=bool)
    repairs = [np.zeros(text_mask.shape, dtype=bool) for _ in ownership_masks]
    for position in reversed(range(len(ownership_masks))):
        ownership_mask = ownership_masks[position]
        relevant_text = _filter_text_ink_over_components(
            ownership_mask, text_ink, text_mask
        )
        nearby_holes = _find_component_text_repairs(
            ownership_mask, relevant_text
        )
        enclosed_holes = _remove_border_connected(~ownership_mask)
        relevant_region = _filter_text_ink_over_components(
            ownership_mask,
            text_mask,
            text_mask,
        )
        supported_region = (
            (relevant_region > 0)
            & semantic_masks[position]
            & unowned_text_region
        )
        supported_region = _solidify_text_repairs(
            supported_region,
            semantic_masks[position],
            relevant_region,
        )
        repair = (
            (
                unowned_text_ink
                & (nearby_holes | enclosed_holes)
            )
            | supported_region
        ) & ~claimed_holes
        repairs[position] = repair
        claimed_holes |= repair
    return repairs


def _solidify_text_repairs(
    repair_seed: np.ndarray,
    semantic_mask: np.ndarray,
    text_regions: np.ndarray,
) -> np.ndarray:
    """Replace glyph-shaped alpha repairs with solid semantic underlays."""
    solid = np.zeros(repair_seed.shape, dtype=bool)
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        (text_regions > 0).astype(np.uint8),
        connectivity=8,
    )
    for label in range(1, count):
        region_seed = repair_seed & (labels == label)
        if not np.any(region_seed):
            continue
        ys, xs = np.where(region_seed)
        x = int(xs.min())
        y = int(ys.min())
        width = int(xs.max()) + 1 - x
        height = int(ys.max()) + 1 - y
        solid[y:y + height, x:x + width] |= semantic_mask[
            y:y + height,
            x:x + width,
        ]
    return solid


def _fill_component_underlay(
    crop_rgb: np.ndarray,
    repair_mask: np.ndarray,
    donor_mask: np.ndarray,
) -> np.ndarray:
    """Fill hidden pixels from the nearest visible pixel of the same component."""
    repair_mask = np.asarray(repair_mask, dtype=bool)
    donor_mask = np.asarray(donor_mask, dtype=bool) & ~repair_mask
    if not np.any(repair_mask) or not np.any(donor_mask):
        return crop_rgb
    _, labels = cv2.distanceTransformWithLabels(
        (~donor_mask).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    donor_colors = crop_rgb[donor_mask]
    filled = crop_rgb.copy()
    filled[repair_mask] = donor_colors[labels[repair_mask] - 1]
    return filled


def repair_exported_component_text(
    components: list[dict],
    text_mask: np.ndarray,
    source_rgb: np.ndarray,
    text_items: list[dict] | None = None,
    cleaned_rgb: np.ndarray | None = None,
) -> None:
    """Inpaint OCR-detected raster text that remains in exported RGBA layers."""
    if cleaned_rgb is not None and text_items:
        cleaned_rgb = np.asarray(cleaned_rgb, dtype=np.uint8)
        if cleaned_rgb.shape != source_rgb.shape:
            raise ValueError("cleaned RGB image shape must match source RGB image")
        loaded = []
        for position, component in enumerate(components):
            with Image.open(component["path"]) as image:
                rgba = np.asarray(image.convert("RGBA")).copy()
            alpha_support = rgba[:, :, 3] > 0
            alpha_support |= _remove_border_connected(~alpha_support)
            loaded.append((component, rgba, alpha_support, position))

        for item in text_items:
            box_x, box_y, box_width, box_height = (
                int(value) for value in item["box"]
            )
            box_x2 = box_x + box_width
            box_y2 = box_y + box_height
            best = None
            for component, rgba, alpha_support, position in loaded:
                x = int(component["x"])
                y = int(component["y"])
                x1 = max(x, box_x)
                y1 = max(y, box_y)
                x2 = min(x + rgba.shape[1], box_x2)
                y2 = min(y + rgba.shape[0], box_y2)
                if x1 >= x2 or y1 >= y2:
                    continue
                alpha = rgba[y1 - y:y2 - y, x1 - x:x2 - x, 3]
                overlap = int(np.count_nonzero(alpha))
                score = (overlap, int(component.get("z_index", position)))
                if overlap and (best is None or score > best[0]):
                    best = (
                        score,
                        component,
                        rgba,
                        alpha_support,
                        x1,
                        y1,
                        x2,
                        y2,
                    )
            if best is None:
                continue
            _, component, rgba, alpha_support, x1, y1, x2, y2 = best
            x = int(component["x"])
            y = int(component["y"])
            region = rgba[y1 - y:y2 - y, x1 - x:x2 - x]
            local_support = alpha_support[y1 - y:y2 - y, x1 - x:x2 - x]
            region[local_support, :3] = cleaned_rgb[y1:y2, x1:x2][local_support]
            region[local_support, 3] = 255

        for component, rgba, _, _ in loaded:
            Image.fromarray(rgba, "RGBA").save(component["path"])
        return

    has_text_item_boxes = bool(text_items) and all(
        "box" in item for item in text_items
    )
    text_ink = (
        _build_text_ink_mask(source_rgb, text_mask)
        if not has_text_item_boxes
        else _build_text_ink_mask(
            source_rgb,
            text_mask,
            text_items=text_items,
        )
    )
    text_ink = cv2.dilate(
        text_ink,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    image_height, image_width = text_mask.shape
    for component in components:
        path = Path(component["path"])
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA")).copy()
        x = int(component["x"])
        y = int(component["y"])
        height, width = rgba.shape[:2]
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)
        local = np.zeros((height, width), dtype=np.uint8)
        local[: y2 - y, : x2 - x] = text_ink[y:y2, x:x2]
        repair = ((local > 0) & (rgba[:, :, 3] > 0)).astype(np.uint8) * 255
        if not np.any(repair):
            continue
        rgba[:, :, :3] = _repair_component_rgb(rgba[:, :, :3], repair)
        Image.fromarray(rgba, "RGBA").save(path)


def _refine_visual_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Refine one visual-element mask without merging it with other elements."""
    binary = np.asarray(mask, dtype=np.uint8)
    original = binary > 0
    original_area = int(np.count_nonzero(original))
    if original_area < 20:
        return original

    dilated = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    trimap = np.full(binary.shape, cv2.GC_BGD, dtype=np.uint8)
    trimap[dilated > 0] = cv2.GC_PR_BGD
    trimap[binary > 0] = cv2.GC_PR_FGD
    trimap[eroded > 0] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            img,
            trimap,
            None,
            bg_model,
            fg_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return original
    refined = (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)
    refined &= original
    refined_area = int(np.count_nonzero(refined))
    if refined_area == 0 or refined_area < original_area * 0.5:
        return original
    refined |= eroded > 0
    return refined


def _soft_alpha(mask: np.ndarray) -> np.ndarray:
    """Feather a hard mask while preserving its eroded interior."""
    hard = mask.astype(np.uint8) * 255
    eroded = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(hard, (3, 3), 0)
    alpha[eroded > 0] = 255
    alpha[hard == 0] = 0
    return alpha.astype(np.uint8)


def split_components(
    img: np.ndarray,
    fg_mask: np.ndarray,
    output_dir: str | Path,
    min_area: int = 20,
    padding: int = 3,
    text_mask: np.ndarray | None = None,
) -> list[dict]:
    """Split foreground mask into independent transparent PNG components.

    Uses connected-component analysis so a connected shape stays intact.

    Args:
        img: Original image (H, W, 3) RGB uint8.
        fg_mask: Foreground binary mask (H, W) uint8.
        output_dir: Directory to save component PNGs.
        min_area: Minimum component area in pixels.
        padding: Pixels to pad around each component bounding box.
        text_mask: Optional OCR text mask used to repair text over components.

    Returns:
        List of component dicts with keys: path, x, y, w, h, area.
        Sorted by area descending.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_h, img_w = img.shape[:2]

    text_ink_mask = (
        _build_text_ink_mask(img, text_mask)
        if text_mask is not None
        else np.zeros_like(fg_mask)
    )
    component_text_ink_mask = _filter_text_ink_over_components(
        fg_mask, text_ink_mask, text_mask
    )

    # Label on a grouping mask that closes narrow text gaps.
    grouping_mask = _build_component_grouping_mask(fg_mask)
    label_map = _label_connected_components(grouping_mask, min_area)

    # Extract each labeled component
    components: list[dict] = []
    num_labels = label_map.max()

    for i in range(1, num_labels + 1):
        label_bool = label_map == i
        label_area = int(np.count_nonzero(label_bool))
        label_ys, label_xs = np.where(label_bool)
        if len(label_ys) == 0:
            continue
        label_x_min, label_x_max = int(label_xs.min()), int(label_xs.max())
        label_y_min, label_y_max = int(label_ys.min()), int(label_ys.max())
        label_w = label_x_max - label_x_min + 1
        label_h = label_y_max - label_y_min + 1

        original_bool = label_bool & (fg_mask > 0)
        repair_bool = _find_component_text_repairs(
            original_bool, component_text_ink_mask
        )

        if _should_use_solid_bbox_alpha(
            label_area, label_w, label_h, img_h * img_w
        ):
            solid_bool = np.zeros_like(label_bool)
            solid_bool[
                label_y_min:label_y_max + 1,
                label_x_min:label_x_max + 1,
            ] = True
            comp_mask_full = (solid_bool | repair_bool).astype(np.uint8) * 255
        else:
            comp_mask_full = (original_bool | repair_bool).astype(np.uint8) * 255
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

        # Use hard alpha and repair text pixels over components, so
        # editable text does not sit on top of original raster text.
        comp_mask = comp_mask_full[y1:y2, x1:x2]
        repair_mask = repair_bool[y1:y2, x1:x2].astype(np.uint8) * 255
        comp_alpha = comp_mask

        # Crop RGB and combine with alpha
        crop_rgb = _repair_component_rgb(img[y1:y2, x1:x2], repair_mask)
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


def _label_connected_components(
    fg_mask: np.ndarray, min_area: int = 20
) -> np.ndarray:
    """Label connected foreground regions without splitting connected shapes.

    Args:
        fg_mask: Binary foreground mask (H, W) uint8.
        min_area: Minimum area for a component.

    Returns:
        Label map (H, W) int32 where each pixel is assigned a component ID.
    """
    num_orig, orig_labels, orig_stats, _ = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8
    )

    label_map = np.zeros_like(orig_labels, dtype=np.int32)
    next_label = 1

    for i in range(1, num_orig):
        area = orig_stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        label_map[orig_labels == i] = next_label
        next_label += 1

    return label_map


def _build_component_grouping_mask(fg_mask: np.ndarray) -> np.ndarray:
    """Close narrow text-shaped gaps only for component grouping."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    return cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _should_use_solid_bbox_alpha(
    area: int,
    width: int,
    height: int,
    total_area: int,
    min_bbox_area_ratio: float = 0.12,
    min_fill_ratio: float = 0.30,
) -> bool:
    """Use a solid crop for large image-like regions with unreliable holes."""
    bbox_area = max(width * height, 1)
    return (
        bbox_area / max(total_area, 1) >= min_bbox_area_ratio
        and area / bbox_area >= min_fill_ratio
    )


def _find_component_text_repairs(
    component_mask: np.ndarray, text_ink_mask: np.ndarray
) -> np.ndarray:
    """Find raster text pixels that sit on top of an existing component."""
    if not np.any(component_mask) or not np.any(text_ink_mask > 0):
        return np.zeros(component_mask.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    nearby_component = cv2.dilate(
        component_mask.astype(np.uint8) * 255, kernel, iterations=1
    ) > 0
    return nearby_component & (text_ink_mask > 0)


def _filter_text_ink_over_components(
    fg_mask: np.ndarray,
    text_ink_mask: np.ndarray,
    text_mask: np.ndarray | None,
    min_component_ratio: float = 0.25,
) -> np.ndarray:
    """Keep text ink only where the OCR box sits on a foreground component."""
    if text_mask is None or not np.any(text_ink_mask > 0):
        return np.zeros_like(text_ink_mask)

    keep = np.zeros_like(text_ink_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (text_mask > 0).astype(np.uint8), connectivity=8
    )

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w <= 0 or h <= 0:
            continue

        box_area = max(w * h, 1)
        fg_pixels = int(np.count_nonzero(fg_mask[y:y + h, x:x + w]))
        if fg_pixels / box_area >= min_component_ratio:
            keep[y:y + h, x:x + w] = text_ink_mask[y:y + h, x:x + w]

    return keep


def _repair_component_rgb(crop_rgb: np.ndarray, repair_mask: np.ndarray) -> np.ndarray:
    """Remove raster text pixels from a component while keeping its base shape."""
    if not np.any(repair_mask > 0):
        return crop_rgb
    bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    repaired = cv2.inpaint(bgr, repair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_text_ink_mask(
    img: np.ndarray,
    text_mask: np.ndarray,
    text_items: list[dict] | None = None,
) -> np.ndarray:
    """Estimate actual glyph pixels inside OCR boxes."""
    ink_mask = np.zeros(text_mask.shape, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if text_items and any("box" not in item for item in text_items):
        text_items = None
    if text_items is None:
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (text_mask > 0).astype(np.uint8), connectivity=8
        )
        regions = [
            (*tuple(int(value) for value in stats[i, :4]), None)
            for i in range(1, count)
        ]
    else:
        regions = [
            (
                *tuple(int(value) for value in item["box"]),
                item.get("color"),
            )
            for item in text_items
        ]

    for x, y, w, h, color in regions:
        x = max(0, x)
        y = max(0, y)
        w = min(gray.shape[1] - x, w)
        h = min(gray.shape[0] - y, h)
        if w < 3 or h < 3:
            continue

        region = gray[y:y + h, x:x + w]
        if float(np.std(region)) < 8.0:
            continue

        thresh, _ = cv2.threshold(
            region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        ink = None
        if isinstance(color, str) and len(color) == 7 and color.startswith("#"):
            target = np.asarray(
                [int(color[index:index + 2], 16) for index in (1, 3, 5)],
                dtype=np.float32,
            )
            rgb_region = img[y:y + h, x:x + w].astype(np.float32)
            color_match = np.linalg.norm(rgb_region - target, axis=2) <= 45.0
            isolated_match = _remove_border_connected(color_match)
            if np.count_nonzero(isolated_match) >= 4:
                ink = isolated_match
        if ink is None:
            ink = _select_text_ink(region, float(thresh))

        ink_uint8 = ink.astype(np.uint8) * 255
        ink_uint8 = cv2.dilate(ink_uint8, np.ones((3, 3), np.uint8), iterations=1)
        box_mask = text_mask[y:y + h, x:x + w] > 0
        ink_mask[y:y + h, x:x + w][(ink_uint8 > 0) & box_mask] = 255

    return ink_mask


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
        if is_full_span and fill_ratio < 0.30:
            continue

        clean[labels == i] = 255

    return clean
