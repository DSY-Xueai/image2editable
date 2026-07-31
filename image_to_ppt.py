#!/usr/bin/env python3
"""Image-to-PPT converter — main entry point.

Converts one or more images into an editable PowerPoint presentation using:
  1. OCR text detection with style estimation
  2. Adaptive background modeling and inpainting repair
  3. Foreground extraction and component splitting
  4. Layered PPTX assembly (background + components + text boxes)

Usage:
    python image_to_ppt.py input.png
    python image_to_ppt.py img1.png img2.png img3.png
    python image_to_ppt.py ./slides_folder/
    python image_to_ppt.py input.png -o output.pptx
    python image_to_ppt.py input.png --lang ch
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import logging
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.bg_model import (
    build_clean_background,
    build_removal_mask,
    build_widescreen_background,
    repair_masked_background,
)
from scripts.fg_extract import (
    _build_text_ink_mask,
    export_visual_components,
    repair_exported_component_text,
)
from scripts.lama_inpaint import (
    inpaint_large_mask,
    inpaint_large_mask_isolated,
    release_model,
)
from scripts.object_detect import (
    ObjectProposal,
    create_object_detector,
    filter_text_overlapping_proposals,
    generate_object_proposals,
)
from scripts.ppt_assemble import assemble_pptx, assemble_pptx_multi
from scripts.text_detect import close_ocr_engines, detect_text
from scripts.visual_segment import (
    MaskCandidate,
    VisualSegmentationError,
    background_residual_metrics,
    create_sam_generator,
    filter_prompt_free_candidates,
    filter_unchanged_residual_candidates,
    generate_mask_candidates,
    generate_prompted_mask_candidates,
    reconcile_residual_candidates,
    recheck_visual_element_holes,
    has_background_residual,
    needs_text_only_fallback,
    require_visual_quality,
    resolve_sam_checkpoint,
    resolve_visual_elements,
    validate_visual_masks,
    visual_difference,
    write_segmentation_diagnostics,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _filter_probable_icon_text_items(items: list[dict]) -> list[dict]:
    """Keep ambiguous compact OCR glyphs in the raster layer as icons."""
    return [
        item for item in items
        if not _is_probable_icon_text_item(item)
    ]


def _is_probable_icon_text_item(item: dict) -> bool:
    box = item.get("box")
    text = "".join(str(item.get("text", "")).split())
    confidence = item.get("confidence")
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or not isinstance(confidence, (int, float))
        or confidence >= 0.9
        or len(text) > 2
    ):
        return False
    width = max(1, int(box[2]))
    height = max(1, int(box[3]))
    return 0.65 <= width / height <= 1.5


def _filter_probable_icon_text_analysis(
    items: list[dict],
    text_mask: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    detected = np.asarray(text_mask, dtype=np.uint8)
    filtered = _filter_probable_icon_text_items(items)
    if len(filtered) == len(items):
        return filtered, detected.copy()

    result = detected.copy()
    height, width = result.shape
    for item in items:
        if not _is_probable_icon_text_item(item):
            continue
        x, y, box_width, box_height = (
            int(value) for value in item["box"]
        )
        result[
            max(0, y):min(height, y + box_height),
            max(0, x):min(width, x + box_width),
        ] = 0
    for item in filtered:
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x, y, box_width, box_height = (int(value) for value in box)
        y1, y2 = max(0, y), min(height, y + box_height)
        x1, x2 = max(0, x), min(width, x + box_width)
        result[y1:y2, x1:x2] = detected[y1:y2, x1:x2]
    return filtered, result


def _build_text_cleanup_mask(
    image: np.ndarray,
    text_mask: np.ndarray,
    text_items: list[dict],
) -> np.ndarray:
    """Cover raster glyphs and antialiasing without replacing whole OCR boxes."""
    source = np.asarray(image, dtype=np.uint8)
    detected = np.asarray(text_mask, dtype=np.uint8)
    if detected.shape != source.shape[:2]:
        raise ValueError("text mask must match image")
    ink = _build_text_ink_mask(
        source,
        detected,
        text_items=text_items or None,
    )
    extended_ink = ink.copy()
    for item in text_items:
        box = item.get("box")
        color = item.get("color")
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or not isinstance(color, str)
            or len(color) != 7
            or not color.startswith("#")
        ):
            continue
        x, y, box_width, box_height = (int(value) for value in box)
        search_pad = max(
            4,
            min(12, int(round(max(box_height, 1) * 0.15))),
        )
        x1 = max(0, x - search_pad)
        y1 = max(0, y - search_pad)
        x2 = min(source.shape[1], x + box_width + search_pad)
        y2 = min(source.shape[0], y + box_height + search_pad)
        target = np.asarray(
            [int(color[index:index + 2], 16) for index in (1, 3, 5)],
            dtype=np.float32,
        )
        region = source[y1:y2, x1:x2].astype(np.float32)
        matching = (
            np.linalg.norm(region - target, axis=2) <= 120.0
        ).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            matching,
            connectivity=8,
        )
        join_radius = max(3, min(4, search_pad // 2))
        seed = cv2.dilate(
            (ink[y1:y2, x1:x2] > 0).astype(np.uint8),
            np.ones(
                (join_radius * 2 + 1, join_radius * 2 + 1),
                dtype=np.uint8,
            ),
            iterations=1,
        ) > 0
        box_support = np.zeros(matching.shape, dtype=bool)
        box_support[
            max(0, y - y1):min(y2 - y1, y + box_height - y1),
            max(0, x - x1):min(x2 - x1, x + box_width - x1),
        ] = True
        for label in range(1, count):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            inside_ratio = (
                np.count_nonzero(component & box_support) / area
                if area
                else 0.0
            )
            boundary_glyph = (
                inside_ratio >= 0.35
                and stats[label, cv2.CC_STAT_WIDTH]
                <= max(8, box_width // 2)
                and stats[label, cv2.CC_STAT_HEIGHT]
                <= max(8, int(box_height * 1.5))
            )
            if np.any(component & seed) or boundary_glyph:
                local_extended = extended_ink[y1:y2, x1:x2]
                local_extended[component] = 255
    ink = extended_ink
    cleanup = np.zeros_like(detected)
    if not np.any(ink):
        return cleanup
    if not text_items:
        return cv2.dilate(
            ink,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )

    height, width = detected.shape
    for item in text_items:
        if "box" not in item:
            continue
        x, y, box_width, box_height = (int(value) for value in item["box"])
        font_size = item.get("font_size")
        if isinstance(font_size, (int, float)) and font_size > 0:
            radius = max(3, min(6, int(round(font_size * 0.5))))
        else:
            radius = max(
                2,
                min(4, int(round(max(box_height, 1) * 0.02))),
            )
        search_pad = max(
            radius,
            max(4, min(12, int(round(max(box_height, 1) * 0.15)))),
        )
        x1 = max(0, x - search_pad)
        y1 = max(0, y - search_pad)
        x2 = min(width, x + box_width + search_pad)
        y2 = min(height, y + box_height + search_pad)
        local = ink[y1:y2, x1:x2]
        if not np.any(local):
            continue
        local_cleanup = cv2.dilate(
            local,
            np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8),
            iterations=1,
        )
        local_gray = cv2.cvtColor(
            source[y1:y2, x1:x2],
            cv2.COLOR_RGB2GRAY,
        )
        edge_map = cv2.Canny(local_gray, 24, 72)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (edge_map > 0).astype(np.uint8),
            connectivity=8,
        )
        protected = np.zeros_like(edge_map)
        box_support = np.zeros_like(edge_map, dtype=bool)
        box_support[
            max(0, y - y1):min(y2 - y1, y + box_height - y1),
            max(0, x - x1):min(x2 - x1, x + box_width - x1),
        ] = True
        for label in range(1, count):
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            component = labels == label
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            inside_ratio = (
                np.count_nonzero(component & box_support) / component_area
                if component_area
                else 0.0
            )
            if (
                (
                    component_width >= max(12, int(box_width * 0.65))
                    or component_height >= max(12, int(box_height * 0.8))
                )
                and (
                    min(component_width, component_height) <= 3
                    or inside_ratio < 0.25
                )
            ):
                protected[component] = 255
        if np.any(protected):
            protected = cv2.dilate(
                protected,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            local_cleanup[protected > 0] = 0
        cleanup[y1:y2, x1:x2] |= local_cleanup
    return cleanup


def _repair_text_background(
    image: np.ndarray,
    cleanup_mask: np.ndarray,
    text_items: list[dict] | None = None,
    large_inpainter=None,
) -> np.ndarray:
    source = np.asarray(image, dtype=np.uint8)
    if text_items:
        modeled = _repair_text_with_local_planes(
            source,
            cleanup_mask,
            text_items,
        )
        modeled_residual = background_residual_metrics(
            source,
            modeled,
            cleanup_mask,
        )
        if not has_background_residual(modeled_residual):
            return modeled

    repaired = repair_masked_background(
        image,
        cleanup_mask,
        large_inpainter=large_inpainter,
    )
    residual = background_residual_metrics(
        source,
        repaired,
        cleanup_mask,
    )
    if not has_background_residual(residual):
        return repaired

    escalated_inpainter = large_inpainter or inpaint_large_mask
    escalated = np.asarray(
        escalated_inpainter(source, cleanup_mask),
        dtype=np.uint8,
    )
    if escalated.shape != source.shape:
        raise ValueError(
            "escalated text inpaint output must match the source image"
        )
    result = escalated.copy()
    result[np.asarray(cleanup_mask) == 0] = source[
        np.asarray(cleanup_mask) == 0
    ]
    return result


def _repair_text_with_local_planes(
    image: np.ndarray,
    cleanup_mask: np.ndarray,
    text_items: list[dict],
) -> np.ndarray:
    source = np.asarray(image, dtype=np.uint8)
    cleanup = np.asarray(cleanup_mask) > 0
    output = source.astype(np.float32).copy()
    height, width = cleanup.shape
    for item in text_items:
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x, y, box_width, box_height = (int(value) for value in box)
        padding = max(8, min(24, int(round(box_height * 0.4))))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + box_width + padding)
        y2 = min(height, y + box_height + padding)
        target = cleanup[y1:y2, x1:x2]
        if not np.any(target):
            continue

        region = source[y1:y2, x1:x2].astype(np.float32)
        gray = cv2.cvtColor(
            region.astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        edges = cv2.dilate(
            cv2.Canny(gray, 24, 72),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ) > 0
        valid = ~target & ~edges
        colors = region[valid]
        if len(colors) < 20:
            continue
        color_median = np.median(colors, axis=0)
        color_spread = float(
            np.percentile(
                np.linalg.norm(colors - color_median, axis=1),
                90,
            )
        )
        distance = cv2.distanceTransform(
            (~target).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        near_background = valid & (distance <= 8)
        if np.count_nonzero(near_background) >= 20:
            background_color = np.median(
                region[near_background],
                axis=0,
            )
            if (
                float(np.max(background_color) - np.min(background_color))
                >= 32
                or float(np.mean(background_color)) < 210
                or color_spread >= 8
            ):
                interpolated = _interpolate_masked_region(region, target)
                local_output = output[y1:y2, x1:x2]
                local_output[target] = interpolated[target]
                continue
        median = np.median(colors, axis=0)
        distances = np.linalg.norm(colors - median, axis=1)
        color_limit = max(24.0, float(np.percentile(distances, 65)))
        keep = distances <= color_limit
        sample_y, sample_x = np.nonzero(valid)
        sample_y = sample_y[keep]
        sample_x = sample_x[keep]
        colors = colors[keep]
        if len(colors) < 20:
            continue

        region_height, region_width = target.shape
        sample_matrix = np.column_stack(
            (
                np.ones(len(sample_x)),
                sample_x / max(1, region_width - 1),
                sample_y / max(1, region_height - 1),
            )
        )
        coefficients = np.linalg.lstsq(
            sample_matrix,
            colors,
            rcond=None,
        )[0]
        grid_y, grid_x = np.indices(target.shape)
        grid_matrix = np.column_stack(
            (
                np.ones(grid_x.size),
                grid_x.ravel() / max(1, region_width - 1),
                grid_y.ravel() / max(1, region_height - 1),
            )
        )
        modeled = np.clip(
            grid_matrix @ coefficients,
            0,
            255,
        ).reshape(region.shape)

        local_output = output[y1:y2, x1:x2]
        local_output[target] = modeled[target]
    return np.clip(output, 0, 255).astype(np.uint8)


def _interpolate_masked_region(
    region: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Interpolate smooth colored backgrounds across glyph-sized holes."""
    source = np.asarray(region, dtype=np.float32)
    target = np.asarray(target, dtype=bool)
    horizontal = np.full_like(source, np.nan)
    vertical = np.full_like(source, np.nan)
    for row in range(target.shape[0]):
        missing = np.flatnonzero(target[row])
        known = np.flatnonzero(~target[row])
        if len(missing) and len(known) >= 2:
            for channel in range(3):
                horizontal[row, missing, channel] = np.interp(
                    missing,
                    known,
                    source[row, known, channel],
                )
    for column in range(target.shape[1]):
        missing = np.flatnonzero(target[:, column])
        known = np.flatnonzero(~target[:, column])
        if len(missing) and len(known) >= 2:
            for channel in range(3):
                vertical[missing, column, channel] = np.interp(
                    missing,
                    known,
                    source[known, column, channel],
                )

    result = source.copy()
    horizontal_valid = np.isfinite(horizontal[:, :, 0])
    vertical_valid = np.isfinite(vertical[:, :, 0])
    target_rows = target[np.any(target, axis=1)]
    target_columns = target[:, np.any(target, axis=0)]
    horizontal_occupancy = (
        float(np.mean(target_rows))
        if target_rows.size
        else 1.0
    )
    vertical_occupancy = (
        float(np.mean(target_columns))
        if target_columns.size
        else 1.0
    )
    if horizontal_occupancy <= vertical_occupancy:
        primary, primary_valid = horizontal, horizontal_valid
        fallback, fallback_valid = vertical, vertical_valid
    else:
        primary, primary_valid = vertical, vertical_valid
        fallback, fallback_valid = horizontal, horizontal_valid
    use_primary = target & primary_valid
    result[use_primary] = primary[use_primary]
    use_fallback = target & ~primary_valid & fallback_valid
    result[use_fallback] = fallback[use_fallback]
    return result


def _interpolate_text_item_boxes(
    image: np.ndarray,
    text_items: list[dict],
    padding: int = 4,
) -> np.ndarray:
    repaired = np.asarray(image, dtype=np.float32).copy()
    height, width = repaired.shape[:2]
    for item in text_items:
        x, y, box_width, box_height = (int(value) for value in item["box"])
        x1 = max(1, x - padding)
        y1 = max(1, y - padding)
        x2 = min(width - 1, x + box_width + padding)
        y2 = min(height - 1, y + box_height + padding)
        if x1 >= x2 or y1 >= y2:
            continue
        region_width = x2 - x1
        region_height = y2 - y1
        horizontal_weight = np.linspace(
            0.0, 1.0, region_width, dtype=np.float32
        )[None, :, None]
        horizontal = (
            repaired[y1:y2, x1 - 1][:, None] * (1.0 - horizontal_weight)
            + repaired[y1:y2, x2][:, None] * horizontal_weight
        )
        vertical_weight = np.linspace(
            0.0, 1.0, region_height, dtype=np.float32
        )[:, None, None]
        vertical = (
            repaired[y1 - 1, x1:x2][None] * (1.0 - vertical_weight)
            + repaired[y2, x1:x2][None] * vertical_weight
        )
        repaired[y1:y2, x1:x2] = (horizontal + vertical) * 0.5
    return np.clip(repaired, 0, 255).astype(np.uint8)


def _compose_exported_components(
    clean_background: np.ndarray,
    components: list[dict],
) -> np.ndarray:
    from PIL import Image

    canvas = Image.fromarray(clean_background).convert("RGBA")
    for component in components:
        with Image.open(component["path"]) as component_image:
            layer = component_image.convert("RGBA")
        canvas.alpha_composite(
            layer,
            dest=(int(component["x"]), int(component["y"])),
        )
    return np.asarray(canvas.convert("RGB"))


def _persist_element_masks(
    work_dir: Path,
    masks: list[np.ndarray],
) -> list[str]:
    masks_dir = (work_dir / "element-masks").resolve()
    masks_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, mask in enumerate(masks):
        mask_path = (masks_dir / f"{index:04d}.png").resolve()
        Image.fromarray(np.asarray(mask)).save(mask_path)
        paths.append(str(mask_path))
    return paths


def _isolated_large_inpainter(work_dir: Path):
    def isolated_inpainter(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        with tempfile.TemporaryDirectory(
            prefix="lama-",
            dir=work_dir,
        ) as temporary_dir:
            temporary_path = Path(temporary_dir)
            input_path = temporary_path / "input.png"
            mask_path = temporary_path / "mask.png"
            output_path = temporary_path / "output.png"
            _save_rgb(str(input_path), image)
            Image.fromarray(
                (np.asarray(mask) > 0).astype(np.uint8) * 255,
                mode="L",
            ).save(mask_path)
            inpaint_large_mask_isolated(
                input_path,
                mask_path,
                output_path,
            )
            return _load_rgb(output_path)

    return isolated_inpainter


def _apply_text_only_fallback(
    slide_data: dict,
    work_dir: Path,
    text_clean: np.ndarray,
    resource_isolation: bool = False,
) -> None:
    background_original_path = work_dir / "background-text-only-fallback.png"
    background_widescreen_path = (
        work_dir / "background-text-only-fallback-16x9.png"
    )
    _save_rgb(str(background_original_path), text_clean)
    background_kwargs = (
        {"large_inpainter": _isolated_large_inpainter(work_dir)}
        if resource_isolation
        else {}
    )
    widescreen_result = build_widescreen_background(
        text_clean,
        **background_kwargs,
    )
    (
        widescreen_background,
        content_offset_x,
        content_offset_y,
        widescreen_background_method,
    ) = widescreen_result
    canvas_height, canvas_width = widescreen_background.shape[:2]
    if widescreen_background_method == "identity":
        background_widescreen_path = background_original_path
    else:
        _save_rgb(str(background_widescreen_path), widescreen_background)
    slide_data.update({
        "background_path": str(background_widescreen_path),
        "background_original_path": str(background_original_path),
        "background_widescreen_path": str(background_widescreen_path),
        "components": [],
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "content_offset_x": content_offset_x,
        "content_offset_y": content_offset_y,
        "widescreen_background_method": widescreen_background_method,
        "conversion_mode": "text_editable_visual_fallback",
    })


def _finalize_slide_quality(
    slide_data: dict,
    lang: str,
    _resource_isolation: bool = False,
    _allow_text_only_fallback: bool = True,
) -> dict:
    work_dir = Path(slide_data.pop("_work_dir"))
    text_mask_path = Path(slide_data.pop("_text_mask_path"))
    text_clean_path_value = slide_data.pop("_text_clean_path", None)
    element_mask_paths = slide_data.pop("_element_mask_paths")
    element_masks = []
    img = None
    clean_background = None
    text_mask = None
    visual_only = None
    try:
        img = _load_rgb(slide_data["original_image_path"])
        clean_background = _load_rgb(slide_data["background_original_path"])
        with Image.open(text_mask_path) as stored_text_mask:
            text_mask = np.asarray(stored_text_mask.convert("L")).copy()
        for mask_path in element_mask_paths:
            with Image.open(mask_path) as stored_mask:
                element_masks.append(np.asarray(stored_mask).copy())

        components = slide_data["components"]
        forced_fallback_reason = None
        visual_only = _compose_exported_components(clean_background, components)
        visual_only_path = work_dir / "visual-only.png"
        _save_rgb(str(visual_only_path), visual_only)

        ocr_kwargs = {}
        if _resource_isolation:
            ocr_kwargs = {"isolated": True, "worker_root": work_dir}
        raster_text_items, raster_text_mask = detect_text(
            visual_only_path, lang=lang, **ocr_kwargs
        )
        raster_text_items, raster_text_mask = (
            _filter_probable_icon_text_analysis(
                raster_text_items,
                raster_text_mask,
            )
        )
        if raster_text_items:
            repair_kwargs = {"text_items": raster_text_items}
            if all("box" in item for item in raster_text_items):
                repair_kwargs["cleaned_rgb"] = _interpolate_text_item_boxes(
                    visual_only,
                    raster_text_items,
                )
            repair_exported_component_text(
                components,
                raster_text_mask,
                visual_only,
                **repair_kwargs,
            )
            visual_only = _compose_exported_components(
                clean_background,
                components,
            )
            _save_rgb(str(visual_only_path), visual_only)
            raster_text_items, raster_text_mask = detect_text(
                visual_only_path, lang=lang, **ocr_kwargs
            )
            raster_text_items, raster_text_mask = (
                _filter_probable_icon_text_analysis(
                    raster_text_items,
                    raster_text_mask,
                )
            )
        if raster_text_items:
            repair_exported_component_text(
                components,
                raster_text_mask,
                visual_only,
                text_items=raster_text_items,
                clear_alpha=True,
            )
            visual_only = _compose_exported_components(
                clean_background,
                components,
            )
            _save_rgb(str(visual_only_path), visual_only)
            raster_text_items, _ = detect_text(
                visual_only_path, lang=lang, **ocr_kwargs
            )
            raster_text_items, _ = _filter_probable_icon_text_analysis(
                raster_text_items,
                np.zeros(img.shape[:2], dtype=np.uint8),
            )
        if raster_text_items:
            forced_fallback_reason = "component_raster_text"

        quality_text_items = slide_data.get("text_items") or []
        quality_text_mask = _build_text_cleanup_mask(
            img,
            text_mask,
            quality_text_items,
        )
        if (
            forced_fallback_reason is None
            and quality_text_items
            and _has_component_text_overlap(
                element_masks,
                quality_text_mask,
            )
        ):
            forced_fallback_reason = "component_text_overlap"
        quality = visual_difference(img, visual_only, quality_text_mask)
        removal_mask = build_removal_mask(
            element_masks,
            quality_text_mask,
        )
        background_residual = background_residual_metrics(
            img,
            clean_background,
            removal_mask,
        )
        slide_data["background_residual"] = background_residual
        diagnostics_dir = (work_dir / "diagnostics").resolve()
        write_segmentation_diagnostics(
            diagnostics_dir,
            source=img,
            masks=element_masks,
            reconstructed=visual_only,
            metrics=quality,
        )
        fallback_reason = None
        if forced_fallback_reason is not None:
            fallback_reason = forced_fallback_reason
        elif has_background_residual(background_residual):
            fallback_reason = "background_residual"
        elif needs_text_only_fallback(quality):
            fallback_reason = "visible_visual_artifacts"
        if fallback_reason is not None:
            if not _allow_text_only_fallback:
                raise VisualSegmentationError(
                    f"agent-managed quality failed: {fallback_reason}"
                )
            if text_clean_path_value is None:
                text_clean = img.copy()
            else:
                text_clean = _load_rgb(text_clean_path_value)
            original_quality = dict(quality)
            _apply_text_only_fallback(
                slide_data,
                work_dir,
                text_clean,
                resource_isolation=_resource_isolation,
            )
            fallback_background_residual = background_residual_metrics(
                img,
                text_clean,
                quality_text_mask,
            )
            slide_data["background_residual"] = (
                fallback_background_residual
            )
            if has_background_residual(fallback_background_residual):
                raise VisualSegmentationError(
                    "text-clean fallback still contains raster residuals"
                )
            quality = visual_difference(img, text_clean, quality_text_mask)
            require_visual_quality(quality)
            slide_data["quality"] = quality
            slide_data["quality_fallback"] = {
                "reason": fallback_reason,
                "original_metrics": original_quality,
                "original_background_residual": background_residual,
            }
            return slide_data
        try:
            require_visual_quality(quality)
        except VisualSegmentationError as exc:
            raise VisualSegmentationError(
                f"{exc}; mae={quality['mae']:.3f}, p95={quality['p95']:.3f}, "
                f"diagnostics={diagnostics_dir}"
            ) from exc
        slide_data["quality"] = quality
        return slide_data
    finally:
        element_masks.clear()
        img = None
        clean_background = None
        text_mask = None
        visual_only = None


def _has_component_text_overlap(
    element_masks: list[np.ndarray],
    text_mask: np.ndarray,
) -> bool:
    text = np.asarray(text_mask) > 0
    if not np.any(text):
        return False
    for element_mask in element_masks:
        component = np.asarray(element_mask) > 0
        component_pixels = int(np.count_nonzero(component))
        if component_pixels == 0:
            continue
        overlap = int(np.count_nonzero(component & text))
        if overlap >= 16 and overlap / component_pixels >= 0.02:
            return True
    return False


def _generate_filtered_object_proposals(
    image: np.ndarray,
    text_mask: np.ndarray,
    detector,
):
    owns_detector = detector is None
    if owns_detector:
        detector = create_object_detector()
    try:
        return filter_text_overlapping_proposals(
            generate_object_proposals(image, detector),
            text_mask,
        )
    finally:
        if owns_detector:
            detector = None
            _release_visual_resources()


def _generate_filtered_object_proposals_isolated(
    image: np.ndarray,
    text_mask: np.ndarray,
    work_dir: Path,
) -> list[ObjectProposal]:
    with tempfile.TemporaryDirectory(
        prefix="object-",
        dir=work_dir,
    ) as temporary_dir:
        temporary_dir = Path(temporary_dir)
        image_path = temporary_dir / "image.png"
        mask_path = temporary_dir / "text-mask.png"
        result_path = temporary_dir / "result.json"
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
            image_path
        )
        Image.fromarray(
            np.asarray(text_mask, dtype=np.uint8),
            mode="L",
        ).save(mask_path)
        module_dir = Path(__file__).resolve().parent
        worker_path = module_dir / "scripts" / "object_worker.py"
        if not worker_path.is_file():
            worker_path = module_dir / "object_worker.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(worker_path),
                "--image",
                str(image_path),
                "--text-mask",
                str(mask_path),
                "--result",
                str(result_path),
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Isolated object worker failed: {detail}")
        if not result_path.is_file():
            raise RuntimeError(
                "Isolated object worker did not create its result"
            )
        records = json.loads(result_path.read_text(encoding="utf-8"))
    return [
        ObjectProposal(
            **{
                **record,
                "box_xyxy": tuple(record["box_xyxy"]),
                "crop_box": tuple(record["crop_box"]),
            }
        )
        for record in records
    ]


def _generate_sam_candidates_isolated(
    image: np.ndarray,
    text_mask: np.ndarray | None,
    proposals: list[ObjectProposal] | None,
    work_dir: Path,
    *,
    mode: str,
) -> list[MaskCandidate]:
    with tempfile.TemporaryDirectory(
        prefix="sam-",
        dir=work_dir,
    ) as temporary_dir:
        temporary_dir = Path(temporary_dir)
        image_path = temporary_dir / "image.png"
        result_path = temporary_dir / "result.json"
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
            image_path
        )
        command = [
            sys.executable,
            str(
                (
                    Path(__file__).resolve().parent
                    / "scripts"
                    / "sam_worker.py"
                )
                if (
                    Path(__file__).resolve().parent
                    / "scripts"
                    / "sam_worker.py"
                ).is_file()
                else Path(__file__).resolve().with_name("sam_worker.py")
            ),
            "--mode",
            mode,
            "--image",
            str(image_path),
            "--result",
            str(result_path),
        ]
        if text_mask is not None:
            text_mask_path = temporary_dir / "text-mask.png"
            Image.fromarray(
                np.asarray(text_mask, dtype=np.uint8),
                mode="L",
            ).save(text_mask_path)
            command.extend(["--text-mask", str(text_mask_path)])
        if proposals is not None:
            proposals_path = temporary_dir / "proposals.json"
            proposals_path.write_text(
                json.dumps(
                    [proposal.__dict__ for proposal in proposals],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            command.extend(["--proposals", str(proposals_path)])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Isolated SAM worker failed: {detail}")
        if not result_path.is_file():
            raise RuntimeError("Isolated SAM worker did not create its result")
        records = json.loads(result_path.read_text(encoding="utf-8"))

    candidates = []
    for record in records:
        mask_shape = tuple(record.pop("mask_shape"))
        packed = np.frombuffer(
            base64.b64decode(record.pop("mask")),
            dtype=np.uint8,
        )
        mask = np.unpackbits(
            packed,
            count=int(np.prod(mask_shape)),
        ).reshape(mask_shape).astype(bool, copy=False)
        if record["crop_box"] is not None:
            record["crop_box"] = tuple(record["crop_box"])
        if record["object_box"] is not None:
            record["object_box"] = tuple(record["object_box"])
        candidates.append(MaskCandidate(mask=mask, **record))
    return candidates


def _packed_mask_fields(mask: np.ndarray, name: str = "mask") -> dict:
    binary = np.asarray(mask, dtype=bool)
    return {
        name: base64.b64encode(
            np.packbits(binary, axis=None).tobytes()
        ).decode("ascii"),
        f"{name}_shape": list(binary.shape),
    }


def _unpack_mask_fields(record: dict, name: str = "mask") -> np.ndarray:
    shape = tuple(record[f"{name}_shape"])
    packed = np.frombuffer(
        base64.b64decode(record[name]),
        dtype=np.uint8,
    )
    return np.unpackbits(
        packed,
        count=int(np.prod(shape)),
    ).reshape(shape).astype(bool, copy=False)


def _recheck_visual_element_holes_isolated(
    image: np.ndarray,
    elements: list,
    work_dir: Path,
) -> None:
    if not elements or not any(
        element.object_box is not None for element in elements
    ):
        return
    with tempfile.TemporaryDirectory(
        prefix="sam-recheck-",
        dir=work_dir,
    ) as temporary_dir:
        temporary_dir = Path(temporary_dir)
        image_path = temporary_dir / "image.png"
        elements_path = temporary_dir / "elements.json"
        result_path = temporary_dir / "result.json"
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
            image_path
        )
        elements_path.write_text(
            json.dumps(
                [
                    {
                        **_packed_mask_fields(element.mask),
                        **_packed_mask_fields(
                            element.semantic_mask,
                            "semantic_mask",
                        ),
                        "z_index": element.z_index,
                        "score": element.score,
                        "source": element.source,
                        "object_box": element.object_box,
                    }
                    for element in elements
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        module_dir = Path(__file__).resolve().parent
        worker_path = module_dir / "scripts" / "sam_worker.py"
        if not worker_path.is_file():
            worker_path = module_dir / "sam_worker.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(worker_path),
                "--mode",
                "recheck",
                "--image",
                str(image_path),
                "--elements",
                str(elements_path),
                "--result",
                str(result_path),
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Isolated SAM worker failed: {detail}")
        if not result_path.is_file():
            raise RuntimeError("Isolated SAM worker did not create its result")
        records = json.loads(result_path.read_text(encoding="utf-8"))

    if len(records) != len(elements):
        raise RuntimeError("Isolated SAM recheck returned the wrong mask count")
    for element, record in zip(elements, records):
        element.mask = _unpack_mask_fields(record)
        element.semantic_mask = _unpack_mask_fields(record, "semantic_mask")


def _process_image_isolated(
    image_path: Path,
    work_dir: Path,
    lang: str,
    text_analysis: dict,
) -> dict:
    request_path = (work_dir / "visual-worker-request.json").resolve()
    result_path = (work_dir / "visual-worker-result.json").resolve()
    request_path.write_text(
        json.dumps({"text_analysis": text_analysis}, ensure_ascii=False),
        encoding="utf-8",
    )
    module_dir = Path(__file__).resolve().parent
    worker_path = module_dir / "scripts" / "visual_worker.py"
    if not worker_path.is_file():
        worker_path = module_dir / "visual_worker.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(worker_path),
            "--image",
            str(image_path),
            "--work-dir",
            str(work_dir),
            "--lang",
            lang,
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Isolated visual worker failed: {detail}")
    if not result_path.is_file():
        raise RuntimeError("Isolated visual worker did not create its result")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _pack_mask_references(references):
    packed_masks = {}
    bindings = []
    for owner, attribute in references:
        mask = getattr(owner, attribute)
        key = id(mask)
        if key not in packed_masks:
            packed_masks[key] = (
                np.packbits(np.asarray(mask, dtype=bool), axis=None),
                mask.shape,
            )
        bindings.append((owner, attribute, key))
        setattr(owner, attribute, None)
    return packed_masks, bindings


def _restore_mask_references(state) -> None:
    packed_masks, bindings = state
    restored = {}
    for key, (packed_mask, shape) in packed_masks.items():
        restored[key] = np.unpackbits(
            packed_mask,
            count=shape[0] * shape[1],
        ).reshape(shape).astype(bool, copy=False)
    for owner, attribute, key in bindings:
        setattr(owner, attribute, restored[key])


def _process_image(
    image_path: Path,
    work_dir: Path,
    object_detector,
    mask_generator,
    lang: str,
    text_analysis: dict | None = None,
    defer_quality: bool = False,
    _resource_isolation: bool = False,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    background_kwargs = (
        {"large_inpainter": _isolated_large_inpainter(work_dir)}
        if _resource_isolation
        else {}
    )
    img = _load_rgb(str(image_path))
    img_h, img_w = img.shape[:2]
    if text_analysis is None:
        text_items, text_mask = detect_text(image_path, lang=lang)
        text_items, text_mask = _filter_probable_icon_text_analysis(
            text_items,
            text_mask,
        )
        text_mask_path = (work_dir / "source-text-mask.png").resolve()
        Image.fromarray(text_mask, mode="L").save(text_mask_path)
    else:
        text_items = text_analysis["items"]
        text_mask_path = Path(text_analysis["mask_path"]).resolve()
        with Image.open(text_mask_path) as stored_text_mask:
            text_mask = np.asarray(stored_text_mask.convert("L")).copy()
    text_ink_mask = _build_text_ink_mask(img, text_mask)
    valid_text_items = bool(text_items) and all(
        "box" in item for item in text_items
    )
    text_cleanup_mask = _build_text_cleanup_mask(
        img,
        text_mask,
        text_items if valid_text_items else [],
    )
    text_clean_image = None
    text_clean_path = None
    if valid_text_items:
        text_clean_path = (
            text_analysis.get("text_clean_path")
            if text_analysis is not None
            else None
        )
        if text_clean_path is None:
            text_clean_image = _repair_text_background(
                img,
                text_cleanup_mask,
                text_items=text_items,
                **background_kwargs,
            )
            text_clean_path = work_dir / "text-clean.png"
            _save_rgb(str(text_clean_path), text_clean_image)
        else:
            text_clean_path = Path(text_clean_path).resolve()
            with Image.open(text_clean_path) as stored_text_clean:
                text_clean_image = np.asarray(
                    stored_text_clean.convert("RGB")
                ).copy()

    proposal_detector = (
        None if _resource_isolation else object_detector
    )
    if _resource_isolation:
        proposals = _generate_filtered_object_proposals_isolated(
            img,
            text_mask,
            work_dir,
        )
    else:
        proposals = _generate_filtered_object_proposals(
            img,
            text_mask,
            proposal_detector,
        )
    if _resource_isolation:
        candidates = _generate_sam_candidates_isolated(
            img,
            text_ink_mask,
            proposals,
            work_dir,
            mode="prompted",
        )
        prompt_free_candidates = _generate_sam_candidates_isolated(
            img,
            None,
            None,
            work_dir,
            mode="automatic",
        )
    else:
        candidates = generate_prompted_mask_candidates(
            img,
            proposals,
            mask_generator,
            text_ink_mask,
        )
        prompt_free_candidates = generate_mask_candidates(
            img,
            mask_generator,
            crop_size=max(img.shape[:2]),
            include_geometry=False,
            min_score=0.90,
        )
    candidates.extend(
        filter_prompt_free_candidates(
            prompt_free_candidates,
            candidates,
            text_ink_mask,
        )
    )
    for round_index in range(3):
        elements = resolve_visual_elements(candidates)
        element_masks = [element.mask for element in elements]
        validate_visual_masks(element_masks)
        clean_background = build_clean_background(
            img,
            element_masks,
            text_cleanup_mask,
            **background_kwargs,
        )
        packed_masks = None
        if _resource_isolation:
            references = [
                *((candidate, "mask") for candidate in candidates),
                *((element, "mask") for element in elements),
                *((element, "semantic_mask") for element in elements),
            ]
            packed_masks = _pack_mask_references(references)
            element_masks.clear()
            mask_generator = None
            _release_visual_resources()
            gc.collect()
        try:
            if _resource_isolation:
                residual_proposals = (
                    _generate_filtered_object_proposals_isolated(
                        clean_background,
                        text_mask,
                        work_dir,
                    )
                )
            else:
                residual_proposals = _generate_filtered_object_proposals(
                    clean_background,
                    text_mask,
                    proposal_detector,
                )
        finally:
            if packed_masks is not None:
                _restore_mask_references(packed_masks)
                element_masks = [element.mask for element in elements]
        if _resource_isolation:
            residual_candidates = _generate_sam_candidates_isolated(
                clean_background,
                text_ink_mask,
                residual_proposals,
                work_dir,
                mode="prompted",
            )
        else:
            residual_candidates = generate_prompted_mask_candidates(
                clean_background,
                residual_proposals,
                mask_generator,
                text_ink_mask,
            )
        residual_candidates = filter_unchanged_residual_candidates(
            img,
            clean_background,
            residual_candidates,
            text_ink_mask,
        )
        residual_candidates, attached_count = reconcile_residual_candidates(
            residual_candidates,
            candidates,
            img.shape[:2],
        )
        if not residual_candidates:
            if attached_count:
                elements = resolve_visual_elements(candidates)
                element_masks = [element.mask for element in elements]
                validate_visual_masks(element_masks)
                clean_background = build_clean_background(
                    img,
                    element_masks,
                    text_cleanup_mask,
                    **background_kwargs,
                )
            break
        residual_diagnostics = work_dir / f"residual-round-{round_index + 1}"
        write_segmentation_diagnostics(
            residual_diagnostics,
            source=img,
            masks=[candidate.mask for candidate in residual_candidates],
            reconstructed=clean_background,
            metrics={"residual_count": len(residual_candidates)},
        )
        if round_index == 2:
            raise VisualSegmentationError(
                "clean background still contains independent visual elements; "
                f"diagnostics={residual_diagnostics.resolve()}"
            )
        candidates.extend(residual_candidates)

    if _resource_isolation:
        _recheck_visual_element_holes_isolated(
            img,
            elements,
            work_dir,
        )
    else:
        recheck_visual_element_holes(img, elements, mask_generator)
    element_masks = [element.mask for element in elements]
    semantic_masks = [element.semantic_mask for element in elements]
    validate_visual_masks(element_masks)
    clean_background = build_clean_background(
        img,
        element_masks,
        text_cleanup_mask,
        **background_kwargs,
    )
    export_kwargs = {"semantic_masks": semantic_masks}
    if valid_text_items:
        export_kwargs["text_items"] = text_items
        export_kwargs["text_clean_image"] = text_clean_image
    components = export_visual_components(
        img,
        element_masks,
        work_dir / "components",
        text_mask,
        **export_kwargs,
    )
    background_original_path = work_dir / "background-original.png"
    background_widescreen_path = work_dir / "background-16x9.png"
    background_removal_mask_path = work_dir / "background-removal-mask.png"
    background_difference_path = work_dir / "background-difference.png"
    _save_rgb(str(background_original_path), clean_background)
    (
        widescreen_background,
        content_offset_x,
        content_offset_y,
        widescreen_background_method,
    ) = build_widescreen_background(
        clean_background,
        **background_kwargs,
    )
    canvas_height, canvas_width = widescreen_background.shape[:2]
    if widescreen_background_method == "identity":
        background_widescreen_path = background_original_path
    else:
        _save_rgb(str(background_widescreen_path), widescreen_background)
    removal_mask = build_removal_mask(element_masks, text_cleanup_mask)
    Image.fromarray(removal_mask, mode="L").save(background_removal_mask_path)
    _save_rgb(
        str(background_difference_path),
        cv2.absdiff(img, clean_background),
    )
    element_mask_paths = _persist_element_masks(work_dir, element_masks)
    slide_data = {
        "background_path": str(background_widescreen_path),
        "background_original_path": str(background_original_path),
        "background_widescreen_path": str(background_widescreen_path),
        "components": components,
        "text_items": text_items,
        "img_width": img_w,
        "img_height": img_h,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "content_offset_x": content_offset_x,
        "content_offset_y": content_offset_y,
        "widescreen_background_method": widescreen_background_method,
        "original_image_path": str(image_path),
        "_work_dir": str(work_dir.resolve()),
        "_text_mask_path": str(text_mask_path),
        "_element_mask_paths": element_mask_paths,
    }
    if text_clean_path is not None:
        slide_data["_text_clean_path"] = str(text_clean_path)
    if defer_quality:
        return slide_data
    return _finalize_slide_quality(slide_data, lang)


_PREPARED_PAGE_SCHEMA_VERSION = 1
_PREPARED_PAGE_NAME = "prepared_page.json"
_PREPARED_PAGE_FIELDS = {
    "schema_version",
    "phase",
    "resource_isolation",
    "initial_component_count",
    "components",
    "text_items",
    "dimensions",
    "assets",
}
_PREPARED_DIMENSION_FIELDS = {
    "img_width",
    "img_height",
    "canvas_width",
    "canvas_height",
    "content_offset_x",
    "content_offset_y",
    "widescreen_background_method",
}
_PREPARED_ASSET_FIELDS = {
    "source_image",
    "ocr_mask",
    "text_clean",
    "element_masks",
    "background_original",
    "background_widescreen",
    "background_removal_mask",
    "background_difference",
}
_PREPARED_COMPONENT_FIELDS = {"x", "y", "w", "h", "area", "z_index"}
_PREPARED_TEXT_FIELDS = {
    "box",
    "text",
    "font_size",
    "color",
    "bold",
    "font",
    "align",
    "confidence",
}


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )


def _validate_prepared_work_dir(work_dir: str | Path) -> Path:
    lexical = Path(os.path.abspath(work_dir))
    if lexical.resolve(strict=False) != lexical:
        raise ValueError(f"work directory resolves through a link: {lexical}")
    if lexical.exists() or lexical.is_symlink():
        status = os.lstat(lexical)
        if _is_link_or_reparse(status):
            raise ValueError(f"work directory is a link or reparse point: {lexical}")
        if not lexical.is_dir():
            raise ValueError(f"work directory is not a directory: {lexical}")
    else:
        lexical.mkdir(parents=True)
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise ValueError(f"work directory resolves through a link: {lexical}")
    return resolved


def _reject_prepared_links(work_dir: Path) -> None:
    for path in work_dir.rglob("*"):
        if _is_link_or_reparse(os.lstat(path)):
            raise ValueError(
                f"prepared asset is a link or reparse point: {path}"
            )


def _prepared_owned_file(
    work_dir: Path,
    value: str | Path,
    label: str,
    *,
    relative_only: bool = False,
) -> Path:
    supplied = Path(value)
    if relative_only and supplied.is_absolute():
        raise ValueError(f"{label} asset path must be relative")
    if ".." in supplied.parts:
        raise ValueError(f"{label} asset path must not contain '..'")
    lexical = supplied if supplied.is_absolute() else work_dir / supplied
    lexical = Path(os.path.abspath(lexical))
    if not lexical.is_relative_to(work_dir):
        raise ValueError(f"{label} asset path is outside the work directory")

    current = lexical
    while True:
        try:
            status = os.lstat(current)
        except FileNotFoundError as exc:
            raise ValueError(f"{label} asset file is missing: {lexical}") from exc
        if _is_link_or_reparse(status):
            raise ValueError(f"{label} asset is a link or reparse point: {current}")
        if current == work_dir:
            break
        current = current.parent
        if not current.is_relative_to(work_dir):
            raise ValueError(f"{label} asset path is outside the work directory")

    resolved = lexical.resolve(strict=True)
    if resolved != lexical or not resolved.is_relative_to(work_dir):
        raise ValueError(f"{label} asset path escapes the work directory")
    if not resolved.is_file():
        raise ValueError(f"{label} asset is not a file: {resolved}")
    return resolved


def _prepared_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_asset_record(work_dir: Path, path: str | Path, label: str) -> dict:
    owned = _prepared_owned_file(work_dir, path, label)
    return {
        "path": owned.relative_to(work_dir).as_posix(),
        "sha256": _prepared_sha256(owned),
    }


def _load_prepared_asset(work_dir: Path, record: object, label: str) -> str:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} asset record is invalid")
    relative = record["path"]
    expected = record["sha256"]
    if not isinstance(relative, str):
        raise ValueError(f"{label} asset path is invalid")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{label} asset sha256 is invalid")
    owned = _prepared_owned_file(
        work_dir,
        relative,
        label,
        relative_only=True,
    )
    if _prepared_sha256(owned) != expected:
        raise ValueError(f"{label} asset sha256 mismatch")
    return str(owned)


def _is_prepared_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_prepared_box(box: object, width: int, height: int, label: str) -> None:
    if (
        not isinstance(box, list)
        or len(box) != 4
        or not all(_is_prepared_int(value) for value in box)
    ):
        raise ValueError(f"prepared page {label} box is invalid")
    x, y, box_width, box_height = box
    if (
        x < 0
        or y < 0
        or box_width <= 0
        or box_height <= 0
        or x + box_width > width
        or y + box_height > height
    ):
        raise ValueError(f"prepared page {label} box is out of bounds")


def _validate_prepared_payload(manifest: dict) -> None:
    dimensions = manifest["dimensions"]
    if not isinstance(dimensions, dict) or set(dimensions) != _PREPARED_DIMENSION_FIELDS:
        raise ValueError("prepared page dimensions are invalid")
    integer_fields = _PREPARED_DIMENSION_FIELDS - {"widescreen_background_method"}
    if any(not _is_prepared_int(dimensions[field]) for field in integer_fields):
        raise ValueError("prepared page dimension values are invalid")
    image_width = dimensions["img_width"]
    image_height = dimensions["img_height"]
    canvas_width = dimensions["canvas_width"]
    canvas_height = dimensions["canvas_height"]
    offset_x = dimensions["content_offset_x"]
    offset_y = dimensions["content_offset_y"]
    if (
        min(image_width, image_height, canvas_width, canvas_height) <= 0
        or min(offset_x, offset_y) < 0
        or offset_x + image_width > canvas_width
        or offset_y + image_height > canvas_height
        or dimensions["widescreen_background_method"]
        not in {"identity", "ambient", "outpaint"}
    ):
        raise ValueError("prepared page dimension values are invalid")

    text_items = manifest["text_items"]
    if not isinstance(text_items, list):
        raise ValueError("prepared page text_items are invalid")
    for item in text_items:
        if not isinstance(item, dict) or set(item) != _PREPARED_TEXT_FIELDS:
            raise ValueError("prepared page text item fields are invalid")
        _validate_prepared_box(item["box"], image_width, image_height, "text item")
        font_size = item["font_size"]
        confidence = item["confidence"]
        color = item["color"]
        if (
            not isinstance(item["text"], str)
            or not item["text"]
            or not isinstance(item["font"], str)
            or not item["font"]
            or not isinstance(font_size, (int, float))
            or isinstance(font_size, bool)
            or not math.isfinite(font_size)
            or font_size <= 0
            or not isinstance(color, str)
            or len(color) != 7
            or not color.startswith("#")
            or any(character not in "0123456789abcdefABCDEF" for character in color[1:])
            or not isinstance(item["bold"], bool)
            or not _is_prepared_int(item["align"])
            or item["align"] not in {0, 1, 2}
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("prepared page text item values are invalid")

    components = manifest["components"]
    if not isinstance(components, list):
        raise ValueError("prepared page components are invalid")
    for component in components:
        if not isinstance(component, dict) or set(component) != {"asset", "metadata"}:
            raise ValueError("prepared page component record is invalid")
        metadata = component["metadata"]
        if (
            not isinstance(metadata, dict)
            or set(metadata) != _PREPARED_COMPONENT_FIELDS
            or not all(_is_prepared_int(value) for value in metadata.values())
        ):
            raise ValueError("prepared page component metadata is invalid")
        if (
            metadata["x"] < 0
            or metadata["y"] < 0
            or metadata["w"] <= 0
            or metadata["h"] <= 0
            or metadata["area"] <= 0
            or metadata["area"] > metadata["w"] * metadata["h"]
            or metadata["z_index"] < 0
            or metadata["x"] + metadata["w"] > image_width
            or metadata["y"] + metadata["h"] > image_height
        ):
            raise ValueError("prepared page component metadata is invalid")


def _write_prepared_page(slide_data: dict, work_dir: Path) -> Path:
    dimensions = {
        field: slide_data[field]
        for field in _PREPARED_DIMENSION_FIELDS
    }
    assets = {
        "source_image": _prepared_asset_record(
            work_dir, slide_data["original_image_path"], "source image"
        ),
        "ocr_mask": _prepared_asset_record(
            work_dir, slide_data["_text_mask_path"], "OCR mask"
        ),
        "text_clean": (
            _prepared_asset_record(
                work_dir, slide_data["_text_clean_path"], "text-clean image"
            )
            if slide_data.get("_text_clean_path") is not None
            else None
        ),
        "element_masks": [
            _prepared_asset_record(work_dir, path, "element mask")
            for path in slide_data["_element_mask_paths"]
        ],
        "background_original": _prepared_asset_record(
            work_dir,
            slide_data["background_original_path"],
            "original background",
        ),
        "background_widescreen": _prepared_asset_record(
            work_dir,
            slide_data["background_widescreen_path"],
            "widescreen background",
        ),
        "background_removal_mask": _prepared_asset_record(
            work_dir,
            slide_data.get(
                "background_removal_mask_path",
                work_dir / "background-removal-mask.png",
            ),
            "background removal mask",
        ),
        "background_difference": _prepared_asset_record(
            work_dir,
            slide_data.get(
                "background_difference_path",
                work_dir / "background-difference.png",
            ),
            "background difference",
        ),
    }
    components = []
    for component in slide_data["components"]:
        metadata = {key: value for key, value in component.items() if key != "path"}
        components.append({
            "asset": _prepared_asset_record(
                work_dir, component["path"], "component RGBA"
            ),
            "metadata": metadata,
        })
    manifest = {
        "schema_version": _PREPARED_PAGE_SCHEMA_VERSION,
        "phase": "initial_layers",
        "resource_isolation": slide_data["_resource_isolation"],
        "initial_component_count": len(components),
        "components": components,
        "text_items": slide_data["text_items"],
        "dimensions": dimensions,
        "assets": assets,
    }
    _validate_prepared_payload(manifest)
    state_path = work_dir / _PREPARED_PAGE_NAME
    if state_path.exists() or state_path.is_symlink():
        if _is_link_or_reparse(os.lstat(state_path)):
            raise ValueError("prepared_page.json is a link or reparse point")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=work_dir,
            prefix=".prepared_page.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(manifest, temporary, ensure_ascii=False, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, state_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return state_path.resolve()


def load_component_layers(state_path: str | Path) -> dict:
    lexical_state = Path(os.path.abspath(state_path))
    if lexical_state.name != _PREPARED_PAGE_NAME:
        raise ValueError(f"state path must name {_PREPARED_PAGE_NAME}")
    work_dir = _validate_prepared_work_dir(lexical_state.parent)
    state_file = _prepared_owned_file(work_dir, lexical_state, "prepared state")
    try:
        manifest = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("prepared page state is invalid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != _PREPARED_PAGE_FIELDS:
        raise ValueError("prepared page state fields are invalid")
    if manifest["schema_version"] != _PREPARED_PAGE_SCHEMA_VERSION:
        raise ValueError("prepared page schema_version is invalid")
    if manifest["phase"] != "initial_layers":
        raise ValueError("prepared page phase is invalid")
    if not isinstance(manifest["resource_isolation"], bool):
        raise ValueError("prepared page resource_isolation is invalid")
    _validate_prepared_payload(manifest)
    initial_count = manifest["initial_component_count"]
    if (
        not isinstance(initial_count, int)
        or isinstance(initial_count, bool)
        or initial_count < 0
        or initial_count != len(manifest["components"])
    ):
        raise ValueError("prepared page initial_component_count is invalid")
    dimensions = manifest["dimensions"]
    assets = manifest["assets"]
    if not isinstance(assets, dict) or set(assets) != _PREPARED_ASSET_FIELDS:
        raise ValueError("prepared page assets are invalid")
    if not isinstance(assets["element_masks"], list):
        raise ValueError("prepared page element masks are invalid")

    loaded_assets = {
        key: _load_prepared_asset(work_dir, assets[key], key)
        for key in (
            "source_image",
            "ocr_mask",
            "background_original",
            "background_widescreen",
            "background_removal_mask",
            "background_difference",
        )
    }
    text_clean = assets["text_clean"]
    if text_clean is not None:
        loaded_assets["text_clean"] = _load_prepared_asset(
            work_dir, text_clean, "text_clean"
        )
    element_mask_paths = [
        _load_prepared_asset(work_dir, record, "element mask")
        for record in assets["element_masks"]
    ]
    components = []
    for record in manifest["components"]:
        components.append({
            **record["metadata"],
            "path": _load_prepared_asset(
                work_dir, record["asset"], "component RGBA"
            ),
        })

    return {
        "phase": "initial_layers",
        "initial_component_count": initial_count,
        "state_path": str(state_file),
        "_resource_isolation": manifest["resource_isolation"],
        **dimensions,
        "components": components,
        "text_items": manifest["text_items"],
        "original_image_path": loaded_assets["source_image"],
        "background_path": loaded_assets["background_widescreen"],
        "background_original_path": loaded_assets["background_original"],
        "background_widescreen_path": loaded_assets["background_widescreen"],
        "background_removal_mask_path": loaded_assets[
            "background_removal_mask"
        ],
        "background_difference_path": loaded_assets["background_difference"],
        "_work_dir": str(work_dir),
        "_text_mask_path": loaded_assets["ocr_mask"],
        "_element_mask_paths": element_mask_paths,
        **(
            {"_text_clean_path": loaded_assets["text_clean"]}
            if text_clean is not None
            else {}
        ),
    }


def prepare_component_layers(
    image_path: str | Path,
    work_dir: str | Path,
    *,
    lang: str,
    resource_isolation: bool,
) -> dict:
    source = _resolve_image_path(image_path)
    owned_work_dir = _validate_prepared_work_dir(work_dir)
    _reject_prepared_links(owned_work_dir)
    owned_source = owned_work_dir / f"source-image{source.suffix.lower()}"
    if source != owned_source:
        shutil.copyfile(source, owned_source)

    text_mask = None
    try:
        ocr_kwargs = (
            {"isolated": True, "worker_root": owned_work_dir}
            if resource_isolation
            else {}
        )
        text_items, text_mask = detect_text(owned_source, lang=lang, **ocr_kwargs)
        text_items, text_mask = _filter_probable_icon_text_analysis(
            text_items,
            text_mask,
        )
        text_mask_path = (owned_work_dir / "source-text-mask.png").resolve()
        Image.fromarray(text_mask, mode="L").save(text_mask_path)
        text_analysis = {
            "items": text_items,
            "mask_path": str(text_mask_path),
        }
    finally:
        text_mask = None
        close_ocr_engines()

    if resource_isolation and text_items and all("box" in item for item in text_items):
        source_image = _load_rgb(owned_source)
        with Image.open(text_analysis["mask_path"]) as stored_text_mask:
            stored_mask = np.asarray(stored_text_mask.convert("L")).copy()
        removal_mask = _build_text_cleanup_mask(
            source_image,
            stored_mask,
            text_items,
        )
        removal_mask_path = owned_work_dir / "text-clean-removal-mask.png"
        Image.fromarray(removal_mask, mode="L").save(removal_mask_path)
        text_clean_path = owned_work_dir / "text-clean.png"
        text_clean = _repair_text_background(
            source_image,
            removal_mask,
            text_items=text_items,
            large_inpainter=_isolated_large_inpainter(owned_work_dir),
        )
        _save_rgb(text_clean_path, text_clean)
        text_analysis["text_clean_path"] = str(text_clean_path)

    object_detector = None
    mask_generator = None
    try:
        if resource_isolation:
            slide_data = _process_image_isolated(
                owned_source,
                owned_work_dir,
                lang,
                text_analysis,
            )
        else:
            object_detector = create_object_detector()
            mask_generator = create_sam_generator(resolve_sam_checkpoint())
            slide_data = _process_image(
                owned_source,
                owned_work_dir,
                object_detector,
                mask_generator,
                lang,
                text_analysis=text_analysis,
                defer_quality=True,
            )
    finally:
        mask_generator = None
        object_detector = None
        _release_visual_resources()

    slide_data["original_image_path"] = str(owned_source)
    slide_data["_resource_isolation"] = resource_isolation
    state_path = _write_prepared_page(slide_data, owned_work_dir)
    return load_component_layers(state_path)


def finalize_component_layers(prepared: dict, accepted, *, lang: str) -> dict:
    if prepared.get("phase") != "initial_layers":
        raise ValueError("prepared layers must be in initial_layers phase")
    work_dir = _validate_prepared_work_dir(prepared["_work_dir"])
    if isinstance(accepted, dict):
        components = accepted.get("components", prepared["components"])
        element_mask_paths = accepted.get(
            "_element_mask_paths",
            accepted.get("element_mask_paths", prepared["_element_mask_paths"]),
        )
    elif isinstance(accepted, list):
        components = accepted
        element_mask_paths = prepared["_element_mask_paths"]
    elif accepted is None:
        components = prepared["components"]
        element_mask_paths = prepared["_element_mask_paths"]
    else:
        raise ValueError("accepted layers must be a dict, component list, or None")
    if not isinstance(components, list) or not isinstance(element_mask_paths, list):
        raise ValueError("accepted components and element masks must be lists")
    initial_count = prepared.get("initial_component_count")
    if initial_count and not components:
        raise VisualSegmentationError(
            "agent-managed quality failed: initial components cannot be empty"
        )

    original = _load_rgb(
        _prepared_owned_file(
            work_dir, prepared["original_image_path"], "source image"
        )
    )
    with Image.open(
        _prepared_owned_file(work_dir, prepared["_text_mask_path"], "OCR mask")
    ) as stored_text_mask:
        text_mask = np.asarray(stored_text_mask.convert("L")).copy()
    masks = []
    for mask_path in element_mask_paths:
        with Image.open(
            _prepared_owned_file(work_dir, mask_path, "element mask")
        ) as stored_mask:
            masks.append(np.asarray(stored_mask).copy())
    quality_text_items = prepared.get("text_items") or []
    quality_text_mask = _build_text_cleanup_mask(
        original,
        text_mask,
        quality_text_items,
    )
    if quality_text_items and _has_component_text_overlap(masks, quality_text_mask):
        raise VisualSegmentationError(
            "agent-managed quality failed: component_text_overlap"
        )

    staged_dir = Path(tempfile.mkdtemp(prefix="quality-components-", dir=work_dir))
    try:
        staged_components = []
        for index, component in enumerate(components):
            source_component = _prepared_owned_file(
                work_dir, component["path"], "component RGBA"
            )
            staged_path = staged_dir / f"component_{index:04d}.png"
            shutil.copyfile(source_component, staged_path)
            staged_components.append({
                **component,
                "path": str(staged_path.resolve()),
            })
        slide_data = {
            key: value
            for key, value in prepared.items()
            if key not in {"phase", "initial_component_count", "state_path"}
        }
        slide_data["components"] = staged_components
        slide_data["_element_mask_paths"] = [
            str(_prepared_owned_file(work_dir, path, "element mask"))
            for path in element_mask_paths
        ]
    except BaseException:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise
    exception_boundary = sys.exc_info()[1]
    primary_exception = None
    primary_traceback = None
    try:
        result = _finalize_slide_quality(
            slide_data,
            lang,
            _resource_isolation=bool(prepared.get("_resource_isolation")),
            _allow_text_only_fallback=False,
        )
    except BaseException as exc:
        primary_exception = exc
        primary_traceback = exc.__traceback__
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise
    finally:
        _run_cleanup_preserving_exception(
            close_ocr_engines,
            "OCR",
            primary_exception,
            primary_traceback,
            exception_boundary,
        )
    if initial_count and not result["components"]:
        raise VisualSegmentationError(
            "agent-managed quality failed: initial components became empty"
        )
    result.update({
        "phase": "quality_accepted",
        "initial_component_count": initial_count,
        "state_path": prepared.get("state_path"),
    })
    return result


def _resolve_image_path(image_path: str | Path) -> Path:
    resolved = Path(image_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Image not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Image path is not a file: {resolved}")
    if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {resolved.suffix.lower()}")
    try:
        with Image.open(resolved) as probe:
            probe.load()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode image: {resolved}") from exc
    return resolved


def _release_visual_resources() -> None:
    release_model()
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_exception_traceback_graph(
    exception: BaseException | None,
    boundary: BaseException | None,
) -> None:
    pending = [exception]
    visited = set()
    while pending:
        current = pending.pop()
        if current is None or current is boundary or id(current) in visited:
            continue
        visited.add(id(current))
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        pending.append(current.__cause__)
        pending.append(current.__context__)


def _run_cleanup_preserving_exception(
    cleanup,
    resource_name: str,
    primary_exception: BaseException | None,
    primary_traceback,
    exception_boundary: BaseException | None,
) -> None:
    if primary_traceback is not None:
        traceback.clear_frames(primary_traceback)
    _clear_exception_traceback_graph(primary_exception, exception_boundary)
    try:
        cleanup()
    except BaseException as cleanup_exception:
        if primary_exception is None:
            raise
        _clear_exception_traceback_graph(cleanup_exception, exception_boundary)
        logger.error("%s cleanup failed; preserving original exception", resource_name)


def _work_directory(root: str | Path | None, index: int) -> Path:
    if root is None:
        return Path(tempfile.mkdtemp(prefix=f"img2ppt_{index}_")).resolve()
    work_dir = Path(root).resolve() / f"page_{index + 1:03d}"
    work_dir.mkdir(parents=True, exist_ok=False)
    return work_dir


def _prepare_single_image(
    image_path: str | Path,
    lang: str,
    _work_root: str | Path | None = None,
    _resource_isolation: bool = False,
) -> tuple[dict, Path]:
    prepare_kwargs = {"_work_root": _work_root}
    if _resource_isolation:
        prepare_kwargs["_resource_isolation"] = True
    slide_data = _prepare_multiple_images(
        [image_path], lang, **prepare_kwargs
    )[0]
    return slide_data, Path(slide_data["background_original_path"]).parent


def _prepare_multiple_images(
    image_paths: list[str | Path],
    lang: str,
    _work_root: str | Path | None = None,
    _resource_isolation: bool = False,
) -> list[dict]:
    resolved_paths = [_resolve_image_path(path) for path in image_paths]
    if not resolved_paths:
        raise ValueError("No valid images provided")

    total = len(resolved_paths)
    print(f"Processing {total} image(s)...\n")
    prepared_pages = []
    for index, image_path in enumerate(resolved_paths):
        work_dir = _work_directory(_work_root, index)
        prepared_pages.append((image_path, work_dir))

    text_analyses = []
    text_mask = None
    exception_boundary = sys.exc_info()[1]
    primary_exception = None
    primary_traceback = None
    try:
        for image_path, work_dir in prepared_pages:
            if _resource_isolation:
                text_items, text_mask = detect_text(
                    image_path,
                    lang=lang,
                    isolated=True,
                    worker_root=work_dir,
                )
            else:
                text_items, text_mask = detect_text(
                    image_path,
                    lang=lang,
                )
            text_items, text_mask = _filter_probable_icon_text_analysis(
                text_items,
                text_mask,
            )
            text_mask_path = (work_dir / "source-text-mask.png").resolve()
            Image.fromarray(text_mask, mode="L").save(text_mask_path)
            text_analyses.append({
                "items": text_items,
                "mask_path": str(text_mask_path),
            })
            text_mask = None
    except BaseException as exc:
        primary_exception = exc
        primary_traceback = exc.__traceback__
        raise
    finally:
        text_mask = None
        _run_cleanup_preserving_exception(
            close_ocr_engines,
            "OCR",
            primary_exception,
            primary_traceback,
            exception_boundary,
        )

    if _resource_isolation:
        for (image_path, work_dir), text_analysis in zip(
            prepared_pages, text_analyses
        ):
            text_items = text_analysis["items"]
            if not text_items or not all(
                "box" in item for item in text_items
            ):
                continue
            source_image = _load_rgb(image_path)
            with Image.open(text_analysis["mask_path"]) as stored_text_mask:
                stored_mask = np.asarray(
                    stored_text_mask.convert("L")
                ).copy()
            removal_mask = _build_text_cleanup_mask(
                source_image,
                stored_mask,
                text_items,
            )
            removal_mask_path = (
                work_dir / "text-clean-removal-mask.png"
            ).resolve()
            Image.fromarray(removal_mask, mode="L").save(removal_mask_path)
            text_clean_path = (work_dir / "text-clean.png").resolve()
            text_clean = _repair_text_background(
                source_image,
                removal_mask,
                text_items=text_items,
                large_inpainter=_isolated_large_inpainter(work_dir),
            )
            _save_rgb(text_clean_path, text_clean)
            text_analysis["text_clean_path"] = str(text_clean_path)
            source_image = None
            stored_mask = None
            text_clean = None

    slides_data = []
    object_detector = None
    mask_generator = None
    exception_boundary = sys.exc_info()[1]
    primary_exception = None
    primary_traceback = None
    try:
        if _resource_isolation:
            mask_generator = None
        else:
            object_detector = create_object_detector()
            mask_generator = create_sam_generator(resolve_sam_checkpoint())
        for index, ((image_path, work_dir), text_analysis) in enumerate(
            zip(prepared_pages, text_analyses)
        ):
            print(f"=== Image {index + 1}/{total}: {image_path.name} ===")
            print(f"  Assets/diagnostics: {work_dir}")
            process_kwargs = {
                "text_analysis": text_analysis,
                "defer_quality": True,
            }
            if _resource_isolation:
                process_kwargs["_resource_isolation"] = True
            if _resource_isolation:
                slide_data = _process_image_isolated(
                    image_path,
                    work_dir,
                    lang,
                    text_analysis,
                )
            else:
                slide_data = _process_image(
                    image_path,
                    work_dir,
                    object_detector,
                    mask_generator,
                    lang,
                    **process_kwargs,
                )
            slides_data.append(slide_data)
            if _resource_isolation and index + 1 < total:
                _release_visual_resources()
    except BaseException as exc:
        primary_exception = exc
        primary_traceback = exc.__traceback__
        raise
    finally:
        mask_generator = None
        object_detector = None
        _run_cleanup_preserving_exception(
            _release_visual_resources,
            "visual resource",
            primary_exception,
            primary_traceback,
            exception_boundary,
        )

    exception_boundary = sys.exc_info()[1]
    primary_exception = None
    primary_traceback = None
    try:
        for index, slide_data in enumerate(slides_data):
            if _resource_isolation:
                slides_data[index] = _finalize_slide_quality(
                    slide_data,
                    lang,
                    _resource_isolation=True,
                )
            else:
                slides_data[index] = _finalize_slide_quality(
                    slide_data,
                    lang,
                )
            print(
                f"         {len(slides_data[index]['components'])} "
                "components extracted\n"
            )
    except BaseException as exc:
        primary_exception = exc
        primary_traceback = exc.__traceback__
        raise
    finally:
        _run_cleanup_preserving_exception(
            close_ocr_engines,
            "OCR",
            primary_exception,
            primary_traceback,
            exception_boundary,
        )
    return slides_data


def _variant_output_paths(
    image_path: str | Path,
    output_path: str | Path | None,
) -> tuple[Path, Path]:
    base = (
        Path(output_path).resolve()
        if output_path is not None
        else Path(image_path).resolve()
    ).with_suffix("")
    return (
        Path(f"{base}_original.pptx"),
        Path(f"{base}_16x9.pptx"),
    )


def _assemble_prepared_slide(
    slide_data: dict,
    output_path: str | Path,
    add_reference: bool,
    slide_size: str,
) -> str:
    background_key = (
        "background_original_path"
        if slide_size == "original"
        else "background_widescreen_path"
    )
    use_canvas = slide_size == "16:9"
    return assemble_pptx(
        background_path=slide_data[background_key],
        components=slide_data["components"],
        text_items=slide_data["text_items"],
        img_width=slide_data["img_width"],
        img_height=slide_data["img_height"],
        output_path=str(output_path),
        add_reference_slide=add_reference,
        original_image_path=slide_data["original_image_path"],
        slide_size=slide_size,
        canvas_width=slide_data.get("canvas_width") if use_canvas else None,
        canvas_height=slide_data.get("canvas_height") if use_canvas else None,
        content_offset_x=slide_data.get("content_offset_x", 0) if use_canvas else 0,
        content_offset_y=slide_data.get("content_offset_y", 0) if use_canvas else 0,
    )


def convert(
    image_path: str | Path,
    output_path: str | Path | None = None,
    lang: str = "ch",
    bg_period: int = 32,
    diff_threshold: float = 20.0,
    min_component_area: int = 20,
    add_reference: bool = False,
    slide_size: str = "16:9",
    _work_root: str | Path | None = None,
    _resource_isolation: bool = False,
) -> str:
    """Full pipeline: image → PPTX.

    Args:
        image_path: Path to input image.
        output_path: Where to save the PPTX. Auto-generated if None.
        lang: OCR language code.
        bg_period: Deprecated compatibility option; ignored by strict SAM pipeline.
        diff_threshold: Deprecated compatibility option; ignored by strict SAM pipeline.
        min_component_area: Deprecated compatibility option; ignored by strict SAM pipeline.
        add_reference: Add a reference slide with the original image.
        slide_size: Use the original image ratio or a 16:9 slide.

    Returns:
        Path to the output PPTX file.
    """
    if slide_size not in {"original", "16:9"}:
        raise ValueError("slide_size must be 'original' or '16:9'")

    if output_path is None:
        output_path = Path(image_path).resolve().with_suffix(".pptx")
    output_path = Path(output_path).resolve()

    prepare_kwargs = {"_work_root": _work_root}
    if _resource_isolation:
        prepare_kwargs["_resource_isolation"] = True
    slide_data, work_dir = _prepare_single_image(
        image_path, lang, **prepare_kwargs
    )
    print("[3/3] Assembling PPTX...")
    result = _assemble_prepared_slide(
        slide_data,
        output_path,
        add_reference,
        slide_size,
    )

    print(f"\nDone!")
    print(f"  Output: {result}")
    print(f"  Components: {len(slide_data['components'])}")
    print(f"  Text boxes: {len(slide_data['text_items'])}")
    print(f"  Assets: {work_dir}")

    return result


def convert_variants(
    image_path: str | Path,
    output_path: str | Path | None = None,
    lang: str = "ch",
    add_reference: bool = False,
    bg_period: int = 32,
    diff_threshold: float = 20.0,
    min_component_area: int = 20,
    _work_root: str | Path | None = None,
    _resource_isolation: bool = False,
) -> dict[str, str]:
    original_output, widescreen_output = _variant_output_paths(
        image_path,
        output_path,
    )
    prepare_kwargs = {"_work_root": _work_root}
    if _resource_isolation:
        prepare_kwargs["_resource_isolation"] = True
    slide_data, work_dir = _prepare_single_image(
        image_path, lang, **prepare_kwargs
    )
    print("[3/3] Assembling original and 16:9 PPTX files...")
    original_result = _assemble_prepared_slide(
        slide_data,
        original_output,
        add_reference,
        "original",
    )
    widescreen_result = _assemble_prepared_slide(
        slide_data,
        widescreen_output,
        add_reference,
        "16:9",
    )

    print("\nDone!")
    print(f"  Original: {original_result}")
    print(f"  16:9: {widescreen_result}")
    print(f"  Assets: {work_dir}")
    return {"original": original_result, "16:9": widescreen_result}


def convert_batch(
    image_paths: list[str | Path],
    output_path: str | Path | None = None,
    lang: str = "ch",
    bg_period: int = 32,
    diff_threshold: float = 20.0,
    min_component_area: int = 20,
    add_reference: bool = False,
    _work_root: str | Path | None = None,
    _resource_isolation: bool = False,
) -> str:
    """Process multiple images into a single multi-slide PPTX.

    Args:
        image_paths: List of paths to input images.
        output_path: Where to save the PPTX. Auto-generated if None.
        lang: OCR language code.
        bg_period: Deprecated compatibility option; ignored by strict SAM pipeline.
        diff_threshold: Deprecated compatibility option; ignored by strict SAM pipeline.
        min_component_area: Deprecated compatibility option; ignored by strict SAM pipeline.
        add_reference: Add reference slides with original images.

    Returns:
        Path to the output PPTX file.
    """
    prepare_kwargs = {"_work_root": _work_root}
    if _resource_isolation:
        prepare_kwargs["_resource_isolation"] = True
    slides_data = _prepare_multiple_images(
        image_paths, lang, **prepare_kwargs
    )
    if output_path is None:
        output_path = Path(slides_data[0]["original_image_path"]).with_suffix(".pptx")
    output_path = Path(output_path).resolve()
    total = len(slides_data)

    # Assemble all slides into one PPTX
    print(f"Assembling {total} slide(s) into PPTX...")
    result = assemble_pptx_multi(
        slides_data=slides_data,
        output_path=str(output_path),
        add_reference=add_reference,
    )

    print(f"\nDone!")
    print(f"  Output: {result}")
    print(f"  Total slides: {total}")

    return result


def convert_batch_variants(
    image_paths: list[str | Path],
    output_path: str | Path | None = None,
    lang: str = "ch",
    add_reference: bool = False,
    include_widescreen: bool = True,
    bg_period: int = 32,
    diff_threshold: float = 20.0,
    min_component_area: int = 20,
    combine_original: bool = False,
    original_aspect_ratio: float | None = None,
    _work_root: str | Path | None = None,
    _resource_isolation: bool = False,
) -> dict[str, str | list[str] | None]:
    prepare_kwargs = {"_work_root": _work_root}
    if _resource_isolation:
        prepare_kwargs["_resource_isolation"] = True
    slides_data = _prepare_multiple_images(
        image_paths, lang, **prepare_kwargs
    )
    source_paths = [
        Path(slide_data["original_image_path"]).resolve()
        for slide_data in slides_data
    ]
    base = (
        Path(output_path).resolve()
        if output_path is not None
        else source_paths[0]
    ).with_suffix("")
    widescreen_output = Path(f"{base}_16x9.pptx")
    original_dir = Path(f"{base}_original")

    widescreen_result = None
    if include_widescreen:
        widescreen_result = assemble_pptx_multi(
            slides_data=slides_data,
            output_path=str(widescreen_output),
            add_reference=add_reference,
        )

    if combine_original:
        original_result = assemble_pptx_multi(
            slides_data=slides_data,
            output_path=str(Path(f"{base}_original.pptx")),
            add_reference=add_reference,
            slide_size="original",
            original_aspect_ratio=original_aspect_ratio,
        )
        return {"16:9": widescreen_result, "original": original_result}

    original_results = []
    stem_totals: dict[str, int] = {}
    for source_path in source_paths:
        stem_key = source_path.stem.casefold()
        stem_totals[stem_key] = stem_totals.get(stem_key, 0) + 1
    reserved_stems = {
        stem_key for stem_key, count in stem_totals.items() if count == 1
    }
    next_suffix: dict[str, int] = {}
    used_stems: set[str] = set()
    for slide_data, source_path in zip(slides_data, source_paths):
        stem_key = source_path.stem.casefold()
        if stem_totals[stem_key] == 1:
            output_stem = source_path.stem
            output_key = stem_key
        else:
            suffix_number = next_suffix.get(stem_key, 1)
            while True:
                output_stem = (
                    source_path.stem
                    if suffix_number == 1
                    else f"{source_path.stem}_{suffix_number}"
                )
                output_key = output_stem.casefold()
                suffix_number += 1
                if output_key not in used_stems and output_key not in reserved_stems:
                    break
            next_suffix[stem_key] = suffix_number
        used_stems.add(output_key)
        original_output = (
            original_dir / f"{output_stem}_original.pptx"
        ).resolve()
        original_results.append(
            _assemble_prepared_slide(
                slide_data,
                original_output,
                add_reference,
                "original",
            )
        )

    return {"16:9": widescreen_result, "original": original_results}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_rgb(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_rgb(path: str, img: np.ndarray) -> None:
    """Save RGB numpy array as image."""
    from PIL import Image
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert image(s) to editable PowerPoint (background + foreground components + text)"
    )
    parser.add_argument(
        "images", nargs="+",
        help="Input image file(s) or directory containing images"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output PPTX path, or filename base when --slide-size=both"
    )
    parser.add_argument(
        "--lang", default="ch",
        help="OCR language (default: ch)"
    )
    parser.add_argument(
        "--period", type=int, default=32,
        help="Deprecated compatibility option; ignored by strict SAM pipeline"
    )
    parser.add_argument(
        "--diff-threshold", type=float, default=20.0,
        help="Deprecated compatibility option; ignored by strict SAM pipeline"
    )
    parser.add_argument(
        "--min-area", type=int, default=20,
        help="Deprecated compatibility option; ignored by strict SAM pipeline"
    )
    parser.add_argument(
        "--no-reference", action="store_true", default=False,
        help="Do not add reference slide with original image (default: no reference)"
    )
    parser.add_argument(
        "--reference", action="store_true", default=False,
        help="Add a reference slide with the original image"
    )
    parser.add_argument(
        "--slide-size",
        choices=("original", "16:9", "both"),
        default="both",
        help="Output slide size (default: both original ratio and 16:9)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    add_reference = _parse_reference_option(args.reference, args.no_reference)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Resolve input paths: expand directories into image files
    image_files = _resolve_inputs(args.images)

    if not image_files:
        print("Error: No valid image files found in the provided input(s).")
        sys.exit(1)

    if len(image_files) == 1 and args.slide_size == "both":
        convert_variants(
            image_path=image_files[0],
            output_path=args.output,
            lang=args.lang,
            add_reference=add_reference,
            bg_period=args.period,
            diff_threshold=args.diff_threshold,
            min_component_area=args.min_area,
            _resource_isolation=True,
        )
    elif len(image_files) == 1:
        convert(
            image_path=image_files[0],
            output_path=args.output,
            lang=args.lang,
            bg_period=args.period,
            diff_threshold=args.diff_threshold,
            min_component_area=args.min_area,
            add_reference=add_reference,
            slide_size=args.slide_size,
            _resource_isolation=True,
        )
    elif args.slide_size == "both":
        convert_batch_variants(
            image_paths=image_files,
            output_path=args.output,
            lang=args.lang,
            add_reference=add_reference,
            bg_period=args.period,
            diff_threshold=args.diff_threshold,
            min_component_area=args.min_area,
            _resource_isolation=True,
        )
    elif args.slide_size == "original":
        convert_batch_variants(
            image_paths=image_files,
            output_path=args.output,
            lang=args.lang,
            add_reference=add_reference,
            include_widescreen=False,
            bg_period=args.period,
            diff_threshold=args.diff_threshold,
            min_component_area=args.min_area,
            _resource_isolation=True,
        )
    else:
        convert_batch(
            image_paths=image_files,
            output_path=args.output,
            lang=args.lang,
            bg_period=args.period,
            diff_threshold=args.diff_threshold,
            min_component_area=args.min_area,
            add_reference=add_reference,
            _resource_isolation=True,
        )


def _parse_reference_option(reference: bool, no_reference: bool) -> bool:
    """Resolve reference-slide flags; explicit --no-reference wins."""
    if no_reference:
        return False
    return bool(reference)


def _merge_foreground_masks(
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> np.ndarray:
    """Restore reliable components lost during background refinement."""
    merged = refined_mask.copy()
    total_area = max(int(initial_mask.shape[0] * initial_mask.shape[1]), 1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (initial_mask > 0).astype(np.uint8), connectivity=8
    )

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        bbox_area = max(w * h, 1)
        bbox_ratio = bbox_area / total_area
        fill_ratio = area / bbox_area

        if bbox_ratio > 0.12 and fill_ratio < 0.30:
            continue

        component = labels == i
        if np.count_nonzero(refined_mask[component]) == 0:
            merged[component] = 255

    return merged


def _resolve_inputs(inputs: list[str]) -> list[Path]:
    """Expand input arguments into a flat list of image file paths.

    Handles both file paths and directory paths. Directories are scanned
    for image files (non-recursive). Results are sorted by filename.
    """
    image_files = []
    for item in inputs:
        p = Path(item).resolve()
        if p.is_dir():
            # Scan directory for image files
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append(f)
        elif p.is_file():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append(p)
            else:
                print(f"Warning: Skipping unsupported file: {p.name}")
        else:
            print(f"Warning: Path not found, skipping: {item}")
    return image_files


if __name__ == "__main__":
    main()
