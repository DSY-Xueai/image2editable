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
import io
import json
import logging
import math
import os
import shutil
import stat
import sys
import tempfile
import traceback
import unicodedata
from difflib import SequenceMatcher
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
    _remove_border_connected,
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
from scripts.text_detect import close_ocr_engines, detect_text, detect_text_batch
from scripts import text_detect as text_detection
from scripts.initial_diagnostics import (
    MAX_INITIAL_DIAGNOSTICS,
    validate_initial_diagnostics,
)
from scripts.visual_segment import (
    MaskCandidate,
    VisualSegmentationError,
    background_residual_metrics,
    combine_residual_candidates,
    create_sam_generator,
    filter_prompt_free_candidates,
    generate_geometry_candidates,
    generate_mask_candidates,
    generate_prompted_mask_candidates,
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
from scripts.worker_resources import run_isolated_worker
from scripts.sam_worker import (
    sam_candidate_batch_output_supported,
    sam_candidate_batch_max_automatic_candidates,
    sam_candidate_batch_max_prompted_candidates,
    sam_candidate_batch_max_proposals,
    sam_candidate_batch_result_max_bytes,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

_TARGETED_OCR_VIEW_SCALES = (2.0, 3.0)
_TARGETED_OCR_VIEW_EDGE_LIMITS = (512, 448)
_TARGETED_OCR_MIN_CONFIDENCE = 0.88
_TARGETED_OCR_MAX_CANDIDATES = 24
_TARGETED_OCR_SINGLE_CROP_PIXELS = 512 * 512
_TARGETED_OCR_TOTAL_CROP_PIXELS = 6 * 1024 * 1024
_TARGETED_OCR_MAX_ITEMS_PER_VIEW = 32
_TEXT_DELTA_CACHE_SCHEMA_VERSION = 1
_TEXT_DELTA_MAX_NODES = 4096
_TEXT_DELTA_CACHE_NAME = "first-visual-cache.json"
_TEXT_DELTA_CACHE_MAX_BYTES = 8 * 1024 * 1024
_TEXT_DELTA_MAX_MASK_CROP_PIXELS = 128 * 1024 * 1024
_TEXT_DELTA_MAX_PAIRWISE_CANDIDATES = 100_000
_TEXT_DELTA_MAX_PAIRWISE_PIXELS = 128 * 1024 * 1024
_TEXT_DELTA_SAM_PROTOCOL_SHA256 = hashlib.sha256(
    b"sam2.1_hiera_large|candidate_batch_v1|visual_pipeline_v1"
).hexdigest()
_TEXT_DELTA_DINO_PROTOCOL_SHA256 = hashlib.sha256(
    b"IDEA-Research/grounding-dino-tiny|visual_pipeline_v1"
).hexdigest()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _normalized_candidate_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(normalized.split())


def _box_intersection_ratio(left: list[int], right: list[int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    return intersection / max(1, min(lw * lh, rw * rh))


def _matches_known_text(item: dict, known_items: list[dict]) -> bool:
    for known in known_items:
        known_box = [int(value) for value in known.get("box", [0, 0, 0, 0])]
        item_box = [int(value) for value in item["box"]]
        overlap = _box_intersection_ratio(
            known_box,
            item_box,
        )
        known_text = _normalized_candidate_text(known.get("text", ""))
        item_text = item["normalized_text"]
        same_text = known_text == item_text
        contained_text = (
            min(len(known_text), len(item_text)) >= 4
            and (known_text in item_text or item_text in known_text)
        )
        _, _, known_width, known_height = known_box
        _, _, item_width, item_height = item_box
        width_ratio = min(known_width, item_width) / max(1, known_width, item_width)
        height_ratio = min(known_height, item_height) / max(1, known_height, item_height)
        known_center = (
            known_box[0] + known_width / 2,
            known_box[1] + known_height / 2,
        )
        item_center = (
            item_box[0] + item_width / 2,
            item_box[1] + item_height / 2,
        )
        similar_geometry = (
            width_ratio >= 0.75
            and height_ratio >= 0.70
            and abs(known_center[0] - item_center[0]) <= max(known_width, item_width) * 0.20
            and abs(known_center[1] - item_center[1]) <= max(known_height, item_height) * 0.25
        )
        text_similarity = SequenceMatcher(None, known_text, item_text).ratio()
        length_ratio = min(len(known_text), len(item_text)) / max(
            1, len(known_text), len(item_text)
        )
        similar_text = similar_geometry and (
            text_similarity >= 0.88
            or (
                max(len(known_text), len(item_text)) >= 20
                and length_ratio <= 0.75
                and text_similarity >= 0.50
            )
        )
        if overlap >= 0.80 and (same_text or contained_text or similar_text):
            return True
        if overlap >= 0.50 and same_text:
            return True
    return False


def _deduplicate_overlapping_text_items(items: list[dict]) -> list[dict]:
    """Keep the most complete OCR reading for the same spatial text line."""
    kept: list[dict] = []
    for item in items:
        text = _normalized_candidate_text(item.get("text", ""))
        box = item.get("box")
        if not text or not isinstance(box, (list, tuple)) or len(box) != 4:
            kept.append(item)
            continue
        duplicate_index = None
        for index, existing in enumerate(kept):
            existing_text = _normalized_candidate_text(existing.get("text", ""))
            existing_box = existing.get("box")
            if (
                not existing_text
                or not isinstance(existing_box, (list, tuple))
                or len(existing_box) != 4
            ):
                continue
            overlap = _box_intersection_ratio(
                [int(value) for value in existing_box],
                [int(value) for value in box],
            )
            if overlap >= 0.80 and (
                text in existing_text or existing_text in text
            ):
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(item)
            continue
        existing = kept[duplicate_index]
        existing_text = _normalized_candidate_text(existing.get("text", ""))
        if (
            len(text), float(item.get("confidence", 0.0))
        ) > (
            len(existing_text), float(existing.get("confidence", 0.0))
        ):
            kept[duplicate_index] = item
    return kept


def _targeted_candidate_ocr_sweep(
    image_path: str | Path,
    components: list[dict],
    known_items: list[dict],
    known_mask: np.ndarray,
    work_dir: str | Path,
    *,
    lang: str,
    isolated: bool,
) -> dict:
    """Recheck bounded visual candidates without retaining page-size copies."""
    source_path = Path(image_path).resolve()
    work_dir = Path(work_dir).resolve()
    mask = np.asarray(known_mask, dtype=np.uint8)
    with Image.open(source_path) as source:
        page_width, page_height = source.size
    if mask.shape != (page_height, page_width):
        raise ValueError("targeted OCR text mask must match the source image")

    page_pixels = page_width * page_height
    candidate_limit = min(
        _TARGETED_OCR_MAX_CANDIDATES,
        max(16, page_pixels // 120_000),
    )
    total_pixel_limit = min(
        _TARGETED_OCR_TOTAL_CROP_PIXELS,
        max(_TARGETED_OCR_SINGLE_CROP_PIXELS, page_pixels * 2),
    )
    selected = []
    for index, component in enumerate(components, start=1):
        try:
            x, y, width, height = (
                int(component[name]) for name in ("x", "y", "w", "h")
            )
            alpha_area = int(component.get("area", width * height))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            x < 0
            or y < 0
            or width < 8
            or height < 8
            or x + width > page_width
            or y + height > page_height
            or (
                page_pixels >= 200_000
                and width * height > page_pixels * 0.10
            )
            or width * height > page_pixels * 0.25
            or max(width / height, height / width) > 14
            or alpha_area / max(width * height, 1) < 0.04
        ):
            continue
        selected.append((index, [x, y, width, height]))
        if len(selected) >= candidate_limit:
            break

    diagnostics = []
    consistent = []
    used_pixels = 0
    with source_path.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    exception_boundary = sys.exc_info()[1]
    primary_exception = None
    primary_traceback = None
    try:
        with tempfile.TemporaryDirectory(prefix="targeted-ocr-", dir=work_dir) as temporary:
            crop_root = Path(temporary)
            pending_views = []
            with Image.open(source_path) as source:
                for component_index, box in selected:
                    x, y, width, height = box
                    views = []
                    candidate_pixels = []
                    for target_scale, edge_limit in zip(
                        _TARGETED_OCR_VIEW_SCALES,
                        _TARGETED_OCR_VIEW_EDGE_LIMITS,
                    ):
                        bounded_scale = min(
                            target_scale,
                            edge_limit / width,
                            edge_limit / height,
                            math.sqrt(
                                _TARGETED_OCR_SINGLE_CROP_PIXELS
                                / max(width * height, 1)
                            ),
                        )
                        view_width = max(1, int(round(width * bounded_scale)))
                        view_height = max(1, int(round(height * bounded_scale)))
                        candidate_pixels.append(view_width * view_height)
                        views.append((bounded_scale, view_width, view_height))
                    if used_pixels + sum(candidate_pixels) > total_pixel_limit:
                        break

                    with source.crop((x, y, x + width, y + height)) as raw_crop:
                        with raw_crop.convert("RGB") as base_crop:
                            for view_index, (scale, view_width, view_height) in enumerate(views):
                                crop_path = crop_root / (
                                    f"candidate-{component_index:04d}-view-{view_index + 1}.png"
                                )
                                with base_crop.resize(
                                    (view_width, view_height), Image.Resampling.LANCZOS
                                ) as resized:
                                    resized.save(crop_path)
                                used_pixels += view_width * view_height
                                pending_views.append({
                                    "component_index": component_index,
                                    "component_box": box,
                                    "scale": scale,
                                    "path": crop_path,
                                })

            view_results = detect_text_batch(
                [view["path"] for view in pending_views],
                lang=lang,
                confidence_threshold=0.70,
                isolated=isolated,
                worker_root=work_dir if isolated else None,
            )
            recognized_by_component = {}
            for view, (items, _) in zip(pending_views, view_results):
                x, y, _, _ = view["component_box"]
                scale = view["scale"]
                mapped_items = []
                for item in text_detection._merge_adjacent_text_items(items):
                    raw_box = item.get("box")
                    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                        continue
                    mapped_box = [
                        max(0, int(round(x + raw_box[0] / scale))),
                        max(0, int(round(y + raw_box[1] / scale))),
                        max(1, int(round(raw_box[2] / scale))),
                        max(1, int(round(raw_box[3] / scale))),
                    ]
                    mapped_box[2] = min(mapped_box[2], page_width - mapped_box[0])
                    mapped_box[3] = min(mapped_box[3], page_height - mapped_box[1])
                    normalized = _normalized_candidate_text(item.get("text", ""))
                    confidence = float(item.get("confidence", 0.0))
                    if (
                        normalized
                        and confidence >= _TARGETED_OCR_MIN_CONFIDENCE
                        and _box_intersection_ratio(
                            mapped_box, view["component_box"],
                        ) >= 0.50
                    ):
                        mapped_items.append({
                            "text": str(item.get("text", "")).strip(),
                            "normalized_text": normalized,
                            "confidence": confidence,
                            "box": mapped_box,
                        })
                recognized_by_component.setdefault(
                    view["component_index"], []
                ).append(sorted(
                    mapped_items,
                    key=lambda value: (value["box"][1], value["box"][0]),
                )[:_TARGETED_OCR_MAX_ITEMS_PER_VIEW])

            for component_index, recognized in recognized_by_component.items():
                if len(recognized) != 2:
                    continue
                unmatched = set(range(len(recognized[1])))
                pairs = []
                for left in recognized[0]:
                    choices = [
                        (index, _box_intersection_ratio(
                            left["box"], recognized[1][index]["box"],
                        ))
                        for index in unmatched
                    ]
                    choices = [choice for choice in choices if choice[1] >= 0.50]
                    if not choices:
                        continue
                    right_index = max(choices, key=lambda choice: choice[1])[0]
                    unmatched.remove(right_index)
                    pairs.append((left, recognized[1][right_index]))
                pairs.sort(key=lambda pair: (
                    min(pair[0]["box"][1], pair[1]["box"][1]),
                    min(pair[0]["box"][0], pair[1]["box"][0]),
                ))
                for pair_index, (left, right) in enumerate(pairs, start=1):
                    if _matches_known_text(left, known_items) or _matches_known_text(
                        right, known_items
                    ):
                        continue
                    if left["normalized_text"] == right["normalized_text"]:
                        consistent.append(max(
                            (left, right), key=lambda item: item["confidence"]
                        ))
                        continue
                    left_box, right_box = left["box"], right["box"]
                    if len(diagnostics) >= MAX_INITIAL_DIAGNOSTICS:
                        continue
                    diagnostics.append({
                        "kind": "unowned_raster_text",
                        "source_sha256": source_sha256,
                        "candidate_id": (
                            f"candidate_{component_index:04d}_{pair_index:02d}"
                        ),
                        "bbox": [
                            min(left_box[0], right_box[0]),
                            min(left_box[1], right_box[1]),
                            max(left_box[0] + left_box[2], right_box[0] + right_box[2]),
                            max(left_box[1] + left_box[3], right_box[1] + right_box[3]),
                        ],
                        "views": [
                            {"normalized_text": left["normalized_text"],
                             "confidence": left["confidence"]},
                            {"normalized_text": right["normalized_text"],
                             "confidence": right["confidence"]},
                        ],
                    })
    except BaseException as exc:
        primary_exception = exc
        primary_traceback = exc.__traceback__
        raise
    finally:
        _run_cleanup_preserving_exception(
            close_ocr_engines,
            "targeted OCR",
            primary_exception,
            primary_traceback,
            exception_boundary,
        )

    recovered = []
    if consistent:
        with Image.open(source_path) as source:
            for item in consistent:
                x, y, width, height = item["box"]
                left, top = max(0, x - 6), max(0, y - 6)
                right = min(page_width, x + width + 6)
                bottom = min(page_height, y + height + 6)
                with source.crop((left, top, right, bottom)).convert("RGB") as crop:
                    pixels = np.asarray(crop).copy()
                local_box = (x - left, y - top, width, height)
                style = text_detection._estimate_style(
                    pixels,
                    local_box,
                    reference_width=page_width,
                )
                font_size = text_detection._adjust_font_size(
                    item["text"], style["font_size"]
                )
                recovered.append({
                "box": item["box"],
                "text": item["text"],
                "font_size": font_size,
                "color": style["color"],
                "bold": (
                    False
                    if text_detection._should_force_regular_weight(
                        item["text"], font_size
                    )
                    else style["bold"]
                ),
                "font": text_detection._select_font(item["text"], font_size),
                "align": 1,
                "confidence": item["confidence"],
                })
    all_items = _deduplicate_overlapping_text_items(
        text_detection._merge_adjacent_text_items(
        [dict(item) for item in known_items] + recovered
        )
    )
    all_items = text_detection._refine_alignment(all_items, page_width)
    updated_mask = text_detection._build_text_mask(
        (page_height, page_width), all_items, padding=6
    )
    return {
        "items": all_items,
        "recovered_items": recovered,
        "text_mask": updated_mask,
        "diagnostics": diagnostics,
        "resource_stats": {
            "candidate_limit": candidate_limit,
            "selected_candidates": len(selected),
            "single_crop_pixel_limit": _TARGETED_OCR_SINGLE_CROP_PIXELS,
            "total_crop_pixel_limit": total_pixel_limit,
            "processed_crop_pixels": used_pixels,
        },
    }


def _remove_owned_first_visual_assets(slide_data: dict, work_dir: Path) -> None:
    owned_root = Path(os.path.abspath(work_dir))
    root_identity = owned_root.lstat()
    if _is_link_or_reparse(root_identity) or not stat.S_ISDIR(root_identity.st_mode):
        raise ValueError("first visual work directory identity is unsafe")
    groups = (
        ("components", [component.get("path") for component in slide_data.get("components", [])]),
        ("element-masks", slide_data.get("_element_mask_paths", [])),
        ("semantic-masks", slide_data.get("_semantic_mask_paths", [])),
    )
    validated = []
    for directory, paths in groups:
        if not paths:
            continue
        owned_directory = owned_root / directory
        directory_status = owned_directory.lstat()
        if (_is_link_or_reparse(directory_status)
                or not stat.S_ISDIR(directory_status.st_mode)):
            raise ValueError("first visual owned directory identity is unsafe")
        for value in paths:
            if not value:
                continue
            path = Path(os.path.abspath(value))
            try:
                path.relative_to(owned_root)
            except ValueError:
                raise ValueError("first visual asset is outside the work directory")
            if path.parent != owned_directory:
                raise ValueError("first visual asset is outside its owned directory")
            before = path.lstat()
            if (_is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1):
                raise ValueError("first visual asset identity is unsafe")
            after = path.lstat()
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise RuntimeError("first visual asset identity changed")
            current_directory = owned_directory.lstat()
            if (
                _is_link_or_reparse(current_directory)
                or not stat.S_ISDIR(current_directory.st_mode)
                or (directory_status.st_dev, directory_status.st_ino)
                != (current_directory.st_dev, current_directory.st_ino)
            ):
                raise RuntimeError("first visual owned directory identity changed")
            validated.append((path, before, owned_directory, directory_status))
    for path, expected, owned_directory, expected_directory in validated:
        current_directory = owned_directory.lstat()
        current = path.lstat()
        if (
            _is_link_or_reparse(current_directory)
            or not stat.S_ISDIR(current_directory.st_mode)
            or (expected_directory.st_dev, expected_directory.st_ino)
            != (current_directory.st_dev, current_directory.st_ino)
            or _is_link_or_reparse(current)
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or (expected.st_dev, expected.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RuntimeError("first visual asset identity changed")
        path.unlink()


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
    deduplicated = _deduplicate_overlapping_text_items(filtered)
    if len(filtered) == len(items):
        return deduplicated, detected.copy()

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
    for item in deduplicated:
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x, y, box_width, box_height = (int(value) for value in box)
        y1, y2 = max(0, y), min(height, y + box_height)
        x1, x2 = max(0, x), min(width, x + box_width)
        result[y1:y2, x1:x2] = detected[y1:y2, x1:x2]
    return deduplicated, result


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
        border_pixels = np.concatenate(
            (region[0], region[-1], region[:, 0], region[:, -1]),
            axis=0,
        )
        background = np.median(border_pixels, axis=0)
        target_distance = float(np.linalg.norm(target - background))
        color_axis = target - background
        axis_length = float(np.dot(color_axis, color_axis))
        if axis_length > 0:
            opacity = np.sum(
                (region - background) * color_axis,
                axis=2,
            ) / axis_length
            expected = background + np.clip(opacity, 0.0, 1.0)[..., None] * color_axis
            corridor_error = np.linalg.norm(region - expected, axis=2)
            corridor_tolerance = min(28.0, max(12.0, target_distance * 0.08))
            matching = (
                (opacity >= 0.05)
                & (opacity <= 1.15)
                & (corridor_error <= corridor_tolerance)
            ).astype(np.uint8)
        else:
            matching = (
                np.linalg.norm(region - target, axis=2) <= 18.0
            ).astype(np.uint8)
        gray_region = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_threshold, _ = cv2.threshold(
            gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        target_gray = float(np.mean(target))
        background_gray = float(np.mean(background))
        secondary = (
            gray_region <= gray_threshold
            if target_gray < background_gray
            else gray_region > gray_threshold
        )
        matching |= _remove_border_connected(secondary).astype(np.uint8)
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
            radius = max(1, min(3, int(round(font_size * 0.08))))
        else:
            radius = max(1, min(2, int(round(max(box_height, 1) * 0.05))))
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


def _persist_visual_masks(
    work_dir: Path,
    directory_name: str,
    masks: list[np.ndarray],
) -> list[str]:
    masks_dir = (work_dir / directory_name).resolve()
    masks_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, mask in enumerate(masks):
        mask_path = (masks_dir / f"{index:04d}.png").resolve()
        Image.fromarray(
            (np.asarray(mask) > 0).astype(np.uint8) * 255,
            mode="L",
        ).save(mask_path)
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


def _finalize_slide_quality(
    slide_data: dict,
    lang: str,
    _resource_isolation: bool = False,
) -> dict:
    slide_data = slide_data.copy()
    slide_data.pop("_prepared_schema_version", None)
    slide_data.pop("_semantic_mask_paths", None)
    slide_data.pop("_foreground_evidence_mask_path", None)
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

        quality_text_items = slide_data.get("text_items") or []
        quality_text_mask = _build_text_cleanup_mask(
            img,
            text_mask,
            quality_text_items,
        )
        forced_fallback_reason = None
        has_component_text_overlap = quality_text_items and _has_component_text_overlap(
            element_masks,
            quality_text_mask,
        )
        overlap_reports = _component_text_overlap_reports(
            element_masks,
            quality_text_mask,
        ) if has_component_text_overlap else []
        if has_component_text_overlap:
            if not overlap_reports:
                overlap_reports = [{
                    "component_id": f"component_{index:04d}",
                    "accepted": False,
                    "metrics": {},
                    "violations": ["component_text_overlap"],
                } for index in range(1, len(element_masks) + 1)]
            slide_data["component_quality_reports"] = overlap_reports
            failed_ids = ",".join(report["component_id"] for report in overlap_reports)
            raise VisualSegmentationError(
                f"component quality failed: component_text_overlap:{failed_ids}"
            )

        components = slide_data["components"]
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
            raise VisualSegmentationError(
                f"component/page quality failed: {fallback_reason}"
            )
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
    return bool(_component_text_overlap_reports(element_masks, text_mask))


def _component_text_overlap_reports(
    element_masks: list[np.ndarray],
    text_mask: np.ndarray,
) -> list[dict]:
    text = np.asarray(text_mask) > 0
    if not np.any(text):
        return []
    reports = []
    for index, element_mask in enumerate(element_masks, start=1):
        component = np.asarray(element_mask) > 0
        component_pixels = int(np.count_nonzero(component))
        if component_pixels == 0:
            continue
        overlap = int(np.count_nonzero(component & text))
        if overlap >= 16 and overlap / component_pixels >= 0.02:
            reports.append({
                "component_id": f"component_{index:04d}",
                "accepted": False,
                "metrics": {
                    "component_pixels": component_pixels,
                    "text_duplicate_pixels": overlap,
                    "text_duplicate_ratio": overlap / component_pixels,
                },
                "violations": ["component_text_overlap"],
            })
    return reports


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
        completed = run_isolated_worker(
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
        completed = run_isolated_worker(
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


_SAM_CANDIDATE_BATCH_SCHEMA_VERSION = 1
_SAM_CANDIDATE_BATCH_MAX_STRING_LENGTH = 256
_SAM_CANDIDATE_BATCH_FIELDS = {
    "mask",
    "mask_shape",
    "score",
    "source",
    "crop_box",
    "touches_crop_edge",
    "label",
    "role",
    "object_box",
}


def _validate_sam_candidate_batch_string(
    value,
    label: str,
    *,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value)
        or len(value) > _SAM_CANDIDATE_BATCH_MAX_STRING_LENGTH
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise RuntimeError(f"SAM candidate batch returned an invalid {label}")
    return value


def _validate_sam_candidate_batch_finite_number(value, label: str):
    if type(value) is int:
        return value
    try:
        finite = math.isfinite(value)
    except (TypeError, OverflowError):
        finite = False
    if type(value) is not float or not finite:
        raise RuntimeError(f"SAM candidate batch returned an invalid {label}")
    return value


def _validate_sam_candidate_batch_crop_box(
    value,
    label: str,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(coordinate) is not int for coordinate in value)
    ):
        raise RuntimeError(f"SAM candidate batch returned an invalid {label}")
    x1, y1, x2, y2 = value
    height, width = image_shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RuntimeError(f"SAM candidate batch returned an invalid {label}")
    return tuple(value)


def _validate_sam_candidate_batch_intersecting_box(
    value,
    label: str,
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise RuntimeError(f"SAM candidate batch returned an invalid {label}")
    coordinates = tuple(
        _validate_sam_candidate_batch_finite_number(coordinate, label)
        for coordinate in value
    )
    x1, y1, x2, y2 = coordinates
    height, width = image_shape
    if not (
        x1 < x2
        and y1 < y2
        and x1 < width
        and y1 < height
        and x2 > 0
        and y2 > 0
    ):
        raise RuntimeError(f"SAM candidate batch returned an invalid {label}")
    return coordinates


def _validate_sam_candidate_batch_score(value):
    return _validate_sam_candidate_batch_finite_number(value, "score")


def _validate_sam_candidate_batch_records(
    records,
    image_shape: tuple[int, int],
    candidate_limit: int,
) -> list[tuple[bytes, dict]]:
    if not isinstance(records, list):
        raise RuntimeError("SAM candidate batch records must be a list")
    if len(records) > candidate_limit:
        raise RuntimeError("SAM candidate batch returned too many candidates")
    validated = []
    expected_bytes = (int(np.prod(image_shape)) + 7) // 8
    expected_base64_length = ((expected_bytes + 2) // 3) * 4
    for record in records:
        if not isinstance(record, dict) or set(record) != _SAM_CANDIDATE_BATCH_FIELDS:
            raise RuntimeError("SAM candidate batch returned an invalid candidate")
        mask_shape = record["mask_shape"]
        if (
            not isinstance(mask_shape, list)
            or len(mask_shape) != 2
            or any(type(value) is not int for value in mask_shape)
            or tuple(mask_shape) != image_shape
        ):
            raise RuntimeError("SAM candidate batch returned the wrong mask shape")
        if (
            not isinstance(record["mask"], str)
            or len(record["mask"]) != expected_base64_length
        ):
            raise RuntimeError("SAM candidate batch returned an invalid mask length")
        try:
            packed_bytes = base64.b64decode(record["mask"], validate=True)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("SAM candidate batch returned an invalid mask") from exc
        if len(packed_bytes) != expected_bytes:
            raise RuntimeError("SAM candidate batch returned an invalid mask length")
        if type(record["touches_crop_edge"]) is not bool:
            raise RuntimeError(
                "SAM candidate batch returned an invalid crop-edge flag"
            )
        candidate_fields = {
            "score": _validate_sam_candidate_batch_score(record["score"]),
            "source": _validate_sam_candidate_batch_string(
                record["source"],
                "source",
            ),
            "crop_box": _validate_sam_candidate_batch_crop_box(
                record["crop_box"],
                "crop box",
                image_shape,
            ),
            "touches_crop_edge": record["touches_crop_edge"],
            "label": _validate_sam_candidate_batch_string(
                record["label"],
                "label",
                allow_empty=True,
            ),
            "role": _validate_sam_candidate_batch_string(
                record["role"],
                "role",
                allow_empty=True,
            ),
            "object_box": _validate_sam_candidate_batch_intersecting_box(
                record["object_box"],
                "object box",
                image_shape,
            ),
        }
        validated.append((packed_bytes, candidate_fields))
    return validated


def _decode_sam_candidate_batch_records(
    records: list[tuple[bytes, dict]],
    image_shape: tuple[int, int],
) -> list[MaskCandidate]:
    candidates = []
    for packed_bytes, candidate_fields in records:
        mask = np.unpackbits(
            np.frombuffer(packed_bytes, dtype=np.uint8),
            count=int(np.prod(image_shape)),
        ).reshape(image_shape).astype(bool, copy=False)
        candidates.append(MaskCandidate(mask=mask, **candidate_fields))
    return candidates


def _read_sam_candidate_batch_result(path: Path, limit: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise RuntimeError(
            "Isolated SAM candidate batch did not create its result"
        ) from exc
    if (
        _is_link_or_reparse(before)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > limit
    ):
        raise RuntimeError("SAM candidate batch result is unsafe or too large")
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size != before.st_size
        ):
            raise RuntimeError("SAM candidate batch result identity changed")
        chunks = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(
                    1024 * 1024,
                    limit + 1 - total,
                ),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise RuntimeError("SAM candidate batch result is too large")
        stable = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identities = {
        (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
        for status in (before, opened, stable, after)
    }
    if (
        len(identities) != 1
        or _is_link_or_reparse(after)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
    ):
        raise RuntimeError("SAM candidate batch result changed while reading")
    return b"".join(chunks)


def _reject_sam_candidate_batch_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _generate_sam_candidate_batch_isolated(
    image: np.ndarray,
    text_mask: np.ndarray,
    proposals: list[ObjectProposal],
    work_dir: Path,
) -> tuple[list[MaskCandidate], list[MaskCandidate]]:
    image_shape = tuple(image.shape[:2])
    maximum_proposals = sam_candidate_batch_max_proposals(image_shape)
    if len(proposals) > maximum_proposals:
        raise RuntimeError("SAM candidate batch has too many proposals")
    result_limit = sam_candidate_batch_result_max_bytes(
        image_shape,
        len(proposals),
    )
    with tempfile.TemporaryDirectory(
        prefix="sam-batch-",
        dir=work_dir,
    ) as temporary_dir:
        temporary_dir = Path(temporary_dir)
        image_path = temporary_dir / "image.png"
        text_mask_path = temporary_dir / "text-mask.png"
        proposals_path = temporary_dir / "proposals.json"
        request_path = temporary_dir / "request.json"
        result_path = temporary_dir / "result.json"
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(image_path)
        Image.fromarray(
            np.asarray(text_mask, dtype=np.uint8),
            mode="L",
        ).save(text_mask_path)
        proposals_path.write_text(
            json.dumps(
                [proposal.__dict__ for proposal in proposals],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        request_path.write_text(
            json.dumps(
                {
                    "schema_version": _SAM_CANDIDATE_BATCH_SCHEMA_VERSION,
                    "operations": [
                        {
                            "id": "prompted",
                            "kind": "prompted",
                            "image": image_path.name,
                            "text_mask": text_mask_path.name,
                            "proposals": proposals_path.name,
                        },
                        {
                            "id": "automatic",
                            "kind": "automatic",
                            "image": image_path.name,
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        module_dir = Path(__file__).resolve().parent
        worker_path = module_dir / "scripts" / "sam_worker.py"
        if not worker_path.is_file():
            worker_path = module_dir / "sam_worker.py"
        completed = run_isolated_worker(
            [
                sys.executable,
                str(worker_path),
                "--mode",
                "batch",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            result_path.unlink(missing_ok=True)
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Isolated SAM candidate batch failed: {detail}")
        try:
            payload = json.loads(
                _read_sam_candidate_batch_result(
                    result_path,
                    result_limit,
                ).decode("utf-8"),
                parse_constant=_reject_sam_candidate_batch_json_constant,
            )
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version",
                "operations",
            }:
                raise RuntimeError("SAM candidate batch returned an invalid result")
            if (
                type(payload["schema_version"]) is not int
                or payload["schema_version"] != _SAM_CANDIDATE_BATCH_SCHEMA_VERSION
            ):
                raise RuntimeError(
                    "SAM candidate batch returned an invalid schema version"
                )
            output_operations = payload["operations"]
            expected_operations = [
                ("prompted", "prompted"),
                ("automatic", "automatic"),
            ]
            if not isinstance(output_operations, list) or len(output_operations) != len(
                expected_operations
            ):
                raise RuntimeError(
                    "SAM candidate batch returned the wrong operation count"
                )
            candidate_groups = []
            for output, expected in zip(output_operations, expected_operations):
                if not isinstance(output, dict) or set(output) != {
                    "id",
                    "kind",
                    "candidates",
                }:
                    raise RuntimeError(
                        "SAM candidate batch returned an invalid operation"
                    )
                if (output["id"], output["kind"]) != expected:
                    raise RuntimeError(
                        "SAM candidate batch returned operations out of order"
                    )
                records = output["candidates"]
                candidate_limit = (
                    sam_candidate_batch_max_prompted_candidates(len(proposals))
                    if expected[1] == "prompted"
                    else sam_candidate_batch_max_automatic_candidates()
                )
                if not isinstance(records, list) or len(records) > candidate_limit:
                    raise RuntimeError(
                        "SAM candidate batch returned an invalid candidate count"
                    )
                candidate_groups.append((records, candidate_limit))
            validated = [
                _validate_sam_candidate_batch_records(
                    records,
                    image_shape,
                    candidate_limit,
                )
                for records, candidate_limit in candidate_groups
            ]
            decoded = [
                _decode_sam_candidate_batch_records(
                    records,
                    image_shape,
                )
                for records in validated
            ]
        except RuntimeError:
            result_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            result_path.unlink(missing_ok=True)
            raise RuntimeError(
                "SAM candidate batch returned an invalid result"
            ) from exc
    return decoded[0], decoded[1]


def _generate_sam_candidate_stage_isolated(
    image: np.ndarray,
    text_mask: np.ndarray,
    proposals: list[ObjectProposal],
    work_dir: Path,
) -> tuple[list[MaskCandidate], list[MaskCandidate]]:
    if sam_candidate_batch_output_supported(work_dir):
        return _generate_sam_candidate_batch_isolated(
            image,
            text_mask,
            proposals,
            work_dir,
        )
    prompted = _generate_sam_candidates_isolated(
        image,
        text_mask,
        proposals,
        work_dir,
        mode="prompted",
    )
    automatic = _generate_sam_candidates_isolated(
        image,
        None,
        None,
        work_dir,
        mode="automatic",
    )
    return prompted, automatic


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
        completed = run_isolated_worker(
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
    _, source_content = _read_prepared_owned_bytes(
        work_dir,
        image_path,
        "isolated visual source",
    )
    source_sha256 = hashlib.sha256(source_content).hexdigest()
    request_path = (work_dir / "visual-worker-request.json").resolve()
    result_path = (work_dir / "visual-worker-result.json").resolve()
    request_path.write_text(
        json.dumps({
            "text_analysis": text_analysis,
            "source_sha256": source_sha256,
            "source_size": len(source_content),
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    module_dir = Path(__file__).resolve().parent
    worker_path = module_dir / "scripts" / "visual_worker.py"
    if not worker_path.is_file():
        worker_path = module_dir / "visual_worker.py"
    completed = run_isolated_worker(
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
    slide_data = json.loads(result_path.read_text(encoding="utf-8"))
    slide_data["_visual_source_sha256"] = source_sha256
    return slide_data


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
    _source_image: np.ndarray | None = None,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    background_kwargs = (
        {"large_inpainter": _isolated_large_inpainter(work_dir)}
        if _resource_isolation
        else {}
    )
    img = (
        np.asarray(_source_image, dtype=np.uint8).copy()
        if _source_image is not None
        else _load_rgb(str(image_path))
    )
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
        (
            candidates,
            prompt_free_candidates,
        ) = _generate_sam_candidate_stage_isolated(
            img,
            text_ink_mask,
            proposals,
            work_dir,
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
            text_clean_image=text_clean_image,
            text_restore_mask=text_mask,
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
            (
                prompted_residual_candidates,
                prompt_free_residual_candidates,
            ) = _generate_sam_candidate_stage_isolated(
                clean_background,
                text_ink_mask,
                residual_proposals,
                work_dir,
            )
            prompt_free_residual_candidates.extend(
                generate_geometry_candidates(clean_background)
            )
        else:
            prompted_residual_candidates = generate_prompted_mask_candidates(
                clean_background,
                residual_proposals,
                mask_generator,
                text_ink_mask,
            )
            prompt_free_residual_candidates = generate_mask_candidates(
                clean_background,
                mask_generator,
                crop_size=max(clean_background.shape[:2]),
                include_geometry=True,
                min_score=0.90,
            )
        residual_candidates, attached_count = combine_residual_candidates(
            source=img,
            clean_background=clean_background,
            prompted=prompted_residual_candidates,
            prompt_free=prompt_free_residual_candidates,
            existing=candidates,
            text_mask=text_ink_mask,
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
                    text_clean_image=text_clean_image,
                    text_restore_mask=text_mask,
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
        candidates.extend(residual_candidates)

    elements = resolve_visual_elements(candidates)
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
        text_clean_image=text_clean_image,
        text_restore_mask=text_mask,
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
    element_mask_paths = _persist_visual_masks(
        work_dir,
        "element-masks",
        element_masks,
    )
    semantic_mask_paths = _persist_visual_masks(
        work_dir,
        "semantic-masks",
        semantic_masks,
    )
    material_foreground = (
        np.logical_or.reduce(semantic_masks)
        if semantic_masks
        else np.zeros(img.shape[:2], dtype=bool)
    )
    foreground_evidence_path = work_dir / "foreground-evidence-mask.png"
    Image.fromarray(
        material_foreground.astype(np.uint8) * 255, mode="L"
    ).save(foreground_evidence_path)
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
        "_semantic_mask_paths": semantic_mask_paths,
        "_foreground_evidence_mask_path": str(foreground_evidence_path),
    }
    if text_clean_path is not None:
        slide_data["_text_clean_path"] = str(text_clean_path)
    if defer_quality:
        return slide_data
    return _finalize_slide_quality(slide_data, lang)


_PREPARED_PAGE_SCHEMA_VERSION = 5
_PREPARED_PAGE_NAME = "prepared_page.json"
_PREPARED_PAGE_SIDECAR_NAME = "prepared_page.sha256"
_PREPARED_PAGE_FIELDS = {
    "schema_version",
    "phase",
    "resource_isolation",
    "initial_component_count",
    "components",
    "text_items",
    "initial_diagnostics",
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
_PREPARED_ASSET_FIELDS_V1 = {
    "source_image",
    "ocr_mask",
    "text_clean",
    "element_masks",
    "background_original",
    "background_widescreen",
    "background_removal_mask",
    "background_difference",
}
_PREPARED_ASSET_FIELDS_V2 = _PREPARED_ASSET_FIELDS_V1 | {"semantic_masks"}
_PREPARED_ASSET_FIELDS_V4 = _PREPARED_ASSET_FIELDS_V2 | {"text_cleanup_mask"}
_PREPARED_ASSET_FIELDS_V5 = _PREPARED_ASSET_FIELDS_V4 | {
    "foreground_evidence_mask"
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


def _validate_prepared_work_dir(
    work_dir: str | Path,
    *,
    create: bool,
) -> Path:
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
        if not create:
            raise ValueError(f"work directory does not exist: {lexical}")
        lexical.mkdir(parents=True)
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise ValueError(f"work directory resolves through a link: {lexical}")
    return resolved


def _reject_prepared_links(work_dir: Path) -> None:
    for path in work_dir.rglob("*"):
        status = os.lstat(path)
        if _is_link_or_reparse(status):
            raise ValueError(
                f"prepared asset is a link or reparse point: {path}"
            )
        if stat.S_ISREG(status.st_mode) and status.st_nlink != 1:
            raise ValueError(f"prepared asset must be a single-link file: {path}")


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
    file_status = os.lstat(resolved)
    if not stat.S_ISREG(file_status.st_mode):
        raise ValueError(f"{label} asset is not a regular file: {resolved}")
    if file_status.st_nlink != 1:
        raise ValueError(f"{label} asset must be a single-link file: {resolved}")
    return resolved


def _read_prepared_owned_bytes(
    work_dir: Path,
    value: str | Path,
    label: str,
    *,
    relative_only: bool = False,
    max_bytes: int | None = None,
) -> tuple[Path, bytes]:
    owned = _prepared_owned_file(
        work_dir,
        value,
        label,
        relative_only=relative_only,
    )
    try:
        path_before = os.lstat(owned)
        with owned.open("rb") as source:
            handle_before = os.fstat(source.fileno())
            if max_bytes is not None and handle_before.st_size > max_bytes:
                raise ValueError(f"{label} asset exceeds its size limit")
            content = source.read() if max_bytes is None else source.read(max_bytes + 1)
            if max_bytes is not None and len(content) > max_bytes:
                raise ValueError(f"{label} asset exceeds its size limit")
            handle_after = os.fstat(source.fileno())
        path_after = os.lstat(owned)
    except OSError as exc:
        raise ValueError(f"{label} asset changed while being read: {owned}") from exc

    statuses = (path_before, handle_before, handle_after, path_after)
    for status in statuses:
        if (
            _is_link_or_reparse(status)
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
        ):
            raise ValueError(f"{label} asset changed while being read: {owned}")
    identities = {
        (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
        )
        for status in statuses
    }
    if len(identities) != 1 or len(content) != handle_before.st_size:
        raise ValueError(f"{label} asset changed while being read: {owned}")
    return owned, content


def _prepared_asset_record(work_dir: Path, path: str | Path, label: str) -> dict:
    owned, content = _read_prepared_owned_bytes(work_dir, path, label)
    return {
        "path": owned.relative_to(work_dir).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _read_prepared_asset_bytes(
    work_dir: Path,
    record: object,
    label: str,
    *,
    max_bytes: int | None = None,
) -> tuple[Path, bytes]:
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
    owned, content = _read_prepared_owned_bytes(
        work_dir,
        relative,
        label,
        relative_only=True,
        max_bytes=max_bytes,
    )
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"{label} asset sha256 mismatch")
    return owned, content


def _text_delta_recompute_scope(
    *,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    graph: dict,
    graph_dir: str | Path,
    source_sha256: str,
    cache_identity: dict,
) -> set[str] | None:
    """Return the complete visual dependency closure, or None if unprovable."""
    try:
        old = np.ascontiguousarray(old_mask, dtype=np.uint8)
        new = np.ascontiguousarray(new_mask, dtype=np.uint8)
        if old.ndim != 2 or old.shape != new.shape or not old.size:
            return None
        if np.any((old > 0) & (new == 0)):
            return None
        difference = (new > 0) & (old == 0)
        if not np.any(difference):
            return None

        identity_fields = {
            "schema_version",
            "cache_key",
            "source_sha256",
            "old_cleanup_mask_sha256",
            "sam_protocol_sha256",
            "dino_protocol_sha256",
            "prepared_manifest_sha256",
        }
        if not isinstance(cache_identity, dict) or set(cache_identity) != identity_fields:
            return None
        digest_fields = identity_fields - {"schema_version"}
        if (
            cache_identity["schema_version"] != _TEXT_DELTA_CACHE_SCHEMA_VERSION
            or any(
                not isinstance(cache_identity[field], str)
                or len(cache_identity[field]) != 64
                or any(character not in "0123456789abcdef"
                       for character in cache_identity[field])
                for field in digest_fields
            )
            or source_sha256 != cache_identity["source_sha256"]
            or hashlib.sha256(old.tobytes()).hexdigest()
            != cache_identity["old_cleanup_mask_sha256"]
            or cache_identity["sam_protocol_sha256"]
            != _TEXT_DELTA_SAM_PROTOCOL_SHA256
            or cache_identity["dino_protocol_sha256"]
            != _TEXT_DELTA_DINO_PROTOCOL_SHA256
        ):
            return None
        if (
            not isinstance(graph, dict)
            or set(graph) != {"schema_version", "cache_key", "nodes"}
            or graph["schema_version"] != _TEXT_DELTA_CACHE_SCHEMA_VERSION
            or graph["cache_key"] != cache_identity["cache_key"]
            or not isinstance(graph["nodes"], list)
            or len(graph["nodes"]) > _TEXT_DELTA_MAX_NODES
        ):
            return None
        expected_cache_key = hashlib.sha256(json.dumps({
            "schema_version": cache_identity["schema_version"],
            "source_sha256": cache_identity["source_sha256"],
            "old_cleanup_mask_sha256": cache_identity["old_cleanup_mask_sha256"],
            "sam_protocol_sha256": cache_identity["sam_protocol_sha256"],
            "dino_protocol_sha256": cache_identity["dino_protocol_sha256"],
            "prepared_manifest_sha256": cache_identity["prepared_manifest_sha256"],
            "nodes": graph["nodes"],
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )).hexdigest()
        if expected_cache_key != cache_identity["cache_key"]:
            return None

        trusted_root = _validate_prepared_work_dir(graph_dir, create=False)
        nodes = {}
        masks = {}
        total_crop_pixels = 0
        for node in graph["nodes"]:
            if (
                not isinstance(node, dict)
                or set(node) != {"id", "mask", "bbox", "parents", "children"}
                or not isinstance(node["id"], str)
                or not node["id"]
                or len(node["id"]) > 128
                or node["id"] in nodes
                or not isinstance(node["parents"], list)
                or not isinstance(node["children"], list)
                or len(set(node["parents"])) != len(node["parents"])
                or len(set(node["children"])) != len(node["children"])
                or not all(isinstance(value, str) and value
                           for value in node["parents"] + node["children"])
            ):
                return None
            _, content = _read_prepared_asset_bytes(
                trusted_root,
                node["mask"],
                f"text delta node {node['id']} mask",
                max_bytes=old.size * 2 + 4096,
            )
            with Image.open(io.BytesIO(content)) as stored:
                if stored.size != (old.shape[1], old.shape[0]):
                    return None
                mask = np.asarray(stored.convert("L"), dtype=np.uint8).copy() > 0
            if not np.any(mask):
                return None
            nodes[node["id"]] = node
            ys, xs = np.nonzero(mask)
            left, top = int(xs.min()), int(ys.min())
            right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
            if node["bbox"] != [left, top, right, bottom]:
                return None
            crop_pixels = (right - left) * (bottom - top)
            total_crop_pixels += crop_pixels
            if total_crop_pixels > _TEXT_DELTA_MAX_MASK_CROP_PIXELS:
                return None
            masks[node["id"]] = (
                (left, top, right, bottom),
                mask[top:bottom, left:right].copy(),
            )
            mask = None

        for node_id, node in nodes.items():
            if any(parent not in nodes or node_id not in nodes[parent]["children"]
                   for parent in node["parents"]):
                return None
            if any(child not in nodes or node_id not in nodes[child]["parents"]
                   for child in node["children"]):
                return None

        adjacency = {
            node_id: set(node["parents"] + node["children"])
            for node_id, node in nodes.items()
        }
        kernel = np.ones((7, 7), dtype=np.uint8)
        node_ids = list(nodes)
        pairwise_candidates = 0
        pairwise_pixels = 0
        for left_index, left_id in enumerate(node_ids):
            (lx1, ly1, lx2, ly2), left_crop = masks[left_id]
            left_dilated = cv2.dilate(
                np.pad(left_crop.astype(np.uint8), 3), kernel, iterations=1
            ) > 0
            for right_id in node_ids[left_index + 1:]:
                pairwise_candidates += 1
                if pairwise_candidates > _TEXT_DELTA_MAX_PAIRWISE_CANDIDATES:
                    return None
                (rx1, ry1, rx2, ry2), right_crop = masks[right_id]
                x1, y1 = max(lx1 - 3, rx1), max(ly1 - 3, ry1)
                x2, y2 = min(lx2 + 3, rx2), min(ly2 + 3, ry2)
                if x1 >= x2 or y1 >= y2:
                    continue
                pairwise_pixels += (x2 - x1) * (y2 - y1)
                if pairwise_pixels > _TEXT_DELTA_MAX_PAIRWISE_PIXELS:
                    return None
                left_view = left_dilated[
                    y1 - (ly1 - 3):y2 - (ly1 - 3),
                    x1 - (lx1 - 3):x2 - (lx1 - 3),
                ]
                right_view = right_crop[
                    y1 - ry1:y2 - ry1,
                    x1 - rx1:x2 - rx1,
                ]
                if np.any(left_view & right_view):
                    adjacency[left_id].add(right_id)
                    adjacency[right_id].add(left_id)

        difference = cv2.dilate(
            difference.astype(np.uint8), kernel, iterations=1
        ) > 0
        scope = set()
        for node_id, (bbox, _) in masks.items():
            left, top, right, bottom = bbox
            if np.any(difference[top:bottom, left:right]):
                scope.add(node_id)
        pending = list(scope)
        while pending:
            node_id = pending.pop()
            for related in adjacency[node_id] - scope:
                scope.add(related)
                pending.append(related)
        return scope
    except (OSError, ValueError, TypeError, KeyError, Image.UnidentifiedImageError):
        return None


def _write_first_visual_cache(
    prepared_manifest: dict,
    work_dir: Path,
    old_cleanup_mask: np.ndarray,
) -> Path:
    assets = prepared_manifest["assets"]
    element_records = assets["element_masks"]
    semantic_records = assets["semantic_masks"]
    nodes = []
    for index, (child, parent) in enumerate(zip(element_records, semantic_records)):
        child_id = f"component_{index:04d}:child"
        parent_id = f"component_{index:04d}:semantic"
        bboxes = []
        for label, record in (("element mask", child), ("semantic mask", parent)):
            _, content = _read_prepared_asset_bytes(work_dir, record, label)
            with Image.open(io.BytesIO(content)) as stored:
                grayscale = stored.convert("L")
                try:
                    bbox = grayscale.getbbox()
                finally:
                    grayscale.close()
            if bbox is None:
                raise ValueError("first visual cache mask is empty")
            bboxes.append(list(bbox))
        nodes.extend((
            {"id": child_id, "mask": child, "bbox": bboxes[0],
             "parents": [parent_id], "children": []},
            {"id": parent_id, "mask": parent, "bbox": bboxes[1],
             "parents": [], "children": [child_id]},
        ))
    graph = {
        "schema_version": _TEXT_DELTA_CACHE_SCHEMA_VERSION,
        "cache_key": "",
        "nodes": nodes,
    }
    old_cleanup = np.ascontiguousarray(old_cleanup_mask, dtype=np.uint8)
    prepared_manifest_sha256 = hashlib.sha256(json.dumps(
        prepared_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")).hexdigest()
    identity_seed = {
        "schema_version": _TEXT_DELTA_CACHE_SCHEMA_VERSION,
        "source_sha256": assets["source_image"]["sha256"],
        "old_cleanup_mask_sha256": hashlib.sha256(old_cleanup.tobytes()).hexdigest(),
        "sam_protocol_sha256": _TEXT_DELTA_SAM_PROTOCOL_SHA256,
        "dino_protocol_sha256": _TEXT_DELTA_DINO_PROTOCOL_SHA256,
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "nodes": nodes,
    }
    cache_key = hashlib.sha256(json.dumps(
        identity_seed, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()
    graph["cache_key"] = cache_key
    payload = {
        "schema_version": _TEXT_DELTA_CACHE_SCHEMA_VERSION,
        "identity": {
            "schema_version": _TEXT_DELTA_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "source_sha256": assets["source_image"]["sha256"],
            "old_cleanup_mask_sha256": identity_seed["old_cleanup_mask_sha256"],
            "sam_protocol_sha256": _TEXT_DELTA_SAM_PROTOCOL_SHA256,
            "dino_protocol_sha256": _TEXT_DELTA_DINO_PROTOCOL_SHA256,
            "prepared_manifest_sha256": prepared_manifest_sha256,
        },
        "graph": graph,
    }
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(content.encode("utf-8")) > _TEXT_DELTA_CACHE_MAX_BYTES:
        raise ValueError("first visual cache exceeds its size limit")
    path = work_dir / _TEXT_DELTA_CACHE_NAME
    _atomic_write_prepared_text(
        work_dir, path, content, encoding="utf-8", label="first visual cache"
    )
    return path


def _read_first_visual_cache(path: Path, work_dir: Path) -> dict:
    _, content = _read_prepared_owned_bytes(
        work_dir, path, "first visual cache", max_bytes=_TEXT_DELTA_CACHE_MAX_BYTES
    )
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("first visual cache is invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "identity", "graph"}
        or payload["schema_version"] != _TEXT_DELTA_CACHE_SCHEMA_VERSION
    ):
        raise ValueError("first visual cache fields are invalid")
    return payload


def _load_prepared_asset(work_dir: Path, record: object, label: str) -> str:
    owned, _ = _read_prepared_asset_bytes(work_dir, record, label)
    return str(owned)


def _is_prepared_int(value: object) -> bool:
    return type(value) is int


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

    diagnostics = manifest["initial_diagnostics"]
    assets = manifest.get("assets")
    source_record = assets.get("source_image") if isinstance(assets, dict) else None
    source_sha256 = (
        source_record.get("sha256") if isinstance(source_record, dict) else None
    )
    validate_initial_diagnostics(
        diagnostics,
        source_sha256=source_sha256,
        image_size=(image_width, image_height),
    )


def _atomic_write_prepared_text(
    work_dir: Path,
    path: Path,
    content: str,
    *,
    encoding: str,
    label: str,
) -> None:
    if path.exists() or path.is_symlink():
        _prepared_owned_file(work_dir, path, label)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="\n",
            dir=work_dir,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


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
        "text_cleanup_mask": (
            _prepared_asset_record(
                work_dir,
                slide_data["_text_cleanup_mask_path"],
                "text cleanup mask",
            )
            if slide_data.get("_text_cleanup_mask_path") is not None
            else None
        ),
        "element_masks": [
            _prepared_asset_record(work_dir, path, "element mask")
            for path in slide_data["_element_mask_paths"]
        ],
        "semantic_masks": [
            _prepared_asset_record(work_dir, path, "semantic mask")
            for path in slide_data["_semantic_mask_paths"]
        ],
        "foreground_evidence_mask": _prepared_asset_record(
            work_dir,
            slide_data["_foreground_evidence_mask_path"],
            "foreground evidence mask",
        ),
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
        "initial_diagnostics": slide_data.get("_initial_diagnostics", []),
        "dimensions": dimensions,
        "assets": assets,
    }
    _validate_prepared_payload(manifest)
    state_path = work_dir / _PREPARED_PAGE_NAME
    state_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    state_bytes = state_text.encode("utf-8")
    _atomic_write_prepared_text(
        work_dir,
        state_path,
        state_text,
        encoding="utf-8",
        label="prepared state",
    )
    state_sha256 = hashlib.sha256(state_bytes).hexdigest()
    _atomic_write_prepared_text(
        work_dir,
        work_dir / _PREPARED_PAGE_SIDECAR_NAME,
        f"{state_sha256}\n",
        encoding="ascii",
        label="prepared state sidecar",
    )
    return state_path.resolve()


def _load_component_layer_state(
    state_path: str | Path,
) -> tuple[dict, dict]:
    lexical_state = Path(os.path.abspath(state_path))
    if lexical_state.name != _PREPARED_PAGE_NAME:
        raise ValueError(f"state path must name {_PREPARED_PAGE_NAME}")
    work_dir = _validate_prepared_work_dir(
        lexical_state.parent,
        create=False,
    )
    sidecar_file, sidecar = _read_prepared_owned_bytes(
        work_dir,
        work_dir / _PREPARED_PAGE_SIDECAR_NAME,
        "prepared state sidecar",
    )
    if (
        len(sidecar) != 65
        or sidecar[-1:] != b"\n"
        or any(character not in b"0123456789abcdef" for character in sidecar[:64])
    ):
        raise ValueError("prepared state sidecar format is invalid")
    state_file, state_bytes = _read_prepared_owned_bytes(
        work_dir,
        lexical_state,
        "prepared state",
    )
    if hashlib.sha256(state_bytes).hexdigest() != sidecar[:64].decode("ascii"):
        raise ValueError("prepared state sha256 mismatch")
    try:
        manifest = json.loads(state_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prepared page state is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("prepared page state fields are invalid")
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2, 3, 4, 5}:
        raise ValueError("prepared page schema_version is invalid")
    legacy_fields = _PREPARED_PAGE_FIELDS - {"initial_diagnostics"}
    expected_fields = (
        _PREPARED_PAGE_FIELDS if schema_version >= 3 else legacy_fields
    )
    if set(manifest) != expected_fields:
        raise ValueError("prepared page state fields are invalid")
    if schema_version < 3:
        manifest = {**manifest, "initial_diagnostics": []}
    if manifest["phase"] != "initial_layers":
        raise ValueError("prepared page phase is invalid")
    if type(manifest["resource_isolation"]) is not bool:
        raise ValueError("prepared page resource_isolation is invalid")
    _validate_prepared_payload(manifest)
    initial_count = manifest["initial_component_count"]
    if (
        type(initial_count) is not int
        or initial_count < 0
        or initial_count != len(manifest["components"])
    ):
        raise ValueError("prepared page initial_component_count is invalid")
    dimensions = manifest["dimensions"]
    assets = manifest["assets"]
    expected_asset_fields = (
        _PREPARED_ASSET_FIELDS_V5 if schema_version >= 5
        else _PREPARED_ASSET_FIELDS_V4 if schema_version >= 4
        else _PREPARED_ASSET_FIELDS_V2 if schema_version >= 2
        else _PREPARED_ASSET_FIELDS_V1
    )
    if not isinstance(assets, dict) or set(assets) != expected_asset_fields:
        raise ValueError("prepared page assets are invalid")
    if not isinstance(assets["element_masks"], list):
        raise ValueError("prepared page element masks are invalid")
    if schema_version >= 2:
        if not isinstance(assets["semantic_masks"], list):
            raise ValueError("prepared page semantic masks are invalid")
        if not (
            len(manifest["components"])
            == len(assets["element_masks"])
            == len(assets["semantic_masks"])
        ):
            raise ValueError("prepared page component and mask counts are inconsistent")

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
    foreground_evidence_mask = None
    if schema_version >= 5:
        evidence_path, evidence_content = _read_prepared_asset_bytes(
            work_dir,
            assets["foreground_evidence_mask"],
            "foreground_evidence_mask",
        )
        try:
            with Image.open(io.BytesIO(evidence_content)) as evidence_image:
                if evidence_image.size != (
                    dimensions["img_width"], dimensions["img_height"]
                ):
                    raise ValueError(
                        "prepared page foreground evidence dimensions are invalid"
                    )
                foreground_evidence_mask = np.asarray(
                    evidence_image.convert("L")
                ).copy() > 0
        except OSError as exc:
            raise ValueError(
                "prepared page foreground evidence is invalid"
            ) from exc
        loaded_assets["foreground_evidence_mask"] = str(evidence_path)
    text_clean = assets["text_clean"]
    if text_clean is not None:
        loaded_assets["text_clean"] = _load_prepared_asset(
            work_dir, text_clean, "text_clean"
        )
    text_cleanup_mask = assets.get("text_cleanup_mask")
    if text_cleanup_mask is not None:
        loaded_assets["text_cleanup_mask"] = _load_prepared_asset(
            work_dir, text_cleanup_mask, "text_cleanup_mask"
        )
    element_mask_paths = []
    semantic_mask_paths = None
    if schema_version == 1:
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

    if schema_version >= 2:
        semantic_mask_paths = []
        expected_size = (dimensions["img_width"], dimensions["img_height"])
        semantic_union = np.zeros(
            (dimensions["img_height"], dimensions["img_width"]), dtype=bool
        )
        for index, (child_record, parent_record) in enumerate(
            zip(assets["element_masks"], assets["semantic_masks"])
        ):
            child_path, child_content = _read_prepared_asset_bytes(
                work_dir,
                child_record,
                "element mask",
            )
            parent_path, parent_content = _read_prepared_asset_bytes(
                work_dir,
                parent_record,
                "semantic mask",
            )
            element_mask_paths.append(str(child_path))
            semantic_mask_paths.append(str(parent_path))
            try:
                with (
                    Image.open(io.BytesIO(child_content)) as stored_child,
                    Image.open(io.BytesIO(parent_content)) as stored_parent,
                ):
                    if (
                        stored_child.size != expected_size
                        or stored_parent.size != expected_size
                    ):
                        raise ValueError(
                            f"prepared page mask pair {index} dimensions are invalid"
                        )
                    child_mask = np.asarray(stored_child.convert("L")).copy()
                    parent_mask = np.asarray(stored_parent.convert("L")).copy()
            except OSError as exc:
                raise ValueError(
                    f"prepared page mask pair {index} is invalid"
                ) from exc
            if not np.any(child_mask) or not np.any(parent_mask):
                raise ValueError(
                    f"prepared page mask pair {index} must be non-empty"
                )
            if np.any((child_mask > 0) & (parent_mask == 0)):
                raise ValueError(
                    f"prepared page child mask {index} must be inside its parent"
                )
            semantic_union |= parent_mask > 0
        if (
            schema_version >= 5
            and not np.array_equal(foreground_evidence_mask, semantic_union)
        ):
            raise ValueError(
                "prepared page foreground evidence does not match semantic masks"
            )

    loaded = {
        "phase": "initial_layers",
        "initial_component_count": initial_count,
        "_prepared_schema_version": schema_version,
        "state_path": str(state_file),
        "_resource_isolation": manifest["resource_isolation"],
        **dimensions,
        "components": components,
        "text_items": manifest["text_items"],
        "initial_diagnostics": manifest["initial_diagnostics"],
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
        **(
            {"_text_cleanup_mask_path": loaded_assets["text_cleanup_mask"]}
            if "text_cleanup_mask" in loaded_assets
            else {}
        ),
        "_element_mask_paths": element_mask_paths,
        **(
            {
                "_foreground_evidence_mask_path": loaded_assets[
                    "foreground_evidence_mask"
                ]
            }
            if "foreground_evidence_mask" in loaded_assets
            else {}
        ),
        **(
            {"_semantic_mask_paths": semantic_mask_paths}
            if semantic_mask_paths is not None
            else {}
        ),
        **(
            {"_text_clean_path": loaded_assets["text_clean"]}
            if text_clean is not None
            else {}
        ),
    }
    return loaded, manifest


def load_component_layers(state_path: str | Path) -> dict:
    """Load a prepared page from an existing, validated state directory."""
    loaded, _ = _load_component_layer_state(state_path)
    return loaded


def _reuse_disjoint_text_delta(
    *,
    cache_path: Path,
    prepared_state_path: Path,
    slide_data: dict,
    work_dir: Path,
    source_image: np.ndarray,
    old_cleanup_mask: np.ndarray,
    new_cleanup_mask: np.ndarray,
    new_text_mask: np.ndarray,
    new_text_items: list[dict],
    text_clean_image: np.ndarray,
    text_clean_path: Path,
    resource_isolation: bool,
) -> bool:
    """Reuse first-pass visuals only when a non-empty text delta is disjoint."""
    try:
        payload = _read_first_visual_cache(cache_path, work_dir)
        cached, manifest = _load_component_layer_state(prepared_state_path)
        manifest_sha256 = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")).hexdigest()
        if payload["identity"].get("prepared_manifest_sha256") != manifest_sha256:
            return False
        expected_nodes = []
        cached_nodes = payload["graph"].get("nodes")
        if not isinstance(cached_nodes, list):
            return False
        for index, (child, parent) in enumerate(zip(
            manifest["assets"]["element_masks"],
            manifest["assets"]["semantic_masks"],
            strict=True,
        )):
            child_id = f"component_{index:04d}:child"
            parent_id = f"component_{index:04d}:semantic"
            if index * 2 + 1 >= len(cached_nodes):
                return False
            expected_nodes.extend((
                {"id": child_id, "mask": child,
                 "bbox": cached_nodes[index * 2].get("bbox"),
                 "parents": [parent_id], "children": []},
                {"id": parent_id, "mask": parent,
                 "bbox": cached_nodes[index * 2 + 1].get("bbox"),
                 "parents": [], "children": [child_id]},
            ))
        if payload["graph"].get("nodes") != expected_nodes:
            return False
        for record in manifest["components"]:
            metadata = record["metadata"]
            _, content = _read_prepared_asset_bytes(
                work_dir,
                record["asset"],
                "component RGBA",
                max_bytes=max(4096, metadata["w"] * metadata["h"] * 8),
            )
            with Image.open(io.BytesIO(content)) as stored:
                if stored.size != (metadata["w"], metadata["h"]):
                    return False
                stored.convert("RGBA").load()
        scope = _text_delta_recompute_scope(
            old_mask=old_cleanup_mask,
            new_mask=new_cleanup_mask,
            graph=payload["graph"],
            graph_dir=work_dir,
            source_sha256=manifest["assets"]["source_image"]["sha256"],
            cache_identity=payload["identity"],
        )
        if scope != set():
            return False
        element_masks = []
        semantic_masks = []
        for label, records, output in (
            ("element mask", manifest["assets"]["element_masks"], element_masks),
            ("semantic mask", manifest["assets"]["semantic_masks"], semantic_masks),
        ):
            for record in records:
                _, content = _read_prepared_asset_bytes(
                    work_dir, record, label,
                    max_bytes=source_image.shape[0] * source_image.shape[1] * 2 + 4096,
                )
                with Image.open(io.BytesIO(content)) as stored:
                    output.append(np.asarray(stored.convert("L")).copy() > 0)
    except (OSError, ValueError, TypeError, KeyError, Image.UnidentifiedImageError):
        return False

    background_kwargs = (
        {"large_inpainter": _isolated_large_inpainter(work_dir)}
        if resource_isolation
        else {}
    )
    clean_background = build_clean_background(
        source_image,
        element_masks,
        new_cleanup_mask,
        text_clean_image=text_clean_image,
        text_restore_mask=new_text_mask,
        **background_kwargs,
    )
    background_original_path = work_dir / "targeted-background-original.png"
    background_widescreen_path = work_dir / "targeted-background-16x9.png"
    background_removal_mask_path = work_dir / "targeted-background-removal-mask.png"
    background_difference_path = work_dir / "targeted-background-difference.png"
    foreground_evidence_path = work_dir / "targeted-foreground-evidence-mask.png"
    _save_rgb(background_original_path, clean_background)
    widescreen, offset_x, offset_y, method = build_widescreen_background(
        clean_background, **background_kwargs
    )
    if method == "identity":
        background_widescreen_path = background_original_path
    else:
        _save_rgb(background_widescreen_path, widescreen)
    removal = build_removal_mask(element_masks, new_cleanup_mask)
    Image.fromarray(removal, mode="L").save(background_removal_mask_path)
    _save_rgb(background_difference_path, cv2.absdiff(source_image, clean_background))
    foreground = (
        np.logical_or.reduce(semantic_masks)
        if semantic_masks
        else np.zeros(source_image.shape[:2], dtype=bool)
    )
    Image.fromarray(foreground.astype(np.uint8) * 255, mode="L").save(
        foreground_evidence_path
    )

    try:
        verified, verified_manifest = _load_component_layer_state(prepared_state_path)
    except (OSError, ValueError, TypeError, KeyError, Image.UnidentifiedImageError):
        return False
    if verified_manifest != manifest:
        return False

    slide_data.update({
        "background_path": str(background_widescreen_path),
        "background_original_path": str(background_original_path),
        "background_widescreen_path": str(background_widescreen_path),
        "background_removal_mask_path": str(background_removal_mask_path),
        "background_difference_path": str(background_difference_path),
        "components": verified["components"],
        "text_items": new_text_items,
        "canvas_width": widescreen.shape[1],
        "canvas_height": widescreen.shape[0],
        "content_offset_x": offset_x,
        "content_offset_y": offset_y,
        "widescreen_background_method": method,
        "_text_clean_path": str(text_clean_path),
        "_foreground_evidence_mask_path": str(foreground_evidence_path),
        "_element_mask_paths": verified["_element_mask_paths"],
        "_semantic_mask_paths": verified["_semantic_mask_paths"],
    })
    return True


def prepare_component_layers(
    image_path: str | Path,
    work_dir: str | Path,
    *,
    lang: str,
    resource_isolation: bool,
) -> dict:
    """Persist recoverable OCR and visual layers for Agent review."""
    source = _resolve_image_path(image_path)
    owned_work_dir = _validate_prepared_work_dir(work_dir, create=True)
    _reject_prepared_links(owned_work_dir)
    owned_source = owned_work_dir / f"source-image{source.suffix.lower()}"
    if source != owned_source:
        shutil.copyfile(source, owned_source)

    text_mask = None
    cleanup_mask_path = None
    exception_boundary = sys.exc_info()[1]
    primary_exception = None
    primary_traceback = None
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

    if resource_isolation and text_items and all("box" in item for item in text_items):
        source_image = None
        stored_text_mask = None
        stored_mask = None
        removal_mask = None
        text_clean = None
        exception_boundary = sys.exc_info()[1]
        primary_exception = None
        primary_traceback = None
        try:
            source_image = _load_rgb(owned_source)
            with Image.open(text_analysis["mask_path"]) as stored_text_mask:
                stored_mask = np.asarray(stored_text_mask.convert("L")).copy()
            removal_mask = _build_text_cleanup_mask(
                source_image,
                stored_mask,
                text_items,
            )
            cleanup_mask_path = (
                owned_work_dir / "text-clean-removal-mask.png"
            ).resolve()
            Image.fromarray(removal_mask, mode="L").save(cleanup_mask_path)
            text_clean_path = owned_work_dir / "text-clean.png"
            text_clean = _repair_text_background(
                source_image,
                removal_mask,
                text_items=text_items,
                large_inpainter=_isolated_large_inpainter(owned_work_dir),
            )
            _save_rgb(text_clean_path, text_clean)
            text_analysis["text_clean_path"] = str(text_clean_path)
        except BaseException as exc:
            primary_exception = exc
            primary_traceback = exc.__traceback__
            raise
        finally:
            source_image = None
            stored_text_mask = None
            stored_mask = None
            removal_mask = None
            text_clean = None
            _run_cleanup_preserving_exception(
                gc.collect,
                "isolated text cleanup arrays",
                primary_exception,
                primary_traceback,
                exception_boundary,
            )

    initial_diagnostics = []
    for visual_pass in range(2):
        object_detector = None
        mask_generator = None
        visual_source_image = None
        visual_source_sha256 = None
        exception_boundary = sys.exc_info()[1]
        primary_exception = None
        primary_traceback = None
        try:
            if resource_isolation:
                slide_data = _process_image_isolated(
                    owned_source,
                    owned_work_dir,
                    lang,
                    text_analysis,
                )
                visual_source_sha256 = slide_data.pop(
                    "_visual_source_sha256", None
                )
            else:
                _, visual_source_content = _read_prepared_owned_bytes(
                    owned_work_dir,
                    owned_source,
                    "visual source",
                )
                visual_source_sha256 = hashlib.sha256(
                    visual_source_content
                ).hexdigest()
                with Image.open(io.BytesIO(visual_source_content)) as stored_source:
                    visual_source_image = np.asarray(
                        stored_source.convert("RGB")
                    ).copy()
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
                    _source_image=visual_source_image,
                )
        except BaseException as exc:
            primary_exception = exc
            primary_traceback = exc.__traceback__
            raise
        finally:
            mask_generator = None
            object_detector = None
            visual_source_image = None
            _run_cleanup_preserving_exception(
                _release_visual_resources,
                "visual resources",
                primary_exception,
                primary_traceback,
                exception_boundary,
            )

        if visual_pass:
            break
        with Image.open(text_analysis["mask_path"]) as stored_text_mask:
            sweep_mask = np.asarray(stored_text_mask.convert("L")).copy()
        sweep = _targeted_candidate_ocr_sweep(
            owned_source,
            slide_data["components"],
            text_analysis["items"],
            sweep_mask,
            owned_work_dir,
            lang=lang,
            isolated=resource_isolation,
        )
        initial_diagnostics = sweep["diagnostics"]
        if not sweep["recovered_items"]:
            sweep_mask = None
            break

        _, source_content = _read_prepared_owned_bytes(
            owned_work_dir,
            owned_source,
            "source image for text delta",
        )
        with Image.open(io.BytesIO(source_content)) as stored_source:
            source_for_delta = np.asarray(stored_source.convert("RGB")).copy()
        source_for_delta_sha256 = hashlib.sha256(source_content).hexdigest()
        old_cleanup_mask = _build_text_cleanup_mask(
            source_for_delta,
            sweep_mask if sweep_mask is not None else np.zeros(
                source_for_delta.shape[:2], dtype=np.uint8
            ),
            text_analysis["items"] if text_analysis["items"] else [],
        )
        first_ocr_mask_path = owned_work_dir / "first-ocr-mask.png"
        first_cleanup_mask_path = owned_work_dir / "first-text-cleanup-mask.png"
        Image.fromarray(sweep_mask, mode="L").save(first_ocr_mask_path)
        Image.fromarray(old_cleanup_mask, mode="L").save(first_cleanup_mask_path)
        slide_data["_text_mask_path"] = str(first_ocr_mask_path)
        slide_data["_text_cleanup_mask_path"] = str(first_cleanup_mask_path)
        slide_data["_resource_isolation"] = resource_isolation
        slide_data["_initial_diagnostics"] = initial_diagnostics
        first_state_path = _write_prepared_page(slide_data, owned_work_dir)
        _, first_manifest = _load_component_layer_state(first_state_path)
        cache_path = None
        if (
            first_manifest["assets"]["source_image"]["sha256"]
            == source_for_delta_sha256
            == visual_source_sha256
        ):
            try:
                cache_path = _write_first_visual_cache(
                    first_manifest, owned_work_dir, old_cleanup_mask
                )
            except (OSError, ValueError, TypeError, KeyError):
                cache_path = None
        slide_data["_text_mask_path"] = str(text_mask_path)
        text_items = sweep["items"]
        Image.fromarray(sweep["text_mask"], mode="L").save(text_mask_path)
        text_analysis = {
            "items": text_items,
            "mask_path": str(text_mask_path),
        }
        source_image = source_for_delta
        stored_mask = sweep["text_mask"]
        removal_mask = None
        text_clean = None
        exception_boundary = sys.exc_info()[1]
        primary_exception = None
        primary_traceback = None
        try:
            removal_mask = _build_text_cleanup_mask(
                source_image,
                stored_mask,
                text_items,
            )
            cleanup_mask_path = (
                owned_work_dir / "text-clean-removal-mask.png"
            ).resolve()
            Image.fromarray(removal_mask, mode="L").save(cleanup_mask_path)
            text_clean_path = owned_work_dir / "targeted-text-clean.png"
            repair_kwargs = (
                {"large_inpainter": _isolated_large_inpainter(owned_work_dir)}
                if resource_isolation
                else {}
            )
            text_clean = _repair_text_background(
                source_image,
                removal_mask,
                text_items=text_items,
                **repair_kwargs,
            )
            _save_rgb(text_clean_path, text_clean)
            text_analysis["text_clean_path"] = str(text_clean_path)
            reused = cache_path is not None and _reuse_disjoint_text_delta(
                    cache_path=cache_path,
                    prepared_state_path=first_state_path,
                    slide_data=slide_data,
                    work_dir=owned_work_dir,
                    source_image=source_image,
                    old_cleanup_mask=old_cleanup_mask,
                    new_cleanup_mask=removal_mask,
                    new_text_mask=stored_mask,
                    new_text_items=text_items,
                    text_clean_image=text_clean,
                    text_clean_path=text_clean_path,
                    resource_isolation=resource_isolation,
                )
        except BaseException as exc:
            primary_exception = exc
            primary_traceback = exc.__traceback__
            raise
        finally:
            source_for_delta = None
            source_content = None
            source_image = None
            stored_mask = None
            removal_mask = None
            text_clean = None
            old_cleanup_mask = None
            _run_cleanup_preserving_exception(
                gc.collect,
                "targeted text cleanup arrays",
                primary_exception,
                primary_traceback,
                exception_boundary,
            )
        if reused:
            sweep_mask = None
            break
        _remove_owned_first_visual_assets(slide_data, owned_work_dir)
        sweep_mask = None

    slide_data["original_image_path"] = str(owned_source)
    slide_data["_resource_isolation"] = resource_isolation
    slide_data["_initial_diagnostics"] = initial_diagnostics
    if (
        cleanup_mask_path is None
        and text_items
        and all("box" in item for item in text_items)
    ):
        source_image = _load_rgb(owned_source)
        with Image.open(text_analysis["mask_path"]) as stored_text_mask:
            stored_mask = np.asarray(stored_text_mask.convert("L")).copy()
        removal_mask = _build_text_cleanup_mask(
            source_image,
            stored_mask,
            text_items,
        )
        cleanup_mask_path = (
            owned_work_dir / "text-clean-removal-mask.png"
        ).resolve()
        Image.fromarray(removal_mask, mode="L").save(cleanup_mask_path)
        source_image = None
        stored_mask = None
        removal_mask = None
        gc.collect()
    slide_data.pop("_text_cleanup_mask_path", None)
    if cleanup_mask_path is not None:
        slide_data["_text_cleanup_mask_path"] = str(cleanup_mask_path)
    state_path = _write_prepared_page(slide_data, owned_work_dir)
    return load_component_layers(state_path)


def _snapshot_prepared_asset(
    work_dir: Path,
    record: object,
    label: str,
    staged_dir: Path,
    name: str,
) -> str:
    owned, content = _read_prepared_asset_bytes(work_dir, record, label)
    staged_path = staged_dir / f"{name}{owned.suffix}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(staged_path, flags, 0o600)
    try:
        written = 0
        view = memoryview(content)
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError(f"staged snapshot write failed: {staged_path}")
            written += count
        os.fsync(descriptor)
        handle_status = os.fstat(descriptor)
        path_status = os.lstat(staged_path)
        if (
            _is_link_or_reparse(path_status)
            or not stat.S_ISREG(handle_status.st_mode)
            or not stat.S_ISREG(path_status.st_mode)
            or handle_status.st_nlink != 1
            or path_status.st_nlink != 1
            or (handle_status.st_dev, handle_status.st_ino)
            != (path_status.st_dev, path_status.st_ino)
            or handle_status.st_size != len(content)
            or path_status.st_size != len(content)
        ):
            raise ValueError(f"staged snapshot identity is invalid: {staged_path}")
    finally:
        os.close(descriptor)
    return str(staged_path)


def finalize_component_layers(prepared: dict, accepted, *, lang: str) -> dict:
    """Finalize the exact current components and ``_element_mask_paths``.

    ``accepted`` must map ``components`` to ``prepared["components"]`` and
    ``element_masks`` to ``prepared["_element_mask_paths"]``.
    """
    if type(prepared) is not dict or type(prepared.get("state_path")) is not str:
        raise ValueError("prepared layers must contain a state_path")
    fresh, manifest = _load_component_layer_state(prepared["state_path"])
    if prepared != fresh:
        raise ValueError("prepared layers do not match fresh prepared state")
    if type(accepted) is not dict or set(accepted) != {
        "components",
        "element_masks",
    }:
        raise ValueError(
            "accepted layers must map components to prepared['components'] and "
            "element_masks to prepared['_element_mask_paths']"
        )
    if (
        type(accepted["components"]) is not list
        or type(accepted["element_masks"]) is not list
        or accepted["components"] != fresh["components"]
        or accepted["element_masks"] != fresh["_element_mask_paths"]
    ):
        raise ValueError(
            "accepted components and element_masks must match current prepared state: "
            "prepared['components'] and prepared['_element_mask_paths']"
        )

    work_dir = Path(fresh["_work_dir"])
    components = fresh["components"]
    initial_count = fresh["initial_component_count"]

    staged_dir = None
    keep_staging = False
    try:
        staged_dir = Path(
            tempfile.mkdtemp(prefix="quality-components-", dir=work_dir)
        )
        assets = manifest["assets"]
        staged_assets = {
            key: _snapshot_prepared_asset(
                work_dir,
                assets[key],
                label,
                staged_dir,
                name,
            )
            for key, label, name in (
                ("source_image", "source image", "source-image"),
                ("ocr_mask", "OCR mask", "ocr-mask"),
                (
                    "background_original",
                    "original background",
                    "background-original",
                ),
                (
                    "background_widescreen",
                    "widescreen background",
                    "background-widescreen",
                ),
                (
                    "background_removal_mask",
                    "background removal mask",
                    "background-removal-mask",
                ),
                (
                    "background_difference",
                    "background difference",
                    "background-difference",
                ),
            )
        }
        staged_text_clean = None
        if assets["text_clean"] is not None:
            staged_text_clean = _snapshot_prepared_asset(
                work_dir,
                assets["text_clean"],
                "text-clean image",
                staged_dir,
                "text-clean",
            )
        staged_element_masks = [
            _snapshot_prepared_asset(
                work_dir,
                record,
                "element mask",
                staged_dir,
                f"element-mask-{index:04d}",
            )
            for index, record in enumerate(assets["element_masks"])
        ]
        staged_components = []
        for index, (component, record) in enumerate(
            zip(components, manifest["components"])
        ):
            staged_component = _snapshot_prepared_asset(
                work_dir,
                record["asset"],
                "component RGBA",
                staged_dir,
                f"component_{index:04d}",
            )
            staged_components.append({
                **component,
                "path": staged_component,
            })
        slide_data = {
            key: value
            for key, value in fresh.items()
            if key not in {"phase", "initial_component_count", "state_path"}
        }
        slide_data.update({
            "original_image_path": staged_assets["source_image"],
            "background_path": staged_assets["background_widescreen"],
            "background_original_path": staged_assets["background_original"],
            "background_widescreen_path": staged_assets[
                "background_widescreen"
            ],
            "background_removal_mask_path": staged_assets[
                "background_removal_mask"
            ],
            "background_difference_path": staged_assets[
                "background_difference"
            ],
            "components": staged_components,
            "_text_mask_path": staged_assets["ocr_mask"],
            "_element_mask_paths": staged_element_masks,
        })
        if staged_text_clean is None:
            slide_data.pop("_text_clean_path", None)
        else:
            slide_data["_text_clean_path"] = staged_text_clean

        exception_boundary = sys.exc_info()[1]
        primary_exception = None
        primary_traceback = None
        try:
            result = _finalize_slide_quality(
                slide_data,
                lang,
                _resource_isolation=fresh["_resource_isolation"],
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
        if initial_count and not result["components"]:
            raise VisualSegmentationError(
                "agent-managed quality failed: initial components became empty"
            )
        result.update({
            "phase": "quality_accepted",
            "initial_component_count": initial_count,
            "state_path": fresh["state_path"],
        })
        keep_staging = True
        return result
    finally:
        if staged_dir is not None and not keep_staging:
            shutil.rmtree(staged_dir, ignore_errors=True)


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
                slide_data.pop("_visual_source_sha256", None)
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
        visual_elements=slide_data.get("visual_elements"),
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
