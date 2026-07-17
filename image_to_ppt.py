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
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.bg_model import (
    build_clean_background,
    build_removal_mask,
    extend_background_to_widescreen,
)
from scripts.fg_extract import (
    _build_text_ink_mask,
    export_visual_components,
    repair_exported_component_text,
)
from scripts.lama_inpaint import inpaint_large_mask
from scripts.object_detect import (
    create_object_detector,
    filter_text_overlapping_proposals,
    generate_object_proposals,
)
from scripts.ppt_assemble import assemble_pptx, assemble_pptx_multi
from scripts.text_detect import detect_text
from scripts.visual_segment import (
    VisualSegmentationError,
    create_sam_generator,
    filter_prompt_free_candidates,
    filter_unchanged_residual_candidates,
    generate_mask_candidates,
    generate_prompted_mask_candidates,
    reconcile_residual_candidates,
    recheck_visual_element_holes,
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


def _build_text_item_removal_mask(
    shape: tuple[int, int],
    text_items: list[dict],
    padding: int = 2,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in text_items:
        x, y, box_width, box_height = (int(value) for value in item["box"])
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + box_width + padding)
        y2 = min(height, y + box_height + padding)
        mask[y1:y2, x1:x2] = 255
    return mask


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


def _process_image(
    image_path: Path,
    work_dir: Path,
    object_detector,
    mask_generator,
    lang: str,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    img = _load_rgb(str(image_path))
    img_h, img_w = img.shape[:2]
    text_items, text_mask = detect_text(image_path, lang=lang)
    text_ink_mask = _build_text_ink_mask(img, text_mask)
    valid_text_items = bool(text_items) and all(
        "box" in item for item in text_items
    )
    text_clean_image = None
    if valid_text_items:
        text_clean_image = inpaint_large_mask(
            img,
            _build_text_item_removal_mask(img.shape[:2], text_items),
        )

    proposals = filter_text_overlapping_proposals(
        generate_object_proposals(img, object_detector), text_mask
    )
    candidates = generate_prompted_mask_candidates(
        img,
        proposals,
        mask_generator,
        text_ink_mask,
    )
    candidates.extend(
        filter_prompt_free_candidates(
            generate_mask_candidates(
                img,
                mask_generator,
                crop_size=max(img.shape[:2]),
            ),
            candidates,
            text_ink_mask,
        )
    )
    for round_index in range(3):
        elements = resolve_visual_elements(candidates)
        element_masks = [element.mask for element in elements]
        validate_visual_masks(element_masks)
        clean_background = build_clean_background(img, element_masks, text_mask)
        residual_proposals = filter_text_overlapping_proposals(
            generate_object_proposals(clean_background, object_detector),
            text_mask,
        )
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
                    img, element_masks, text_mask
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

    recheck_visual_element_holes(img, elements, mask_generator)
    element_masks = [element.mask for element in elements]
    semantic_masks = [element.semantic_mask for element in elements]
    validate_visual_masks(element_masks)
    clean_background = build_clean_background(img, element_masks, text_mask)
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
    _save_rgb(
        str(background_widescreen_path),
        extend_background_to_widescreen(clean_background, 1920, 1080),
    )
    removal_mask = build_removal_mask(element_masks, text_mask)
    Image.fromarray(removal_mask, mode="L").save(background_removal_mask_path)
    _save_rgb(
        str(background_difference_path),
        cv2.absdiff(img, clean_background),
    )

    visual_only = _compose_exported_components(clean_background, components)
    visual_only_path = work_dir / "visual-only.png"
    _save_rgb(str(visual_only_path), visual_only)

    raster_text_items, raster_text_mask = detect_text(visual_only_path, lang=lang)
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
        visual_only = _compose_exported_components(clean_background, components)
        _save_rgb(str(visual_only_path), visual_only)
        raster_text_items, _ = detect_text(visual_only_path, lang=lang)
    if raster_text_items:
        raise VisualSegmentationError(
            "visual components still contain raster text after cleanup"
        )

    quality = visual_difference(img, visual_only, text_mask)
    diagnostics_dir = (work_dir / "diagnostics").resolve()
    write_segmentation_diagnostics(
        diagnostics_dir,
        source=img,
        masks=element_masks,
        reconstructed=visual_only,
        metrics=quality,
    )
    try:
        require_visual_quality(quality)
    except VisualSegmentationError as exc:
        raise VisualSegmentationError(
            f"{exc}; mae={quality['mae']:.3f}, p95={quality['p95']:.3f}, "
            f"diagnostics={diagnostics_dir}"
        ) from exc

    return {
        "background_path": str(background_widescreen_path),
        "background_original_path": str(background_original_path),
        "background_widescreen_path": str(background_widescreen_path),
        "components": components,
        "text_items": text_items,
        "img_width": img_w,
        "img_height": img_h,
        "original_image_path": str(image_path),
        "quality": quality,
    }


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


def _prepare_single_image(
    image_path: str | Path,
    lang: str,
) -> tuple[dict, Path]:
    resolved = _resolve_image_path(image_path)
    print("[1/3] Loading visual models...")
    object_detector = create_object_detector()
    mask_generator = create_sam_generator(resolve_sam_checkpoint())
    work_dir = Path(tempfile.mkdtemp(prefix="img2ppt_")).resolve()
    print(f"[2/3] Decomposing image: {resolved.name}")
    print(f"  Assets/diagnostics: {work_dir}")
    slide_data = _process_image(
        resolved,
        work_dir,
        object_detector,
        mask_generator,
        lang,
    )
    return slide_data, work_dir


def _prepare_multiple_images(
    image_paths: list[str | Path],
    lang: str,
) -> list[dict]:
    resolved_paths = [_resolve_image_path(path) for path in image_paths]
    if not resolved_paths:
        raise ValueError("No valid images provided")

    total = len(resolved_paths)
    print(f"Processing {total} image(s)...\n")
    object_detector = create_object_detector()
    mask_generator = create_sam_generator(resolve_sam_checkpoint())
    slides_data = []
    for index, image_path in enumerate(resolved_paths):
        print(f"=== Image {index + 1}/{total}: {image_path.name} ===")
        work_dir = Path(tempfile.mkdtemp(prefix=f"img2ppt_{index}_")).resolve()
        print(f"  Assets/diagnostics: {work_dir}")
        slide_data = _process_image(
            image_path,
            work_dir,
            object_detector,
            mask_generator,
            lang,
        )
        slides_data.append(slide_data)
        print(f"         {len(slide_data['components'])} components extracted\n")
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

    slide_data, work_dir = _prepare_single_image(image_path, lang)
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
) -> dict[str, str]:
    original_output, widescreen_output = _variant_output_paths(
        image_path,
        output_path,
    )
    slide_data, work_dir = _prepare_single_image(image_path, lang)
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
    slides_data = _prepare_multiple_images(image_paths, lang)
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
) -> dict[str, str | list[str] | None]:
    slides_data = _prepare_multiple_images(image_paths, lang)
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
