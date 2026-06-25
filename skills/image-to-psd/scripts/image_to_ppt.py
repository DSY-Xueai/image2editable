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
    python image_to_ppt.py input.png --lang ch --period 32
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.bg_model import build_background
from scripts.fg_extract import extract_foreground_mask, split_components
from scripts.ppt_assemble import assemble_pptx, assemble_pptx_multi
from scripts.text_detect import detect_text

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def convert(
    image_path: str | Path,
    output_path: str | Path | None = None,
    lang: str = "ch",
    bg_period: int = 32,
    diff_threshold: float = 20.0,
    min_component_area: int = 20,
    add_reference: bool = False,
) -> str:
    """Full pipeline: image → PPTX.

    Args:
        image_path: Path to input image.
        output_path: Where to save the PPTX. Auto-generated if None.
        lang: OCR language code.
        bg_period: Tile period for background modeling.
        diff_threshold: Foreground detection sensitivity.
        min_component_area: Minimum component area in pixels.
        add_reference: Add a reference slide with the original image.

    Returns:
        Path to the output PPTX file.
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = image_path.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {ext}")

    if output_path is None:
        output_path = image_path.with_suffix(".pptx")
    output_path = Path(output_path).resolve()

    # Load image
    print(f"[1/5] Loading image: {image_path.name}")
    img = _load_rgb(str(image_path))
    img_h, img_w = img.shape[:2]
    print(f"      Image size: {img_w} x {img_h}")

    # Step 1: Text detection
    print(f"[2/5] Detecting text (OCR)...")
    text_items, text_mask = detect_text(image_path, lang=lang)
    print(f"      Found {len(text_items)} text regions")

    # Step 2: Background modeling (first pass)
    print(f"[3/5] Building background model...")
    bg = build_background(img, text_mask=text_mask, period=bg_period)

    # Step 3: Foreground extraction
    print(f"[4/5] Extracting foreground components...")
    fg_mask = extract_foreground_mask(
        img, bg, text_mask, diff_threshold=diff_threshold
    )

    # Iterative refinement: use foreground mask to improve background
    bg = build_background(
        img, text_mask=text_mask, fg_hint_mask=fg_mask, period=bg_period
    )

    # Re-extract foreground with improved background
    refined_fg_mask = extract_foreground_mask(
        img, bg, text_mask, diff_threshold=diff_threshold
    )
    fg_mask = _merge_foreground_masks(fg_mask, refined_fg_mask)

    # Split into components
    work_dir = tempfile.mkdtemp(prefix="img2ppt_")
    bg_path = Path(work_dir) / "background.png"
    _save_rgb(str(bg_path), bg)

    comp_dir = Path(work_dir) / "components"
    components = split_components(
        img, fg_mask, comp_dir, min_area=min_component_area, text_mask=text_mask
    )
    print(f"      {len(components)} components extracted")

    # Step 4: Assemble PPTX
    print(f"[5/5] Assembling PPTX...")
    result = assemble_pptx(
        background_path=str(bg_path),
        components=components,
        text_items=text_items,
        img_width=img_w,
        img_height=img_h,
        output_path=str(output_path),
        add_reference_slide=add_reference,
        original_image_path=str(image_path),
    )

    print(f"\nDone!")
    print(f"  Output: {result}")
    print(f"  Components: {len(components)}")
    print(f"  Text boxes: {len(text_items)}")
    print(f"  Assets: {work_dir}")

    return result


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
        bg_period: Tile period for background modeling.
        diff_threshold: Foreground detection sensitivity.
        min_component_area: Minimum component area in pixels.
        add_reference: Add reference slides with original images.

    Returns:
        Path to the output PPTX file.
    """
    resolved_paths = []
    for p in image_paths:
        p = Path(p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        ext = p.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {ext} ({p.name})")
        resolved_paths.append(p)

    if not resolved_paths:
        raise ValueError("No valid images provided")

    if output_path is None:
        output_path = resolved_paths[0].with_suffix(".pptx")
    output_path = Path(output_path).resolve()

    total = len(resolved_paths)
    print(f"Processing {total} image(s) into one PPTX...\n")

    slides_data = []
    for i, img_path in enumerate(resolved_paths):
        print(f"=== Image {i + 1}/{total}: {img_path.name} ===")

        # Load image
        print(f"  [1/4] Loading image...")
        img = _load_rgb(str(img_path))
        img_h, img_w = img.shape[:2]
        print(f"         Size: {img_w} x {img_h}")

        # Text detection
        print(f"  [2/4] Detecting text (OCR)...")
        text_items, text_mask = detect_text(img_path, lang=lang)
        print(f"         Found {len(text_items)} text regions")

        # Background modeling
        print(f"  [3/4] Building background model...")
        bg = build_background(img, text_mask=text_mask, period=bg_period)

        # Foreground extraction
        print(f"  [4/4] Extracting foreground components...")
        fg_mask = extract_foreground_mask(
            img, bg, text_mask, diff_threshold=diff_threshold
        )

        # Iterative refinement
        bg = build_background(
            img, text_mask=text_mask, fg_hint_mask=fg_mask, period=bg_period
        )
        refined_fg_mask = extract_foreground_mask(
            img, bg, text_mask, diff_threshold=diff_threshold
        )
        fg_mask = _merge_foreground_masks(fg_mask, refined_fg_mask)

        # Split components
        work_dir = tempfile.mkdtemp(prefix=f"img2ppt_{i}_")
        bg_path = Path(work_dir) / "background.png"
        _save_rgb(str(bg_path), bg)

        comp_dir = Path(work_dir) / "components"
        components = split_components(
            img, fg_mask, comp_dir, min_area=min_component_area, text_mask=text_mask
        )
        print(f"         {len(components)} components extracted\n")

        slides_data.append({
            "background_path": str(bg_path),
            "components": components,
            "text_items": text_items,
            "img_width": img_w,
            "img_height": img_h,
            "original_image_path": str(img_path),
        })

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert image(s) to editable PowerPoint (background + foreground components + text)"
    )
    parser.add_argument(
        "images", nargs="+",
        help="Input image file(s) or directory containing images"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output PPTX path (default: same name as first input with .pptx)"
    )
    parser.add_argument(
        "--lang", default="ch",
        help="OCR language (default: ch)"
    )
    parser.add_argument(
        "--period", type=int, default=32,
        help="Background tile period (default: 32)"
    )
    parser.add_argument(
        "--diff-threshold", type=float, default=20.0,
        help="Foreground detection threshold (default: 20.0)"
    )
    parser.add_argument(
        "--min-area", type=int, default=20,
        help="Minimum component area in pixels (default: 20)"
    )
    parser.add_argument(
        "--no-reference", action="store_true", default=False,
        help="Do not add reference slide with original image (default: no reference)"
    )
    parser.add_argument(
        "--reference", action="store_true", default=False,
        help="Add a reference slide with the original image"
    )

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

    if len(image_files) == 1:
        # Single image: use original convert() for full backward compatibility
        convert(
            image_path=image_files[0],
            output_path=args.output,
            lang=args.lang,
            bg_period=args.period,
            diff_threshold=args.diff_threshold,
            min_component_area=args.min_area,
            add_reference=add_reference,
        )
    else:
        # Multiple images: batch into one PPTX
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
