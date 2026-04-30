#!/usr/bin/env python3
"""Image-to-PPT converter — main entry point.

Converts an image into an editable PowerPoint presentation using:
  1. OCR text detection with style estimation
  2. Adaptive background modeling and inpainting repair
  3. Foreground extraction and component splitting
  4. Layered PPTX assembly (background + components + text boxes)

Usage:
    python image_to_ppt.py input.png
    python image_to_ppt.py input.png -o output.pptx
    python image_to_ppt.py input.png --lang ch --period 32
    python image_to_ppt.py input.png --no-reference
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from bg_model import build_background
from fg_extract import extract_foreground_mask, split_components
from ppt_assemble import assemble_pptx
from text_detect import detect_text

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
    fg_mask = extract_foreground_mask(
        img, bg, text_mask, diff_threshold=diff_threshold
    )

    # Split into components
    work_dir = tempfile.mkdtemp(prefix="img2ppt_")
    bg_path = Path(work_dir) / "background.png"
    _save_rgb(str(bg_path), bg)

    comp_dir = Path(work_dir) / "components"
    components = split_components(
        img, fg_mask, comp_dir, min_area=min_component_area
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_rgb(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
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
        description="Convert image to editable PowerPoint (background + foreground components + text)"
    )
    parser.add_argument("image", help="Input image file (PNG, JPG, etc.)")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output PPTX path (default: same name as input with .pptx)"
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
        "--no-reference", action="store_true", default=True,
        help="Do not add reference slide with original image (default: no reference)"
    )
    parser.add_argument(
        "--reference", action="store_true", default=False,
        help="Add a reference slide with the original image"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    convert(
        image_path=args.image,
        output_path=args.output,
        lang=args.lang,
        bg_period=args.period,
        diff_threshold=args.diff_threshold,
        min_component_area=args.min_area,
        add_reference=args.reference,
    )


if __name__ == "__main__":
    main()
