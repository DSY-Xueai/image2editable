#!/usr/bin/env python3
"""Image-to-PSD converter.

Converts one or more images into layered PSD files using the same OCR,
background repair, and foreground extraction pipeline as image_to_ppt.py.
PSD text export requires a licensed Aspose.PSD runtime.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

skill_root = str(Path(__file__).resolve().parents[1])
if skill_root not in sys.path:
    sys.path.insert(0, skill_root)

from image_to_ppt import (
    IMAGE_EXTENSIONS,
    _load_rgb,
    _merge_foreground_masks,
    _resolve_inputs,
    _save_rgb,
)
from scripts.bg_model import build_background
from scripts.fg_extract import extract_foreground_mask, split_components
from scripts.psd_assemble import (
    AsposePsdLicenseError,
    assemble_psd,
    ensure_aspose_psd_license,
)
from scripts.text_detect import detect_text

logger = logging.getLogger(__name__)


def convert(
    image_path: str | Path,
    output_path: str | Path | None = None,
    lang: str = "ch",
    bg_period: int = 32,
    diff_threshold: float = 20.0,
    min_component_area: int = 20,
) -> str:
    """Full pipeline: image -> layered PSD."""
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix.lower()}")

    output = _resolve_output_paths([image_path], output_path)[0].resolve()
    ensure_aspose_psd_license()

    print(f"[1/5] Loading image: {image_path.name}")
    img = _load_rgb(str(image_path))
    img_h, img_w = img.shape[:2]
    print(f"      Image size: {img_w} x {img_h}")

    print("[2/5] Detecting text (OCR)...")
    text_items, text_mask = detect_text(image_path, lang=lang)
    print(f"      Found {len(text_items)} text regions")

    print("[3/5] Building background model...")
    bg = build_background(img, text_mask=text_mask, period=bg_period)

    print("[4/5] Extracting foreground components...")
    fg_mask = extract_foreground_mask(
        img, bg, text_mask, diff_threshold=diff_threshold
    )
    bg = build_background(
        img, text_mask=text_mask, fg_hint_mask=fg_mask, period=bg_period
    )
    refined_fg_mask = extract_foreground_mask(
        img, bg, text_mask, diff_threshold=diff_threshold
    )
    fg_mask = _merge_foreground_masks(fg_mask, refined_fg_mask)

    work_dir = tempfile.mkdtemp(prefix="img2psd_")
    bg_path = Path(work_dir) / "background.png"
    _save_rgb(str(bg_path), bg)

    comp_dir = Path(work_dir) / "components"
    components = split_components(
        img, fg_mask, comp_dir, min_area=min_component_area, text_mask=text_mask
    )
    print(f"      {len(components)} components extracted")

    print("[5/5] Assembling PSD...")
    result = assemble_psd(
        background_path=bg_path,
        components=components,
        text_items=text_items,
        img_width=img_w,
        img_height=img_h,
        output_path=output,
    )

    print("\nDone!")
    print(f"  Output: {result}")
    print(f"  Components: {len(components)}")
    print(f"  Text layers: {len(text_items)}")
    print(f"  Assets: {work_dir}")

    return result


def convert_batch(
    image_paths: list[str | Path],
    output_path: str | Path | None = None,
    lang: str = "ch",
    bg_period: int = 32,
    diff_threshold: float = 20.0,
    min_component_area: int = 20,
) -> list[str]:
    """Process multiple images into one PSD file per image."""
    resolved_paths = [Path(p).resolve() for p in image_paths]
    output_paths = _resolve_output_paths(resolved_paths, output_path)

    results = []
    total = len(resolved_paths)
    for idx, img_path in enumerate(resolved_paths):
        print(f"=== Image {idx + 1}/{total}: {img_path.name} ===")
        results.append(
            convert(
                image_path=img_path,
                output_path=output_paths[idx],
                lang=lang,
                bg_period=bg_period,
                diff_threshold=diff_threshold,
                min_component_area=min_component_area,
            )
        )
        print()
    return results


def _resolve_output_paths(
    image_paths: list[Path],
    output_path: str | Path | None,
) -> list[Path]:
    if not image_paths:
        raise ValueError("No valid images provided")

    if len(image_paths) == 1:
        if output_path is None:
            return [image_paths[0].with_suffix(".psd")]
        output = Path(output_path)
        if output.suffix.lower() == ".psd":
            return [output]
        return [output / f"{image_paths[0].stem}.psd"]

    if output_path is None:
        return [p.with_suffix(".psd") for p in image_paths]

    out_dir = Path(output_path)
    if out_dir.suffix.lower() == ".psd":
        raise ValueError("Multiple PSD outputs require an output directory")
    return [out_dir / f"{p.stem}.psd" for p in image_paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert image(s) to layered PSD files"
    )
    parser.add_argument(
        "images", nargs="+",
        help="Input image file(s) or directory containing images"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output PSD path for one image, or output directory for multiple images",
    )
    parser.add_argument("--lang", default="ch", help="OCR language (default: ch)")
    parser.add_argument(
        "--period", type=int, default=32,
        help="Background tile period (default: 32)",
    )
    parser.add_argument(
        "--diff-threshold", type=float, default=20.0,
        help="Foreground detection threshold (default: 20.0)",
    )
    parser.add_argument(
        "--min-area", type=int, default=20,
        help="Minimum component area in pixels (default: 20)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    image_files = _resolve_inputs(args.images)
    if not image_files:
        print("Error: No valid image files found in the provided input(s).")
        sys.exit(1)

    try:
        if len(image_files) == 1:
            convert(
                image_path=image_files[0],
                output_path=args.output,
                lang=args.lang,
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
            )
    except AsposePsdLicenseError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
