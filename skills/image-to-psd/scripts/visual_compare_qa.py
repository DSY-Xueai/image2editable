#!/usr/bin/env python3
"""Generate visual QA artifacts for a source image and a rendered PPT preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat


def write_visual_compare(
    source_path: str | Path,
    preview_path: str | Path,
    out_dir: str | Path,
) -> dict:
    """Write side-by-side, blend, heatmap, and JSON diff metrics."""
    source_path = Path(source_path)
    preview_path = Path(preview_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as src, Image.open(preview_path) as prv:
        preview = prv.convert("RGB")
        source = src.convert("RGB").resize(preview.size)

    source.save(out_dir / "source_resized.png")
    preview.save(out_dir / "preview.png")

    side_by_side = Image.new("RGB", (preview.width * 2, preview.height), (18, 18, 18))
    side_by_side.paste(source, (0, 0))
    side_by_side.paste(preview, (preview.width, 0))
    draw = ImageDraw.Draw(side_by_side)
    draw.rectangle([0, 0, 120, 28], fill=(0, 0, 0))
    draw.rectangle([preview.width, 0, preview.width + 120, 28], fill=(0, 0, 0))
    draw.text((8, 7), "source", fill=(255, 255, 255))
    draw.text((preview.width + 8, 7), "preview", fill=(255, 255, 255))
    side_by_side.save(out_dir / "side_by_side.png")

    blend = Image.blend(source, preview, 0.5)
    blend.save(out_dir / "blend.png")

    diff = ImageChops.difference(source, preview)
    stat = ImageStat.Stat(diff)
    mean_abs = sum(stat.mean) / 3.0
    rms = (sum(v * v for v in stat.rms) / 3.0) ** 0.5

    gray = ImageOps.grayscale(diff)
    hist = gray.histogram()
    total = max(preview.width * preview.height, 1)
    changed_32 = sum(hist[32:]) / total
    changed_64 = sum(hist[64:]) / total

    heat_alpha = gray.point(lambda v: min(220, int(v * 1.35)))
    heat = Image.new("RGBA", preview.size, (255, 0, 0, 0))
    heat.putalpha(heat_alpha)
    heat_base = source.convert("RGBA")
    heat_base.alpha_composite(heat)
    heat_base.convert("RGB").save(out_dir / "diff_heatmap.png")

    report = {
        "source": str(source_path),
        "preview": str(preview_path),
        "preview_size": list(preview.size),
        "mean_abs_diff_0_255": round(mean_abs, 4),
        "rms_diff_0_255": round(rms, 4),
        "changed_pixel_fraction_threshold_32": round(changed_32, 6),
        "changed_pixel_fraction_threshold_64": round(changed_64, 6),
        "artifacts": {
            "source_resized": str(out_dir / "source_resized.png"),
            "preview": str(out_dir / "preview.png"),
            "side_by_side": str(out_dir / "side_by_side.png"),
            "blend": str(out_dir / "blend.png"),
            "diff_heatmap": str(out_dir / "diff_heatmap.png"),
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Original source image.")
    parser.add_argument("preview", help="Rendered PPT preview image.")
    parser.add_argument("--out-dir", required=True, help="Directory for QA artifacts.")
    args = parser.parse_args()

    report = write_visual_compare(args.source, args.preview, args.out_dir)
    print(f"Wrote {report['artifacts']['side_by_side']}")
    print(f"Wrote {report['artifacts']['blend']}")
    print(f"Wrote {report['artifacts']['diff_heatmap']}")
    print(f"Wrote {Path(args.out_dir) / 'report.json'}")


if __name__ == "__main__":
    main()
