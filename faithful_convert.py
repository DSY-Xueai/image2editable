#!/usr/bin/env python3
"""Faithful document-to-PPT converter.

Two modes:
  --mode ai   : AI vision full-element rebuild (no background image, all native editable)
  --mode ocr  : Background image + OCR text overlay (fallback)

Usage:
    python faithful_convert.py input.png                    # auto-detect mode
    python faithful_convert.py input.png --mode ai          # force AI mode
    python faithful_convert.py input.pdf --mode ocr         # force OCR mode
    python faithful_convert.py input.pdf --format ppt169 -o output_dir
"""

from __future__ import annotations

import argparse
import base64
import html
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "skills" / "ppt-master" / "scripts"

CANVAS_FORMATS = {
    "ppt169": {"width": 1280, "height": 720},
    "ppt43": {"width": 1024, "height": 768},
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
RENDER_SCALE = 2  # render at 2x for quality


def _extract_pdf_page_text(doc, page_idx: int) -> list[dict]:
    """Extract text spans with layout info from a PDF page using PyMuPDF."""
    page = doc[page_idx]
    page_rect = page.rect
    spans = []

    data = page.get_text("dict")
    for block in data["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                bbox = span["bbox"]  # (x0, y0, x1, y1) in PDF points
                color = span["color"]
                font_size = span["size"]
                font_name = span["font"]
                flags = span["flags"]

                spans.append({
                    "text": text,
                    "x": bbox[0],
                    "y": bbox[1],
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1],
                    "font_size": font_size,
                    "font_name": font_name,
                    "color": f"#{color:06x}",
                    "bold": bool(flags & 2**4),
                    "italic": bool(flags & 2**1),
                    "page_width": page_rect.width,
                    "page_height": page_rect.height,
                })
    return spans


def _render_pdf_page(doc, page_idx: int, output_path: Path) -> Path:
    """Render a PDF page to PNG."""
    page = doc[page_idx]
    mat = __import__("fitz").Matrix(RENDER_SCALE, RENDER_SCALE)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(output_path))
    return output_path


def _image_to_base64(image_path: Path) -> tuple[str, int, int]:
    """Read image and return base64 string + dimensions."""
    with Image.open(image_path) as img:
        w, h = img.size
    data = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        suffix.lstrip("."), "image/png"
    )
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}", w, h


def _generate_faithful_svg(
    bg_image_path: Path,
    text_spans: list[dict],
    source_width: float,
    source_height: float,
    canvas_width: int,
    canvas_height: int,
) -> str:
    """Generate SVG that faithfully reproduces the source page."""
    bg_data_uri, bg_w, bg_h = _image_to_base64(bg_image_path)

    # Scale factors from source coordinates to canvas
    scale_x = canvas_width / source_width
    scale_y = canvas_height / source_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">',
        "",
        "  <!-- Background image (full page render for visual fidelity) -->",
        f'  <image x="0" y="0" width="{canvas_width}" height="{canvas_height}" '
        f'href="{bg_data_uri}" preserveAspectRatio="none"/>',
        "",
    ]

    if text_spans:
        parts.append("  <!-- Editable text overlays (extracted from source) -->")
        for span in text_spans:
            x = span["x"] * scale_x
            y = span["y"] * scale_y
            font_size = span["font_size"] * scale_y
            color = span["color"]
            text = html.escape(span["text"])

            # Map font name to a PPT-safe font
            font_name = _map_font(span["font_name"])

            weight = ' font-weight="bold"' if span.get("bold") else ""
            style = ' font-style="italic"' if span.get("italic") else ""

            # y offset: SVG text y is baseline, we got top-left from PDF
            baseline_y = y + font_size * 0.85

            # Text is transparent overlay on background image.
            # User can delete background image in PPT to reveal editable text.
            parts.append(
                f'  <text x="{x:.1f}" y="{baseline_y:.1f}" '
                f'font-family="{font_name}" font-size="{font_size:.1f}" '
                f'fill="{color}"{weight}{style} fill-opacity="0.01">'
                f"{text}</text>"
            )

    parts.append("</svg>")
    return "\n".join(parts)


def _map_font(pdf_font: str) -> str:
    """Map PDF font name to a PPT-safe font."""
    lower = pdf_font.lower()
    if any(k in lower for k in ["simhei", "heiti", "hei", "yahei", "msyh"]):
        return "Microsoft YaHei"
    if any(k in lower for k in ["simsun", "songti", "song", "stsung"]):
        return "SimSun"
    if any(k in lower for k in ["simkai", "kaiti", "kai", "stkaiti"]):
        return "KaiTi"
    if any(k in lower for k in ["fangsong", "fang"]):
        return "FangSong"
    if any(k in lower for k in ["arial", "helvetica", "sans"]):
        return "Arial"
    if any(k in lower for k in ["times", "serif"]):
        return "Times New Roman"
    if any(k in lower for k in ["courier", "mono", "consol"]):
        return "Courier New"
    if any(k in lower for k in ["calibri"]):
        return "Calibri"
    # Default: keep original or fall back
    return "Microsoft YaHei"


def _try_ocr_image(image_path: Path) -> list[dict]:
    """Try to OCR an image. Returns text spans or empty list."""
    # Try pytesseract
    try:
        import pytesseract
        from PIL import Image as PILImage

        # Configure Tesseract path on Windows
        tesseract_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if tesseract_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

        img = PILImage.open(image_path)
        data = pytesseract.image_to_data(
            img, lang="chi_sim+eng", output_type=pytesseract.Output.DICT
        )
        spans = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text or data["conf"][i] < 30:
                continue
            spans.append({
                "text": text,
                "x": data["left"][i],
                "y": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "font_size": data["height"][i] * 0.75,
                "font_name": "default",
                "color": "#000000",
                "bold": False,
                "italic": False,
                "page_width": img.size[0],
                "page_height": img.size[1],
            })
        return spans
    except Exception:
        pass

    # Try PaddleOCR
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(lang="ch")
        result = ocr.predict(str(image_path))
        spans = []
        if result:
            for item in result:
                for i, text in enumerate(item.rec_texts):
                    box = item.dt_polys[i]
                    x1, y1 = box[0]
                    x2, y2 = box[2]
                    h = y2 - y1
                    with Image.open(image_path) as img:
                        pw, ph = img.size
                    spans.append({
                        "text": text,
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(h),
                        "font_size": float(h) * 0.75,
                        "font_name": "default",
                        "color": "#000000",
                        "bold": False,
                        "italic": False,
                        "page_width": pw,
                        "page_height": ph,
                    })
        return spans
    except Exception:
        pass

    return []


def _generate_ai_svg(
    elements: dict, canvas_width: int, canvas_height: int,
    source_image: Path | None = None,
) -> str:
    """Generate SVG from AI-analyzed element descriptions.

    For image_region elements: crop from source image and embed as base64.
    For text/rect/line/circle: generate native editable SVG elements.
    """
    bg = elements.get("background", {})
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">',
        "",
    ]

    # Background
    bg_type = bg.get("type", "solid")
    if bg_type == "gradient":
        direction = bg.get("gradient_direction", "top_to_bottom")
        x1, y1, x2, y2 = "0", "0", "0", str(canvas_height)
        if direction == "left_to_right":
            x1, y1, x2, y2 = "0", "0", str(canvas_width), "0"
        elif direction == "diagonal":
            x2, y2 = str(canvas_width), str(canvas_height)
        parts.append("  <defs>")
        parts.append(
            f'    <linearGradient id="bg_grad" x1="{x1}" y1="{y1}" '
            f'x2="{x2}" y2="{y2}" gradientUnits="userSpaceOnUse">'
        )
        parts.append(
            f'      <stop offset="0%" stop-color="{bg.get("gradient_start", "#ffffff")}"/>'
        )
        parts.append(
            f'      <stop offset="100%" stop-color="{bg.get("gradient_end", "#ffffff")}"/>'
        )
        parts.append("    </linearGradient>")
        parts.append("  </defs>")
        parts.append(
            f'  <rect x="0" y="0" width="{canvas_width}" height="{canvas_height}" '
            f'fill="url(#bg_grad)"/>'
        )
    else:
        color = bg.get("color", "#ffffff")
        parts.append(
            f'  <rect x="0" y="0" width="{canvas_width}" height="{canvas_height}" '
            f'fill="{color}"/>'
        )
    parts.append("")

    # Elements
    for elem in elements.get("elements", []):
        etype = elem.get("type", "")
        x = elem.get("x_pct", 0) / 100 * canvas_width
        y = elem.get("y_pct", 0) / 100 * canvas_height
        w = elem.get("width_pct", 0) / 100 * canvas_width
        h = elem.get("height_pct", 0) / 100 * canvas_height

        if etype == "rect":
            fill = elem.get("fill", "#cccccc")
            stroke = elem.get("stroke")
            sw = elem.get("stroke_width", 1)
            rx = elem.get("corner_radius", 0)
            opacity = elem.get("opacity", 1)
            stroke_attr = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
            opacity_attr = f' fill-opacity="{opacity}"' if opacity < 1 else ""
            parts.append(
                f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'rx="{rx}" fill="{fill}"{stroke_attr}{opacity_attr}/>'
            )

        elif etype == "circle":
            cx = elem.get("cx_pct", 0) / 100 * canvas_width
            cy = elem.get("cy_pct", 0) / 100 * canvas_height
            r = elem.get("r_pct", 0) / 100 * canvas_width
            fill = elem.get("fill", "#cccccc")
            stroke = elem.get("stroke")
            stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
            parts.append(
                f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                f'fill="{fill}"{stroke_attr}/>'
            )

        elif etype == "line":
            x1 = elem.get("x1_pct", 0) / 100 * canvas_width
            y1 = elem.get("y1_pct", 0) / 100 * canvas_height
            x2 = elem.get("x2_pct", 0) / 100 * canvas_width
            y2 = elem.get("y2_pct", 0) / 100 * canvas_height
            stroke = elem.get("stroke", "#000000")
            sw = elem.get("stroke_width", 1)
            parts.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{stroke}" stroke-width="{sw}"/>'
            )

        elif etype == "text":
            text_content = html.escape(elem.get("text", ""))
            font_size = elem.get("font_size_pct", 3) / 100 * canvas_height
            color = elem.get("color", "#000000")
            weight = elem.get("font_weight", "normal")
            align = elem.get("align", "left")

            if align == "center":
                text_x = x + w / 2
                anchor = "middle"
            elif align == "right":
                text_x = x + w
                anchor = "end"
            else:
                text_x = x
                anchor = "start"

            text_y = y + font_size * 0.85
            weight_attr = ' font-weight="bold"' if weight == "bold" else ""

            parts.append(
                f'  <text x="{text_x:.1f}" y="{text_y:.1f}" '
                f'font-family="Microsoft YaHei" font-size="{font_size:.1f}" '
                f'fill="{color}" text-anchor="{anchor}"{weight_attr}>'
                f"{text_content}</text>"
            )

        elif etype == "image_region" and source_image and source_image.exists():
            # Crop this region from the source image and embed as base64
            crop_data = _crop_and_encode(source_image, elem, canvas_width, canvas_height)
            if crop_data:
                parts.append(
                    f'  <image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                    f'href="{crop_data}" preserveAspectRatio="xMidYMid meet"/>'
                )

        elif etype == "image_region":
            # No source image available, placeholder
            fill = elem.get("fill", "#e0e0e0")
            parts.append(
                f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'fill="{fill}" stroke="#cccccc" stroke-width="0.5"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def _crop_and_encode(
    image_path: Path, elem: dict, canvas_width: int, canvas_height: int
) -> str | None:
    """Crop a region from the source image and return as base64 data URI."""
    try:
        with Image.open(image_path) as img:
            iw, ih = img.size
            # Convert percentage to pixel coordinates in source image
            x1 = int(elem.get("x_pct", 0) / 100 * iw)
            y1 = int(elem.get("y_pct", 0) / 100 * ih)
            x2 = int((elem.get("x_pct", 0) + elem.get("width_pct", 0)) / 100 * iw)
            y2 = int((elem.get("y_pct", 0) + elem.get("height_pct", 0)) / 100 * ih)

            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            cropped = img.crop((x1, y1, x2, y2))
            import io
            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def convert_image_ai(
    image_path: Path, output_dir: Path, canvas_format: str = "ppt169",
    provider: str | None = None,
) -> list[Path]:
    """Convert image using AI vision — full element rebuild, no background image."""
    from vision_analyzer import analyze_image

    canvas = CANVAS_FORMATS[canvas_format]
    cw, ch = canvas["width"], canvas["height"]

    svg_dir = output_dir / "svg_output"
    svg_dir.mkdir(parents=True, exist_ok=True)

    elements = analyze_image(image_path, provider=provider)
    svg_content = _generate_ai_svg(elements, cw, ch, source_image=image_path)

    svg_path = svg_dir / "slide_01.svg"
    svg_path.write_text(svg_content, encoding="utf-8")
    return [svg_path]


def convert_pdf_ai(
    pdf_path: Path, output_dir: Path, canvas_format: str = "ppt169",
    provider: str | None = None,
) -> list[Path]:
    """Convert PDF using AI vision — render each page, then AI rebuilds."""
    import fitz
    from vision_analyzer import analyze_image

    canvas = CANVAS_FORMATS[canvas_format]
    cw, ch = canvas["width"], canvas["height"]

    doc = fitz.open(str(pdf_path))
    svg_files = []

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = output_dir / "svg_output"
    svg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(doc)):
        # Render page to image for AI analysis
        bg_path = assets_dir / f"page_{i+1:03d}.png"
        _render_pdf_page(doc, i, bg_path)

        print(f"  Page {i+1}/{len(doc)}:")
        elements = analyze_image(bg_path, provider=provider)
        svg_content = _generate_ai_svg(elements, cw, ch, source_image=bg_path)

        svg_path = svg_dir / f"slide_{i+1:02d}.svg"
        svg_path.write_text(svg_content, encoding="utf-8")
        svg_files.append(svg_path)

    doc.close()
    return svg_files


def convert_pdf(
    pdf_path: Path, output_dir: Path, canvas_format: str = "ppt169"
) -> list[Path]:
    """Convert PDF to faithful SVG pages."""
    import fitz

    canvas = CANVAS_FORMATS[canvas_format]
    cw, ch = canvas["width"], canvas["height"]

    doc = fitz.open(str(pdf_path))
    svg_files = []

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = output_dir / "svg_output"
    svg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(doc)):
        page = doc[i]
        page_rect = page.rect

        # Render page as background
        bg_path = assets_dir / f"page_{i+1:03d}.png"
        _render_pdf_page(doc, i, bg_path)

        # Extract native text
        spans = _extract_pdf_page_text(doc, i)

        # If no native text, try OCR on the rendered image
        if not spans:
            print(f"  Page {i+1}: no native text, trying OCR...")
            ocr_spans = _try_ocr_image(bg_path)
            if ocr_spans:
                # OCR coordinates are in rendered image space (scaled)
                with Image.open(bg_path) as img:
                    render_w, render_h = img.size
                for s in ocr_spans:
                    s["page_width"] = render_w
                    s["page_height"] = render_h
                spans = ocr_spans
                source_w, source_h = render_w, render_h
            else:
                print(f"  Page {i+1}: OCR unavailable, background-only mode")
                source_w, source_h = page_rect.width, page_rect.height
        else:
            source_w, source_h = page_rect.width, page_rect.height

        print(f"  Page {i+1}: {len(spans)} text spans extracted")

        svg_content = _generate_faithful_svg(
            bg_path, spans, source_w, source_h, cw, ch
        )
        svg_path = svg_dir / f"slide_{i+1:02d}.svg"
        svg_path.write_text(svg_content, encoding="utf-8")
        svg_files.append(svg_path)

    doc.close()
    return svg_files


def convert_image(
    image_path: Path, output_dir: Path, canvas_format: str = "ppt169"
) -> list[Path]:
    """Convert image to faithful SVG page."""
    canvas = CANVAS_FORMATS[canvas_format]
    cw, ch = canvas["width"], canvas["height"]

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = output_dir / "svg_output"
    svg_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        source_w, source_h = img.size

    # Try OCR
    spans = _try_ocr_image(image_path)
    if spans:
        print(f"  OCR extracted {len(spans)} text spans")
    else:
        print("  OCR unavailable, background-only mode")

    svg_content = _generate_faithful_svg(
        image_path, spans, source_w, source_h, cw, ch
    )
    svg_path = svg_dir / "slide_01.svg"
    svg_path.write_text(svg_content, encoding="utf-8")
    return [svg_path]


def export_pptx(project_dir: Path) -> Path | None:
    """Run ppt-master post-processing and export."""
    svg_dir = project_dir / "svg_output"
    if not list(svg_dir.glob("*.svg")):
        print("[ERROR] No SVG files to export")
        return None

    # Finalize SVG
    print("\n[STEP] Finalizing SVGs...")
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "finalize_svg.py"), str(project_dir)],
        cwd=str(REPO_ROOT),
    )

    # Export PPTX
    print("\n[STEP] Exporting to PPTX...")
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "svg_to_pptx.py"),
         str(project_dir), "-s", "final"],
        cwd=str(REPO_ROOT),
    )

    # Find exported file
    exports_dir = project_dir / "exports"
    exports_dir.mkdir(exist_ok=True)
    pptx_files = sorted(exports_dir.glob("*.pptx"), reverse=True)
    native = [f for f in pptx_files if "_svg" not in f.name]
    return native[0] if native else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faithful document-to-PPT converter"
    )
    parser.add_argument("input", help="Input file (PDF, image)")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: auto-generated in projects/)"
    )
    parser.add_argument(
        "--format", default="ppt169", choices=list(CANVAS_FORMATS.keys()),
        help="Canvas format (default: ppt169)"
    )
    parser.add_argument(
        "--mode", default="auto", choices=["auto", "ai", "ocr"],
        help="Conversion mode: ai (full rebuild), ocr (background+text), auto (try ai first)"
    )
    parser.add_argument(
        "--provider", default=None, choices=["anthropic", "openai"],
        help="AI provider for --mode ai (default: auto-detect)"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    ext = input_path.suffix.lower()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "projects" / f"faithful_{input_path.stem}_{ts}"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exports").mkdir(exist_ok=True)

    # Determine mode
    mode = args.mode
    if mode == "auto":
        try:
            from vision_analyzer import detect_provider
            detect_provider()
            mode = "ai"
            print("[INFO] AI provider detected, using AI full-rebuild mode")
        except Exception:
            mode = "ocr"
            print("[INFO] No AI provider, falling back to OCR mode")

    print(f"[INFO] Input: {input_path.name}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Format: {args.format}")
    print(f"[INFO] Mode: {mode}")
    print()

    # Convert
    if ext in {".pdf"}:
        if mode == "ai":
            print("[STEP] Converting PDF pages (AI full rebuild)...")
            svg_files = convert_pdf_ai(input_path, output_dir, args.format, args.provider)
        else:
            print("[STEP] Converting PDF pages (OCR + background)...")
            svg_files = convert_pdf(input_path, output_dir, args.format)
    elif ext in IMAGE_EXTENSIONS:
        if mode == "ai":
            print("[STEP] Converting image (AI full rebuild)...")
            svg_files = convert_image_ai(input_path, output_dir, args.format, args.provider)
        else:
            print("[STEP] Converting image (OCR + background)...")
            svg_files = convert_image(input_path, output_dir, args.format)
    else:
        print(f"[ERROR] Unsupported format: {ext}")
        sys.exit(1)

    print(f"\n[OK] Generated {len(svg_files)} SVG page(s)")

    # Export to PPTX
    result = export_pptx(output_dir)
    if result:
        print(f"\n{'='*60}")
        print(f"[OK] Faithful PPTX exported: {result}")
        print(f"{'='*60}")
    else:
        print("\n[WARN] PPTX export failed, SVGs are in svg_output/")


if __name__ == "__main__":
    main()
