#!/usr/bin/env python3
"""PPTX assembly module — compose background, foreground components, and text.

Builds a PowerPoint presentation with layered structure:
  - Bottom: repaired background image (full slide)
  - Middle: independent transparent foreground component images
  - Top: editable text boxes with font styling

Usage:
    from ppt_assemble import assemble_pptx
    assemble_pptx(bg_path, components, text_items, img_w, img_h, output_path)
"""

from __future__ import annotations

import logging
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

logger = logging.getLogger(__name__)

# Slide width in inches (standard widescreen reference)
SLIDE_WIDTH_INCHES = 13.333


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_pptx(
    background_path: str | Path,
    components: list[dict],
    text_items: list[dict],
    img_width: int,
    img_height: int,
    output_path: str | Path,
    add_reference_slide: bool = False,
    original_image_path: str | Path | None = None,
) -> str:
    """Assemble a PPTX from background, foreground components, and text.

    Args:
        background_path: Path to the clean background PNG.
        components: List of component dicts (path, x, y, w, h, area).
        text_items: List of text dicts (box, text, font_size, color, bold, font, align).
        img_width: Original image width in pixels.
        img_height: Original image height in pixels.
        output_path: Where to save the PPTX.
        add_reference_slide: If True, add a second slide with the original image.
        original_image_path: Path to original image (for reference slide).

    Returns:
        The output path as string.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()

    # Set slide dimensions to match image aspect ratio
    slide_w = SLIDE_WIDTH_INCHES
    slide_h = slide_w * img_height / img_width

    prs.slide_width = Inches(slide_w)
    prs.slide_height = Inches(slide_h)

    # Use blank layout
    blank_layout = prs.slide_layouts[6]

    # --- Main slide ---
    slide = prs.slides.add_slide(blank_layout)

    # Layer 1 (bottom): Background image
    slide.shapes.add_picture(
        str(background_path), 0, 0, Inches(slide_w), Inches(slide_h)
    )
    logger.info("Added background layer.")

    # Layer 2 (middle): Foreground components
    for comp in components:
        left = Inches(comp["x"] / img_width * slide_w)
        top = Inches(comp["y"] / img_height * slide_h)
        width = Inches(comp["w"] / img_width * slide_w)
        height = Inches(comp["h"] / img_height * slide_h)
        slide.shapes.add_picture(comp["path"], left, top, width, height)

    logger.info("Added %d foreground components.", len(components))

    # Layer 3 (top): Editable text boxes
    for item in text_items:
        _add_textbox(slide, item, img_width, img_height, slide_w, slide_h)

    logger.info("Added %d text boxes.", len(text_items))

    # --- Reference slide (optional) ---
    if add_reference_slide and original_image_path:
        original_image_path = Path(original_image_path)
        if original_image_path.exists():
            ref_slide = prs.slides.add_slide(blank_layout)
            ref_slide.shapes.add_picture(
                str(original_image_path), 0, 0, Inches(slide_w), Inches(slide_h)
            )
            logger.info("Added reference slide with original image.")

    prs.save(str(output_path))
    logger.info("Saved PPTX: %s", output_path)

    return str(output_path)


def assemble_pptx_multi(
    slides_data: list[dict],
    output_path: str | Path,
    add_reference: bool = False,
) -> str:
    """Assemble a multi-slide PPTX from multiple images' data.

    Each entry in slides_data is a dict with keys:
        background_path, components, text_items, img_width, img_height,
        original_image_path (optional).

    Args:
        slides_data: List of per-image data dicts.
        output_path: Where to save the PPTX.
        add_reference: If True, add a reference slide after each content slide.

    Returns:
        The output path as string.
    """
    if not slides_data:
        raise ValueError("slides_data must not be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()

    # Use first image to set slide dimensions
    first = slides_data[0]
    slide_w = SLIDE_WIDTH_INCHES
    slide_h = slide_w * first["img_height"] / first["img_width"]

    prs.slide_width = Inches(slide_w)
    prs.slide_height = Inches(slide_h)

    blank_layout = prs.slide_layouts[6]

    for idx, data in enumerate(slides_data):
        img_w = data["img_width"]
        img_h = data["img_height"]
        # Per-slide aspect ratio (height may differ from first slide)
        s_h = slide_w * img_h / img_w

        # --- Content slide ---
        slide = prs.slides.add_slide(blank_layout)

        # Layer 1: Background
        slide.shapes.add_picture(
            str(data["background_path"]), 0, 0, Inches(slide_w), Inches(s_h)
        )

        # Layer 2: Foreground components
        for comp in data["components"]:
            left = Inches(comp["x"] / img_w * slide_w)
            top = Inches(comp["y"] / img_h * s_h)
            width = Inches(comp["w"] / img_w * slide_w)
            height = Inches(comp["h"] / img_h * s_h)
            slide.shapes.add_picture(comp["path"], left, top, width, height)

        # Layer 3: Text boxes
        for item in data["text_items"]:
            _add_textbox(slide, item, img_w, img_h, slide_w, s_h)

        logger.info("Slide %d: bg + %d components + %d text boxes.",
                    idx + 1, len(data["components"]), len(data["text_items"]))

        # --- Reference slide (optional) ---
        if add_reference and data.get("original_image_path"):
            orig = Path(data["original_image_path"])
            if orig.exists():
                ref_slide = prs.slides.add_slide(blank_layout)
                ref_slide.shapes.add_picture(
                    str(orig), 0, 0, Inches(slide_w), Inches(s_h)
                )

    prs.save(str(output_path))
    logger.info("Saved multi-slide PPTX (%d slides): %s", len(slides_data), output_path)

    return str(output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to (r, g, b) tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0, 0, 0)
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def _add_textbox(
    slide,
    item: dict,
    img_w: int,
    img_h: int,
    slide_w: float,
    slide_h: float,
) -> None:
    """Add an editable text box to the slide.

    Alignment is applied inside the OCR-detected bounds so nearby text keeps
    its original horizontal position.
    """
    x, y, w, h = item["box"]

    # Map vertical position and height from image to slide
    top = Inches(y / img_h * slide_h)
    height = Inches(h / img_h * slide_h)

    left = Inches(x / img_w * slide_w)
    width = Inches(w / img_w * slide_w)

    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = False
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    from pptx.enum.text import MSO_VERTICAL_ANCHOR
    tf.vertical_anchor = MSO_VERTICAL_ANCHOR.MIDDLE
    tf.clear()

    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = item.get("text", "")

    font = run.font
    _set_run_font(run, item.get("font", "Microsoft YaHei"))
    font.size = Pt(item.get("font_size", 12))
    font.bold = item.get("bold", False)

    color = _hex_to_rgb(item.get("color", "#000000"))
    font.color.rgb = RGBColor(*color)

    # Alignment: 0=left, 1=center, 2=right
    from pptx.enum.text import PP_ALIGN
    align_map = {0: PP_ALIGN.LEFT, 1: PP_ALIGN.CENTER, 2: PP_ALIGN.RIGHT}
    p.alignment = align_map.get(item.get("align", 1), PP_ALIGN.CENTER)


def _set_run_font(run, font_name: str) -> None:
    """Set both Latin and East Asian font names for PowerPoint."""
    run.font.name = font_name
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea"):
        node = rpr.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            rpr.append(node)
        node.set("typeface", font_name)
