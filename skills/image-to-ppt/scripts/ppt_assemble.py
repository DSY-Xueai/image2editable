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
import math
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from numbers import Integral, Real
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

logger = logging.getLogger(__name__)

# Slide width in inches (standard widescreen reference)
SLIDE_WIDTH_INCHES = 40 / 3
SLIDE_HEIGHT_INCHES = 7.5


@dataclass(frozen=True)
class ContainTransform:
    slide_width: float
    slide_height: float
    content_width: float
    content_height: float
    offset_x: float
    offset_y: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_slide_transform(
    img_width: int,
    img_height: int,
    slide_size: str,
) -> ContainTransform:
    """Map an image to its original aspect ratio or a fixed widescreen slide."""
    if slide_size not in {"original", "16:9"}:
        raise ValueError("slide_size must be 'original' or '16:9'")

    scale = min(
        SLIDE_WIDTH_INCHES / img_width,
        SLIDE_HEIGHT_INCHES / img_height,
    )
    content_width = img_width * scale
    content_height = img_height * scale
    if slide_size == "original":
        if min(content_width, content_height) < 1.0:
            size_scale = 1.0 / min(content_width, content_height)
            content_width *= size_scale
            content_height *= size_scale
        if max(content_width, content_height) > 56.0 + 1e-9:
            raise ValueError(
                "original slide aspect ratio exceeds PowerPoint's 1-56 inch range"
            )
        return ContainTransform(
            slide_width=content_width,
            slide_height=content_height,
            content_width=content_width,
            content_height=content_height,
            offset_x=0,
            offset_y=0,
        )

    return ContainTransform(
        slide_width=SLIDE_WIDTH_INCHES,
        slide_height=SLIDE_HEIGHT_INCHES,
        content_width=content_width,
        content_height=content_height,
        offset_x=(SLIDE_WIDTH_INCHES - content_width) / 2,
        offset_y=(SLIDE_HEIGHT_INCHES - content_height) / 2,
    )


def compute_contain_transform(img_width: int, img_height: int) -> ContainTransform:
    """Map an image into the center of a fixed widescreen slide."""
    return compute_slide_transform(img_width, img_height, "16:9")


def assemble_pptx(
    background_path: str | Path | None,
    components: list[dict],
    text_items: list[dict],
    img_width: int,
    img_height: int,
    output_path: str | Path,
    add_reference_slide: bool = False,
    original_image_path: str | Path | None = None,
    slide_size: str = "16:9",
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    content_offset_x: int = 0,
    content_offset_y: int = 0,
    visual_elements: list[dict] | None = None,
    background_rgb: list[int] | None = None,
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
        canvas_width: Optional exact-16:9 canvas width for widescreen placement.
        canvas_height: Optional exact-16:9 canvas height for widescreen placement.
        content_offset_x: Source-image horizontal offset inside the canvas.
        content_offset_y: Source-image vertical offset inside the canvas.

    Returns:
        The output path as string.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()

    use_canvas = False
    if slide_size == "16:9":
        use_canvas = _validate_canvas(
            img_width,
            img_height,
            canvas_width,
            canvas_height,
            content_offset_x,
            content_offset_y,
        )
    if use_canvas:
        transform = compute_slide_transform(canvas_width, canvas_height, slide_size)
    else:
        transform = compute_slide_transform(img_width, img_height, slide_size)

    prs.slide_width = Inches(transform.slide_width)
    prs.slide_height = Inches(transform.slide_height)
    prs._element.sldSz.set(
        "type",
        "screen16x9" if slide_size == "16:9" else "custom",
    )

    # Use blank layout
    blank_layout = prs.slide_layouts[6]

    # --- Main slide ---
    slide = prs.slides.add_slide(blank_layout)

    # Layer 1 (bottom): Background image
    if background_rgb is None:
        slide.shapes.add_picture(
            str(background_path), 0, 0,
            Inches(transform.slide_width), Inches(transform.slide_height)
        )
    else:
        _set_slide_background(slide, background_rgb)
    logger.info("Added background layer.")

    # Layer 2 (middle): Foreground visual objects
    elements = visual_elements
    if elements is None:
        elements = [
            {
                "route": "raster_component",
                "z_index": index,
                "component": component,
            }
            for index, component in enumerate(components)
        ]
    for element in sorted(elements, key=lambda item: item["z_index"]):
        _add_visual_element(
            slide,
            element,
            img_width,
            img_height,
            transform,
            canvas_width if use_canvas else None,
            canvas_height if use_canvas else None,
            content_offset_x if use_canvas else 0,
            content_offset_y if use_canvas else 0,
        )

    logger.info("Added %d foreground visual objects.", len(elements))

    # Layer 3 (top): Editable text boxes
    for item in text_items:
        _add_textbox(
            slide,
            item,
            img_width,
            img_height,
            transform,
            canvas_width if use_canvas else None,
            canvas_height if use_canvas else None,
            content_offset_x if use_canvas else 0,
            content_offset_y if use_canvas else 0,
        )

    logger.info("Added %d text boxes.", len(text_items))

    # --- Reference slide (optional) ---
    if add_reference_slide and original_image_path:
        original_image_path = Path(original_image_path)
        if original_image_path.exists():
            ref_slide = prs.slides.add_slide(blank_layout)
            if background_rgb is None:
                ref_slide.shapes.add_picture(
                    str(background_path), 0, 0,
                    Inches(transform.slide_width), Inches(transform.slide_height)
                )
            else:
                _set_slide_background(ref_slide, background_rgb)
            left, top, width, height = _map_bbox(
                0,
                0,
                img_width,
                img_height,
                img_width,
                img_height,
                transform,
                canvas_width if use_canvas else None,
                canvas_height if use_canvas else None,
                content_offset_x if use_canvas else 0,
                content_offset_y if use_canvas else 0,
            )
            ref_slide.shapes.add_picture(
                str(original_image_path),
                Inches(left), Inches(top), Inches(width), Inches(height)
            )
            logger.info("Added reference slide with original image.")

    prs.save(str(output_path))
    logger.info("Saved PPTX: %s", output_path)

    return str(output_path)


def assemble_pptx_multi(
    slides_data: list[dict],
    output_path: str | Path,
    add_reference: bool = False,
    slide_size: str = "16:9",
    original_aspect_ratio: float | None = None,
) -> str:
    """Assemble a multi-slide PPTX from multiple images' data.

    Each entry in slides_data is a dict with keys:
        background_path, components, text_items, img_width, img_height,
        original_image_path (optional), canvas_width, canvas_height,
        content_offset_x, content_offset_y (all canvas fields optional).

    Args:
        slides_data: List of per-image data dicts.
        output_path: Where to save the PPTX.
        add_reference: If True, add a reference slide after each content slide.

    Returns:
        The output path as string.
    """
    if slide_size not in {"original", "16:9"}:
        raise ValueError("slide_size must be 'original' or '16:9'")
    if not slides_data:
        raise ValueError("slides_data must not be empty")

    original_transform = None
    if slide_size == "original":
        for data in slides_data:
            img_w = data["img_width"]
            img_h = data["img_height"]
            if img_w <= 0 or img_h <= 0:
                raise ValueError("original slides require positive image dimensions")
        if original_aspect_ratio is None:
            first_width = slides_data[0]["img_width"]
            first_height = slides_data[0]["img_height"]
            original_transform = compute_slide_transform(
                first_width,
                first_height,
                "original",
            )
        else:
            if (
                isinstance(original_aspect_ratio, bool)
                or not isinstance(original_aspect_ratio, Real)
                or not math.isfinite(original_aspect_ratio)
                or original_aspect_ratio <= 0
            ):
                raise ValueError("original_aspect_ratio must be positive and finite")
            for data in slides_data:
                img_w = data["img_width"]
                img_h = data["img_height"]
                lower = (img_w - 1) / img_h
                upper = img_w / (img_h - 1) if img_h > 1 else math.inf
                above_lower = original_aspect_ratio >= lower or math.isclose(
                    original_aspect_ratio, lower, rel_tol=1e-12, abs_tol=1e-12
                )
                below_upper = original_aspect_ratio <= upper or math.isclose(
                    original_aspect_ratio, upper, rel_tol=1e-12, abs_tol=1e-12
                )
                if not above_lower or not below_upper:
                    raise ValueError("original slides must have the same aspect ratio")
            original_transform = compute_slide_transform(
                original_aspect_ratio,
                1.0,
                "original",
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()

    if original_transform is None:
        prs.slide_width = Inches(SLIDE_WIDTH_INCHES)
        prs.slide_height = Inches(SLIDE_HEIGHT_INCHES)
        prs._element.sldSz.set("type", "screen16x9")
    else:
        prs.slide_width = Inches(original_transform.slide_width)
        prs.slide_height = Inches(original_transform.slide_height)
        prs._element.sldSz.set("type", "custom")

    blank_layout = prs.slide_layouts[6]

    for idx, data in enumerate(slides_data):
        img_w = data["img_width"]
        img_h = data["img_height"]
        if original_transform is None:
            canvas_width = data.get("canvas_width")
            canvas_height = data.get("canvas_height")
            content_offset_x = data.get("content_offset_x", 0)
            content_offset_y = data.get("content_offset_y", 0)
            use_canvas = _validate_canvas(
                img_w,
                img_h,
                canvas_width,
                canvas_height,
                content_offset_x,
                content_offset_y,
            )
            if use_canvas:
                transform = compute_contain_transform(canvas_width, canvas_height)
            else:
                transform = compute_contain_transform(img_w, img_h)
        else:
            canvas_width = None
            canvas_height = None
            content_offset_x = 0
            content_offset_y = 0
            use_canvas = False
            scale = min(
                original_transform.slide_width / img_w,
                original_transform.slide_height / img_h,
            )
            content_width = img_w * scale
            content_height = img_h * scale
            transform = ContainTransform(
                slide_width=original_transform.slide_width,
                slide_height=original_transform.slide_height,
                content_width=content_width,
                content_height=content_height,
                offset_x=(original_transform.slide_width - content_width) / 2,
                offset_y=(original_transform.slide_height - content_height) / 2,
            )
        background_key = (
            "background_original_path" if slide_size == "original" else "background_path"
        )

        # --- Content slide ---
        slide = prs.slides.add_slide(blank_layout)

        # Layer 1: Background
        if data.get("background_rgb") is not None:
            _set_slide_background(slide, data["background_rgb"])
        elif original_transform is None:
            slide.shapes.add_picture(
                str(data[background_key]), 0, 0,
                Inches(transform.slide_width), Inches(transform.slide_height)
            )
        else:
            slide.shapes.add_picture(
                str(data[background_key]),
                Inches(transform.offset_x),
                Inches(transform.offset_y),
                Inches(transform.content_width),
                Inches(transform.content_height),
            )

        # Layer 2: Foreground visual objects
        elements = data.get("visual_elements")
        if elements is None:
            elements = [
                {
                    "route": "raster_component",
                    "z_index": index,
                    "component": component,
                }
                for index, component in enumerate(data["components"])
            ]
        for element in sorted(elements, key=lambda item: item["z_index"]):
            _add_visual_element(
                slide,
                element,
                img_w,
                img_h,
                transform,
                canvas_width if use_canvas else None,
                canvas_height if use_canvas else None,
                content_offset_x if use_canvas else 0,
                content_offset_y if use_canvas else 0,
            )

        # Layer 3: Text boxes
        for item in data["text_items"]:
            _add_textbox(
                slide,
                item,
                img_w,
                img_h,
                transform,
                canvas_width if use_canvas else None,
                canvas_height if use_canvas else None,
                content_offset_x if use_canvas else 0,
                content_offset_y if use_canvas else 0,
            )

        logger.info("Slide %d: bg + %d components + %d text boxes.",
                    idx + 1, len(data["components"]), len(data["text_items"]))

        # --- Reference slide (optional) ---
        if add_reference and data.get("original_image_path"):
            orig = Path(data["original_image_path"])
            if orig.exists():
                ref_slide = prs.slides.add_slide(blank_layout)
                if data.get("background_rgb") is not None:
                    _set_slide_background(ref_slide, data["background_rgb"])
                elif original_transform is None:
                    ref_slide.shapes.add_picture(
                        str(data[background_key]), 0, 0,
                        Inches(transform.slide_width), Inches(transform.slide_height)
                    )
                else:
                    ref_slide.shapes.add_picture(
                        str(data[background_key]),
                        Inches(transform.offset_x),
                        Inches(transform.offset_y),
                        Inches(transform.content_width),
                        Inches(transform.content_height),
                    )
                left, top, width, height = _map_bbox(
                    0,
                    0,
                    img_w,
                    img_h,
                    img_w,
                    img_h,
                    transform,
                    canvas_width if use_canvas else None,
                    canvas_height if use_canvas else None,
                    content_offset_x if use_canvas else 0,
                    content_offset_y if use_canvas else 0,
                )
                ref_slide.shapes.add_picture(
                    str(orig),
                    Inches(left), Inches(top), Inches(width), Inches(height)
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


def _set_slide_background(slide, rgb: list[int]) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(*rgb)


def _validate_canvas(
    img_width: int,
    img_height: int,
    canvas_width: int | None,
    canvas_height: int | None,
    content_offset_x: int,
    content_offset_y: int,
) -> bool:
    """Validate an optional widescreen canvas and its source-image rectangle."""
    if not all(
        isinstance(value, Integral) and not isinstance(value, bool)
        for value in (content_offset_x, content_offset_y)
    ):
        raise ValueError("canvas offsets must be integers")
    if canvas_width is None and canvas_height is None:
        if content_offset_x != 0 or content_offset_y != 0:
            raise ValueError("canvas offsets require canvas dimensions")
        return False
    if canvas_width is None or canvas_height is None:
        raise ValueError("canvas_width and canvas_height must be provided together")
    if not all(
        isinstance(value, Integral) and not isinstance(value, bool)
        for value in (canvas_width, canvas_height)
    ):
        raise ValueError("canvas dimensions must be integers")
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("canvas dimensions must be positive")
    if canvas_width * 9 != canvas_height * 16:
        raise ValueError("canvas dimensions must have an exact 16:9 ratio")
    if (
        content_offset_x < 0
        or content_offset_y < 0
        or content_offset_x + img_width > canvas_width
        or content_offset_y + img_height > canvas_height
    ):
        raise ValueError("source image must fit completely inside the canvas")
    return True


def _map_bbox(
    x: int,
    y: int,
    width: int,
    height: int,
    img_w: int,
    img_h: int,
    transform: ContainTransform,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    content_offset_x: int = 0,
    content_offset_y: int = 0,
) -> tuple[float, float, float, float]:
    """Map a source-image bounding box to slide inches."""
    if canvas_width is not None and canvas_height is not None:
        return (
            (x + content_offset_x) / canvas_width * transform.slide_width,
            (y + content_offset_y) / canvas_height * transform.slide_height,
            width / canvas_width * transform.slide_width,
            height / canvas_height * transform.slide_height,
        )

    return (
        transform.offset_x + x / img_w * transform.content_width,
        transform.offset_y + y / img_h * transform.content_height,
        width / img_w * transform.content_width,
        height / img_h * transform.content_height,
    )


def _add_component(
    slide,
    component: dict,
    img_w: int,
    img_h: int,
    transform: ContainTransform,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    content_offset_x: int = 0,
    content_offset_y: int = 0,
    component_id: str | None = None,
) -> None:
    left, top, width, height = _map_bbox(
        component["x"],
        component["y"],
        component["w"],
        component["h"],
        img_w,
        img_h,
        transform,
        canvas_width,
        canvas_height,
        content_offset_x,
        content_offset_y,
    )
    image_source = (
        BytesIO(component["blob"])
        if isinstance(component.get("blob"), bytes)
        else component["path"]
    )
    picture = slide.shapes.add_picture(
        image_source,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    crop = component.get("crop")
    if crop is not None:
        if not isinstance(crop, dict) or set(crop) != {
            "left", "top", "right", "bottom"
        }:
            raise ValueError("component crop is invalid")
        values = [crop[name] for name in ("left", "top", "right", "bottom")]
        if (
            any(type(value) not in {int, float} or not math.isfinite(value) for value in values)
            or any(value < 0 or value >= 1 for value in values)
            or values[0] + values[2] >= 1
            or values[1] + values[3] >= 1
        ):
            raise ValueError("component crop is invalid")
        (
            picture.crop_left,
            picture.crop_top,
            picture.crop_right,
            picture.crop_bottom,
        ) = (float(value) for value in values)
    component_id = component_id or component.get("component_id")
    if isinstance(component_id, str) and component_id:
        picture.name = f"image2editable:{component_id}"


def _add_visual_element(
    slide,
    element: dict,
    img_w: int,
    img_h: int,
    transform: ContainTransform,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    content_offset_x: int = 0,
    content_offset_y: int = 0,
) -> None:
    canvas_args = (
        canvas_width,
        canvas_height,
        content_offset_x,
        content_offset_y,
    )
    route = element["route"]
    if route == "raster_component":
        _add_component(
            slide,
            element["component"],
            img_w,
            img_h,
            transform,
            *canvas_args,
            element.get("object_id"),
        )
        return
    if route == "native_image":
        _add_component(
            slide,
            element["component"],
            img_w,
            img_h,
            transform,
            *canvas_args,
            element.get("object_id"),
        )
        return
    if route == "native_text":
        box = _add_textbox(
            slide,
            element["text"],
            img_w,
            img_h,
            transform,
            *canvas_args,
        )
        box.name = f"image2editable:{element['object_id']}"
        return
    if route == "native_shape":
        _add_native_shape(
            slide,
            element,
            img_w,
            img_h,
            transform,
            *canvas_args,
        )
        return
    raise ValueError(f"Unsupported visual route: {route}")


def _add_native_shape(
    slide,
    element: dict,
    img_w: int,
    img_h: int,
    transform: ContainTransform,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    content_offset_x: int = 0,
    content_offset_y: int = 0,
) -> None:
    payload = element["shape"]
    shape_type = payload["shape_type"]
    fill_rgb = payload["fill_rgb"]
    stroke_rgb = payload.get("stroke_rgb")
    canvas_args = (
        canvas_width,
        canvas_height,
        content_offset_x,
        content_offset_y,
    )
    if shape_type == "line":
        start = payload["line_start"]
        end = payload["line_end"]
        start_left, start_top, _, _ = _map_bbox(
            start[0], start[1], 0, 0, img_w, img_h, transform, *canvas_args
        )
        end_left, end_top, _, _ = _map_bbox(
            end[0], end[1], 0, 0, img_w, img_h, transform, *canvas_args
        )
        _, _, line_width, _ = _map_bbox(
            0,
            0,
            payload["line_width"],
            payload["line_width"],
            img_w,
            img_h,
            transform,
            *canvas_args,
        )
        shape = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT,
            Inches(start_left),
            Inches(start_top),
            Inches(end_left),
            Inches(end_top),
        )
        shape.line.color.rgb = RGBColor(*(stroke_rgb or fill_rgb))
        shape.line.width = Inches(line_width)
        _set_solid_fill_opacity(
            shape._element.spPr.find(qn("a:ln")),
            payload.get("stroke_opacity", 1.0),
        )
    else:
        shape_types = {
            "rectangle": MSO_SHAPE.RECTANGLE,
            "rounded_rectangle": MSO_SHAPE.ROUNDED_RECTANGLE,
            "ellipse": MSO_SHAPE.OVAL,
        }
        if shape_type not in shape_types:
            raise ValueError(f"Unsupported native shape: {shape_type}")
        left, top, right, bottom = element["bbox"]
        mapped = _map_bbox(
            left,
            top,
            right - left,
            bottom - top,
            img_w,
            img_h,
            transform,
            *canvas_args,
        )
        shape = slide.shapes.add_shape(
            shape_types[shape_type], *(Inches(value) for value in mapped)
        )
        if fill_rgb is None:
            shape.fill.background()
        else:
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor(*fill_rgb)
            _set_solid_fill_opacity(
                shape._element.spPr,
                payload.get("fill_opacity", 1.0),
            )
        if stroke_rgb is None:
            shape.line.fill.background()
        else:
            shape.line.color.rgb = RGBColor(*stroke_rgb)
            _, _, line_width, _ = _map_bbox(
                0,
                0,
                payload.get("line_width", 1.0),
                payload.get("line_width", 1.0),
                img_w,
                img_h,
                transform,
                *canvas_args,
            )
            shape.line.width = Inches(line_width)
            _set_solid_fill_opacity(
                shape._element.spPr.find(qn("a:ln")),
                payload.get("stroke_opacity", 1.0),
            )
    shape.name = f"image2editable:{element['object_id']}"


@lru_cache(maxsize=32)
def _ocr_measurement_font(font_name: str, bold: bool, italic: bool):
    from PIL import ImageFont
    names = {
        "Arial": "arialbi.ttf" if bold and italic else "arialbd.ttf" if bold else "ariali.ttf" if italic else "arial.ttf",
        "Microsoft YaHei": "msyhbd.ttc" if bold else "msyh.ttc",
    }
    if font_name in names:
        try:
            return ImageFont.truetype(names[font_name], 1000)
        except OSError:
            pass
    from scripts.font_match import resolve_font
    return resolve_font(font_name, bold, italic)


def _fit_ocr_font_size(text, font_name, bold, italic, font_size, width):
    font = _ocr_measurement_font(font_name, bold, italic)
    if font is None or not text:
        return font_size
    # Measure advances and italic overhangs; keep the detected frame unchanged.
    measured = max(
        max(font.getlength(line), font.getbbox(line)[2] - min(0, font.getbbox(line)[0]))
        for line in text.split("\n")
    )
    return min(font_size, width * 72 * 1000 / measured) if measured > 0 else font_size


def _add_textbox(
    slide,
    item: dict,
    img_w: int,
    img_h: int,
    transform: ContainTransform,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    content_offset_x: int = 0,
    content_offset_y: int = 0,
) -> None:
    """Add an editable text box to the slide.

    Alignment is applied inside the OCR-detected bounds so nearby text keeps
    its original horizontal position.
    """
    if "runs" in item:
        return _add_positioned_text(
            slide, item, img_w, img_h, transform, canvas_width, canvas_height,
            content_offset_x, content_offset_y,
        )
    x, y, w, h = item["box"]

    left, top, width, height = _map_bbox(
        x,
        y,
        w,
        h,
        img_w,
        img_h,
        transform,
        canvas_width,
        canvas_height,
        content_offset_x,
        content_offset_y,
    )

    rotation = item.get("rotation", 0)
    if type(rotation) is not int or rotation not in {0, 90, 180, 270}:
        raise ValueError("text rotation must be one of 0, 90, 180, or 270")
    if rotation in {90, 270}:
        center_x = left + width / 2
        center_y = top + height / 2
        width, height = height, width
        left = center_x - width / 2
        top = center_y - height / 2

    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.rotation = rotation
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
    if canvas_width is not None:
        font_scale = img_w / canvas_width
    else:
        font_scale = transform.content_width / SLIDE_WIDTH_INCHES
    if "font_size_pt" in item:
        font_size = item["font_size_pt"] * transform.content_width * 72 / img_w
    else:
        font_size = item.get("font_size", 12) * font_scale
        if item.get("box_kind") != "ink":
            font_size = _fit_ocr_font_size(
                run.text, item.get("font", "Microsoft YaHei"),
                item.get("bold", False), item.get("italic", False), font_size, width,
            )
        if 0 < font_size < 1:
            from pptx.enum.text import MSO_AUTO_SIZE
            tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
            tf.word_wrap = True
            font_size = 1
    font.size = Pt(font_size)
    if item.get("box_kind") == "ink":
        measured_font = _ocr_measurement_font(
            item.get("font", "Microsoft YaHei"), item.get("bold", False), item.get("italic", False),
        )
        if measured_font is None:
            raise ValueError("text runs ink layout requires installed font metrics")
        mask, offset = measured_font.getmask2(run.text)
        bounds = mask.getbbox()
        if bounds is None:
            raise ValueError("text runs ink layout requires visible glyphs")
        ink_left, ink_top, ink_right, ink_bottom = bounds
        scale = font_size / 72000
        ascent, descent = measured_font.getmetrics()
        # A tight ink box locates visible strokes, not the font's advance box.
        box.left = Inches(left + width / 2 - (ink_left + ink_right + 2 * offset[0]) * scale / 2)
        box.top = Inches(top + height / 2 - (ink_top + ink_bottom + 2 * offset[1]) * scale / 2)
        box.width = Inches(max(measured_font.getlength(run.text), ink_right + offset[0]) * scale)
        box.height = Inches((ascent + descent) * scale)
    character_spacing = item.get("character_spacing_pt", 0.0)
    if (
        type(character_spacing) not in {int, float}
        or not math.isfinite(character_spacing)
        or not -4000.0 <= character_spacing <= 201168.0
    ):
        raise ValueError("text character spacing is invalid")
    if character_spacing:
        spacing = character_spacing * transform.content_width * 72 / img_w
        run._r.get_or_add_rPr().set("spc", str(round(spacing * 100)))
    font.bold = item.get("bold", False)
    font.italic = item.get("italic", False)

    color = _hex_to_rgb(item.get("color", "#000000"))
    font.color.rgb = RGBColor(*color)
    _set_solid_fill_opacity(
        run._r.get_or_add_rPr(), item.get("fill_opacity", 1.0)
    )
    if "gradient" in item:
        rpr = run._r.get_or_add_rPr()
        solid = rpr.find(qn("a:solidFill"))
        gradient = OxmlElement("a:gradFill")
        gradient.set("rotWithShape", "1")
        stops = OxmlElement("a:gsLst")
        for position, value in zip((0, 100000), item["gradient"]["colors"]):
            stop = OxmlElement("a:gs")
            stop.set("pos", str(position))
            color = OxmlElement("a:srgbClr")
            color.set("val", value.lstrip("#"))
            stop.append(color)
            stops.append(stop)
        gradient.append(stops)
        linear = OxmlElement("a:lin")
        linear.set("ang", str(round(item["gradient"]["angle"]*60000)))
        linear.set("scaled", "0")
        gradient.append(linear)
        rpr.replace(solid, gradient)
    if item.get("outline_width", 0):
        outline = OxmlElement("a:ln")
        outline.set("w", str(round(item["outline_width"] * font_scale * 12700)))
        fill = OxmlElement("a:solidFill")
        stroke_color = OxmlElement("a:srgbClr")
        stroke_color.set("val", item["outline_color"].lstrip("#"))
        fill.append(stroke_color)
        outline.append(fill)
        outline.append(OxmlElement("a:round"))
        run._r.get_or_add_rPr().insert(0, outline)

    # Alignment: 0=left, 1=center, 2=right
    from pptx.enum.text import PP_ALIGN
    align_map = {0: PP_ALIGN.LEFT, 1: PP_ALIGN.CENTER, 2: PP_ALIGN.RIGHT}
    p.alignment = PP_ALIGN.LEFT if item.get("box_kind") == "ink" else align_map.get(item.get("align", 1), PP_ALIGN.CENTER)
    return box


def _add_positioned_text(
    slide, item, img_w, img_h, transform, canvas_width, canvas_height,
    content_offset_x, content_offset_y,
):
    from scripts.text_runs import validate_text_runs

    validate_text_runs(item)
    rotation = item.get("rotation", 0)
    if type(rotation) is not int or rotation not in {0, 90, 180, 270}:
        raise ValueError("text rotation must be one of 0, 90, 180, or 270")
    x, y, width, height = item["box"]
    logical_width, logical_height = (height, width) if rotation in {90, 270} else (width, height)
    cosine, sine = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
    group = slide.shapes.add_group_shape()
    for run in item["runs"]:
        rx, ry, rw, rh = run["box"]
        dx = (rx + rw / 2 - 0.5) * logical_width
        dy = (ry + rh / 2 - 0.5) * logical_height
        center_x = x + width / 2 + dx * cosine - dy * sine
        center_y = y + height / 2 + dx * sine + dy * cosine
        run_width, run_height = rw * logical_width, rh * logical_height
        child_item = {key: value for key, value in item.items() if key not in {"runs", "words"}}
        child_item.update(run)
        child_item.update({
            "box": [center_x - run_width / 2, center_y - run_height / 2, run_width, run_height],
            "rotation": 0,
        })
        child = _add_textbox(
            group, child_item, img_w, img_h, transform, canvas_width, canvas_height,
            content_offset_x, content_offset_y,
        )
        child.rotation = (rotation + run.get("rotation", 0)) % 360
        if run.get("box_kind") == "ink":
            mapped_x, mapped_y, _, _ = _map_bbox(
                center_x, center_y, 0, 0, img_w, img_h, transform,
                canvas_width, canvas_height, content_offset_x, content_offset_y,
            )
            dx = child.left + child.width / 2 - Inches(mapped_x)
            dy = child.top + child.height / 2 - Inches(mapped_y)
            angle = math.radians(child.rotation)
            child.left += round(dx * math.cos(angle) - dy * math.sin(angle) - dx)
            child.top += round(dx * math.sin(angle) + dy * math.cos(angle) - dy)
            group.shapes._recalculate_extents()
    return group


def _set_solid_fill_opacity(root, opacity: float) -> None:
    if (
        type(opacity) not in {int, float}
        or not math.isfinite(opacity)
        or not 0.0 <= opacity <= 1.0
    ):
        raise ValueError("fill opacity must be between 0 and 1")
    if opacity == 1.0:
        return
    solid_fill = root.find(qn("a:solidFill")) if root is not None else None
    color = solid_fill.find(qn("a:srgbClr")) if solid_fill is not None else None
    if color is None:
        raise ValueError("fill opacity requires an RGB solid fill")
    alpha = color.find(qn("a:alpha"))
    if alpha is None:
        alpha = OxmlElement("a:alpha")
        color.append(alpha)
    alpha.set("val", str(round(float(opacity) * 100000)))


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
