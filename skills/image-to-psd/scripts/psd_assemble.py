#!/usr/bin/env python3
"""PSD assembly module.

Builds a layered PSD with repaired background, foreground pixel layers,
and real Photoshop text layers. Text-layer creation requires a licensed
Aspose.PSD runtime configured through ASPOSE_PSD_LICENSE.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from PIL import Image


class AsposePsdLicenseError(RuntimeError):
    """Raised when PSD text-layer export cannot use a licensed Aspose.PSD."""


def assemble_psd(
    background_path: str | Path,
    components: list[dict],
    text_items: list[dict],
    img_width: int,
    img_height: int,
    output_path: str | Path,
) -> str:
    """Assemble a layered PSD from background, foreground components, and text."""
    ensure_aspose_psd_license()

    from aspose.psd import Color, Rectangle
    from aspose.psd.fileformats.psd import PsdImage

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    psd = PsdImage(int(img_width), int(img_height))
    try:
        psd.layers = []
        psd.add_layer(_make_pixel_layer(background_path, "Background"))

        for idx, comp in enumerate(components, start=1):
            layer = _make_pixel_layer(comp["path"], f"Foreground {idx:03d}")
            layer.left = int(comp["x"])
            layer.top = int(comp["y"])
            layer.right = int(comp["x"] + comp["w"])
            layer.bottom = int(comp["y"] + comp["h"])
            psd.add_layer(layer)

        for idx, item in enumerate(text_items, start=1):
            x, y, w, h = [int(v) for v in item["box"]]
            rect = Rectangle(x, y, max(w, 1), max(h, 1))
            layer = psd.add_text_layer(item.get("text", ""), rect)
            layer.display_name = f"Text {idx:03d}"
            _style_text_layer(layer, item, Color)

        psd.save(str(output_path))
    finally:
        psd.dispose()

    return str(output_path)


def ensure_aspose_psd_license() -> None:
    license_path = os.environ.get("ASPOSE_PSD_LICENSE")
    if not license_path:
        raise AsposePsdLicenseError(
            "PSD export requires a licensed Aspose.PSD runtime. "
            "Set ASPOSE_PSD_LICENSE to your Aspose.PSD .lic file."
        )

    path = Path(license_path).expanduser()
    if not path.exists():
        raise AsposePsdLicenseError(f"ASPOSE_PSD_LICENSE does not exist: {path}")

    try:
        from aspose.psd import License

        License().set_license(str(path))
    except Exception as exc:
        raise AsposePsdLicenseError(
            f"Failed to load Aspose.PSD license from {path}: {exc}"
        ) from exc


def _make_pixel_layer(image_path: str | Path, name: str):
    from aspose.psd.fileformats.psd.layers import Layer

    with Image.open(image_path) as img:
        rgba = img.convert("RGBA")
        buffer = BytesIO()
        rgba.save(buffer, format="PNG")
    buffer.seek(0)
    layer = Layer(buffer)
    layer.display_name = name
    return layer


def _style_text_layer(layer, item: dict, color_cls) -> None:
    color = color_cls.from_argb(255, *_hex_to_rgb(item.get("color", "#000000")))
    font_size = float(item.get("font_size", 12))
    bold = bool(item.get("bold", False))

    text_data = getattr(layer, "text_data", None)
    if text_data is not None:
        try:
            portions = list(getattr(text_data, "items", []))
            if not portions:
                portions = [text_data.produce_portion()]
            for portion in portions:
                style = portion.style
                style.fill_color = color
                style.font_size = font_size
                if hasattr(style, "faux_bold"):
                    style.faux_bold = bold
            text_data.update_layer_data()
            return
        except Exception:
            pass

    try:
        layer.text_color = color
    except Exception:
        pass

    try:
        layer.font_size = font_size
    except Exception:
        pass


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0, 0, 0)
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )
