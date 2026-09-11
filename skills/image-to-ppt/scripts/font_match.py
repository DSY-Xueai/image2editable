"""Match visible glyphs to installed, editable font faces and rotations."""

from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=1)
def installed_faces():
    roots = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
             Path(os.environ.get("LOCALAPPDATA", ".")) / "Microsoft/Windows/Fonts",
             Path("/usr/share/fonts"), Path.home() / ".local/share/fonts"]
    faces = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".ttf", ".ttc", ".otf"}:
                continue
            for index in range(32 if path.suffix.lower() == ".ttc" else 1):
                try:
                    family, style = ImageFont.truetype(str(path), 32, index=index).getname()
                except OSError:
                    break
                bold = any(name in style.lower() for name in ("bold", "black", "heavy"))
                italic = any(name in style.lower() for name in ("italic", "oblique"))
                faces.setdefault((family, bold, italic), (str(path), index))
    return tuple((*key, *value) for key, value in faces.items())


@lru_cache(maxsize=128)
def resolve_font(font_name, bold=False, italic=False, size=1000):
    candidates = [face for face in installed_faces() if face[0].casefold() == font_name.casefold()]
    if not candidates:
        return None
    face = min(candidates, key=lambda face: (face[2] != italic, face[1] != bold))
    try:
        return ImageFont.truetype(face[3], size, index=face[4])
    except OSError:
        return None


@lru_cache(maxsize=4096)
def _glyph(face, text):
    font = ImageFont.truetype(face[3], 128, index=face[4])
    missing = font.getmask("\U0010ffff")
    missing_signature = (missing.size, bytes(missing))
    for char in text:
        mask = font.getmask(char)
        if not char.isspace() and (mask.size, bytes(mask)) == missing_signature:
            return None
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("L", (right-left+16, bottom-top+16))
    ImageDraw.Draw(image).text((8-left, 8-top), text, font=font, fill=255)
    return image


def _normalize(mask):
    bounds = cv2.boundingRect(mask.astype(np.uint8))
    x, y, width, height = bounds
    if not width or not height:
        return None
    cropped = mask[y:y+height, x:x+width].astype(np.float32)
    return cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_AREA), width, height


def match_glyph(mask, text, preferred_font="Arial"):
    """Return native face, clockwise angle, pixel size and measured fit IoU.

    The score measures a local glyph match, not final slide quality.
    """
    target = _normalize(mask)
    if target is None:
        return None
    target_pixels, target_width, target_height = target
    faces = sorted((face for face in installed_faces() if not face[2]),
                   key=lambda face: (face[0] != preferred_font, not face[1]))
    best = None

    def measure(face, glyph, angle):
        nonlocal best
        rotated = glyph.rotate(-angle, Image.Resampling.BICUBIC, expand=True)
        candidate = _normalize(np.asarray(rotated) > 127)
        if candidate is None:
            return
        candidate_pixels, width, height = candidate
        intersection = np.minimum(candidate_pixels, target_pixels).sum()
        union = np.maximum(candidate_pixels, target_pixels).sum()
        iou = float(intersection / max(1, union))
        aspect_error = abs(np.log((width/height)/(target_width/target_height)))
        score = iou - .15*aspect_error
        if face[0].casefold() == preferred_font.casefold():
            score += .05
        if best is None or score > best[0]:
            size = 128*(width*target_width+height*target_height)/(width*width+height*height)
            best = (score, face, glyph, angle, size, iou)

    for face in faces:
        try:
            glyph = _glyph(face, text)
        except OSError:
            # Some installed color/bitmap faces cannot render a text mask.
            continue
        if glyph is None:
            continue
        for angle in range(-35, 36, 5):
            measure(face, glyph, angle)
        if best is not None and best[0] > .975:
            break
    if best is None:
        return None
    _, face, glyph, coarse_angle, _, _ = best
    for angle in range(coarse_angle-4, coarse_angle+5):
        measure(face, glyph, angle)
    _, face, _, angle, size, iou = best
    return {"font": face[0], "bold": face[1], "rotation": angle,
            "font_size": size, "fit_iou": iou}
