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
                try:
                    variations = ImageFont.truetype(str(path), 32, index=index).get_variation_names()
                except OSError:
                    variations = []
                if b'Regular' in variations and b'Bold' in variations:
                    faces.setdefault((family, True, italic), (str(path), index))
    return tuple((*key, *value) for key, value in faces.items())


@lru_cache(maxsize=128)
def resolve_font(font_name, bold=False, italic=False, size=1000):
    candidates = [face for face in installed_faces() if face[0].casefold() == font_name.casefold()]
    if not candidates:
        return None
    face = min(candidates, key=lambda face: (face[2] != italic, face[1] != bold))
    try:
        return _load_face(face, size)
    except OSError:
        return None


def _load_face(face, size):
    font = ImageFont.truetype(face[3], size, index=face[4])
    try:
        weight = b'Bold' if face[1] else b'Regular'
        if weight in font.get_variation_names():
            font.set_variation_by_name(weight)
    except OSError:
        pass
    return font


@lru_cache(maxsize=4096)
def _glyph(face, text, size=128):
    font = _load_face(face, size)
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


@lru_cache(maxsize=128)
def match_text_face(pixels: bytes, width: int, height: int, text: str):
    """Match a straight text line locally; whitespace does not determine weight."""
    from scripts.text_detect import _normalized_ink

    region = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float32)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    contrast = np.abs(gray - np.median(border))
    # A cell border at the OCR crop edge is not part of the glyph height.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (contrast > contrast.max() * .2).astype(np.uint8), 8,
    )
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        if (h == height and w <= 2 and (x == 0 or x + w == width)) or (
            w == width and h <= 2 and (y == 0 or y + h == height)
        ):
            contrast[labels == label] = 0
    # OCR boxes may include the descenders of the preceding line. Do not
    # measure that fragment as part of this line's font height.
    rows = np.flatnonzero((contrast > contrast.max() * .2).any(axis=1))
    bands = np.split(rows, np.flatnonzero(np.diff(rows) > max(2, height * .05)) + 1)
    if len(bands) > 1:
        weights = [float(contrast[band].sum()) for band in bands]
        selected = int(np.argmax(weights))
        if sum(weights) - weights[selected] > weights[selected] * .25:
            return None
        keep = bands[selected]
        contrast[:keep[0]] = 0
        contrast[keep[-1] + 1:] = 0
    target = _normalized_ink(contrast)
    if target is None:
        return None
    compact = target[:, target.max(axis=0) > .2]
    edges = np.diff(np.pad((target.max(axis=0) > .2).astype(np.int8), 1))
    intervals = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    chars = [char for char in text if not char.isspace()]
    letters = {}
    if len(chars) == len(intervals):
        for char, (left, right) in zip(chars, intervals):
            if char.isalnum() and len(letters) < 6:
                letters.setdefault(char, _normalized_ink(target[:, left:right]))

    def similarity(observed, reference):
        fitted = cv2.resize(reference, (observed.shape[1], observed.shape[0]), interpolation=cv2.INTER_AREA)
        fitted = _normalized_ink(fitted)
        if fitted.shape != observed.shape:
            fitted = cv2.resize(fitted, (observed.shape[1], observed.shape[0]), interpolation=cv2.INTER_AREA)
        overlap = np.minimum(observed, fitted).sum() / np.maximum(observed, fitted).sum()
        aspect = abs(np.log((reference.shape[1] / reference.shape[0]) / (observed.shape[1] / observed.shape[0])))
        return float(overlap - .15 * aspect)

    best = None
    measured_faces = []
    for face in installed_faces():
        if face[2]:
            continue
        try:
            glyph = _glyph.__wrapped__(face, text)
            if glyph is None:
                continue
            reference = _normalized_ink(np.asarray(glyph, dtype=np.float32))
            if reference is None:
                continue
            size = 128 * target.shape[0] / reference.shape[0]
            measured_faces.append((face, round(size)))
            score = max(similarity(target, reference), similarity(
                compact, reference[:, reference.max(axis=0) > .2],
            ))
            if len(letters) >= 3:
                letter_scores = []
                for char, observed in letters.items():
                    letter = _glyph(face, char)
                    if letter is None:
                        break
                    letter_scores.append(similarity(observed, _normalized_ink(np.asarray(letter, dtype=np.float32))))
                if len(letter_scores) == len(letters):
                    score = max(score, sum(letter_scores) / len(letter_scores))
            if best is None or score > best[0]:
                best = (score, face, size)
        except OSError:
            continue
    if best is not None and best[0] < .75:
        # Small raster text is hinted at its actual size. Downsampling a large
        # reference can reject the right face, so retry at the measured size.
        for face, size in measured_faces:
            try:
                for pixels in range(max(1, size - 1), size + 2):
                    glyph = _glyph.__wrapped__(face, text, pixels)
                    reference = _normalized_ink(np.asarray(glyph, dtype=np.float32))
                    if reference is None:
                        continue
                    score = similarity(target, reference)
                    if score > best[0]:
                        best = (score, face, pixels)
            except OSError:
                continue
    if best is None or best[0] < .75:
        return None
    score, face, size = best
    ys, xs = np.nonzero(contrast > contrast.max() * .2)
    return {'font': face[0], 'bold': face[1], 'font_size_px': size,
            'ink_box': [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]}


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
