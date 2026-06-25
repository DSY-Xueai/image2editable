#!/usr/bin/env python3
"""Text detection module — OCR-based text extraction with style estimation.

Uses PaddleOCR (preferred) or pytesseract (fallback) to detect text regions,
then estimates font size, color, bold, and alignment from the image.

Usage:
    from text_detect import detect_text
    text_items, text_mask = detect_text("slide.png")
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import re

logger = logging.getLogger(__name__)

# Characters considered "noise" — lines consisting only of these are filtered
_NOISE_PATTERN = re.compile(r'^[\s\-_=.|/\\:;,!?~`@#$%^&*(){}\[\]<>+\'\"]+$')

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_text(
    image_path: str | Path,
    lang: str = "ch",
    confidence_threshold: float = 0.7,
    mask_padding: int = 6,
) -> tuple[list[dict], np.ndarray]:
    """Detect text regions and estimate styling.

    Args:
        image_path: Path to the input image.
        lang: OCR language ("ch" for PaddleOCR, "chi_sim+eng" for Tesseract).
        confidence_threshold: Minimum confidence to keep a detection.
        mask_padding: Pixels to pad around each text bbox in the mask.

    Returns:
        text_items: List of dicts with keys:
            box (x, y, w, h), text, font_size, color, bold, font, align, confidence
        text_mask: Binary mask (H, W) uint8 where text regions = 255.
    """
    image_path = Path(image_path)
    img_rgb = _load_rgb(image_path)
    h, w = img_rgb.shape[:2]

    raw_boxes = _ocr_detect(image_path, lang, confidence_threshold)

    if not raw_boxes:
        logger.warning("No text detected by OCR.")
        return [], np.zeros((h, w), dtype=np.uint8)

    # Filter out noise lines (pure symbols, very short, etc.)
    raw_boxes = _filter_noise(raw_boxes)

    # Clean up text (strip leading/trailing symbols)
    for rb in raw_boxes:
        rb["text"] = rb["text"].strip(" |/\\-_=.,:;!?~`'\"")

    # Remove boxes that became empty after cleanup
    raw_boxes = [rb for rb in raw_boxes if rb["text"]]

    if not raw_boxes:
        logger.warning("All OCR detections filtered as noise.")
        return [], np.zeros((h, w), dtype=np.uint8)

    # Estimate styling for each detection
    text_items = []
    for rb in raw_boxes:
        box = rb["box"]  # (x, y, w, h) in pixels
        style = _estimate_style(img_rgb, box)

        # Filter out tiny text (likely decorative labels, icon text, noise)
        if style["font_size"] < 8.0:
            continue

        text = rb["text"]
        font_size = _adjust_font_size(text, style["font_size"])
        text_items.append({
            "box": list(box),
            "text": text,
            "font_size": font_size,
            "color": style["color"],
            "bold": False if _should_force_regular_weight(text, font_size) else style["bold"],
            "font": _select_font(text, font_size),
            "align": 1,  # default center; refined below
            "confidence": rb["confidence"],
        })

    # Refine alignment by grouping nearby lines
    text_items = _refine_alignment(text_items, w)

    # Build mask
    text_mask = _build_text_mask((h, w), text_items, padding=mask_padding)

    logger.info("Detected %d text regions.", len(text_items))
    return text_items, text_mask


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------


def _ocr_detect(
    image_path: Path, lang: str, conf_threshold: float
) -> list[dict]:
    """Try PaddleOCR first, fall back to pytesseract."""
    results = _try_paddleocr(image_path, lang, conf_threshold)
    if results is not None:
        return results

    results = _try_tesseract(image_path, conf_threshold, lang=lang)
    if results is not None:
        return results

    logger.error("No OCR engine available (tried PaddleOCR, pytesseract).")
    return []


def _try_paddleocr(
    image_path: Path, lang: str, conf_threshold: float
) -> list[dict] | None:
    """Detect text with PaddleOCR. Returns None if unavailable."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.debug("PaddleOCR not installed, skipping.")
        return None

    try:
        # Fix OneDNN bug in PaddlePaddle 3.x on Windows:
        # Default run_mode='mkldnn' triggers a ConvertPirAttribute2RuntimeAttribute
        # error. Force run_mode='paddle' to use plain CPU inference.
        _patch_paddle_mkldnn()

        ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        result = ocr.predict(str(image_path))
        if not result:
            return []

        boxes: list[dict] = []
        for item in result:
            # PaddleOCR v3.5+ returns dict-like OCRResult
            texts = item.get("rec_texts", []) if isinstance(item, dict) else getattr(item, "rec_texts", [])
            scores = item.get("rec_scores", []) if isinstance(item, dict) else getattr(item, "rec_scores", [])
            polys = item.get("dt_polys", []) if isinstance(item, dict) else getattr(item, "dt_polys", [])

            if not texts:
                continue

            for i, text in enumerate(texts):
                conf = float(scores[i]) if i < len(scores) else 0.0
                if conf < conf_threshold:
                    continue
                text = text.strip()
                if not text:
                    continue
                poly = polys[i]
                x1, y1 = poly[0]
                x2, y2 = poly[2]
                bx = int(min(x1, x2))
                by = int(min(y1, y2))
                bw = int(abs(x2 - x1))
                bh = int(abs(y2 - y1))
                if bw < 2 or bh < 2:
                    continue
                boxes.append({
                    "box": (bx, by, bw, bh),
                    "text": text,
                    "confidence": conf,
                })
        return boxes
    except Exception as exc:
        logger.warning("PaddleOCR failed: %s", exc)
        return None


def _patch_paddle_mkldnn() -> None:
    """Patch PaddlePaddle's default engine config to disable mkldnn.

    PaddlePaddle 3.x defaults to run_mode='mkldnn' on CPU, which triggers
    an OneDNN bug (ConvertPirAttribute2RuntimeAttribute) on some Windows
    systems. This patches the config resolver to force run_mode='paddle'.

    Also pre-imports torch before paddle to prevent DLL search path conflicts
    on Windows where paddle's DLL loading can break torch's shm.dll.
    """
    try:
        # Import torch first to prevent DLL path pollution from paddle
        try:
            import torch  # noqa: F401
        except ImportError:
            pass

        import paddlex.inference.models.runners.paddle_static.runner as runner_mod
        _orig_resolve = runner_mod.resolve_paddle_static_engine_config

        def _patched_resolve(model_name, config):
            result = _orig_resolve(model_name, config)
            if result.get("run_mode") == "mkldnn":
                result["run_mode"] = "paddle"
            return result

        # Only patch once
        if not getattr(runner_mod, '_mkldnn_patched', False):
            runner_mod.resolve_paddle_static_engine_config = _patched_resolve
            runner_mod._mkldnn_patched = True
    except Exception:
        pass


def _try_tesseract(
    image_path: Path, conf_threshold: float, lang: str = "ch"
) -> list[dict] | None:
    """Detect text with pytesseract at line level. Returns None if unavailable.

    Groups word-level detections by (block, paragraph, line) to produce
    complete text lines instead of individual characters/words.
    """
    try:
        import pytesseract
    except ImportError:
        logger.debug("pytesseract not installed, skipping.")
        return None

    try:
        # Configure Tesseract path on Windows
        tesseract_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if tesseract_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

        tess_lang = _to_tesseract_lang(lang)
        img = Image.open(image_path)
        data = pytesseract.image_to_data(
            img, lang=tess_lang, output_type=pytesseract.Output.DICT
        )

        # Group words by (block, paragraph, line)
        lines: dict[tuple, list[int]] = {}
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if key not in lines:
                lines[key] = []
            lines[key].append(i)

        boxes: list[dict] = []
        for key, indices in lines.items():
            # Merge all words in this line
            texts = []
            confs = []
            x_min, y_min = float("inf"), float("inf")
            x_max, y_max = 0, 0

            for i in indices:
                word = data["text"][i].strip()
                if not word:
                    continue
                texts.append(word)
                conf = float(data["conf"][i])
                if conf >= 0:
                    confs.append(conf)

                wx = int(data["left"][i])
                wy = int(data["top"][i])
                ww = int(data["width"][i])
                wh = int(data["height"][i])
                x_min = min(x_min, wx)
                y_min = min(y_min, wy)
                x_max = max(x_max, wx + ww)
                y_max = max(y_max, wy + wh)

            line_text = "".join(texts)
            if not line_text:
                continue

            avg_conf = sum(confs) / len(confs) if confs else 0
            if avg_conf < conf_threshold * 100:
                continue

            bw = x_max - x_min
            bh = y_max - y_min
            if bw < 2 or bh < 2:
                continue

            boxes.append({
                "box": (int(x_min), int(y_min), int(bw), int(bh)),
                "text": line_text,
                "confidence": avg_conf / 100.0,
            })

        return boxes
    except Exception as exc:
        logger.warning("pytesseract failed: %s", exc)
        return None


def _to_tesseract_lang(lang: str) -> str:
    """Map public OCR language names to Tesseract language packs."""
    if lang in {"ch", "zh", "cn"}:
        return "chi_sim+eng"
    if lang == "en":
        return "eng"
    return lang


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------


def _filter_noise(boxes: list[dict]) -> list[dict]:
    """Filter out OCR detections that are likely noise.

    Removes:
    - Lines consisting only of punctuation/symbols
    - Lines where most characters are symbols/noise
    - Very short meaningless detections
    """
    filtered = []
    for b in boxes:
        text = b["text"].strip()

        # Skip empty
        if not text:
            continue

        # Skip pure symbol/punctuation lines
        if _NOISE_PATTERN.match(text):
            continue

        if _is_likely_vertical_decorative_fragment(b):
            continue

        # Count meaningful characters (letters, digits, CJK)
        meaningful = sum(
            1 for c in text
            if c.isalnum() or '\u4e00' <= c <= '\u9fff'  # CJK unified
            or '\u3400' <= c <= '\u4dbf'  # CJK extension A
        )
        total = len(text.replace(" ", ""))

        # If less than 60% of characters are meaningful, it's likely noise
        if total > 0 and meaningful / total < 0.6:
            continue

        # Skip single-char lines that are common OCR artifacts
        if len(text) == 1 and not text.isalnum() and not ('\u4e00' <= text <= '\u9fff'):
            continue

        # Skip garbled text: mostly uppercase with separators, e.g.
        # e.g. "MCOULE ST:SETMP", "NOOOLE SX.TEET"
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) >= 4:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
            has_garbled_separator = any(c in text for c in ":;./\\")
            if upper_ratio > 0.8 and has_garbled_separator and not has_cjk:
                continue

        filtered.append(b)

    return filtered


def _is_likely_vertical_decorative_fragment(box: dict) -> bool:
    """Identify OCR fragments from large vertical/decorative background text."""
    text = box["text"].strip()
    x, y, w, h = box.get("box", (0, 0, 0, 0))
    if w <= 0 or h <= 0:
        return False

    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
    has_latin_or_digit = any(c.isascii() and c.isalnum() for c in text)

    if h / w >= 1.8 and w <= 36 and (has_cjk or has_latin_or_digit):
        return True
    if has_cjk and len(text) == 1 and h >= 120 and w >= 80:
        return True
    return False


# ---------------------------------------------------------------------------
# Style estimation
# ---------------------------------------------------------------------------


def _estimate_style(img_rgb: np.ndarray, box: tuple) -> dict:
    """Estimate font_size, color, bold from the image region."""
    x, y, w, h = box
    ih, iw = img_rgb.shape[:2]

    # Clamp
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(iw, x + w)
    y2 = min(ih, y + h)

    region = img_rgb[y1:y2, x1:x2]
    if region.size == 0:
        return {"font_size": 12.0, "color": "#000000", "bold": False}

    # --- Font size estimation ---
    # The bbox height in pixels corresponds to the text line height.
    # To convert to PowerPoint points:
    #   slide_width_inches = 13.333 (our PPTX slide width)
    #   pixels_per_inch = image_width / slide_width_inches
    #   bbox_height_inches = bbox_height_px / pixels_per_inch
    #   font_size_pt = bbox_height_inches * 72
    # Apply a correction factor: OCR bboxes include padding around text,
    # and larger text tends to have proportionally more padding.
    pixels_per_inch = iw / 13.333
    bbox_inches = h / pixels_per_inch
    raw_pt = bbox_inches * 72.0

    # Non-linear correction: larger bboxes have more relative padding
    # Correction ranges from ~0.75 for small text to ~0.65 for large text
    correction = 0.75 - 0.001 * min(raw_pt, 100)
    font_size = raw_pt * correction
    font_size = max(6.0, min(font_size, 200.0))

    # --- Color estimation ---
    color_hex = _sample_text_color(region)

    # --- Bold estimation ---
    bold = _estimate_bold(region)

    return {"font_size": round(font_size, 1), "color": color_hex, "bold": bold}


def _select_font(text: str, font_size: float) -> str:
    """Choose an editable font that better matches common Chinese slide styles."""
    if _has_cjk(text) and font_size >= 36.0:
        return "华文行楷"
    return "Microsoft YaHei"


def _adjust_font_size(text: str, font_size: float) -> float:
    """Constrain large Chinese title text so editable text does not wrap."""
    if _has_cjk(text) and font_size >= 80.0:
        return round(font_size * 0.88, 1)
    if _has_cjk(text) and font_size >= 48.0:
        return round(font_size * 0.90, 1)
    return font_size


def _should_force_regular_weight(text: str, font_size: float) -> bool:
    """Avoid synthetic bold on calligraphy fonts."""
    return _has_cjk(text) and font_size >= 36.0


def _has_cjk(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def _sample_text_color(region: np.ndarray) -> str:
    """Sample the dominant text (foreground) color in a text region.

    Uses Otsu thresholding to separate text from background, then uses
    border pixels to determine which class is background. This handles
    both dark-on-light and light-on-dark text correctly.
    """
    if region.size == 0 or region.shape[0] < 3 or region.shape[1] < 3:
        return "#000000"

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Otsu threshold to separate two classes (text vs background)
    thresh_val, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Use border pixels to determine which class is background
    # Border pixels are more likely to be background than text
    border_vals = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ]).astype(np.float32)
    border_mean = float(np.mean(border_vals))

    flat = region.reshape(-1, 3).astype(np.float32)
    gray_flat = gray.reshape(-1).astype(np.float32)

    if border_mean > thresh_val:
        # Border is bright → background is bright → text is dark class
        text_pixels = flat[gray_flat <= thresh_val]
    else:
        # Border is dark → background is dark → text is bright class
        text_pixels = flat[gray_flat > thresh_val]

    if len(text_pixels) < 3:
        # Fallback: use pixels most different from border
        bg_color = np.median(
            flat[np.argsort(np.abs(gray_flat - border_mean))[:max(1, len(flat)//3)]],
            axis=0,
        )
        dists = np.linalg.norm(flat - bg_color, axis=1)
        text_pixels = flat[dists > np.percentile(dists, 60)]

    if len(text_pixels) == 0:
        return "#000000"

    median_color = np.median(text_pixels, axis=0).astype(int)
    r, g, b = np.clip(median_color, 0, 255)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _estimate_bold(region: np.ndarray) -> bool:
    """Estimate whether text is bold by measuring ink density.

    Bold text has thicker strokes, resulting in higher ink-to-background ratio.
    Uses Otsu binarization for robust foreground/background separation.
    """
    if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
        return False

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    # Otsu threshold for robust binarization
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Ink density: ratio of foreground pixels to total
    ink_ratio = np.count_nonzero(binary) / binary.size

    # Bold Chinese/English text typically has ink_ratio > 0.20
    # Regular text typically has ink_ratio 0.10-0.18
    return ink_ratio > 0.20


# ---------------------------------------------------------------------------
# Alignment refinement
# ---------------------------------------------------------------------------


def _refine_alignment(text_items: list[dict], img_width: int) -> list[dict]:
    """Refine text alignment by analyzing horizontal positions.

    Wide text (>= 50% of image width) near the image center → center aligned
    with full-width text box for proper PowerPoint centering.

    Narrow text (< 50% of image width) → placed at detected position using
    left/right alignment based on which side of the image it's on.
    This handles column layouts where text is left-aligned within a column.
    """
    for item in text_items:
        x, y, w, h = item["box"]
        center_x = x + w / 2
        img_center = img_width / 2

        is_wide = w >= img_width * 0.5
        # Tight center check: any text very close to image center
        is_near_center = abs(center_x - img_center) < img_width * 0.05

        if is_near_center:
            # Any text (narrow or wide) very close to center → center
            item["align"] = 1
        elif is_wide and abs(center_x - img_center) < img_width * 0.15:
            # Wide text near center → full-width centered box
            item["align"] = 1
        else:
            # Non-centered text: position at detected location, left-aligned
            # The text box is placed at the OCR-detected coordinates,
            # so left alignment within the box matches the original layout.
            item["align"] = 0

    return text_items


# ---------------------------------------------------------------------------
# Mask building
# ---------------------------------------------------------------------------


def _build_text_mask(
    shape: tuple, text_items: list[dict], padding: int = 6
) -> np.ndarray:
    """Build a binary mask covering all text bounding boxes."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for item in text_items:
        x, y, bw, bh = item["box"]
        x1 = max(0, int(x - padding))
        y1 = max(0, int(y - padding))
        x2 = min(w, int(x + bw + padding))
        y2 = min(h, int(y + bh + padding))
        mask[y1:y2, x1:x2] = 255

    return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_rgb(path: Path) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
