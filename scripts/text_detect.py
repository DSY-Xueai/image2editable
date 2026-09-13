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
import json
from functools import lru_cache
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import re

_MODULE_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _MODULE_ROOT:
    while _MODULE_ROOT in sys.path:
        sys.path.remove(_MODULE_ROOT)
    sys.path.insert(0, _MODULE_ROOT)

from scripts.worker_resources import run_isolated_worker
from scripts.ocr_worker import _validated_words, _words_from_polys

logger = logging.getLogger(__name__)

_PADDLE_OCR_ENGINES: dict[str, object] = {}

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
    *,
    isolated: bool = False,
    worker_root: str | Path | None = None,
    style_reference_width: int | None = None,
    worker_pool=None,
    performance_trace=None,
    page_id: str | None = None,
    recover_empty: bool = False,
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

    raw_boxes = _ocr_detect(
        image_path,
        lang,
        confidence_threshold,
        isolated=isolated,
        worker_root=worker_root,
        worker_pool=worker_pool,
        performance_trace=performance_trace,
        page_id=page_id,
        **({"recover_empty": True} if recover_empty else {}),
    )

    return _build_text_result(
        img_rgb,
        raw_boxes,
        confidence_threshold,
        mask_padding,
        style_reference_width=style_reference_width,
    )


def detect_text_batch(
    image_paths: list[str | Path],
    lang: str = "ch",
    confidence_threshold: float = 0.7,
    mask_padding: int = 6,
    *,
    isolated: bool = False,
    worker_root: str | Path | None = None,
    worker_pool=None,
    performance_trace=None,
    page_id: str | None = None,
    recover_empty: bool = False,
) -> list[tuple[list[dict], np.ndarray]]:
    """Detect text in several images while sharing one isolated OCR lifecycle."""
    paths = [Path(path) for path in image_paths]
    if not paths:
        return []
    if not isolated:
        return [
            detect_text(
                path, lang=lang, confidence_threshold=confidence_threshold,
                mask_padding=mask_padding,
                **({"recover_empty": True} if recover_empty else {}),
            )
            for path in paths
        ]
    raw_results = _try_isolated_paddleocr_batch(
        paths,
        lang,
        confidence_threshold,
        worker_root=worker_root,
        worker_pool=worker_pool,
        performance_trace=performance_trace,
        page_id=page_id,
        **({"recover_empty": True} if recover_empty else {}),
    )
    if raw_results is None:
        return [
            detect_text(
                path, lang=lang, confidence_threshold=confidence_threshold,
                mask_padding=mask_padding,
                isolated=True,
                worker_root=worker_root,
                worker_pool=worker_pool,
                performance_trace=performance_trace,
                page_id=page_id,
                **({"recover_empty": True} if recover_empty else {}),
            )
            for path in paths
        ]
    return [
        _build_text_result(
            _load_rgb(path), raw_boxes, confidence_threshold, mask_padding,
        )
        for path, raw_boxes in zip(paths, raw_results)
    ]


def _build_text_result(
    img_rgb: np.ndarray,
    raw_boxes: list[dict],
    confidence_threshold: float,
    mask_padding: int,
    *,
    style_reference_width: int | None = None,
) -> tuple[list[dict], np.ndarray]:
    if style_reference_width is not None and (
        type(style_reference_width) is not int or style_reference_width <= 0
    ):
        raise ValueError("reference_width must be a positive integer")
    h, w = img_rgb.shape[:2]
    if not raw_boxes:
        logger.warning("No text detected by OCR.")
        return [], np.zeros((h, w), dtype=np.uint8)

    # Filter out noise lines (pure symbols, very short, etc.)
    raw_boxes = _filter_noise(raw_boxes, confidence_threshold)

    # Clean up OCR edge noise while preserving semantic sentence endings.
    for rb in raw_boxes:
        original_box = tuple(rb["box"])
        text = rb["text"].strip()
        if (
            len(text) == 1 and text in "?!" and rb.get("confidence", 0) >= .9
        ) or re.fullmatch(r"[+-]\d+(?:[.,]\d+)?%?", text) or (
            rb.get("confidence", 0) >= .99 and text.endswith(("/", "\\"))
        ):
            rb["text"] = text
        else:
            rb["text"] = text.lstrip("|/\\-_=.,:;!?~`'\"").rstrip("|/\\-_=,:;~`'\"")
        rb["text"], rb["box"] = _recover_trailing_heading_period(
            img_rgb,
            rb["text"],
            rb["box"],
        )

        if tuple(rb["box"]) != original_box:
            rb.pop("words", None)

    # Remove boxes that became empty after cleanup
    raw_boxes = [rb for rb in raw_boxes if rb["text"]]

    if not raw_boxes:
        logger.warning("All OCR detections filtered as noise.")
        return [], np.zeros((h, w), dtype=np.uint8)

    # Estimate styling for each detection
    text_items = []
    for rb in raw_boxes:
        box = rb["box"]  # (x, y, w, h) in pixels
        text = rb["text"]
        if style_reference_width is None:
            style = _estimate_style(img_rgb, box, text=text)
        else:
            style = _estimate_style(
                img_rgb,
                box,
                reference_width=style_reference_width,
                text=text,
            )

        # Retain confident small labels; size alone is not evidence of noise.
        if style["font_size"] < 8.0 and rb["confidence"] < 0.9:
            continue

        font_size = _adjust_font_size(
            text,
            style["font_size"],
            bbox_height=box[3],
            reference_width=style_reference_width or w,
        )
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

        words = _validated_words(text, rb.get("words"))
        if words:
            text_items[-1]["words"] = words

    text_items = _merge_adjacent_text_items(text_items)

    # Refine alignment by grouping nearby lines
    text_items = _refine_alignment(text_items, w)

    text_items = refine_text_ink_bounds(img_rgb, text_items)

    # Build mask
    text_mask = _build_text_mask((h, w), text_items, padding=mask_padding)

    logger.info("Detected %d text regions.", len(text_items))
    return text_items, text_mask


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------


def _ocr_detect(
    image_path: Path,
    lang: str,
    conf_threshold: float,
    *,
    isolated: bool = False,
    worker_root: str | Path | None = None,
    worker_pool=None,
    performance_trace=None,
    page_id: str | None = None,
    recover_empty: bool = False,
) -> list[dict]:
    """Try PaddleOCR first, fall back to pytesseract."""
    if isolated:
        results = _try_isolated_paddleocr(
            image_path,
            lang,
            conf_threshold,
            worker_root=worker_root,
            worker_pool=worker_pool,
            performance_trace=performance_trace,
            page_id=page_id,
            **({"recover_empty": True} if recover_empty else {}),
        )
    else:
        results = _try_paddleocr(
            image_path, lang, conf_threshold,
            **({"recover_empty": True} if recover_empty else {}),
        )
    if results is not None:
        return results

    results = _try_tesseract(image_path, conf_threshold, lang=lang)
    if results is not None:
        return results

    logger.error("No OCR engine available (tried PaddleOCR, pytesseract).")
    return []


def _try_isolated_paddleocr(
    image_path: Path,
    lang: str,
    conf_threshold: float,
    *,
    worker_root: str | Path | None,
    worker_pool=None,
    performance_trace=None,
    page_id: str | None = None,
    recover_empty: bool = False,
) -> list[dict] | None:
    if worker_pool is not None:
        results = _try_isolated_paddleocr_batch(
            [image_path],
            lang,
            conf_threshold,
            worker_root=worker_root,
            worker_pool=worker_pool,
            performance_trace=performance_trace,
            page_id=page_id,
            **({"recover_empty": True} if recover_empty else {}),
        )
        return None if results is None else results[0]
    try:
        with tempfile.TemporaryDirectory(
            prefix="ocr-",
            dir=worker_root,
        ) as temporary:
            work_dir = Path(temporary)
            detection_result = work_dir / "detection.json"
            recognition_result = work_dir / "recognition.json"
            commands = [
                [
                    sys.executable,
                    str(Path(__file__).with_name("ocr_worker.py").resolve()),
                    "detect",
                    "--image",
                    str(image_path),
                    "--work-dir",
                    str(work_dir),
                    "--result",
                    str(detection_result),
                ],
                [
                    sys.executable,
                    str(Path(__file__).with_name("ocr_worker.py").resolve()),
                    "recognize",
                    "--detection-result",
                    str(detection_result),
                    "--result",
                    str(recognition_result),
                    "--lang",
                    lang,
                ],
            ]
            for stage, command, result_path in (
                ("detection", commands[0], detection_result),
                ("recognition", commands[1], recognition_result),
            ):
                if recover_empty and stage == "detection":
                    command.append("--recover-empty")
                completed = run_isolated_worker(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode or not result_path.is_file():
                    diagnostic = completed.stderr.strip()
                    logger.warning(
                        "Isolated OCR %s failed (exit=%s): %s",
                        stage,
                        completed.returncode,
                        diagnostic or f"missing result {result_path}",
                    )
                    return None
            payload = json.loads(
                recognition_result.read_text(encoding="utf-8")
            )
    except Exception as error:
        logger.warning("Isolated OCR failed: %s", error)
        return None

    boxes = []
    for item in payload.get("items", []):
        confidence = float(item.get("score", 0.0))
        text = str(item.get("text", "")).strip()
        if confidence < conf_threshold or not text:
            continue
        poly = item["poly"]
        bx, by, width, height = _poly_to_box(poly)
        if width < 2 or height < 2:
            continue
        boxes.append(
            {
                "box": (bx, by, width, height),
                "text": text,
                "confidence": confidence,
            }
        )
        words = _validated_words(text, item.get("words"))
        if words:
            boxes[-1]["words"] = words
    return boxes


def _try_isolated_paddleocr_batch(
    image_paths: list[Path],
    lang: str,
    conf_threshold: float,
    *,
    worker_root: str | Path | None,
    worker_pool=None,
    performance_trace=None,
    page_id: str | None = None,
    recover_empty: bool = False,
    recognition_only: bool = False,
) -> list[list[dict]] | None:
    try:
        with tempfile.TemporaryDirectory(
            prefix="ocr-batch-", dir=worker_root,
        ) as temporary:
            work_dir = Path(temporary)
            result_path = work_dir / "result.json"
            resolved_paths = [str(path.resolve()) for path in image_paths]
            if worker_pool is not None:
                worker_pool.request(
                    {"images": resolved_paths, "result": str(result_path), "lang": lang,
                     **({"recover_empty": True} if recover_empty else {}),
                     **({"recognition_only": True} if recognition_only else {})},
                    performance_trace=performance_trace,
                    page_id=page_id,
                )
            else:
                manifest_path = work_dir / "manifest.json"
                manifest_path.write_text(
                    json.dumps({"images": resolved_paths}), encoding="utf-8",
                )
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("ocr_worker.py").resolve()),
                    "batch",
                    "--manifest", str(manifest_path),
                    "--result", str(result_path),
                    "--lang", lang,
                ]
                if recover_empty:
                    command.append("--recover-empty")
                if recognition_only:
                    command.append("--recognition-only")
                completed = run_isolated_worker(
                    command, capture_output=True, text=True, check=False,
                )
                if completed.returncode or not result_path.is_file():
                    logger.warning(
                        "Isolated OCR batch failed (exit=%s): %s",
                        completed.returncode,
                        completed.stderr.strip() or f"missing result {result_path}",
                    )
                    return None
            if not result_path.is_file():
                logger.warning("Isolated OCR batch did not create its result")
                return None
            payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.warning("Isolated OCR batch failed: %s", error)
        return None

    images = payload.get("images", [])
    if len(images) != len(image_paths):
        logger.warning("Isolated OCR batch returned the wrong image count")
        return None
    results = []
    for image in images:
        boxes = []
        for item in image.get("items", []):
            confidence = float(item.get("score", 0.0))
            text = str(item.get("text", "")).strip()
            if confidence < conf_threshold or not text:
                continue
            bx, by, width, height = _poly_to_box(item["poly"])
            if width < 2 or height < 2:
                continue
            boxes.append({
                "box": (bx, by, width, height),
                "text": text,
                "confidence": confidence,
            })
            words = _validated_words(text, item.get("words"))
            if words:
                boxes[-1]["words"] = words
        results.append(boxes)
    return results


def _poly_to_box(poly: object) -> tuple[int, int, int, int]:
    x_values = [point[0] for point in poly]
    y_values = [point[1] for point in poly]
    x1, x2 = min(x_values), max(x_values)
    y1, y2 = min(y_values), max(y_values)
    return (
        int(x1),
        int(y1),
        int(x2 - x1),
        int(y2 - y1),
    )


def _try_paddleocr(
    image_path: Path, lang: str, conf_threshold: float,
    *, recover_empty: bool = False,
) -> list[dict] | None:
    """Detect text with PaddleOCR. Returns None if unavailable."""
    try:
        ocr = _get_paddleocr(lang)
    except ImportError:
        logger.debug("PaddleOCR not installed, skipping.")
        return None
    except Exception as exc:
        logger.warning("PaddleOCR failed: %s", exc)
        return None

    try:
        result = list(ocr.predict(str(image_path), return_word_box=True))
        if recover_empty and not any(
            item.get("rec_texts", []) if isinstance(item, dict)
            else getattr(item, "rec_texts", [])
            for item in result
        ):
            result = list(ocr.predict(
                str(image_path), text_det_thresh=0.15, text_det_box_thresh=0.3,
                return_word_box=True,
            ))
            conf_threshold = max(conf_threshold, 0.9)
        if not result:
            return []

        boxes: list[dict] = []
        for item in result:
            # PaddleOCR v3.5+ returns dict-like OCRResult
            texts = item.get("rec_texts", []) if isinstance(item, dict) else getattr(item, "rec_texts", [])
            scores = item.get("rec_scores", []) if isinstance(item, dict) else getattr(item, "rec_scores", [])
            polys = item.get("rec_polys", item.get("dt_polys", [])) if isinstance(item, dict) else getattr(item, "rec_polys", getattr(item, "dt_polys", []))

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
                bx, by, bw, bh = _poly_to_box(poly)
                if bw < 2 or bh < 2:
                    continue
                boxes.append({
                    "box": (bx, by, bw, bh),
                    "text": text,
                    "confidence": conf,
                })
                tokens = item.get("text_word", []) if isinstance(item, dict) else getattr(item, "text_word", [])
                regions = item.get("text_word_region", []) if isinstance(item, dict) else getattr(item, "text_word_region", [])
                if i < len(tokens) and i < len(regions):
                    words = _words_from_polys(text, tokens[i], regions[i], (bx, by, bw, bh))
                    if words:
                        boxes[-1]["words"] = words
        return boxes
    except Exception as exc:
        logger.warning("PaddleOCR failed: %s", exc)
        return None


def _create_paddleocr(lang: str) -> object:
    from paddleocr import PaddleOCR

    _patch_paddle_mkldnn()
    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_batch_size=1,
        cpu_threads=1,
        enable_mkldnn=False,
    )


def _get_paddleocr(lang: str) -> object:
    if lang not in _PADDLE_OCR_ENGINES:
        _PADDLE_OCR_ENGINES[lang] = _create_paddleocr(lang)
    return _PADDLE_OCR_ENGINES[lang]


def close_ocr_engines() -> None:
    _PADDLE_OCR_ENGINES.clear()


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


def _is_spaced_semantic_separator_text(text: str) -> bool:
    parts = re.split(r"\s+[/\-]\s+", text)
    if len(parts) == 1:
        return False

    for part in parts:
        meaningful = 0
        for index, char in enumerate(part):
            if (
                char.isalnum()
                or "\u4e00" <= char <= "\u9fff"
                or "\u3400" <= char <= "\u4dbf"
            ):
                meaningful += 1
                continue
            if char.isspace():
                continue
            if (
                char in ".-"
                and index > 0
                and index + 1 < len(part)
                and part[index - 1].isalnum()
                and part[index + 1].isalnum()
            ):
                continue
            return False
        if meaningful < 2:
            return False
    return True


def _filter_noise(
    boxes: list[dict], confidence_threshold: float = 0.7
) -> list[dict]:
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

        if float(b.get("confidence", 0.0)) < confidence_threshold:
            continue

        preserve_symbol = len(text) == 1 and text in "?!" and float(b.get("confidence", 0.0)) >= .9
        if preserve_symbol:
            filtered.append(b)
            continue
        if re.fullmatch(r"[+-]\d+(?:[.,]\d+)?%?", text):
            filtered.append(b)
            continue
        if _NOISE_PATTERN.match(text) and not preserve_symbol:
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

        # Paired brackets delimit labels; they are not noise in the label content.
        if meaningful and text[0] + text[-1] in (
            "[]", "()", "{}", "\uff08\uff09", "\u3010\u3011",
            "\u3014\u3015", "\u3008\u3009", "\u300a\u300b",
        ):
            total -= 2

        # If less than 60% of characters are meaningful, it's likely noise
        if total > 0 and meaningful / total < 0.6:
            continue

        technical_text = text.rstrip(".!?") or text
        technical_separators = "-_./"
        has_technical_separator = any(c in technical_separators for c in technical_text)
        compact_label = all(
            c.isalnum() or c in technical_separators for c in technical_text
        )
        valid_technical_label = (
            compact_label
            and technical_text[0].isalnum()
            and technical_text[-1].isalnum()
            and not any(
                left in technical_separators and right in technical_separators
                for left, right in zip(technical_text, technical_text[1:])
            )
            and meaningful >= (4 if has_technical_separator else 2)
        )

        if has_technical_separator and compact_label and not valid_technical_label:
            continue

        spaced_semantic_separator = _is_spaced_semantic_separator_text(text)
        if re.search(r"\s[/\-]\s", text) and not spaced_semantic_separator:
            continue

        # Skip single-char lines that are common OCR artifacts
        if len(text) == 1 and not text.isalnum() and not ('\u4e00' <= text <= '\u9fff') and not preserve_symbol:
            continue

        # Skip garbled text: mostly uppercase with separators, e.g.
        # e.g. "MCOULE ST:SETMP", "NOOOLE SX.TEET"
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) >= 4:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
            has_garbled_separator = any(c in text for c in ":;./\\")
            if (
                upper_ratio > 0.8
                and has_garbled_separator
                and not has_cjk
                and not valid_technical_label
                and not spaced_semantic_separator
            ):
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


def refine_text_ink_bounds(img_rgb: np.ndarray, items: list[dict]) -> list[dict]:
    """Include a recognized terminal slash whose ink lies outside its OCR box."""
    result = []
    for item in items:
        result.append(item)
        text = item.get("text", "").rstrip()
        if not text.endswith(("/", "\\")) or item.get("rotation") or item.get("runs"):
            continue
        x, y, width, height = map(int, item["box"])
        right = x + width
        left = max(0, right - height)
        end = min(img_rgb.shape[1], right + height)
        top, bottom = max(0, y), min(img_rgb.shape[0], y + height)
        for other in items:
            if other is item:
                continue
            ox, oy, ow, oh = other["box"]
            if ox >= right and min(bottom, oy + oh) > max(top, oy):
                end = min(end, int(ox))
        color = item.get("color", "")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color) or end <= left or bottom <= top:
            continue
        rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)])
        ink = np.max(np.abs(img_rgb[top:bottom, left:end].astype(np.int16) - rgb), axis=2) <= 24
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        matches = []
        for label in range(1, count):
            bx, by, bw, bh, area = stats[label]
            if not (.12 * height <= bw <= .65 * height and .5 * height <= bh <= height
                    and area >= 6 and bx + bw < end - left):
                continue
            ys, xs = np.nonzero(labels == label)
            slope = float(np.corrcoef(xs, ys)[0, 1])
            if slope * (-1 if text.endswith("/") else 1) > .8:
                matches.append((int(bx), int(bw)))
        if len(matches) == 1:
            bx, bw = matches[0]
            if left + bx >= right and left + bx + bw > right:
                updated = {**item, "box": [x, y, left + bx + bw - x, height]}
                updated.pop("words", None)
                result[-1] = updated
    return result


def _recover_trailing_heading_period(
    img_rgb: np.ndarray,
    text: str,
    box: tuple,
) -> tuple[str, tuple]:
    letters = [char for char in text if char.isalpha()]
    x, y, width, height = (int(value) for value in box)
    if (
        text.endswith((".", "!", "?"))
        or height < 48
        or len(letters) < 4
        or not all(char.isascii() and char.isupper() for char in letters)
    ):
        return text, box

    image_height, image_width = img_rgb.shape[:2]
    right = x + width
    scan_width = min(image_width - right, max(8, int(round(height * 0.45))))
    if scan_width <= 0:
        return text, box
    top = max(0, y)
    bottom = min(image_height, y + height)
    text_region = img_rgb[top:bottom, max(0, x):min(image_width, right)]
    candidate_region = img_rgb[top:bottom, right:right + scan_width]
    if text_region.size == 0 or candidate_region.size == 0:
        return text, box

    color = _sample_text_color(text_region)
    color_rgb = np.array(
        [int(color[index:index + 2], 16) for index in (1, 3, 5)],
        dtype=np.int16,
    )
    color_distance = np.linalg.norm(
        candidate_region.astype(np.int16) - color_rgb,
        axis=2,
    )
    candidate_mask = (color_distance <= 48.0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8,
    )
    matches = []
    for index in range(1, count):
        left = int(stats[index, cv2.CC_STAT_LEFT])
        candidate_top = int(stats[index, cv2.CC_STAT_TOP])
        candidate_width = int(stats[index, cv2.CC_STAT_WIDTH])
        candidate_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        fill_ratio = area / max(1, candidate_width * candidate_height)
        if (
            left <= height * 0.15
            and candidate_top >= height * 0.58
            and height * 0.08 <= candidate_width <= height * 0.30
            and height * 0.08 <= candidate_height <= height * 0.30
            and fill_ratio >= 0.45
            and left + candidate_width < scan_width
        ):
            matches.append((left, candidate_width))
    if len(matches) != 1:
        return text, box
    left, candidate_width = matches[0]
    return f"{text}.", (x, y, width + left + candidate_width, height)


# ---------------------------------------------------------------------------
# Style estimation
# ---------------------------------------------------------------------------


def _estimate_style(
    img_rgb: np.ndarray,
    box: tuple,
    *,
    reference_width: int | None = None,
    text: str = "",
) -> dict:
    """Estimate font_size, color, bold from the image region."""
    if reference_width is not None and (
        type(reference_width) is not int or reference_width <= 0
    ):
        raise ValueError("reference_width must be a positive integer")
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
    pixels_per_inch = (reference_width if reference_width is not None else iw) / 13.333
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
    bold = _estimate_bold(region, text=text)

    # OCR rectangles include variable padding; measure glyphs for known fonts.
    reference_font = _weight_reference_font(_has_cjk(text), bold) if text and "\n" not in text else None
    if reference_font is not None:
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float32)
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        contrast = np.abs(gray - float(np.median(border)))
        foreground = (contrast > float(contrast.max()) * 0.1).astype(np.uint8)
        horizontal = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN,
            np.ones((1, max(13, int(region.shape[1] * 0.8) | 1)), dtype=np.uint8),
        )
        contrast[horizontal > 0] = 0
        rows = np.flatnonzero(np.any(contrast > float(contrast.max()) * 0.5, axis=1))
        bounds = reference_font.getbbox(text)
        glyph_height = bounds[3] - bounds[1]
        if len(rows) and glyph_height > 0:
            size_px = (rows[-1] - rows[0] + 1) * reference_font.size / glyph_height
            font_size = max(6.0, min(size_px * 72.0 / pixels_per_inch, 200.0))

    return {"font_size": round(font_size, 1), "color": color_hex, "bold": bold}


def _select_font(text: str, font_size: float) -> str:
    """Choose an editable font that better matches common Chinese slide styles."""
    return "Microsoft YaHei" if _has_cjk(text) else "Arial"


def refine_plain_text_fonts(image: np.ndarray, items: list[dict]) -> list[dict]:
    from scripts.font_match import match_text_face

    refined = []
    for item in items:
        text = item.get('text', '')
        x, y, width, height = map(int, item['box'])
        if (item.get('runs') or item.get('rotation') or item.get('italic')
                or item.get('outline_width') or item.get('gradient')
                or 'font_size_pt' in item or item.get('box_kind') == 'ink'
                or not 3 <= len(text) <= 128 or '\n' in text
                or x < 0 or y < 0 or width * height > 170000):
            refined.append(item)
            continue
        crop = np.ascontiguousarray(image[y:y+height, x:x+width])
        if crop.shape != (height, width, 3) or not crop.size:
            refined.append(item)
            continue
        match = match_text_face(crop.tobytes(), width, height, text)
        if match is None:
            refined.append(item)
            continue
        left, top, ink_width, ink_height = match['ink_box']
        refined.append({**item, 'font': match['font'], 'bold': match['bold'],
                        'font_size': match['font_size_px'] * 13.333 * 72 / image.shape[1],
                        'box': [x+left, y+top, ink_width, ink_height], 'box_kind': 'ink'})
    return refined


def _adjust_font_size(
    text: str,
    font_size: float,
    *,
    bbox_height: int | None = None,
    reference_width: int | None = None,
) -> float:
    """Constrain large Chinese title text so editable text does not wrap."""
    letters = [char for char in text if char.isalpha()]
    if (
        font_size >= 30.0
        and len(letters) >= 4
        and all(char.isascii() and char.isupper() for char in letters)
        and bbox_height is not None
        and reference_width is not None
    ):
        pixels_per_inch = reference_width / 13.333
        return round(bbox_height / pixels_per_inch * 72.0 * 1.08, 1)
    if _has_cjk(text) and font_size >= 80.0:
        return round(font_size * 0.88, 1)
    if _has_cjk(text) and font_size >= 48.0:
        return round(font_size * 0.90, 1)
    return font_size


def _should_force_regular_weight(text: str, font_size: float) -> bool:
    """Keep the detected weight for the sans-serif fallback fonts."""
    return False


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
    contrast = np.abs(gray.astype(np.float32) - float(np.median(border_vals)))
    foreground = (contrast > float(contrast.max()) * 0.1).astype(np.uint8)
    structure = np.zeros(gray.shape, dtype=np.uint8)
    for shape in ((1, max(13, int(w * 0.8) | 1)),
                  (max(13, int(h * 0.8) | 1), 1)):
        structure |= cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, np.ones(shape, dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
    non_structure = structure.reshape(-1) == 0
    # A narrow glyph can itself resemble a rule; retain the dominant foreground.
    if np.count_nonzero(foreground.reshape(-1) & non_structure) >= (
        0.5 * np.count_nonzero(foreground)
    ):
        flat = flat[non_structure]
        gray_flat = gray_flat[non_structure]

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

    # Ink cores preserve the original color; antialiased edges blend with the background.
    text_luma = text_pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    contrast = np.abs(text_luma - border_mean)
    text_pixels = text_pixels[contrast >= np.percentile(contrast, 90)]
    median_color = np.median(text_pixels, axis=0).astype(int)
    r, g, b = np.clip(median_color, 0, 255)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _normalized_ink(contrast: np.ndarray) -> np.ndarray | None:
    """Keep fractional edge coverage and remove background-only padding."""
    if contrast.size == 0 or float(contrast.max()) == 0:
        return None
    foreground = contrast[contrast > float(contrast.max()) * 0.1]
    ink = np.clip(contrast / float(np.percentile(foreground, 95)), 0, 1)
    ys, xs = np.nonzero(ink > 0.2)
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


@lru_cache(maxsize=4)
def _weight_reference_font(cjk: bool, bold: bool):
    filenames = (
        ("msyhbd.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
        if bold else
        ("msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    ) if cjk else (
        ("arialbd.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf")
        if bold else
        ("arial.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf")
    )
    for filename in filenames:
        try:
            return ImageFont.truetype(filename, 128)
        except OSError:
            continue
    return None


def _estimate_reference_bold(region: np.ndarray, text: str) -> bool | None:
    """Compare the same glyphs in the editable regular and bold fallback fonts."""
    if not text.strip() or region.size == 0:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float32)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    difference = gray - float(np.median(border))
    magnitude = np.abs(difference)
    foreground = difference[magnitude > float(magnitude.max()) * 0.1]
    if foreground.size == 0:
        return None
    polarity = 1 if float(np.median(foreground)) > 0 else -1
    ink = _normalized_ink(np.maximum(difference * polarity, 0))
    if ink is None:
        return None
    height, width = ink.shape
    densities = []
    for bold in (False, True):
        font = _weight_reference_font(_has_cjk(text), bold)
        if font is None:
            return None
        left, top, right, bottom = font.getbbox(text)
        if right <= left or bottom <= top:
            return None
        reference = Image.new("L", (right - left + 8, bottom - top + 8), 0)
        ImageDraw.Draw(reference).text((4 - left, 4 - top), text, font=font, fill=255)
        reference_ink = _normalized_ink(np.asarray(reference, dtype=np.float32))
        if reference_ink is None:
            return None
        # Match raster scale before normalizing so small antialiased glyphs are comparable.
        reference_ink = _normalized_ink(cv2.resize(
            reference_ink, (width, height), interpolation=cv2.INTER_AREA,
        ))
        densities.append(float(reference_ink.mean()))
    observed = float(ink.mean())
    return abs(observed - densities[1]) < abs(observed - densities[0])


def _estimate_bold(region: np.ndarray, *, text: str = "") -> bool:
    """Estimate bold weight from ink density and relative stroke width."""
    if text:
        reference_bold = _estimate_reference_bold(region, text)
        if reference_bold is not None:
            return reference_bold
    if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
        return False

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    threshold, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    border = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ])
    if float(np.mean(border)) > threshold:
        ink = gray <= threshold
    else:
        ink = gray > threshold
    ink_ratio = np.count_nonzero(ink) / ink.size
    stroke_depth = cv2.distanceTransform(
        ink.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    strokes = stroke_depth[stroke_depth > 0]
    if strokes.size == 0:
        return False
    relative_stroke = float(np.percentile(strokes, 90)) / region.shape[0]
    return ink_ratio > 0.20 and relative_stroke >= 0.05


# ---------------------------------------------------------------------------
# Alignment refinement
# ---------------------------------------------------------------------------


def _merge_adjacent_text_items(text_items: list[dict]) -> list[dict]:
    """Merge same-style OCR fragments that belong to one visual line."""
    merged = [dict(item) for item in text_items]

    while True:
        best_pair = None
        best_score = None
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                left, right = sorted((merged[i], merged[j]), key=lambda item: item["box"][0])
                if not _can_merge_text_items(left, right):
                    continue
                gap = right["box"][0] - (left["box"][0] + left["box"][2])
                center_gap = abs(
                    (left["box"][1] + left["box"][3] / 2)
                    - (right["box"][1] + right["box"][3] / 2)
                )
                score = (max(gap, 0), center_gap)
                if best_score is None or score < best_score:
                    best_pair = (i, j, left, right)
                    best_score = score

        if best_pair is None:
            break
        i, j, left, right = best_pair
        for index in sorted((i, j), reverse=True):
            merged.pop(index)
        merged.append(_merge_text_pair(left, right))

    return sorted(merged, key=lambda item: (item["box"][1], item["box"][0]))


def _can_merge_text_items(left: dict, right: dict) -> bool:
    if "runs" in left or "runs" in right:
        return False
    if bool(left.get("words")) != bool(right.get("words")):
        return False
    lx, ly, lw, lh = left["box"]
    rx, ry, rw, rh = right["box"]
    if rx < lx:
        return False

    overlap = max(0, min(ly + lh, ry + rh) - max(ly, ry))
    if overlap / max(1, min(lh, rh)) < 0.60:
        return False

    gap = rx - (lx + lw)
    max_height = max(lh, rh)
    if gap < -0.35 * max_height or gap > max(6, 0.45 * max_height):
        return False

    left_size = float(left.get("font_size", 12))
    right_size = float(right.get("font_size", 12))
    if abs(left_size - right_size) / max(left_size, right_size, 1) > 0.25:
        return False
    if left.get("bold", False) != right.get("bold", False):
        return False
    return _colors_are_close(left.get("color", "#000000"), right.get("color", "#000000"))


def _colors_are_close(left: str, right: str, max_distance: float = 48.0) -> bool:
    try:
        lrgb = np.array([int(left[i:i + 2], 16) for i in (1, 3, 5)])
        rrgb = np.array([int(right[i:i + 2], 16) for i in (1, 3, 5)])
    except (TypeError, ValueError):
        return left == right
    return float(np.linalg.norm(lrgb - rrgb)) <= max_distance


def _merge_text_pair(left: dict, right: dict) -> dict:
    lx, ly, lw, lh = left["box"]
    rx, ry, rw, rh = right["box"]
    left_char = left["text"][-1:]
    right_char = right["text"][:1]
    separator = ""
    if not (
        (_has_cjk(left_char) and _has_cjk(right_char))
        or (left_char.isdigit() and _has_cjk(right_char))
    ):
        separator = " "
    merged = dict(left)
    merged["box"] = [
        min(lx, rx),
        min(ly, ry),
        max(lx + lw, rx + rw) - min(lx, rx),
        max(ly + lh, ry + rh) - min(ly, ry),
    ]
    merged["text"] = left["text"] + separator + right["text"]
    merged["font_size"] = max(float(left.get("font_size", 12)), float(right.get("font_size", 12)))
    merged["font"] = _select_font(merged["text"], merged["font_size"])
    merged["confidence"] = min(float(left.get("confidence", 1)), float(right.get("confidence", 1)))
    merged.pop("words", None)
    if left.get("words") and right.get("words"):
        mx, my, mw, mh = merged["box"]
        words = []
        for item in (left, right):
            x, y, width, height = item["box"]
            for word in _validated_words(item["text"], item["words"]):
                wx, wy, ww, wh = word["box"]
                words.append({"text": word["text"], "box": [(x + wx * width - mx) / mw,
                    (y + wy * height - my) / mh, ww * width / mw, wh * height / mh]})
        words = _validated_words(merged["text"], words)
        if words:
            merged["words"] = words
    return merged


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
