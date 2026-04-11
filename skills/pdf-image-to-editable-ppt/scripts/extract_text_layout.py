from __future__ import annotations

from pathlib import Path


_ALIGNMENTS = {0: "left", 1: "center", 2: "right", 3: "justify"}


def _color_to_hex(value: int | None) -> str:
    if value is None:
        return "#000000"
    return f"#{value:06x}"


def _normalize_native_blocks(page_dict: dict) -> list[dict]:
    blocks: list[dict] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox", [0.0, 0.0, 0.0, 0.0])
        spans = [
            span
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]
        if not spans:
            continue
        text = " ".join(span["text"].strip() for span in spans)
        blocks.append(
            {
                "text": text,
                "left": float(bbox[0]),
                "top": float(bbox[1]),
                "width": float(bbox[2] - bbox[0]),
                "height": float(bbox[3] - bbox[1]),
                "font_size": float(spans[0].get("size", bbox[3] - bbox[1])),
                "color": _color_to_hex(spans[0].get("color")),
                "alignment": _ALIGNMENTS.get(block.get("align", 0), "left"),
                "confidence": 0.99,
            }
        )
    return blocks


def _extract_with_pytesseract(image) -> list[dict]:
    try:
        import pytesseract
    except ImportError:
        return []

    try:
        data = pytesseract.image_to_data(image, output_type=None, config="--psm 6")
    except Exception:
        return []
    text_items: list[dict] = []
    for text, left, top, width, height, conf in zip(
        data.get("text", []),
        data.get("left", []),
        data.get("top", []),
        data.get("width", []),
        data.get("height", []),
        data.get("conf", []),
    ):
        content = str(text).strip()
        confidence = float(conf) / 100.0 if str(conf).strip() not in {"", "-1"} else -1.0
        if not content or confidence < 0:
            continue
        text_items.append(
            {
                "text": content,
                "left": float(left),
                "top": float(top),
                "width": float(width),
                "height": float(height),
                "font_size": float(height),
                "color": "#000000",
                "alignment": "left",
                "confidence": confidence,
            }
        )
    return text_items


def _extract_with_ocr(image) -> list[dict]:
    pytesseract_items = _extract_with_pytesseract(image)
    if pytesseract_items:
        return pytesseract_items

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return []

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    results = ocr.ocr(image)
    text_items: list[dict] = []
    for line in results or []:
        for bbox, payload in line:
            text, confidence = payload
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            text_items.append(
                {
                    "text": text,
                    "left": float(min(xs)),
                    "top": float(min(ys)),
                    "width": float(max(xs) - min(xs)),
                    "height": float(max(ys) - min(ys)),
                    "font_size": float(max(ys) - min(ys)),
                    "color": "#000000",
                    "alignment": "left",
                    "confidence": float(confidence),
                }
            )
    return text_items

def extract_text_layout(input_path: str, *, page_number: int) -> list[dict]:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz
        except ImportError:
            return []

        with fitz.open(str(source)) as document:
            page = document.load_page(page_number - 1)
            blocks = _normalize_native_blocks(page.get_text("dict"))
            if blocks:
                return blocks
        return []

    from PIL import Image

    image = Image.open(source)
    try:
        return _extract_with_ocr(image)
    finally:
        close = getattr(image, "close", None)
        if callable(close):
            close()
