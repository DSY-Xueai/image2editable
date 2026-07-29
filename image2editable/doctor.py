from __future__ import annotations

import importlib.util
import shutil
import sys
from typing import Any


def _module_status(name: str) -> tuple[bool, str]:
    try:
        available = importlib.util.find_spec(name) is not None
    except Exception as error:
        return False, f"{name}: {error}"
    return available, name if available else f"{name}: not found"


def _module_check(name: str, *, required: bool = True) -> dict[str, Any]:
    available, detail = _module_status(name)
    return {"ok": available, "required": required, "detail": detail}


def _ocr_check() -> dict[str, Any]:
    paddleocr_ok, paddleocr_detail = _module_status("paddleocr")
    paddle_ok, paddle_detail = _module_status("paddle")
    pytesseract_ok, pytesseract_detail = _module_status("pytesseract")
    tesseract_path = shutil.which("tesseract")
    paddle_route_ok = paddleocr_ok and paddle_ok
    tesseract_route_ok = pytesseract_ok and tesseract_path is not None
    return {
        "ok": paddle_route_ok or tesseract_route_ok,
        "required": True,
        "detail": {
            "paddle": {
                "ok": paddle_route_ok,
                "paddleocr": paddleocr_detail,
                "paddle": paddle_detail,
            },
            "tesseract": {
                "ok": tesseract_route_ok,
                "pytesseract": pytesseract_detail,
                "binary": tesseract_path or "tesseract: not found",
            },
        },
    }


def check_environment() -> dict[str, Any]:
    checks = {
        "python": {
            "ok": (3, 10) <= sys.version_info[:2] < (3, 13),
            "required": True,
            "detail": sys.version.split()[0],
        },
        "python-pptx": _module_check("pptx"),
        "opencv": _module_check("cv2"),
        "pillow": _module_check("PIL"),
        "numpy": _module_check("numpy"),
        "pdfium": _module_check("pypdfium2"),
        "transformers": _module_check("transformers"),
        "sam2": _module_check("sam2"),
        "lama": _module_check("simple_lama_inpainting"),
        "ocr": _ocr_check(),
        "aspose-psd": _module_check("aspose.psd", required=False),
    }
    return {
        "ready": all(
            check["ok"] for check in checks.values() if check["required"]
        ),
        "checks": checks,
    }
