from __future__ import annotations

import importlib.util


_DEPENDENCY_PROBES = {
    "pymupdf": "fitz",
    "pillow": "PIL",
    "paddleocr": "paddleocr",
    "python_pptx": "pptx",
}


def probe_dependencies() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(module_name) is not None
        for name, module_name in _DEPENDENCY_PROBES.items()
    }
