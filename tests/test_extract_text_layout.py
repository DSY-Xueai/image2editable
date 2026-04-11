from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from conftest import load_skill_module


extract_text_layout = load_skill_module("extract_text_layout").extract_text_layout


def test_extract_text_layout_raises_for_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text_layout(str(tmp_path / "missing.pdf"), page_number=1)


def test_extract_text_layout_prefers_native_pdf_text(tmp_path, monkeypatch):
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF-1.4")

    class FakePage:
        def get_text(self, mode: str = "dict"):
            assert mode == "dict"
            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": [10.0, 20.0, 110.0, 60.0],
                        "lines": [
                            {
                                "spans": [
                                    {
                                        "text": "Hello world",
                                        "size": 18.0,
                                        "color": 0x112233,
                                    }
                                ]
                            }
                        ],
                        "align": 1,
                    }
                ]
            }

    class FakeDoc:
        def __init__(self, path: str):
            self.path = path

        def load_page(self, index: int):
            assert index == 0
            return FakePage()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_fitz = types.SimpleNamespace(open=lambda path: FakeDoc(path))
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    result = extract_text_layout(str(input_path), page_number=1)

    assert len(result) == 1
    assert result[0]["text"] == "Hello world"
    assert result[0]["alignment"] == "center"
    assert result[0]["color"] == "#112233"
    assert result[0]["confidence"] == 0.99


def test_extract_text_layout_falls_back_to_ocr_for_images(tmp_path, monkeypatch):
    input_path = tmp_path / "sample.png"
    input_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    class FakeImage:
        size = (200, 100)

    monkeypatch.setattr(
        "pytesseract.image_to_data",
        lambda image, output_type=None, config=None: {
            "text": ["Hello", ""],
            "left": [12, 0],
            "top": [34, 0],
            "width": [56, 0],
            "height": [20, 0],
            "conf": ["87", "-1"],
        },
    )
    monkeypatch.setattr("PIL.Image.open", lambda path: FakeImage())

    result = extract_text_layout(str(input_path), page_number=1)

    assert len(result) == 1
    assert result[0]["text"] == "Hello"
    assert result[0]["confidence"] == pytest.approx(0.87)
    assert result[0]["font_size"] == 20.0
