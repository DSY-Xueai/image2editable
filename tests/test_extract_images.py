from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from conftest import load_skill_module


extract_images = load_skill_module("extract_images").extract_images


def test_extract_images_raises_for_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_images(str(tmp_path / "missing.pdf"), page_number=1, output_dir=tmp_path / "images")


def test_extract_images_uses_native_pdf_images_when_available(tmp_path, monkeypatch):
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "images"

    class FakePage:
        def get_images(self, full: bool = True):
            assert full is True
            return [(7, 0, 0, 0, 0, 0, 0, 0)]

    class FakeDoc:
        def __init__(self, path: str):
            self.path = path

        def load_page(self, index: int):
            assert index == 0
            return FakePage()

        def extract_image(self, xref: int):
            assert xref == 7
            image_path = tmp_path / "native.png"
            Image.new("RGB", (20, 10), "red").save(image_path)
            return {"image": image_path.read_bytes(), "ext": "png"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_fitz = types.SimpleNamespace(open=lambda path: FakeDoc(path))
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    result = extract_images(str(input_path), page_number=1, output_dir=output_dir)

    assert len(result) == 1
    assert Path(result[0]["path"]).exists()
    assert result[0]["extractable"] is True
    assert result[0]["confidence"] == 0.99


def test_extract_images_detects_high_confidence_crop_from_raster_image(tmp_path):
    input_path = tmp_path / "poster.png"
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 20, 150, 90), fill="black")
    image.save(input_path)

    output_dir = tmp_path / "crops"
    result = extract_images(str(input_path), page_number=1, output_dir=output_dir)

    assert len(result) == 1
    assert Path(result[0]["path"]).exists()
    assert result[0]["extractable"] is True
    assert result[0]["confidence"] >= 0.9
    assert result[0]["width"] > 0
    assert result[0]["height"] > 0
