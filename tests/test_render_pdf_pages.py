from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from conftest import load_skill_module


render_pdf_pages = load_skill_module("render_pdf_pages").render_pdf_pages


def test_render_pdf_pages_raises_for_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_pdf_pages(str(tmp_path / "missing.pdf"), tmp_path / "pages")


def test_render_pdf_pages_uses_fitz_when_available(tmp_path, monkeypatch):
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "pages"

    class FakePixmap:
        def __init__(self, page_number: int):
            self.page_number = page_number

        def save(self, target: str) -> None:
            Path(target).write_bytes(f"page-{self.page_number}".encode("utf-8"))

    class FakePage:
        def __init__(self, page_number: int):
            self.page_number = page_number

        def get_pixmap(self, matrix=None, alpha=False):
            return FakePixmap(self.page_number)

    class FakeDoc:
        def __init__(self, path: str):
            self.path = path
            self.pages = [FakePage(1), FakePage(2)]

        def __len__(self):
            return len(self.pages)

        def load_page(self, index: int):
            return self.pages[index]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_fitz = types.SimpleNamespace(
        Matrix=lambda x, y: (x, y),
        open=lambda path: FakeDoc(path),
    )
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    rendered = render_pdf_pages(str(input_path), output_dir)

    assert len(rendered) == 2
    assert all(Path(path).exists() for path in rendered)
    assert all(path.endswith(".png") for path in rendered)
