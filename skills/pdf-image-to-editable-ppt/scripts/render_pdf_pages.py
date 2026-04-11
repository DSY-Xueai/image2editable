from __future__ import annotations

from pathlib import Path


def render_pdf_pages(input_path: str, output_dir: Path) -> list[str]:
    """Adapter boundary for PDF-to-image rendering."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return []
