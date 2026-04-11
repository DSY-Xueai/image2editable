from __future__ import annotations

from pathlib import Path


def render_pdf_pages(input_path: str, output_dir: Path) -> list[str]:
    """Render PDF pages to PNG backgrounds when PyMuPDF is available."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import fitz
    except ImportError:
        return []

    rendered_paths: list[str] = []
    with fitz.open(str(source)) as document:
        for index in range(len(document)):
            page = document.load_page(index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            target = output_dir / f"page-{index + 1}.png"
            pixmap.save(str(target))
            rendered_paths.append(str(target))
    return rendered_paths
