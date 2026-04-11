from __future__ import annotations

from pathlib import Path


def extract_images(input_path: str, *, page_number: int, output_dir: Path) -> list[dict]:
    """Adapter boundary for extracting reusable image assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return []
