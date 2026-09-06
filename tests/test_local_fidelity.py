from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from image2editable.local_fidelity import build_local_fidelity_components


def _save_rgb(path: Path, pixels: np.ndarray) -> Path:
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)
    return path


def _save_mask(path: Path, pixels: np.ndarray) -> Path:
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)
    return path


def test_residual_pixels_become_bounded_transparent_components(
    tmp_path: Path,
) -> None:
    source = np.full((60, 100, 3), 255, dtype=np.uint8)
    source[12:32, 18:42] = (10, 80, 220)
    candidate = np.full_like(source, 255)
    text_mask = np.zeros(source.shape[:2], dtype=np.uint8)

    result = build_local_fidelity_components(
        source_path=_save_rgb(tmp_path / "source.png", source),
        candidate_path=_save_rgb(tmp_path / "candidate.png", candidate),
        reliable_text_mask_path=_save_mask(tmp_path / "text.png", text_mask),
        output_dir=tmp_path / "fidelity",
        max_component_coverage=0.35,
    )

    assert result["residual_pixels"] == 20 * 24
    assert result["uncovered_pixels"] == 0
    assert all(item["coverage"] <= 0.35 for item in result["components"])
    assert all(Path(item["path"]).is_file() for item in result["components"])
    with Image.open(result["components"][0]["path"]) as patch:
        assert patch.mode == "RGBA"
        assert patch.getchannel("A").getbbox() is not None


def test_reliable_text_pixels_are_excluded_from_fidelity_components(
    tmp_path: Path,
) -> None:
    source = np.full((40, 60, 3), 255, dtype=np.uint8)
    source[5:12, 5:20] = 0
    source[20:30, 35:45] = (220, 30, 40)
    candidate = np.full_like(source, 255)
    text_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    text_mask[4:13, 4:21] = 255

    result = build_local_fidelity_components(
        source_path=_save_rgb(tmp_path / "source.png", source),
        candidate_path=_save_rgb(tmp_path / "candidate.png", candidate),
        reliable_text_mask_path=_save_mask(tmp_path / "text.png", text_mask),
        output_dir=tmp_path / "fidelity",
    )

    assert result["residual_pixels"] == 100
    for item in result["components"]:
        left, top, right, bottom = item["bbox"]
        with Image.open(item["path"]) as patch:
            alpha = np.asarray(patch.getchannel("A")) > 0
        full = np.zeros(text_mask.shape, dtype=bool)
        full[top:bottom, left:right] = alpha
        assert not np.any(full & (text_mask > 0))


def test_large_connected_residual_is_split_into_bounded_components(
    tmp_path: Path,
) -> None:
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    candidate = np.full_like(source, 255)
    text_mask = np.zeros(source.shape[:2], dtype=np.uint8)

    result = build_local_fidelity_components(
        source_path=_save_rgb(tmp_path / "source.png", source),
        candidate_path=_save_rgb(tmp_path / "candidate.png", candidate),
        reliable_text_mask_path=_save_mask(tmp_path / "text.png", text_mask),
        output_dir=tmp_path / "fidelity",
        max_component_coverage=0.2,
    )

    assert len(result["components"]) > 1
    assert all(item["coverage"] <= 0.2 for item in result["components"])
    assert result["residual_pixels"] == 10_000
    assert result["uncovered_pixels"] == 0


def test_empty_residual_creates_no_files(tmp_path: Path) -> None:
    source = np.full((20, 30, 3), 127, dtype=np.uint8)
    text_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    output_dir = tmp_path / "fidelity"

    result = build_local_fidelity_components(
        source_path=_save_rgb(tmp_path / "source.png", source),
        candidate_path=_save_rgb(tmp_path / "candidate.png", source),
        reliable_text_mask_path=_save_mask(tmp_path / "text.png", text_mask),
        output_dir=output_dir,
    )

    assert result == {
        "components": [],
        "residual_pixels": 0,
        "uncovered_pixels": 0,
    }
    assert not output_dir.exists()
