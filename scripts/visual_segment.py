from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


SAM21_LARGE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_large.pt"
)
SAM21_LARGE_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class VisualSegmentationError(RuntimeError):
    pass


@dataclass
class MaskCandidate:
    mask: np.ndarray
    score: float
    source: str


@dataclass
class VisualElement:
    mask: np.ndarray
    z_index: int
    score: float
    source: str


def resolve_sam_checkpoint(cache_dir=None, downloader=urlretrieve) -> Path:
    cache_root = Path(
        cache_dir
        or os.environ.get("IMAGE2EDITABLE_MODEL_CACHE")
        or Path.home() / ".cache" / "image2editable"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = cache_root / "sam2.1_hiera_large.pt"
    if checkpoint_path.exists() and checkpoint_path.stat().st_size:
        return checkpoint_path

    partial_path = checkpoint_path.with_suffix(".pt.part")
    try:
        downloader(SAM21_LARGE_URL, str(partial_path))
        if not partial_path.exists() or not partial_path.stat().st_size:
            raise VisualSegmentationError("SAM 2.1 checkpoint download was empty")
        partial_path.replace(checkpoint_path)
    except VisualSegmentationError:
        partial_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial_path.unlink(missing_ok=True)
        raise VisualSegmentationError(
            f"Unable to download SAM 2.1 checkpoint: {exc}"
        ) from exc

    return checkpoint_path


def create_sam_generator(checkpoint_path, device=None):
    try:
        torch = importlib.import_module("torch")
        build_sam = importlib.import_module("sam2.build_sam")
        mask_generator = importlib.import_module("sam2.automatic_mask_generator")
    except ModuleNotFoundError as exc:
        raise VisualSegmentationError(
            "SAM 2.1 is required. Install project segmentation dependencies."
        ) from exc

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sam.build_sam2(
        SAM21_LARGE_CONFIG,
        str(checkpoint_path),
        device=selected_device,
        apply_postprocessing=False,
    )
    return mask_generator.SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,
        points_per_batch=16 if str(selected_device).startswith("cuda") else 4,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=20,
    )
