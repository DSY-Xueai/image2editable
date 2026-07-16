from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
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


def _crop_origins(length: int, crop_size: int, overlap: int) -> list[int]:
    if length <= crop_size:
        return [0]
    step = max(crop_size - overlap, 1)
    origins = list(range(0, max(length - crop_size, 0) + 1, step))
    last = length - crop_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def generate_mask_candidates(
    image: np.ndarray,
    generator,
    crop_size: int = 768,
    overlap: int = 128,
    include_geometry: bool = True,
) -> list[MaskCandidate]:
    if crop_size <= 0 or overlap < 0 or overlap >= crop_size:
        raise ValueError(
            "crop_size must be > 0 and overlap must satisfy 0 <= overlap < crop_size"
        )

    height, width = image.shape[:2]
    crop_boxes = [(0, 0, width, height)]
    if height > crop_size or width > crop_size:
        crop_height = min(crop_size, height)
        crop_width = min(crop_size, width)
        for y in _crop_origins(height, crop_height, overlap):
            for x in _crop_origins(width, crop_width, overlap):
                crop_boxes.append(
                    (x, y, min(x + crop_width, width), min(y + crop_height, height))
                )

    candidates = []
    seen_boxes = set()
    for x1, y1, x2, y2 in crop_boxes:
        box = (x1, y1, x2, y2)
        if box in seen_boxes:
            continue
        seen_boxes.add(box)
        crop = image[y1:y2, x1:x2]
        for record in generator.generate(crop):
            mask = np.asarray(record["segmentation"], dtype=bool)
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y1:y2, x1:x2] = mask
            score = min(
                float(record.get("predicted_iou", 0.0)),
                float(record.get("stability_score", 0.0)),
            )
            candidates.append(MaskCandidate(full_mask, score, "sam"))

    if include_geometry:
        candidates.extend(generate_geometry_candidates(image))
    return candidates


def generate_geometry_candidates(
    image: np.ndarray,
    min_area: int = 20,
) -> list[MaskCandidate]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []
    max_area = image.shape[0] * image.shape[1] * 0.9
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        candidates.append(MaskCandidate(mask > 0, 0.70, "geometry"))
    return candidates


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
