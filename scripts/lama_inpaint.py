"""LaMa adapter for repairing large masked image regions."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_MODULE_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _MODULE_ROOT:
    while _MODULE_ROOT in sys.path:
        sys.path.remove(_MODULE_ROOT)
    sys.path.insert(0, _MODULE_ROOT)

from scripts.worker_resources import run_isolated_worker


class LargeMaskInpaintError(RuntimeError):
    """Raised when LaMa cannot repair a large masked region."""


_MODEL = None


def _dependency_error(detail: str) -> LargeMaskInpaintError:
    return LargeMaskInpaintError(
        f"{detail} Install simple-lama-inpainting==0.1.2."
    )


def _prepare_image_and_mask(image, mask, *, device):
    if isinstance(image, Image.Image):
        image_array = np.array(image, copy=True)
    elif isinstance(image, np.ndarray):
        image_array = image.copy()
    else:
        raise TypeError("image must be a NumPy array or PIL image")
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask, copy=True)
    elif isinstance(mask, np.ndarray):
        mask_array = mask.copy()
    else:
        raise TypeError("mask must be a NumPy array or PIL image")

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image must be an RGB array with shape (H, W, 3)")
    if mask_array.ndim != 2:
        raise ValueError("mask must be an L array with shape (H, W)")
    if mask_array.shape != image_array.shape[:2]:
        raise ValueError("mask must match the image height and width")

    image_array = np.transpose(
        image_array.astype(np.float32) / 255,
        (2, 0, 1),
    )
    mask_array = (mask_array.astype(np.float32) / 255)[None, ...]
    height, width = mask_array.shape[1:]
    padding = ((0, 0), (0, (-height) % 8), (0, (-width) % 8))
    if padding[1][1] or padding[2][1]:
        image_array = np.pad(image_array, padding, mode="symmetric")
        mask_array = np.pad(mask_array, padding, mode="symmetric")

    torch = importlib.import_module("torch")
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).to(device)
    mask_tensor = (mask_tensor > 0) * 1
    return image_tensor, mask_tensor


def _create_model():
    try:
        from simple_lama_inpainting import SimpleLama
    except ModuleNotFoundError as exc:
        raise _dependency_error("LaMa dependency is unavailable.") from exc

    try:
        return SimpleLama()
    except Exception as exc:
        raise _dependency_error("LaMa model initialization failed.") from exc


def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = _create_model()
        except LargeMaskInpaintError:
            raise
        except Exception as exc:
            raise _dependency_error("LaMa model initialization failed.") from exc
    return _MODEL


def release_model() -> None:
    global _MODEL
    _MODEL = None


def inpaint_large_mask_isolated(
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path).resolve()
    completed = run_isolated_worker(
        [
            sys.executable,
            str(Path(__file__).with_name("lama_worker.py").resolve()),
            "--image",
            str(Path(image_path).resolve()),
            "--mask",
            str(Path(mask_path).resolve()),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LargeMaskInpaintError(
            f"Isolated LaMa worker failed: {detail}"
        )
    if not output_path.is_file():
        raise LargeMaskInpaintError(
            "Isolated LaMa worker did not create the output image"
        )


def inpaint_large_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Repair a large mask with LaMa while preserving every unmasked pixel."""
    source = np.asarray(image)
    removal = np.asarray(mask)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("image must be an RGB array with shape (H, W, 3)")
    if removal.ndim != 2 or removal.shape != source.shape[:2]:
        raise ValueError("mask must match the image height and width")

    source = source.astype(np.uint8, copy=False)
    binary = (removal > 0).astype(np.uint8) * 255
    model = _get_model()
    try:
        repaired = model(
            Image.fromarray(source, mode="RGB"),
            Image.fromarray(binary, mode="L"),
        )
        repaired = np.asarray(repaired, dtype=np.uint8)
    except Exception as exc:
        raise LargeMaskInpaintError("LaMa inference failed.") from exc

    if repaired.ndim != 3 or repaired.shape[2] != source.shape[2]:
        raise LargeMaskInpaintError(
            f"LaMa returned shape {repaired.shape}, expected {source.shape}."
        )
    source_height, source_width = source.shape[:2]
    padded_height = ((source_height + 7) // 8) * 8
    padded_width = ((source_width + 7) // 8) * 8
    actual_spatial = repaired.shape[:2]
    allowed_spatial = (
        (source_height, source_width),
        (padded_height, padded_width),
    )
    if actual_spatial not in allowed_spatial:
        raise LargeMaskInpaintError(
            f"LaMa returned invalid spatial shape: actual={actual_spatial}, "
            f"allowed={allowed_spatial}."
        )
    repaired = repaired[:source_height, :source_width]
    if repaired.shape != source.shape:
        raise LargeMaskInpaintError(
            f"LaMa returned shape {repaired.shape}, expected {source.shape}."
        )

    output = repaired.copy()
    output[binary == 0] = source[binary == 0]
    return output
