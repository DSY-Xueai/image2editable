"""LaMa adapter for repairing large masked image regions."""

from __future__ import annotations

import numpy as np
from PIL import Image


class LargeMaskInpaintError(RuntimeError):
    """Raised when LaMa cannot repair a large masked region."""


_MODEL = None


def _dependency_error(detail: str) -> LargeMaskInpaintError:
    return LargeMaskInpaintError(
        f"{detail} Install simple-lama-inpainting==0.1.2."
    )


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

    if repaired.shape != source.shape:
        raise LargeMaskInpaintError(
            f"LaMa returned shape {repaired.shape}, expected {source.shape}."
        )

    output = repaired.copy()
    output[binary == 0] = source[binary == 0]
    return output
