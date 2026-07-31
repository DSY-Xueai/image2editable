from __future__ import annotations

import numpy as np


def _page_shape(shape: object) -> tuple[int, int]:
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ValueError("shape must contain positive integer height and width")
    return shape[0], shape[1]


def _numeric_mask(mask: object, label: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype.kind not in "biuf":
        raise ValueError(f"{label} must be a two-dimensional numeric mask")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    if array.dtype.kind in "if" and np.any(array < 0):
        raise ValueError(f"{label} contains negative values")
    return array


def _exact_mask(mask: object, shape: tuple[int, int], label: str) -> np.ndarray:
    array = _numeric_mask(mask, label)
    if array.shape != shape:
        raise ValueError(f"{label} shape must match page shape")
    return array if array.dtype == np.bool_ else array > 0


def _project_component_mask(
    mask: object,
    shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    array = _numeric_mask(mask, "component mask")
    active = array if array.dtype == np.bool_ else array > 0
    if active.shape == shape:
        return active, 0
    height = min(shape[0], active.shape[0])
    width = min(shape[1], active.shape[1])
    projected = np.zeros(shape, dtype=bool)
    projected[:height, :width] = active[:height, :width]
    out_of_bounds = int(np.count_nonzero(active)) - int(
        np.count_nonzero(active[:height, :width])
    )
    return projected, out_of_bounds


def validate_pixel_ownership(
    component_masks: list[np.ndarray],
    text_mask: np.ndarray,
    shape: tuple[int, int],
    *,
    foreground_mask: np.ndarray | None = None,
) -> dict:
    """Report ownership defects without modifying or repairing any mask.

    ``missing_pixels`` is meaningful only when ``foreground_mask`` is given.
    Every non-zero alpha value counts as source evidence owned by that component.
    """

    page_shape = _page_shape(shape)
    text = _exact_mask(text_mask, page_shape, "text mask")
    claimed = np.zeros(page_shape, dtype=bool)
    duplicate = np.zeros(page_shape, dtype=bool)
    out_of_bounds = 0
    for mask in component_masks:
        projected, outside = _project_component_mask(mask, page_shape)
        duplicate |= claimed & projected
        claimed |= projected
        out_of_bounds += outside

    if foreground_mask is None:
        missing_pixels = 0
    else:
        foreground = _exact_mask(
            foreground_mask,
            page_shape,
            "foreground mask",
        )
        missing_pixels = int(np.count_nonzero(foreground & ~claimed))

    report = {
        "duplicate_pixels": int(np.count_nonzero(duplicate)),
        "missing_pixels": missing_pixels,
        "text_duplicate_pixels": int(np.count_nonzero(text & claimed)),
        "out_of_bounds_pixels": out_of_bounds,
    }
    return {"valid": not any(report.values()), **report}
