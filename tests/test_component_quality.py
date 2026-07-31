from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest


def _validate_pixel_ownership(*args, **kwargs):
    assert importlib.util.find_spec("image2editable.component_quality") is not None
    module = importlib.import_module("image2editable.component_quality")
    return module.validate_pixel_ownership(*args, **kwargs)


def _mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    y1, x1, y2, x2 = box
    mask[y1:y2, x1:x2] = 255
    return mask


def test_each_foreground_pixel_has_one_active_owner() -> None:
    first = _mask((14, 14), (2, 2, 8, 8))
    second = _mask((14, 14), (6, 6, 12, 12))

    report = _validate_pixel_ownership(
        [first, second],
        text_mask=np.zeros(first.shape, dtype=np.uint8),
        shape=first.shape,
    )

    assert report == {
        "valid": False,
        "duplicate_pixels": 4,
        "missing_pixels": 0,
        "text_duplicate_pixels": 0,
        "out_of_bounds_pixels": 0,
    }


def test_missing_pixels_require_an_explicit_foreground_mask() -> None:
    owner = _mask((8, 8), (1, 1, 4, 4))
    expected = owner.copy()
    expected[5:7, 5:7] = 255

    without_expected = _validate_pixel_ownership(
        [owner],
        text_mask=np.zeros(owner.shape, dtype=np.uint8),
        shape=owner.shape,
    )
    with_expected = _validate_pixel_ownership(
        [owner],
        text_mask=np.zeros(owner.shape, dtype=np.uint8),
        shape=owner.shape,
        foreground_mask=expected,
    )

    assert without_expected["missing_pixels"] == 0
    assert without_expected["valid"] is True
    assert with_expected["missing_pixels"] == 4
    assert with_expected["valid"] is False


def test_text_duplicate_and_out_of_bounds_pixels_are_reported_separately() -> None:
    oversized = np.zeros((7, 7), dtype=np.uint8)
    oversized[1:4, 1:4] = 255
    oversized[6, 2:5] = 255
    text = _mask((6, 6), (2, 2, 4, 4))

    report = _validate_pixel_ownership(
        [oversized],
        text_mask=text,
        shape=(6, 6),
    )

    assert report["duplicate_pixels"] == 0
    assert report["missing_pixels"] == 0
    assert report["text_duplicate_pixels"] == 4
    assert report["out_of_bounds_pixels"] == 3
    assert report["valid"] is False


def test_alpha_shadow_and_antialias_evidence_still_has_one_owner() -> None:
    gradient = np.zeros((8, 8), dtype=np.uint8)
    gradient[2:6, 2:6] = np.array(
        [
            [32, 64, 64, 32],
            [64, 255, 255, 64],
            [64, 255, 255, 64],
            [32, 64, 64, 32],
        ],
        dtype=np.uint8,
    )
    shadow = np.zeros_like(gradient)
    shadow[6, 3] = 1

    valid = _validate_pixel_ownership(
        [gradient, shadow],
        text_mask=np.zeros(gradient.shape, dtype=np.uint8),
        shape=gradient.shape,
    )
    shadow[5, 3] = 1
    duplicate = _validate_pixel_ownership(
        [gradient, shadow],
        text_mask=np.zeros(gradient.shape, dtype=np.uint8),
        shape=gradient.shape,
    )

    assert valid["valid"] is True
    assert duplicate["duplicate_pixels"] == 1
    assert duplicate["valid"] is False


def test_ownership_validation_does_not_modify_masks() -> None:
    component = _mask((6, 6), (1, 1, 5, 5))
    text = _mask((6, 6), (2, 2, 3, 3))
    foreground = component.copy()
    originals = (component.copy(), text.copy(), foreground.copy())

    _validate_pixel_ownership(
        [component],
        text_mask=text,
        shape=component.shape,
        foreground_mask=foreground,
    )

    assert np.array_equal(component, originals[0])
    assert np.array_equal(text, originals[1])
    assert np.array_equal(foreground, originals[2])


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[object()]], dtype=object),
        np.array([["mask"]]),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[-1]], dtype=np.int16),
    ],
)
def test_ownership_rejects_invalid_mask_values(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="mask"):
        _validate_pixel_ownership(
            [invalid],
            text_mask=np.zeros((1, 1), dtype=np.uint8),
            shape=(1, 1),
        )


def test_exact_bool_mask_projection_reuses_input_memory() -> None:
    assert importlib.util.find_spec("image2editable.component_quality") is not None
    module = importlib.import_module("image2editable.component_quality")
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    projected, outside = module._project_component_mask(mask, mask.shape)

    assert np.shares_memory(projected, mask)
    assert outside == 0


def test_ownership_accumulator_uses_only_boolean_page_buffers(monkeypatch) -> None:
    assert importlib.util.find_spec("image2editable.component_quality") is not None
    module = importlib.import_module("image2editable.component_quality")
    real_zeros = module.np.zeros
    allocated_dtypes = []

    def recording_zeros(*args, **kwargs):
        allocated_dtypes.append(kwargs.get("dtype"))
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(module.np, "zeros", recording_zeros)
    first = np.zeros((8, 8), dtype=bool)
    first[1:4, 1:4] = True
    second = np.zeros((8, 8), dtype=bool)
    second[5:7, 5:7] = True

    report = module.validate_pixel_ownership(
        [first, second],
        text_mask=np.zeros((8, 8), dtype=bool),
        shape=(8, 8),
    )

    assert report["valid"] is True
    assert allocated_dtypes
    assert set(allocated_dtypes) <= {bool, np.bool_}
