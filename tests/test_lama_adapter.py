from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts import lama_inpaint


def test_lama_module_import_does_not_eagerly_import_torch() -> None:
    root = Path(__file__).resolve().parent.parent
    probe = """
import builtins
import sys

actual_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("torch imported during module import")
    return actual_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from scripts import lama_inpaint
assert "torch" not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_prepare_image_and_mask_matches_simple_lama_numpy_semantics() -> None:
    image = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
    mask = np.zeros((5, 9), dtype=np.uint8)
    mask[1:4, 3:7] = 127
    original_image = image.copy()
    original_mask = mask.copy()

    prepared_image, prepared_mask = lama_inpaint._prepare_image_and_mask(
        image,
        mask,
        device=torch.device("cpu"),
    )

    expected_image = np.pad(
        np.transpose(image.astype(np.float32) / 255, (2, 0, 1)),
        ((0, 0), (0, 3), (0, 7)),
        mode="symmetric",
    )
    expected_mask = (
        np.pad(
            (mask.astype(np.float32) / 255)[None, ...],
            ((0, 0), (0, 3), (0, 7)),
            mode="symmetric",
        ) > 0
    ) * 1
    assert prepared_image.dtype is torch.float32
    assert prepared_mask.dtype is torch.int64
    assert prepared_image.shape == (1, 3, 8, 16)
    assert prepared_mask.shape == (1, 1, 8, 16)
    assert prepared_image.device.type == "cpu"
    assert prepared_mask.device.type == "cpu"
    np.testing.assert_array_equal(prepared_image[0].numpy(), expected_image)
    np.testing.assert_array_equal(prepared_mask[0].numpy(), expected_mask)
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(mask, original_mask)


def test_prepare_image_and_mask_accepts_pil_rgb_and_l_without_mutation() -> None:
    image_array = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3)
    mask_array = np.zeros((6, 10), dtype=np.uint8)
    mask_array[2:5, 4:9] = 255
    image = Image.fromarray(image_array, mode="RGB")
    mask = Image.fromarray(mask_array, mode="L")
    image_bytes = image.tobytes()
    mask_bytes = mask.tobytes()

    prepared_image, prepared_mask = lama_inpaint._prepare_image_and_mask(
        image,
        mask,
        device="cpu",
    )

    assert prepared_image.shape == (1, 3, 8, 16)
    assert prepared_mask.shape == (1, 1, 8, 16)
    assert image.mode == "RGB"
    assert mask.mode == "L"
    assert image.tobytes() == image_bytes
    assert mask.tobytes() == mask_bytes


def test_prepare_image_and_mask_does_not_pad_multiple_of_eight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lama_inpaint.np,
        "pad",
        lambda *args, **kwargs: pytest.fail("aligned inputs must not be padded"),
    )

    prepared_image, prepared_mask = lama_inpaint._prepare_image_and_mask(
        np.zeros((8, 16, 3), dtype=np.uint8),
        np.zeros((8, 16), dtype=np.uint8),
        device="cpu",
    )

    assert prepared_image.shape == (1, 3, 8, 16)
    assert prepared_mask.shape == (1, 1, 8, 16)


def test_prepare_image_and_mask_lazily_imports_torch_and_moves_to_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class FakeTensor:
        def __init__(self, array: np.ndarray) -> None:
            self.array = array

        def unsqueeze(self, axis: int) -> FakeTensor:
            events.append(("unsqueeze", axis))
            self.array = np.expand_dims(self.array, axis)
            return self

        def to(self, device: object) -> FakeTensor:
            events.append(("to", device))
            return self

        def __gt__(self, value: object) -> FakeTensor:
            events.append(("gt", value))
            self.array = self.array > value
            return self

        def __mul__(self, value: object) -> FakeTensor:
            events.append(("mul", value))
            self.array = self.array * value
            return self

    fake_torch = types.SimpleNamespace(
        from_numpy=lambda array: events.append(("from_numpy", array.shape))
        or FakeTensor(array)
    )
    monkeypatch.setattr(
        lama_inpaint.importlib,
        "import_module",
        lambda name: events.append(("import", name)) or fake_torch,
    )
    device = object()

    lama_inpaint._prepare_image_and_mask(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8), dtype=np.uint8),
        device=device,
    )

    assert events == [
        ("import", "torch"),
        ("from_numpy", (3, 8, 8)),
        ("unsqueeze", 0),
        ("to", device),
        ("from_numpy", (1, 8, 8)),
        ("unsqueeze", 0),
        ("to", device),
        ("gt", 0),
        ("mul", 1),
    ]


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((1, 8, 8, 3), dtype=np.uint8),
    ],
)
def test_prepare_image_and_mask_rejects_non_rgb_image(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="RGB"):
        lama_inpaint._prepare_image_and_mask(
            image,
            np.zeros((8, 8), dtype=np.uint8),
            device="cpu",
        )


def test_prepare_image_and_mask_rejects_non_l_mask() -> None:
    with pytest.raises(ValueError, match="mask"):
        lama_inpaint._prepare_image_and_mask(
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
            device="cpu",
        )


def test_prepare_image_and_mask_rejects_mismatched_spatial_shape() -> None:
    with pytest.raises(ValueError, match="height and width"):
        lama_inpaint._prepare_image_and_mask(
            np.zeros((8, 9, 3), dtype=np.uint8),
            np.zeros((9, 8), dtype=np.uint8),
            device="cpu",
        )
