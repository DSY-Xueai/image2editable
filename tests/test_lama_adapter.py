from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import traceback
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts import lama_inpaint


_REAL_EQUIVALENCE_MANIFEST_SHA256 = (
    "dc93d38da73367705d32efe4f95ddf36a120d27a35c83bddc630b2788a12352e"
)
_REAL_EQUIVALENCE_CONTRACT = {
    "checkpoint_sha256": (
        "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
    ),
    "reference": {
        "device": "cpu",
        "implementation": "simple_lama_inpainting.SimpleLama",
        "version": "0.1.2",
    },
    "cases": {
        "rectangle": {
            "input": {
                "dtype": "uint8",
                "sha256": "1563b9c9f1dabd7b4adebb655cfe1d0dae63fd625d2fc070d54a44cde5de004c",
                "shape": [67, 65, 3],
            },
            "mask": {
                "dtype": "uint8",
                "sha256": "5130c37d5ffdc7ee4e337d907109d4394fcfc342e323dcbb77b28874c3f10034",
                "shape": [67, 65],
            },
            "reference": {
                "dtype": "uint8",
                "file": "cpu-reference-rectangle.npy",
                "file_sha256": "48bb8ad42dcab20924f7ca5cc5bb6dcdc9847d7cff6918e9993ec06ec6b3fc0e",
                "sha256": "da85d8464e05d11055eab997943c6e89e01cac5eb7ae888fa52ca9e3a5933842",
                "shape": [67, 65, 3],
            },
        },
        "thin-line": {
            "input": {
                "dtype": "uint8",
                "sha256": "1563b9c9f1dabd7b4adebb655cfe1d0dae63fd625d2fc070d54a44cde5de004c",
                "shape": [67, 65, 3],
            },
            "mask": {
                "dtype": "uint8",
                "sha256": "f9099158a32d42c80ea7e8522e75a76d89c74b30a7d9c6526a12dc1da8b50adf",
                "shape": [67, 65],
            },
            "reference": {
                "dtype": "uint8",
                "file": "cpu-reference-thin-line.npy",
                "file_sha256": "20fe2fc3ad959edbc4208234d013eddc47889789b202b500cf1ff6c1aab23325",
                "sha256": "635949d4aba34fc9f47cce6167792a6a751a4237d453265a7bf1c675f4e17ca2",
                "shape": [67, 65, 3],
            },
        },
        "non-modulo-boundary": {
            "input": {
                "dtype": "uint8",
                "sha256": "1563b9c9f1dabd7b4adebb655cfe1d0dae63fd625d2fc070d54a44cde5de004c",
                "shape": [67, 65, 3],
            },
            "mask": {
                "dtype": "uint8",
                "sha256": "0609b8ab7cb332665d3afdef176b5b06aa7113dbdc0838e74a42acfc1828c9f7",
                "shape": [67, 65],
            },
            "reference": {
                "dtype": "uint8",
                "file": "cpu-reference-non-modulo-boundary.npy",
                "file_sha256": "02b4f911bcf32f311dd668e0587768ce036a3f041cfa9072f525063936e5d7d9",
                "sha256": "09c3b37f1b9cb082c94ff2f1809ebd2d0a5b97059b859689d6a3b76347354db3",
                "shape": [67, 65, 3],
            },
        },
        "full-rectangle": {
            "input": {
                "dtype": "uint8",
                "sha256": "4ec67cf9c058dba2af19fbf36949efe3f9d3d1e546bee850db855fc698719a49",
                "shape": [936, 1665, 3],
            },
            "mask": {
                "dtype": "uint8",
                "sha256": "a4bab30ee091b96ac586d11cdaa860417ae9304c2b690efbe9761b0f969dac9d",
                "shape": [936, 1665],
            },
            "reference": {
                "dtype": "uint8",
                "file": "cpu-reference-full-rectangle.npy",
                "file_sha256": "54dbcb7c34eac143d0fac197bde5515eb931230397d5abd6d9436045e01da319",
                "sha256": "9e282f2f01addce51831db160165b07bcad11551eb9aad19e66bd69b39964677",
                "shape": [936, 1665, 3],
            },
        },
    },
}


def test_resolve_lama_checkpoint_uses_runtime_model_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "big-lama.pt"
    calls: list[str] = []
    monkeypatch.setattr(
        lama_inpaint,
        "resolve_runtime_model_path",
        lambda name: calls.append(name) or checkpoint,
    )

    assert lama_inpaint.resolve_lama_checkpoint() == checkpoint
    assert calls == ["big_lama"]


def test_resolve_lama_checkpoint_maps_bridge_error_without_path_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = lama_inpaint.RuntimeModelPathError(
        "LAMA_MODEL model file failed integrity verification"
    )
    monkeypatch.setattr(
        lama_inpaint,
        "resolve_runtime_model_path",
        lambda _name: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(
        lama_inpaint.LargeMaskInpaintError,
        match="failed integrity verification",
    ) as caught:
        lama_inpaint.resolve_lama_checkpoint()

    assert caught.value.__cause__ is None

class _FakeBigLamaOutput:
    def __init__(self, payload, events: list) -> None:
        self.payload = payload
        self.events = events

    def __getitem__(self, index: int):
        self.events.append(("output", index))
        return self

    def permute(self, *axes: int):
        self.events.append(("permute", axes))
        return self

    def detach(self):
        self.events.append("detach")
        return self

    def cpu(self):
        self.events.append("cpu")
        return self

    def numpy(self):
        self.events.append("numpy")
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _install_fake_big_lama_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload=None,
    cuda_available: bool = False,
    load_error: Exception | None = None,
    inference_error: Exception | None = None,
):
    events = []
    output = _FakeBigLamaOutput(
        np.array([[[1.2, -0.1, 0.5]]], dtype=np.float32)
        if payload is None
        else payload,
        events,
    )

    class Model:
        def eval(self):
            events.append("eval")
            return self

        def to(self, device):
            events.append(("model-to", device))
            return self

        def __call__(self, image_tensor, mask_tensor):
            events.append(("model", image_tensor, mask_tensor))
            if inference_error is not None:
                raise inference_error
            return output

    class InferenceMode:
        def __enter__(self):
            events.append("inference-enter")

        def __exit__(self, *_args):
            events.append("inference-exit")

    def load(path: str, *, map_location):
        events.append(("load", path, map_location))
        if load_error is not None:
            raise load_error
        return Model()

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: events.append("cuda-available")
            or cuda_available
        ),
        device=lambda name: events.append(("device", name)) or f"device:{name}",
        jit=types.SimpleNamespace(load=load),
        inference_mode=lambda: InferenceMode(),
    )
    monkeypatch.setattr(
        lama_inpaint.importlib,
        "import_module",
        lambda name: events.append(("import", name)) or fake_torch,
    )
    return events


def test_big_lama_matches_torchscript_adapter_call_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "private-model-name.pt"
    events = _install_fake_big_lama_torch(monkeypatch)
    image = object()
    mask = object()
    image_tensor = object()
    mask_tensor = object()
    monkeypatch.setattr(
        lama_inpaint,
        "_prepare_image_and_mask",
        lambda actual_image, actual_mask, *, device: events.append(
            ("prepare", actual_image, actual_mask, device)
        )
        or (image_tensor, mask_tensor),
    )

    model = lama_inpaint._BigLama(checkpoint, "device:cpu")
    output = model(image, mask)

    assert output.mode == "RGB"
    np.testing.assert_array_equal(
        np.asarray(output),
        np.array([[[255, 0, 127]]], dtype=np.uint8),
    )
    assert events == [
        ("import", "torch"),
        ("load", str(checkpoint), "device:cpu"),
        "eval",
        ("model-to", "device:cpu"),
        ("prepare", image, mask, "device:cpu"),
        "inference-enter",
        ("model", image_tensor, mask_tensor),
        ("output", 0),
        ("permute", (1, 2, 0)),
        "detach",
        "cpu",
        "numpy",
        "inference-exit",
    ]


def test_big_lama_maps_model_load_error_without_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "secret-checkpoint-name.pt"
    failure = RuntimeError(f"cannot load {checkpoint}")
    _install_fake_big_lama_torch(monkeypatch, load_error=failure)

    with pytest.raises(lama_inpaint.LargeMaskInpaintError) as raised:
        lama_inpaint._BigLama(checkpoint, "device:cpu")

    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert str(checkpoint) not in str(raised.value)


def test_big_lama_maps_inference_error_without_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "secret-checkpoint-name.pt"
    failure = RuntimeError(f"inference failed for {checkpoint}")
    _install_fake_big_lama_torch(monkeypatch, inference_error=failure)
    monkeypatch.setattr(
        lama_inpaint,
        "_prepare_image_and_mask",
        lambda *_args, **_kwargs: (object(), object()),
    )
    model = lama_inpaint._BigLama(checkpoint, "device:cpu")

    with pytest.raises(lama_inpaint.LargeMaskInpaintError) as raised:
        model(object(), object())

    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert str(checkpoint) not in str(raised.value)


def test_big_lama_maps_output_error_without_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "secret-checkpoint-name.pt"
    failure = RuntimeError(f"bad output for {checkpoint}")
    _install_fake_big_lama_torch(monkeypatch, payload=failure)
    monkeypatch.setattr(
        lama_inpaint,
        "_prepare_image_and_mask",
        lambda *_args, **_kwargs: (object(), object()),
    )
    model = lama_inpaint._BigLama(checkpoint, "device:cpu")

    with pytest.raises(lama_inpaint.LargeMaskInpaintError) as raised:
        model(object(), object())

    rendered = "".join(
        traceback.format_exception(
            type(raised.value),
            raised.value,
            raised.value.__traceback__,
        )
    )
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert str(checkpoint) not in rendered


@pytest.mark.parametrize(
    "payload",
    [
        np.zeros((4, 5), dtype=np.float32),
        np.zeros((4, 5, 4), dtype=np.float32),
        "not-an-array",
    ],
)
def test_big_lama_rejects_invalid_output_type_or_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload,
) -> None:
    events = _install_fake_big_lama_torch(monkeypatch, payload=payload)
    monkeypatch.setattr(
        lama_inpaint,
        "_prepare_image_and_mask",
        lambda *_args, **_kwargs: (object(), object()),
    )
    model = lama_inpaint._BigLama(tmp_path / "model.pt", "device:cpu")

    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="invalid output"):
        model(object(), object())

    assert "inference-enter" in events
    assert "inference-exit" in events


@pytest.mark.parametrize(
    ("cuda_available", "expected_device"),
    [(True, "cuda"), (False, "cpu")],
)
def test_create_model_selects_device_and_loads_resolved_checkpoint_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    expected_device: str,
) -> None:
    checkpoint = tmp_path / "bound-checkpoint.pt"
    resolve_calls = []
    monkeypatch.setattr(
        lama_inpaint,
        "resolve_lama_checkpoint",
        lambda: resolve_calls.append("resolve") or checkpoint,
    )
    events = _install_fake_big_lama_torch(
        monkeypatch,
        cuda_available=cuda_available,
    )

    model = lama_inpaint._create_model()

    assert isinstance(model, lama_inpaint._BigLama)
    assert resolve_calls == ["resolve"]
    assert events == [
        ("import", "torch"),
        "cuda-available",
        ("device", expected_device),
        ("import", "torch"),
        ("load", str(checkpoint), f"device:{expected_device}"),
        "eval",
        ("model-to", f"device:{expected_device}"),
    ]


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


def test_real_checkpoint_equivalence_against_simple_lama_baseline() -> None:
    equivalence_dir = os.environ.get("IMAGE2EDITABLE_LAMA_EQUIVALENCE_DIR")
    checkpoint_value = os.environ.get("LAMA_MODEL")
    if not equivalence_dir or not checkpoint_value:
        pytest.skip(
            "requires IMAGE2EDITABLE_LAMA_EQUIVALENCE_DIR and LAMA_MODEL"
        )

    root = Path(equivalence_dir)
    old = root / "old"
    references = root / "cpu-reference"
    manifest = json.loads(
        (references / "manifest.json").read_text(encoding="utf-8")
    )
    canonical_manifest = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(canonical_manifest).hexdigest() == (
        _REAL_EQUIVALENCE_MANIFEST_SHA256
    )
    assert manifest == _REAL_EQUIVALENCE_CONTRACT
    checkpoint = Path(checkpoint_value)
    assert lama_inpaint.resolve_lama_checkpoint() == checkpoint.resolve()

    height, width = 67, 65
    y, x = np.indices((height, width), dtype=np.uint16)
    small_image = np.stack(
        (
            (x * 7 + y * 3) % 256,
            (x * 5 + 17) % 256,
            (y * 11 + x) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    rectangle = np.zeros((height, width), dtype=np.uint8)
    rectangle[18:47, 19:52] = 255
    thin_line = np.zeros((height, width), dtype=np.uint8)
    thin_line[5:62, 32] = 255
    boundary = np.zeros((height, width), dtype=np.uint8)
    boundary[63:67, 60:65] = 255
    fixtures = {
        "rectangle": (small_image, rectangle),
        "thin-line": (small_image, thin_line),
        "non-modulo-boundary": (small_image, boundary),
        "full-rectangle": (
            np.load(old / "input.npy"),
            np.load(old / "mask-rectangle.npy"),
        ),
    }

    previous_model = lama_inpaint._MODEL
    lama_inpaint._MODEL = lama_inpaint._BigLama(
        checkpoint,
        torch.device("cpu"),
    )
    try:
        for name, (image, mask) in fixtures.items():
            case = manifest["cases"][name]
            for value, expected in ((image, case["input"]), (mask, case["mask"])):
                assert list(value.shape) == expected["shape"]
                assert str(value.dtype) == expected["dtype"]
                assert hashlib.sha256(
                    np.ascontiguousarray(value).tobytes()
                ).hexdigest() == expected["sha256"]

            reference = case["reference"]
            reference_path = references / reference["file"]
            assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == (
                reference["file_sha256"]
            )
            expected = np.load(reference_path)
            assert list(expected.shape) == reference["shape"]
            assert str(expected.dtype) == reference["dtype"]
            assert hashlib.sha256(
                np.ascontiguousarray(expected).tobytes()
            ).hexdigest() == reference["sha256"]

            actual = lama_inpaint.inpaint_large_mask(image, mask)
            assert actual.shape == expected.shape
            delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            assert int(delta.max(initial=0)) <= 1
            assert int(delta[mask == 0].max(initial=0)) == 0
    finally:
        lama_inpaint._MODEL = previous_model
