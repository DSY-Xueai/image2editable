from __future__ import annotations

import hashlib
import os
import stat
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


_SMALL_CHECKPOINT = b"fixed-test-checkpoint"


@pytest.fixture
def small_checkpoint(monkeypatch: pytest.MonkeyPatch) -> bytes:
    monkeypatch.setattr(
        lama_inpaint,
        "BIG_LAMA_MODEL_SIZE",
        len(_SMALL_CHECKPOINT),
    )
    monkeypatch.setattr(
        lama_inpaint,
        "BIG_LAMA_MODEL_SHA256",
        hashlib.sha256(_SMALL_CHECKPOINT).hexdigest(),
    )
    return _SMALL_CHECKPOINT


def _write_download(url: str, destination: str | Path) -> None:
    assert url == lama_inpaint.BIG_LAMA_MODEL_URL
    destination = Path(destination)
    assert destination.exists()
    destination.write_bytes(_SMALL_CHECKPOINT)


def test_big_lama_checkpoint_constants_are_immutable_release_identity() -> None:
    assert lama_inpaint.BIG_LAMA_MODEL_URL == (
        "https://github.com/enesmsahin/simple-lama-inpainting/releases/"
        "download/v0.1.0/big-lama.pt"
    )
    assert lama_inpaint.BIG_LAMA_MODEL_SIZE == 205803670
    assert lama_inpaint.BIG_LAMA_MODEL_SHA256 == (
        "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
    )


def test_resolve_lama_checkpoint_downloads_to_private_preallocated_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_checkpoint: bytes,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    downloads = []
    fsynced = []
    actual_fsync = os.fsync
    monkeypatch.setattr(
        lama_inpaint.os,
        "fsync",
        lambda descriptor: fsynced.append(descriptor) or actual_fsync(descriptor),
    )

    def downloader(url: str, destination: str | Path) -> None:
        destination = Path(destination)
        status = destination.lstat()
        downloads.append((url, destination, stat.S_IMODE(status.st_mode)))
        assert destination.parent == cache
        assert destination.name != "big-lama.pt"
        destination.write_bytes(small_checkpoint)

    resolved = lama_inpaint.resolve_lama_checkpoint(
        cache_dir=cache,
        downloader=downloader,
    )

    assert resolved == cache / "big-lama.pt"
    assert resolved.read_bytes() == small_checkpoint
    assert downloads[0][:2] == (
        lama_inpaint.BIG_LAMA_MODEL_URL,
        downloads[0][1],
    )
    if os.name != "nt":
        assert downloads[0][2] == 0o600
    assert len(fsynced) == 1
    assert list(cache.iterdir()) == [resolved]


def test_resolve_lama_checkpoint_uses_environment_cache_and_reuses_valid_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_checkpoint: bytes,
) -> None:
    cache = tmp_path / "models"
    cache.mkdir()
    checkpoint = cache / "big-lama.pt"
    checkpoint.write_bytes(small_checkpoint)
    monkeypatch.setenv("IMAGE2EDITABLE_MODEL_CACHE", str(cache))
    monkeypatch.delenv("LAMA_MODEL", raising=False)

    resolved = lama_inpaint.resolve_lama_checkpoint(
        downloader=lambda *_: pytest.fail("valid cache must not be downloaded")
    )

    assert resolved == checkpoint


def test_resolve_lama_checkpoint_uses_default_user_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_checkpoint: bytes,
) -> None:
    monkeypatch.delenv("IMAGE2EDITABLE_MODEL_CACHE", raising=False)
    monkeypatch.delenv("LAMA_MODEL", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    resolved = lama_inpaint.resolve_lama_checkpoint(downloader=_write_download)

    assert resolved == (
        tmp_path
        / ".cache"
        / "image2editable"
        / "models"
        / "runtime"
        / "big-lama.pt"
    )
    assert resolved.read_bytes() == small_checkpoint


@pytest.mark.parametrize("payload", [b"short", b"wrong-checkpoint-data"])
def test_resolve_lama_checkpoint_rejects_invalid_existing_default_cache(
    tmp_path: Path,
    small_checkpoint: bytes,
    payload: bytes,
) -> None:
    checkpoint = tmp_path / "big-lama.pt"
    checkpoint.write_bytes(payload)

    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="integrity"):
        lama_inpaint.resolve_lama_checkpoint(
            cache_dir=tmp_path,
            downloader=lambda *_: pytest.fail("bad cache must not be overwritten"),
        )

    assert checkpoint.read_bytes() == payload


def test_resolve_lama_checkpoint_accepts_custom_model_with_distinct_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = tmp_path / "custom-lama.pt"
    custom.write_bytes(b"custom model payload")
    monkeypatch.setenv("LAMA_MODEL", str(custom))

    resolved = lama_inpaint.resolve_lama_checkpoint(
        cache_dir=tmp_path / "unused",
        downloader=lambda *_: pytest.fail("custom model must not be downloaded"),
    )

    assert resolved == custom


def test_checkpoint_identity_contains_no_absolute_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "custom-lama.pt"
    payload = b"custom model payload"
    checkpoint.write_bytes(payload)

    identity = lama_inpaint.checkpoint_identity(checkpoint)

    assert identity == {
        "basename": checkpoint.name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert str(tmp_path) not in repr(identity)


def test_checkpoint_identity_rejects_missing_and_nonregular_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="missing"):
        lama_inpaint.checkpoint_identity(tmp_path / "missing.pt")
    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="regular"):
        lama_inpaint.checkpoint_identity(tmp_path)


def test_checkpoint_identity_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"model")
    hardlink = tmp_path / "hardlink.pt"
    os.link(source, hardlink)

    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="hard link"):
        lama_inpaint.checkpoint_identity(source)

    symlink = tmp_path / "symlink.pt"
    try:
        symlink.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(
        lama_inpaint.LargeMaskInpaintError,
        match="link or reparse",
    ):
        lama_inpaint.checkpoint_identity(symlink)


def test_checkpoint_identity_rejects_reparse_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    actual_lstat = Path.lstat

    def fake_lstat(path: Path):
        status = actual_lstat(path)
        if path == checkpoint:
            values = {
                name: getattr(status, name)
                for name in (
                    "st_mode",
                    "st_dev",
                    "st_ino",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                )
            }
            values["st_file_attributes"] = getattr(
                stat,
                "FILE_ATTRIBUTE_REPARSE_POINT",
                0x400,
            )
            return types.SimpleNamespace(**values)
        return status

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(
        lama_inpaint.LargeMaskInpaintError,
        match="link or reparse",
    ):
        lama_inpaint.checkpoint_identity(checkpoint)


def test_checkpoint_identity_rejects_file_identity_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    actual_fstat = os.fstat
    calls = 0

    def changed_fstat(descriptor: int):
        nonlocal calls
        status = actual_fstat(descriptor)
        calls += 1
        if calls == 2:
            values = {
                name: getattr(status, name)
                for name in (
                    "st_mode",
                    "st_dev",
                    "st_ino",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                )
            }
            values["st_ino"] += 1
            return types.SimpleNamespace(**values)
        return status

    monkeypatch.setattr(lama_inpaint.os, "fstat", changed_fstat)

    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="changed"):
        lama_inpaint.checkpoint_identity(checkpoint)


def test_checkpoint_identity_rejects_parent_chain_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "models"
    parent.mkdir()
    checkpoint = parent / "model.pt"
    checkpoint.write_bytes(b"model")
    actual_lstat = Path.lstat
    parent_calls = 0

    def changed_parent_lstat(path: Path):
        nonlocal parent_calls
        status = actual_lstat(path)
        if path == parent:
            parent_calls += 1
            if parent_calls > 1:
                values = {
                    name: getattr(status, name)
                    for name in (
                        "st_mode",
                        "st_dev",
                        "st_ino",
                        "st_nlink",
                        "st_size",
                        "st_mtime_ns",
                    )
                }
                values["st_ino"] += 1
                values["st_file_attributes"] = getattr(
                    status,
                    "st_file_attributes",
                    0,
                )
                return types.SimpleNamespace(**values)
        return status

    monkeypatch.setattr(Path, "lstat", changed_parent_lstat)

    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="parent.*changed"):
        lama_inpaint.checkpoint_identity(checkpoint)


def test_resolve_lama_checkpoint_cleans_own_temporary_after_download_error(
    tmp_path: Path,
    small_checkpoint: bytes,
) -> None:
    failure = RuntimeError("download failed")

    def downloader(_url: str, _destination: str | Path) -> None:
        raise failure

    with pytest.raises(RuntimeError) as raised:
        lama_inpaint.resolve_lama_checkpoint(
            cache_dir=tmp_path,
            downloader=downloader,
        )

    assert raised.value is failure
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("payload", [b"short", b"wrong-checkpoint-data"])
def test_resolve_lama_checkpoint_cleans_invalid_download(
    tmp_path: Path,
    small_checkpoint: bytes,
    payload: bytes,
) -> None:
    def downloader(_url: str, destination: str | Path) -> None:
        Path(destination).write_bytes(payload)

    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="integrity"):
        lama_inpaint.resolve_lama_checkpoint(
            cache_dir=tmp_path,
            downloader=downloader,
        )

    assert list(tmp_path.iterdir()) == []


def test_resolve_lama_checkpoint_never_overwrites_racing_target(
    tmp_path: Path,
    small_checkpoint: bytes,
) -> None:
    target = tmp_path / "big-lama.pt"
    attacker = b"racing target"

    def downloader(_url: str, destination: str | Path) -> None:
        Path(destination).write_bytes(small_checkpoint)
        target.write_bytes(attacker)

    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="integrity"):
        lama_inpaint.resolve_lama_checkpoint(
            cache_dir=tmp_path,
            downloader=downloader,
        )

    assert target.read_bytes() == attacker
    assert list(tmp_path.iterdir()) == [target]


def test_resolve_lama_checkpoint_rejects_parent_replacement_after_publish(
    tmp_path: Path,
    small_checkpoint: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_link = os.link
    actual_lstat = Path.lstat
    published = False

    def publish_then_replace_identity(*args, **kwargs) -> None:
        nonlocal published
        actual_link(*args, **kwargs)
        published = True

    def changed_parent_lstat(path: Path):
        status = actual_lstat(path)
        if path == tmp_path and published:
            values = {
                name: getattr(status, name)
                for name in (
                    "st_mode",
                    "st_dev",
                    "st_ino",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                )
            }
            values["st_ino"] += 1
            values["st_file_attributes"] = getattr(
                status,
                "st_file_attributes",
                0,
            )
            return types.SimpleNamespace(**values)
        return status

    monkeypatch.setattr(lama_inpaint.os, "link", publish_then_replace_identity)
    monkeypatch.setattr(Path, "lstat", changed_parent_lstat)

    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="parent.*changed"):
        lama_inpaint.resolve_lama_checkpoint(
            cache_dir=tmp_path,
            downloader=_write_download,
        )


def test_resolve_lama_checkpoint_does_not_delete_replaced_temporary_path(
    tmp_path: Path,
    small_checkpoint: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attacker = b"attacker replacement"
    temporary = None
    actual_lstat = Path.lstat

    def downloader(_url: str, destination: str | Path) -> None:
        nonlocal temporary
        temporary = Path(destination)
        temporary.write_bytes(small_checkpoint)

    replaced = False
    temporary_lstats = 0

    def replace_before_cleanup(path: Path):
        nonlocal replaced, temporary_lstats
        status = actual_lstat(path)
        if temporary is not None and path == temporary:
            temporary_lstats += 1
        if temporary_lstats >= 3 and not replaced:
            replaced = True
            path.unlink()
            path.write_bytes(attacker)
            status = actual_lstat(path)
        return status

    monkeypatch.setattr(Path, "lstat", replace_before_cleanup)
    monkeypatch.setattr(
        lama_inpaint,
        "BIG_LAMA_MODEL_SHA256",
        hashlib.sha256(b"different expected payload").hexdigest(),
    )

    with pytest.raises(lama_inpaint.LargeMaskInpaintError, match="integrity"):
        lama_inpaint.resolve_lama_checkpoint(
            cache_dir=tmp_path,
            downloader=downloader,
        )

    assert temporary is not None
    assert temporary.read_bytes() == attacker


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
