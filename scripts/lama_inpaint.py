"""LaMa adapter for repairing large masked image regions."""

from __future__ import annotations

import hashlib
import importlib
import os
import secrets
import stat
import sys
from pathlib import Path
from urllib.request import urlretrieve

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

BIG_LAMA_MODEL_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/"
    "download/v0.1.0/big-lama.pt"
)
BIG_LAMA_MODEL_SIZE = 205803670
BIG_LAMA_MODEL_SHA256 = (
    "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
)
_CHECKPOINT_NAME = "big-lama.pt"
_CHECKPOINT_CHUNK_SIZE = 1024 * 1024


def _dependency_error(detail: str) -> LargeMaskInpaintError:
    return LargeMaskInpaintError(
        f"{detail} Install simple-lama-inpainting==0.1.2."
    )


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse
    )


def _directory_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        getattr(status, "st_file_attributes", 0),
    )


def _snapshot_parent_chain(
    parent: Path,
) -> list[tuple[Path, tuple[int, int, int, int]]]:
    anchor = Path(parent.anchor)
    current = anchor
    chain = []
    for part in (Path(), *parent.relative_to(anchor).parts):
        if part != Path():
            current /= part
        try:
            status = current.lstat()
        except OSError as exc:
            raise LargeMaskInpaintError(
                f"LaMa checkpoint parent is missing: {current}"
            ) from exc
        if _is_link_or_reparse(status) or not stat.S_ISDIR(status.st_mode):
            raise LargeMaskInpaintError(
                f"LaMa checkpoint parent is unsafe: {current}"
            )
        chain.append((current, _directory_identity(status)))
    return chain


def _require_parent_chain(
    chain: list[tuple[Path, tuple[int, int, int, int]]],
) -> None:
    for path, expected in chain:
        try:
            status = path.lstat()
        except OSError as exc:
            raise LargeMaskInpaintError(
                f"LaMa checkpoint parent identity changed: {path}"
            ) from exc
        if (
            _is_link_or_reparse(status)
            or not stat.S_ISDIR(status.st_mode)
            or _directory_identity(status) != expected
        ):
            raise LargeMaskInpaintError(
                f"LaMa checkpoint parent identity changed: {path}"
            )


def _validate_checkpoint_status(path: Path, status: os.stat_result) -> None:
    if _is_link_or_reparse(status):
        raise LargeMaskInpaintError(
            f"LaMa checkpoint is a link or reparse point: {path}"
        )
    if not stat.S_ISREG(status.st_mode):
        raise LargeMaskInpaintError(
            f"LaMa checkpoint is not a regular file: {path}"
        )
    if status.st_nlink != 1:
        raise LargeMaskInpaintError(
            f"LaMa checkpoint is an unsafe hard link: {path}"
        )


def _read_checkpoint_identity(
    path: Path,
    *,
    descriptor: int | None = None,
) -> dict[str, str | int]:
    path = _absolute_path(path)
    chain = _snapshot_parent_chain(path.parent)
    try:
        path_status = path.lstat()
    except FileNotFoundError as exc:
        raise LargeMaskInpaintError(
            f"LaMa checkpoint is missing: {path}"
        ) from exc
    _validate_checkpoint_status(path, path_status)

    owned_descriptor = descriptor is None
    if descriptor is None:
        flags = os.O_RDONLY
        for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= getattr(os, name, 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise LargeMaskInpaintError(
                f"LaMa checkpoint cannot be opened safely: {path}"
            ) from exc
    try:
        _require_parent_chain(chain)
        opened = os.fstat(descriptor)
        _validate_checkpoint_status(path, opened)
        if (opened.st_dev, opened.st_ino) != (
            path_status.st_dev,
            path_status.st_ino,
        ):
            raise LargeMaskInpaintError(
                f"LaMa checkpoint identity changed: {path}"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _CHECKPOINT_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        stable = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError as exc:
            raise LargeMaskInpaintError(
                f"LaMa checkpoint identity changed: {path}"
            ) from exc
        _validate_checkpoint_status(path, stable)
        _validate_checkpoint_status(path, current)
        expected = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        )
        if expected != (
            stable.st_dev,
            stable.st_ino,
            stable.st_size,
            stable.st_mtime_ns,
        ) or expected != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ):
            raise LargeMaskInpaintError(
                f"LaMa checkpoint changed while being read: {path}"
            )
        _require_parent_chain(chain)
        return {
            "basename": path.name,
            "size": opened.st_size,
            "sha256": digest.hexdigest(),
        }
    finally:
        if owned_descriptor:
            os.close(descriptor)


def checkpoint_identity(path: str | Path) -> dict[str, str | int]:
    return _read_checkpoint_identity(_absolute_path(path))


def _validate_default_checkpoint(path: Path) -> None:
    identity = _read_checkpoint_identity(path)
    if (
        identity["size"] != BIG_LAMA_MODEL_SIZE
        or identity["sha256"] != BIG_LAMA_MODEL_SHA256
    ):
        raise LargeMaskInpaintError(
            "Big-LaMa checkpoint integrity verification failed"
        )


def _create_private_checkpoint(parent: Path) -> tuple[Path, int, tuple[int, int]]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    for _ in range(100):
        path = parent / f".big-lama-{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            continue
        status = os.fstat(descriptor)
        return path, descriptor, (status.st_dev, status.st_ino)
    raise LargeMaskInpaintError("Cannot allocate a private LaMa checkpoint file")


def _cleanup_owned_checkpoint(path: Path, identity: tuple[int, int]) -> None:
    try:
        status = path.lstat()
    except OSError:
        return
    if (
        _is_link_or_reparse(status)
        or not stat.S_ISREG(status.st_mode)
        or (status.st_dev, status.st_ino) != identity
    ):
        return
    try:
        path.unlink()
    except OSError:
        return


def resolve_lama_checkpoint(
    *,
    cache_dir: str | Path | None = None,
    downloader=urlretrieve,
) -> Path:
    custom = os.environ.get("LAMA_MODEL")
    if custom:
        path = _absolute_path(custom)
        checkpoint_identity(path)
        return path

    if cache_dir is None:
        cache_dir = os.environ.get("IMAGE2EDITABLE_MODEL_CACHE")
    if cache_dir is None:
        cache_dir = Path.home() / ".cache/image2editable/models/runtime"
    cache = _absolute_path(cache_dir)
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LargeMaskInpaintError(
            f"LaMa checkpoint cache cannot be created: {cache}"
        ) from exc
    chain = _snapshot_parent_chain(cache)
    target = cache / _CHECKPOINT_NAME
    if os.path.lexists(target):
        _validate_default_checkpoint(target)
        _require_parent_chain(chain)
        return target

    temporary, descriptor, temporary_identity = _create_private_checkpoint(cache)
    try:
        _require_parent_chain(chain)
        downloader(BIG_LAMA_MODEL_URL, temporary)
        os.fsync(descriptor)
        downloaded = _read_checkpoint_identity(
            temporary,
            descriptor=descriptor,
        )
        if (
            downloaded["size"] != BIG_LAMA_MODEL_SIZE
            or downloaded["sha256"] != BIG_LAMA_MODEL_SHA256
        ):
            raise LargeMaskInpaintError(
                "Big-LaMa checkpoint download failed integrity verification"
            )
        _require_parent_chain(chain)
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            _require_parent_chain(chain)
            _validate_default_checkpoint(target)
            _require_parent_chain(chain)
            return target
        _require_parent_chain(chain)
        os.close(descriptor)
        descriptor = None
        _cleanup_owned_checkpoint(temporary, temporary_identity)
        _require_parent_chain(chain)
        _validate_default_checkpoint(target)
        _require_parent_chain(chain)
        return target
    finally:
        if descriptor is not None:
            os.close(descriptor)
        _cleanup_owned_checkpoint(temporary, temporary_identity)


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
