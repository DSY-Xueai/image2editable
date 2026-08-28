from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


class RuntimeModelPathError(RuntimeError):
    """Raised when inference cannot resolve a verified local model."""


FILE_MODELS = {
    "sam2_large": (
        "SAM2_MODEL",
        898083611,
        "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    ),
    "big_lama": (
        "LAMA_MODEL",
        205803670,
        "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c",
    ),
}
MODEL_ENV = {
    "sam2_large": "SAM2_MODEL",
    "big_lama": "LAMA_MODEL",
    "grounding_dino": "GROUNDING_DINO_MODEL",
}
_CHUNK_SIZE = 1024 * 1024


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse
    )


def _absolute_override(value: str, env_name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeModelPathError(f"{env_name} must be an absolute local path")
    return path


def _verified_file(path: Path, env_name: str, size: int, sha256: str) -> Path:
    try:
        before = path.lstat()
    except OSError:
        raise RuntimeModelPathError(f"{env_name} model file is missing") from None
    if (
        _is_link_or_reparse(before)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise RuntimeModelPathError(f"{env_name} must name a regular non-link file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise RuntimeModelPathError(
            f"{env_name} model file cannot be opened safely"
        ) from None
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeModelPathError(f"{env_name} model file identity changed")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError:
            raise RuntimeModelPathError(
                f"{env_name} model file identity changed"
            ) from None
    finally:
        os.close(descriptor)
    expected = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != expected
        or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        != expected
        or opened.st_nlink != 1
        or after.st_nlink != 1
        or current.st_nlink != 1
        or opened.st_size != size
        or digest.hexdigest() != sha256
    ):
        raise RuntimeModelPathError(
            f"{env_name} model file failed integrity verification"
        )
    return path.resolve()


def _explicit_model_path(name: str, value: str) -> Path:
    env_name = MODEL_ENV[name]
    path = _absolute_override(value, env_name)
    if name in FILE_MODELS:
        _, size, sha256 = FILE_MODELS[name]
        return _verified_file(path, env_name, size, sha256)
    # The explicit directory is an operator-trusted override; product defaults
    # remain bound to the strict runtime receipt resolver below.
    try:
        status = path.lstat()
    except OSError:
        raise RuntimeModelPathError(f"{env_name} snapshot directory is missing") from None
    if _is_link_or_reparse(status) or not stat.S_ISDIR(status.st_mode):
        raise RuntimeModelPathError(f"{env_name} must name a non-link directory")
    return path.resolve()


def _product_runtime_model_path(name: str) -> Path:
    from image2editable.runtime_models import runtime_model_path

    return runtime_model_path(name)


def resolve_runtime_model_path(name: str) -> Path:
    try:
        env_name = MODEL_ENV[name]
    except KeyError:
        raise RuntimeModelPathError(f"Unknown runtime model: {name}") from None
    override = os.environ.get(env_name)
    if override:
        return _explicit_model_path(name, override)
    try:
        return Path(_product_runtime_model_path(name))
    except ModuleNotFoundError as exc:
        if exc.name not in {"image2editable", "image2editable.runtime_models"}:
            raise
        raise RuntimeModelPathError(
            f"image2editable is unavailable; set {env_name} to an absolute local path"
        ) from None
    except RuntimeError as exc:
        raise RuntimeModelPathError(str(exc)) from None
