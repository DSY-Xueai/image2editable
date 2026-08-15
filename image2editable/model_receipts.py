from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat


_CHUNK_SIZE = 1024 * 1024


def canonical_sha256(document: object) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse
    )


def _unsafe_file(status: os.stat_result) -> bool:
    return (
        _is_link_or_reparse(status)
        or not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
    )


def _identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_strict_file(
    path: Path,
    boundary: Path,
    *,
    capture: bool = False,
) -> tuple[dict[str, object], bytes | None]:
    status = path.lstat()
    if _unsafe_file(status):
        raise RuntimeError(f"model file is not a private regular file: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(boundary.resolve()):
        raise RuntimeError(f"model file is outside the model cache: {path}")
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _unsafe_file(opened) or _identity(opened)[:2] != _identity(status)[:2]:
            raise RuntimeError(f"model file identity changed: {path}")
        digest = hashlib.sha256()
        chunks = [] if capture else None
        total = 0
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
                total += len(chunk)
                if total > 4 * 1024 * 1024:
                    raise RuntimeError(f"model receipt is too large: {path}")
        stable = os.fstat(descriptor)
        current = path.lstat()
        if (
            _identity(opened) != _identity(stable)
            or _identity(opened) != _identity(current)
            or _unsafe_file(current)
        ):
            raise RuntimeError(f"model file changed while being read: {path}")
        return (
            {
                "path": path.name,
                "size": opened.st_size,
                "sha256": digest.hexdigest(),
            },
            b"".join(chunks) if chunks is not None else None,
        )
    finally:
        os.close(descriptor)


def strict_file_record(path: Path, boundary: Path) -> dict[str, object]:
    record, _ = _read_strict_file(path, boundary)
    return record


def read_strict_json(path: Path, boundary: Path) -> object:
    _, payload = _read_strict_file(path, boundary, capture=True)
    if payload is None:
        raise RuntimeError(f"model receipt could not be read: {path}")
    return json.loads(payload.decode("utf-8"))


def manifest_files(
    snapshot: Path,
    boundary: Path,
    *,
    strict: bool = False,
) -> list[dict[str, object]]:
    files = []
    for path in sorted(snapshot.rglob("*"), key=lambda value: value.as_posix()):
        if strict:
            status = path.lstat()
            if _is_link_or_reparse(status):
                raise RuntimeError(f"model snapshot contains a link: {path}")
            if stat.S_ISDIR(status.st_mode):
                continue
            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise RuntimeError(f"model snapshot contains an unsafe file: {path}")
            record = strict_file_record(path, boundary)
            record["path"] = path.relative_to(snapshot).as_posix()
            files.append(record)
            continue
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(boundary.resolve()):
            raise RuntimeError("model snapshot contains a file outside the cache")
        files.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise RuntimeError("downloaded model snapshot is empty")
    return files


def validate_manifest(
    snapshot: Path,
    boundary: Path,
    files: object,
    *,
    strict: bool = False,
    require_sorted: bool = False,
) -> None:
    if not isinstance(files, list) or not files:
        raise RuntimeError("receipt file manifest is empty")
    expected = {}
    ordered_paths = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise RuntimeError("receipt file entry is invalid")
        relative = Path(item["path"])
        if (
            not isinstance(item["path"], str)
            or not item["path"]
            or relative.is_absolute()
            or ".." in relative.parts
            or item["path"] in expected
        ):
            raise RuntimeError(f"snapshot file path is invalid: {item['path']}")
        if (
            type(item["size"]) is not int
            or item["size"] < 0
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            raise RuntimeError(f"receipt file entry is invalid: {item['path']}")
        expected[item["path"]] = item
        ordered_paths.append(item["path"])
    if require_sorted and ordered_paths != sorted(ordered_paths):
        raise RuntimeError("receipt file manifest is not sorted")
    actual = {
        item["path"]: item
        for item in manifest_files(snapshot, boundary, strict=strict)
    }
    if set(actual) != set(expected):
        raise RuntimeError("snapshot file set does not match receipt")
    for relative_path, item in expected.items():
        if (
            actual[relative_path]["size"] != item["size"]
            or actual[relative_path]["sha256"] != item["sha256"]
        ):
            raise RuntimeError(f"snapshot file checksum mismatch: {relative_path}")
