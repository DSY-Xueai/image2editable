from __future__ import annotations

import hashlib
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from image2editable.contracts import SCHEMA_VERSION, RunStatus
from image2editable.store import RunStore


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
_HASH_CHUNK_SIZE = 1024 * 1024


def resolve_image_inputs(inputs: Iterable[str | Path]) -> list[Path]:
    resolved_inputs: list[Path] = []
    for value in inputs:
        path = Path(value).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_file():
            if path.suffix.casefold() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image input: {path}")
            resolved_inputs.append(path)
        elif path.is_dir():
            resolved_inputs.extend(
                child.resolve()
                for child in sorted(
                    path.iterdir(),
                    key=lambda child: (child.name.casefold(), child.name),
                )
                if not child.is_symlink()
                and child.is_file()
                and child.suffix.casefold() in IMAGE_EXTENSIONS
            )
    if not resolved_inputs:
        raise ValueError("No supported image inputs")
    return resolved_inputs


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _new_job_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def prepare_image_job(
    inputs: Iterable[str | Path],
    *,
    run_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    slide_size: str = "both",
    lang: str = "ch",
) -> Path:
    if slide_size not in {"original", "16:9", "both"}:
        raise ValueError(f"Unsupported slide_size: {slide_size}")

    source_paths = resolve_image_inputs(inputs)
    job_id = _new_job_id()
    root = Path(run_dir).resolve() if run_dir is not None else Path.cwd() / "runs" / job_id
    store = RunStore.create(root)
    try:
        items = []
        page_ids = []
        for index, source_path in enumerate(source_paths, start=1):
            page_id = f"page_{index:03d}"
            copied_relative = Path("input") / f"{index:03d}_{source_path.name}"
            copied_path = store.root / copied_relative
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, copied_path)
            digest = sha256_file(copied_path)
            relative_source = copied_relative.as_posix()
            store.write_json(
                Path("pages") / page_id / "page_request.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "page_id": page_id,
                    "source": relative_source,
                    "sha256": digest,
                },
            )
            items.append(
                {
                    "original_path": str(source_path),
                    "source": relative_source,
                    "sha256": digest,
                }
            )
            page_ids.append(page_id)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "input": {"type": "images", "items": items},
            "output_format": "pptx",
            "options": {
                "lang": lang,
                "slide_size": slide_size,
                "output_path": (
                    str(Path(output_path).resolve())
                    if output_path is not None
                    else None
                ),
            },
            "pages": page_ids,
        }
        store.initialize(manifest, page_ids)
        store.transition_run(RunStatus.PREPARED)
        return store.root
    except Exception:
        shutil.rmtree(store.root)
        store.root.mkdir(parents=True)
        raise
