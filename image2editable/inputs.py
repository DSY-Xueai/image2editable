from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Sequence

from image2editable.contracts import SCHEMA_VERSION, RunStatus
from image2editable.resources import safe_default_policy
from image2editable.store import RunStore


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
_HASH_CHUNK_SIZE = 1024 * 1024
InputType = Literal["images", "pdf", "pptx"]


def classify_inputs(
    inputs: str | Path | Iterable[str | Path],
) -> tuple[InputType, list[Path]]:
    values = (inputs,) if isinstance(inputs, (str, Path)) else list(inputs)
    if not values:
        raise ValueError("No inputs provided")

    resolved_paths = [Path(value).resolve() for value in values]
    for path in resolved_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

    document_paths = [
        path
        for path in resolved_paths
        if path.is_file() and path.suffix.casefold() in {".pdf", ".pptx"}
    ]
    if document_paths:
        if len(resolved_paths) != 1:
            raise ValueError("Inputs must contain one PDF or one PPTX")
        if document_paths[0].suffix.casefold() == ".pdf":
            return "pdf", resolved_paths
        return "pptx", resolved_paths

    return "images", resolve_image_inputs(values)


def resolve_image_inputs(
    inputs: str | Path | Iterable[str | Path],
) -> list[Path]:
    resolved_inputs: list[Path] = []
    values = (inputs,) if isinstance(inputs, (str, Path)) else inputs
    for value in values:
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


def new_job_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def validate_pptx_output_path(
    output_path: str | Path | None,
    *,
    source_paths: Sequence[Path],
    run_root: Path,
) -> Path | None:
    if output_path is None:
        return None
    resolved_output = Path(output_path).resolve()
    if resolved_output.suffix.casefold() != ".pptx":
        raise ValueError(f"Invalid output path; expected .pptx: {resolved_output}")
    if resolved_output.is_dir():
        raise ValueError(f"Invalid output path; path is a directory: {resolved_output}")
    if any(
        resolved_output == source.resolve()
        or (
            resolved_output.exists()
            and source.exists()
            and os.path.samefile(resolved_output, source)
        )
        for source in source_paths
    ):
        raise ValueError(f"Invalid output path; overwrites source: {resolved_output}")
    root = run_root.resolve()
    if resolved_output.is_relative_to(root) and not resolved_output.is_relative_to(
        root / "final"
    ):
        raise ValueError(
            f"Invalid output path; run outputs must be under final: {resolved_output}"
        )
    return resolved_output


def prepare_image_job(
    inputs: str | Path | Iterable[str | Path],
    *,
    run_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    slide_size: str = "both",
    lang: str = "ch",
) -> Path:
    if slide_size not in {"original", "16:9", "both"}:
        raise ValueError(f"Unsupported slide_size: {slide_size}")

    source_paths = resolve_image_inputs(inputs)
    job_id = new_job_id()
    root = Path(run_dir).resolve() if run_dir is not None else Path.cwd() / "runs" / job_id
    resolved_output = validate_pptx_output_path(
        output_path, source_paths=source_paths, run_root=root
    )
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
                    str(resolved_output) if resolved_output is not None else None
                ),
                "resource_policy": safe_default_policy(),
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
