from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from image2editable.contracts import (
    SCHEMA_VERSION,
    PageStatus,
    RunStatus,
    transition_page_document,
    transition_run_document,
    utc_now,
    validate_schema_version,
)


class RunStore:
    """P0 single-writer store; the parent runtime serializes state transitions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    @classmethod
    def create(cls, root: str | Path) -> RunStore:
        resolved_root = Path(root).resolve()
        if resolved_root.exists():
            if not resolved_root.is_dir() or any(resolved_root.iterdir()):
                raise FileExistsError(f"Run directory is not empty: {resolved_root}")
        else:
            resolved_root.mkdir(parents=True)
        return cls(resolved_root)

    @classmethod
    def open(cls, root: str | Path) -> RunStore:
        store = cls(root)
        for relative in ("job_manifest.json", "run_state.json", "page_jobs.json"):
            path = store._resolve(relative)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Not an image2editable run; missing {path}"
                )
            document = store.read_json(relative)
            try:
                validate_schema_version(document)
            except ValueError as error:
                raise ValueError(f"Invalid {path}: {error}") from error
        return store

    def _resolve(self, relative: str | Path) -> Path:
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"Path is outside run directory: {relative}")
        return target

    def write_json(self, relative: str | Path, document: dict[str, Any]) -> None:
        target = self._resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                json.dump(
                    document,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                file.write("\n")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def read_json(self, relative: str | Path) -> dict[str, Any]:
        path = self._resolve(relative)
        try:
            with path.open(encoding="utf-8") as file:
                document = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}: {error.msg}") from error
        if not isinstance(document, dict):
            raise ValueError(f"JSON document must be an object: {path}")
        return document

    def initialize(self, job_manifest: dict[str, Any], page_ids: list[str]) -> None:
        validate_schema_version(job_manifest)
        now = utc_now()
        self.write_json(
            "run_state.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": RunStatus.CREATED.value,
                "updated_at": now,
            },
        )
        self.write_json(
            "page_jobs.json",
            {
                "schema_version": SCHEMA_VERSION,
                "pages": {
                    page_id: {
                        "schema_version": SCHEMA_VERSION,
                        "status": PageStatus.PENDING.value,
                        "updated_at": now,
                    }
                    for page_id in page_ids
                }
            },
        )
        self.write_json("job_manifest.json", job_manifest)

    def transition_run(self, target: RunStatus) -> dict[str, Any]:
        updated = transition_run_document(self.read_json("run_state.json"), target)
        self.write_json("run_state.json", updated)
        return updated

    def transition_page(self, page_id: str, target: PageStatus) -> dict[str, Any]:
        page_jobs = self.read_json("page_jobs.json")
        pages = page_jobs["pages"]
        if page_id not in pages:
            raise KeyError(f"Unknown page_id: {page_id}")
        updated = transition_page_document(pages[page_id], target)
        pages[page_id] = updated
        self.write_json("page_jobs.json", page_jobs)
        return updated
