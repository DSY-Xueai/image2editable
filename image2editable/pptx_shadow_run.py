from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from pptx import Presentation

from image2editable.inputs import sha256_file
from image2editable.pptx_input import (
    _publish_pptx_no_clobber,
    scan_pptx,
)
from image2editable.pptx_reconstruct import build_reconstruction_donor, build_reconstruction_donor_from_result
from image2editable.pptx_shadow import patch_slide_background


def run_shadow_replacements(
    source_pptx: str | Path,
    output_pptx: str | Path,
    plans: list[dict],
    *,
    run_root: str | Path,
    lang: str = "ch",
) -> dict:
    """Apply approved screenshot replacements sequentially with page fallback."""
    source = Path(source_pptx).resolve()
    output = Path(output_pptx).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trusted_run_root = _trusted_directory(Path(run_root), "run")

    current = source
    current_is_temporary = False
    page_results = []
    published_temporary = None
    try:
        for plan in plans:
            result, staged = _run_page(
                current,
                output.parent,
                trusted_run_root,
                plan,
                lang,
            )
            page_results.append(result)
            if staged is None:
                continue
            if current_is_temporary:
                current.unlink(missing_ok=True)
            current = staged
            current_is_temporary = True

        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        published_temporary = Path(temporary_name)
        shutil.copyfile(current, published_temporary)
        output_identity = _publish_pptx_no_clobber(
            published_temporary,
            output,
        )
        return {
            "page_results": page_results,
            "output_sha256": sha256_file(output),
            "_output_identity": output_identity,
        }
    finally:
        if current_is_temporary:
            current.unlink(missing_ok=True)
        if published_temporary is not None:
            published_temporary.unlink(missing_ok=True)


def _run_page(
    current: Path,
    staging_root: Path,
    run_root: Path,
    plan: dict,
    lang: str,
) -> tuple[dict, Path | None]:
    page_id = plan["page_id"]
    page_root = _trusted_page_root(run_root, page_id)
    result_path = page_root / "replacement_result.json"
    staged = _unused_path(staging_root, page_id)
    try:
        work_root = _trusted_work_root(page_root, plan.get("work_root"))
        donor = work_root / "donor.pptx"
        if plan.get("conflict_warning"):
            raise RuntimeError(plan["conflict_warning"])
        # Host runs may only consume the durable component-result boundary.
        # While the Host Agent is awaiting a plan there is no accepted result;
        # never fall back to the legacy CV/OCR donor builder in that state.
        if plan.get("provider") == "host" and not plan.get("component_result_path"):
            raise RuntimeError(
                "host shadow replacement requires an accepted component result"
            )
        if plan.get("component_result_path"):
            donor_kwargs = {
                "source_screenshot_sha256": plan["source_screenshot_sha256"],
                "provider": plan.get("provider", "host"),
                "initial_component_count": plan.get("initial_component_count", 0),
            }
            if plan.get("component_result_sha256") is not None:
                donor_kwargs.update({
                    "expected_result_sha256": plan["component_result_sha256"],
                    "run_root": run_root,
                })
            reconstruction = build_reconstruction_donor_from_result(
                plan["component_result_path"], donor, work_root, **donor_kwargs
            )
        else:
            reconstruction = build_reconstruction_donor(
                plan["image_path"], donor, work_root,
                decision=plan["decision"], lang=lang,
            )
        replacement = patch_slide_background(
            current,
            donor,
            staged,
            slide_part=plan["slide_part"],
            source_shape_id=plan["decision"]["source_shape_id"],
        )
        _validate_shadow_patch(current, staged, plan)
    except Exception as error:
        staged.unlink(missing_ok=True)
        result = {
            "schema_version": 1,
            "page_id": page_id,
            "status": "preserved_with_warning",
            "warning": str(error),
            "error_type": type(error).__name__,
        }
        _write_json(result_path, result)
        return result, None

    result = {
        "schema_version": 1,
        "page_id": page_id,
        "status": "replaced",
        "reconstruction": reconstruction,
        "replacement": replacement,
    }
    _write_json(result_path, result)
    return result, staged


def _trusted_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise RuntimeError(f"PPTX {label} directory must be absolute: {path}")
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"PPTX {label} directory is missing: {path}"
        ) from error
    if _is_link_or_reparse(status):
        raise RuntimeError(
            f"PPTX {label} directory is a link or reparse point: {path}"
        )
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"PPTX {label} path is not a directory: {path}")
    return path.resolve()


def _trusted_page_root(run_root: Path, page_id: object) -> Path:
    if (
        not isinstance(page_id, str)
        or not page_id
        or Path(page_id).name != page_id
    ):
        raise RuntimeError(f"PPTX page_id is invalid: {page_id!r}")
    current = run_root
    for name in ("pages", page_id):
        current = _trusted_directory(current / name, "page")
        if not current.is_relative_to(run_root):
            raise RuntimeError(
                f"PPTX page directory is outside run directory: {current}"
            )
    return current


def _trusted_work_root(page_root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("PPTX plan work_root must be a string")
    planned = Path(value)
    expected = page_root / "reconstruction"
    if (
        not planned.is_absolute()
        or os.path.normcase(os.path.abspath(planned))
        != os.path.normcase(os.path.abspath(expected))
    ):
        raise RuntimeError(
            "PPTX plan work_root must be the page reconstruction directory"
        )
    try:
        status = expected.lstat()
    except FileNotFoundError:
        return expected
    if _is_link_or_reparse(status):
        raise RuntimeError(
            "PPTX reconstruction directory is a link or reparse point: "
            f"{expected}"
        )
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(
            f"PPTX reconstruction path is not a directory: {expected}"
        )
    resolved = expected.resolve()
    if not resolved.is_relative_to(page_root):
        raise RuntimeError(
            "PPTX reconstruction directory is outside its page directory: "
            f"{expected}"
        )
    return resolved


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )


def _validate_shadow_patch(
    before_path: Path,
    after_path: Path,
    plan: dict,
) -> None:
    before = scan_pptx(before_path)
    after = scan_pptx(after_path)
    if (
        before["slide_count"] != after["slide_count"]
        or before["slide_width"] != after["slide_width"]
        or before["slide_height"] != after["slide_height"]
    ):
        raise RuntimeError("shadow PPTX changed presentation structure")

    slide_part = plan["slide_part"]
    before_slide = _slide_by_part(before, slide_part)
    after_slide = _slide_by_part(after, slide_part)
    if before_slide["notes_sha256"] != after_slide["notes_sha256"]:
        raise RuntimeError("shadow PPTX changed slide notes")

    source_shape_id = plan["decision"]["source_shape_id"]
    protected = Counter(
        item["xml_c14n_sha256"]
        for item in before_slide["objects"]
        if item["shape_id"] != source_shape_id
    )
    actual = Counter(
        item["xml_c14n_sha256"] for item in after_slide["objects"]
    )
    if protected - actual:
        raise RuntimeError("shadow PPTX changed protected native objects")
    if any(
        item["shape_id"] == source_shape_id
        for item in after_slide["objects"]
    ):
        raise RuntimeError("shadow PPTX kept the screenshot background")

    before_other = {
        item["slide_part"]: item["sha256"]
        for item in before["slides"]
        if item["slide_part"] != slide_part
    }
    after_other = {
        item["slide_part"]: item["sha256"]
        for item in after["slides"]
        if item["slide_part"] != slide_part
    }
    if before_other != after_other:
        raise RuntimeError("shadow PPTX changed an unrelated slide")
    reopened = Presentation(after_path)
    if len(reopened.slides) != after["slide_count"]:
        raise RuntimeError("shadow PPTX cannot be reopened safely")


def _slide_by_part(inventory: dict, slide_part: str) -> dict:
    for slide in inventory["slides"]:
        if slide["slide_part"] == slide_part:
            return slide
    raise RuntimeError(f"shadow PPTX slide is missing: {slide_part}")


def _unused_path(directory: Path, page_id: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=directory,
        prefix=f".{page_id}-",
        suffix=".pptx",
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
