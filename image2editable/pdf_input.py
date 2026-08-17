from __future__ import annotations

import hashlib
import math
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Sequence

import pypdfium2 as pdfium

from image2editable.component_contracts import validate_agent_provider
from image2editable.contracts import SCHEMA_VERSION, RunStatus
from image2editable.inputs import (
    new_job_id,
    sha256_file,
    validate_pptx_output_path,
)
from image2editable.resources import safe_default_policy
from image2editable.store import RunStore


STANDARD_DPI = 200.0
DETAIL_DPI = 300.0
SHORT_EDGE_FLOOR = 1200
STANDARD_LONG_EDGE_CEILING = 2560
LONG_EDGE_CEILING = 6000
PIXEL_COUNT_CEILING = 24_000_000

RenderProfile = Literal["standard", "detail"]


def pdf_page_count(path: str | Path) -> int:
    try:
        document = pdfium.PdfDocument(str(Path(path).resolve()))
        try:
            count = len(document)
        finally:
            document.close()
    except pdfium.PdfiumError as error:
        if error.err_code == pdfium.raw.FPDF_ERR_PASSWORD:
            raise ValueError(f"Cannot open encrypted PDF: {path}") from error
        raise ValueError(f"Cannot open PDF: {path}") from error
    except Exception as error:
        raise ValueError(f"Cannot open PDF: {path}") from error
    if count == 0:
        raise ValueError(f"Cannot open PDF: {path} has no pages")
    return count


def prepare_pdf_job(
    source: str | Path,
    *,
    run_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    slide_size: str = "both",
    lang: str = "ch",
    agent_provider: str = "host",
) -> Path:
    agent_provider = validate_agent_provider(agent_provider)
    source_path = Path(source).resolve()
    if not source_path.is_file() or source_path.suffix.casefold() != ".pdf":
        raise ValueError(f"PDF input must be an existing .pdf file: {source_path}")
    if slide_size not in {"original", "16:9", "both"}:
        raise ValueError(f"Unsupported slide_size: {slide_size}")

    page_count = pdf_page_count(source_path)
    job_id = new_job_id()
    root = Path(run_dir).resolve() if run_dir is not None else Path.cwd() / "runs" / job_id
    resolved_output = validate_pptx_output_path(
        output_path, source_paths=[source_path], run_root=root
    )
    store = RunStore.create(root)
    try:
        copied_relative = Path("input") / "original.pdf"
        copied_path = store.root / copied_relative
        copied_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, copied_path)
        digest = sha256_file(copied_path)
        page_ids = [f"page_{index:03d}" for index in range(1, page_count + 1)]
        output_paths = [
            store.root / "pages" / page_id / "source.png" for page_id in page_ids
        ]
        renders = render_pdf_document(copied_path, output_paths, profile="standard")
        ratios = [record["width_pt"] / record["height_pt"] for record in renders]
        page_ratios_equal = math.isfinite(ratios[0]) and all(
            math.isfinite(ratio)
            and math.isclose(ratio, ratios[0], rel_tol=1e-4, abs_tol=1e-6)
            for ratio in ratios[1:]
        )
        page_aspect_ratio = ratios[0] if page_ratios_equal else None
        for page_id, render in zip(page_ids, renders):
            source_relative = (Path("pages") / page_id / "source.png").as_posix()
            store.write_json(
                Path("pages") / page_id / "page_request.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "page_id": page_id,
                    "source_type": "pdf",
                    "source": source_relative,
                    "sha256": render["sha256"],
                    "render": render,
                },
            )
            store.write_json(
                Path("pages") / page_id / "render_history.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "renders": [render],
                    "detail_used": False,
                },
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "input": {
                "type": "pdf",
                "original_path": str(source_path),
                "source": copied_relative.as_posix(),
                "sha256": digest,
                "page_count": page_count,
                "page_ratios_equal": page_ratios_equal,
                "page_aspect_ratio": page_aspect_ratio,
            },
            "output_format": "pptx",
            "options": {
                "agent_provider": agent_provider,
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
    except Exception as error:
        cleanup_error: Exception | None = None
        try:
            shutil.rmtree(store.root)
        except Exception as caught:
            cleanup_error = caught
        try:
            store.root.mkdir(parents=True, exist_ok=True)
        except Exception as caught:
            if cleanup_error is None:
                cleanup_error = caught
        if cleanup_error is not None:
            raise error from cleanup_error
        raise


def _remove_files(paths: Sequence[Path]) -> Exception | None:
    first_error: Exception | None = None
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception as error:
            if first_error is None:
                first_error = error
    return first_error


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode):
        raise RuntimeError(f"PDF detail source is not a regular file: {path}")
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _validate_pdf_detail_source(
    store: RunStore,
    input_record: dict[str, object],
) -> tuple[Path, tuple[int, int, int, int, int], str]:
    source_value = input_record.get("source")
    if type(source_value) is not str:
        raise RuntimeError("PDF detail source must be a relative path")
    relative = Path(source_value)
    if (
        not source_value
        or relative.is_absolute()
        or relative.as_posix() != source_value
        or ".." in relative.parts
        or relative.suffix.casefold() != ".pdf"
    ):
        raise RuntimeError(f"Invalid PDF detail source path: {source_value}")
    expected_sha256 = input_record.get("sha256")
    if not _is_sha256(expected_sha256):
        raise RuntimeError("PDF detail source sha256 is invalid")

    lexical_path = store.root / relative
    current = store.root
    try:
        for part in relative.parts:
            current /= part
            status = current.lstat()
            if stat.S_ISLNK(status.st_mode):
                raise RuntimeError(
                    f"PDF detail source contains a symlink: {source_value}"
                )
        resolved = lexical_path.resolve()
    except RuntimeError:
        raise
    except OSError as error:
        raise RuntimeError(
            f"PDF detail source is missing: {source_value}"
        ) from error
    if resolved != lexical_path or not resolved.is_relative_to(store.root):
        raise RuntimeError(
            f"PDF detail source contains a symlink: {source_value}"
        )

    identity = _file_identity(lexical_path)
    digest = sha256_file(lexical_path)
    if _file_identity(lexical_path) != identity:
        raise RuntimeError(
            "PDF detail source changed during verification"
        )
    if digest != expected_sha256:
        raise RuntimeError("PDF detail source hash does not match manifest")
    return lexical_path, identity, expected_sha256


def _verify_pdf_detail_source_after_render(
    path: Path,
    source_file: BinaryIO,
    expected_identity: tuple[int, int, int, int, int],
    expected_sha256: str,
) -> None:
    try:
        identity = _open_file_identity(source_file)
        digest = _sha256_open_file(source_file)
        stable_identity = _open_file_identity(source_file)
        path_identity = _file_identity(path)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            "PDF detail source changed during rendering"
        ) from error
    if (
        identity != expected_identity
        or stable_identity != expected_identity
        or path_identity != expected_identity
    ):
        raise RuntimeError("PDF detail source changed during rendering")
    if digest != expected_sha256:
        raise RuntimeError(
            "PDF detail source hash changed during rendering"
        )


def _open_file_identity(source_file: BinaryIO) -> tuple[int, int, int, int, int]:
    status = os.fstat(source_file.fileno())
    if not stat.S_ISREG(status.st_mode):
        raise RuntimeError("PDF detail source descriptor is not a regular file")
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _sha256_open_file(source_file: BinaryIO) -> str:
    digest = hashlib.sha256()
    source_file.seek(0)
    while chunk := source_file.read(1024 * 1024):
        digest.update(chunk)
    source_file.seek(0)
    return digest.hexdigest()


def _copy_pdf_detail_snapshot(
    source_file: BinaryIO, snapshot_file: BinaryIO
) -> None:
    source_file.seek(0)
    snapshot_file.seek(0)
    snapshot_file.truncate()
    while chunk := source_file.read(1024 * 1024):
        snapshot_file.write(chunk)
    snapshot_file.flush()
    source_file.seek(0)
    snapshot_file.seek(0)


def _verify_pdf_detail_snapshot_after_render(
    snapshot_file: BinaryIO,
    expected_identity: tuple[int, int, int, int, int],
    expected_sha256: str,
) -> None:
    identity = _open_file_identity(snapshot_file)
    digest = _sha256_open_file(snapshot_file)
    stable_identity = _open_file_identity(snapshot_file)
    if identity != expected_identity or stable_identity != expected_identity:
        raise RuntimeError("PDF detail snapshot changed during rendering")
    if digest != expected_sha256:
        raise RuntimeError("PDF detail snapshot hash changed during rendering")


def _open_verified_pdf_detail_source(
    path: Path,
    expected_identity: tuple[int, int, int, int, int],
    expected_sha256: str,
) -> BinaryIO:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(
            "PDF detail source changed before rendering"
        ) from error
    try:
        source_file = os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise
    try:
        identity = _open_file_identity(source_file)
        digest = _sha256_open_file(source_file)
        stable_identity = _open_file_identity(source_file)
        path_identity = _file_identity(path)
        if (
            identity != expected_identity
            or stable_identity != expected_identity
            or path_identity != expected_identity
        ):
            raise RuntimeError(
                "PDF detail source changed before rendering"
            )
        if digest != expected_sha256:
            raise RuntimeError(
                "PDF detail source hash changed before rendering"
            )
        return source_file
    except Exception:
        source_file.close()
        raise


def rerender_pdf_page(run_dir: str | Path, page_id: str) -> dict[str, bool]:
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    if manifest.get("input", {}).get("type") != "pdf":
        raise RuntimeError("Detail rendering is only available for PDF runs")
    if store.read_json("run_state.json")["status"] != RunStatus.PREPARED.value:
        raise RuntimeError("Run must be prepared for detail rendering")
    page_jobs = store.read_json("page_jobs.json")["pages"]
    if page_id not in page_jobs:
        raise KeyError(f"Unknown page_id: {page_id}")
    if page_jobs[page_id]["status"] != "pending":
        raise RuntimeError(f"Page must be pending for detail rendering: {page_id}")

    request_path = Path("pages") / page_id / "page_request.json"
    history_path = Path("pages") / page_id / "render_history.json"
    request = store.read_json(request_path)
    history = store.read_json(history_path)
    if history["detail_used"]:
        raise RuntimeError(f"Detail rendering already used for page: {page_id}")
    source_path, source_identity, source_sha256 = (
        _validate_pdf_detail_source(store, manifest["input"])
    )

    standard = history["renders"][0]
    standard_request = dict(request)
    standard_request.update(
        {
            "source": (Path("pages") / page_id / "source.png").as_posix(),
            "sha256": standard["sha256"],
            "render": standard,
        }
    )
    detail_relative = (Path("pages") / page_id / "source_detail.png").as_posix()
    detail_path = store.root / detail_relative
    detail_temp_path = detail_path.with_name(f"{detail_path.name}.tmp")
    recovery_error: Exception | None = None
    try:
        store.write_json(request_path, standard_request)
    except Exception as error:
        recovery_error = error
    cleanup_error = _remove_files((detail_temp_path, detail_path))
    if recovery_error is not None:
        if cleanup_error is not None:
            raise recovery_error from cleanup_error
        raise recovery_error
    if cleanup_error is not None:
        raise cleanup_error

    try:
        with _open_verified_pdf_detail_source(
            source_path,
            source_identity,
            source_sha256,
        ) as source_file:
            with tempfile.TemporaryFile(mode="w+b") as snapshot_file:
                _copy_pdf_detail_snapshot(source_file, snapshot_file)
                snapshot_identity = _open_file_identity(snapshot_file)
                snapshot_digest = _sha256_open_file(snapshot_file)
                stable_snapshot_identity = _open_file_identity(snapshot_file)
                if (
                    snapshot_identity != stable_snapshot_identity
                    or snapshot_digest != source_sha256
                ):
                    raise RuntimeError(
                        "PDF detail snapshot hash does not match manifest"
                    )
                if (
                    _open_file_identity(source_file) != source_identity
                    or _file_identity(source_path) != source_identity
                ):
                    raise RuntimeError(
                        "PDF detail source changed while creating snapshot"
                    )
                detail = _render_pdf_page_from_stream(
                    snapshot_file,
                    standard["page_index"],
                    detail_temp_path,
                    profile="detail",
                )
                _verify_pdf_detail_snapshot_after_render(
                    snapshot_file,
                    snapshot_identity,
                    source_sha256,
                )
            _verify_pdf_detail_source_after_render(
                source_path,
                source_file,
                source_identity,
                source_sha256,
            )
        activated = (
            detail["pixel_width"] > standard["pixel_width"]
            and detail["pixel_height"] > standard["pixel_height"]
        )
        if activated:
            detail["result"] = "detail_activated"
            detail["source"] = detail_relative
            request["source"] = detail_relative
            request["sha256"] = detail["sha256"]
            request["render"] = detail
            os.replace(detail_temp_path, detail_path)
            store.write_json(request_path, request)
        else:
            detail_temp_path.unlink(missing_ok=True)
            detail["result"] = "detail_not_higher"
            detail["source"] = None
        history["renders"].append(detail)
        history["detail_used"] = True
        store.write_json(history_path, history)
    except Exception as error:
        cleanup_error: Exception | None = None
        try:
            store.write_json(request_path, standard_request)
        except Exception as caught:
            cleanup_error = caught
        file_cleanup_error = _remove_files((detail_temp_path, detail_path))
        if cleanup_error is None:
            cleanup_error = file_cleanup_error
        if cleanup_error is not None:
            raise error from cleanup_error
        raise
    return {"detail_used": True, "activated": activated}


@dataclass(frozen=True)
class PdfRenderPlan:
    profile: RenderProfile
    target_dpi: float
    effective_dpi: float
    scale: float
    pixel_width: int
    pixel_height: int
    reasons: tuple[str, ...]


def _pixel_dimensions(width_pt: float, height_pt: float, scale: float) -> tuple[int, int]:
    return max(1, math.ceil(width_pt * scale)), max(1, math.ceil(height_pt * scale))


def _fits_hard_limits(width_pt: float, height_pt: float, scale: float) -> bool:
    width, height = _pixel_dimensions(width_pt, height_pt, scale)
    return max(width, height) <= LONG_EDGE_CEILING and width * height <= PIXEL_COUNT_CEILING


def _integer_safe_scale(width_pt: float, height_pt: float, scale: float) -> float:
    if _fits_hard_limits(width_pt, height_pt, scale):
        return scale
    low = 0.0
    high = scale
    for _ in range(64):
        middle = (low + high) / 2
        if _fits_hard_limits(width_pt, height_pt, middle):
            low = middle
        else:
            high = middle
    return math.nextafter(low, 0.0)


def plan_pdf_render(width_pt: float, height_pt: float, profile: RenderProfile) -> PdfRenderPlan:
    if not isinstance(width_pt, (int, float)) or not isinstance(height_pt, (int, float)):
        raise ValueError("PDF page dimensions must be positive numbers")
    if not math.isfinite(width_pt) or not math.isfinite(height_pt) or width_pt <= 0 or height_pt <= 0:
        raise ValueError("PDF page dimensions must be positive numbers")
    if profile not in ("standard", "detail"):
        raise ValueError(f"Unsupported PDF render profile: {profile}")

    target_dpi = STANDARD_DPI if profile == "standard" else DETAIL_DPI
    scale = target_dpi / 72.0
    reasons: list[str] = []
    if profile == "standard" and min(width_pt, height_pt) * scale < SHORT_EDGE_FLOOR:
        scale = min(SHORT_EDGE_FLOOR / min(width_pt, height_pt), DETAIL_DPI / 72.0)
        reasons.append("short_edge_floor")

    if profile == "standard" and max(width_pt, height_pt) * scale > STANDARD_LONG_EDGE_CEILING:
        scale = STANDARD_LONG_EDGE_CEILING / max(width_pt, height_pt)
        reasons.append("standard_long_edge_ceiling")

    if max(width_pt, height_pt) * scale > LONG_EDGE_CEILING:
        scale = LONG_EDGE_CEILING / max(width_pt, height_pt)
        reasons.append("long_edge_ceiling")
    pixel_cap_scale = (
        math.sqrt(PIXEL_COUNT_CEILING)
        / math.sqrt(width_pt)
        / math.sqrt(height_pt)
    )
    if scale > pixel_cap_scale:
        scale = pixel_cap_scale
        reasons.append("pixel_count_ceiling")

    scale = _integer_safe_scale(width_pt, height_pt, scale)
    pixel_width, pixel_height = _pixel_dimensions(width_pt, height_pt, scale)
    return PdfRenderPlan(
        profile=profile,
        target_dpi=target_dpi,
        effective_dpi=scale * 72.0,
        scale=scale,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        reasons=tuple(reasons),
    )


def _box_values(box: Sequence[float] | None) -> list[float] | None:
    return [float(value) for value in box] if box is not None else None


def _renderer_version() -> str:
    version = getattr(pdfium, "__version__", None)
    if version is not None:
        return str(version)
    return str(pdfium.version.PYPDFIUM_INFO)


def _same_file(first: Path, second: Path) -> bool:
    return first == second or (
        first.exists() and second.exists() and os.path.samefile(first, second)
    )


def _normalize_render_paths(
    source: str | Path, outputs: Sequence[str | Path]
) -> tuple[Path, tuple[Path, ...]]:
    source_path = Path(source).resolve()
    output_paths = tuple(Path(output).resolve() for output in outputs)
    for index, output in enumerate(output_paths):
        if _same_file(source_path, output):
            raise ValueError(f"PDF output must not overwrite source: {output}")
        if any(_same_file(output, prior) for prior in output_paths[:index]):
            raise ValueError(f"PDF outputs must be unique: {output}")
    return source_path, output_paths


def _render_open_page(
    document: pdfium.PdfDocument,
    index: int,
    output: str | Path,
    profile: RenderProfile,
) -> dict[str, object]:
    page = document[index]
    try:
        width_pt = float(page.get_width())
        height_pt = float(page.get_height())
        plan = plan_pdf_render(width_pt, height_pt, profile)
        media_box = _box_values(page.get_mediabox(fallback_ok=False))
        crop_box = _box_values(page.get_cropbox(fallback_ok=False))
        bitmap = page.render(scale=plan.scale)
        try:
            image = bitmap.to_pil()
            try:
                target = Path(output)
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target, format="PNG")
                pixel_width, pixel_height = image.size
            finally:
                image.close()
        finally:
            bitmap.close()
        return {
            "page_index": index,
            "page_number": index + 1,
            "width_pt": width_pt,
            "height_pt": height_pt,
            "rotation": int(page.get_rotation()),
            "media_box": media_box,
            "crop_box": crop_box,
            "profile": profile,
            "target_dpi": plan.target_dpi,
            "effective_dpi": plan.effective_dpi,
            "scale": plan.scale,
            "reasons": list(plan.reasons),
            "pixel_width": pixel_width,
            "pixel_height": pixel_height,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "renderer": "pypdfium2",
            "renderer_version": _renderer_version(),
        }
    finally:
        page.close()


def render_pdf_page(
    source: str | Path,
    index: int,
    output: str | Path,
    *,
    profile: RenderProfile,
) -> dict[str, object]:
    source_path, (output_path,) = _normalize_render_paths(source, [output])
    document = pdfium.PdfDocument(str(source_path))
    try:
        return _render_open_page(document, index, output_path, profile)
    finally:
        document.close()


def _render_pdf_page_from_stream(
    source_file: BinaryIO,
    index: int,
    output: str | Path,
    *,
    profile: RenderProfile,
) -> dict[str, object]:
    document = pdfium.PdfDocument(source_file)
    try:
        return _render_open_page(document, index, Path(output).resolve(), profile)
    finally:
        document.close()


def render_pdf_document(
    source: str | Path,
    outputs: Sequence[str | Path],
    *,
    profile: RenderProfile,
) -> list[dict[str, object]]:
    source_path, output_paths = _normalize_render_paths(source, outputs)
    document = pdfium.PdfDocument(str(source_path))
    try:
        if len(output_paths) != len(document):
            raise ValueError("outputs must contain one path for every PDF page")
        return [
            _render_open_page(document, index, output, profile)
            for index, output in enumerate(output_paths)
        ]
    finally:
        document.close()
