from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from PIL import Image
import pypdfium2 as pdfium


STANDARD_DPI = 200.0
DETAIL_DPI = 300.0
SHORT_EDGE_FLOOR = 1200
LONG_EDGE_CEILING = 6000
PIXEL_COUNT_CEILING = 24_000_000

RenderProfile = Literal["standard", "detail"]


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

    if max(width_pt, height_pt) * scale > LONG_EDGE_CEILING:
        scale = LONG_EDGE_CEILING / max(width_pt, height_pt)
        reasons.append("long_edge_ceiling")
    if width_pt * height_pt * scale * scale > PIXEL_COUNT_CEILING:
        scale *= math.sqrt(PIXEL_COUNT_CEILING / (width_pt * height_pt * scale * scale))
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


def _box_values(box: Sequence[float] | None, fallback: list[float]) -> list[float]:
    return [float(value) for value in box] if box is not None else fallback


def _renderer_version() -> str:
    version = getattr(pdfium, "__version__", None)
    if version is not None:
        return str(version)
    return str(pdfium.version.PYPDFIUM_INFO)


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
        media_box = _box_values(page.get_mediabox(), [0.0, 0.0, width_pt, height_pt])
        crop_box = _box_values(page.get_cropbox(), media_box)
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
    document = pdfium.PdfDocument(str(source))
    try:
        return _render_open_page(document, index, output, profile)
    finally:
        document.close()


def render_pdf_document(
    source: str | Path,
    outputs: Sequence[str | Path],
    *,
    profile: RenderProfile,
) -> list[dict[str, object]]:
    document = pdfium.PdfDocument(str(source))
    try:
        if len(outputs) != len(document):
            raise ValueError("outputs must contain one path for every PDF page")
        return [
            _render_open_page(document, index, output, profile)
            for index, output in enumerate(outputs)
        ]
    finally:
        document.close()
