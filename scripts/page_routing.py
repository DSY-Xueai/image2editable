"""Deterministic, content-free page routing for fast reconstruction.

The router intentionally uses only bounded numeric signals.  It does not call
an LLM or a vision model, so deciding whether a page needs segmentation is
cheap and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


_SOURCE_KINDS = frozenset({"image", "pdf", "pdf_native", "pptx"})
_ROUTES = frozenset({"native", "direct", "local_refine", "strict"})


@dataclass(frozen=True)
class PageSignals:
    source_kind: str
    ocr_items: int = 0
    ocr_mean_confidence: float = 0.0
    text_coverage: float = 0.0
    regular_geometry_ratio: float = 0.0
    overlap_ratio: float = 0.0
    transparency_ratio: float = 0.0
    edge_density: float = 0.0
    scan_noise: float = 0.0
    visual_regions: int = 0


@dataclass(frozen=True)
class PagePolicy:
    route: str
    confidence: float
    reasons: tuple[str, ...]
    automatic_sam: bool
    max_residual_rounds: int
    hole_recheck: bool
    max_lama_calls: int
    host_agent_allowed: bool


def strict_page_policy() -> PagePolicy:
    """Return the legacy behavior contract for callers without routing."""
    return PagePolicy(
        route="strict",
        confidence=0.0,
        reasons=("strict_mode",),
        automatic_sam=True,
        max_residual_rounds=3,
        hole_recheck=True,
        max_lama_calls=2,
        host_agent_allowed=True,
    )


def _bounded_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return min(1.0, max(0.0, result))


def _validate_signals(signals: PageSignals) -> None:
    if signals.source_kind not in _SOURCE_KINDS:
        raise ValueError(f"Unsupported page source kind: {signals.source_kind}")
    if type(signals.ocr_items) is not int or signals.ocr_items < 0:
        raise ValueError("ocr_items must be a non-negative integer")
    if type(signals.visual_regions) is not int or signals.visual_regions < 0:
        raise ValueError("visual_regions must be a non-negative integer")
    for name in (
        "ocr_mean_confidence", "text_coverage", "regular_geometry_ratio",
        "overlap_ratio", "transparency_ratio", "edge_density", "scan_noise",
    ):
        _bounded_float(getattr(signals, name), name)


def classify_page(signals: PageSignals) -> PagePolicy:
    """Select the cheapest route that has enough evidence to preserve quality."""
    _validate_signals(signals)
    if signals.source_kind == "pdf_native":
        return PagePolicy(
            route="native", confidence=0.99, reasons=("native_pdf_objects",),
            automatic_sam=False, max_residual_rounds=0, hole_recheck=False,
            max_lama_calls=0, host_agent_allowed=False,
        )

    confidence = 0.0
    reasons: list[str] = []
    if signals.ocr_items and signals.ocr_mean_confidence >= 0.90:
        confidence += 0.25
        reasons.append("high_ocr_confidence")
    if signals.regular_geometry_ratio >= 0.70:
        confidence += 0.25
        reasons.append("regular_geometry")
    if signals.text_coverage >= 0.05:
        confidence += 0.10
        reasons.append("structured_text")
    if signals.overlap_ratio <= 0.08:
        confidence += 0.15
        reasons.append("low_overlap")
    if signals.transparency_ratio <= 0.12:
        confidence += 0.10
        reasons.append("low_transparency")
    if signals.edge_density <= 0.45:
        confidence += 0.05
    if signals.scan_noise <= 0.12:
        confidence += 0.10
    confidence = min(1.0, max(0.0, confidence))

    difficult = (
        signals.ocr_mean_confidence < 0.55
        or signals.overlap_ratio > 0.30
        or signals.transparency_ratio > 0.35
        or signals.scan_noise > 0.28
        or signals.edge_density > 0.75
        or signals.visual_regions >= 128
    )
    if difficult or confidence < 0.35:
        return PagePolicy(
            route="strict", confidence=confidence,
            reasons=tuple(reasons or ("low_confidence",)),
            automatic_sam=True, max_residual_rounds=3, hole_recheck=True,
            max_lama_calls=2, host_agent_allowed=False,
        )
    if confidence >= 0.75 and signals.visual_regions < 24:
        return PagePolicy(
            route="direct", confidence=confidence, reasons=tuple(reasons),
            automatic_sam=False, max_residual_rounds=0, hole_recheck=False,
            max_lama_calls=1, host_agent_allowed=False,
        )
    return PagePolicy(
        route="local_refine", confidence=confidence,
        reasons=tuple(reasons or ("partial_confidence",)),
        automatic_sam=False, max_residual_rounds=1, hole_recheck=False,
        max_lama_calls=1, host_agent_allowed=False,
    )


__all__ = ["PageSignals", "PagePolicy", "classify_page", "strict_page_policy"]
