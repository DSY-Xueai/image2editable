from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PageCalibration:
    noise_l1: float
    local_contrast: float
    edge_width_px: int
    text_halo_px: int
    min_component_pixels: int


@dataclass(frozen=True)
class _PageQualityContext:
    source_rgb: np.ndarray
    background_rgb: np.ndarray
    reconstructed_rgb: np.ndarray
    reconstruction_delta: np.ndarray
    background_delta: np.ndarray
    source_luma: np.ndarray
    text: np.ndarray
    exterior_owner_count: np.ndarray


_CHECK_STATES = frozenset({"pass", "fail", "unknown"})


def calibrate_page(source: np.ndarray, text_mask: np.ndarray) -> PageCalibration:
    image = np.asarray(source)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype.kind not in "buif":
        raise ValueError("source must be an RGB numeric image")
    text = _exact_mask(text_mask, image.shape[:2], "text mask")
    rgb = np.clip(image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    median = cv2.medianBlur(gray, 3)
    noise_l1 = float(np.median(np.abs(gray.astype(np.float32) - median.astype(np.float32))))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    local_mean = cv2.blur(lab, (9, 9))
    local_contrast = float(np.median(np.linalg.norm(lab - local_mean, axis=2)))
    if np.any(text):
        distance = cv2.distanceTransform(text.astype(np.uint8), cv2.DIST_L2, 5)
        text_halo_px = max(1, int(round(float(np.percentile(distance[text], 50)))))
    else:
        text_halo_px = 1
    edge_width_px = max(
        1,
        text_halo_px,
        int(round(max(noise_l1, 1.0) ** 0.5)),
    )
    min_component_pixels = max(20, int(round(image.shape[0] * image.shape[1] * 1e-5)))
    return PageCalibration(noise_l1, local_contrast, edge_width_px, text_halo_px, min_component_pixels)


def _check_state(checks: dict | None, name: str) -> str:
    if checks is None or name not in checks:
        return "unknown"
    state = checks[name]
    if state not in _CHECK_STATES:
        raise ValueError(f"{name} check state is invalid")
    return state


def _ratio(numerator: np.ndarray, denominator: int) -> float:
    return float(np.count_nonzero(numerator)) / max(denominator, 1)


def _largest_region(mask: np.ndarray) -> tuple[int, np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    if count <= 1:
        return 0, np.zeros(mask.shape, dtype=bool)
    label = max(
        range(1, count),
        key=lambda value: int(stats[value, cv2.CC_STAT_AREA]),
    )
    return int(stats[label, cv2.CC_STAT_AREA]), labels == label


def _rgb_image(value: object, shape: tuple[int, int] | None, label: str) -> np.ndarray:
    image = np.asarray(value)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype.kind not in "buif"
        or (image.dtype.kind == "f" and not np.all(np.isfinite(image)))
    ):
        raise ValueError(f"{label} must be a finite RGB numeric image")
    if shape is not None and image.shape[:2] != shape:
        raise ValueError(f"{label} shape must match source")
    return np.clip(image, 0, 255).astype(np.uint8)


def _prepare_page_quality_context(
    source: np.ndarray,
    background: np.ndarray,
    reconstructed: np.ndarray,
    text_mask: np.ndarray,
    *,
    component_masks: list[np.ndarray] | None = None,
) -> _PageQualityContext:
    source_rgb = _rgb_image(source, None, "source")
    shape = source_rgb.shape[:2]
    background_rgb = _rgb_image(background, shape, "background")
    reconstructed_rgb = _rgb_image(reconstructed, shape, "reconstructed")
    reconstruction_delta = np.max(
        np.abs(source_rgb.astype(np.int16) - reconstructed_rgb.astype(np.int16)), axis=2
    )
    background_delta = np.max(
        np.abs(source_rgb.astype(np.int16) - background_rgb.astype(np.int16)), axis=2
    )
    exterior_owner_count = np.zeros(shape, dtype=np.uint16)
    boundary_kernel = np.ones((3, 3), dtype=np.uint8)
    for mask in component_masks or []:
        support, _ = _project_component_mask(mask, shape)
        adjacent = cv2.dilate(support.astype(np.uint8), boundary_kernel) > 0
        adjacent &= ~support
        exterior_owner_count += adjacent.astype(np.uint16)
    return _PageQualityContext(
        source_rgb=source_rgb,
        background_rgb=background_rgb,
        reconstructed_rgb=reconstructed_rgb,
        reconstruction_delta=reconstruction_delta,
        background_delta=background_delta,
        source_luma=cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32),
        text=_exact_mask(text_mask, shape, "text mask"),
        exterior_owner_count=exterior_owner_count,
    )


def component_metrics(
    source: np.ndarray,
    background: np.ndarray,
    reconstructed: np.ndarray,
    node: dict,
    graph: dict,
    calibration: PageCalibration,
    *,
    component_mask: np.ndarray,
    text_mask: np.ndarray,
    _page_context: _PageQualityContext | None = None,
) -> dict:
    context = _page_context or _prepare_page_quality_context(
        source, background, reconstructed, text_mask
    )
    source_rgb = context.source_rgb
    shape = source_rgb.shape[:2]
    support, outside = _project_component_mask(component_mask, shape)
    support_pixels = int(np.count_nonzero(support))
    adaptive_tolerance = max(
        3.0,
        calibration.noise_l1 * 4.0 + calibration.local_contrast * 0.08,
    )
    hard_tolerance = 3.0
    reconstruction_delta = context.reconstruction_delta
    background_delta = context.background_delta
    missing = support & (reconstruction_delta > hard_tolerance)
    duplicate = support & (background_delta <= hard_tolerance)
    missing_pixels, missing_region = _largest_region(missing)
    duplicate_pixels, _ = _largest_region(duplicate)
    radius = calibration.edge_width_px
    edge_kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    edge = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_GRADIENT, edge_kernel) > 0
    far_radius = max(4, radius * 4)
    far_kernel = np.ones((2 * far_radius + 1, 2 * far_radius + 1), dtype=np.uint8)
    far = cv2.dilate(support.astype(np.uint8), far_kernel) > 0
    far &= ~support
    baseline_pixels = source_rgb[far]
    if baseline_pixels.size == 0:
        baseline_pixels = source_rgb.reshape(-1, 3)
    baseline = np.median(baseline_pixels.astype(np.float32), axis=0)
    source_luma = context.source_luma
    baseline_luma = float(cv2.cvtColor(
        np.asarray([[baseline]], dtype=np.uint8), cv2.COLOR_RGB2GRAY
    )[0, 0])
    largest_inner_shadow, _ = _largest_region(
        duplicate & (source_luma < baseline_luma - 6.0)
    )
    largest_inner_alpha, _ = _largest_region(
        duplicate & edge & (source_luma >= baseline_luma - 3.0)
    )
    adjacent = cv2.dilate(support.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    adjacent &= ~support
    exterior_duplicate = adjacent & (background_delta <= hard_tolerance)
    exterior_duplicate &= reconstruction_delta <= hard_tolerance
    exterior_changed = np.abs(source_luma - baseline_luma) > 6.0
    unique_exterior = exterior_duplicate & exterior_changed & (
        context.exterior_owner_count == 1
    )
    ambiguous_exterior = exterior_duplicate & exterior_changed & (
        context.exterior_owner_count > 1
    )
    exterior_shadow, _ = _largest_region(
        unique_exterior & (source_luma < baseline_luma - 6.0)
    )
    exterior_alpha, _ = _largest_region(
        unique_exterior & (source_luma >= baseline_luma - 3.0)
    )
    largest_shadow = max(largest_inner_shadow, exterior_shadow)
    largest_alpha = max(largest_inner_alpha, exterior_alpha)
    text_radius = max(calibration.text_halo_px, calibration.edge_width_px)
    text_kernel = np.ones((2 * text_radius + 1, 2 * text_radius + 1), dtype=np.uint8)
    text_support = context.text & (
        cv2.dilate(support.astype(np.uint8), text_kernel) > 0
    )
    text_ghost = text_support & (reconstruction_delta <= hard_tolerance)
    active_states = {"pending", "pending_gate", "frozen"}
    nodes = {value.get("id"): value for value in graph.get("nodes", []) if isinstance(value, dict)}
    parent_child_double = any(
        value.get("parent_id") == node.get("id")
        and value.get("state") in active_states
        and node.get("state") in active_states
        for value in nodes.values()
    ) or (
        node.get("parent_id") in nodes
        and node.get("state") in active_states
        and nodes[node["parent_id"]].get("state") in active_states
    )
    return {
        "component_pixels": support_pixels,
        "missing_pixels": missing_pixels,
        "missing_ratio": missing_pixels / max(support_pixels, 1),
        "duplicate_pixels": duplicate_pixels,
        "duplicate_ratio": duplicate_pixels / max(support_pixels, 1),
        "edge_missing_ratio": _ratio(missing_region & edge, int(np.count_nonzero(edge))),
        "shadow_duplicate_ratio": largest_shadow / max(support_pixels, 1),
        "alpha_duplicate_ratio": largest_alpha / max(int(np.count_nonzero(edge)), 1),
        "exterior_shadow_pixels": exterior_shadow,
        "exterior_alpha_pixels": exterior_alpha,
        "orphan_residual_pixels": int(np.count_nonzero(ambiguous_exterior)),
        "text_support_pixels": int(np.count_nonzero(text_support)),
        "text_duplicate_ratio": _ratio(text_ghost, int(np.count_nonzero(text_support))),
        "ownership_out_of_bounds_pixels": outside,
        "parent_child_double": parent_child_double,
        "noise_l1": calibration.noise_l1,
        "local_contrast": calibration.local_contrast,
        "edge_width_px": calibration.edge_width_px,
        "text_halo_px": calibration.text_halo_px,
        "adaptive_pixel_tolerance": adaptive_tolerance,
        "hard_pixel_tolerance": hard_tolerance,
    }


def evaluate_component(
    source: np.ndarray,
    background: np.ndarray,
    reconstructed: np.ndarray,
    node: dict,
    graph: dict,
    calibration: PageCalibration,
    *,
    component_mask: np.ndarray,
    text_mask: np.ndarray,
    page_checks: dict | None = None,
    agent_confidence: float | None = None,
    previous_metrics: dict | None = None,
    _page_context: _PageQualityContext | None = None,
) -> dict:
    metrics = component_metrics(
        source, background, reconstructed, node, graph, calibration,
        component_mask=component_mask, text_mask=text_mask, _page_context=_page_context,
    )
    violations = []
    hard_pixel_ratio = max(
        0.01,
        calibration.min_component_pixels / max(metrics["component_pixels"], 1),
    )
    hard_pixel_ratio = min(1.0, hard_pixel_ratio)
    if metrics["shadow_duplicate_ratio"] >= hard_pixel_ratio:
        violations.append("duplicate_shadow")
    if metrics["missing_ratio"] > 0.02 or metrics["edge_missing_ratio"] > 0.05:
        violations.append("missing_edge")
    text_pixel_floor = calibration.text_halo_px ** 2
    if (
        metrics["text_duplicate_ratio"] >= 0.02
        and metrics["text_duplicate_ratio"] * metrics["text_support_pixels"] >= text_pixel_floor
    ):
        violations.append("text_ghost")
    if metrics["alpha_duplicate_ratio"] >= 0.02:
        violations.append("alpha_halo")
    if metrics["parent_child_double"]:
        violations.append("parent_child_double")
    if metrics["duplicate_ratio"] >= hard_pixel_ratio:
        violations.append("duplicate_pixels")
    if metrics["ownership_out_of_bounds_pixels"]:
        violations.append("out_of_bounds")
    native_state = _check_state(page_checks, "protected_native_overlap")
    if native_state == "unknown":
        violations.append("protected_native_overlap_unknown")
    elif native_state == "fail":
        violations.append("protected_native_overlap")
    previous = previous_metrics or {}
    improvement = {}
    for key in ("missing_ratio", "duplicate_ratio", "edge_missing_ratio", "shadow_duplicate_ratio", "alpha_duplicate_ratio", "text_duplicate_ratio"):
        if key not in previous:
            continue
        prior = float(previous[key])
        if not np.isfinite(prior):
            raise ValueError("previous component metrics must be finite")
        improvement[key] = prior - float(metrics[key])
    return {
        "component_id": node["id"],
        "accepted": not violations,
        "metrics": metrics,
        "improvement": improvement,
        "violations": sorted(set(violations)),
        "checks": {"protected_native_overlap": native_state},
        "agent_confidence": agent_confidence,
    }


def evaluate_page_quality(
    component_reports: list[dict],
    *,
    visual_metrics: dict,
    page_checks: dict | None = None,
    expected_component_ids: list[str],
    initial_component_count: int,
) -> dict:
    required_visual = {"mae", "p95", "changed_ratio"}
    if not isinstance(visual_metrics, dict) or not required_visual <= set(visual_metrics):
        raise ValueError("visual_metrics fields are incomplete")
    for key in required_visual:
        value = visual_metrics[key]
        if type(value) not in {int, float} or not np.isfinite(value) or value < 0:
            raise ValueError("visual_metrics values must be finite and non-negative")
    if (
        type(expected_component_ids) is not list
        or any(type(value) is not str for value in expected_component_ids)
        or len(expected_component_ids) != len(set(expected_component_ids))
        or type(initial_component_count) is not int
        or initial_component_count < 0
    ):
        raise ValueError("component report expectations are invalid")
    if initial_component_count and not expected_component_ids:
        raise ValueError("component reports cannot be empty for a nonempty page")
    report_ids = []
    violations = []
    for report in component_reports:
        if (
            not isinstance(report, dict)
            or type(report.get("component_id")) is not str
            or type(report.get("accepted")) is not bool
            or type(report.get("violations")) is not list
            or (not report["accepted"] and not report["violations"])
        ):
            raise ValueError("component reports are invalid")
        report_ids.append(report["component_id"])
        violations.extend(report["violations"])
        metrics = report.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("component report metrics are invalid")
        if int(metrics.get("orphan_residual_pixels", 0)) > 0:
            violations.append("orphan_residual")
    if sorted(report_ids) != sorted(expected_component_ids):
        raise ValueError("component reports do not match expected IDs")
    reopen_state = _check_state(page_checks, "pptx_reopen")
    if reopen_state == "unknown":
        violations.append("pptx_reopen_unknown")
    elif reopen_state == "fail":
        violations.append("pptx_reopen")
    if (
        float(visual_metrics["mae"]) > 8.0
        or float(visual_metrics["p95"]) > 32.0
        or float(visual_metrics["changed_ratio"]) > 0.02
    ):
        violations.append("visual_difference")
    return {
        "accepted": not violations,
        "violations": sorted(set(violations)),
        "component_reports": component_reports,
        "visual_metrics": dict(visual_metrics),
        "checks": {"pptx_reopen": reopen_state},
    }


def _page_shape(shape: object) -> tuple[int, int]:
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ValueError("shape must contain positive integer height and width")
    return shape[0], shape[1]


def _numeric_mask(mask: object, label: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype.kind not in "biuf":
        raise ValueError(f"{label} must be a two-dimensional numeric mask")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    if array.dtype.kind in "if" and np.any(array < 0):
        raise ValueError(f"{label} contains negative values")
    return array


def _exact_mask(mask: object, shape: tuple[int, int], label: str) -> np.ndarray:
    array = _numeric_mask(mask, label)
    if array.shape != shape:
        raise ValueError(f"{label} shape must match page shape")
    return array if array.dtype == np.bool_ else array > 0


def _project_component_mask(
    mask: object,
    shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    array = _numeric_mask(mask, "component mask")
    active = array if array.dtype == np.bool_ else array > 0
    if active.shape == shape:
        return active, 0
    height = min(shape[0], active.shape[0])
    width = min(shape[1], active.shape[1])
    projected = np.zeros(shape, dtype=bool)
    projected[:height, :width] = active[:height, :width]
    out_of_bounds = int(np.count_nonzero(active)) - int(
        np.count_nonzero(active[:height, :width])
    )
    return projected, out_of_bounds


def validate_pixel_ownership(
    component_masks: list[np.ndarray],
    text_mask: np.ndarray,
    shape: tuple[int, int],
    *,
    foreground_mask: np.ndarray | None = None,
) -> dict:
    """Report ownership defects without modifying or repairing any mask.

    ``missing_pixels`` is meaningful only when ``foreground_mask`` is given.
    Every non-zero alpha value counts as source evidence owned by that component.
    """

    page_shape = _page_shape(shape)
    text = _exact_mask(text_mask, page_shape, "text mask")
    claimed = np.zeros(page_shape, dtype=bool)
    duplicate = np.zeros(page_shape, dtype=bool)
    out_of_bounds = 0
    for mask in component_masks:
        projected, outside = _project_component_mask(mask, page_shape)
        duplicate |= claimed & projected
        claimed |= projected
        out_of_bounds += outside

    if foreground_mask is None:
        missing_pixels = 0
    else:
        foreground = _exact_mask(
            foreground_mask,
            page_shape,
            "foreground mask",
        )
        missing_pixels = int(np.count_nonzero(foreground & ~claimed))

    report = {
        "duplicate_pixels": int(np.count_nonzero(duplicate)),
        "missing_pixels": missing_pixels,
        "text_duplicate_pixels": int(np.count_nonzero(text & claimed)),
        "out_of_bounds_pixels": out_of_bounds,
    }
    return {"valid": not any(report.values()), **report}
