from __future__ import annotations

import importlib
import importlib.util
import hashlib
import weakref

import numpy as np
import pytest
from PIL import Image

import image2editable.component_quality as component_quality

from image2editable.component_quality import (
    calibrate_page,
    evaluate_component,
    evaluate_page_quality,
    validate_component_quality_report,
)
from image2editable.component_repair import evaluate_component_quality_round


def _validate_pixel_ownership(*args, **kwargs):
    assert importlib.util.find_spec("image2editable.component_quality") is not None
    module = importlib.import_module("image2editable.component_quality")
    return module.validate_pixel_ownership(*args, **kwargs)


def _mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    y1, x1, y2, x2 = box
    mask[y1:y2, x1:x2] = 255
    return mask


def _leaf_calibration(min_component_pixels: int = 20) -> component_quality.PageCalibration:
    return component_quality.PageCalibration(0.0, 1.0, 1, 1, min_component_pixels)


def test_spatially_independent_absorbed_candidates_form_multiple_leaf_clusters() -> None:
    masks = [
        _mask((30, 40), (2, 2, 8, 8)),
        _mask((30, 40), (18, 28, 24, 34)),
    ]

    assert component_quality.absorbed_leaf_cluster_count(
        masks, _leaf_calibration()
    ) == 2


def test_overlapping_or_contained_duplicate_masks_form_one_leaf_cluster() -> None:
    masks = [
        _mask((30, 40), (3, 3, 8, 8)),
        _mask((30, 40), (4, 4, 8, 9)),
        _mask((30, 40), (3, 3, 7, 8)),
    ]

    assert component_quality.absorbed_leaf_cluster_count(
        masks, _leaf_calibration()
    ) == 1


def test_fragment_below_page_noise_floor_does_not_add_leaf_cluster() -> None:
    masks = [
        _mask((30, 40), (3, 3, 11, 11)),
        _mask((30, 40), (20, 30, 22, 32)),
    ]

    assert component_quality.absorbed_leaf_cluster_count(
        masks, _leaf_calibration(min_component_pixels=20)
    ) == 1


def test_broad_container_mask_does_not_bridge_independent_leaf_clusters() -> None:
    masks = [
        _mask((30, 40), (3, 3, 9, 9)),
        _mask((30, 40), (18, 28, 24, 34)),
        _mask((30, 40), (1, 1, 27, 37)),
    ]

    assert component_quality.absorbed_leaf_cluster_count(
        masks, _leaf_calibration()
    ) == 3


def test_nearby_nonoverlapping_edge_fragments_form_one_leaf_cluster() -> None:
    masks = [
        _mask((30, 40), (5, 3, 20, 18)),
        _mask((30, 40), (10, 19, 13, 27)),
    ]
    calibration = component_quality.PageCalibration(0.0, 1.0, 2, 1, 20)

    assert component_quality.absorbed_leaf_cluster_count(masks, calibration) == 1


def test_gap_fragment_does_not_bridge_two_complete_leaf_clusters() -> None:
    masks = [
        _mask((30, 40), (5, 2, 20, 14)),
        _mask((30, 40), (7, 15, 17, 17)),
        _mask((30, 40), (5, 18, 20, 30)),
    ]
    calibration = component_quality.PageCalibration(0.0, 1.0, 2, 1, 20)

    assert component_quality.absorbed_leaf_cluster_count(masks, calibration) == 2


def test_adjacent_complete_rectangles_form_independent_leaf_clusters() -> None:
    masks = [
        _mask((30, 40), (5, 3, 15, 13)),
        _mask((30, 40), (5, 14, 15, 24)),
    ]
    calibration = component_quality.PageCalibration(0.0, 1.0, 2, 1, 20)

    assert component_quality.absorbed_leaf_cluster_count(masks, calibration) == 2


def test_offset_shadow_forms_one_leaf_cluster() -> None:
    masks = [
        _mask((30, 40), (5, 5, 15, 15)),
        _mask((30, 40), (5, 10, 15, 20)),
    ]
    calibration = component_quality.PageCalibration(0.0, 1.0, 2, 1, 20)

    assert component_quality.absorbed_leaf_cluster_count(masks, calibration) == 1


def test_small_page_calibration_keeps_far_valid_small_leaf_masks() -> None:
    source = np.zeros((30, 40, 3), dtype=np.uint8)
    calibration = calibrate_page(source, np.zeros(source.shape[:2], dtype=np.uint8))
    masks = [
        _mask(source.shape[:2], (2, 2, 4, 4)),
        _mask(source.shape[:2], (24, 34, 26, 36)),
    ]

    assert calibration.min_component_pixels == 1
    assert component_quality.absorbed_leaf_cluster_count(masks, calibration) == 2


def test_leaf_clustering_releases_full_page_masks_while_consuming_iterator() -> None:
    references: list[weakref.ReferenceType[np.ndarray]] = []
    live_counts = []

    def masks():
        for index in range(12):
            live_counts.append(sum(reference() is not None for reference in references))
            mask = np.zeros((512, 512), dtype=bool)
            start = index * 16
            mask[4:12, start:start + 8] = True
            references.append(weakref.ref(mask))
            yield mask

    assert component_quality.absorbed_leaf_cluster_count(
        masks(), _leaf_calibration(min_component_pixels=1)
    ) == 12
    assert max(live_counts) <= 2


def test_strict_quality_report_rejects_empty_metrics() -> None:
    report = {
        "accepted": False,
        "violations": [],
        "component_reports": [{
            "component_id": "component_0001",
            "accepted": False,
            "metrics": {},
            "improvement": {},
            "violations": [],
            "checks": {"protected_native_overlap": "pass"},
            "agent_confidence": None,
        }],
        "visual_metrics": {},
        "checks": {"protected_native_overlap": "pass", "pptx_reopen": "unknown"},
    }

    with pytest.raises(ValueError, match="metrics fields"):
        validate_component_quality_report(
            report,
            expected_component_ids=["component_0001"],
            initial_component_count=1,
        )


def test_each_foreground_pixel_has_one_active_owner() -> None:
    first = _mask((14, 14), (2, 2, 8, 8))
    second = _mask((14, 14), (6, 6, 12, 12))

    report = _validate_pixel_ownership(
        [first, second],
        text_mask=np.zeros(first.shape, dtype=np.uint8),
        shape=first.shape,
    )

    assert report == {
        "valid": False,
        "duplicate_pixels": 4,
        "missing_pixels": 0,
        "text_duplicate_pixels": 0,
        "out_of_bounds_pixels": 0,
    }


def test_missing_pixels_require_an_explicit_foreground_mask() -> None:
    owner = _mask((8, 8), (1, 1, 4, 4))
    expected = owner.copy()
    expected[5:7, 5:7] = 255

    without_expected = _validate_pixel_ownership(
        [owner],
        text_mask=np.zeros(owner.shape, dtype=np.uint8),
        shape=owner.shape,
    )
    with_expected = _validate_pixel_ownership(
        [owner],
        text_mask=np.zeros(owner.shape, dtype=np.uint8),
        shape=owner.shape,
        foreground_mask=expected,
    )

    assert without_expected["missing_pixels"] == 0
    assert without_expected["valid"] is True
    assert with_expected["missing_pixels"] == 4
    assert with_expected["valid"] is False


def test_text_duplicate_and_out_of_bounds_pixels_are_reported_separately() -> None:
    oversized = np.zeros((7, 7), dtype=np.uint8)
    oversized[1:4, 1:4] = 255
    oversized[6, 2:5] = 255
    text = _mask((6, 6), (2, 2, 4, 4))

    report = _validate_pixel_ownership(
        [oversized],
        text_mask=text,
        shape=(6, 6),
    )

    assert report["duplicate_pixels"] == 0
    assert report["missing_pixels"] == 0
    assert report["text_duplicate_pixels"] == 4
    assert report["out_of_bounds_pixels"] == 3
    assert report["valid"] is False


def test_alpha_shadow_and_antialias_evidence_still_has_one_owner() -> None:
    gradient = np.zeros((8, 8), dtype=np.uint8)
    gradient[2:6, 2:6] = np.array(
        [
            [32, 64, 64, 32],
            [64, 255, 255, 64],
            [64, 255, 255, 64],
            [32, 64, 64, 32],
        ],
        dtype=np.uint8,
    )
    shadow = np.zeros_like(gradient)
    shadow[6, 3] = 1

    valid = _validate_pixel_ownership(
        [gradient, shadow],
        text_mask=np.zeros(gradient.shape, dtype=np.uint8),
        shape=gradient.shape,
    )
    shadow[5, 3] = 1
    duplicate = _validate_pixel_ownership(
        [gradient, shadow],
        text_mask=np.zeros(gradient.shape, dtype=np.uint8),
        shape=gradient.shape,
    )

    assert valid["valid"] is True
    assert duplicate["duplicate_pixels"] == 1
    assert duplicate["valid"] is False


def test_ownership_validation_does_not_modify_masks() -> None:
    component = _mask((6, 6), (1, 1, 5, 5))
    text = _mask((6, 6), (2, 2, 3, 3))
    foreground = component.copy()
    originals = (component.copy(), text.copy(), foreground.copy())

    _validate_pixel_ownership(
        [component],
        text_mask=text,
        shape=component.shape,
        foreground_mask=foreground,
    )

    assert np.array_equal(component, originals[0])
    assert np.array_equal(text, originals[1])
    assert np.array_equal(foreground, originals[2])


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[object()]], dtype=object),
        np.array([["mask"]]),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[-1]], dtype=np.int16),
    ],
)
def test_ownership_rejects_invalid_mask_values(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="mask"):
        _validate_pixel_ownership(
            [invalid],
            text_mask=np.zeros((1, 1), dtype=np.uint8),
            shape=(1, 1),
        )


def test_exact_bool_mask_projection_reuses_input_memory() -> None:
    assert importlib.util.find_spec("image2editable.component_quality") is not None
    module = importlib.import_module("image2editable.component_quality")
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    projected, outside = module._project_component_mask(mask, mask.shape)

    assert np.shares_memory(projected, mask)
    assert outside == 0


def test_ownership_accumulator_uses_only_boolean_page_buffers(monkeypatch) -> None:
    assert importlib.util.find_spec("image2editable.component_quality") is not None
    module = importlib.import_module("image2editable.component_quality")
    real_zeros = module.np.zeros
    allocated_dtypes = []

    def recording_zeros(*args, **kwargs):
        allocated_dtypes.append(kwargs.get("dtype"))
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(module.np, "zeros", recording_zeros)
    first = np.zeros((8, 8), dtype=bool)
    first[1:4, 1:4] = True
    second = np.zeros((8, 8), dtype=bool)
    second[5:7, 5:7] = True

    report = module.validate_pixel_ownership(
        [first, second],
        text_mask=np.zeros((8, 8), dtype=bool),
        shape=(8, 8),
    )

    assert report["valid"] is True
    assert allocated_dtypes
    assert set(allocated_dtypes) <= {bool, np.bool_}


def _synthetic_quality_case(
    *,
    scale: int = 1,
    noise: int = 0,
    contrast: int = 90,
    defect: str | None = None,
) -> dict:
    shape = (48 * scale, 64 * scale)
    source = np.full((*shape, 3), 96, dtype=np.uint8)
    component_mask = np.zeros(shape, dtype=bool)
    component_mask[12 * scale:36 * scale, 16 * scale:48 * scale] = True
    source[component_mask] = 96 + contrast
    if noise:
        rng = np.random.default_rng(7)
        source = np.clip(
            source.astype(np.int16) + rng.integers(-noise, noise + 1, source.shape),
            0,
            255,
        ).astype(np.uint8)
    text = np.zeros(shape, dtype=bool)
    text[20 * scale:24 * scale, 24 * scale:40 * scale] = True
    component_mask &= ~text
    source[text] = 20
    background = np.full_like(source, 96)
    reconstructed = background.copy()
    reconstructed[component_mask] = source[component_mask]
    graph = {"nodes": [{"id": "component_0001", "kind": "parent", "parent_id": None,
                        "state": "pending_gate", "mask": "masks/component_0001.png",
                        "mask_sha256": "a" * 64, "bbox": [16 * scale, 12 * scale, 48 * scale, 36 * scale],
                        "z_index": 0, "text_ids": []}]}
    if defect == "missing_edge":
        reconstructed[12 * scale:14 * scale, 16 * scale:48 * scale] = 0
    elif defect == "duplicate_shadow":
        component_mask[36 * scale:40 * scale, 18 * scale:46 * scale] = True
        source[36 * scale:40 * scale, 18 * scale:46 * scale] = 55
        background[36 * scale:40 * scale, 18 * scale:46 * scale] = 55
        reconstructed[36 * scale:40 * scale, 18 * scale:46 * scale] = 55
    elif defect == "text_ghost":
        reconstructed[text] = source[text]
    elif defect == "alpha_halo":
        component_mask[11 * scale:12 * scale, 16 * scale:48 * scale] = True
        component_mask[36 * scale:37 * scale, 16 * scale:48 * scale] = True
        source[11 * scale:12 * scale, 16 * scale:48 * scale] = 138
        source[36 * scale:37 * scale, 16 * scale:48 * scale] = 138
        background[11 * scale:12 * scale, 16 * scale:48 * scale] = 138
        background[36 * scale:37 * scale, 16 * scale:48 * scale] = 138
        reconstructed[11 * scale:12 * scale, 16 * scale:48 * scale] = 138
        reconstructed[36 * scale:37 * scale, 16 * scale:48 * scale] = 138
    elif defect == "parent_child_double":
        graph["nodes"].append({
            "id": "component_0002", "kind": "child", "parent_id": "component_0001",
            "state": "pending_gate", "mask": "masks/component_0002.png",
            "mask_sha256": "b" * 64, "bbox": [20 * scale, 16 * scale, 44 * scale, 32 * scale],
            "z_index": 1, "text_ids": [],
        })
    ys, xs = np.where(component_mask)
    graph["nodes"][0]["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return {
        "source": source,
        "background": background,
        "reconstructed": reconstructed,
        "node": graph["nodes"][0],
        "graph": graph,
        "component_mask": component_mask,
        "text_mask": text,
    }


def _evaluate_synthetic(case: dict, *, confidence: float = 0.95, previous_metrics: dict | None = None) -> dict:
    calibration = calibrate_page(case["source"], case["text_mask"])
    return evaluate_component(
        case["source"], case["background"], case["reconstructed"],
        case["node"], case["graph"], calibration,
        component_mask=case["component_mask"], text_mask=case["text_mask"],
        page_checks={"protected_native_overlap": "pass"},
        agent_confidence=confidence, previous_metrics=previous_metrics,
    )


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_gate_decision_is_stable_across_resolution_and_ocr_scale(scale: int) -> None:
    report = _evaluate_synthetic(_synthetic_quality_case(scale=scale, contrast=80))
    assert report["accepted"] is True


def test_calibration_adapts_to_noise_and_contrast_without_relaxing_clean_decision() -> None:
    flat = _synthetic_quality_case(noise=0, contrast=30)
    photo = _synthetic_quality_case(noise=18, contrast=120)
    flat_calibration = calibrate_page(flat["source"], flat["text_mask"])
    photo_calibration = calibrate_page(photo["source"], photo["text_mask"])
    assert photo_calibration.noise_l1 > flat_calibration.noise_l1
    assert photo_calibration.local_contrast > flat_calibration.local_contrast
    assert photo_calibration.edge_width_px >= flat_calibration.edge_width_px
    assert _evaluate_synthetic(flat)["accepted"] is True
    assert _evaluate_synthetic(photo)["accepted"] is True


@pytest.mark.parametrize(
    "defect",
    ["duplicate_shadow", "missing_edge", "text_ghost", "alpha_halo", "parent_child_double"],
)
@pytest.mark.parametrize("scale", [1, 2, 4])
@pytest.mark.parametrize("noise", [0, 18])
@pytest.mark.parametrize("contrast", [12, 90])
def test_hard_safety_defects_never_pass(defect: str, scale: int, noise: int, contrast: int) -> None:
    report = _evaluate_synthetic(
        _synthetic_quality_case(defect=defect, scale=scale, noise=noise, contrast=contrast),
        confidence=1.0,
    )
    assert report["accepted"] is False
    assert defect in report["violations"]


def test_low_contrast_noisy_clean_component_does_not_false_fail() -> None:
    report = _evaluate_synthetic(_synthetic_quality_case(contrast=12, noise=18))
    assert report["accepted"] is True


def test_clean_adjacent_gradient_and_line_are_not_component_residuals() -> None:
    case = _synthetic_quality_case()
    height, width = case["source"].shape[:2]
    gradient = np.linspace(20, 220, width, dtype=np.uint8)
    case["source"][:10, :, :] = gradient[None, :, None]
    case["background"][:10, :, :] = case["source"][:10, :, :]
    case["reconstructed"][:10, :, :] = case["source"][:10, :, :]
    case["source"][:, 8:10] = 12
    case["background"][:, 8:10] = 12
    case["reconstructed"][:, 8:10] = 12
    assert _evaluate_synthetic(case)["accepted"] is True


def test_small_component_full_duplicate_still_fails_hard_gate() -> None:
    case = _synthetic_quality_case()
    small = np.zeros(case["component_mask"].shape, dtype=bool)
    small[14:16, 18:20] = True
    case["component_mask"] = small
    case["background"][small] = case["source"][small]
    report = _evaluate_synthetic(case)
    assert report["accepted"] is False
    assert "duplicate_pixels" in report["violations"]


def test_sparse_child_fails_against_its_text_excluded_parent() -> None:
    case = _synthetic_quality_case()
    parent_mask = case["component_mask"].copy()
    child_mask = np.zeros_like(parent_mask)
    child_mask[20:24, 24:28] = True
    case["component_mask"] = child_mask
    case["node"]["kind"] = "child"
    case["node"]["parent_id"] = "parent_0001"
    case["graph"]["nodes"].append({
        "id": "parent_0001", "kind": "parent", "parent_id": None,
        "state": "inactive", "mask": "masks/parent_0001.png",
        "mask_sha256": "b" * 64, "bbox": [16, 12, 48, 36],
        "z_index": 1, "text_ids": [],
    })
    calibration = calibrate_page(case["source"], case["text_mask"])

    report = evaluate_component(
        case["source"], case["background"], case["reconstructed"],
        case["node"], case["graph"], calibration,
        component_mask=child_mask, parent_mask=parent_mask,
        text_mask=case["text_mask"],
        page_checks={"protected_native_overlap": "pass"},
    )

    assert report["metrics"]["parent_coverage_ratio"] < 0.25
    assert "incomplete_child" in report["violations"]


def test_generic_internal_duplicate_is_not_misclassified_as_shadow_or_alpha() -> None:
    case = _synthetic_quality_case()
    duplicate = np.zeros(case["component_mask"].shape, dtype=bool)
    duplicate[26:34, 24:32] = True
    case["background"][duplicate] = case["source"][duplicate]
    violations = _evaluate_synthetic(case)["violations"]
    assert "duplicate_pixels" in violations
    assert "duplicate_shadow" not in violations
    assert "alpha_halo" not in violations


def test_nonhierarchical_component_overlap_fails_quality_gate() -> None:
    case = _synthetic_quality_case()
    other = np.zeros_like(case["component_mask"])
    other[26:34, 24:32] = True
    case["graph"]["nodes"].append({
        "id": "component_0002", "kind": "parent", "parent_id": None,
        "state": "frozen", "mask": "masks/component_0002.png",
        "mask_sha256": "b" * 64, "bbox": [24, 26, 32, 34],
        "z_index": 1, "text_ids": [],
    })
    module = importlib.import_module("image2editable.component_quality")
    calibration = calibrate_page(case["source"], case["text_mask"])
    context = module._prepare_page_quality_context(
        case["source"], case["background"], case["reconstructed"],
        case["text_mask"], calibration=calibration,
        component_masks=[case["component_mask"], other],
    )

    report = evaluate_component(
        case["source"], case["background"], case["reconstructed"],
        case["node"], case["graph"], calibration,
        component_mask=case["component_mask"], text_mask=case["text_mask"],
        page_checks={"protected_native_overlap": "pass"},
        _page_context=context,
    )

    assert "component_overlap" in report["violations"]


def test_canvas_colored_component_fill_is_not_a_visible_duplicate() -> None:
    case = _synthetic_quality_case()
    fill = np.zeros(case["component_mask"].shape, dtype=bool)
    fill[26:34, 24:32] = True
    case["source"][fill] = 97
    case["background"][fill] = 97
    case["reconstructed"][fill] = 97

    assert "duplicate_pixels" not in _evaluate_synthetic(case)["violations"]


def test_agent_confidence_cannot_relax_hard_gate() -> None:
    case = _synthetic_quality_case(defect="text_ghost")
    assert _evaluate_synthetic(case, confidence=0.01)["violations"] == _evaluate_synthetic(case, confidence=1.0)["violations"]


def _text_isolation_case() -> dict:
    import cv2

    shape = (96, 180)
    source = np.full((*shape, 3), 228, dtype=np.uint8)
    component_mask = np.zeros(shape, dtype=bool)
    component_mask[18:78, 20:160] = True
    source[component_mask] = (70, 125, 190)
    text_mask = np.zeros(shape, dtype=bool)
    text_mask[34:67, 42:140] = True
    cv2.putText(
        source, "EDIT", (45, 62), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (238, 238, 238), 2, cv2.LINE_AA,
    )
    background = np.full_like(source, 228)
    reconstructed = background.copy()
    reconstructed[component_mask] = (70, 125, 190)
    node = {
        "id": "component_0001", "kind": "parent", "parent_id": None,
        "state": "pending_gate", "mask": "masks/component_0001.png",
        "mask_sha256": "a" * 64, "bbox": [20, 18, 160, 78],
        "z_index": 0, "text_ids": [],
    }
    return {
        "source": source, "background": background,
        "reconstructed": reconstructed, "component_mask": component_mask,
        "text_mask": text_mask, "node": node, "graph": {"nodes": [node]},
    }


def test_component_full_alpha_with_source_glyph_pixels_fails_isolation_gate() -> None:
    case = _text_isolation_case()
    glyph = case["text_mask"] & np.any(case["source"] != (70, 125, 190), axis=2)
    case["reconstructed"][glyph] = case["source"][glyph]

    report = _evaluate_synthetic(case)

    assert report["metrics"]["component_text_residual_ratio"] > 0
    assert "component_text_residual" in report["violations"]


def test_background_with_source_glyph_pixels_fails_text_isolation_gate() -> None:
    case = _text_isolation_case()
    glyph = case["text_mask"] & np.any(case["source"] != (70, 125, 190), axis=2)
    case["background"][glyph] = case["source"][glyph]

    report = _evaluate_synthetic(case)

    assert report["metrics"]["background_text_residual_ratio"] > 0
    assert "background_text_residual" in report["violations"]


def test_reliable_ocr_requires_exactly_one_editable_text_contribution() -> None:
    report = evaluate_page_quality(
        [],
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={"pptx_reopen": "pass", "editable_text_once": "fail"},
        expected_component_ids=[], initial_component_count=0,
    )

    assert report["accepted"] is False
    assert "editable_text_once" in report["violations"]


def test_clean_component_and_background_pass_text_isolation_gate() -> None:
    report = _evaluate_synthetic(_text_isolation_case())

    assert "component_text_residual" not in report["violations"]
    assert "background_text_residual" not in report["violations"]


def test_text_ghost_is_only_attributed_to_the_adjacent_component() -> None:
    case = _synthetic_quality_case(defect="text_ghost")
    remote_mask = np.zeros(case["component_mask"].shape, dtype=bool)
    remote_mask[2:8, 52:60] = True
    remote_node = dict(case["node"], id="component_0002", parent_id=None)
    case["graph"]["nodes"].append(remote_node)
    calibration = calibrate_page(case["source"], case["text_mask"])
    adjacent = evaluate_component(
        case["source"], case["background"], case["reconstructed"],
        case["node"], case["graph"], calibration,
        component_mask=case["component_mask"], text_mask=case["text_mask"],
        page_checks={"protected_native_overlap": "pass"},
    )
    remote = evaluate_component(
        case["source"], case["background"], case["reconstructed"],
        remote_node, case["graph"], calibration,
        component_mask=remote_mask, text_mask=case["text_mask"],
        page_checks={"protected_native_overlap": "pass"},
    )
    assert "text_ghost" in adjacent["violations"]
    assert "text_ghost" not in remote["violations"]


def test_text_box_background_is_not_counted_as_a_text_ghost() -> None:
    case = _synthetic_quality_case()
    text = case["text_mask"]
    case["source"][text] = 96
    glyph = np.zeros_like(text)
    glyph[21:23, 27:37] = True
    case["source"][glyph] = 20

    report = _evaluate_synthetic(case)

    assert "text_ghost" not in report["violations"]


def test_clean_component_fill_inside_text_box_is_not_a_text_ghost() -> None:
    case = _synthetic_quality_case()
    text = case["text_mask"]
    case["source"][text] = 186
    case["reconstructed"][text] = 186

    assert "text_ghost" not in _evaluate_synthetic(case)["violations"]


def test_structural_divider_crossing_text_box_is_not_a_text_ghost() -> None:
    case = _synthetic_quality_case()
    text = case["text_mask"]
    case["source"][text] = 186
    divider = np.zeros_like(text)
    divider[20:24, 38:40] = True
    case["source"][divider] = 20
    case["reconstructed"][divider] = 20

    assert "text_ghost" not in _evaluate_synthetic(case)["violations"]


def test_solid_colored_header_fill_is_not_a_text_ghost() -> None:
    import cv2
    from image2editable.component_quality import PageCalibration

    shape = (120, 320)
    source = np.full((*shape, 3), 245, dtype=np.uint8)
    component_mask = np.zeros(shape, dtype=bool)
    component_mask[50:110, 170:310] = True
    fill = np.array([29, 140, 57], dtype=np.uint8)
    source[component_mask] = fill
    cv2.putText(
        source, "GOOD", (190, 94), cv2.FONT_HERSHEY_SIMPLEX,
        1.1, (255, 255, 255), 2, cv2.LINE_AA,
    )
    text_mask = np.zeros(shape, dtype=bool)
    text_mask[54:107, 187:276] = True
    text_mask[5:54, 20:230] = True
    background = np.full_like(source, 245)
    reconstructed = background.copy()
    reconstructed[component_mask] = fill
    node = {
        "id": "component_0001", "kind": "parent", "parent_id": None,
        "state": "pending_gate", "mask": "masks/component_0001.png",
        "mask_sha256": "a" * 64, "bbox": [170, 50, 310, 110],
        "z_index": 0, "text_ids": [],
    }

    report = evaluate_component(
        source, background, reconstructed, node, {"nodes": [node]},
        PageCalibration(0.0, 1.0, 16, 16, 20),
        component_mask=component_mask, text_mask=text_mask,
        page_checks={"protected_native_overlap": "pass"},
    )

    assert "text_ghost" not in report["violations"]


def test_internal_page_context_reuses_full_page_conversions(monkeypatch) -> None:
    case = _synthetic_quality_case()
    module = importlib.import_module("image2editable.component_quality")
    original = module._rgb_image
    original_median = module.cv2.medianBlur
    original_distance = module.cv2.distanceTransform
    calls = []
    median_calls = 0
    distance_calls = 0
    calibration = calibrate_page(case["source"], case["text_mask"])

    def counted(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    def counted_median(*args, **kwargs):
        nonlocal median_calls
        median_calls += 1
        return original_median(*args, **kwargs)

    def counted_distance(*args, **kwargs):
        nonlocal distance_calls
        distance_calls += 1
        return original_distance(*args, **kwargs)

    monkeypatch.setattr(module, "_rgb_image", counted)
    monkeypatch.setattr(module.cv2, "medianBlur", counted_median)
    monkeypatch.setattr(module.cv2, "distanceTransform", counted_distance)
    context = module._prepare_page_quality_context(
        case["source"], case["background"], case["reconstructed"],
        case["text_mask"], calibration=calibration,
    )
    for component_id in ("component_0001", "component_0002"):
        node = dict(case["node"], id=component_id)
        evaluate_component(
            case["source"], case["background"], case["reconstructed"],
            node, {"nodes": [node]}, calibration,
            component_mask=case["component_mask"], text_mask=case["text_mask"],
            page_checks={"protected_native_overlap": "pass"}, _page_context=context,
        )
    assert calls == ["source", "background", "reconstructed"]
    assert median_calls == 3
    assert distance_calls == 1


def test_exterior_shadow_requires_unique_boundary_attribution() -> None:
    case = _synthetic_quality_case()
    exterior = np.zeros(case["component_mask"].shape, dtype=bool)
    exterior[36:38, 20:44] = True
    case["source"][exterior] = 50
    case["background"][exterior] = 50
    case["reconstructed"][exterior] = 50
    calibration = calibrate_page(case["source"], case["text_mask"])
    module = importlib.import_module("image2editable.component_quality")
    context = module._prepare_page_quality_context(
        case["source"], case["background"], case["reconstructed"],
        case["text_mask"], calibration=calibration,
        component_masks=[case["component_mask"]],
    )
    report = evaluate_component(
        case["source"], case["background"], case["reconstructed"],
        case["node"], case["graph"], calibration,
        component_mask=case["component_mask"], text_mask=case["text_mask"],
        page_checks={"protected_native_overlap": "pass"}, _page_context=context,
    )
    assert "duplicate_shadow" in report["violations"]
    assert report["metrics"]["exterior_shadow_pixels"] > 0


def test_ambiguous_exterior_residual_is_page_level_orphan() -> None:
    shape = (40, 48)
    source = np.full((*shape, 3), 96, dtype=np.uint8)
    background = source.copy()
    reconstructed = source.copy()
    left = np.zeros(shape, dtype=bool)
    right = np.zeros(shape, dtype=bool)
    left[10:30, 8:20] = True
    right[10:30, 21:33] = True
    source[left | right] = 180
    reconstructed[left | right] = 180
    shared = np.zeros(shape, dtype=bool)
    shared[12:28, 20] = True
    source[shared] = 50
    background[shared] = 50
    reconstructed[shared] = 50
    text = np.zeros(shape, dtype=bool)
    calibration = calibrate_page(source, text)
    module = importlib.import_module("image2editable.component_quality")
    context = module._prepare_page_quality_context(
        source, background, reconstructed, text, calibration=calibration,
        component_masks=[left, right],
    )
    graph = {"nodes": []}
    reports = []
    for component_id, mask in (("left", left), ("right", right)):
        node = {"id": component_id, "kind": "parent", "parent_id": None,
                "state": "pending_gate"}
        graph["nodes"].append(node)
        reports.append(evaluate_component(
            source, background, reconstructed, node, graph, calibration,
            component_mask=mask, text_mask=text,
            page_checks={"protected_native_overlap": "pass"}, _page_context=context,
        ))
    page = evaluate_page_quality(
        reports, visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={"pptx_reopen": "pass"},
        expected_component_ids=["left", "right"], initial_component_count=2,
    )
    assert all("duplicate_shadow" not in report["violations"] for report in reports)
    assert "orphan_residual" in page["violations"]


def test_report_tracks_relative_improvement_from_previous_round() -> None:
    case = _synthetic_quality_case(defect="missing_edge")
    previous = {"missing_ratio": 0.50, "duplicate_ratio": 0.10}
    report = _evaluate_synthetic(case, previous_metrics=previous)
    assert report["improvement"]["missing_ratio"] > 0


@pytest.mark.parametrize("status", [None, "unknown", "fail"])
def test_protected_native_overlap_is_fail_closed(status: str | None) -> None:
    case = _synthetic_quality_case()
    checks = {} if status is None else {"protected_native_overlap": status}
    calibration = calibrate_page(case["source"], case["text_mask"])
    report = evaluate_component(
        case["source"], case["background"], case["reconstructed"],
        case["node"], case["graph"], calibration,
        component_mask=case["component_mask"], text_mask=case["text_mask"], page_checks=checks,
    )
    assert report["accepted"] is False
    assert "protected_native_overlap_unknown" in report["violations"] or "protected_native_overlap" in report["violations"]


@pytest.mark.parametrize("status", [None, "unknown", "fail"])
def test_pptx_reopen_is_page_hard_gate(status: str | None) -> None:
    checks = {"protected_native_overlap": "pass"}
    if status is not None:
        checks["pptx_reopen"] = status
    report = evaluate_page_quality(
        [],
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks=checks,
        expected_component_ids=[],
        initial_component_count=0,
    )
    assert report["accepted"] is False
    assert "pptx_reopen_unknown" in report["violations"] or "pptx_reopen" in report["violations"]


def test_repair_quality_round_requires_authenticated_masks_and_external_pass_checks(tmp_path) -> None:
    case = _synthetic_quality_case()
    graph_dir = tmp_path / "round"
    mask_path = graph_dir / "masks/component_0001.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(case["component_mask"].astype(np.uint8) * 255).save(mask_path)
    case["graph"]["nodes"][0]["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    accepted = evaluate_component_quality_round(
        case["source"], case["background"], case["reconstructed"],
        case["graph"], graph_dir=graph_dir, text_mask=case["text_mask"],
        trusted_root=tmp_path,
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
        initial_component_count=1,
        expected_component_ids=["component_0001"],
    )
    unknown = evaluate_component_quality_round(
        case["source"], case["background"], case["reconstructed"],
        case["graph"], graph_dir=graph_dir, text_mask=case["text_mask"],
        trusted_root=tmp_path,
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={},
        initial_component_count=1,
        expected_component_ids=["component_0001"],
    )
    assert accepted["accepted"] is True
    assert unknown["accepted"] is False
    assert {"protected_native_overlap_unknown", "pptx_reopen_unknown"} <= set(unknown["violations"])


def test_repair_quality_round_rejects_reliable_text_without_editable_object(tmp_path) -> None:
    case = _synthetic_quality_case()
    graph_dir = tmp_path / "round"
    mask_path = graph_dir / "masks/component_0001.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(case["component_mask"].astype(np.uint8) * 255).save(mask_path)
    case["graph"]["nodes"][0]["mask_sha256"] = hashlib.sha256(
        mask_path.read_bytes()
    ).hexdigest()

    report = evaluate_component_quality_round(
        case["source"], case["background"], case["reconstructed"],
        case["graph"], graph_dir=graph_dir, text_mask=case["text_mask"],
        trusted_root=tmp_path,
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
        initial_component_count=1,
        expected_component_ids=["component_0001"], text_items=[],
    )

    assert "editable_text_once" in report["violations"]


def test_repair_quality_round_requires_one_editable_item_per_text_node(tmp_path) -> None:
    case = _synthetic_quality_case()
    graph_dir = tmp_path / "round"
    mask_path = graph_dir / "masks/component_0001.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(case["component_mask"].astype(np.uint8) * 255).save(mask_path)
    case["graph"]["nodes"][0]["mask_sha256"] = hashlib.sha256(
        mask_path.read_bytes()
    ).hexdigest()
    for index in range(2):
        case["graph"]["nodes"].append({
            "id": f"text_{index + 1:04d}", "kind": "text",
            "parent_id": None, "state": "frozen",
            "mask": f"masks/text_{index + 1:04d}.png",
            "mask_sha256": chr(ord("b") + index) * 64,
            "bbox": [24, 20, 40, 24], "z_index": index + 1,
            "text_ids": [],
        })

    report = evaluate_component_quality_round(
        case["source"], case["background"], case["reconstructed"],
        case["graph"], graph_dir=graph_dir, text_mask=case["text_mask"],
        trusted_root=tmp_path,
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
        initial_component_count=1,
        expected_component_ids=["component_0001"],
        text_items=[{"text": "only one", "box": [24, 20, 40, 24]}],
    )

    assert "editable_text_once" in report["violations"]


def test_repair_quality_owner_map_includes_frozen_components(tmp_path) -> None:
    shape = (40, 48)
    source = np.full((*shape, 3), 96, dtype=np.uint8)
    background = source.copy()
    reconstructed = source.copy()
    left = np.zeros(shape, dtype=bool)
    right = np.zeros(shape, dtype=bool)
    left[10:30, 8:20] = True
    right[10:30, 21:33] = True
    source[left | right] = 180
    reconstructed[left | right] = 180
    shared = np.zeros(shape, dtype=bool)
    shared[12:28, 20] = True
    source[shared] = background[shared] = reconstructed[shared] = 50
    graph_dir = tmp_path / "round"
    masks_dir = graph_dir / "masks"
    masks_dir.mkdir(parents=True)
    nodes = []
    for component_id, state, mask in (
        ("pending", "pending_gate", left), ("frozen", "frozen", right)
    ):
        path = masks_dir / f"{component_id}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        ys, xs = np.where(mask)
        nodes.append({
            "id": component_id, "kind": "parent", "parent_id": None,
            "state": state, "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "z_index": len(nodes), "text_ids": [],
        })
    page = evaluate_component_quality_round(
        source, background, reconstructed, {"nodes": nodes},
        graph_dir=graph_dir, trusted_root=tmp_path,
        text_mask=np.zeros(shape, dtype=bool),
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
        initial_component_count=2, expected_component_ids=["pending"],
    )
    assert "duplicate_shadow" not in page["component_reports"][0]["violations"]
    assert "orphan_residual" in page["violations"]


@pytest.mark.parametrize(
    "visual_metrics",
    [
        {"mae": 0.0, "p95": 0.0},
        {"mae": float("nan"), "p95": 0.0, "changed_ratio": 0.0},
    ],
)
def test_page_quality_rejects_incomplete_or_nonfinite_visual_metrics(visual_metrics: dict) -> None:
    with pytest.raises(ValueError, match="visual_metrics"):
        evaluate_page_quality(
            [], visual_metrics=visual_metrics,
            page_checks={"pptx_reopen": "pass"},
            expected_component_ids=[], initial_component_count=0,
        )


def test_page_quality_cannot_accept_empty_reports_for_nonempty_page() -> None:
    with pytest.raises(ValueError, match="component reports"):
        evaluate_page_quality(
            [], visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
            page_checks={"pptx_reopen": "pass"},
            expected_component_ids=["component_0001"], initial_component_count=1,
        )


def test_repair_quality_round_rejects_graph_hash_tampering(tmp_path) -> None:
    case = _synthetic_quality_case()
    graph_dir = tmp_path / "round"
    mask_path = graph_dir / "masks/component_0001.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(case["component_mask"].astype(np.uint8) * 255).save(mask_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        evaluate_component_quality_round(
            case["source"], case["background"], case["reconstructed"], case["graph"],
            graph_dir=graph_dir, text_mask=case["text_mask"],
            trusted_root=tmp_path,
            visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
            page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
            initial_component_count=1, expected_component_ids=["component_0001"],
        )


def test_repair_quality_round_rejects_expected_id_mismatch(tmp_path) -> None:
    case = _synthetic_quality_case()
    graph_dir = tmp_path / "round"
    mask_path = graph_dir / "masks/component_0001.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(case["component_mask"].astype(np.uint8) * 255).save(mask_path)
    case["graph"]["nodes"][0]["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="expected IDs"):
        evaluate_component_quality_round(
            case["source"], case["background"], case["reconstructed"], case["graph"],
            graph_dir=graph_dir, text_mask=case["text_mask"],
            trusted_root=tmp_path,
            visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
            page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
            initial_component_count=1, expected_component_ids=["other"],
        )


def test_repair_quality_round_rejects_symlinked_ancestor(tmp_path) -> None:
    case = _synthetic_quality_case()
    outside = tmp_path / "outside"
    mask_path = outside / "round/masks/component_0001.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(case["component_mask"].astype(np.uint8) * 255).save(mask_path)
    case["graph"]["nodes"][0]["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")
    with pytest.raises(ValueError, match="directory chain"):
        evaluate_component_quality_round(
            case["source"], case["background"], case["reconstructed"], case["graph"],
            graph_dir=linked / "round", trusted_root=tmp_path, text_mask=case["text_mask"],
            visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
            page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
            initial_component_count=1, expected_component_ids=["component_0001"],
        )


def test_repair_quality_round_detects_graph_directory_replacement(tmp_path, monkeypatch) -> None:
    import scripts.visual_segment as visual_segment

    case = _synthetic_quality_case()
    graph_dir = tmp_path / "round"
    mask_path = graph_dir / "masks/component_0001.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(case["component_mask"].astype(np.uint8) * 255).save(mask_path)
    case["graph"]["nodes"][0]["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    real_read = visual_segment._read_action_mask

    def replace_after_read(*args, **kwargs):
        loaded = real_read(*args, **kwargs)
        graph_dir.rename(tmp_path / "original-round")
        graph_dir.mkdir()
        return loaded

    monkeypatch.setattr(visual_segment, "_read_action_mask", replace_after_read)
    with pytest.raises(RuntimeError, match="directory identity changed"):
        evaluate_component_quality_round(
            case["source"], case["background"], case["reconstructed"], case["graph"],
            graph_dir=graph_dir, trusted_root=tmp_path, text_mask=case["text_mask"],
            visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
            page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
            initial_component_count=1, expected_component_ids=["component_0001"],
        )


def test_repair_quality_round_rejects_dotdot_escape(tmp_path) -> None:
    case = _synthetic_quality_case()
    trusted = tmp_path / "trusted"
    graph_dir = tmp_path / "outside/round"
    mask_path = graph_dir / "masks/component_0001.png"
    trusted.mkdir()
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(case["component_mask"].astype(np.uint8) * 255).save(mask_path)
    case["graph"]["nodes"][0]["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    escaped = trusted / ".." / "outside" / "round"
    with pytest.raises(ValueError, match="semantic path segments"):
        evaluate_component_quality_round(
            case["source"], case["background"], case["reconstructed"], case["graph"],
            graph_dir=escaped, trusted_root=trusted, text_mask=case["text_mask"],
            visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
            page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
            initial_component_count=1, expected_component_ids=["component_0001"],
        )
