from __future__ import annotations

import copy
import json
import hashlib
import hmac
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import stat
import subprocess
import sys

import pytest
import cv2

from image2editable.component_repair import (
    EVIDENCE_NAMES,
    advance_component_repair,
    build_component_agent_request as _build_component_agent_request,
    initialize_component_repair_state,
    load_component_agent_graph,
    load_component_agent_request,
    record_component_execution,
    record_next_component_request,
    record_parent_fallback_execution,
    record_parent_fallback_quality,
    record_component_quality,
    record_local_component_plan,
)
import image2editable.component_repair as component_repair
import image2editable.component_quality as component_quality
import image2editable.legacy as legacy
import image2editable.runtime as runtime
from image2editable.execution import ExecutionLease
from scripts.fg_extract import _fill_component_underlay
from scripts.visual_segment import (
    RecoverableComponentPlanError,
    VisualSegmentationError,
    _publish_action_directory,
    execute_component_actions,
)
from scripts.sam_worker import component_prompt_mask, run_component_prompt_worker
from PIL import Image
import numpy as np


def _page_quality_key_case(**metrics: float) -> dict:
    return {
        "violations": ["background_text_residual"],
        "visual_metrics": {
            "largest_unexplained_region_pixels": 120,
            "unexplained_visual_pixels": 120,
            "mae": 1.0,
            "p95": 2.0,
            **metrics,
        },
    }


def test_page_progress_key_orders_improved_equal_and_worse_quality() -> None:
    quality = _page_quality_key_case()
    improved = _page_quality_key_case(largest_unexplained_region_pixels=119)
    worse = _page_quality_key_case(largest_unexplained_region_pixels=121)

    key = component_repair._page_progress_key(quality)
    assert component_repair._page_progress_key(improved) < key
    assert component_repair._page_progress_key(copy.deepcopy(quality)) == key
    assert component_repair._page_progress_key(worse) > key


def test_page_progress_key_compares_legacy_quality_deterministically() -> None:
    quality = {
        "violations": ["background_text_residual"],
        "visual_metrics": {
            "mae": 1.0,
            "p95": 2.0,
        },
    }

    assert component_repair._page_progress_key(quality)[:2] == (0.0, 0.0)


def test_page_progress_key_counts_resolved_blocking_violation_as_progress() -> None:
    previous = _page_quality_key_case()
    previous["violations"].append("contained_parent_review")
    current = _page_quality_key_case()

    assert component_repair._page_progress_key(current) < (
        component_repair._page_progress_key(previous)
    )


@pytest.mark.parametrize("quality", [{}, {"violations": [], "visual_metrics": []}])
def test_page_progress_key_rejects_invalid_quality(quality: dict) -> None:
    with pytest.raises(ValueError, match="progress check"):
        component_repair._page_progress_key(quality)


def test_unowned_raster_text_check_accepts_diagnostic_owned_by_editable_text() -> None:
    diagnostics = [{"bbox": [1045, 335, 1461, 366]}]
    text_items = [{"box": [1047, 337, 436, 30]}]

    assert component_repair._unowned_raster_text_check(
        diagnostics, text_items
    ) == "pass"


def test_unowned_raster_text_check_rejects_spatially_uncovered_diagnostic() -> None:
    diagnostics = [{"bbox": [1045, 335, 1461, 366]}]
    text_items = [{"box": [200, 100, 300, 30]}]

    assert component_repair._unowned_raster_text_check(
        diagnostics, text_items
    ) == "fail"


def _rounded_rectangle_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (22, 8), (73, 55), 255, thickness=-1)
    cv2.rectangle(mask, (12, 18), (83, 45), 255, thickness=-1)
    cv2.circle(mask, (22, 18), 10, 255, thickness=-1)
    cv2.circle(mask, (73, 18), 10, 255, thickness=-1)
    cv2.circle(mask, (22, 45), 10, 255, thickness=-1)
    cv2.circle(mask, (73, 45), 10, 255, thickness=-1)
    return mask.astype(bool)


def _interior_gradient_jump_p95(rgb: np.ndarray, hole: np.ndarray) -> float:
    interior = cv2.erode(
        hole.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    laplacian = np.max(np.abs(
        cv2.Laplacian(rgb.astype(np.float32), cv2.CV_32F, ksize=1)
    ), axis=2)
    return float(np.percentile(laplacian[interior], 95))


def test_build_presentation_layer_repairs_gradient_component_holes() -> None:
    from scripts.component_underlay import build_presentation_layer

    height, width = 64, 96
    y, x = np.mgrid[:height, :width]
    source = np.dstack((2 * x, 2 * y, x + y)).astype(np.uint8)
    text_clean = source.copy()
    semantic = _rounded_rectangle_mask(height, width)
    child = np.zeros((height, width), dtype=bool)
    child[16:48, 32:64] = True
    text = np.zeros((height, width), dtype=bool)
    text[30:35, 20:32] = True
    ownership = semantic & ~(child | text)
    higher_z = child.copy()
    text_clean[text] = (17, 31, 43)

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=text_clean,
        ownership_mask=ownership,
        semantic_mask=semantic,
        higher_layer_mask=higher_z,
        text_mask=text,
    )

    assert set(layer) == {
        "rgb", "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask", "metrics",
    }
    assert np.all(layer["ownership_mask"] <= ownership)
    assert not np.any(layer["presentation_alpha_mask"] & ~semantic)
    assert np.all(
        layer["generated_underlay_mask"]
        >= (semantic & ~ownership & (child | text))
    )
    assert np.all(layer["presentation_alpha_mask"][layer["generated_underlay_mask"]])
    assert not np.array_equal(layer["rgb"][text], text_clean[text])
    assert layer["metrics"]["boundary_color_mae"] <= 3.0
    assert layer["metrics"]["gradient_jump_p95"] <= 6.0
    assert all(np.isfinite(value) for value in layer["metrics"].values())


def test_presentation_layer_keeps_source_pixels_owned_by_visual() -> None:
    from scripts.component_underlay import build_presentation_layer

    source = np.full((7, 7, 3), 240, dtype=np.uint8)
    source[2:5, 2:5] = (20, 80, 180)
    text_clean = source.copy()
    text_clean[2:5, 2:5] = 240
    ownership = np.zeros((7, 7), dtype=bool)
    ownership[2:5, 2:5] = True

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=text_clean,
        ownership_mask=ownership,
        semantic_mask=ownership,
        higher_layer_mask=np.zeros_like(ownership),
        text_mask=np.zeros_like(ownership),
    )

    assert np.array_equal(layer["rgb"][ownership], source[ownership])


def test_presentation_layer_removes_active_text_from_visual_ownership() -> None:
    from scripts.component_underlay import build_presentation_layer

    source = np.full((9, 11, 3), (30, 150, 70), dtype=np.uint8)
    source[3:6, 4:7] = 255
    text_clean = np.full_like(source, (30, 150, 70))
    semantic = np.ones((9, 11), dtype=bool)
    text = np.zeros_like(semantic)
    text[3:6, 4:7] = True

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=text_clean,
        ownership_mask=semantic,
        semantic_mask=semantic,
        higher_layer_mask=np.zeros_like(semantic),
        text_mask=text,
    )

    assert not np.any(layer["ownership_mask"] & text)
    assert np.all(layer["generated_underlay_mask"][text])
    assert np.array_equal(layer["rgb"][text], text_clean[text])


def test_presentation_layer_preserves_verified_text_clean_underlay() -> None:
    from scripts.component_underlay import build_presentation_layer

    green = np.array((30, 150, 70), dtype=np.uint8)
    source = np.full((15, 21, 3), green, dtype=np.uint8)
    text = np.zeros((15, 21), dtype=bool)
    text[6:9, 8:13] = True
    source[text] = 255
    text_clean = np.full_like(source, green)
    semantic = np.ones(text.shape, dtype=bool)

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=text_clean,
        ownership_mask=semantic,
        semantic_mask=semantic,
        higher_layer_mask=np.zeros_like(semantic),
        text_mask=text,
    )

    assert np.array_equal(layer["rgb"][text], text_clean[text])


def test_presentation_layer_repairs_discontinuous_text_clean_on_gradient() -> None:
    from scripts.component_underlay import build_presentation_layer

    height, width = 64, 96
    y, x = np.mgrid[:height, :width]
    gradient = np.dstack((2 * x, 2 * y, x + y)).astype(np.uint8)
    semantic = _rounded_rectangle_mask(height, width)
    text = np.zeros((height, width), dtype=bool)
    text[22:42, 32:64] = True
    source = gradient.copy()
    source[text] = 0
    text_clean = gradient.copy()
    text_clean[text] = (240, 20, 20)

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=text_clean,
        ownership_mask=semantic,
        semantic_mask=semantic,
        higher_layer_mask=np.zeros_like(semantic),
        text_mask=text,
    )

    assert not np.array_equal(layer["rgb"][text], text_clean[text])
    assert layer["metrics"]["boundary_color_mae"] <= 6.0
    assert layer["metrics"]["gradient_jump_p95"] <= 12.0


def test_presentation_layer_handles_saturated_gradient_beside_text_hole() -> None:
    from scripts.component_underlay import build_presentation_layer

    source = np.full((64, 96, 3), (90, 168, 112), dtype=np.uint8)
    semantic = np.zeros(source.shape[:2], dtype=bool)
    semantic[5:59, 5:91] = True
    source[5:40, 5:91] = (12, 129, 44)
    text = np.zeros_like(semantic)
    text[15:39, 25:71] = True
    text_clean = source.copy()
    text_clean[text] = (40, 180, 60)

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=text_clean,
        ownership_mask=semantic & ~text,
        semantic_mask=semantic,
        higher_layer_mask=np.zeros_like(semantic),
        text_mask=text,
    )

    assert layer["metrics"]["boundary_color_mae"] <= 6.0
    assert layer["metrics"]["gradient_jump_p95"] <= 12.0


def test_presentation_layer_offers_smooth_repair_for_visual_holes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import component_underlay

    source = np.full((20, 30, 3), 180, dtype=np.uint8)
    semantic = np.ones(source.shape[:2], dtype=bool)
    higher = np.zeros_like(semantic)
    higher[7:13, 11:19] = True
    calls: list[bool] = []
    real_choose = component_underlay._choose_visual_fill

    def observe(**kwargs):
        calls.append(kwargs["allow_smooth_surface"])
        return real_choose(**kwargs)

    monkeypatch.setattr(component_underlay, "_choose_visual_fill", observe)
    component_underlay.build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=semantic & ~higher,
        semantic_mask=semantic,
        higher_layer_mask=higher,
        text_mask=np.zeros_like(semantic),
    )

    assert calls == [True]


def test_underlay_gradient_avoids_nearest_donor_seams() -> None:
    from scripts.component_underlay import build_presentation_layer

    height, width = 64, 96
    y, x = np.mgrid[:height, :width]
    source = np.dstack((2 * x, 2 * y, x + y)).astype(np.uint8)
    semantic = _rounded_rectangle_mask(height, width)
    child = np.zeros((height, width), dtype=bool)
    child[16:48, 32:64] = True
    ownership = semantic & ~child
    source_with_child = source.copy()
    source_with_child[child] = (240, 20, 20)
    legacy = _fill_component_underlay(source_with_child, child, ownership)

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=ownership,
        semantic_mask=semantic,
        higher_layer_mask=child,
        text_mask=np.zeros_like(child),
    )

    legacy_jump = _interior_gradient_jump_p95(legacy, child)
    new_jump = _interior_gradient_jump_p95(layer["rgb"], child)
    assert legacy_jump > 30.0
    assert new_jump <= 6.0
    assert legacy_jump >= new_jump * 5.0
    assert layer["metrics"]["boundary_color_mae"] <= 3.0
    assert layer["metrics"]["gradient_jump_p95"] <= 6.0


def test_presentation_layer_metrics_ignore_visible_child_pixels() -> None:
    from scripts.component_underlay import build_presentation_layer

    height, width = 64, 96
    y, x = np.mgrid[:height, :width]
    parent_truth = np.dstack((2 * x, 2 * y, x + y)).astype(np.uint8)
    semantic = _rounded_rectangle_mask(height, width)
    child = np.zeros((height, width), dtype=bool)
    child[16:48, 32:64] = True
    ownership = semantic & ~child
    source_with_child = parent_truth.copy()
    source_with_child[child] = (240, 20, 20)
    arguments = {
        "ownership_mask": ownership,
        "semantic_mask": semantic,
        "higher_layer_mask": child,
        "text_mask": np.zeros_like(child),
    }

    reference = build_presentation_layer(
        source_rgb=parent_truth, text_clean_rgb=parent_truth, **arguments,
    )
    layer = build_presentation_layer(
        source_rgb=source_with_child,
        text_clean_rgb=source_with_child,
        **arguments,
    )

    true_mae = np.abs(
        layer["rgb"][child].astype(np.int16)
        - parent_truth[child].astype(np.int16)
    ).mean()
    assert true_mae <= 8.0
    assert layer["metrics"] == reference["metrics"]
    assert layer["metrics"]["boundary_color_mae"] <= 3.0
    assert layer["metrics"]["gradient_jump_p95"] <= 6.0


def test_visual_fill_selects_smaller_lexicographic_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import component_underlay

    y, x = np.mgrid[:9, :9]
    source = np.dstack((2 * x, 2 * y, x + y)).astype(np.uint8)
    semantic = np.ones((9, 9), dtype=bool)
    hole = np.zeros((9, 9), dtype=bool)
    hole[3:6, 3:6] = True

    def fake_inpaint(
        image: np.ndarray, mask: np.ndarray, radius: int, method: int,
    ) -> np.ndarray:
        candidate = image.copy()
        if method == cv2.INPAINT_TELEA:
            candidate[mask.astype(bool)] = 200
        return candidate

    monkeypatch.setattr(component_underlay.cv2, "inpaint", fake_inpaint)
    layer = component_underlay.build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=semantic & ~hole,
        semantic_mask=semantic,
        higher_layer_mask=hole,
        text_mask=np.zeros_like(hole),
    )

    assert np.array_equal(layer["rgb"][hole], source[hole])


def test_visual_fill_prefers_existing_clean_candidate_when_better(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import component_underlay

    source = np.full((5, 5, 3), 100, dtype=np.uint8)
    semantic = np.ones((5, 5), dtype=bool)
    hole = np.zeros((5, 5), dtype=bool)
    hole[2, 2] = True
    ownership = semantic & ~hole
    telea = source.copy()
    telea[hole] = 90
    navier_stokes = source.copy()
    navier_stokes[hole] = 110

    assert component_underlay._visual_metrics(
        telea, source, ownership, hole,
    ) == component_underlay._visual_metrics(
        navier_stokes, source, ownership, hole,
    )

    def fake_inpaint(
        image: np.ndarray, mask: np.ndarray, radius: int, method: int,
    ) -> np.ndarray:
        return telea if method == cv2.INPAINT_TELEA else navier_stokes

    monkeypatch.setattr(component_underlay.cv2, "inpaint", fake_inpaint)
    layer = component_underlay.build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=ownership,
        semantic_mask=semantic,
        higher_layer_mask=hole,
        text_mask=np.zeros_like(semantic),
    )

    assert np.all(layer["rgb"][hole] == 100)
    assert np.array_equal(layer["rgb"][~hole], source[~hole])


def test_visual_fill_prefers_clean_local_fill_over_interior_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import component_underlay

    source = np.full((50, 70, 3), 220, dtype=np.uint8)
    semantic = np.ones((50, 70), dtype=bool)
    hole = np.zeros((50, 70), dtype=bool)
    large_hole = np.zeros_like(hole)
    large_hole[12:38, 20:50] = True
    hole |= large_hole
    hole[3, 3] = True

    def patched_inpaint(image, mask, radius, method):
        candidate = image.copy()
        y, x = np.indices(mask.shape)
        selected = mask.astype(bool)
        checker = ((x + y) % 2)[selected][:, None]
        candidate[selected] = np.where(checker, 255, 0)
        return candidate

    monkeypatch.setattr(component_underlay.cv2, "inpaint", patched_inpaint)
    layer = component_underlay.build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=semantic & ~hole,
        semantic_mask=semantic,
        higher_layer_mask=hole,
        text_mask=np.zeros_like(hole),
    )

    assert (
        np.max(layer["rgb"][large_hole])
        - np.min(layer["rgb"][large_hole])
        <= 1
    )
    assert layer["metrics"]["added_high_frequency_pixels"] == 0


def test_visual_metrics_count_only_interior_high_frequency() -> None:
    from scripts.component_underlay import _visual_metrics

    source = np.full((30, 40, 3), 120, dtype=np.uint8)
    hole = np.zeros((30, 40), dtype=bool)
    hole[8:22, 10:30] = True
    donor = ~hole
    boundary_only = source.copy()
    boundary = hole & ~cv2.erode(
        hole.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    boundary_only[boundary] = 200
    patched = boundary_only.copy()
    patched[12:18, 15:25:2] = 255

    boundary_metrics = _visual_metrics(boundary_only, source, donor, hole)
    patched_metrics = _visual_metrics(patched, source, donor, hole)

    assert boundary_metrics["added_high_frequency_pixels"] == 0
    assert patched_metrics["added_high_frequency_pixels"] > 0


def test_visual_fill_handles_multiple_holes_near_page_edge() -> None:
    from scripts.component_underlay import build_presentation_layer

    source = np.arange(3 * 7 * 3, dtype=np.uint8).reshape(3, 7, 3)
    semantic = np.ones((3, 7), dtype=bool)
    holes = np.zeros((3, 7), dtype=bool)
    holes[1, 1] = True
    holes[1, 4] = True

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=semantic & ~holes,
        semantic_mask=semantic,
        higher_layer_mask=holes,
        text_mask=np.zeros_like(holes),
    )

    assert np.array_equal(layer["rgb"][~holes], source[~holes])
    assert all(np.isfinite(value) for value in layer["metrics"].values())


def test_visual_fill_rebuilds_smooth_canvas_hole_from_outside_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import component_underlay

    y, x = np.mgrid[:70, :100]
    source = np.dstack((180 + x // 3, 190 + y // 4, 200 + (x + y) // 6))
    source = source.astype(np.uint8)
    hole = np.zeros(source.shape[:2], dtype=bool)
    hole[20:50, 25:75] = True
    damaged = source.copy()
    damaged[hole] = np.where(((x + y)[hole] % 2)[:, None], 0, 255)

    monkeypatch.setattr(
        component_underlay.cv2,
        "inpaint",
        lambda image, mask, radius, method: np.where(
            mask[:, :, None] > 0, 255, image
        ).astype(np.uint8),
    )
    rebuilt, _ = component_underlay._choose_visual_fill(
        rgb=damaged,
        source_rgb=source,
        semantic_mask=hole,
        donor_mask=~hole,
        visual_hole=hole,
        allow_smooth_surface=True,
    )

    assert float(np.abs(rebuilt[hole].astype(int) - source[hole]).mean()) <= 2.0


def test_gradient_continuation_keeps_text_hole_boundary_smooth() -> None:
    from scripts import component_underlay

    height, width = 48, 80
    y, x = np.mgrid[:height, :width]
    source = np.dstack((80 + 2 * x, 60 + y, 40 + x + y)).astype(np.uint8)
    hole = np.zeros((height, width), dtype=bool)
    hole[14:34, 24:56] = True
    damaged = source.copy()
    damaged[hole] = (240, 20, 20)

    repaired = component_underlay._continue_boundary_gradient(damaged, ~hole, hole)
    metrics = component_underlay._visual_metrics(repaired, source, ~hole, hole)

    assert metrics["boundary_color_mae"] <= 6.0
    assert metrics["gradient_jump_p95"] <= 12.0
    assert metrics["added_high_frequency_pixels"] == 0.0


def test_gradient_continuation_smooths_noisy_hole_interior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import component_underlay

    source = np.full((48, 80, 3), 120, dtype=np.uint8)
    hole = np.zeros(source.shape[:2], dtype=bool)
    hole[10:38, 20:60] = True

    def noisy_inpaint(image, mask, radius, method):
        output = image.copy()
        y, x = np.indices(mask.shape)
        selected = mask.astype(bool)
        output[selected] = np.where(
            ((x + y) % 2)[selected, None], 255, 0,
        )
        return output

    monkeypatch.setattr(component_underlay.cv2, "inpaint", noisy_inpaint)
    repaired = component_underlay._continue_boundary_gradient(
        source, ~hole, hole,
    )
    metrics = component_underlay._visual_metrics(
        repaired, source, ~hole, hole,
    )

    assert metrics["gradient_jump_p95"] <= 12.0
    assert metrics["added_high_frequency_pixels"] == 0.0


def test_presentation_layer_removes_higher_layer_antialias_halo() -> None:
    from scripts.component_underlay import build_presentation_layer

    height, width = 80, 120
    y, x = np.mgrid[:height, :width]
    truth = np.dstack((220 + x // 30, 225 + y // 30, 230 + x // 40)).astype(
        np.uint8
    )
    semantic = np.zeros((height, width), dtype=bool)
    semantic[10:70, 10:110] = True
    higher = np.zeros((height, width), dtype=bool)
    higher[32:48, 52:68] = True
    contaminated = cv2.dilate(
        higher.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
    ).astype(bool)
    source = truth.copy()
    source[contaminated] = (30, 80, 190)
    ownership = semantic & ~higher

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=ownership,
        semantic_mask=semantic,
        higher_layer_mask=higher,
        text_mask=np.zeros_like(higher),
    )

    assert not np.any(layer["ownership_mask"] & contaminated)
    assert np.all(layer["generated_underlay_mask"][contaminated & semantic])
    repaired_error = np.abs(
        layer["rgb"][contaminated].astype(np.int16)
        - truth[contaminated].astype(np.int16)
    )
    assert repaired_error.mean() <= 8.0


def test_presentation_layer_ignores_adjacent_higher_layer_at_semantic_edge() -> None:
    from scripts.component_underlay import build_presentation_layer

    source = np.full((64, 112, 3), 220, dtype=np.uint8)
    semantic = np.zeros(source.shape[:2], dtype=bool)
    semantic[12:52, 12:100] = True
    higher = np.zeros_like(semantic)
    higher[10:14, 20:92] = True

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=semantic,
        semantic_mask=semantic,
        higher_layer_mask=higher,
        text_mask=np.zeros_like(semantic),
    )

    assert np.array_equal(layer["ownership_mask"], semantic & ~higher)
    assert not np.any(layer["generated_underlay_mask"])


def test_presentation_layer_skips_visual_hole_without_ownership_donor() -> None:
    from scripts.component_underlay import build_presentation_layer

    source = np.zeros((5, 5, 3), dtype=np.uint8)
    semantic = np.ones((5, 5), dtype=bool)

    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=np.zeros_like(semantic),
        semantic_mask=semantic,
        higher_layer_mask=semantic,
        text_mask=np.zeros_like(semantic),
    )

    assert not np.any(layer["presentation_alpha_mask"])
    assert not np.any(layer["generated_underlay_mask"])


def test_visual_fill_never_changes_pixels_outside_hole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import component_underlay

    source = np.arange(9 * 9 * 3, dtype=np.uint8).reshape(9, 9, 3)
    semantic = np.ones((9, 9), dtype=bool)
    hole = np.zeros((9, 9), dtype=bool)
    hole[3:6, 3:6] = True

    monkeypatch.setattr(
        component_underlay.cv2,
        "inpaint",
        lambda image, mask, radius, method: np.full_like(image, 255),
    )
    layer = component_underlay.build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=semantic & ~hole,
        semantic_mask=semantic,
        higher_layer_mask=hole,
        text_mask=np.zeros_like(hole),
    )

    assert np.array_equal(layer["rgb"][~hole], source[~hole])


def test_advance_without_state_only_reports_needs_initialization(tmp_path: Path) -> None:
    from image2editable.inputs import prepare_image_job
    from image2editable.store import RunStore

    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)

    outcome = advance_component_repair(store, "page_001")

    assert outcome == {"status": "needs_initialization", "page_id": "page_001"}
    reconstruction = run_dir / "pages/page_001/reconstruction"
    assert not (reconstruction / "component_state.json").exists()
    assert not (reconstruction / "agent").exists()


def test_component_repair_rejects_unheld_execution_lease(page_session: dict) -> None:
    from image2editable.execution import ExecutionLease
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    lease = ExecutionLease(store.root / "execution.lock", run_root=store.root)

    with pytest.raises(RuntimeError, match="held Run execution lease"):
        initialize_component_repair_state(
            store, "page_001", request_path=request_path,
            initial_component_count=2, _lease=lease,
        )


def test_execution_lease_authorizes_only_its_held_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    lease = ExecutionLease(run / "execution.lock", run_root=run)

    with pytest.raises(RuntimeError, match="not held"):
        lease.assert_authorizes(run)

    with lease:
        lease.assert_authorizes(run)
        with pytest.raises(RuntimeError, match="different Run"):
            lease.assert_authorizes(other)

    with pytest.raises(RuntimeError, match="not held"):
        lease.assert_authorizes(run)


@pytest.mark.parametrize(
    ("lease_path", "run_root"),
    [
        ("execution.lock", None),
        ("not-execution.lock", "run"),
    ],
)
def test_execution_lease_rejects_unbound_or_nonstandard_run_lock(
    tmp_path: Path, lease_path: str, run_root: str | None,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    lease = ExecutionLease(
        run / lease_path,
        run_root=run if run_root is not None else None,
    )

    with lease:
        with pytest.raises(RuntimeError, match="different Run"):
            lease.assert_authorizes(run)


def test_execution_lease_rejects_replaced_lock_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    lease = ExecutionLease(run / "execution.lock", run_root=run)

    with lease:
        if os.name == "nt":
            original_lstat = Path.lstat

            class ReplacedPathStatus:
                def __init__(self, status: os.stat_result) -> None:
                    self._status = status

                def __getattr__(self, name: str) -> object:
                    if name == "st_ino":
                        return self._status.st_ino + 1
                    return getattr(self._status, name)

            def replaced_lstat(path: Path) -> object:
                result = original_lstat(path)
                if path == lease.path:
                    return ReplacedPathStatus(result)
                return result

            monkeypatch.setattr(Path, "lstat", replaced_lstat)
        else:
            replacement = run / "replacement.lock"
            replacement.write_bytes(b"replacement")
            os.replace(replacement, lease.path)

        with pytest.raises(RuntimeError):
            lease.assert_authorizes(run)


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX parent descriptor required",
)
def test_execution_lease_rejects_missing_posix_parent_descriptor(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    lease = ExecutionLease(run / "execution.lock", run_root=run)

    with lease:
        descriptor = lease._parent_descriptor
        assert descriptor is not None
        lease._parent_descriptor = None
        try:
            with pytest.raises(RuntimeError, match="parent is not held"):
                lease.assert_authorizes(run)
        finally:
            lease._parent_descriptor = descriptor


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX parent descriptor required",
)
def test_execution_lease_rejects_unlocked_posix_parent_descriptor(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    lease = ExecutionLease(run / "execution.lock", run_root=run)

    with lease:
        assert lease._parent_descriptor is not None
        assert lease._parent_locked
        lease._parent_locked = False
        try:
            with pytest.raises(RuntimeError, match="parent is not held"):
                lease.assert_authorizes(run)
        finally:
            lease._parent_locked = True


@pytest.mark.skipif(
    os.name != "nt", reason="Windows parent validator delegation required",
)
def test_execution_lease_rejects_unlocked_parent_descriptor_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import image2editable.execution as execution

    run = tmp_path / "run"
    run.mkdir()
    lease = ExecutionLease(run / "execution.lock", run_root=run)

    with lease:
        monkeypatch.setattr(execution, "_validate_open_parent", lambda *_: None)
        lease._parent_descriptor = -1
        try:
            with pytest.raises(RuntimeError, match="parent is not held"):
                lease.assert_authorizes(run)
        finally:
            lease._parent_descriptor = None


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX parent descriptor required",
)
def test_execution_lease_rejects_replaced_posix_parent_path(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    moved_run = tmp_path / "moved-run"
    run.mkdir()
    lease = ExecutionLease(run / "execution.lock", run_root=run)

    with lease:
        run.rename(moved_run)
        run.mkdir()
        with pytest.raises(RuntimeError, match="parent identity changed"):
            lease.assert_authorizes(run)


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX execution lease covers publication",
)
def test_component_request_reuses_held_execution_lease(
    page_session: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    run_root = reconstruction.parents[2]

    def reject_nested_publication_lease(path: Path) -> object:
        raise AssertionError(f"nested publication lease acquired: {path}")

    monkeypatch.setattr(
        component_repair,
        "_run_publication_lease",
        reject_nested_publication_lease,
    )

    with ExecutionLease(
        run_root / "execution.lock", run_root=run_root,
    ) as lease:
        request_path = build_component_agent_request(
            page_session, repair_round=1, _lease=lease,
        )

    assert request_path.is_file()


def test_direct_component_request_uses_publication_lease(
    page_session: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered: list[Path] = []

    class TrackingPublicationLease:
        def __init__(self, reconstruction: Path) -> None:
            self.reconstruction = reconstruction

        def __enter__(self) -> None:
            entered.append(self.reconstruction)

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        component_repair,
        "_run_publication_lease",
        TrackingPublicationLease,
    )

    request_path = build_component_agent_request(page_session, repair_round=1)

    assert request_path.is_file()
    assert entered == [Path(page_session["reconstruction_dir"])]


def test_component_request_rejects_execution_lease_for_different_run(
    page_session: dict, tmp_path: Path,
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    with ExecutionLease(
        other / "execution.lock", run_root=other,
    ) as lease:
        with pytest.raises(RuntimeError, match="different Run"):
            build_component_agent_request(
                page_session, repair_round=1, _lease=lease,
            )


def test_component_request_rejects_released_execution_lease(
    page_session: dict,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    run_root = reconstruction.parents[2]
    lease = ExecutionLease(run_root / "execution.lock", run_root=run_root)
    with lease:
        pass

    with pytest.raises(RuntimeError, match="not held"):
        build_component_agent_request(
            page_session, repair_round=1, _lease=lease,
        )


def test_initialized_state_points_to_hash_bound_current_request(page_session: dict) -> None:
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    run_root = request_path.parents[5]
    store = RunStore(run_root)
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {"schema_version": 1, "status": "prepared"})
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "processing"}
    }})
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )

    assert state["phase"] == "request_published"
    assert state["repair_round"] == 1
    assert state["plan_count"] == 0
    assert state["delivery_checks"] == {"pptx_reopen": "unknown"}
    assert state["current_round"]["request_ref"]["sha256"] == hashlib.sha256(
        request_path.read_bytes()
    ).hexdigest()
    assert state["current_round"]["request_ref"]["path"] == (
        "pages/page_001/reconstruction/agent/round-01/component_agent_request.json"
    )
    assert advance_component_repair(store, "page_001")["status"] == "awaiting_agent"
    persisted = store.read_json("pages/page_001/reconstruction/component_state.json")
    assert persisted["phase"] == "awaiting_plan"


def test_local_plan_is_hash_bound_and_recorded_without_host_state(
    page_session: dict,
) -> None:
    from image2editable.store import RunStore

    local_session = {**page_session, "provider": "local"}
    request_path = build_component_agent_request(local_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json(
        "job_manifest.json",
        {
            "schema_version": 1,
            "pages": ["page_001"],
            "options": {"agent_provider": "local"},
        },
    )
    initialize_component_repair_state(
        store,
        "page_001",
        request_path=request_path,
        initial_component_count=2,
    )
    assert advance_component_repair(store, "page_001")["status"] == "awaiting_agent"
    plan = {
        "schema_version": 1,
        "kind": "component_plan",
        "page_id": "page_001",
        "provider": "local",
        "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("accept", ["candidate_b"])],
    }

    recorded = record_local_component_plan(store, "page_001", plan=plan)

    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    assert state["phase"] == "plan_recorded"
    assert state["plan_count"] == 1
    assert state["current_round"]["plan_ref"] == recorded["plan_ref"]
    assert not (store.root / "host_capabilities.json").exists()


def test_recoverable_plan_rejection_reopens_same_local_round_and_preserves_plans(
    page_session: dict,
) -> None:
    from image2editable.store import RunStore

    page_session["provider"] = "local"
    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    advance_component_repair(store, "page_001")
    request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
    rejected_plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "local", "repair_round": 1,
        "request_sha256": request_sha256,
        "actions": [_action("absorb_residual", ["candidate_b"])],
    }
    first = record_local_component_plan(
        store, "page_001", plan=rejected_plan,
    )
    rejected_path = store.root / first["plan_ref"]["path"]
    rejected_payload = rejected_path.read_bytes()
    recorded = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    first_rejection_name = (
        f"component-plan-rejection-{recorded['revision'] + 1:08d}.json"
    )
    store.write_json(
        f"pages/page_001/reconstruction/{first_rejection_name}",
        {
            "schema_version": 1,
            "page_id": "page_001",
            "repair_round": 1,
            "request_ref": recorded["current_round"]["request_ref"],
            "rejected_plan_ref": recorded["current_round"]["plan_ref"],
            "reason": "unrelated_residual_target",
        },
    )
    assert runtime._local_plan_correction_context(
        RunStore(store.root), "page_001", request_path
    ) is None

    component_repair.reject_recoverable_component_plan(
        store,
        "page_001",
        repair_round=1,
        request_ref=recorded["current_round"]["request_ref"],
        plan_ref=recorded["current_round"]["plan_ref"],
    )

    reopened = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    assert reopened["phase"] == "awaiting_plan"
    assert reopened["repair_round"] == 1
    assert reopened["current_round"]["request_ref"] == (
        recorded["current_round"]["request_ref"]
    )
    assert reopened["current_round"]["plan_ref"] is None
    assert reopened["current_round"]["execution_ref"] is None
    assert reopened["current_round"]["quality_ref"] is None
    assert reopened["graph_ref"] == recorded["graph_ref"]
    assert reopened["plan_count"] == recorded["plan_count"]
    half_committed_plan = copy.deepcopy(rejected_plan)
    half_committed_plan["actions"] = [_action("discard", ["candidate_b"])]
    half_committed_payload = json.dumps(
        half_committed_plan,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    half_committed_path = rejected_path.with_name(
        f"{rejected_path.stem}-retry-"
        f"{hashlib.sha256(half_committed_payload).hexdigest()[:12]}.json"
    )
    half_committed_path.write_bytes(half_committed_payload)
    resumed = RunStore(store.root)
    correction_context = runtime._local_plan_correction_context(
        resumed, "page_001", request_path
    )
    assert correction_context["rejected_plan"] == rejected_plan
    assert "do not change request_sha256" in correction_context["instruction"]

    second_rejected_plan = copy.deepcopy(rejected_plan)
    second_rejected_plan["actions"][0]["evidence"] = [
        "second unrelated residual target"
    ]
    second = record_local_component_plan(
        store, "page_001", plan=second_rejected_plan,
    )
    second_recorded = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    component_repair.reject_recoverable_component_plan(
        store,
        "page_001",
        repair_round=1,
        request_ref=second_recorded["current_round"]["request_ref"],
        plan_ref=second_recorded["current_round"]["plan_ref"],
    )
    rejection_paths = sorted(
        rejected_path.parent.glob("component-plan-rejection-*.json")
    )
    assert len(rejection_paths) == 2
    assert rejection_paths[0].name == first_rejection_name
    second_context = runtime._local_plan_correction_context(
        RunStore(store.root), "page_001", request_path
    )
    assert second_context["rejected_plan"] == second_rejected_plan
    assert (store.root / second["plan_ref"]["path"]).is_file()

    corrected_plan = copy.deepcopy(rejected_plan)
    corrected_plan["actions"] = [_action("discard", ["candidate_b"])]
    corrected = record_local_component_plan(
        store, "page_001", plan=corrected_plan,
    )
    repeated = record_local_component_plan(
        store, "page_001", plan=corrected_plan,
    )

    assert corrected["plan_ref"] != first["plan_ref"]
    assert "-retry-" in corrected["plan_ref"]["path"]
    assert repeated["plan_ref"] == corrected["plan_ref"]
    assert repeated["recovered"] is True
    assert rejected_path.read_bytes() == rejected_payload

    assert (store.root / corrected["plan_ref"]["path"]).is_file()


def test_recoverable_split_rejection_exposes_bound_correction_context(
    page_session: dict,
) -> None:
    from image2editable.store import RunStore

    page_session["provider"] = "local"
    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    advance_component_repair(store, "page_001")
    first_plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "local", "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("absorb_residual", ["candidate_b"])],
    }
    record_local_component_plan(store, "page_001", plan=first_plan)
    recorded = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    component_repair.reject_recoverable_component_plan(
        store,
        "page_001",
        repair_round=1,
        request_ref=recorded["current_round"]["request_ref"],
        plan_ref=recorded["current_round"]["plan_ref"],
    )
    first_context = runtime._local_plan_correction_context(
        RunStore(store.root), "page_001", request_path
    )
    assert first_context["forbidden_action_pairs"] == [
        ["absorb_residual", "candidate_b"]
    ]

    split_plan = {**first_plan, "actions": [
        _action("split", ["candidate_b"], {"parts": 2})
    ]}
    record_local_component_plan(store, "page_001", plan=split_plan)
    recorded = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    component_repair.reject_recoverable_component_plan(
        store,
        "page_001",
        repair_round=1,
        request_ref=recorded["current_round"]["request_ref"],
        plan_ref=recorded["current_round"]["plan_ref"],
        reason="invalid_split_target",
    )

    context = runtime._local_plan_correction_context(
        RunStore(store.root), "page_001", request_path
    )
    assert context["rejected_plan"] == split_plan
    assert context["forbidden_action_pairs"] == [
        ["absorb_residual", "candidate_b"],
        ["split", "candidate_b"],
    ]
    assert "exact requested number of connected proposals" in context["instruction"]


def test_recoverable_host_plan_execution_returns_to_awaiting_agent(
    page_session: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image2editable.host_agent import record_host_plan
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    manifest = {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    }
    store.write_json("job_manifest.json", manifest)
    store.write_json("run_state.json", {
        "schema_version": 1, "status": "awaiting_agent", "updated_at": "now",
    })
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {
            "schema_version": 1, "status": "awaiting_agent", "updated_at": "now",
        },
    }})
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    advance_component_repair(store, "page_001")
    request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
    rejected_plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "host", "repair_round": 1,
        "request_sha256": request_sha256,
        "actions": [_action("absorb_residual", ["candidate_b"])],
    }
    rejected_source = tmp_path / "rejected-host-plan.json"
    rejected_source.write_text(json.dumps(rejected_plan), encoding="utf-8")
    first = record_host_plan(store.root, rejected_source)
    rejected_path = Path(first["plan_path"])
    rejected_payload = rejected_path.read_bytes()
    recorded = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    store.transition_page("page_001", runtime.PageStatus.PROCESSING)
    store.transition_run(runtime.RunStatus.RUNNING)

    monkeypatch.setattr(
        legacy,
        "execute_component_action_round",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RecoverableComponentPlanError(
                "absorb_residual found an unrelated residual region"
            )
        ),
    )
    with ExecutionLease(store.root / "execution.lock", run_root=store.root) as lease:
        summary = runtime._advance_legacy_pages(
            store, manifest, ["page_001"], lease,
        )

    reopened = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    assert summary["status"] == "awaiting_agent"
    assert store.read_json("run_state.json")["status"] == "awaiting_agent"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == (
        "awaiting_agent"
    )
    assert reopened["phase"] == "awaiting_plan"
    assert reopened["repair_round"] == recorded["repair_round"]
    assert reopened["current_round"]["request_ref"] == (
        recorded["current_round"]["request_ref"]
    )
    assert reopened["current_round"]["plan_ref"] is None
    assert reopened["current_round"]["execution_ref"] is None
    assert reopened["current_round"]["quality_ref"] is None
    assert reopened["graph_ref"] == recorded["graph_ref"]
    assert not (
        Path(page_session["reconstruction_dir"]) / "execution-01"
    ).exists()
    assert rejected_path.read_bytes() == rejected_payload

    rejected_sha256 = hashlib.sha256(rejected_payload).hexdigest()
    short_rejected_path = store.root / (
        "host-component-plan-page_001-01-retry-"
        f"{rejected_sha256[:12]}.json"
    )
    short_rejected_path.write_bytes(rejected_payload)
    rejection_path = (
        Path(page_session["reconstruction_dir"])
        / f"component-plan-rejection-{reopened['revision']:08d}.json"
    )
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    rejection["rejected_plan_ref"] = {
        "path": short_rejected_path.relative_to(store.root).as_posix(),
        "sha256": rejected_sha256,
    }
    rejection_path.write_text(json.dumps(rejection), encoding="utf-8")

    import image2editable.host_agent as host_agent

    handshake = host_agent.next_host_agent_item(store.root)
    challenge = host_agent._load_or_create_challenge(store)
    capability_source = tmp_path / "capability-response.json"
    capability_source.write_text(json.dumps({
        "schema_version": 1,
        "kind": "host_capability_response",
        "challenge_id": handshake["challenge_id"],
        "observed": challenge["expected"],
    }), encoding="utf-8")
    record_host_plan(store.root, capability_source)
    retry_request = host_agent.next_host_agent_item(store.root)
    assert retry_request["correction_context"] == {
        "instruction": (
            "The previous plan was rejected because an absorb_residual target had "
            "no containment or 3px adjacency with the signed residual. Modify or "
            "remove the related absorb_residual action; do not change request_sha256."
            ),
            "rejected_plan": rejected_plan,
            "forbidden_action_pairs": [["absorb_residual", "candidate_b"]],
        }
    assert retry_request["request_sha256"] == request_sha256
    assert retry_request["repair_round"] == 1
    assert rejected_path.read_bytes() == rejected_payload
    short_rejected_path.unlink()

    corrected_plan = copy.deepcopy(rejected_plan)
    corrected_plan["actions"] = [_action("discard", ["candidate_b"])]
    corrected_source = tmp_path / "corrected-host-plan.json"
    corrected_source.write_text(json.dumps(corrected_plan), encoding="utf-8")
    real_transition = RunStore.transition_run
    failed = False

    def fail_corrected_transition(current_store, target):
        nonlocal failed
        if target is runtime.RunStatus.PREPARED and not failed:
            failed = True
            raise OSError("corrected transition failure")
        return real_transition(current_store, target)

    monkeypatch.setattr(RunStore, "transition_run", fail_corrected_transition)
    with pytest.raises(OSError, match="corrected transition failure"):
        record_host_plan(store.root, corrected_source)
    retry_paths = list(store.root.glob("host-component-plan-*-retry-*.json"))
    assert len(retry_paths) == 1
    retry_sha256 = hashlib.sha256(retry_paths[0].read_bytes()).hexdigest()
    assert retry_paths[0].name == (
        f"host-component-plan-page_001-01-retry-{retry_sha256[:12]}.json"
    )
    assert request_sha256 not in retry_paths[0].name

    corrected = record_host_plan(store.root, corrected_source)

    assert corrected["plan_path"] != first["plan_path"]
    assert "-retry-" in Path(corrected["plan_path"]).name
    assert corrected["recovered"] is True
    assert Path(corrected["plan_path"]) == retry_paths[0]
    assert rejected_path.read_bytes() == rejected_payload


@pytest.mark.parametrize(
    "error",
    [VisualSegmentationError("ordinary visual failure"), RuntimeError("SAM failure")],
)
def test_nonrecoverable_execution_errors_remain_fatal(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    from image2editable.store import RunStore

    page_session["provider"] = "local"
    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    advance_component_repair(store, "page_001")
    record_local_component_plan(store, "page_001", plan={
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "local", "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("accept", ["candidate_b"])],
    })
    monkeypatch.setattr(
        legacy,
        "execute_component_action_round",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with ExecutionLease(store.root / "execution.lock", run_root=store.root) as lease:
        with pytest.raises(type(error), match=str(error)):
            legacy.advance_legacy_page(store, "page_001", _lease=lease)

    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    assert state["phase"] == "plan_recorded"
    assert state["current_round"]["plan_ref"] is not None


def test_execution_refreshes_candidates_after_discard(
    page_session: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image2editable.store import RunStore

    evidence_root = Path(page_session["reconstruction_dir"]) / "evidence-source"
    graph_path = evidence_root / "component-graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    for component_id, z_index in (("candidate_c", 2), ("candidate_d", 3)):
        candidate = dict(next(
            node for node in graph["nodes"] if node["id"] == "candidate_b"
        ))
        candidate.update({
            "id": component_id, "mask": f"masks/{component_id}.png",
            "z_index": z_index,
        })
        mask_path = evidence_root / candidate["mask"]
        shutil.copyfile(evidence_root / "masks/candidate_b.png", mask_path)
        candidate["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        graph["nodes"].append(candidate)
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    page_session["provider"] = "local"
    _refresh_test_presentation_manifest(page_session)

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=3,
    )
    advance_component_repair(store, "page_001")
    plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "local", "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("discard", ["candidate_b"])],
    }
    record_local_component_plan(store, "page_001", plan=plan)
    output_dir = request_path.parents[2] / "execution-01"
    next_graph = execute_component_actions(
        np.zeros((2, 2, 3), dtype=np.uint8), graph, plan["actions"],
        sam_runner=None, input_dir=request_path.parent, output_dir=output_dir,
    )
    next_graph_path = output_dir / "component-graph.json"
    execution = {
        "schema_version": 1, "page_id": "page_001", "provider": "local",
        "repair_round": 1, "request_sha256": plan["request_sha256"],
        "input_graph_sha256": hashlib.sha256(
            (request_path.parent / "component-graph.json").read_bytes()
        ).hexdigest(),
        "output_graph_sha256": hashlib.sha256(next_graph_path.read_bytes()).hexdigest(),
        "executable_action_count": 1,
        "quality_input_refs": _quality_input_refs(
            output_dir, store, next_graph_path
        ),
    }
    execution_path = output_dir / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    state = record_component_execution(
        store, "page_001", execution_path=execution_path,
        output_graph_path=next_graph_path,
    )

    assert state["candidate_ids"] == ["candidate_c", "candidate_d"]
    assert next(
        node for node in next_graph["nodes"] if node["id"] == "candidate_b"
    )["state"] == "inactive"

    passed = _strict_quality_report("candidate_c", True)["component_reports"][0]
    failed = _strict_quality_report("candidate_d", False)["component_reports"][0]
    monkeypatch.setattr(
        component_repair, "evaluate_component_quality_round",
        lambda *args, **kwargs: _quality_report_with_unexplained({
            "accepted": False,
            "violations": [
                "missing_edge", "pptx_reopen_unknown", "visual_difference",
            ],
            "component_reports": [passed, failed],
            "visual_metrics": {"mae": 30.0, "p95": 60.0, "changed_ratio": 0.2},
            "checks": {"pptx_reopen": "unknown"},
        }, **kwargs),
    )
    record_component_quality(store, "page_001")
    advance_component_repair(store, "page_001")
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    assert sorted(state["frozen"]) == ["candidate_c", "frozen_a"]
    assert state["failed_ids"] == ["candidate_d"]


def test_page_only_background_residual_enters_next_round(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image2editable.store import RunStore

    page_session["provider"] = "local"
    evidence_graph_path = page_session["evidence"]["component-graph.json"]
    evidence_graph = json.loads(evidence_graph_path.read_text(encoding="utf-8"))
    for component_id, point in (("candidate_b", (0, 0)), ("frozen_a", (1, 1))):
        node = next(
            item for item in evidence_graph["nodes"]
            if item["id"] == component_id
        )
        mask = np.zeros((2, 2), dtype=np.uint8)
        mask[point[1], point[0]] = 255
        mask_path = evidence_graph_path.parent / node["mask"]
        Image.fromarray(mask, mode="L").save(mask_path)
        node["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        node["bbox"] = [point[0], point[1], point[0] + 1, point[1] + 1]
    evidence_graph_path.write_text(json.dumps(evidence_graph), encoding="utf-8")
    _refresh_test_presentation_manifest(page_session)
    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    initialized = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    assert initialized["quality_gate_version"] == 2
    advance_component_repair(store, "page_001")
    graph = load_component_agent_graph(request_path)
    frozen_before = dict(next(
        node for node in graph["nodes"] if node["id"] == "frozen_a"
    ))
    plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "local", "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [
            _action("discard", ["candidate_b"]),
            _action(
                "rebuild_background", ["candidate_b"], {"margin_ratio": 0.03}
            ),
        ],
    }
    record_local_component_plan(store, "page_001", plan=plan)
    execution_dir = request_path.parents[2] / "execution-01"
    execute_component_actions(
        np.zeros((2, 2, 3), dtype=np.uint8), graph, plan["actions"],
        sam_runner=None, input_dir=request_path.parent, output_dir=execution_dir,
    )
    graph_path = execution_dir / "component-graph.json"
    execution = {
        "schema_version": 1, "page_id": "page_001", "provider": "local",
        "repair_round": 1, "request_sha256": plan["request_sha256"],
        "input_graph_sha256": hashlib.sha256(
            (request_path.parent / "component-graph.json").read_bytes()
        ).hexdigest(),
        "output_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "executable_action_count": 2,
        "quality_input_refs": _quality_input_refs(
            execution_dir, store, graph_path
        ),
    }
    execution_path = execution_dir / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    state = record_component_execution(
        store, "page_001", execution_path=execution_path,
        output_graph_path=graph_path,
    )
    monkeypatch.setattr(
        component_repair, "evaluate_component_quality_round",
        lambda *args, **kwargs: component_quality.evaluate_page_quality(
            [],
            visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
            page_checks={
                "pptx_reopen": "unknown",
                "editable_text_once": "pass",
                "background_text_clean": "fail",
                "unowned_raster_text": "pass",
            },
            expected_component_ids=[], initial_component_count=2,
            active_visual_count=1,
        ),
    )

    assert state["phase"] == "actions_executed"
    assert state["candidate_ids"] == []
    unexplained_path = execution_dir / "unexplained-mask.png"
    unexplained = np.zeros((2, 2), dtype=np.uint8)
    unexplained[0, 1] = 255
    Image.fromarray(unexplained, mode="L").save(unexplained_path)
    state = record_component_quality(store, "page_001")
    quality = json.loads(
        (store.root / state["current_round"]["quality_ref"]["path"])
        .read_text(encoding="utf-8")
    )
    assert quality["report"]["component_reports"] == []
    assert quality["report"]["accepted"] is False
    assert quality["unexplained_mask_ref"]["sha256"] == hashlib.sha256(
        unexplained_path.read_bytes()
    ).hexdigest()
    after = json.loads(graph_path.read_text(encoding="utf-8"))
    assert next(node for node in after["nodes"] if node["id"] == "frozen_a") == frozen_before
    assert advance_component_repair(store, "page_001")["status"] == "freeze_committed"
    state_after = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    assert state_after["frozen"] == {}
    assert state_after["failed_ids"] == ["frozen_a"]
    next_round = advance_component_repair(store, "page_001")
    assert next_round == {
        "status": "needs_next_round",
        "page_id": "page_001",
        "repair_round": 2,
        "candidate_ids": ["frozen_a"],
        "page_violations": ["background_text_residual"],
    }


def test_page_residual_owner_selects_adjacent_pending_component(
    tmp_path: Path,
) -> None:
    from image2editable.store import RunStore

    store = RunStore(tmp_path / "run")
    graph_root = store.root / "evidence"
    masks_root = graph_root / "masks"
    masks_root.mkdir(parents=True)
    residual = np.zeros((40, 40), dtype=np.uint8)
    residual[10:13, 10:13] = 255
    residual_path = graph_root / "unexplained-mask.png"
    Image.fromarray(residual, mode="L").save(residual_path)
    nodes = []
    for component_id, box in (
        ("adjacent", (5, 10, 8, 13)),
        ("separate", (25, 25, 30, 30)),
    ):
        mask = np.zeros_like(residual)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
        mask_path = masks_root / f"{component_id}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        nodes.append({
            "id": component_id, "kind": "parent", "parent_id": None,
            "state": "pending", "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "bbox": list(box), "z_index": len(nodes), "text_ids": [],
        })
    quality = {"unexplained_mask_ref": {
        "path": residual_path.relative_to(store.root).as_posix(),
        "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
    }}

    owners = component_repair._page_residual_owner_ids(
        store, quality=quality, graph={"nodes": nodes}, graph_root=graph_root,
    )

    assert owners == {"adjacent"}


def test_page_residual_owner_selects_one_actionable_component_per_region(
    tmp_path: Path,
) -> None:
    from image2editable.store import RunStore

    store = RunStore(tmp_path / "run")
    graph_root = store.root / "evidence"
    masks_root = graph_root / "masks"
    masks_root.mkdir(parents=True)
    residual = np.zeros((40, 40), dtype=np.uint8)
    residual[18:22, 18:22] = 255
    residual_path = graph_root / "unexplained-mask.png"
    Image.fromarray(residual, mode="L").save(residual_path)
    nodes = []
    for component_id, box in (
        ("near_small", (14, 18, 17, 22)),
        ("near_large", (12, 16, 17, 24)),
    ):
        mask = np.zeros_like(residual)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
        mask_path = masks_root / f"{component_id}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        nodes.append({
            "id": component_id, "kind": "parent", "parent_id": None,
            "state": "pending", "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "bbox": list(box), "z_index": len(nodes), "text_ids": [],
        })
    quality = {"unexplained_mask_ref": {
        "path": residual_path.relative_to(store.root).as_posix(),
        "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
    }}

    owners = component_repair._page_residual_owner_ids(
        store, quality=quality, graph={"nodes": nodes}, graph_root=graph_root,
    )

    assert owners == {"near_small"}


def test_page_residual_owner_selects_smallest_containing_visual_component(
    tmp_path: Path,
) -> None:
    from image2editable.store import RunStore

    store = RunStore(tmp_path / "run")
    graph_root = store.root / "evidence"
    masks_root = graph_root / "masks"
    masks_root.mkdir(parents=True)
    residual = np.zeros((50, 50), dtype=np.uint8)
    residual[20:24, 20:22] = 255
    residual_path = graph_root / "unexplained-mask.png"
    Image.fromarray(residual, mode="L").save(residual_path)
    nodes = []
    for component_id, box in (
        ("small_container", (10, 10, 40, 40)),
        ("large_container", (2, 2, 48, 48)),
    ):
        mask = np.zeros_like(residual)
        x1, y1, x2, y2 = box
        mask[y1, x1:x2] = 255
        mask[y2 - 1, x1:x2] = 255
        mask[y1:y2, x1] = 255
        mask[y1:y2, x2 - 1] = 255
        mask_path = masks_root / f"{component_id}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        nodes.append({
            "id": component_id, "kind": "parent", "parent_id": None,
            "state": "frozen", "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "bbox": list(box), "z_index": len(nodes), "text_ids": [],
        })
    quality = {"unexplained_mask_ref": {
        "path": residual_path.relative_to(store.root).as_posix(),
        "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
    }}

    owners = component_repair._page_residual_owner_ids(
        store, quality=quality, graph={"nodes": nodes}, graph_root=graph_root,
    )

    assert owners == {"small_container"}


def test_small_page_residual_selects_adjacent_large_frozen_component(
    tmp_path: Path,
) -> None:
    from image2editable.store import RunStore

    store = RunStore(tmp_path / "run")
    graph_root = store.root / "evidence"
    masks_root = graph_root / "masks"
    masks_root.mkdir(parents=True)
    residual = np.zeros((600, 600), dtype=np.uint8)
    residual[200:206, 96:99] = 255
    residual_path = graph_root / "unexplained-mask.png"
    Image.fromarray(residual, mode="L").save(residual_path)
    nodes = []
    for component_id, state, box in (
        ("large_panel", "frozen", (100, 100, 500, 500)),
        ("unrelated", "pending", (520, 520, 540, 540)),
    ):
        mask = np.zeros_like(residual)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
        mask_path = masks_root / f"{component_id}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        nodes.append({
            "id": component_id, "kind": "parent", "parent_id": None,
            "state": state, "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "bbox": list(box), "z_index": len(nodes), "text_ids": [],
        })
    quality = {"unexplained_mask_ref": {
        "path": residual_path.relative_to(store.root).as_posix(),
        "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
    }}

    owners = component_repair._page_residual_owner_ids(
        store, quality=quality, graph={"nodes": nodes}, graph_root=graph_root,
    )

    assert owners == {"large_panel"}


@pytest.mark.parametrize(
    ("component_accepted", "residual_owner", "expected_failed"),
    [
        (True, "candidate_b", ["candidate_b"]),
        (False, "frozen_a", ["candidate_b", "frozen_a"]),
    ],
)
def test_page_residual_owner_is_reopened_with_component_failures(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
    component_accepted: bool,
    residual_owner: str,
    expected_failed: list[str],
) -> None:
    from image2editable.store import RunStore

    page_session["provider"] = "local"
    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    state = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    quality = _strict_quality_report("candidate_b", component_accepted)
    quality["violations"] = [
        "pptx_reopen_unknown", "unexplained_visual_residual",
    ]
    quality_path = store.root / "quality.json"
    quality_path.write_text(
        json.dumps({
            "report": quality,
            "contained_parent_pairs": [],
            "approved_contained_parent_pairs": [],
        }),
        encoding="utf-8",
    )
    quality_ref = {
        "path": quality_path.relative_to(store.root).as_posix(),
        "sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
    }
    state["current_round"].update({
        "plan_ref": quality_ref,
        "execution_ref": quality_ref,
        "quality_ref": quality_ref,
    })
    state["phase"] = "quality_recorded"
    state["plan_count"] = 1
    state["round_history"] = [{
        "round": 1,
        "plan_sha256": quality_ref["sha256"],
        "normalized_plan_sha256": quality_ref["sha256"],
        "execution_sha256": quality_ref["sha256"],
        "quality_sha256": None,
        "frozen_ids": [],
        "failed_ids": [],
    }]
    monkeypatch.setattr(
        component_repair, "_page_residual_owner_ids",
        lambda *args, **kwargs: {residual_owner},
    )

    result = component_repair._commit_component_freeze(
        store, state, "page_001"
    )

    assert result["failed_ids"] == expected_failed
    updated = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    assert "candidate_b" not in updated["frozen"]
    graph = json.loads(
        (store.root / updated["graph_ref"]["path"]).read_text(encoding="utf-8")
    )
    assert next(
        node for node in graph["nodes"] if node["id"] == "candidate_b"
    )["state"] == "pending"


def test_record_suppress_text_preserves_linked_frozen_visual_assets(
    page_session: dict,
) -> None:
    from image2editable.store import RunStore

    graph_path = page_session["evidence"]["component-graph.json"]
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    frozen_visual = next(
        node for node in graph["nodes"] if node["id"] == "frozen_a"
    )
    frozen_visual["text_ids"] = ["text_0001"]
    text = _node("text_0001", "frozen", 2)
    text["kind"] = "text"
    graph["nodes"].append(text)
    text_mask = graph_path.parent / text["mask"]
    Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(text_mask)
    text["mask_sha256"] = hashlib.sha256(text_mask.read_bytes()).hexdigest()
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    _refresh_test_presentation_manifest(page_session)
    page_session["provider"] = "local"
    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1,
        "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    advance_component_repair(store, "page_001")
    plan = {
        "schema_version": 1, "kind": "component_plan",
        "page_id": "page_001", "provider": "local", "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("suppress_text", ["text_0001"])],
    }
    record_local_component_plan(store, "page_001", plan=plan)
    output_dir = request_path.parents[2] / "execution-01"
    next_graph = execute_component_actions(
        np.zeros((2, 2, 3), dtype=np.uint8), graph, plan["actions"],
        sam_runner=lambda **_: np.ones((2, 2), dtype=np.uint8),
        input_dir=request_path.parent,
        output_dir=output_dir,
    )
    next_graph_path = output_dir / "component-graph.json"
    refs = _quality_input_refs(output_dir, store, next_graph_path)
    execution = {
        "schema_version": 1, "page_id": "page_001", "provider": "local",
        "repair_round": 1, "request_sha256": plan["request_sha256"],
        "input_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "output_graph_sha256": hashlib.sha256(next_graph_path.read_bytes()).hexdigest(),
        "executable_action_count": 1, "quality_input_refs": refs,
    }
    execution_path = output_dir / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    state = record_component_execution(
        store, "page_001", execution_path=execution_path,
        output_graph_path=next_graph_path,
    )

    assert state["phase"] == "actions_executed"
    before_manifest = json.loads(
        (request_path.parent / "presentation-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    after_manifest = json.loads(
        (store.root / refs["presentation_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    before_frozen = next(
        item for item in before_manifest["components"]
        if item["component_id"] == "frozen_a"
    )
    after_frozen = next(
        item for item in after_manifest["components"]
        if item["component_id"] == "frozen_a"
    )
    for field in (
        "rgba", "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask",
    ):
        assert after_frozen[field]["sha256"] == before_frozen[field]["sha256"]
    assert next(
        node for node in next_graph["nodes"] if node["id"] == "frozen_a"
    )["text_ids"] == []


def test_execution_quality_consumes_exact_presentation_underlay_and_freezes(
    page_session: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from image2editable.host_agent import record_host_plan
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {
        "schema_version": 1, "status": "awaiting_agent", "updated_at": "now"
    })
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "awaiting_agent", "updated_at": "now"}
    }})
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    advance_component_repair(store, "page_001")
    request = load_component_agent_request(request_path)
    plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "host", "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("accept", ["candidate_b"])],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    record_host_plan(store.root, plan_path)
    assert advance_component_repair(store, "page_001")["status"] == "needs_execution"

    graph = json.loads((request_path.parent / "component-graph.json").read_text(encoding="utf-8"))
    next_graph = json.loads(json.dumps(graph))
    next(node for node in next_graph["nodes"] if node["id"] == "candidate_b")["state"] = "pending_gate"
    execution_dir = request_path.parents[2] / "execution-01"
    execution_dir.mkdir()
    shutil.copytree(request_path.parent / "masks", execution_dir / "masks")
    graph_path = execution_dir / "component-graph.json"
    graph_path.write_text(json.dumps(next_graph), encoding="utf-8")
    quality_input_refs = _quality_input_refs(execution_dir, store, graph_path)
    missing_evidence_refs = dict(quality_input_refs)
    missing_evidence_refs.pop("foreground_evidence")
    missing_evidence_execution = {
        "schema_version": 1, "page_id": "page_001", "provider": "host",
        "repair_round": 1, "request_sha256": plan["request_sha256"],
        "input_graph_sha256": request["graph_sha256"],
        "output_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "executable_action_count": 1,
        "quality_input_refs": missing_evidence_refs,
    }
    missing_evidence_path = execution_dir / "missing-evidence-execution.json"
    missing_evidence_path.write_text(
        json.dumps(missing_evidence_execution), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="foreground_evidence"):
        record_component_execution(
            store,
            "page_001",
            execution_path=missing_evidence_path,
            output_graph_path=graph_path,
        )
    foreground_evidence = execution_dir / "foreground-evidence-mask.png"
    Image.new("L", (2, 2), 255).save(foreground_evidence)
    quality_input_refs["foreground_evidence"] = {
        "path": foreground_evidence.relative_to(store.root).as_posix(),
        "sha256": hashlib.sha256(foreground_evidence.read_bytes()).hexdigest(),
    }
    background_responsibility = execution_dir / "background-responsibility.png"
    Image.fromarray(np.array([[0, 0], [0, 255]], dtype=np.uint8)).save(
        background_responsibility
    )
    quality_input_refs["background_responsibility"] = {
        "path": background_responsibility.relative_to(store.root).as_posix(),
        "sha256": hashlib.sha256(
            background_responsibility.read_bytes()
        ).hexdigest(),
    }
    manifest_path = store.root / quality_input_refs["presentation_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["components"][0]
    exact_masks = {
        "ownership_mask": np.array([[255, 0], [0, 0]], dtype=np.uint8),
        "presentation_alpha_mask": np.array([[255, 255], [0, 0]], dtype=np.uint8),
        "generated_underlay_mask": np.array([[0, 255], [0, 0]], dtype=np.uint8),
    }
    for name, array in exact_masks.items():
        path = store.root / first[name]["path"]
        Image.fromarray(array).save(path)
        first[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rgba_path = store.root / first["rgba"]["path"]
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[:, :, 3] = exact_masks["presentation_alpha_mask"]
    Image.fromarray(rgba, mode="RGBA").save(rgba_path)
    first["rgba"]["sha256"] = hashlib.sha256(rgba_path.read_bytes()).hexdigest()
    first["metrics"] = {
        "boundary_color_mae": 1.25,
        "gradient_jump_p95": 2.5,
        "added_high_frequency_pixels": 0.0,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    quality_input_refs["presentation_manifest"]["sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    execution = {
        "schema_version": 1, "page_id": "page_001", "provider": "host",
        "repair_round": 1, "request_sha256": plan["request_sha256"],
        "input_graph_sha256": request["graph_sha256"],
        "output_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "executable_action_count": 1,
        "quality_input_refs": quality_input_refs,
    }
    execution_path = execution_dir / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    record_component_execution(
        store, "page_001", execution_path=execution_path, output_graph_path=graph_path,
    )
    assert advance_component_repair(store, "page_001")["status"] == "needs_quality"

    observed_visual = {}
    observed_layers = []
    observed_foreground = {}
    observed_responsibility = {}
    import scripts.visual_segment as visual_segment
    real_visual_difference = visual_segment.visual_difference

    def mutate_manifest_after_quality_inputs_are_bound(*args, **kwargs):
        manifest_path.write_text("{", encoding="utf-8")
        return real_visual_difference(*args, **kwargs)

    monkeypatch.setattr(
        visual_segment,
        "visual_difference",
        mutate_manifest_after_quality_inputs_are_bound,
    )

    def quality_evaluator(*args, **kwargs):
        observed_visual.update(kwargs["visual_metrics"])
        observed_layers.extend(kwargs["presentation_layers"])
        observed_foreground["mask"] = kwargs["material_foreground"]
        observed_foreground["output"] = kwargs["unexplained_output_path"]
        observed_responsibility["mask"] = kwargs[
            "background_responsibility"
        ]
        return _quality_report_with_unexplained(
            _strict_quality_report("candidate_b", True), **kwargs
        )

    monkeypatch.setattr(
        component_repair, "evaluate_component_quality_round", quality_evaluator,
    )
    record_component_quality(store, "page_001")
    assert observed_visual["mae"] > 0
    assert [layer["component_id"] for layer in observed_layers] == [
        "candidate_b", "frozen_a",
    ]
    assert np.array_equal(
        observed_layers[0]["ownership_mask"], exact_masks["ownership_mask"] > 0
    )
    assert np.array_equal(
        observed_layers[0]["presentation_alpha_mask"],
        exact_masks["presentation_alpha_mask"] > 0,
    )
    assert np.array_equal(
        observed_layers[0]["generated_underlay_mask"],
        exact_masks["generated_underlay_mask"] > 0,
    )
    assert observed_layers[0]["metrics"] == first["metrics"]
    assert np.all(observed_foreground["mask"] == 255)
    assert np.array_equal(
        observed_responsibility["mask"],
        np.array([[False, False], [False, True]]),
    )
    assert observed_foreground["output"] == execution_dir / "unexplained-mask.png"
    quality_state = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    quality_artifact = json.loads(
        (store.root / quality_state["current_round"]["quality_ref"]["path"])
        .read_text(encoding="utf-8")
    )
    assert set(quality_artifact["input_refs"]) == {
        "source", "background", "reconstructed", "text_mask", "native_check",
        "presentation_manifest", "foreground_evidence",
        "background_responsibility",
    }
    assert quality_artifact["contained_parent_pairs"] == []
    assert advance_component_repair(store, "page_001")["status"] == "freeze_committed"
    ready = advance_component_repair(store, "page_001")
    assert ready["status"] == "ready_for_assembly"
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    assert state["frozen"]["candidate_b"] == state["parent_assets"]["candidate_b"]["sha256"]
    assert state["delivery_checks"] == {"pptx_reopen": "unknown"}
    assert state["result_ref"] is not None
    result = store.read_json("pages/page_001/reconstruction/component_result.json")
    assert set(result["accepted_asset_refs"]) == {
        "source", "background", "reconstructed", "text_mask", "native_check",
        "presentation_manifest", "foreground_evidence",
        "background_responsibility",
    }


def test_page_only_violation_stops_when_quality_does_not_improve(
    page_session: dict,
) -> None:
    from image2editable.store import RunStore

    first = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(first.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {"schema_version": 1, "status": "prepared"})
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "processing"}
    }})
    state = initialize_component_repair_state(
        store, "page_001", request_path=first, initial_component_count=2,
    )

    def bind_synthetic_quality(current: dict) -> dict:
        request_path = (
            store.root / current["current_round"]["request_ref"]["path"]
        )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        manifest_record = request["evidence"]["presentation-manifest.json"]
        manifest_path = request_path.parent / manifest_record["path"]
        quality_path = Path(page_session["reconstruction_dir"]) / (
            f"synthetic-quality-{current['repair_round']:02d}.json"
        )
        strict_report = _strict_quality_report("candidate_b", False)
        strict_report["checks"]["background_text_clean"] = "fail"
        quality_path.write_text(json.dumps({
            "schema_version": 1,
            "page_id": "page_001",
            "provider": "host",
            "repair_round": current["repair_round"],
            "request_sha256": current["current_round"]["request_ref"]["sha256"],
            "input_graph_sha256": current["graph_ref"]["sha256"],
            "quality_gate_version": current["quality_gate_version"],
            "expected_component_ids": ["candidate_b"],
            "initial_component_count": 2,
            "initial_diagnostics": [],
            "contained_parent_pairs": [],
            "approved_contained_parent_pairs": [],
            "input_refs": {
                **{
                    name: current["current_round"]["request_ref"]
                    for name in (
                        "background", "reconstructed", "text_mask",
                        "native_check",
                    )
                },
                "presentation_manifest": {
                    "path": manifest_path.relative_to(store.root).as_posix(),
                    "sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                },
                "source": {
                    "path": current["current_round"]["request_ref"]["path"],
                    "sha256": current["source_sha256"],
                },
            },
            "report": component_quality.evaluate_page_quality(
                strict_report["component_reports"],
                visual_metrics=strict_report["visual_metrics"],
                page_checks=strict_report["checks"],
                expected_component_ids=["candidate_b"],
                initial_component_count=2,
                active_visual_count=2,
            ),
        }), encoding="utf-8")
        page_session["evidence"]["quality-report.json"] = quality_path
        return {
            "path": quality_path.relative_to(store.root).as_posix(),
            "sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
        }

    graph = json.loads(
        page_session["evidence"]["component-graph.json"].read_text(
            encoding="utf-8"
        )
    )
    candidate = next(
        node for node in graph["nodes"] if node["id"] == "candidate_b"
    )
    candidate["state"] = "frozen"
    page_session["evidence"]["component-graph.json"].write_text(
        json.dumps(graph), encoding="utf-8"
    )
    _refresh_test_presentation_manifest(page_session)
    state["phase"] = "freeze_committed"
    state["candidate_ids"] = state["failed_ids"] = []
    state["frozen"] = {
        "candidate_b": state["parent_assets"]["candidate_b"]["sha256"],
        "frozen_a": next(
            node["mask_sha256"] for node in graph["nodes"]
            if node["id"] == "frozen_a"
        ),
    }
    state["current_round"]["plan_ref"] = state["current_round"]["request_ref"]
    state["current_round"]["execution_ref"] = state["current_round"]["request_ref"]
    state["current_round"]["quality_ref"] = bind_synthetic_quality(state)
    store.write_json("pages/page_001/reconstruction/component_state.json", state)
    request_path = build_component_agent_request(page_session, repair_round=2)
    updated = record_next_component_request(
        store, "page_001", request_path=request_path
    )
    assert updated["repair_round"] == 2
    assert updated["candidate_ids"] == []
    updated["phase"] = "freeze_committed"
    updated["current_round"]["plan_ref"] = updated["current_round"]["request_ref"]
    updated["current_round"]["execution_ref"] = updated["current_round"]["request_ref"]
    updated["current_round"]["quality_ref"] = bind_synthetic_quality(updated)
    store.write_json("pages/page_001/reconstruction/component_state.json", updated)

    third_request = build_component_agent_request(page_session, repair_round=3)
    with pytest.raises(RuntimeError, match="quality did not improve"):
        record_next_component_request(
            store, "page_001", request_path=third_request
        )
    assert advance_component_repair(store, "page_001") == {
        "status": "fallback_required",
        "page_id": "page_001",
        "repair_round": 2,
        "stop_reason": "no_quality_improvement",
    }
    assert not (first.parent.parent / "round-04").exists()


@pytest.mark.parametrize(
    ("candidate_c_accepted", "expected_frozen", "expected_failed"),
    [
        (False, ["candidate_b"], ["candidate_c"]),
        (True, ["candidate_b", "candidate_c"], []),
    ],
)
def test_unowned_raster_text_page_violation_preserves_leaf_component_freeze(
    page_session: dict,
    candidate_c_accepted: bool,
    expected_frozen: list[str],
    expected_failed: list[str],
) -> None:
    from image2editable.store import RunStore

    graph_path = page_session["evidence"]["component-graph.json"]
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    candidate_c = _node("candidate_c", "pending", 2)
    mask_path = graph_path.parent / "masks/candidate_c.png"
    Image.fromarray(np.full((2, 2), 253, dtype=np.uint8)).save(mask_path)
    candidate_c["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    graph["nodes"].append(candidate_c)
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    _refresh_test_presentation_manifest(page_session)
    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    quality = {
        "report": {
            **_strict_quality_report("candidate_b", True),
            "accepted": False,
            "violations": ["unowned_raster_text"],
            "component_reports": [
                _strict_quality_report("candidate_b", True)["component_reports"][0],
                _strict_quality_report(
                    "candidate_c", candidate_c_accepted
                )["component_reports"][0],
            ],
        }
    }
    quality_path = store.root / "sticky-quality.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    state["phase"] = "quality_recorded"
    state["plan_count"] = 1
    state["current_round"]["plan_ref"] = state["current_round"]["request_ref"]
    state["current_round"]["execution_ref"] = state["current_round"]["request_ref"]
    state["current_round"]["quality_ref"] = {
        "path": "sticky-quality.json",
        "sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
    }
    state["round_history"] = [{
        "round": 1,
        "plan_sha256": None,
        "normalized_plan_sha256": None,
        "execution_sha256": None,
        "quality_sha256": None,
        "frozen_ids": [],
        "failed_ids": [],
    }]
    store.write_json("pages/page_001/reconstruction/component_state.json", state)

    outcome = component_repair._commit_component_freeze(
        store, state, "page_001"
    )

    assert outcome["frozen_ids"] == expected_frozen
    assert outcome["failed_ids"] == expected_failed
    if not expected_failed:
        preserved = advance_component_repair(store, "page_001")
        assert preserved["status"] == "preserved_with_warning"
        state = store.read_json(
            "pages/page_001/reconstruction/component_state.json"
        )
        assert state["repair_round"] == 1
        assert state["stop_reason"] == "unowned_raster_text"


def _initial_unowned_diagnostic(source_sha256: str, *, text: str = "ny") -> dict:
    return {
        "kind": "unowned_raster_text",
        "source_sha256": source_sha256,
        "candidate_id": "candidate_0001_01",
        "bbox": [0, 0, 2, 2],
        "views": [
            {"normalized_text": "nx", "confidence": 0.96},
            {"normalized_text": text, "confidence": 0.95},
        ],
    }


def _failed_diagnostic_round_one(page_session: dict, monkeypatch):
    from image2editable.store import RunStore

    page_session["provider"] = "local"
    source_sha256 = hashlib.sha256(
        page_session["evidence"]["source.png"].read_bytes()
    ).hexdigest()
    diagnostics = [_initial_unowned_diagnostic(source_sha256)]
    page_session["evidence"]["quality-report.json"].write_text(json.dumps({
        "schema_version": 1,
        "phase": "initial_layers",
        "text_items": [],
        "initial_diagnostics": diagnostics,
        "violations": ["unowned_raster_text"],
    }), encoding="utf-8")
    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=1,
    )
    advance_component_repair(store, "page_001")
    failed = _strict_quality_report("candidate_b", False)
    failed["violations"] = sorted({
        *failed["violations"], "unowned_raster_text",
    })
    failed["checks"]["unowned_raster_text"] = "fail"
    monkeypatch.setattr(
        component_repair, "evaluate_component_quality_round",
        lambda *args, **kwargs: _quality_report_with_unexplained(failed, **kwargs),
    )
    graph = load_component_agent_graph(request_path)
    _, freeze = _execute_composite_quality_round(
        store, request_path, graph,
        action=_action("accept", ["candidate_b"]), shape=(2, 2),
        initial_diagnostics=diagnostics,
    )
    assert freeze["status"] == "freeze_committed"
    assert freeze["failed_ids"] == ["candidate_b"]
    assert advance_component_repair(store, "page_001")["status"] == "needs_next_round"
    return store, diagnostics


def _next_round_session(page_session: dict, store, quality: dict) -> dict:
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    evidence = dict(page_session["evidence"])
    evidence["component-graph.json"] = store.root / state["graph_ref"]["path"]
    quality_path = Path(page_session["reconstruction_dir"]) / "round-02-quality.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    evidence["quality-report.json"] = quality_path
    session = {**page_session, "provider": "local", "evidence": evidence}
    _refresh_test_presentation_manifest(session)
    return session


def _failed_underlay_round_one(page_session: dict):
    from image2editable.store import RunStore

    page_session["provider"] = "local"
    first_request = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(first_request.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=first_request, initial_component_count=2,
    )
    advance_component_repair(store, "page_001")
    _, first_freeze = _execute_composite_quality_round(
        store,
        first_request,
        load_component_agent_graph(first_request),
        action=_action("accept", ["candidate_b"]),
        shape=(2, 2),
        presentation_metrics_by_id={"candidate_b": {
            "boundary_color_mae": 1000.0,
            "gradient_jump_p95": 0.0,
            "added_high_frequency_pixels": 0.0,
        }},
    )
    assert first_freeze["failed_ids"] == ["candidate_b"]
    assert advance_component_repair(store, "page_001")["status"] == "needs_next_round"
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    return store, store.root / state["current_round"]["quality_ref"]["path"]


def _real_next_round_session(page_session: dict, store, quality_path: Path) -> dict:
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    evidence = dict(page_session["evidence"])
    evidence["component-graph.json"] = store.root / state["graph_ref"]["path"]
    evidence["quality-report.json"] = quality_path
    session = {**page_session, "provider": "local", "evidence": evidence}
    _refresh_test_presentation_manifest(session)
    return session


def test_next_legacy_request_reuses_execution_lease(
    page_session: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _failed_diagnostic_round_one(page_session, monkeypatch)
    state = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    quality_ref = state["current_round"]["quality_ref"]
    quality_path = store.root / quality_ref["path"]
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    native_ref = quality["input_refs"]["native_check"]
    native_path = store.root / native_ref["path"]
    native_check = json.loads(native_path.read_text(encoding="utf-8"))
    native_check["text_items"] = []
    native_path.write_text(json.dumps(native_check), encoding="utf-8")
    native_ref["sha256"] = hashlib.sha256(native_path.read_bytes()).hexdigest()
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    quality_ref["sha256"] = hashlib.sha256(quality_path.read_bytes()).hexdigest()
    store.write_json(
        "pages/page_001/reconstruction/component_state.json", state,
    )
    received_leases = []
    real_build_request = legacy.build_component_agent_request

    def build_request(
        session: dict,
        *,
        repair_round: int,
        _lease: ExecutionLease | None = None,
    ) -> Path:
        received_leases.append(_lease)
        if _lease is None:
            return real_build_request(session, repair_round=repair_round)
        return real_build_request(
            session, repair_round=repair_round, _lease=_lease,
        )

    monkeypatch.setattr(legacy, "build_component_agent_request", build_request)
    with ExecutionLease(
        store.root / "execution.lock", run_root=store.root,
    ) as lease:
        legacy._publish_next_legacy_request(store, "page_001", 2, lease)

    assert received_leases == [lease]


@pytest.mark.parametrize(
    "field",
    [
        "rgba", "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask",
    ],
)
def test_next_round_rejects_replaced_frozen_presentation_asset(
    page_session: dict, field: str,
) -> None:
    store, quality_path = _failed_underlay_round_one(page_session)
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    graph_path = store.root / state["graph_ref"]["path"]
    replacement_dir = Path(page_session["reconstruction_dir"]) / (
        f"round-02-frozen-{field}"
    )
    replacement_dir.mkdir()
    manifest_path = _write_test_presentation_manifest(
        store.root,
        replacement_dir,
        source_sha256=state["source_sha256"],
        graph_path=graph_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = next(
        item for item in manifest["components"]
        if item["component_id"] == "frozen_a"
    )
    asset_path = store.root / frozen[field]["path"]
    asset_path.write_bytes(asset_path.read_bytes() + b"round-two-replacement")
    frozen[field]["sha256"] = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    session = _real_next_round_session(page_session, store, quality_path)
    session["evidence"]["presentation-manifest.json"] = manifest_path
    request_path = build_component_agent_request(session, repair_round=2)

    with pytest.raises(ValueError, match="frozen presentation"):
        record_next_component_request(
            store, "page_001", request_path=request_path,
        )


def test_real_round_two_uses_bound_previous_quality_for_improvement(
    page_session: dict,
) -> None:
    store, quality_path = _failed_underlay_round_one(page_session)
    session = _real_next_round_session(page_session, store, quality_path)
    second_request = build_component_agent_request(session, repair_round=2)
    record_next_component_request(store, "page_001", request_path=second_request)
    advance_component_repair(store, "page_001")

    second_report, _ = _execute_composite_quality_round(
        store,
        second_request,
        load_component_agent_graph(second_request),
        action=_action("accept", ["candidate_b"]),
        shape=(2, 2),
    )

    candidate = second_report["component_reports"][0]
    assert candidate["improvement"]["underlay_boundary_color_mae"] == 1000.0


def test_next_request_must_reference_the_state_quality_artifact(
    page_session: dict,
) -> None:
    store, quality_path = _failed_underlay_round_one(page_session)
    copied_quality = Path(page_session["reconstruction_dir"]) / "copied-quality.json"
    copied_quality.write_bytes(quality_path.read_bytes() + b"\n")
    session = _real_next_round_session(page_session, store, copied_quality)
    second_request = build_component_agent_request(session, repair_round=2)

    with pytest.raises(ValueError, match="previous quality"):
        record_next_component_request(
            store, "page_001", request_path=second_request,
        )


@pytest.mark.parametrize(
    "mutation", ["identity", "report", "input_ref", "unexplained_ref", "pairs"],
)
def test_previous_quality_artifact_identity_is_strict(
    page_session: dict,
    mutation: str,
) -> None:
    store, quality_path = _failed_underlay_round_one(page_session)
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    session = _real_next_round_session(page_session, store, quality_path)
    request_path = build_component_agent_request(session, repair_round=2)
    request = load_component_agent_request(request_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if mutation == "identity":
        quality["page_id"] = "wrong_page"
    elif mutation == "report":
        quality["report"] = {}
    elif mutation == "input_ref":
        quality["input_refs"]["background"]["path"] = 1
    elif mutation == "unexplained_ref":
        quality["unexplained_mask_ref"]["path"] = 1
    else:
        quality["contained_parent_pairs"] = [["candidate_b"]]

    with pytest.raises(
        ValueError,
        match="previous component quality artifact|component quality report",
    ):
        component_repair._previous_component_reports(
            quality,
            state={**state, "repair_round": 2},
            request=request,
            active_component_ids=["candidate_b", "frozen_a"],
        )


def test_previous_quality_allows_failed_candidate_to_be_replaced(
    page_session: dict,
) -> None:
    store, quality_path = _failed_underlay_round_one(page_session)
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    session = _real_next_round_session(page_session, store, quality_path)
    request_path = build_component_agent_request(session, repair_round=2)
    request = load_component_agent_request(request_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    reports = component_repair._previous_component_reports(
        quality,
        state={**state, "repair_round": 2},
        request=request,
        active_component_ids=["frozen_a"],
    )

    assert "candidate_b" in reports


def test_previous_quality_allows_bound_background_responsibility(
    page_session: dict,
) -> None:
    store, quality_path = _failed_underlay_round_one(page_session)
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    session = _real_next_round_session(page_session, store, quality_path)
    request_path = build_component_agent_request(session, repair_round=2)
    request = load_component_agent_request(request_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["input_refs"]["background_responsibility"] = dict(
        quality["input_refs"]["foreground_evidence"]
    )

    reports = component_repair._previous_component_reports(
        quality,
        state={**state, "repair_round": 2},
        request=request,
        active_component_ids=["candidate_b", "frozen_a"],
    )

    assert "candidate_b" in reports


def test_previous_quality_allows_prior_failed_candidate_to_be_merged(
    page_session: dict,
) -> None:
    store, quality_path = _failed_underlay_round_one(page_session)
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    session = _real_next_round_session(page_session, store, quality_path)
    request_path = build_component_agent_request(session, repair_round=2)
    request = load_component_agent_request(request_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["expected_component_ids"] = ["frozen_a"]
    quality["report"] = _strict_quality_report("frozen_a", True)
    state["failed_ids"] = ["merge_0001"]

    reports = component_repair._previous_component_reports(
        quality,
        state={**state, "repair_round": 2},
        request=request,
        active_component_ids=["frozen_a", "merge_0001"],
    )

    assert reports["frozen_a"]["accepted"] is True


def test_previous_quality_identity_allows_unapproved_pair_reactivation(
    page_session: dict,
) -> None:
    store, quality_path = _failed_underlay_round_one(page_session)
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    session = _real_next_round_session(page_session, store, quality_path)
    request_path = build_component_agent_request(session, repair_round=2)
    request = load_component_agent_request(request_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["contained_parent_pairs"] = [["candidate_b", "frozen_a"]]
    quality["approved_contained_parent_pairs"] = []
    request["candidate_ids"] = ["candidate_b", "frozen_a"]
    request["frozen_ids"] = []

    reports = component_repair._previous_component_reports(
        quality,
        state={**state, "repair_round": 2},
        request=request,
        active_component_ids=["candidate_b", "frozen_a"],
    )

    assert "candidate_b" in reports


def test_initial_diagnostics_continue_through_real_round_two(
    page_session: dict,
    monkeypatch,
) -> None:
    store, diagnostics = _failed_diagnostic_round_one(page_session, monkeypatch)
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    quality_path = store.root / state["current_round"]["quality_ref"]["path"]
    session = _real_next_round_session(page_session, store, quality_path)
    request_path = build_component_agent_request(session, repair_round=2)

    state = record_next_component_request(
        store, "page_001", request_path=request_path
    )
    request = load_component_agent_request(request_path)
    quality_path = request_path.parent / request["evidence"]["quality-report.json"]["path"]
    assert json.loads(quality_path.read_text(encoding="utf-8"))[
        "initial_diagnostics"
    ] == diagnostics
    assert state["repair_round"] == 2
    advance_component_repair(store, "page_001")
    graph = load_component_agent_graph(request_path)
    _, freeze = _execute_composite_quality_round(
        store, request_path, graph,
        action=_action("accept", ["candidate_b"]), shape=(2, 2),
        initial_diagnostics=diagnostics,
    )
    assert freeze["status"] == "freeze_committed"


@pytest.mark.parametrize("replacement", [[], "replacement"])
def test_next_round_rejects_deleted_or_replaced_initial_diagnostics(
    page_session: dict,
    monkeypatch,
    replacement: object,
) -> None:
    store, diagnostics = _failed_diagnostic_round_one(page_session, monkeypatch)
    next_diagnostics = (
        [_initial_unowned_diagnostic(diagnostics[0]["source_sha256"], text="nz")]
        if replacement == "replacement" else []
    )
    session = _next_round_session(page_session, store, {
        "schema_version": 1,
        "initial_diagnostics": next_diagnostics,
        "violations": ["unowned_raster_text"] if next_diagnostics else [],
    })
    request_path = build_component_agent_request(session, repair_round=2)

    with pytest.raises(ValueError, match="diagnostic|previous quality"):
        record_next_component_request(store, "page_001", request_path=request_path)


@pytest.mark.parametrize("replacement", [[], "replacement"])
def test_round_two_execution_rejects_deleted_or_replaced_native_diagnostics(
    page_session: dict,
    monkeypatch,
    replacement: object,
) -> None:
    store, diagnostics = _failed_diagnostic_round_one(page_session, monkeypatch)
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    quality_path = store.root / state["current_round"]["quality_ref"]["path"]
    session = _real_next_round_session(page_session, store, quality_path)
    request_path = build_component_agent_request(session, repair_round=2)
    record_next_component_request(store, "page_001", request_path=request_path)
    advance_component_repair(store, "page_001")
    native_diagnostics = (
        [_initial_unowned_diagnostic(diagnostics[0]["source_sha256"], text="nz")]
        if replacement == "replacement" else []
    )

    with pytest.raises(ValueError, match="diagnostic"):
        _execute_composite_quality_round(
            store, request_path, load_component_agent_graph(request_path),
            action=_action("accept", ["candidate_b"]), shape=(2, 2),
            initial_diagnostics=native_diagnostics,
        )


def test_agent_request_rejects_tampered_component_mask(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    graph_path = request_path.parent / request["evidence"]["component-graph.json"]["path"]
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    mask_path = request_path.parent / graph["nodes"][0]["mask"]
    mask_path.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="mask hash mismatch"):
        load_component_agent_request(request_path)


def test_quality_recorder_rejects_external_self_report(page_session: dict) -> None:
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    with pytest.raises(TypeError, match="quality_path"):
        record_component_quality(
            store, "page_001", quality_path=request_path.parent / "quality-report.json"
        )


def test_fallback_state_requires_dedicated_refs(page_session: dict) -> None:
    from image2editable.component_contracts import validate_component_repair_state
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    state["phase"] = "fallback_executed"
    state["stop_reason"] = "round_limit"
    state["fallback"] = {"status": "parent_pending", "parent_ids": ["candidate_b"]}

    with pytest.raises(ValueError, match="fallback execution references"):
        validate_component_repair_state(state)


def test_same_normalized_plan_twice_stops_before_execution(page_session: dict) -> None:
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {"schema_version": 1, "status": "prepared"})
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "processing"}
    }})
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "host", "repair_round": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [_action("accept", ["candidate_b"])],
    }
    plan_path = store.root / "same-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    state["phase"] = "plan_recorded"
    state["plan_count"] = 1
    state["current_round"]["plan_ref"] = {
        "path": "same-plan.json",
        "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    }
    state["last_normalized_plan_sha256"] = component_repair._normalized_plan_sha256(plan)
    store.write_json("pages/page_001/reconstruction/component_state.json", state)

    outcome = advance_component_repair(store, "page_001")

    assert outcome["status"] == "fallback_required"
    assert outcome["stop_reason"] == "repeated_plan"
    assert not (request_path.parents[1] / "execution-01").exists()


def test_zero_executable_actions_stops_without_quality(page_session: dict) -> None:
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {"schema_version": 1, "status": "prepared"})
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "processing"}
    }})
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    execution = {"executable_action_count": 0}
    execution_path = store.root / "execution-zero.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    state["phase"] = "actions_executed"
    state["current_round"]["plan_ref"] = state["current_round"]["request_ref"]
    state["current_round"]["execution_ref"] = {
        "path": "execution-zero.json",
        "sha256": hashlib.sha256(execution_path.read_bytes()).hexdigest(),
    }
    store.write_json("pages/page_001/reconstruction/component_state.json", state)

    outcome = advance_component_repair(store, "page_001")

    assert outcome["status"] == "fallback_required"
    assert outcome["stop_reason"] == "no_executable_actions"
    assert not (request_path.parents[2] / "component_result.json").exists()


@pytest.mark.parametrize(
    ("accepted", "expected"),
    [(True, "ready_for_assembly"), (False, "preserved_with_warning")],
)
def test_intact_parent_gate_controls_fallback_result(
    page_session: dict, accepted: bool, expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, graph_path, quality_input_refs = _fallback_execution_case(page_session)
    record_parent_fallback_execution(
        store, "page_001", graph_path=graph_path,
        quality_input_refs=quality_input_refs,
    )
    monkeypatch.setattr(
        component_repair, "evaluate_component_quality_round",
        lambda *args, **kwargs: _quality_report_with_unexplained(
            _strict_quality_report("candidate_b", accepted), **kwargs
        ),
    )
    record_parent_fallback_quality(store, "page_001")

    result = advance_component_repair(store, "page_001")

    assert result["status"] == expected
    final_state = store.read_json("pages/page_001/reconstruction/component_state.json")
    if accepted:
        assert final_state["fallback"]["status"] == "parent_preserved"
        assert final_state["frozen"]["candidate_b"] == final_state["parent_assets"]["candidate_b"]["sha256"]
    else:
        assert final_state["fallback"]["status"] == "warning"


def _fallback_execution_case(page_session: dict):
    from image2editable.store import RunStore

    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "host"},
    })
    store.write_json("run_state.json", {"schema_version": 1, "status": "prepared"})
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "processing"}
    }})
    state = initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    state["phase"] = "fallback_required"
    state["stop_reason"] = "round_limit"
    state["fallback"] = {"status": "required", "parent_ids": ["candidate_b"]}
    store.write_json("pages/page_001/reconstruction/component_state.json", state)
    graph = load_component_agent_graph(request_path)
    next(node for node in graph["nodes"] if node["id"] == "candidate_b")[
        "state"
    ] = "pending_gate"
    fallback_dir = request_path.parents[2] / "fallback"
    fallback_dir.mkdir()
    (fallback_dir / "masks").mkdir()
    for component_id in ("candidate_b", "frozen_a"):
        shutil.copy2(
            request_path.parent / f"masks/{component_id}.png",
            fallback_dir / f"masks/{component_id}.png",
        )
    graph_path = fallback_dir / "component-graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    return store, graph_path, _quality_input_refs(fallback_dir, store, graph_path)


@pytest.mark.parametrize(
    "field",
    [
        "rgba", "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask",
    ],
)
def test_parent_fallback_rejects_replaced_frozen_presentation_asset(
    page_session: dict, field: str,
) -> None:
    store, graph_path, quality_input_refs = _fallback_execution_case(page_session)
    manifest_ref = quality_input_refs["presentation_manifest"]
    manifest_path = store.root / manifest_ref["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = next(
        item for item in manifest["components"]
        if item["component_id"] == "frozen_a"
    )
    asset_path = store.root / frozen[field]["path"]
    asset_path.write_bytes(asset_path.read_bytes() + b"fallback-replacement")
    frozen[field]["sha256"] = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_ref["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="frozen presentation"):
        record_parent_fallback_execution(
            store,
            "page_001",
            graph_path=graph_path,
            quality_input_refs=quality_input_refs,
        )


def test_parent_fallback_quality_history_keeps_replaced_child_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = {"nodes": [
        {"id": "parent_0001", "kind": "parent", "state": "pending_gate"},
        {"id": "component_0001", "kind": "visual", "state": "inactive"},
        {"id": "text_0001", "kind": "text", "state": "frozen"},
    ]}
    monkeypatch.setattr(
        component_repair, "validate_component_graph", lambda value: value
    )

    assert component_repair._quality_history_component_ids(
        graph, parent_fallback=True
    ) == ["parent_0001", "component_0001"]


def test_parent_candidate_falls_back_to_its_intact_paired_parent() -> None:
    node = {
        "id": "component_0008", "kind": "parent", "parent_id": None,
    }

    assert component_repair._fallback_parent_id(
        node, {"parent_0008": {"path": "intact-parent.png"}}
    ) == "parent_0008"


def _node(component_id: str, state: str, z_index: int) -> dict:
    return {
        "id": component_id,
        "kind": "parent",
        "parent_id": None,
        "state": state,
        "mask": f"masks/{component_id}.png",
        "mask_sha256": "a" * 64,
        "bbox": [0, 0, 2, 2],
        "z_index": z_index,
        "text_ids": [],
    }


def _action(action: str, object_ids: list[str], parameters: dict | None = None) -> dict:
    return {"action": action, "object_ids": object_ids, "parameters": parameters or {},
            "confidence": 0.95, "evidence": ["visible relationship"]}


def _strict_quality_report(component_id: str, accepted: bool) -> dict:
    metrics = {name: 0.0 for name in component_quality._METRIC_FIELDS}
    metrics.update({
        "component_pixels": 4,
        "parent_coverage_ratio": 1.0,
        "parent_child_double": False,
        "local_contrast": 1.0,
        "edge_width_px": 1,
        "text_halo_px": 1,
        "adaptive_pixel_tolerance": 3.0,
        "hard_pixel_tolerance": 3.0,
    })
    component_violations = [] if accepted else ["missing_edge"]
    page_violations = sorted(component_violations + ["pptx_reopen_unknown"])
    return {
        "accepted": False, "violations": page_violations,
        "component_reports": [{
            "component_id": component_id, "accepted": accepted, "metrics": metrics,
            "improvement": {}, "violations": component_violations,
            "checks": {"protected_native_overlap": "pass"},
            "agent_confidence": None,
        }],
        "visual_metrics": {"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        "checks": {"pptx_reopen": "unknown"},
    }


def _quality_report_with_unexplained(report: dict, **kwargs) -> dict:
    output = kwargs.get("unexplained_output_path")
    material = kwargs.get("material_foreground")
    if output is not None and material is not None:
        Image.fromarray(np.zeros(np.asarray(material).shape, dtype=np.uint8)).save(output)
    return report


def _quality_input_refs(directory: Path, store, graph_path: Path) -> dict:
    paths = {}
    for name in ("background", "reconstructed", "text-mask"):
        path = directory / f"{name}.png"
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)
        paths[name] = path
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    if state["quality_gate_version"] >= 2:
        foreground = directory / "foreground-evidence.png"
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(foreground)
        paths["foreground-evidence"] = foreground
    native = directory / "native-check.json"
    native.write_text(json.dumps({
        "schema_version": 1, "page_id": "page_001",
        "source_sha256": state["source_sha256"],
        "protected_native_overlap": "pass",
        "initial_diagnostics": [],
    }), encoding="utf-8")
    paths["native-check"] = native
    paths["presentation-manifest"] = _write_test_presentation_manifest(
        store.root,
        directory,
        source_sha256=state["source_sha256"],
        graph_path=graph_path,
    )
    return {
        name.replace("-", "_"): {
            "path": path.resolve().relative_to(store.root.resolve()).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
    }


@pytest.mark.parametrize(
    "native_diagnostics",
    [
        None,
        [],
        [{"kind": "unowned_raster_text"}],
        [{
            "kind": "unowned_raster_text", "source_sha256": "a" * 64,
            "candidate_id": "candidate_0002_01", "bbox": [1, 1, 3, 3],
            "views": [
                {"normalized_text": "x", "confidence": 0.96},
                {"normalized_text": "z", "confidence": 0.95},
            ],
        }],
    ],
)
def test_quality_native_diagnostics_must_match_request_evidence(
    tmp_path: Path,
    native_diagnostics,
) -> None:
    from image2editable.store import RunStore

    store = RunStore(tmp_path / "run")
    source_sha = "a" * 64
    expected = [{
        "kind": "unowned_raster_text", "source_sha256": source_sha,
        "candidate_id": "candidate_0001_01", "bbox": [1, 1, 3, 3],
        "views": [
            {"normalized_text": "x", "confidence": 0.96},
            {"normalized_text": "y", "confidence": 0.95},
        ],
    }]
    request_dir = store.root / "request"
    request_dir.mkdir(parents=True)
    quality_path = request_dir / "quality-report.json"
    quality_path.write_text(json.dumps({
        "initial_diagnostics": expected,
    }), encoding="utf-8")
    request_path = request_dir / "component_agent_request.json"
    request_path.write_text("{}", encoding="utf-8")
    request = {
        "graph_sha256": "b" * 64,
        "evidence": {"quality-report.json": {
            "path": "quality-report.json",
            "sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
        }},
    }
    native = {
        "schema_version": 1, "page_id": "page_001",
        "source_sha256": source_sha, "protected_native_overlap": "pass",
    }
    if native_diagnostics is not None:
        native["initial_diagnostics"] = native_diagnostics
    refs = {}
    for name in ("background", "reconstructed", "text_mask"):
        path = request_dir / f"{name}.bin"
        path.write_bytes(b"x")
        refs[name] = {
            "path": path.relative_to(store.root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    native_path = request_dir / "native-check.json"
    native_path.write_text(json.dumps(native), encoding="utf-8")
    refs["native_check"] = {
        "path": native_path.relative_to(store.root).as_posix(),
        "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
    }
    manifest_path = _write_test_presentation_manifest(
        store.root,
        request_dir,
        source_sha256=source_sha,
        graph_sha256=request["graph_sha256"],
    )
    refs["presentation_manifest"] = {
        "path": manifest_path.relative_to(store.root).as_posix(),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }

    with pytest.raises(ValueError, match="diagnostic|native"):
        component_repair._verify_quality_input_refs(
            store,
            {"page_id": "page_001", "source_sha256": source_sha},
            refs,
            request=request,
            request_path=request_path,
        )


def _action_case(tmp_path: Path) -> tuple[np.ndarray, dict, Path]:
    root = tmp_path / "round-01"
    masks = root / "masks"
    masks.mkdir(parents=True)
    values = {
        "parent": np.ones((12, 16), dtype=bool),
        "left": np.pad(np.ones((4, 4), dtype=bool), ((2, 6), (2, 10))),
        "right": np.pad(np.ones((4, 4), dtype=bool), ((2, 6), (10, 2))),
        "frozen": np.pad(np.ones((2, 2), dtype=bool), ((9, 1), (7, 7))),
        "text": np.pad(np.ones((1, 1), dtype=bool), ((0, 11), (0, 15))),
    }
    nodes = []
    specs = [
        ("parent", "parent", None, "inactive", 0),
        ("left", "child", "parent", "pending", 1),
        ("right", "child", "parent", "pending", 2),
        ("frozen", "parent", None, "frozen", 3),
        ("text", "text", None, "frozen", 4),
    ]
    for component_id, kind, parent_id, state, z_index in specs:
        path = masks / f"{component_id}.png"
        Image.fromarray(values[component_id].astype(np.uint8) * 255).save(path)
        ys, xs = np.where(values[component_id])
        nodes.append({"id": component_id, "kind": kind, "parent_id": parent_id,
                      "state": state, "mask": f"masks/{component_id}.png",
                      "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                      "z_index": z_index, "text_ids": []})
    return np.zeros((12, 16, 3), dtype=np.uint8), {"nodes": nodes}, root


def test_retry_actions_are_batched_before_graph_mutation(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    calls = []
    left = np.asarray(Image.open(input_dir / "masks/left.png")) > 0
    right = np.asarray(Image.open(input_dir / "masks/right.png")) > 0

    def batch_runner(*, image: np.ndarray, prompts: list[dict]) -> list[dict]:
        calls.append(prompts)
        return [
            {"component_id": "left", "mask": left},
            {"component_id": "right", "mask": right},
        ]

    result = execute_component_actions(
        image,
        graph,
        [
            _action(
                "retry_with_box",
                ["left"],
                {"box": [0.25, 0.25, 0.75, 0.75]},
            ),
            _action(
                "retry_with_points",
                ["right"],
                {"positive": [[0.8, 0.3]], "negative": [[0.5, 0.5]]},
            ),
        ],
        sam_batch_runner=batch_runner,
        input_dir=input_dir,
        output_dir=tmp_path / "round-batched-retry",
    )

    assert len(calls) == 1
    assert calls[0] == [
        {
            "component_id": "left",
            "box": [4.0, 3.0, 12.0, 9.0],
            "positive": [],
            "negative": [],
        },
        {
            "component_id": "right",
            "box": None,
            "positive": [[12.0, 3.3]],
            "negative": [[7.5, 5.5]],
        },
    ]
    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["left"]["state"] == "pending"
    assert by_id["right"]["state"] == "pending"


def test_retry_batch_rejects_invalid_second_mask_without_graph_mutation_or_output(
    tmp_path: Path,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    original_graph = copy.deepcopy(graph)
    valid = np.asarray(Image.open(input_dir / "masks/left.png")) > 0
    invalid = np.zeros(image.shape[:2], dtype=bool)
    output_dir = tmp_path / "round-invalid-batched-retry"

    with pytest.raises(VisualSegmentationError, match="invalid mask"):
        execute_component_actions(
            image,
            graph,
            [
                _action(
                    "retry_with_box",
                    ["left"],
                    {"box": [0.25, 0.25, 0.75, 0.75]},
                ),
                _action(
                    "retry_with_points",
                    ["right"],
                    {"positive": [[0.8, 0.3]], "negative": []},
                ),
            ],
            sam_batch_runner=lambda **_: [
                {"component_id": "left", "mask": valid},
                {"component_id": "right", "mask": invalid},
            ],
            input_dir=input_dir,
            output_dir=output_dir,
        )

    assert graph == original_graph
    assert not output_dir.exists()


def test_retry_batch_rejects_reordered_results_before_graph_mutation(
    tmp_path: Path,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    original_graph = copy.deepcopy(graph)
    left = np.asarray(Image.open(input_dir / "masks/left.png")) > 0
    right = np.asarray(Image.open(input_dir / "masks/right.png")) > 0
    output_dir = tmp_path / "round-reordered-batched-retry"

    with pytest.raises(VisualSegmentationError, match="order"):
        execute_component_actions(
            image,
            graph,
            [
                _action(
                    "retry_with_box", ["left"], {"box": [0.25, 0.25, 0.75, 0.75]},
                ),
                _action(
                    "retry_with_points", ["right"], {"positive": [[0.8, 0.3]], "negative": []},
                ),
            ],
            sam_batch_runner=lambda **_: [
                {"component_id": "right", "mask": right},
                {"component_id": "left", "mask": left},
            ],
            input_dir=input_dir,
            output_dir=output_dir,
        )

    assert graph == original_graph
    assert not output_dir.exists()


def test_single_retry_runner_preserves_uint8_mask_compatibility(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    uint8_mask = np.asarray(Image.open(input_dir / "masks/left.png"), dtype=np.uint8)

    result = execute_component_actions(
        image,
        graph,
        [_action("retry_with_box", ["left"], {"box": [0.25, 0.25, 0.75, 0.75]})],
        sam_runner=lambda **_: uint8_mask,
        input_dir=input_dir,
        output_dir=tmp_path / "round-single-uint8",
    )

    assert next(node for node in result["nodes"] if node["id"] == "left")["state"] == "pending"


def test_component_plan_rejects_excessive_prompt_points() -> None:
    from image2editable.component_contracts import (
        MAX_COMPONENT_PROMPT_POINTS,
        validate_component_action,
    )

    with pytest.raises(ValueError, match="too many"):
        validate_component_action(_action(
            "retry_with_points",
            ["component"],
            {
                "positive": [[0.5, 0.5]] * (MAX_COMPONENT_PROMPT_POINTS + 1),
                "negative": [],
            },
        ))

    from scripts import sam_worker

    assert MAX_COMPONENT_PROMPT_POINTS == sam_worker._COMPONENT_MAX_POINTS_PER_PROMPT


def test_execute_accept_is_pending_gate_and_preserves_frozen_hash(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    output = tmp_path / "round-02"
    result = execute_component_actions(
        image, graph, [_action("accept", ["left"])], sam_runner=None,
        input_dir=input_dir, output_dir=output,
    )
    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["left"]["state"] == "pending_gate"
    assert by_id["frozen"] == next(node for node in graph["nodes"] if node["id"] == "frozen")
    assert (output / "component-graph.json").is_file()


def test_accept_completion_does_not_create_active_overlap(tmp_path: Path) -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    accepted_before = np.zeros(image.shape[:2], dtype=bool)
    accepted_before[1:11, 2:12] = True
    active_visual = np.zeros_like(accepted_before)
    inactive_visual = np.zeros_like(accepted_before)
    frozen_text = np.zeros_like(accepted_before)
    active_visual[4, 5] = True
    inactive_visual[6, 7] = True
    frozen_text[8, 9] = True
    accepted_before[active_visual | inactive_visual | frozen_text] = False
    assert not np.any(accepted_before & active_visual)

    input_dir = tmp_path / "accept-overlap-input"
    mask_dir = input_dir / "masks"
    mask_dir.mkdir(parents=True)
    specs = [
        ("accepted", "parent", "pending", accepted_before),
        ("active", "parent", "frozen", active_visual),
        ("inactive", "parent", "inactive", inactive_visual),
        ("text", "text", "frozen", frozen_text),
    ]
    nodes = []
    for z_index, (component_id, kind, state, mask) in enumerate(specs):
        path = mask_dir / f"{component_id}.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)
        ys, xs = np.where(mask)
        nodes.append({
            "id": component_id,
            "kind": kind,
            "parent_id": None,
            "state": state,
            "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bbox": [
                int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1,
            ],
            "z_index": z_index,
            "text_ids": [],
        })

    output_dir = tmp_path / "accept-overlap-output"
    result = execute_component_actions(
        image,
        {"nodes": nodes},
        [_action("accept", ["accepted"])],
        sam_runner=None,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    accepted = next(node for node in result["nodes"] if node["id"] == "accepted")
    actual = np.asarray(Image.open(output_dir / accepted["mask"])) > 0
    added = actual & ~accepted_before
    assert accepted["state"] == "pending_gate"
    assert np.all(actual[accepted_before])
    assert not np.any(added & active_visual)
    assert np.all(actual[inactive_visual])
    assert np.all(actual[frozen_text])


def test_execute_accept_can_detach_confirmed_independent_visual(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    left = next(node for node in graph["nodes"] if node["id"] == "left")
    assert left["parent_id"] == "parent"

    result = execute_component_actions(
        image, graph,
        [_action("accept", ["left"], {"independent": True})],
        sam_runner=None, input_dir=input_dir,
        output_dir=tmp_path / "round-independent-accept",
    )

    accepted = next(node for node in result["nodes"] if node["id"] == "left")
    assert accepted["state"] == "pending_gate"
    assert accepted["kind"] == "parent"
    assert accepted["parent_id"] is None


def test_independent_accept_restores_parent_backing_only_under_editable_text(
    tmp_path: Path,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    by_id = {node["id"]: node for node in graph["nodes"]}
    text = np.zeros(image.shape[:2], dtype=np.uint8)
    text[2:4, 2:4] = 255
    text_path = input_dir / by_id["text"]["mask"]
    Image.fromarray(text, mode="L").save(text_path)
    by_id["text"]["mask_sha256"] = hashlib.sha256(text_path.read_bytes()).hexdigest()
    by_id["text"]["bbox"] = [2, 2, 4, 4]
    left_path = input_dir / by_id["left"]["mask"]
    left_before = np.asarray(Image.open(left_path)) > 0
    left_before[2:4, 2:4] = False
    Image.fromarray(left_before.astype(np.uint8) * 255, mode="L").save(left_path)
    by_id["left"]["mask_sha256"] = hashlib.sha256(left_path.read_bytes()).hexdigest()

    output = tmp_path / "round-independent-text-backing"
    result = execute_component_actions(
        image, graph,
        [_action("accept", ["left"], {"independent": True})],
        sam_runner=None, input_dir=input_dir, output_dir=output,
    )

    accepted = next(node for node in result["nodes"] if node["id"] == "left")
    actual = np.asarray(Image.open(output / accepted["mask"])) > 0
    expected = left_before.copy()
    expected[2:4, 2:4] = True
    assert np.array_equal(actual, expected)


def test_execute_suppress_text_rebuilds_false_ocr_as_visual_component(
    tmp_path: Path,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    next(node for node in graph["nodes"] if node["id"] == "frozen")[
        "text_ids"
    ] = ["text"]
    before = {
        node["id"]: dict(node)
        for node in graph["nodes"]
    }
    proposed = np.zeros(image.shape[:2], dtype=bool)
    proposed[3:7, 4:8] = True
    calls = []

    def batch_runner(*, image: np.ndarray, prompts: list[dict]) -> list[dict]:
        calls.append(prompts)
        return [{"component_id": "text", "mask": proposed}]

    result = execute_component_actions(
        image,
        graph,
        [_action("suppress_text", ["text"])],
        sam_batch_runner=batch_runner,
        input_dir=input_dir,
        output_dir=tmp_path / "round-suppress-text",
    )

    by_id = {node["id"]: node for node in result["nodes"]}
    promoted = next(node for node in result["nodes"] if node["id"] not in before)
    assert calls == [[{
        "component_id": "text",
        "box": [0.0, 0.0, 1.0, 1.0],
        "positive": [],
        "negative": [],
    }]]
    assert by_id["text"] == {**before["text"], "state": "inactive"}
    assert by_id["frozen"] == {**before["frozen"], "text_ids": []}
    assert by_id["left"] == before["left"]
    assert promoted["kind"] == "parent"
    assert promoted["parent_id"] is None
    assert promoted["state"] == "pending"
    assert promoted["text_ids"] == []
    actual = np.asarray(Image.open(
        tmp_path / "round-suppress-text" / promoted["mask"]
    )) > 0
    assert np.array_equal(actual, proposed)


def test_execute_suppress_text_rejects_frozen_visual(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)

    with pytest.raises(ValueError, match="text"):
        execute_component_actions(
            image,
            graph,
            [_action("suppress_text", ["frozen"])],
            sam_runner=None,
            input_dir=input_dir,
            output_dir=tmp_path / "round-invalid-suppress-text",
        )


def test_execute_discard_inactivates_redundant_candidate(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)

    result = execute_component_actions(
        image, graph, [_action("discard", ["left"])], sam_runner=None,
        input_dir=input_dir, output_dir=tmp_path / "round-discard",
    )

    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["left"]["state"] == "inactive"
    assert by_id["right"]["state"] == "pending"


def test_execute_background_rebuild_action_preserves_component_graph(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)

    result = execute_component_actions(
        image, graph,
        [_action("rebuild_background", ["left", "right"], {"margin_ratio": 0.01})],
        sam_runner=None, input_dir=input_dir,
        output_dir=tmp_path / "round-background",
    )

    assert result == graph


def test_execute_background_rebuild_can_clear_frozen_visuals(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    next(node for node in graph["nodes"] if node["id"] == "left")[
        "state"
    ] = "frozen"

    result = execute_component_actions(
        image, graph,
        [_action("rebuild_background", ["left", "right"], {"margin_ratio": 0.01})],
        sam_runner=None, input_dir=input_dir,
        output_dir=tmp_path / "round-background-frozen",
    )

    assert result == graph


def test_execute_absorb_unions_visuals_into_one_parent(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    parent = next(node for node in graph["nodes"] if node["id"] == "parent")
    parent_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    parent_mask[0, 0] = 255
    parent_path = input_dir / parent["mask"]
    Image.fromarray(parent_mask).save(parent_path)
    parent["mask_sha256"] = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    parent["bbox"] = [0, 0, 1, 1]

    result = execute_component_actions(
        image, graph,
        [
            _action("absorb_into_parent", ["parent", "left", "right"]),
            _action("rebuild_background", ["left"], {"margin_ratio": 0.01}),
        ],
        sam_runner=None, input_dir=input_dir,
        output_dir=tmp_path / "round-absorb",
    )

    by_id = {node["id"]: node for node in result["nodes"]}
    absorbed = np.asarray(Image.open(
        tmp_path / "round-absorb" / by_id["parent"]["mask"]
    )) > 0
    assert int(absorbed.sum()) == 33
    assert by_id["parent"]["state"] == "pending"
    assert by_id["left"]["state"] == by_id["right"]["state"] == "inactive"


def _execute_composite_quality_round(
    store,
    request_path: Path,
    graph: dict,
    *,
    action: dict,
    shape: tuple[int, int],
    before_execution_record=None,
    before_execution_submit=None,
    before_quality=None,
    initial_diagnostics: list[dict] | None = None,
    presentation_metrics_by_id: dict[str, dict] | None = None,
    background_responsibility: np.ndarray | None = None,
) -> tuple[dict, dict]:
    request = load_component_agent_request(request_path)
    plan = {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "local", "repair_round": request["repair_round"],
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [action],
    }
    record_local_component_plan(store, "page_001", plan=plan)
    execution_dir = request_path.parents[2] / f"execution-{request['repair_round']:02d}"
    execute_component_actions(
        np.zeros((*shape, 3), dtype=np.uint8), graph, plan["actions"],
        sam_runner=None, input_dir=request_path.parent, output_dir=execution_dir,
    )
    graph_path = execution_dir / "component-graph.json"
    quality_paths = {}
    for name in ("background", "reconstructed", "text-mask"):
        path = execution_dir / f"{name}.png"
        Image.fromarray(np.zeros((*shape, 3), dtype=np.uint8)).save(path)
        quality_paths[name.replace("-", "_")] = path
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    if state["quality_gate_version"] >= 2:
        foreground = execution_dir / "foreground-evidence.png"
        Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(foreground)
        quality_paths["foreground_evidence"] = foreground
    if background_responsibility is not None:
        responsibility = execution_dir / "background-responsibility.png"
        Image.fromarray(background_responsibility).save(responsibility)
        quality_paths["background_responsibility"] = responsibility
    native = execution_dir / "native-check.json"
    native.write_text(json.dumps({
        "schema_version": 1, "page_id": "page_001",
        "source_sha256": state["source_sha256"],
        "protected_native_overlap": "pass",
        "initial_diagnostics": initial_diagnostics or [],
    }), encoding="utf-8")
    quality_paths["native_check"] = native
    quality_paths["presentation_manifest"] = _write_test_presentation_manifest(
        store.root,
        execution_dir,
        source_sha256=state["source_sha256"],
        graph_path=graph_path,
    )
    if presentation_metrics_by_id:
        manifest_path = quality_paths["presentation_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for component in manifest["components"]:
            if component["component_id"] in presentation_metrics_by_id:
                component["metrics"] = presentation_metrics_by_id[
                    component["component_id"]
                ]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    quality_refs = {
        name: {
            "path": path.resolve().relative_to(store.root.resolve()).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in quality_paths.items()
    }
    execution = {
        "schema_version": 1, "page_id": "page_001", "provider": "local",
        "repair_round": request["repair_round"],
        "request_sha256": plan["request_sha256"],
        "input_graph_sha256": request["graph_sha256"],
        "output_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "executable_action_count": 1, "quality_input_refs": quality_refs,
    }
    execution_path = execution_dir / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    if before_execution_record is not None:
        before_execution_record(graph_path)
        execution["output_graph_sha256"] = hashlib.sha256(
            graph_path.read_bytes()
        ).hexdigest()
        manifest_path = _write_test_presentation_manifest(
            store.root,
            execution_dir,
            source_sha256=state["source_sha256"],
            graph_path=graph_path,
        )
        execution["quality_input_refs"]["presentation_manifest"] = {
            "path": manifest_path.relative_to(store.root).as_posix(),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
    if before_execution_submit is not None:
        before_execution_submit(
            quality_paths["presentation_manifest"], execution
        )
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
    record_component_execution(
        store, "page_001", execution_path=execution_path,
        output_graph_path=graph_path,
    )
    if before_quality is not None:
        before_quality()
    record_component_quality(store, "page_001")
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    quality = json.loads(
        (store.root / state["current_round"]["quality_ref"]["path"])
        .read_text(encoding="utf-8")
    )
    return quality["report"], advance_component_repair(store, "page_001")


def test_background_responsibility_artifact_must_be_binary(
    page_session: dict,
) -> None:
    store, request_path = _start_quality_mutation_round(page_session)

    with pytest.raises(
        ValueError, match="background responsibility is invalid"
    ):
        _execute_composite_quality_round(
            store,
            request_path,
            load_component_agent_graph(request_path),
            action=_action("accept", ["candidate_b"]),
            shape=(2, 2),
            background_responsibility=np.array(
                [[0, 1], [0, 255]], dtype=np.uint8
            ),
        )


def test_background_responsibility_artifact_is_hash_bound(
    page_session: dict,
) -> None:
    store, request_path = _start_quality_mutation_round(page_session)

    def tamper() -> None:
        path = request_path.parents[2] / (
            "execution-01/background-responsibility.png"
        )
        path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        _execute_composite_quality_round(
            store,
            request_path,
            load_component_agent_graph(request_path),
            action=_action("accept", ["candidate_b"]),
            shape=(2, 2),
            background_responsibility=np.array(
                [[0, 0], [0, 255]], dtype=np.uint8
            ),
            before_quality=tamper,
        )


def _start_quality_mutation_round(page_session: dict):
    from image2editable.store import RunStore

    page_session["provider"] = "local"
    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    advance_component_repair(store, "page_001")
    return store, request_path


@pytest.mark.parametrize(
    "field",
    [
        "rgba", "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask",
    ],
)
def test_frozen_presentation_asset_cannot_be_replaced_during_execution(
    page_session: dict, field: str,
) -> None:
    store, request_path = _start_quality_mutation_round(page_session)

    def replace_frozen_asset(manifest_path: Path, execution: dict) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen = next(
            item for item in manifest["components"]
            if item["component_id"] == "frozen_a"
        )
        asset_path = store.root / frozen[field]["path"]
        asset_path.write_bytes(asset_path.read_bytes() + b"replacement")
        frozen[field]["sha256"] = hashlib.sha256(
            asset_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        execution["quality_input_refs"]["presentation_manifest"]["sha256"] = (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        )

    with pytest.raises(ValueError, match="frozen presentation"):
        _execute_composite_quality_round(
            store,
            request_path,
            load_component_agent_graph(request_path),
            action=_action("accept", ["candidate_b"]),
            shape=(2, 2),
            before_execution_submit=replace_frozen_asset,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("manifest_ref", "component repair artifact hash mismatch"),
        ("component_order", "components do not match graph"),
        ("nonbinary", "masks must be binary"),
        ("alpha_union", "mask alpha union is invalid"),
    ],
)
def test_record_quality_fails_closed_on_presentation_mutation(
    page_session: dict, mutation: str, error: str,
) -> None:
    store, request_path = _start_quality_mutation_round(page_session)

    def mutate_quality_input() -> None:
        state_path = "pages/page_001/reconstruction/component_state.json"
        state = store.read_json(state_path)
        execution_path = store.root / state["current_round"]["execution_ref"]["path"]
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        manifest_ref = execution["quality_input_refs"]["presentation_manifest"]
        manifest_path = store.root / manifest_ref["path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        component = manifest["components"][0]
        if mutation == "component_order":
            manifest["components"].reverse()
        elif mutation in {"nonbinary", "alpha_union"}:
            field = (
                "ownership_mask" if mutation == "nonbinary"
                else "presentation_alpha_mask"
            )
            asset_path = store.root / component[field]["path"]
            with Image.open(asset_path) as image:
                mask = np.asarray(image.convert("L")).copy()
            if mutation == "nonbinary":
                mask.flat[0] = 127
            else:
                mask[:] = 0
            Image.fromarray(mask).save(asset_path)
            component[field]["sha256"] = hashlib.sha256(
                asset_path.read_bytes()
            ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if mutation == "manifest_ref":
            manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
            return
        manifest_ref["sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
        state["current_round"]["execution_ref"]["sha256"] = hashlib.sha256(
            execution_path.read_bytes()
        ).hexdigest()
        store.write_json(state_path, state)

    with pytest.raises((ValueError, RuntimeError), match=error):
        _execute_composite_quality_round(
            store,
            request_path,
            load_component_agent_graph(request_path),
            action=_action("accept", ["candidate_b"]),
            shape=(2, 2),
            before_quality=mutate_quality_input,
        )


def test_record_quality_rejects_exact_empty_presentation_component(
    page_session: dict,
) -> None:
    store, request_path = _start_quality_mutation_round(page_session)

    def empty_candidate_assets() -> None:
        state_path = "pages/page_001/reconstruction/component_state.json"
        state = store.read_json(state_path)
        execution_path = store.root / state["current_round"]["execution_ref"]["path"]
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        manifest_ref = execution["quality_input_refs"]["presentation_manifest"]
        manifest_path = store.root / manifest_ref["path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        component = manifest["components"][0]
        for field in (
            "ownership_mask", "presentation_alpha_mask",
            "generated_underlay_mask", "rgba",
        ):
            asset_path = store.root / component[field]["path"]
            mode = "RGBA" if field == "rgba" else "L"
            shape = (2, 2, 4) if field == "rgba" else (2, 2)
            Image.fromarray(np.zeros(shape, dtype=np.uint8), mode=mode).save(asset_path)
            component[field]["sha256"] = hashlib.sha256(
                asset_path.read_bytes()
            ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_ref["sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
        state["current_round"]["execution_ref"]["sha256"] = hashlib.sha256(
            execution_path.read_bytes()
        ).hexdigest()
        store.write_json(state_path, state)

    report, freeze = _execute_composite_quality_round(
        store,
        request_path,
        load_component_agent_graph(request_path),
        action=_action("accept", ["candidate_b"]),
        shape=(2, 2),
        before_quality=empty_candidate_assets,
    )

    candidate = report["component_reports"][0]
    assert "empty_component" in candidate["violations"]
    assert freeze["failed_ids"] == ["candidate_b"]


@pytest.mark.parametrize(
    "field",
    [
        "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask", "rgba",
    ],
)
def test_record_quality_rejects_post_execution_asset_tamper(
    page_session: dict,
    field: str,
) -> None:
    store, request_path = _start_quality_mutation_round(page_session)

    def tamper_after_execution_record() -> None:
        state = store.read_json(
            "pages/page_001/reconstruction/component_state.json"
        )
        execution = json.loads((
            store.root / state["current_round"]["execution_ref"]["path"]
        ).read_text(encoding="utf-8"))
        manifest_ref = execution["quality_input_refs"]["presentation_manifest"]
        manifest = json.loads((store.root / manifest_ref["path"]).read_text(
            encoding="utf-8"
        ))
        asset_path = store.root / manifest["components"][0][field]["path"]
        asset_path.write_bytes(asset_path.read_bytes() + b"tampered")

    with pytest.raises(
        RuntimeError,
        match=rf"presentation asset hash mismatch: candidate_b/{field}",
    ):
        _execute_composite_quality_round(
            store,
            request_path,
            load_component_agent_graph(request_path),
            action=_action("accept", ["candidate_b"]),
            shape=(2, 2),
            before_quality=tamper_after_execution_record,
        )


def _record_composite_quality(
    page_session: dict,
    *,
    left_box: tuple[int, int, int, int],
    right_box: tuple[int, int, int, int],
    action_name: str = "absorb_into_parent",
    before_execution_record=None,
    before_quality=None,
) -> tuple[dict, dict, object, dict]:
    from image2editable.store import RunStore

    evidence_root = Path(page_session["reconstruction_dir"]) / "evidence-source"
    shape = (30, 40)
    masks_dir = evidence_root / "masks"
    shutil.rmtree(masks_dir)
    masks_dir.mkdir()
    values = {
        "parent": _box_mask(shape, (0, 0, 1, 1)),
        "left": _box_mask(shape, left_box),
        "right": _box_mask(shape, right_box),
    }
    if action_name == "collapse_to_parent":
        values["parent"] = np.maximum(values["left"], values["right"])
    nodes = []
    for component_id, kind, parent_id, state, z_index in (
        ("parent", "parent", None, "inactive", 0),
        ("left", "child", "parent", "pending", 1),
        ("right", "child", "parent", "pending", 2),
    ):
        mask_path = masks_dir / f"{component_id}.png"
        Image.fromarray(values[component_id]).save(mask_path)
        ys, xs = np.where(values[component_id] > 0)
        nodes.append({
            "id": component_id, "kind": kind, "parent_id": parent_id,
            "state": state, "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "z_index": z_index, "text_ids": [],
        })
    graph = {"nodes": nodes}
    (evidence_root / "component-graph.json").write_text(
        json.dumps(graph), encoding="utf-8"
    )
    for name in EVIDENCE_NAMES:
        path = evidence_root / name
        if path.suffix == ".png" and name != "component-graph.json":
            Image.fromarray(np.zeros((*shape, 3), dtype=np.uint8)).save(path)
    page_session["provider"] = "local"
    _refresh_test_presentation_manifest(page_session)
    request_path = build_component_agent_request(page_session, repair_round=1)
    store = RunStore(request_path.parents[5])
    store.write_json("job_manifest.json", {
        "schema_version": 1, "pages": ["page_001"],
        "options": {"agent_provider": "local"},
    })
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=2,
    )
    advance_component_repair(store, "page_001")
    object_ids = {
        "absorb_into_parent": ["parent", "left", "right"],
        "collapse_to_parent": ["parent"],
        "merge": ["left", "right"],
    }[action_name]
    report, freeze = _execute_composite_quality_round(
        store, request_path, graph,
        action=_action(action_name, object_ids), shape=shape,
        before_execution_record=before_execution_record,
        before_quality=before_quality,
    )
    return report, freeze, store, page_session


def _box_mask(
    shape: tuple[int, int], box: tuple[int, int, int, int]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    y1, x1, y2, x2 = box
    mask[y1:y2, x1:x2] = 255
    return mask


def _rewrite_graph_mask_as(
    graph_path: Path, source_id: str, replacement_id: str
) -> None:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    by_id = {node["id"]: node for node in graph["nodes"]}
    source_path = graph_path.parent / by_id[source_id]["mask"]
    replacement_path = graph_path.parent / by_id[replacement_id]["mask"]
    source_path.write_bytes(replacement_path.read_bytes())
    by_id[source_id]["mask_sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    by_id[source_id]["bbox"] = list(by_id[replacement_id]["bbox"])
    graph_path.write_text(json.dumps(graph), encoding="utf-8")


def _rewrite_graph_source_role(graph_path: Path, source_id: str) -> None:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    source = next(node for node in graph["nodes"] if node["id"] == source_id)
    source["kind"] = "text"
    source["parent_id"] = None
    graph_path.write_text(json.dumps(graph), encoding="utf-8")


def _rewrite_graph_source_id(graph_path: Path, source_id: str) -> None:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    source = next(node for node in graph["nodes"] if node["id"] == source_id)
    source["id"] = f"{source_id}_changed"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")


def test_absorbing_independent_candidates_is_hard_over_merge_failure(
    page_session: dict,
) -> None:
    report, freeze, _, _ = _record_composite_quality(
        page_session, left_box=(2, 2, 8, 8), right_box=(18, 28, 24, 34)
    )

    parent = report["component_reports"][0]
    assert "over_merged_component" in parent["violations"]
    assert parent["accepted"] is False
    assert freeze["failed_ids"] == ["parent"]
    assert "parent" not in freeze["frozen_ids"]


def test_top_level_parent_ignores_unrelated_top_level_sources() -> None:
    target = {"id": "target", "kind": "parent", "parent_id": None}
    unrelated = {"id": "unrelated", "kind": "parent", "parent_id": None}
    nodes_by_id = {node["id"]: node for node in (target, unrelated)}

    assert component_repair._is_related_inactive_source(
        target, unrelated, nodes_by_id
    ) is False


def test_contained_pair_approval_requires_explicit_cross_evidence() -> None:
    pair = {("inner", "outer")}
    actions = [
        _action("accept", ["inner"]),
        _action("accept", ["outer"]),
    ]
    actions[0]["confidence"] = 0.94
    actions[1]["confidence"] = 0.93
    for action in actions:
        action["evidence"] = ["inner", "outer", "independent visual units"]

    assert component_repair._approved_contained_parent_pairs(
        {"actions": actions}, pair
    ) == pair

    actions[0]["evidence"] = ["inner only"]
    assert component_repair._approved_contained_parent_pairs(
        {"actions": actions}, pair
    ) == set()


def test_contained_pair_approval_does_not_survive_without_mask_binding() -> None:
    previous = {
        "approved_contained_parent_pairs": [
            ["inner", "outer"], ["old_inner", "old_outer"],
        ],
    }

    assert component_repair._carried_contained_parent_pairs(
        previous, {("inner", "outer"), ("new_inner", "new_outer")}
    ) == set()


def test_explicit_retry_authorizes_one_next_round_without_prior_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        component_repair, "_page_quality_progressed", lambda *_: False,
    )

    assert component_repair._next_round_progress_allowed(
        None, {"stop_reason": "no_quality_improvement"}
    ) is True
    assert component_repair._next_round_progress_allowed(
        None, {"stop_reason": None}
    ) is False


def test_unapproved_contained_pair_reopens_both_visual_owners() -> None:
    quality = {
        "contained_parent_pairs": [
            ["frozen_inner", "pending_outer"],
            ["approved_inner", "approved_outer"],
        ],
        "approved_contained_parent_pairs": [
            ["approved_inner", "approved_outer"],
        ],
    }

    assert component_repair._unapproved_contained_parent_ids(quality) == {
        "frozen_inner", "pending_outer",
    }


def test_component_overlap_reopens_only_affected_frozen_owner() -> None:
    graph = {"nodes": [
        {"id": "pending", "kind": "parent", "state": "pending"},
        {"id": "affected", "kind": "parent", "state": "frozen"},
        {"id": "unrelated", "kind": "parent", "state": "frozen"},
    ]}
    report = {"component_reports": [{
        "component_id": "pending",
        "violations": ["component_overlap"],
        "overlap_component_ids": ["affected"],
    }]}

    assert component_repair._failed_overlap_dependency_ids(report, graph) == {
        "affected"
    }


def test_pair_approval_changes_normalized_plan_without_accepting_rewording() -> None:
    actions = [_action("accept", ["inner"]), _action("accept", ["outer"])]
    first = {"actions": json.loads(json.dumps(actions))}
    approved = {"actions": json.loads(json.dumps(actions))}
    reworded = {"actions": json.loads(json.dumps(actions))}
    for action in approved["actions"]:
        action["evidence"] = ["inner", "outer", "independent units"]
    for action in reworded["actions"]:
        action["evidence"] = ["different prose only"]

    assert component_repair._normalized_plan_sha256(first) != (
        component_repair._normalized_plan_sha256(approved)
    )
    assert component_repair._normalized_plan_sha256(first) == (
        component_repair._normalized_plan_sha256(reworded)
    )


def test_absorbing_adjacent_complete_rectangles_is_hard_over_merge_failure(
    page_session: dict,
) -> None:
    report, freeze, _, _ = _record_composite_quality(
        page_session, left_box=(5, 3, 15, 13), right_box=(5, 14, 15, 24)
    )

    parent = report["component_reports"][0]
    assert "over_merged_component" in parent["violations"]
    assert parent["accepted"] is False
    assert freeze["failed_ids"] == ["parent"]
    assert "parent" not in freeze["frozen_ids"]


@pytest.mark.parametrize("action_name", ["absorb_into_parent", "merge", "collapse_to_parent"])
def test_execution_rejects_rewritten_newly_inactive_source_mask(
    page_session: dict,
    action_name: str,
) -> None:
    def rewrite_source(graph_path: Path) -> None:
        _rewrite_graph_mask_as(graph_path, "left", "right")

    with pytest.raises(ValueError, match="source|provenance|inactive"):
        _record_composite_quality(
            page_session,
            left_box=(2, 2, 8, 8),
            right_box=(18, 28, 24, 34),
            action_name=action_name,
            before_execution_record=rewrite_source,
        )


@pytest.mark.parametrize("action_name", ["absorb_into_parent", "merge", "collapse_to_parent"])
def test_execution_rejects_rewritten_newly_inactive_source_role(
    page_session: dict,
    action_name: str,
) -> None:
    with pytest.raises(ValueError, match="source|provenance|inactive"):
        _record_composite_quality(
            page_session,
            left_box=(2, 2, 8, 8),
            right_box=(18, 28, 24, 34),
            action_name=action_name,
            before_execution_record=lambda graph_path: _rewrite_graph_source_role(
                graph_path, "left"
            ),
        )


def test_execution_rejects_rewritten_newly_inactive_source_id(
    page_session: dict,
) -> None:
    with pytest.raises(ValueError, match="source|provenance|inactive|identity"):
        _record_composite_quality(
            page_session,
            left_box=(2, 2, 8, 8),
            right_box=(18, 28, 24, 34),
            before_execution_record=lambda graph_path: _rewrite_graph_source_id(
                graph_path, "left"
            ),
        )


def test_absorbing_overlapping_duplicate_masks_can_freeze_parent(
    page_session: dict,
) -> None:
    report, freeze, _, _ = _record_composite_quality(
        page_session, left_box=(3, 3, 13, 13), right_box=(4, 4, 12, 12)
    )

    parent = report["component_reports"][0]
    assert "over_merged_component" not in parent["violations"]
    assert parent["accepted"] is True
    assert freeze["frozen_ids"] == ["parent"]


@pytest.mark.parametrize(
    ("action_name", "target_id"),
    [("merge", "merge_0001"), ("collapse_to_parent", "parent")],
)
def test_other_composite_actions_reject_independent_leaf_candidates(
    page_session: dict,
    action_name: str,
    target_id: str,
) -> None:
    report, freeze, _, _ = _record_composite_quality(
        page_session,
        left_box=(2, 2, 8, 8),
        right_box=(18, 28, 24, 34),
        action_name=action_name,
    )

    target = next(
        item for item in report["component_reports"]
        if item["component_id"] == target_id
    )
    assert "over_merged_component" in target["violations"]
    assert freeze["failed_ids"] == [target_id]
    assert target_id not in freeze["frozen_ids"]


def test_over_merge_violation_survives_next_round_accept_of_unchanged_parent(
    page_session: dict,
) -> None:
    _, first_freeze, store, session = _record_composite_quality(
        page_session,
        left_box=(2, 2, 8, 8),
        right_box=(18, 28, 24, 34),
    )
    assert first_freeze["failed_ids"] == ["parent"]
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    session = _real_next_round_session(
        session,
        store,
        store.root / state["current_round"]["quality_ref"]["path"],
    )
    second_request = build_component_agent_request(session, repair_round=2)
    record_next_component_request(
        store, "page_001", request_path=second_request
    )
    advance_component_repair(store, "page_001")
    graph = load_component_agent_graph(second_request)

    report, second_freeze = _execute_composite_quality_round(
        store,
        second_request,
        graph,
        action=_action("accept", ["parent"]),
        shape=(30, 40),
    )

    parent = report["component_reports"][0]
    assert "over_merged_component" in parent["violations"]
    assert second_freeze["failed_ids"] == ["parent"]
    assert "parent" not in second_freeze["frozen_ids"]
    state_before = store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    third_session = _real_next_round_session(
        session,
        store,
        store.root / state_before["current_round"]["quality_ref"]["path"],
    )
    third_request = build_component_agent_request(third_session, repair_round=3)
    with pytest.raises(RuntimeError, match="quality did not improve"):
        record_next_component_request(
            store, "page_001", request_path=third_request
        )
    assert store.read_json(
        "pages/page_001/reconstruction/component_state.json"
    ) == state_before
    outcome = advance_component_repair(store, "page_001")
    assert outcome["status"] == "fallback_required"
    assert outcome["stop_reason"] == "no_quality_improvement"


def test_page_quality_progresses_when_round_freezes_new_components() -> None:
    state = {
        "repair_round": 2,
        "round_history": [
            {"round": 1, "frozen_ids": []},
            {"round": 2, "frozen_ids": ["isolated_panel"]},
        ],
    }

    assert component_repair._page_quality_progressed(None, state) is True


def test_page_quality_progresses_when_candidate_topology_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "repair_round": 2,
        "failed_ids": ["row_1", "row_2"],
        "round_history": [
            {"round": 1, "frozen_ids": []},
            {"round": 2, "frozen_ids": []},
        ],
        "current_round": {
            "request_ref": {"path": "current-request.json"},
        },
    }
    monkeypatch.setattr(
        component_repair,
        "load_component_agent_request",
        lambda _: {"candidate_ids": ["composite"]},
    )
    store = type("Store", (), {"root": tmp_path})()

    assert component_repair._page_quality_progressed(store, state) is True


def test_page_quality_progresses_when_repair_makes_new_component_acceptable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def report(*, repaired: bool, unexplained_pixels: int) -> dict:
        return {
            "component_reports": [
                {
                    "component_id": "component_0009",
                    "accepted": repaired,
                },
                {
                    "component_id": "component_0013",
                    "accepted": not repaired,
                },
            ],
            "violations": ["unexplained_visual_residual"],
            "visual_metrics": {
                "largest_unexplained_region_pixels": unexplained_pixels,
                "unexplained_visual_pixels": unexplained_pixels,
                "mae": float(unexplained_pixels),
                "p95": float(unexplained_pixels),
            },
        }

    previous_payload = json.dumps({
        "report": report(repaired=False, unexplained_pixels=100),
    }).encode("utf-8")
    current_payload = json.dumps({
        "report": report(repaired=True, unexplained_pixels=200),
    }).encode("utf-8")
    (tmp_path / "previous-quality.json").write_bytes(previous_payload)
    (tmp_path / "current-quality.json").write_bytes(current_payload)
    monkeypatch.setattr(
        component_repair,
        "load_component_agent_request",
        lambda _: {
            "candidate_ids": ["component_0009"],
            "evidence": {
                "quality-report.json": {
                    "path": "previous-quality.json",
                    "sha256": hashlib.sha256(previous_payload).hexdigest(),
                },
            },
        },
    )
    state = {
        "repair_round": 2,
        "failed_ids": ["component_0009"],
        "round_history": [
            {"round": 1, "frozen_ids": []},
            {"round": 2, "frozen_ids": []},
        ],
        "current_round": {
            "request_ref": {"path": "current-request.json"},
            "quality_ref": {
                "path": "current-quality.json",
                "sha256": hashlib.sha256(current_payload).hexdigest(),
            },
        },
    }
    store = type("Store", (), {"root": tmp_path})()

    assert component_repair._page_quality_progressed(store, state) is True


def test_execution_rejects_rewritten_retained_inactive_source_mask(
    page_session: dict,
) -> None:
    _, first_freeze, store, session = _record_composite_quality(
        page_session,
        left_box=(2, 2, 8, 8),
        right_box=(18, 28, 24, 34),
    )
    assert first_freeze["failed_ids"] == ["parent"]
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    session = _real_next_round_session(
        session,
        store,
        store.root / state["current_round"]["quality_ref"]["path"],
    )
    second_request = build_component_agent_request(session, repair_round=2)
    record_next_component_request(
        store, "page_001", request_path=second_request
    )
    advance_component_repair(store, "page_001")

    with pytest.raises(ValueError, match="source|provenance|inactive"):
        _execute_composite_quality_round(
            store,
            second_request,
            load_component_agent_graph(second_request),
            action=_action("accept", ["parent"]),
            shape=(30, 40),
            before_execution_record=lambda graph_path: _rewrite_graph_mask_as(
                graph_path, "left", "right"
            ),
        )


def test_execution_rejects_rewritten_retained_inactive_source_role(
    page_session: dict,
) -> None:
    _, first_freeze, store, session = _record_composite_quality(
        page_session,
        left_box=(2, 2, 8, 8),
        right_box=(18, 28, 24, 34),
    )
    assert first_freeze["failed_ids"] == ["parent"]
    state = store.read_json("pages/page_001/reconstruction/component_state.json")
    session = _real_next_round_session(
        session,
        store,
        store.root / state["current_round"]["quality_ref"]["path"],
    )
    second_request = build_component_agent_request(session, repair_round=2)
    record_next_component_request(
        store, "page_001", request_path=second_request
    )
    advance_component_repair(store, "page_001")

    with pytest.raises(ValueError, match="source|provenance|inactive"):
        _execute_composite_quality_round(
            store,
            second_request,
            load_component_agent_graph(second_request),
            action=_action("accept", ["parent"]),
            shape=(30, 40),
            before_execution_record=lambda graph_path: _rewrite_graph_source_role(
                graph_path, "left"
            ),
        )


def test_quality_rejects_absorbed_mask_ancestor_replaced_during_read(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.visual_segment as visual_segment

    real_read = visual_segment._read_action_mask
    real_lstat = Path.lstat
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    replaced_root = None

    class ReparseStatus:
        def __init__(self, wrapped: os.stat_result) -> None:
            self._wrapped = wrapped
            self.st_file_attributes = (
                getattr(wrapped, "st_file_attributes", 0) | reparse_flag
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    def install_swap() -> None:
        def swap_then_read(path: Path, *args, **kwargs):
            nonlocal replaced_root
            if replaced_root is None:
                replaced_root = Path(path).parents[1]
            return real_read(path, *args, **kwargs)

        def flagged_lstat(path: Path) -> object:
            status = real_lstat(path)
            return ReparseStatus(status) if path == replaced_root else status

        monkeypatch.setattr(visual_segment, "_read_action_mask", swap_then_read)
        monkeypatch.setattr(Path, "lstat", flagged_lstat)

    with pytest.raises((RuntimeError, ValueError), match="directory|identity|unsafe|reparse"):
        _record_composite_quality(
            page_session,
            left_box=(2, 2, 8, 8),
            right_box=(18, 28, 24, 34),
            before_quality=install_swap,
        )
    assert replaced_root is not None
    reconstruction = Path(page_session["reconstruction_dir"])
    assert not list(reconstruction.rglob("component-quality.json"))


def test_frozen_mask_keeps_nonstandard_relative_path(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    frozen = next(node for node in graph["nodes"] if node["id"] == "frozen")
    custom = input_dir / "masks/archive/frozen-original.png"
    custom.parent.mkdir()
    (input_dir / frozen["mask"]).replace(custom)
    frozen["mask"] = "masks/archive/frozen-original.png"
    output = tmp_path / "round-custom"
    result = execute_component_actions(
        image, graph, [_action("accept", ["left"])], sam_runner=None,
        input_dir=input_dir, output_dir=output,
    )
    published = next(node for node in result["nodes"] if node["id"] == "frozen")
    assert published == frozen
    assert (output / frozen["mask"]).read_bytes() == custom.read_bytes()


def test_execute_merge_unions_masks_and_inactivates_sources(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    output = tmp_path / "round-02"
    result = execute_component_actions(
        image, graph, [_action("merge", ["left", "right"])], sam_runner=None,
        input_dir=input_dir, output_dir=output,
    )
    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["left"]["state"] == by_id["right"]["state"] == "inactive"
    merged = np.asarray(Image.open(output / by_id["merge_0001"]["mask"])) > 0
    assert int(merged.sum()) == 32
    assert by_id["merge_0001"]["kind"] == "child"
    assert by_id["merge_0001"]["parent_id"] == "parent"


def test_absorb_residual_unions_only_bound_unexplained_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    residual = np.zeros(image.shape[:2], dtype=np.uint8)
    residual[6:8, 2:4] = 255
    residual_path = input_dir / "unexplained-mask.png"
    Image.fromarray(residual, mode="L").save(residual_path)
    request = {"evidence": {"unexplained-mask.png": {
        "path": "unexplained-mask.png",
        "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
    }}}
    (input_dir / "component_agent_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    monkeypatch.setattr(
        component_repair, "load_component_agent_request", lambda _: request,
    )
    left_before = np.asarray(Image.open(
        input_dir / next(node for node in graph["nodes"] if node["id"] == "left")["mask"]
    )) > 0

    result = execute_component_actions(
        image, graph, [_action("absorb_residual", ["left"])], sam_runner=None,
        input_dir=input_dir, output_dir=tmp_path / "round-residual",
    )

    left = next(node for node in result["nodes"] if node["id"] == "left")
    actual = np.asarray(Image.open(tmp_path / "round-residual" / left["mask"])) > 0
    assert np.array_equal(actual, left_before | (residual > 0))


def test_accept_then_absorb_residual_preserves_gate_and_pair_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    residual = np.zeros(image.shape[:2], dtype=np.uint8)
    residual[6:8, 2:4] = 255
    residual_path = input_dir / "unexplained-mask.png"
    Image.fromarray(residual, mode="L").save(residual_path)
    request = {"evidence": {"unexplained-mask.png": {
        "path": "unexplained-mask.png",
        "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
    }}}
    (input_dir / "component_agent_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    monkeypatch.setattr(
        component_repair, "load_component_agent_request", lambda _: request,
    )
    left_before = np.asarray(Image.open(
        input_dir / next(node for node in graph["nodes"] if node["id"] == "left")["mask"]
    )) > 0
    actions = [
        _action("accept", ["left"]),
        _action("absorb_residual", ["left"]),
        _action("accept", ["right"]),
    ]
    for action in (actions[0], actions[2]):
        action["evidence"] = ["left", "right", "independent visual units"]

    result = execute_component_actions(
        image, graph, actions, sam_runner=None,
        input_dir=input_dir, output_dir=tmp_path / "round-accept-residual",
    )

    left = next(node for node in result["nodes"] if node["id"] == "left")
    actual = np.asarray(Image.open(
        tmp_path / "round-accept-residual" / left["mask"]
    )) > 0
    assert left["state"] == "pending_gate"
    assert np.array_equal(actual, left_before | (residual > 0))
    assert component_repair._approved_contained_parent_pairs(
        {"actions": actions}, {("left", "right")}
    ) == {("left", "right")}


def test_absorb_residual_partitions_disconnected_regions_by_nearest_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    residual = np.zeros(image.shape[:2], dtype=np.uint8)
    residual[3:5, 0:2] = 255
    residual[3:5, 14:16] = 255
    residual_path = input_dir / "unexplained-mask.png"
    Image.fromarray(residual, mode="L").save(residual_path)
    request = {"evidence": {"unexplained-mask.png": {
        "path": "unexplained-mask.png",
        "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
    }}}
    (input_dir / "component_agent_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    monkeypatch.setattr(
        component_repair, "load_component_agent_request", lambda _: request,
    )
    before = {
        node["id"]: np.asarray(Image.open(input_dir / node["mask"])) > 0
        for node in graph["nodes"]
        if node["id"] in {"left", "right"}
    }

    result = execute_component_actions(
        image, graph,
        [
            _action("absorb_residual", ["left"]),
            _action("absorb_residual", ["right"]),
        ],
        sam_runner=None, input_dir=input_dir,
        output_dir=tmp_path / "round-partitioned-residual",
    )

    by_id = {node["id"]: node for node in result["nodes"]}
    left = np.asarray(Image.open(
        tmp_path / "round-partitioned-residual" / by_id["left"]["mask"]
    )) > 0
    right = np.asarray(Image.open(
        tmp_path / "round-partitioned-residual" / by_id["right"]["mask"]
    )) > 0
    expected_left = before["left"].copy()
    expected_left[3:5, 0:2] = True
    expected_right = before["right"].copy()
    expected_right[3:5, 14:16] = True
    assert np.array_equal(left, expected_left)
    assert np.array_equal(right, expected_right)


def test_absorb_residual_rejects_region_unrelated_to_every_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    residual = np.zeros(image.shape[:2], dtype=np.uint8)
    residual[10:12, 14:16] = 255
    residual_path = input_dir / "unexplained-mask.png"
    Image.fromarray(residual, mode="L").save(residual_path)
    request = {"evidence": {"unexplained-mask.png": {
            "path": "unexplained-mask.png",
            "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
        }}}
    (input_dir / "component_agent_request.json").write_text(
        json.dumps(request),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        component_repair, "load_component_agent_request", lambda _: request,
    )

    graph_before = copy.deepcopy(graph)
    output_dir = tmp_path / "round-unrelated-residual"
    with pytest.raises(VisualSegmentationError, match="no related residual") as error:
        execute_component_actions(
            image,
            graph,
            [_action("absorb_residual", ["left"])],
            sam_runner=None,
            input_dir=input_dir,
            output_dir=output_dir,
        )

    assert type(error.value).__name__ == "RecoverableComponentPlanError"
    assert graph == graph_before
    assert not output_dir.exists()


def test_absorb_residual_leaves_unmatched_region_for_same_round_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    residual = np.zeros(image.shape[:2], dtype=np.uint8)
    residual[3:5, 0:2] = 255
    residual[3:5, 14:16] = 255
    residual_path = input_dir / "unexplained-mask.png"
    Image.fromarray(residual, mode="L").save(residual_path)
    request = {"evidence": {"unexplained-mask.png": {
        "path": "unexplained-mask.png",
        "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
    }}}
    (input_dir / "component_agent_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    monkeypatch.setattr(
        component_repair, "load_component_agent_request", lambda _: request,
    )
    right = np.asarray(Image.open(input_dir / "masks/right.png")) > 0
    output_dir = tmp_path / "round-absorb-and-retry"

    result = execute_component_actions(
        image,
        graph,
        [
            _action("absorb_residual", ["left"]),
            _action(
                "retry_with_box", ["right"],
                {"box": [0.6, 0.1, 1.0, 0.6]},
            ),
        ],
        sam_batch_runner=lambda **_: [
            {"component_id": "right", "mask": right}
        ],
        input_dir=input_dir,
        output_dir=output_dir,
    )

    by_id = {node["id"]: node for node in result["nodes"]}
    left = np.asarray(Image.open(output_dir / by_id["left"]["mask"])) > 0
    assert np.all(left[3:5, 0:2])
    assert not np.any(left[3:5, 14:16])
    assert by_id["right"]["state"] == "pending"


def test_skill_absorb_residual_validates_bound_request_without_product_import(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    skill_root = Path(__file__).parents[1] / "skills" / "image-to-ppt"
    script = """
import builtins
from pathlib import Path
from scripts import visual_segment
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name.startswith('image2editable'):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
mask = visual_segment._read_bound_residual_mask(Path(__import__('sys').argv[1]).parent, (2, 2))
assert mask.shape == (2, 2)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(request_path)],
        cwd=skill_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    marker_path = request_path.parent / "publication-marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["hmac_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    rejected = subprocess.run(
        [sys.executable, "-c", script, str(request_path)],
        cwd=skill_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert rejected.returncode != 0
    assert "publication signature mismatch" in rejected.stderr


def test_split_without_connected_proposals_fails_without_output(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    output = tmp_path / "round-02"
    with pytest.raises(
        RecoverableComponentPlanError, match="connected proposals"
    ) as error:
        execute_component_actions(
            image, graph, [_action("split", ["left"], {"parts": 2})], sam_runner=None,
            input_dir=input_dir, output_dir=output,
        )
    assert error.value.reason == "invalid_split_target"
    assert not output.exists()


def test_split_rejects_extra_connected_proposals_instead_of_losing_pixels(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    left = next(node for node in graph["nodes"] if node["id"] == "left")
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[1:3, 1:3] = mask[5:7, 5:7] = mask[9:11, 9:11] = 255
    path = input_dir / left["mask"]
    Image.fromarray(mask).save(path)
    left["mask_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    left["bbox"] = [1, 1, 11, 11]
    output = tmp_path / "round-extra-parts"
    with pytest.raises(
        RecoverableComponentPlanError, match="exact connected proposals"
    ) as error:
        execute_component_actions(
            image, graph, [_action("split", ["left"], {"parts": 2})], sam_runner=None,
            input_dir=input_dir, output_dir=output,
        )
    assert error.value.reason == "invalid_split_target"
    assert not output.exists()


def test_split_two_connected_proposals_preserves_pixels_and_layer(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    left = next(node for node in graph["nodes"] if node["id"] == "left")
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[1:3, 1:3] = mask[7:10, 8:11] = 255
    path = input_dir / left["mask"]
    Image.fromarray(mask).save(path)
    left["mask_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    left["bbox"] = [1, 1, 11, 10]
    output = tmp_path / "round-split"
    result = execute_component_actions(
        image, graph, [_action("split", ["left"], {"parts": 2})], sam_runner=None,
        input_dir=input_dir, output_dir=output,
    )
    by_id = {node["id"]: node for node in result["nodes"]}
    children = [node for node in result["nodes"] if node["id"].startswith("split_")]
    union = np.zeros(image.shape[:2], dtype=bool)
    for child in children:
        payload = (output / child["mask"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == child["mask_sha256"]
        union |= np.asarray(Image.open(output / child["mask"])) > 0
        assert child["kind"] == "child" and child["parent_id"] == "parent"
    assert by_id["left"]["state"] == "inactive"
    assert len(children) == 2
    assert np.array_equal(union, mask > 0)


def test_action_failure_does_not_delete_replacement_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    output = tmp_path / "round-replaced"
    real_save = Image.Image.save
    replacement = None

    def replace_staging_then_fail(self: Image.Image, path: object, *args: object, **kwargs: object) -> None:
        nonlocal replacement
        staging = Path(path).parent.parent
        owned = staging.with_name(staging.name + "-owned")
        staging.rename(owned)
        staging.mkdir()
        replacement = staging / "attacker.txt"
        replacement.write_text("keep", encoding="utf-8")
        raise OSError("simulated save failure")

    monkeypatch.setattr(Image.Image, "save", replace_staging_then_fail)
    with pytest.raises(OSError, match="save failure"):
        execute_component_actions(
            image, graph, [_action("accept", ["left"])], sam_runner=None,
            input_dir=input_dir, output_dir=output,
        )
    assert replacement is not None and replacement.read_text(encoding="utf-8") == "keep"
    monkeypatch.setattr(Image.Image, "save", real_save)


def test_sam_prompt_coordinates_and_attach_text_do_not_merge_pixels(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    calls = []
    def runner(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.asarray(Image.open(input_dir / "masks/left.png")) > 0
    output = tmp_path / "round-02"
    result = execute_component_actions(
        image, graph,
        [_action("retry_with_points", ["left"], {"positive": [[1.0, 1.0]], "negative": [[0.0, 0.0]]}),
         _action("attach_text", ["right", "text"])],
        sam_runner=runner, input_dir=input_dir, output_dir=output,
    )
    assert calls[0]["positive"] == [[15.0, 11.0]]
    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["right"]["text_ids"] == ["text"]
    assert int((np.asarray(Image.open(output / by_id["right"]["mask"])) > 0).sum()) == 16


def test_prompt_retry_promotes_small_visual_from_overbroad_parent(
    tmp_path: Path,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)

    result = execute_component_actions(
        image, graph,
        [_action("retry_with_points", ["left"], {
            "positive": [[0.2, 0.2]], "negative": [[0.8, 0.8]],
            "independent": True,
        })],
        sam_runner=lambda **kwargs: np.asarray(
            Image.open(input_dir / "masks/left.png")
        ) > 0,
        input_dir=input_dir,
        output_dir=tmp_path / "round-promote-small-retry",
    )

    node = next(node for node in result["nodes"] if node["id"] == "left")
    assert node["kind"] == "parent"
    assert node["parent_id"] is None


def test_expand_stays_inside_parent_and_collapse_activates_parent(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    expanded_dir = tmp_path / "round-02"
    expanded = execute_component_actions(
        image, graph, [_action("expand", ["left"], {"margin_ratio": 0.2})],
        sam_runner=None, input_dir=input_dir, output_dir=expanded_dir,
    )
    by_id = {node["id"]: node for node in expanded["nodes"]}
    expanded_mask = np.asarray(Image.open(expanded_dir / by_id["left"]["mask"])) > 0
    parent_mask = np.asarray(Image.open(input_dir / "masks/parent.png")) > 0
    assert not np.any(expanded_mask & ~parent_mask)
    collapsed_dir = tmp_path / "round-03"
    collapsed = execute_component_actions(
        image, expanded, [_action("collapse_to_parent", ["parent"])],
        sam_runner=None, input_dir=expanded_dir, output_dir=collapsed_dir,
    )
    states = {node["id"]: node["state"] for node in collapsed["nodes"]}
    assert states["parent"] == "pending"
    assert states["left"] == states["right"] == "inactive"


def test_action_margin_uses_page_short_edge_for_different_component_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    right = next(node for node in graph["nodes"] if node["id"] == "right")
    right_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    right_mask[1:9, 7:15] = 255
    right_path = input_dir / right["mask"]
    Image.fromarray(right_mask).save(right_path)
    right["mask_sha256"] = hashlib.sha256(right_path.read_bytes()).hexdigest()
    right["bbox"] = [7, 1, 15, 9]
    kernel_sizes = []
    real_kernel = cv2.getStructuringElement

    def record_kernel(shape: int, size: tuple[int, int]) -> np.ndarray:
        kernel_sizes.append(size)
        return real_kernel(shape, size)

    monkeypatch.setattr(cv2, "getStructuringElement", record_kernel)
    execute_component_actions(
        image, graph, [_action("expand", ["left"], {"margin_ratio": 0.25})],
        sam_runner=None, input_dir=input_dir, output_dir=tmp_path / "round-margin-left",
    )
    execute_component_actions(
        image, graph, [_action("expand", ["right"], {"margin_ratio": 0.25})],
        sam_runner=None, input_dir=input_dir, output_dir=tmp_path / "round-margin-right",
    )
    assert kernel_sizes == [(7, 7), (7, 7)]


def test_shrink_uses_page_margin_and_publishes_nonempty_mask(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    result = execute_component_actions(
        image, graph, [_action("shrink", ["left"], {"margin_ratio": 0.1})],
        sam_runner=None, input_dir=input_dir, output_dir=tmp_path / "round-shrink",
    )
    left = next(node for node in result["nodes"] if node["id"] == "left")
    mask = np.asarray(Image.open(tmp_path / "round-shrink" / left["mask"])) > 0
    assert 0 < int(mask.sum()) < 16


def test_publish_action_directory_never_replaces_existing_empty_target(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "target"
    staging.mkdir()
    (staging / "component-graph.json").write_text("new", encoding="utf-8")
    target.mkdir()
    with pytest.raises(FileExistsError):
        _publish_action_directory(staging, target)
    assert target.is_dir() and not list(target.iterdir())
    assert (staging / "component-graph.json").read_text(encoding="utf-8") == "new"


def test_multi_action_round_is_validated_before_mutation(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    result = execute_component_actions(
        image,
        graph,
        [_action("merge", ["left", "right"]), _action("collapse_to_parent", ["parent"])],
        sam_runner=None,
        input_dir=input_dir,
        output_dir=tmp_path / "round-batch",
    )
    by_id = {node["id"]: node for node in result["nodes"]}
    assert by_id["parent"]["state"] == "pending"


def test_invalid_later_action_is_rejected_before_sam_side_effect(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    calls = []

    def runner(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.ones(image.shape[:2], dtype=bool)

    invalid = _action("accept", ["text"])
    with pytest.raises(ValueError, match="text kind"):
        execute_component_actions(
            image, graph,
            [_action("retry_with_box", ["left"], {"box": [0.1, 0.1, 0.9, 0.9]}), invalid],
            sam_runner=runner, input_dir=input_dir, output_dir=tmp_path / "round-invalid",
        )
    assert calls == []


def test_retry_with_box_can_reconsider_inactive_visual(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    left = next(node for node in graph["nodes"] if node["id"] == "left")
    left["state"] = "inactive"
    proposed = np.asarray(Image.open(input_dir / left["mask"])) > 0

    result = execute_component_actions(
        image, graph,
        [_action("retry_with_box", ["left"], {"box": [0.1, 0.1, 0.9, 0.9]})],
        sam_runner=lambda **_: proposed,
        input_dir=input_dir, output_dir=tmp_path / "round-inactive",
    )

    reconsidered = next(node for node in result["nodes"] if node["id"] == "left")
    assert reconsidered["state"] == "pending"


def test_retry_with_box_completes_subtle_antialias_edges(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    image[1:11, 2:12] = 20
    image[2:10, 3:11] = 80
    proposed = np.zeros(image.shape[:2], dtype=bool)
    proposed[2:10, 3:11] = True
    output = tmp_path / "round-retry-antialias"

    result = execute_component_actions(
        image,
        graph,
        [_action("retry_with_box", ["left"], {"box": [0.1, 0.1, 0.9, 0.9]})],
        sam_runner=lambda **_: proposed,
        input_dir=input_dir,
        output_dir=output,
    )

    left = next(node for node in result["nodes"] if node["id"] == "left")
    stored = np.asarray(Image.open(output / left["mask"])) > 0
    assert np.all(stored[1:11, 2:12])


def test_absorb_into_parent_completes_subtle_antialias_edges(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    image[1:11, 2:14] = 20
    image[2:10, 3:13] = 80
    for component_id, x0, x1 in (("parent", 3, 7), ("left", 7, 10), ("right", 10, 13)):
        node = next(node for node in graph["nodes"] if node["id"] == component_id)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[2:10, x0:x1] = 255
        path = input_dir / node["mask"]
        Image.fromarray(mask).save(path)
        node["mask_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        node["bbox"] = [x0, 2, x1, 10]
    output = tmp_path / "round-absorb-antialias"

    result = execute_component_actions(
        image,
        graph,
        [_action("absorb_into_parent", ["parent", "left", "right"])],
        sam_runner=None,
        input_dir=input_dir,
        output_dir=output,
    )

    updated = next(node for node in result["nodes"] if node["id"] == "parent")
    stored = np.asarray(Image.open(output / updated["mask"])) > 0
    assert np.all(stored[1:11, 2:14])


def test_retried_inactive_visual_can_rebuild_background_in_same_round(
    tmp_path: Path,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    left = next(node for node in graph["nodes"] if node["id"] == "left")
    left["state"] = "inactive"
    proposed = np.asarray(Image.open(input_dir / left["mask"])) > 0

    result = execute_component_actions(
        image,
        graph,
        [
            _action(
                "retry_with_box", ["left"], {"box": [0.1, 0.1, 0.9, 0.9]},
            ),
            _action(
                "rebuild_background", ["left"], {"margin_ratio": 0.01},
            ),
        ],
        sam_runner=lambda **_: proposed,
        input_dir=input_dir,
        output_dir=tmp_path / "round-retry-background",
    )

    assert next(node for node in result["nodes"] if node["id"] == "left")[
        "state"
    ] == "pending"


def test_non_retry_action_cannot_reactivate_inactive_visual(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    left = next(node for node in graph["nodes"] if node["id"] == "left")
    left["state"] = "inactive"

    with pytest.raises(ValueError, match="pending component"):
        execute_component_actions(
            image, graph, [_action("accept", ["left"])], sam_runner=None,
            input_dir=input_dir, output_dir=tmp_path / "round-inactive-accept",
        )


def test_retry_box_maps_normalized_page_coordinates(tmp_path: Path) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    calls = []
    def runner(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.asarray(Image.open(input_dir / "masks/left.png")) > 0
    execute_component_actions(
        image, graph,
        [_action("retry_with_box", ["left"], {"box": [0.25, 0.25, 0.75, 0.75]})],
        sam_runner=runner, input_dir=input_dir, output_dir=tmp_path / "round-02",
    )
    assert calls[0]["box"] == [4.0, 3.0, 12.0, 9.0]


def test_retry_outside_original_semantic_parent_promotes_exact_mask_to_top_level(
    tmp_path: Path,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    parent = next(node for node in graph["nodes"] if node["id"] == "parent")
    parent_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    parent_mask[1:7, 1:7] = 255
    parent_path = input_dir / parent["mask"]
    Image.fromarray(parent_mask).save(parent_path)
    parent["mask_sha256"] = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    parent["bbox"] = [1, 1, 7, 7]
    proposed = np.zeros(image.shape[:2], dtype=bool)
    proposed[8:11, 12:15] = True

    output = tmp_path / "round-retry-detached"
    result = execute_component_actions(
        image,
        graph,
        [_action("retry_with_box", ["left"], {"box": [0.7, 0.6, 1.0, 1.0]})],
        sam_runner=lambda **_: proposed,
        input_dir=input_dir,
        output_dir=output,
    )

    retried = next(node for node in result["nodes"] if node["id"] == "left")
    stored = np.asarray(Image.open(output / retried["mask"])) > 0
    assert retried["kind"] == "parent"
    assert retried["parent_id"] is None
    assert retried["z_index"] == 1
    assert np.array_equal(stored, proposed)


def test_retry_inside_original_semantic_parent_keeps_child_relationship(
    tmp_path: Path,
) -> None:
    image, graph, input_dir = _action_case(tmp_path)
    proposed = np.zeros(image.shape[:2], dtype=bool)
    proposed[3:6, 3:6] = True

    result = execute_component_actions(
        image,
        graph,
        [_action("retry_with_points", ["left"], {"positive": [[0.2, 0.3]], "negative": []})],
        sam_runner=lambda **_: proposed,
        input_dir=input_dir,
        output_dir=tmp_path / "round-retry-contained",
    )

    retried = next(node for node in result["nodes"] if node["id"] == "left")
    assert retried["kind"] == "child"
    assert retried["parent_id"] == "parent"


def test_retry_outside_semantic_parent_builds_bound_presentation_assets(
    tmp_path: Path,
) -> None:
    import types
    from image2editable import legacy

    image, graph, input_dir = _action_case(tmp_path)
    by_id = {node["id"]: node for node in graph["nodes"]}
    by_id["right"]["state"] = "inactive"
    by_id["frozen"]["state"] = "inactive"
    parent_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    parent_mask[1:7, 1:7] = 255
    parent_path = input_dir / by_id["parent"]["mask"]
    Image.fromarray(parent_mask).save(parent_path)
    by_id["parent"]["mask_sha256"] = hashlib.sha256(
        parent_path.read_bytes()
    ).hexdigest()
    by_id["parent"]["bbox"] = [1, 1, 7, 7]
    proposed = np.zeros(image.shape[:2], dtype=bool)
    proposed[8:11, 12:15] = True
    output = tmp_path / "round-retry-presentation"
    result = execute_component_actions(
        image,
        graph,
        [_action("retry_with_box", ["left"], {"box": [0.7, 0.6, 1.0, 1.0]})],
        sam_runner=lambda **_: proposed,
        input_dir=input_dir,
        output_dir=output,
    )
    retried = next(node for node in result["nodes"] if node["id"] == "left")
    graph_path = output / "component-graph.json"
    graph_payload = graph_path.read_bytes()
    source = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    source_path = tmp_path / "source.png"
    Image.fromarray(source, mode="RGB").save(source_path)

    manifest_path = legacy._build_presentation_assets(
        types.SimpleNamespace(root=tmp_path),
        source_path=source_path,
        text_clean_path=source_path,
        graph_path=graph_path,
        output_dir=output,
    )

    assert graph_path.read_bytes() == graph_payload
    assert retried["z_index"] == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["graph_sha256"] == hashlib.sha256(graph_payload).hexdigest()
    assert manifest["components"][0]["component_id"] == "left"
    entry = manifest["components"][0]
    decoded = {}
    for name in (
        "rgba", "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask",
    ):
        reference = entry[name]
        asset_path = tmp_path / Path(reference["path"])
        payload = asset_path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == reference["sha256"]
        with Image.open(asset_path) as stored:
            decoded[name] = np.asarray(stored.convert(
                "RGBA" if name == "rgba" else "L"
            )).copy()
    assert np.array_equal(decoded["ownership_mask"] > 0, proposed)
    assert np.array_equal(decoded["presentation_alpha_mask"] > 0, proposed)
    assert not np.any(decoded["generated_underlay_mask"])
    assert np.array_equal(decoded["rgba"][:, :, 3] > 0, proposed)
    assert np.array_equal(decoded["rgba"][proposed, :3], source[proposed])

    quality = component_repair.evaluate_component_quality_round(
        source, source, source, result,
        graph_dir=output, trusted_root=tmp_path,
        text_mask=proposed,
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
        initial_component_count=1, expected_component_ids=["left"],
        presentation_layers=[{
            "component_id": "left",
            "ownership_mask": decoded["ownership_mask"],
            "presentation_alpha_mask": decoded["presentation_alpha_mask"],
            "generated_underlay_mask": decoded["generated_underlay_mask"],
            "metrics": entry["metrics"],
        }],
    )
    component_report = quality["component_reports"][0]
    assert component_report["metrics"]["component_pixels"] == 0
    assert component_report["violations"] == ["empty_component"]
    assert component_report["accepted"] is False


def test_presentation_assets_use_refined_text_mask_instead_of_ocr_box(
    tmp_path: Path,
) -> None:
    import types
    from image2editable import legacy

    height, width = 24, 48
    source = np.zeros((height, width, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(width, dtype=np.uint8) * 4
    source[:, :, 1] = 90
    source[:, :, 2] = 40
    visual = np.zeros((height, width), dtype=bool)
    visual[4:20, 4:44] = True
    ocr_box = np.zeros_like(visual)
    ocr_box[8:16, 14:34] = True
    refined = np.zeros_like(visual)
    refined[10:14, 20:28] = True

    graph_dir = tmp_path / "graph"
    masks_dir = graph_dir / "masks"
    masks_dir.mkdir(parents=True)
    nodes = []
    for component_id, kind, mask, bbox, z_index in (
        ("visual", "parent", visual, [4, 4, 44, 20], 0),
        ("text", "text", ocr_box, [14, 8, 34, 16], 1),
    ):
        path = masks_dir / f"{component_id}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        nodes.append({
            "id": component_id,
            "kind": kind,
            "parent_id": None,
            "state": "pending" if kind == "parent" else "frozen",
            "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bbox": bbox,
            "z_index": z_index,
            "text_ids": ["text"] if kind == "parent" else [],
        })
    graph_path = graph_dir / "component-graph.json"
    graph_path.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")
    source_path = tmp_path / "source.png"
    Image.fromarray(source, mode="RGB").save(source_path)
    refined_path = tmp_path / "refined-text-mask.png"
    Image.fromarray(refined.astype(np.uint8) * 255).save(refined_path)

    manifest_path = legacy._build_presentation_assets(
        types.SimpleNamespace(root=tmp_path),
        source_path=source_path,
        text_clean_path=source_path,
        text_mask_path=refined_path,
        graph_path=graph_path,
        output_dir=graph_dir,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated_path = tmp_path / manifest["components"][0][
        "generated_underlay_mask"
    ]["path"]
    generated = np.asarray(Image.open(generated_path).convert("L")) > 0
    assert np.all(generated[refined])
    assert not np.any(generated[ocr_box & ~refined])


def test_opaque_mask_completion_repairs_solid_holes_but_keeps_line_art() -> None:
    from scripts.visual_segment import _complete_opaque_mask_regions

    solid = np.zeros((40, 80), dtype=bool)
    solid[5:35, 5:35] = True
    solid[12:28, 12:28:3] = False
    solid[14:20, 5] = False
    line_art = np.zeros_like(solid)
    line_art = cv2.circle(
        np.zeros_like(solid, dtype=np.uint8), (58, 20), 12, 1, 2
    ).astype(bool)
    source = solid | line_art

    completed = _complete_opaque_mask_regions(source)

    assert np.all(completed[5:35, 5:35])
    assert np.array_equal(completed[:, 40:], line_art[:, 40:])


def test_opaque_mask_completion_restores_colored_node_on_smooth_canvas() -> None:
    from scripts.visual_segment import _complete_opaque_mask_regions

    image = np.full((80, 120, 3), 248, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.line(mask, (20, 40), (100, 40), 1, 3)
    cv2.circle(mask, (60, 40), 12, 1, 3)
    cv2.line(image, (20, 40), (100, 40), (20, 70, 140), 3)
    cv2.circle(image, (60, 40), 12, (40, 130, 220), cv2.FILLED)

    completed = _complete_opaque_mask_regions(mask > 0, image)

    assert completed[40, 60]
    assert not completed[20, 60]


def test_opaque_mask_completion_restores_subtle_antialias_edge() -> None:
    from scripts.visual_segment import _complete_opaque_mask_regions

    image = np.full((60, 100, 3), 248, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    image[28:33, 20:81] = (242, 244, 246)
    image[29:32, 20:81] = (20, 70, 140)
    mask[29:32, 20:81] = 1

    completed = _complete_opaque_mask_regions(mask > 0, image)

    assert np.all(completed[28, 20:81])
    assert np.all(completed[32, 20:81])


def test_opaque_mask_completion_preserves_legitimate_transparent_hole() -> None:
    from scripts.visual_segment import _complete_opaque_mask_regions

    image = np.full((70, 70, 3), 250, dtype=np.uint8)
    image[10:60, 10:60] = (30, 110, 210)
    cv2.circle(image, (35, 35), 8, (250, 250, 250), cv2.FILLED)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[10:60, 10:60] = 1
    cv2.circle(mask, (35, 35), 8, 0, cv2.FILLED)

    completed = _complete_opaque_mask_regions(mask > 0, image)

    assert not completed[35, 35]


def test_opaque_mask_completion_does_not_absorb_touching_different_color() -> None:
    from scripts.visual_segment import _complete_opaque_mask_regions

    image = np.full((60, 100, 3), 248, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.line(mask, (20, 30), (60, 30), 1, 3)
    cv2.line(image, (20, 30), (60, 30), (30, 90, 210), 3)
    image[27:34, 62:66] = (60, 120, 180)

    completed = _complete_opaque_mask_regions(mask > 0, image)

    assert not np.any(
        completed[27:34, 62:66] & ~(mask[27:34, 62:66] > 0)
    )


def test_opaque_mask_completion_follows_contact_continuity_across_gradient() -> None:
    from scripts.visual_segment import _complete_opaque_mask_regions

    image = np.full((80, 120, 3), 248, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.line(mask, (20, 40), (100, 40), 1, 3)
    cv2.line(image, (20, 40), (100, 40), (20, 70, 140), 3)
    for y in range(33, 48):
        color = (40 + (y - 33), 120 + (y - 33), 200)
        cv2.line(image, (58, y), (62, y), color, 1)

    completed = _complete_opaque_mask_regions(mask > 0, image)

    assert completed[34, 60]
    assert completed[46, 60]


def test_sam_worker_component_prompt_selects_best_mask_and_can_run_twice() -> None:
    class Predictor:
        def set_image(self, image: np.ndarray) -> None:
            self.image = image
        def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
            masks = np.zeros((2, 4, 5), dtype=bool)
            masks[1, 1:3, 1:4] = True
            return masks, np.asarray([0.1, 0.9]), None
    generator = type("Generator", (), {"predictor": Predictor()})()
    prompt = {"box": [1, 1, 4, 3], "positive": [], "negative": []}
    first = component_prompt_mask(generator, np.zeros((4, 5, 3), dtype=np.uint8), prompt)
    second = component_prompt_mask(generator, np.zeros((4, 5, 3), dtype=np.uint8), prompt)
    assert np.array_equal(first, second)
    assert int(first.sum()) == 6


def test_component_prompt_batch_sets_source_image_once_and_preserves_order() -> None:
    from scripts.sam_worker import component_prompt_masks

    class Predictor:
        def __init__(self) -> None:
            self.set_image_calls = 0

        def set_image(self, image: np.ndarray) -> None:
            self.set_image_calls += 1

        def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
            masks = np.zeros((1, 4, 5), dtype=bool)
            box = kwargs["box"]
            column = int(np.asarray(box)[0]) if box is not None else 0
            masks[0, 1:3, column:column + 1] = True
            return masks, np.asarray([0.9]), None

    predictor = Predictor()
    generator = type("Generator", (), {"predictor": predictor})()
    masks = component_prompt_masks(
        generator,
        np.zeros((4, 5, 3), dtype=np.uint8),
        [
            {"component_id": "left", "box": [1, 1, 2, 3], "positive": [], "negative": []},
            {"component_id": "right", "box": [3, 1, 4, 3], "positive": [], "negative": []},
        ],
    )

    assert predictor.set_image_calls == 1
    assert [int(np.where(mask)[1].min()) for mask in masks] == [1, 3]


def test_component_prompt_batch_worker_entry_loads_generator_once_and_publishes_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import sam_worker

    image_path = tmp_path / "image.png"
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(image_path)
    prompts = [
        {"component_id": "left", "box": [1, 1, 2, 3], "positive": [], "negative": []},
        {"component_id": "right", "box": [3, 1, 4, 3], "positive": [], "negative": []},
    ]
    request_path.write_text(json.dumps({
        "schema_version": sam_worker._BATCH_SCHEMA_VERSION,
        "image": image_path.name,
        "prompts": prompts,
    }), encoding="utf-8")
    loads = []

    class Predictor:
        def set_image(self, image: np.ndarray) -> None:
            self.image = image

        def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
            mask = np.zeros((1, 4, 5), dtype=bool)
            column = int(np.asarray(kwargs["box"])[0])
            mask[0, 1:3, column] = True
            return mask, np.asarray([0.9]), None

    generator = type("Generator", (), {"predictor": Predictor()})()
    monkeypatch.setattr(
        sam_worker,
        "_load_tools",
        lambda: (None, lambda *args, **kwargs: loads.append(1) or generator, None, None,
                 lambda: tmp_path / "sam.pt", None, None),
    )
    published = []
    monkeypatch.setattr(
        sam_worker,
        "_write_bound_json_result",
        lambda binding, payload, limit: published.append(payload),
    )

    assert sam_worker._run_component_prompt_batch(request_path, result_path) == 0
    assert loads == [1]
    assert [record["component_id"] for record in published[0]] == ["left", "right"]


def test_component_sam_batch_subprocess_runner_uses_one_worker_and_cleans_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import sam_worker

    calls = []
    monkeypatch.setattr(sam_worker, "sam_candidate_batch_output_supported", lambda _: True)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((command, kwargs))
        result_path = Path(command[command.index("--result") + 1])
        first = np.zeros((4, 5), dtype=bool)
        first[1:3, 1] = True
        second = np.zeros((4, 5), dtype=bool)
        second[1:3, 3] = True
        result_path.write_text(
            json.dumps([
                {"component_id": "left", **sam_worker._mask_record(first)},
                {"component_id": "right", **sam_worker._mask_record(second)},
            ]),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sam_worker, "run_isolated_worker", fake_run)
    prompts = [
        {"component_id": "left", "box": [1, 1, 2, 3], "positive": [], "negative": []},
        {"component_id": "right", "box": [3, 1, 4, 3], "positive": [], "negative": []},
    ]
    masks = sam_worker.run_component_prompt_batch_worker(
        np.zeros((4, 5, 3), dtype=np.uint8),
        prompts,
        work_dir=tmp_path,
    )

    assert len(calls) == 1
    assert calls[0][0][calls[0][0].index("--mode") + 1] == "component_batch"
    assert calls[0][1]["timeout"] == 600
    assert [int(np.where(mask)[1].min()) for mask in masks] == [1, 3]
    assert not list(tmp_path.glob("component-sam-batch-*"))


def test_component_sam_batch_rejects_mask_shape_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import sam_worker

    monkeypatch.setattr(sam_worker, "sam_candidate_batch_output_supported", lambda _: True)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        result_path = Path(command[command.index("--result") + 1])
        result_path.write_text(json.dumps([{
            "component_id": "left",
            "mask": "AA==",
            "mask_shape": [1_000_000_000, 1_000_000_000],
        }]), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sam_worker, "run_isolated_worker", fake_run)
    monkeypatch.setattr(
        np,
        "unpackbits",
        lambda *args, **kwargs: pytest.fail("invalid shape must be rejected before decode"),
    )

    with pytest.raises(RuntimeError, match="shape"):
        sam_worker.run_component_prompt_batch_worker(
            np.zeros((4, 5, 3), dtype=np.uint8),
            [{
                "component_id": "left",
                "box": [1, 1, 2, 3],
                "positive": [],
                "negative": [],
            }],
            work_dir=tmp_path,
        )


def test_component_sam_batch_over_capacity_uses_single_workers_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import sam_worker

    monkeypatch.setattr(sam_worker, "sam_candidate_batch_output_supported", lambda _: True)
    monkeypatch.setattr(
        sam_worker,
        "run_isolated_worker",
        lambda *args, **kwargs: pytest.fail("over-capacity batch must not spawn"),
    )
    calls = []

    def single(image: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[0, 0] = True
        return mask

    monkeypatch.setattr(sam_worker, "run_component_prompt_worker", single)
    prompts = [
        {
            "component_id": f"component_{index:04d}",
            "box": [0, 0, 1, 1],
            "positive": [],
            "negative": [],
        }
        for index in range(sam_worker._COMPONENT_BATCH_MAX_PROMPTS + 1)
    ]

    masks = sam_worker.run_component_prompt_batch_worker(
        np.zeros((2, 2, 3), dtype=np.uint8), prompts, work_dir=tmp_path
    )

    assert len(calls) == len(prompts)
    assert len(masks) == len(prompts)


def test_component_sam_batch_oversize_request_uses_single_workers_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import sam_worker

    monkeypatch.setattr(sam_worker, "sam_candidate_batch_output_supported", lambda _: True)
    monkeypatch.setattr(
        sam_worker,
        "run_isolated_worker",
        lambda *args, **kwargs: pytest.fail("oversize batch must not spawn"),
    )
    calls = []

    def single(image: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[0, 0] = True
        return mask

    monkeypatch.setattr(sam_worker, "run_component_prompt_worker", single)
    prompts = [
        {
            "component_id": f"component_{index:04d}_" + "x" * 220,
            "box": [0, 0, 1, 1],
            "positive": [],
            "negative": [],
        }
        for index in range(220)
    ]

    masks = sam_worker.run_component_prompt_batch_worker(
        np.zeros((2, 2, 3), dtype=np.uint8), prompts, work_dir=tmp_path
    )

    assert len(calls) == len(prompts)
    assert len(masks) == len(prompts)


def test_component_sam_batch_unsupported_uses_equal_quality_single_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import sam_worker

    calls = []
    monkeypatch.setattr(sam_worker, "sam_candidate_batch_output_supported", lambda _: False)

    def single(image: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[1:3, len(calls)] = True
        return mask

    monkeypatch.setattr(sam_worker, "run_component_prompt_worker", single)
    prompts = [
        {"component_id": "left", "box": [1, 1, 2, 3], "positive": [], "negative": []},
        {"component_id": "right", "box": None, "positive": [[3, 2]], "negative": []},
    ]
    masks = sam_worker.run_component_prompt_batch_worker(
        np.zeros((4, 5, 3), dtype=np.uint8),
        prompts,
        work_dir=tmp_path,
    )

    assert len(calls) == 2
    assert [int(np.where(mask)[1].min()) for mask in masks] == [1, 2]


def test_component_sam_subprocess_runner_reads_result_and_cleans_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import sam_worker

    calls = []

    def run(command: list[str], **kwargs: object) -> None:
        calls.append((command, kwargs))
        result = Path(command[command.index("--result") + 1])
        mask = np.zeros((4, 5), dtype=bool)
        mask[1:3, 1:4] = True
        packed = np.packbits(mask, axis=None).tobytes()
        import base64
        result.write_text(json.dumps([{
            "mask": base64.b64encode(packed).decode("ascii"),
            "mask_shape": [4, 5],
        }]), encoding="utf-8")

    monkeypatch.setattr(
        sam_worker,
        "run_isolated_worker",
        run,
        raising=False,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "component SAM must use the shared runner"
        ),
    )
    mask = run_component_prompt_worker(
        np.zeros((4, 5, 3), dtype=np.uint8),
        box=[1, 1, 4, 3], positive=[], negative=[], work_dir=tmp_path,
    )
    assert int(mask.sum()) == 6
    assert calls[0][1]["check"] is True
    assert calls[0][1]["timeout"] == 600
    assert not list(tmp_path.glob("component-sam-*"))


@pytest.fixture
def page_session(tmp_path: Path) -> dict:
    return _make_page_session(tmp_path, "page_001")


def _write_test_presentation_manifest(
    run_root: Path,
    output_dir: Path,
    *,
    source_sha256: str,
    graph_path: Path | None = None,
    graph_sha256: str | None = None,
) -> Path:
    if graph_path is not None:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph_sha256 = hashlib.sha256(graph_path.read_bytes()).hexdigest()
        active_nodes = [
            node for node in graph["nodes"]
            if node["kind"] != "text"
            and node["state"] in {"pending", "pending_gate", "frozen"}
        ]
        raw_masks = []
        for node in active_nodes:
            with Image.open(graph_path.parent / node["mask"]) as image:
                raw_masks.append(np.asarray(image.convert("L")) > 0)
        ownership_masks = component_quality.resolve_visual_mask_ownership(
            active_nodes, raw_masks
        )
        component_ids = [node["id"] for node in active_nodes]
        ownership_by_id = dict(zip(component_ids, ownership_masks, strict=True))
    else:
        component_ids = []
        ownership_by_id = {}
    assert graph_sha256 is not None
    assets_dir = output_dir / f"presentation-assets-{graph_sha256[:12]}"
    assets_dir.mkdir(exist_ok=True)
    components = []
    for index, component_id in enumerate(component_ids, start=1):
        ownership = ownership_by_id[component_id]
        generated = np.zeros_like(ownership)
        alpha = ownership | generated
        paths = {}
        for name in (
            "rgba", "ownership-mask", "presentation-alpha-mask",
            "generated-underlay-mask",
        ):
            path = assets_dir / f"{index:04d}-{name}.png"
            mode = "RGBA" if name == "rgba" else "L"
            if mode == "RGBA":
                array = np.zeros((*ownership.shape, 4), dtype=np.uint8)
                array[:, :, 3] = alpha.astype(np.uint8) * 255
            else:
                masks = {
                    "ownership-mask": ownership,
                    "presentation-alpha-mask": alpha,
                    "generated-underlay-mask": generated,
                }
                array = masks[name].astype(np.uint8) * 255
            Image.fromarray(array, mode=mode).save(path)
            paths[name.replace("-", "_")] = {
                "path": path.relative_to(run_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        components.append({
            "component_id": component_id,
            **paths,
            "metrics": {
                "boundary_color_mae": 0.0,
                "gradient_jump_p95": 0.0,
                "added_high_frequency_pixels": 0.0,
            },
        })
    manifest_path = output_dir / "presentation-manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "source_sha256": source_sha256,
        "graph_sha256": graph_sha256,
        "components": components,
    }), encoding="utf-8")
    return manifest_path


def _refresh_test_presentation_manifest(page_session: dict) -> None:
    evidence = page_session["evidence"]
    manifest_path = Path(evidence["presentation-manifest.json"])
    graph_path = Path(evidence["component-graph.json"])
    source_path = Path(evidence["source.png"])
    if graph_path.stat().st_size > component_repair.GRAPH_JSON_LIMIT:
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError):
        return
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    graph_sha256 = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    if (
        manifest["source_sha256"] != source_sha256
        or manifest["graph_sha256"] != graph_sha256
    ):
        _write_test_presentation_manifest(
            Path(page_session["reconstruction_dir"]).parents[2],
            manifest_path.parent,
            source_sha256=source_sha256,
            graph_path=graph_path,
        )


def _prepare_round_two_review_session(page_session: dict) -> None:
    evidence = page_session["evidence"]
    size = (12, 12)
    for name, path in evidence.items():
        if Path(path).suffix == ".png":
            Image.fromarray(np.zeros(size, dtype=np.uint8)).save(path)
    Image.fromarray(np.full(size, 220, dtype=np.uint8)).save(evidence["source.png"])
    masks = Path(evidence["component-graph.json"]).parent / "masks"
    nodes = []
    definitions = [
        ("failed,one", "child", "parent", "pending", [2, 2, 3, 6], (2, 2, 3, 6)),
        ("parent", "parent", None, "inactive", [1, 1, 7, 7], (1, 1, 7, 7)),
        ("contained", "parent", None, "frozen", [2, 2, 5, 5], (2, 2, 5, 5)),
        ("overlap", "parent", None, "frozen", [5, 5, 9, 9], (5, 5, 9, 9)),
        ("residual_neighbor", "parent", None, "frozen", [9, 9, 11, 11], (9, 9, 11, 11)),
        ("unrelated", "parent", None, "frozen", [0, 9, 2, 11], (0, 9, 2, 11)),
    ]
    for z_index, (component_id, kind, parent_id, state, bbox, area) in enumerate(definitions):
        mask = np.zeros(size, dtype=np.uint8)
        left, top, right, bottom = area
        mask[top:bottom, left:right] = 255
        mask_path = masks / f"{component_id}.png"
        Image.fromarray(mask).save(mask_path)
        nodes.append({
            "id": component_id,
            "kind": kind,
            "parent_id": parent_id,
            "state": state,
            "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "bbox": bbox,
            "z_index": z_index,
            "text_ids": [],
        })
    evidence["component-graph.json"].write_text(
        json.dumps({"nodes": nodes}), encoding="utf-8"
    )
    residual = np.zeros(size, dtype=np.uint8)
    residual[8, 8] = 255
    Image.fromarray(residual).save(evidence["unexplained-mask.png"])
    evidence["quality-report.json"].write_text(json.dumps({
        "schema_version": 1,
        "report": {
            "violations": [
                "unexplained_visual_residual", "contained_parent_review"
            ],
        },
        "contained_parent_pairs": [["contained", "failed,one"]],
    }), encoding="utf-8")
    _refresh_test_presentation_manifest(page_session)


def test_first_round_review_evidence_preserves_full_visual_message(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = load_component_agent_request(request_path)

    assert request["review_evidence"] == [
        *component_repair.FULL_COMPONENT_REVIEW_EVIDENCE
    ]


def test_round_two_review_contains_failed_node_and_every_dependency_neighbor(
    page_session: dict,
) -> None:
    _prepare_round_two_review_session(page_session)

    request_path = build_component_agent_request(page_session, repair_round=2)
    request = load_component_agent_request(request_path)
    atlas_path = request_path.parent / request["evidence"]["round-review.png"]["path"]
    with Image.open(atlas_path) as atlas:
        assert json.loads(atlas.info["component_ids"]) == [
            "contained", "failed,one", "overlap", "parent", "residual_neighbor"
        ]
        assert json.loads(atlas.info["panel_names"]) == [
            "source", "isolation", "ownership", "reconstructed", "difference",
            "residual",
        ]
        rows = json.loads(atlas.info["rows"])
        failed_row = next(row for row in rows if row["id"] == "failed,one")
        assert failed_row["crop"] == [2, 2, 3, 6]
        assert atlas.width >= failed_row["label_width"]
        assert np.all(np.asarray(atlas.convert("RGB"))[40:43, 0:3] == 220)
    assert request["review_evidence"] == [
        "source.png", "reconstructed.png", "difference.png",
        "unexplained-mask.png", "quality-report.json", "round-review.png",
    ]


def test_round_review_write_failure_falls_back_to_full_evidence(
    page_session: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_round_two_review_session(page_session)
    monkeypatch.setattr(
        component_repair,
        "_write_round_review",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    request_path = build_component_agent_request(page_session, repair_round=2)
    request = load_component_agent_request(request_path)

    assert "round-review.png" not in request["evidence"]
    assert request["review_evidence"] == [
        *component_repair.FULL_COMPONENT_REVIEW_EVIDENCE
    ]


def test_partial_round_review_write_is_removed_before_full_fallback(
    page_session: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_round_two_review_session(page_session)

    partial_target = Path(page_session["reconstruction_dir"]) / "partial-target.bin"
    partial_target.write_bytes(b"partial")

    def fail_after_partial_write(*, staging: Path, **kwargs):
        (staging / "round-review.png").hardlink_to(partial_target)
        raise OSError("write interrupted")

    monkeypatch.setattr(component_repair, "_write_round_review", fail_after_partial_write)

    request_path = build_component_agent_request(page_session, repair_round=2)

    assert "round-review.png" not in load_component_agent_request(request_path)[
        "evidence"
    ]


def test_round_review_mask_work_budget_falls_back_before_unbounded_decode(
    page_session: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_round_two_review_session(page_session)
    monkeypatch.setattr(component_repair, "ROUND_REVIEW_MAX_MASK_PIXELS", 100)

    request_path = build_component_agent_request(page_session, repair_round=2)
    request = load_component_agent_request(request_path)

    assert "round-review.png" not in request["evidence"]
    assert request["review_evidence"] == [
        *component_repair.FULL_COMPONENT_REVIEW_EVIDENCE
    ]


def test_round_review_working_budget_falls_back_before_full_image_decode(
    page_session: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_round_two_review_session(page_session)
    monkeypatch.setattr(component_repair, "ROUND_REVIEW_MAX_WORKING_BYTES", 1)
    monkeypatch.setattr(
        component_repair,
        "_round_review_image",
        lambda *args, **kwargs: pytest.fail("full image decoded before budget check"),
    )
    monkeypatch.setattr(
        component_repair,
        "_round_review_mask",
        lambda *args, **kwargs: pytest.fail("full mask decoded before budget check"),
    )

    request_path = build_component_agent_request(page_session, repair_round=2)
    request = load_component_agent_request(request_path)

    assert "round-review.png" not in request["evidence"]
    assert request["review_evidence"] == [
        *component_repair.FULL_COMPONENT_REVIEW_EVIDENCE
    ]


def test_round_review_tamper_is_rejected_with_full_request_hash_checks(
    page_session: dict,
) -> None:
    _prepare_round_two_review_session(page_session)
    request_path = build_component_agent_request(page_session, repair_round=2)
    request = load_component_agent_request(request_path)
    atlas_path = request_path.parent / request["evidence"]["round-review.png"]["path"]
    atlas_path.write_bytes(atlas_path.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="evidence hash mismatch"):
        load_component_agent_request(request_path)


build_component_agent_request = _build_component_agent_request


def _make_page_session(run_root: Path, page_id: str) -> dict:
    reconstruction = run_root / "pages" / page_id / "reconstruction"
    evidence_root = reconstruction / "evidence-source"
    evidence_root.mkdir(parents=True)
    graph = {
        "nodes": [
            _node("candidate_b", "pending", 1),
            _node("frozen_a", "frozen", 0),
        ]
    }
    masks = evidence_root / "masks"
    masks.mkdir()
    for index, node in enumerate(graph["nodes"]):
        mask_path = masks / f"{node['id']}.png"
        Image.fromarray(np.full((2, 2), 255 - index, dtype=np.uint8)).save(mask_path)
        node["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    sources = {}
    for name in EVIDENCE_NAMES:
        path = evidence_root / name
        if name == "component-graph.json":
            path.write_text(json.dumps(graph), encoding="utf-8")
        elif name == "quality-report.json":
            path.write_text(json.dumps({
                "schema_version": 1,
                "phase": "initial_layers",
                "text_items": [],
                "initial_diagnostics": [],
                "violations": [],
            }), encoding="utf-8")
        elif path.suffix == ".png":
            value = 255 if name == "source.png" else 0
            Image.fromarray(np.full((2, 2), value, dtype=np.uint8)).save(path)
        else:
            path.write_bytes((name + " data").encode())
        sources[name] = path
    manifest_path = _write_test_presentation_manifest(
        run_root,
        evidence_root,
        source_sha256=hashlib.sha256(sources["source.png"].read_bytes()).hexdigest(),
        graph_path=sources["component-graph.json"],
    )
    sources["presentation-manifest.json"] = manifest_path
    return {
        "page_id": page_id,
        "provider": "host",
        "reconstruction_dir": reconstruction,
        "evidence": sources,
    }


def test_build_request_hash_binds_every_evidence_file(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert set(request["evidence"]) == set(EVIDENCE_NAMES)
    assert all(len(record["sha256"]) == 64 for record in request["evidence"].values())
    assert request["candidate_ids"] == ["candidate_b"]
    assert request["frozen_ids"] == ["frozen_a"]
    assert request_path.parent.name == "round-01"


def test_presentation_asset_tamper_is_rejected_on_request_load(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    manifest = json.loads(
        (request_path.parent / "presentation-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    run_root = request_path.parents[5]
    asset = run_root / manifest["components"][0]["rgba"]["path"]
    asset.write_bytes(asset.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="presentation asset hash mismatch"):
        load_component_agent_request(request_path)


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("source_sha256", "presentation manifest source hash mismatch"),
        ("graph_sha256", "presentation manifest graph hash mismatch"),
    ],
)
def test_presentation_manifest_binds_request_source_and_graph_hashes(
    page_session: dict,
    field: str,
    error: str,
) -> None:
    manifest_path = page_session["evidence"]["presentation-manifest.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=error):
        _build_component_agent_request(page_session, repair_round=1)


@pytest.mark.parametrize("path", ["../rgba.png", "/rgba.png"])
def test_presentation_manifest_rejects_asset_path_outside_run_root(
    page_session: dict,
    path: str,
) -> None:
    manifest_path = page_session["evidence"]["presentation-manifest.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["components"][0]["rgba"]["path"] = path
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="presentation asset path"):
        _build_component_agent_request(page_session, repair_round=1)


def test_presentation_manifest_components_must_match_active_graph_order(
    page_session: dict,
) -> None:
    manifest_path = page_session["evidence"]["presentation-manifest.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["components"].reverse()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="components.*graph"):
        _build_component_agent_request(page_session, repair_round=1)


def test_quality_layers_use_the_already_bound_manifest_payload(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = load_component_agent_request(request_path)
    manifest_path = request_path.parent / "presentation-manifest.json"
    reconstruction = Path(page_session["reconstruction_dir"])
    run_root = reconstruction.parents[2]
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset_paths = {
        (run_root / reference["path"]).resolve()
        for component in raw_manifest["components"]
        for name, reference in component.items()
        if name in {
            "rgba", "ownership_mask", "presentation_alpha_mask",
            "generated_underlay_mask",
        }
    }
    real_read = component_repair._read_bound_file
    read_counts = {path: 0 for path in asset_paths}

    def bounded_read(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved in read_counts:
            read_counts[resolved] += 1
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(component_repair, "_read_bound_file", bounded_read)
    monkeypatch.setattr(
        component_repair,
        "_hash_bound_file",
        lambda *args, **kwargs: pytest.fail("quality must not unbounded-hash assets"),
    )
    manifest = component_repair._validate_presentation_manifest_payload(
        manifest_path.read_bytes(),
        reconstruction,
        source_sha256=request["source_sha256"],
        graph_sha256=request["graph_sha256"],
        run_root=run_root,
        expected_component_ids=["candidate_b", "frozen_a"],
    )

    manifest_path.write_text("{", encoding="utf-8")
    layers = list(component_repair._iter_quality_presentation_layers(
        run_root=run_root,
        reconstruction=reconstruction,
        manifest=manifest,
        page_shape=(2, 2),
    ))

    assert [layer["component_id"] for layer in layers] == [
        "candidate_b", "frozen_a",
    ]
    assert set(read_counts.values()) == {1}


def test_request_presentation_asset_preflight_uses_bounded_reads(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = json.loads(
        page_session["evidence"]["presentation-manifest.json"].read_text(
            encoding="utf-8"
        )
    )
    asset_paths = {
        (Path(page_session["reconstruction_dir"]).parents[2] / reference["path"])
        .resolve()
        for component in manifest["components"]
        for name, reference in component.items()
        if name in {
            "rgba", "ownership_mask", "presentation_alpha_mask",
            "generated_underlay_mask",
        }
    }
    real_hash = component_repair._hash_bound_file

    def reject_unbounded_asset_hash(path, reconstruction):
        if Path(path).resolve() in asset_paths:
            pytest.fail("request asset preflight must be bounded")
        return real_hash(path, reconstruction)

    monkeypatch.setattr(
        component_repair, "_hash_bound_file", reject_unbounded_asset_hash,
    )

    build_component_agent_request(page_session, repair_round=1)


def test_round_one_initial_quality_schema_has_no_improvement_baseline() -> None:
    assert component_repair._previous_component_reports(
        {
            "schema_version": 1,
            "phase": "initial_layers",
            "text_items": [],
            "initial_diagnostics": [],
            "violations": [],
        },
        state={
            "repair_round": 1,
            "source_sha256": "a" * 64,
        },
        request={},
        active_component_ids=["candidate_b"],
    ) == {}


def test_presentation_manifest_rejects_boolean_schema_version(
    page_session: dict,
) -> None:
    manifest_path = page_session["evidence"]["presentation-manifest.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        _build_component_agent_request(page_session, repair_round=1)


def test_validate_request_rejects_changed_overlay(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    overlay = request_path.parent / "ocr-overlay.png"
    overlay.write_bytes(overlay.read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match="evidence hash"):
        load_component_agent_request(request_path)


def test_validate_request_rejects_changed_unexplained_mask(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    unexplained = request_path.parent / "unexplained-mask.png"
    unexplained.write_bytes(unexplained.read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match="evidence hash"):
        load_component_agent_request(request_path)


@pytest.mark.parametrize("repair_round", [0, 6, True, 1.0])
def test_build_request_rejects_round_outside_fixed_limit(
    page_session: dict,
    repair_round: object,
) -> None:
    with pytest.raises(ValueError, match="repair_round"):
        build_component_agent_request(page_session, repair_round=repair_round)


def test_published_request_cannot_be_overwritten(page_session: dict) -> None:
    first = build_component_agent_request(page_session, repair_round=1)
    before = first.read_bytes()

    with pytest.raises(RuntimeError, match="already published"):
        build_component_agent_request(page_session, repair_round=1)

    assert first.read_bytes() == before


def _publish(session: dict, ready: object, result: object) -> None:
    ready.wait(10)
    try:
        build_component_agent_request(session, repair_round=1)
    except RuntimeError:
        result.put("rejected")
    else:
        result.put("published")


def _publish_round(session: dict, repair_round: int, started: object, result: object) -> None:
    started.set()
    try:
        build_component_agent_request(session, repair_round=repair_round)
    except RuntimeError:
        result.put("rejected")
    else:
        result.put("published")


def _publish_round_with_execution_lease(
    session: dict,
    repair_round: int,
    ready: object,
    release: object,
    result: object,
) -> None:
    real_build = component_repair._build_component_agent_request_locked

    def hold_build(*args: object, **kwargs: object) -> Path:
        ready.set()
        release.wait(10)
        return real_build(*args, **kwargs)

    component_repair._build_component_agent_request_locked = hold_build
    reconstruction = Path(session["reconstruction_dir"])
    run_root = reconstruction.parents[2]
    try:
        with ExecutionLease(
            run_root / "execution.lock", run_root=run_root,
        ) as lease:
            component_repair.build_component_agent_request(
                session, repair_round=repair_round, _lease=lease,
            )
    except Exception as error:
        result.put(("error", type(error).__name__, str(error)))
    else:
        result.put(("published",))


def _hold_publication_lease(reconstruction: str, ready: object, release: object) -> None:
    with component_repair._run_publication_lease(Path(reconstruction)):
        component_repair._load_integrity_key(Path(reconstruction))
        ready.set()
        release.wait(10)


def _refresh_marker_for_contract_test(request_path: Path) -> None:
    marker_path = request_path.parent / "publication-marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["request_sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    key_path = (
        request_path.parents[5]
        / ".component-agent-integrity"
        / "key.bin"
    )
    fields = {key: value for key, value in marker.items() if key != "hmac_sha256"}
    canonical = json.dumps(
        fields,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    marker["hmac_sha256"] = hmac.new(
        key_path.read_bytes(), canonical, hashlib.sha256
    ).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def test_same_page_round_has_one_concurrent_publisher(page_session: dict) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    result = context.Queue()
    processes = [
        context.Process(target=_publish, args=(page_session, ready, result))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0

    assert sorted(result.get(timeout=2) for _ in processes) == [
        "published",
        "rejected",
    ]


def test_build_request_rejects_cross_page_evidence(page_session: dict, tmp_path: Path) -> None:
    outside = tmp_path / "pages" / "page_002" / "reconstruction" / "source.png"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"other page")
    page_session["evidence"]["source.png"] = outside

    with pytest.raises(RuntimeError, match="outside page reconstruction"):
        build_component_agent_request(page_session, repair_round=1)


def test_load_rejects_hard_linked_evidence(page_session: dict, tmp_path: Path) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    evidence = request_path.parent / "ownership.png"
    outside = tmp_path / "outside.png"
    evidence.replace(outside)
    try:
        os.link(outside, evidence)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="hard link"):
        load_component_agent_request(request_path)


def test_load_rejects_request_unknown_field(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["unexpected"] = True
    request_path.write_text(json.dumps(request), encoding="utf-8")
    _refresh_marker_for_contract_test(request_path)

    with pytest.raises(ValueError, match="request fields"):
        load_component_agent_request(request_path)


def test_load_rejects_evidence_path_traversal(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["evidence"]["source.png"]["path"] = "../source.png"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    _refresh_marker_for_contract_test(request_path)

    with pytest.raises(ValueError, match="evidence path"):
        load_component_agent_request(request_path)


def test_build_rejects_symlinked_evidence(page_session: dict, tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    source = page_session["evidence"]["source.png"]
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="link|safely"):
        build_component_agent_request(page_session, repair_round=1)


def test_load_rejects_candidate_ids_not_bound_to_graph(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["candidate_ids"] = []
    request_path.write_text(json.dumps(request), encoding="utf-8")
    _refresh_marker_for_contract_test(request_path)

    with pytest.raises(RuntimeError, match="component ids"):
        load_component_agent_request(request_path)


def test_build_rejects_similar_but_non_pages_directory(tmp_path: Path) -> None:
    reconstruction = tmp_path / "not-pages" / "page_001" / "reconstruction"
    evidence_root = reconstruction / "evidence-source"
    evidence_root.mkdir(parents=True)
    graph = {"nodes": [_node("candidate", "pending", 0)]}
    evidence = {}
    for name in EVIDENCE_NAMES:
        path = evidence_root / name
        if name == "component-graph.json":
            path.write_text(json.dumps(graph), encoding="utf-8")
        elif name == "quality-report.json":
            path.write_text('{"valid": false}', encoding="utf-8")
        else:
            path.write_bytes(name.encode())
        evidence[name] = path

    with pytest.raises(RuntimeError, match="pages/.+reconstruction"):
        build_component_agent_request(
            {
                "page_id": "page_001",
                "provider": "host",
                "reconstruction_dir": reconstruction,
                "evidence": evidence,
            },
            repair_round=1,
        )


def test_load_rejects_request_moved_under_similar_fake_directory(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    pages_dir = Path(page_session["reconstruction_dir"]).parent.parent
    fake_dir = pages_dir.with_name("not-pages")
    pages_dir.rename(fake_dir)
    moved_request = fake_dir / request_path.relative_to(pages_dir)

    with pytest.raises(RuntimeError, match="pages/.+reconstruction"):
        load_component_agent_request(moved_request)


def test_load_requires_external_request_digest_marker(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    marker = request_path.parent / "publication-marker.json"
    marker.unlink()

    with pytest.raises(RuntimeError, match="marker"):
        load_component_agent_request(request_path)


def test_load_rejects_provider_changed_after_publish(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["provider"] = "local"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(RuntimeError, match="request hash"):
        load_component_agent_request(request_path)


def test_load_rejects_evidence_and_request_hash_changed_together(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    evidence_path = request_path.parent / "ocr-overlay.png"
    evidence_path.write_bytes(b"coordinated replacement")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["evidence"]["ocr-overlay.png"]["sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(RuntimeError, match="request hash"):
        load_component_agent_request(request_path)


def test_load_rejects_synchronized_request_evidence_and_marker_without_key(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    evidence_path = request_path.parent / "ocr-overlay.png"
    evidence_path.write_bytes(b"coordinated replacement")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["evidence"]["ocr-overlay.png"]["sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    request_path.write_text(json.dumps(request), encoding="utf-8")
    marker_path = request_path.parent / "publication-marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["request_sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    marker["hmac_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="signature"):
        load_component_agent_request(request_path)


def test_marker_write_failure_does_not_publish_round_and_can_retry(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = component_repair._write_exclusive
    failed = False

    def fail_marker(path: Path, payload: bytes, reconstruction: Path) -> None:
        nonlocal failed
        if path.name == "publication-marker.json" and not failed:
            failed = True
            raise OSError("simulated marker failure")
        real_write(path, payload, reconstruction)

    monkeypatch.setattr(component_repair, "_write_exclusive", fail_marker)
    with pytest.raises(OSError, match="marker failure"):
        build_component_agent_request(page_session, repair_round=1)

    round_dir = Path(page_session["reconstruction_dir"]) / "agent" / "round-01"
    assert not round_dir.exists()
    monkeypatch.setattr(component_repair, "_write_exclusive", real_write)
    assert build_component_agent_request(page_session, repair_round=1).is_file()


def test_round_rename_failure_leaves_no_publication_and_can_retry(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    component_repair._load_or_create_integrity_key(reconstruction)
    real_rename = Path.rename
    failed = False

    def fail_round_rename(path: Path, target: Path) -> Path:
        nonlocal failed
        if path.name.startswith(".round-01.tmp-") and not failed:
            failed = True
            raise OSError("simulated pre-publication crash")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_round_rename)
    with pytest.raises(RuntimeError, match="already published"):
        build_component_agent_request(page_session, repair_round=1)

    round_dir = reconstruction / "agent" / "round-01"
    assert not round_dir.exists()
    monkeypatch.setattr(Path, "rename", real_rename)
    assert build_component_agent_request(page_session, repair_round=1).is_file()


def test_damaged_integrity_key_fails_closed_without_rotation(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    key_path = request_path.parents[5] / ".component-agent-integrity" / "key.bin"
    key_path.write_bytes(b"damaged")

    with pytest.raises(RuntimeError, match="integrity key"):
        load_component_agent_request(request_path)

    assert key_path.read_bytes() == b"damaged"


def test_two_pages_concurrently_reuse_one_complete_integrity_key(
    tmp_path: Path,
) -> None:
    first = _make_page_session(tmp_path, "page_001")
    second = _make_page_session(tmp_path, "page_002")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    result = context.Queue()
    processes = [
        context.Process(target=_publish, args=(session, ready, result))
        for session in (first, second)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0

    assert sorted(result.get(timeout=2) for _ in processes) == [
        "published",
        "published",
    ]
    key_path = tmp_path / ".component-agent-integrity" / "key.bin"
    assert len(key_path.read_bytes()) == 32
    assert key_path.stat().st_nlink == 1
    assert load_component_agent_request(
        tmp_path / "pages/page_001/reconstruction/agent/round-01/component_agent_request.json"
    )["page_id"] == "page_001"
    assert load_component_agent_request(
        tmp_path / "pages/page_002/reconstruction/agent/round-01/component_agent_request.json"
    )["page_id"] == "page_002"


def test_integrity_key_hard_link_fails_closed(page_session: dict, tmp_path: Path) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    key_path = request_path.parents[5] / ".component-agent-integrity" / "key.bin"
    outside = tmp_path / "stolen-key"
    key_path.replace(outside)
    try:
        os.link(outside, key_path)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="integrity key|hard link"):
        load_component_agent_request(request_path)


def test_integrity_key_directory_symlink_fails_closed(
    page_session: dict,
    tmp_path: Path,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    anchor = request_path.parents[5] / ".component-agent-integrity"
    outside = tmp_path / "outside-key-anchor"
    anchor.rename(outside)
    try:
        anchor.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        outside.rename(anchor)
        pytest.skip(f"directory symbolic links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="integrity key|link|reparse"):
        load_component_agent_request(request_path)


def test_missing_key_after_published_round_is_not_rotated(page_session: dict) -> None:
    first = build_component_agent_request(page_session, repair_round=1)
    anchor = first.parents[5] / ".component-agent-integrity"
    shutil.rmtree(anchor)

    with pytest.raises(RuntimeError, match="published.+integrity key"):
        build_component_agent_request(page_session, repair_round=2)

    assert not anchor.exists()


def test_build_streams_large_evidence_in_bounded_reads(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    large = Path(page_session["evidence"]["source.png"])
    large.write_bytes(b"x" * (3 * 1024 * 1024 + 17))
    _refresh_test_presentation_manifest(page_session)
    real_fdopen = os.fdopen
    read_sizes: list[int] = []

    class TrackingFile:
        def __init__(self, wrapped: object) -> None:
            self._wrapped = wrapped

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

        def __enter__(self) -> object:
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> object:
            return self._wrapped.__exit__(exc_type, exc, traceback)

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self._wrapped.read(size)

    def tracking_fdopen(descriptor: int, mode: str) -> object:
        wrapped = real_fdopen(descriptor, mode)
        return TrackingFile(wrapped) if "r" in mode else wrapped

    monkeypatch.setattr(component_repair.os, "fdopen", tracking_fdopen)
    build_component_agent_request(page_session, repair_round=1)

    assert read_sizes
    assert -1 not in read_sizes
    assert max(read_sizes) <= 1024 * 1024


def test_build_rejects_component_graph_over_json_limit(page_session: dict) -> None:
    graph = Path(page_session["evidence"]["component-graph.json"])
    graph.write_bytes(b" " * (16 * 1024 * 1024 + 1))

    with pytest.raises(RuntimeError, match="JSON size limit"):
        build_component_agent_request(page_session, repair_round=1)


def test_load_rejects_marker_over_json_limit(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    marker = request_path.parent / "publication-marker.json"
    marker.write_bytes(marker.read_bytes() + b" " * (64 * 1024))

    with pytest.raises(RuntimeError, match="JSON size limit"):
        load_component_agent_request(request_path)


def test_load_rejects_request_over_json_limit(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request_path.write_bytes(request_path.read_bytes() + b" " * (4 * 1024 * 1024))

    with pytest.raises(RuntimeError, match="JSON size limit"):
        load_component_agent_request(request_path)


def test_load_rejects_component_graph_over_json_limit(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    graph = request_path.parent / "component-graph.json"
    graph.write_bytes(b" " * (16 * 1024 * 1024 + 1))

    with pytest.raises(RuntimeError, match="JSON size limit"):
        load_component_agent_request(request_path)


def test_staging_replacement_before_rename_is_removed_and_retryable(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    component_repair._load_or_create_integrity_key(reconstruction)
    real_rename = Path.rename
    replaced = False

    def replace_staging(path: Path, target: Path) -> Path:
        nonlocal replaced
        if path.name.startswith(".round-01.tmp-") and not replaced:
            original = path.with_name(path.name + ".original")
            attacker = path.with_name(path.name + ".attacker")
            real_rename(path, original)
            attacker.mkdir()
            (attacker / "forged").write_bytes(b"forged")
            real_rename(attacker, path)
            replaced = True
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", replace_staging)
    with pytest.raises(RuntimeError, match="staging identity"):
        build_component_agent_request(page_session, repair_round=1)

    round_dir = reconstruction / "agent" / "round-01"
    assert not round_dir.exists()
    quarantines = list((reconstruction / "agent").glob(".quarantine-round-*"))
    assert any((path / "forged").read_bytes() == b"forged" for path in quarantines)
    monkeypatch.setattr(Path, "rename", real_rename)
    assert build_component_agent_request(page_session, repair_round=1).is_file()


def test_simulated_windows_reparse_flag_on_key_anchor_fails_closed(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    anchor = request_path.parents[5] / ".component-agent-integrity"
    real_lstat = Path.lstat
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    class ReparseStatus:
        def __init__(self, wrapped: os.stat_result) -> None:
            self._wrapped = wrapped
            self.st_file_attributes = (
                getattr(wrapped, "st_file_attributes", 0) | reparse_flag
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    def flagged_lstat(path: Path) -> object:
        status = real_lstat(path)
        return ReparseStatus(status) if path == anchor else status

    monkeypatch.setattr(Path, "lstat", flagged_lstat)

    with pytest.raises(RuntimeError, match="reparse"):
        load_component_agent_request(request_path)


def test_missing_key_scan_rejects_simulated_reparse_agent_directory(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    anchor = request_path.parents[5] / ".component-agent-integrity"
    shutil.rmtree(anchor)
    agent = request_path.parent.parent
    real_lstat = Path.lstat
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    class ReparseStatus:
        def __init__(self, wrapped: os.stat_result) -> None:
            self._wrapped = wrapped
            self.st_file_attributes = (
                getattr(wrapped, "st_file_attributes", 0) | reparse_flag
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    def flagged_lstat(path: Path) -> object:
        status = real_lstat(path)
        return ReparseStatus(status) if path == agent else status

    monkeypatch.setattr(Path, "lstat", flagged_lstat)

    with pytest.raises(RuntimeError, match="agent.+reparse"):
        build_component_agent_request(page_session, repair_round=2)
    assert not anchor.exists()


def test_cleanup_quarantines_replaced_staging_without_deleting_unknown_content(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    component_repair._load_or_create_integrity_key(reconstruction)
    real_write = component_repair._write_exclusive
    real_rename = Path.rename
    swapped = False

    def fail_marker(path: Path, payload: bytes, reconstruction: Path) -> None:
        if path.name == "publication-marker.json":
            raise OSError("marker failure")
        real_write(path, payload, reconstruction)

    def swap_before_quarantine(path: Path, target: Path) -> Path:
        nonlocal swapped
        if (
            not swapped
            and path.name.startswith(".round-01.tmp-")
            and ".quarantine-" in target.name
        ):
            original = path.with_name(path.name + ".original")
            attacker = path.with_name(path.name + ".attacker")
            real_rename(path, original)
            attacker.mkdir()
            (attacker / "unknown.txt").write_text("do not delete", encoding="utf-8")
            real_rename(attacker, path)
            swapped = True
        return real_rename(path, target)

    monkeypatch.setattr(component_repair, "_write_exclusive", fail_marker)
    monkeypatch.setattr(Path, "rename", swap_before_quarantine)

    with pytest.raises(OSError, match="marker failure"):
        build_component_agent_request(page_session, repair_round=1)

    quarantines = list((reconstruction / "agent").glob(".quarantine-*"))
    assert swapped is True
    assert any((path / "unknown.txt").read_text(encoding="utf-8") == "do not delete" for path in quarantines if (path / "unknown.txt").is_file())
    assert not (reconstruction / "agent" / "round-01").exists()


def test_normal_marker_failure_cleans_owned_quarantine_without_residue(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    real_write = component_repair._write_exclusive

    def fail_marker(path: Path, payload: bytes, reconstruction: Path) -> None:
        if path.name == "publication-marker.json":
            raise OSError("marker failure")
        real_write(path, payload, reconstruction)

    monkeypatch.setattr(component_repair, "_write_exclusive", fail_marker)
    with pytest.raises(OSError, match="marker failure"):
        build_component_agent_request(page_session, repair_round=1)

    agent = reconstruction / "agent"
    assert not list(agent.glob(".quarantine-*"))
    assert not list(agent.glob(".delete-*"))
    assert not (agent / "round-01").exists()


def test_run_publication_lease_blocks_key_rotation_during_inflight_publish(
    page_session: dict,
) -> None:
    first = build_component_agent_request(page_session, repair_round=1)
    reconstruction = Path(page_session["reconstruction_dir"])
    anchor = first.parents[5] / ".component-agent-integrity"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_publication_lease,
        args=(str(reconstruction), ready, release),
    )
    holder.start()
    assert ready.wait(10)
    shutil.rmtree(anchor)

    started = context.Event()
    result = context.Queue()
    contender = context.Process(
        target=_publish_round,
        args=(page_session, 2, started, result),
    )
    contender.start()
    assert started.wait(10)
    contender.join(0.5)
    assert contender.is_alive()
    with pytest.raises(queue.Empty):
        result.get_nowait()

    release.set()
    holder.join(10)
    contender.join(10)
    assert holder.exitcode == 0
    assert contender.exitcode == 0
    assert result.get(timeout=2) == "rejected"
    assert not anchor.exists()


@pytest.mark.skipif(
    os.name != "nt", reason="Windows execution and publication locks differ",
)
def test_windows_execution_lease_publication_blocks_direct_process(
    page_session: dict,
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder_result = context.Queue()
    holder = context.Process(
        target=_publish_round_with_execution_lease,
        args=(page_session, 1, ready, release, holder_result),
    )
    started = context.Event()
    contender_result = context.Queue()
    contender = context.Process(
        target=_publish_round,
        args=(page_session, 2, started, contender_result),
    )
    started_processes = []
    try:
        holder.start()
        started_processes.append(holder)
        assert ready.wait(10)
        contender.start()
        started_processes.append(contender)
        assert started.wait(10)
        contender.join(0.5)
        assert contender.is_alive()
        with pytest.raises(queue.Empty):
            contender_result.get_nowait()
    finally:
        release.set()
        for process in started_processes:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(10)

    assert holder.exitcode == 0
    assert contender.exitcode == 0
    assert holder_result.get(timeout=2) == ("published",)
    assert contender_result.get(timeout=2) == "published"


def test_build_detects_parent_replaced_between_check_and_open(
    page_session: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = Path(page_session["evidence"]["component-graph.json"]).parent
    original_root = evidence_root.with_name("evidence-original")
    attacker_root = evidence_root.with_name("evidence-attacker")
    shutil.copytree(evidence_root, attacker_root)
    attacker_graph = attacker_root / "component-graph.json"
    attacker_graph.write_text(
        json.dumps({"nodes": [_node("attacker", "pending", 0)]}),
        encoding="utf-8",
    )
    real_open = os.open
    replaced = False

    def replace_parent_before_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if not replaced and Path(path) == evidence_root / "component-graph.json":
            evidence_root.rename(original_root)
            attacker_root.rename(evidence_root)
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(component_repair.os, "open", replace_parent_before_open)

    with pytest.raises(RuntimeError, match="directory identity"):
        build_component_agent_request(page_session, repair_round=1)

    assert replaced is True
    assert json.loads(
        (evidence_root / "component-graph.json").read_text(encoding="utf-8")
    )["nodes"][0]["id"] == "attacker"


def test_build_detects_parent_changed_to_symlink_before_open(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = Path(page_session["evidence"]["component-graph.json"]).parent
    original_root = evidence_root.with_name("evidence-original")
    attacker_root = evidence_root.with_name("evidence-attacker")
    shutil.copytree(evidence_root, attacker_root)
    try:
        probe = evidence_root.with_name("symlink-probe")
        probe.symlink_to(attacker_root, target_is_directory=True)
        probe.unlink()
    except OSError as error:
        pytest.skip(f"directory symbolic links are unavailable: {error}")
    real_open = os.open
    replaced = False

    def replace_with_symlink(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if not replaced and Path(path) == evidence_root / "component-graph.json":
            evidence_root.rename(original_root)
            evidence_root.symlink_to(attacker_root, target_is_directory=True)
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(component_repair.os, "open", replace_with_symlink)

    with pytest.raises(RuntimeError, match="directory identity"):
        build_component_agent_request(page_session, repair_round=1)


def test_marker_write_detects_agent_directory_replacement(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    agent_dir = reconstruction / "agent"
    original_agent = reconstruction / "agent-original"
    attacker_agent = reconstruction / "agent-attacker"
    real_open = os.open
    replaced = False

    def replace_agent_before_marker(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if not replaced and Path(path).name == "publication-marker.json":
            agent_dir.rename(original_agent)
            attacker_agent.mkdir()
            attacker_agent.rename(agent_dir)
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(component_repair.os, "open", replace_agent_before_marker)

    with pytest.raises(RuntimeError, match="directory identity"):
        build_component_agent_request(page_session, repair_round=1)

    assert replaced is True
