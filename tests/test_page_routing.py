import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from image2editable.page_routing import (
    PageSignals,
    classify_page,
    strict_page_policy,
)


_IMAGE_TO_PPT_PATH = (
    Path(__file__).parents[1]
    / "skills"
    / "image-to-ppt"
    / "scripts"
    / "image_to_ppt.py"
)
_IMAGE_TO_PPT_SPEC = importlib.util.spec_from_file_location(
    "skill_image_to_ppt", _IMAGE_TO_PPT_PATH
)
image_to_ppt = importlib.util.module_from_spec(_IMAGE_TO_PPT_SPEC)
assert _IMAGE_TO_PPT_SPEC.loader is not None
_IMAGE_TO_PPT_SPEC.loader.exec_module(image_to_ppt)


def test_high_confidence_regular_page_uses_direct_policy():
    result = classify_page(PageSignals(
        source_kind="image", ocr_items=14, ocr_mean_confidence=0.96,
        text_coverage=0.31, regular_geometry_ratio=0.82,
        overlap_ratio=0.01, transparency_ratio=0.0,
        edge_density=0.18, scan_noise=0.02,
    ))
    assert result.route == "direct"
    assert result.automatic_sam is False
    assert result.max_residual_rounds == 0
    assert result.host_agent_allowed is False


def test_low_confidence_page_uses_strict_policy_without_agent():
    result = classify_page(PageSignals(
        source_kind="image", ocr_items=0, ocr_mean_confidence=0.0,
        text_coverage=0.0, regular_geometry_ratio=0.08,
        overlap_ratio=0.42, transparency_ratio=0.35,
        edge_density=0.81, scan_noise=0.34,
    ))
    assert result.route == "strict"
    assert result.host_agent_allowed is False


def test_native_pdf_bypasses_visual_models():
    result = classify_page(PageSignals(source_kind="pdf_native"))
    assert result.route == "native"
    assert result.automatic_sam is False
    assert result.max_lama_calls == 0


def test_medium_confidence_page_uses_one_local_refinement():
    result = classify_page(PageSignals(
        source_kind="pdf", ocr_items=3, ocr_mean_confidence=0.95,
        text_coverage=0.2, regular_geometry_ratio=0.45,
        overlap_ratio=0.15, transparency_ratio=0.2,
        edge_density=0.5, scan_noise=0.15,
    ))
    assert result.route == "local_refine"
    assert result.max_residual_rounds == 1
    assert result.automatic_sam is False


def test_pdf_raster_with_duplicate_ocr_boxes_does_not_force_strict():
    result = classify_page(PageSignals(
        source_kind="pdf", ocr_items=21, ocr_mean_confidence=0.978,
        text_coverage=0.23, regular_geometry_ratio=0.0,
        overlap_ratio=1.0, transparency_ratio=0.0,
        edge_density=0.059, scan_noise=0.076, visual_regions=73,
    ))
    assert result.route == "local_refine"
    assert result.max_residual_rounds == 1
    assert result.automatic_sam is False


def test_strict_policy_keeps_legacy_defaults():
    policy = strict_page_policy()
    assert policy.route == "strict"
    assert policy.automatic_sam is True
    assert policy.max_residual_rounds == 3
    assert policy.hole_recheck is True


def test_process_image_direct_skips_model_segmentation_and_hole_recheck(
    monkeypatch,
    tmp_path: Path,
):
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "source.png"
    Image.fromarray(source, mode="RGB").save(image_path)
    text_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    calls = []
    captured_candidates = []
    flat = np.zeros(source.shape[:2], dtype=bool)
    flat[2:98, 1:99] = True
    duplicate_geometry = np.zeros_like(flat)
    duplicate_geometry[1:99, 1:99] = True
    independent_geometry = np.zeros_like(flat)
    independent_geometry[0, :50] = True

    monkeypatch.setattr(
        image_to_ppt,
        "detect_text",
        lambda *args, **kwargs: ([], text_mask.copy()),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "_generate_filtered_object_proposals",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct route should skip DINO")
        ),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "generate_prompted_mask_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct route should skip prompted SAM")
        ),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "generate_mask_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("automatic SAM should be skipped")
        ),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "filter_prompt_free_candidates",
        lambda candidates, *args, **kwargs: captured_candidates.extend(candidates) or [],
    )
    monkeypatch.setattr(
        image_to_ppt,
        "generate_flat_color_candidates",
        lambda *args, **kwargs: calls.append("flat") or [
            image_to_ppt.MaskCandidate(flat, 0.98, "flat_color")
        ],
    )
    geometry_kwargs = {}

    def fake_geometry(*args, **kwargs):
        calls.append("geometry")
        geometry_kwargs.update(kwargs)
        return [
            image_to_ppt.MaskCandidate(duplicate_geometry, 0.70, "geometry"),
            image_to_ppt.MaskCandidate(independent_geometry, 0.70, "geometry"),
        ]

    monkeypatch.setattr(
        image_to_ppt,
        "generate_geometry_candidates",
        fake_geometry,
    )
    monkeypatch.setattr(image_to_ppt, "resolve_visual_elements", lambda _: [])
    monkeypatch.setattr(image_to_ppt, "validate_visual_masks", lambda _: None)
    monkeypatch.setattr(
        image_to_ppt,
        "build_clean_background",
        lambda image, *args, **kwargs: image.copy(),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "export_visual_components",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        image_to_ppt,
        "build_widescreen_background",
        lambda background, **kwargs: (background.copy(), 0, 0, "identity"),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "recheck_visual_element_holes",
        lambda *args, **kwargs: calls.append("hole"),
    )

    policy = classify_page(PageSignals(
        source_kind="image",
        ocr_items=14,
        ocr_mean_confidence=0.96,
        text_coverage=0.31,
        regular_geometry_ratio=0.82,
        overlap_ratio=0.01,
        edge_density=0.18,
        scan_noise=0.02,
    ))
    image_to_ppt._process_image(
        image_path,
        tmp_path / "work",
        object_detector=None,
        mask_generator=None,
        lang="en",
        defer_quality=True,
        page_policy=policy,
    )

    assert calls == ["flat", "geometry"]
    assert [candidate.source for candidate in captured_candidates] == [
        "flat_color", "geometry",
    ]
    assert geometry_kwargs["min_area_fraction"] == 0.0005
    assert geometry_kwargs["text_mask"].shape == source.shape[:2]
    assert np.array_equal(captured_candidates[1].mask, independent_geometry)


def test_process_image_direct_uses_local_background_repair(
    monkeypatch,
    tmp_path: Path,
):
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "source.png"
    Image.fromarray(source, mode="RGB").save(image_path)
    policy = classify_page(PageSignals(
        source_kind="image",
        ocr_items=14,
        ocr_mean_confidence=0.96,
        text_coverage=0.31,
        regular_geometry_ratio=0.82,
        overlap_ratio=0.01,
        edge_density=0.18,
        scan_noise=0.02,
    ))
    captured = []
    isolated_marker = object()

    monkeypatch.setattr(
        image_to_ppt,
        "_isolated_large_inpainter",
        lambda work_dir: isolated_marker,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "build_clean_background",
        lambda image, *args, **kwargs: captured.append(kwargs) or image.copy(),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "export_visual_components",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(image_to_ppt, "resolve_visual_elements", lambda _: [])
    monkeypatch.setattr(image_to_ppt, "validate_visual_masks", lambda _: None)
    monkeypatch.setattr(
        image_to_ppt,
        "build_widescreen_background",
        lambda background, **kwargs: (background.copy(), 0, 0, "identity"),
    )

    image_to_ppt._process_image(
        image_path,
        tmp_path / "work",
        object_detector=None,
        mask_generator=None,
        lang="en",
        defer_quality=True,
        _resource_isolation=True,
        page_policy=policy,
    )

    assert captured
    assert all(
        kwargs.get("large_inpainter") is not isolated_marker
        for kwargs in captured
    )


def test_fast_router_does_not_call_clean_ui_edges_scan_noise():
    image = np.full((200, 300, 3), 240, dtype=np.uint8)
    import cv2

    cv2.rectangle(image, (10, 10), (290, 190), (20, 80, 120), thickness=-1)
    cv2.rectangle(image, (20, 20), (280, 180), (240, 240, 240), thickness=-1)
    items = [
        {"box": [35, 35, 70, 20], "confidence": 0.98},
        {"box": [35, 70, 70, 20], "confidence": 0.98},
        {"box": [35, 105, 70, 20], "confidence": 0.98},
    ]

    policy = image_to_ppt._infer_page_policy(
        image,
        items,
        source_kind="image",
        pipeline_mode="fast",
    )

    assert policy.route != "strict"


def test_fast_router_escalates_pages_with_many_visual_regions():
    result = classify_page(PageSignals(
        source_kind="image", ocr_items=18, ocr_mean_confidence=0.97,
        text_coverage=0.08, regular_geometry_ratio=0.75,
        overlap_ratio=0.01, transparency_ratio=0.02,
        edge_density=0.03, scan_noise=0.04, visual_regions=30,
    ))

    assert result.route == "local_refine"
    assert result.automatic_sam is False
    assert result.max_residual_rounds == 1


def test_fast_router_ignores_text_contours_when_counting_visual_regions():
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    for x in range(image.shape[1]):
        image[:, x] = (45 + x // 20, 60 + x // 40, 115 + x // 16)
    import cv2

    cv2.ellipse(image, (1330, 90), (390, 370), 0, 0, 360, (155, 105, 105), -1)
    cv2.ellipse(image, (60, 820), (380, 300), 0, 0, 360, (45, 95, 145), -1)
    boxes = [
        {"box": [122, 121, 116, 25], "confidence": 0.95},
        {"box": [122, 262, 428, 87], "confidence": 0.99},
        {"box": [119, 357, 544, 83], "confidence": 0.99},
        {"box": [128, 661, 389, 32], "confidence": 0.98},
    ]
    for item in boxes:
        x, y, w, h = item["box"]
        for offset in range(4, max(5, h - 3), 8):
            cv2.line(image, (x + 4, y + offset), (x + w - 4, y + offset), (250, 250, 250), 2)

    policy = image_to_ppt._infer_page_policy(
        image,
        boxes,
        source_kind="image",
        pipeline_mode="fast",
    )

    assert policy.route == "direct"


def test_fast_geometry_detects_large_flat_ellipses_without_segmentation():
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    for x in range(image.shape[1]):
        image[:, x] = (45 + x // 8, 60 + x // 16, 115 + x // 6)
    import cv2

    cv2.ellipse(image, (292, 36), (110, 100), 0, 0, 360, (155, 105, 105), -1)
    cv2.ellipse(image, (20, 222), (130, 105), 0, 0, 360, (45, 95, 145), -1)

    candidates = image_to_ppt.generate_flat_color_candidates(image)
    boxes = []
    for candidate in candidates:
        ys, xs = np.nonzero(candidate.mask)
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    assert len(candidates) == 2
    assert any(box[0] >= 170 and box[1] == 0 for box in boxes)
    assert any(box[0] == 0 and box[1] >= 100 for box in boxes)


def test_flat_color_candidates_reuse_same_color_connectivity(monkeypatch):
    from scripts import visual_segment

    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    image[20:50, 20:60] = (20, 80, 180)
    image[20:50, 120:160] = (20, 80, 180)
    calls = 0
    full_page_counts = 0
    connected_components = visual_segment.cv2.connectedComponents
    count_nonzero = visual_segment.np.count_nonzero

    def counted_connected_components(*args, **kwargs):
        nonlocal calls
        calls += 1
        return connected_components(*args, **kwargs)

    def counted_nonzero(value, *args, **kwargs):
        nonlocal full_page_counts
        if np.asarray(value).size == image.shape[0] * image.shape[1]:
            full_page_counts += 1
        return count_nonzero(value, *args, **kwargs)

    monkeypatch.setattr(
        visual_segment.cv2,
        "connectedComponents",
        counted_connected_components,
    )
    monkeypatch.setattr(visual_segment.np, "count_nonzero", counted_nonzero)

    candidates = image_to_ppt.generate_flat_color_candidates(image)

    assert len(candidates) == 2
    assert calls == 1
    assert full_page_counts <= 3


def test_geometry_candidates_prefilter_locally_without_changing_kept_mask(
    monkeypatch,
):
    from scripts import visual_segment
    import cv2

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    small = np.asarray([[[2, 2]], [[7, 2]], [[7, 7]], [[2, 7]]])
    kept = np.asarray([[[20, 20]], [[50, 20]], [[50, 50]], [[20, 50]]])
    monkeypatch.setattr(
        visual_segment.cv2,
        "findContours",
        lambda *args, **kwargs: ([small, kept], None),
    )

    candidates = image_to_ppt.generate_geometry_candidates(
        image,
        text_mask=np.zeros(image.shape[:2], dtype=np.uint8),
        min_area_fraction=0.05,
    )

    expected = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(expected, [kept], -1, 255, thickness=-1)
    assert len(candidates) == 1
    assert np.array_equal(candidates[0].mask, expected > 0)
