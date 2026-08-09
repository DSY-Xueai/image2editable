import numpy as np

from image2editable.render_qa import compare_rendered_page


def test_local_error_is_not_hidden_by_page_average() -> None:
    source = np.full((100, 100, 3), 255, dtype=np.uint8)
    baseline = source.copy()
    candidate = source.copy()
    candidate[20:40, 20:40] = 0

    report = compare_rendered_page(
        source,
        baseline,
        candidate,
        object_regions={"shape_1": [20, 20, 40, 40]},
        repeated_baseline=baseline.copy(),
    )

    assert report["accepted"] is False
    assert "shape_1" in report["failed_object_ids"]
    assert "render_difference" in report["violations"]


def test_candidate_equal_to_baseline_passes() -> None:
    source = np.full((40, 60, 3), 120, dtype=np.uint8)
    baseline = source.copy()

    report = compare_rendered_page(
        source,
        baseline,
        baseline.copy(),
        object_regions={},
        repeated_baseline=baseline.copy(),
    )

    assert report["accepted"] is True
    assert report["violations"] == []
    assert set(report) == {
        "schema_version",
        "accepted",
        "metrics",
        "failed_object_ids",
        "violations",
        "noise_tolerance",
    }


def test_missing_ocr_text_fails() -> None:
    source = np.full((40, 60, 3), 120, dtype=np.uint8)

    report = compare_rendered_page(
        source,
        source.copy(),
        source.copy(),
        object_regions={},
        repeated_baseline=source.copy(),
        expected_text_items=[{"text": "Hello", "bbox": [5, 5, 30, 20]}],
        rendered_text_items=[],
    )

    assert report["accepted"] is False
    assert "text_mismatch" in report["violations"]


def test_ocr_bbox_misalignment_fails() -> None:
    source = np.full((40, 80, 3), 120, dtype=np.uint8)

    report = compare_rendered_page(
        source,
        source.copy(),
        source.copy(),
        object_regions={},
        repeated_baseline=source.copy(),
        expected_text_items=[{"text": "Hello", "bbox": [5, 5, 30, 20]}],
        rendered_text_items=[{"text": " hello ", "bbox": [45, 5, 70, 20]}],
    )

    assert report["accepted"] is False
    assert "text_alignment_mismatch" in report["violations"]


def test_object_edge_metric_catches_one_pixel_shift() -> None:
    source = np.full((80, 80, 3), 255, dtype=np.uint8)
    source[20:60, 20:60] = 0
    candidate = np.full_like(source, 255)
    candidate[20:60, 21:61] = 0

    report = compare_rendered_page(
        source,
        source.copy(),
        candidate,
        object_regions={"shape_1": [15, 15, 65, 65]},
        repeated_baseline=source.copy(),
    )

    assert report["accepted"] is False
    assert report["metrics"]["objects"]["shape_1"]["candidate"][
        "edge_symmetric_difference_ratio"
    ] > 0
    assert report["failed_object_ids"] == ["shape_1"]


def test_candidate_difference_within_repeat_noise_passes() -> None:
    source = np.full((40, 60, 3), 120, dtype=np.uint8)
    baseline = source.copy()
    repeated = baseline.copy()
    repeated[10:14, 20:24] = 125

    report = compare_rendered_page(
        source,
        baseline,
        repeated.copy(),
        object_regions={"shape_1": [15, 5, 30, 20]},
        repeated_baseline=repeated,
    )

    assert report["accepted"] is True
    assert report["failed_object_ids"] == []
