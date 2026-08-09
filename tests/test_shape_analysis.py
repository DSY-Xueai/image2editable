import cv2
import numpy as np
import pytest

from image2editable.shape_analysis import analyze_shape_candidate


def _canvas(height: int = 100, width: int = 140) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    return rgba, mask


def _paint(rgba: np.ndarray, mask: np.ndarray) -> None:
    rgba[mask > 0] = (30, 90, 180, 255)


def _rounded_rectangle(
    mask: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    radius: int,
) -> None:
    left, top = top_left
    right, bottom = bottom_right
    cv2.rectangle(mask, (left + radius, top), (right - radius, bottom), 255, -1)
    cv2.rectangle(mask, (left, top + radius), (right, bottom - radius), 255, -1)
    for center in (
        (left + radius, top + radius),
        (right - radius, top + radius),
        (left + radius, bottom - radius),
        (right - radius, bottom - radius),
    ):
        cv2.circle(mask, center, radius, 255, -1)


def test_solid_rectangle_produces_native_candidate() -> None:
    rgba, mask = _canvas()
    cv2.rectangle(mask, (20, 25), (110, 75), 255, -1)
    _paint(rgba, mask)

    candidate = analyze_shape_candidate(rgba, mask)

    assert candidate is not None
    assert candidate["shape_type"] == "rectangle"
    assert candidate["geometry_score"] >= 0.99
    assert candidate["fill_rgb"] == [30, 90, 180]


def test_rounded_rectangle_produces_native_candidate() -> None:
    rgba, mask = _canvas()
    _rounded_rectangle(mask, (20, 20), (115, 80), 12)
    _paint(rgba, mask)

    candidate = analyze_shape_candidate(rgba, mask)

    assert candidate is not None
    assert candidate["shape_type"] == "rounded_rectangle"
    assert candidate["geometry_score"] >= 0.99


def test_ellipse_produces_native_candidate() -> None:
    rgba, mask = _canvas()
    cv2.ellipse(mask, (70, 50), (35, 20), 0, 0, 360, 255, -1)
    _paint(rgba, mask)

    candidate = analyze_shape_candidate(rgba, mask)

    assert candidate is not None
    assert candidate["shape_type"] == "ellipse"
    assert candidate["geometry_score"] >= 0.99


def test_line_produces_native_candidate() -> None:
    rgba, mask = _canvas()
    cv2.line(mask, (20, 75), (115, 25), 255, 5, cv2.LINE_8)
    _paint(rgba, mask)

    candidate = analyze_shape_candidate(rgba, mask)

    assert candidate is not None
    assert candidate["shape_type"] == "line"
    assert candidate["geometry_score"] >= 0.95


def test_gradient_fill_records_high_color_variation() -> None:
    rgba, mask = _canvas()
    cv2.ellipse(mask, (70, 50), (35, 20), 0, 0, 360, 255, -1)
    for x in range(rgba.shape[1]):
        rgba[:, x, :3] = x
    rgba[:, :, 3] = mask

    candidate = analyze_shape_candidate(rgba, mask)

    assert candidate is not None
    assert candidate["shape_type"] == "ellipse"
    assert candidate["color_mad"] > 3.0


def test_translucent_interior_is_rejected() -> None:
    rgba, mask = _canvas()
    cv2.rectangle(mask, (20, 25), (110, 75), 255, -1)
    _paint(rgba, mask)
    rgba[40:60, 40:90, 3] = 200

    assert analyze_shape_candidate(rgba, mask) is None


@pytest.mark.parametrize("kind", ["hole", "components", "irregular"])
def test_unsafe_shape_is_rejected(kind: str) -> None:
    rgba, mask = _canvas()
    if kind == "hole":
        cv2.rectangle(mask, (20, 20), (110, 80), 255, -1)
        cv2.circle(mask, (65, 50), 12, 0, -1)
    elif kind == "components":
        cv2.circle(mask, (35, 50), 18, 255, -1)
        cv2.circle(mask, (100, 50), 18, 255, -1)
    else:
        points = np.array([[20, 20], [115, 20], [115, 40], [55, 40], [55, 80], [20, 80]])
        cv2.fillPoly(mask, [points], 255)
    _paint(rgba, mask)

    assert analyze_shape_candidate(rgba, mask) is None


def test_text_like_mask_is_rejected() -> None:
    rgba, mask = _canvas()
    cv2.putText(mask, "A", (35, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 7)
    _paint(rgba, mask)

    assert analyze_shape_candidate(rgba, mask) is None


def test_result_is_deterministic_and_scale_stable() -> None:
    results = []
    for scale in (1, 2, 4):
        rgba, mask = _canvas(100 * scale, 140 * scale)
        cv2.ellipse(
            mask,
            (70 * scale, 50 * scale),
            (35 * scale, 20 * scale),
            0,
            0,
            360,
            255,
            -1,
        )
        _paint(rgba, mask)
        first = analyze_shape_candidate(rgba, mask)
        second = analyze_shape_candidate(rgba, mask)
        assert first == second
        results.append(first)

    assert all(result is not None for result in results)
    assert {result["shape_type"] for result in results} == {"ellipse"}
    assert max(result["geometry_score"] for result in results) - min(
        result["geometry_score"] for result in results
    ) <= 0.002
