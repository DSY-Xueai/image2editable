from __future__ import annotations

import numpy as np

from scripts.object_detect import (
    ObjectProposal,
    filter_object_proposals,
    generate_object_proposals,
)


def _proposal(
    box: tuple[float, float, float, float],
    label: str,
    role: str,
    *,
    score: float = 0.9,
    source: str = "full",
    crop_box: tuple[int, int, int, int] = (0, 0, 200, 200),
    touches_crop_edge: bool = False,
) -> ObjectProposal:
    return ObjectProposal(
        box_xyxy=box,
        score=score,
        label=label,
        role=role,
        source=source,
        crop_box=crop_box,
        touches_crop_edge=touches_crop_edge,
    )


def test_object_proposals_map_tiled_boxes_to_full_image() -> None:
    image = np.zeros((900, 900, 3), dtype=np.uint8)

    class Detector:
        def detect(self, crop, prompt, box_threshold, text_threshold):
            return [{"box_xyxy": (10, 10, 40, 40), "score": 0.9, "label": "icon"}]

    proposals = generate_object_proposals(image, Detector())

    assert any(item.box_xyxy[0] >= 140 and item.box_xyxy[1] >= 140 for item in proposals)
    assert any(item.source.startswith("tile_") for item in proposals)
    assert all(item.crop_box is not None for item in proposals)


def test_object_filter_rejects_crop_wide_portrait() -> None:
    portrait = _proposal(
        (0, 0, 90, 90),
        "portrait",
        "person",
        source="tile_1",
        crop_box=(0, 0, 100, 100),
    )

    assert filter_object_proposals([portrait], (200, 200)) == []


def test_object_filter_preserves_parent_child_but_drops_crop_truncation() -> None:
    parent = _proposal((10, 10, 110, 110), "panel", "container", score=0.8)
    child = _proposal((20, 20, 100, 100), "panel", "container", score=0.9)
    truncated = _proposal(
        (10, 10, 108, 110),
        "panel",
        "container",
        score=0.99,
        source="tile_2",
        crop_box=(0, 0, 108, 200),
        touches_crop_edge=True,
    )

    retained = filter_object_proposals([parent, child, truncated], (200, 200))

    assert parent in retained
    assert child in retained
    assert truncated not in retained


def test_object_filter_rejects_weak_chart_spanning_people_but_keeps_panel() -> None:
    left = _proposal((10, 10, 30, 60), "person", "person")
    right = _proposal((70, 10, 90, 60), "player", "person")
    chart = _proposal((0, 0, 100, 70), "chart", "container")
    panel = _proposal((0, 0, 100, 100), "panel", "container")

    retained = filter_object_proposals([left, right, chart, panel], (200, 200))

    assert left in retained and right in retained
    assert chart not in retained
    assert panel in retained
