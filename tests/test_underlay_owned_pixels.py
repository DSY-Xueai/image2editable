from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(params=["scripts", "skills/image-to-ppt/scripts", "skills/image-to-psd/scripts"])
def underlay_engine(request):
    path = Path(__file__).resolve().parents[1] / request.param / "component_underlay.py"
    spec = importlib.util.spec_from_file_location("owned_pixel_underlay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("detail", ["uniform", "outline", "gradient", "same_surface"])
def test_visible_lower_surface_stays_owned_beside_higher_object(underlay_engine, detail):
    y, x = np.mgrid[:80, :120]
    source = np.full((80, 120, 3), (220, 230, 240), dtype=np.uint8)
    semantic = np.zeros((80, 120), dtype=bool)
    semantic[10:70, 10:110] = True
    higher = np.zeros_like(semantic)
    higher[32:48, 52:68] = True
    if detail == "outline":
        source[30:32, 46:74] = (30, 40, 50)
    elif detail == "gradient":
        source = np.dstack((160 + x // 2, 175 + y // 2, 180 + x // 3)).astype(np.uint8)
    if detail != "same_surface":
        source[higher] = (180, 40, 70)
    visible = semantic & ~higher
    layer = underlay_engine.build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=visible,
        semantic_mask=semantic,
        higher_layer_mask=higher,
        text_mask=np.zeros_like(higher),
    )

    assert np.array_equal(layer["ownership_mask"], visible)
    assert np.array_equal(layer["rgb"][visible], source[visible])
    assert not np.any(layer["presentation_alpha_mask"] & higher)
    assert not np.any(layer["generated_underlay_mask"])
    moved = np.full_like(source, 255)
    moved_layer = np.roll(layer["rgb"], 12, axis=1)
    moved_alpha = np.roll(layer["presentation_alpha_mask"], 12, axis=1)
    moved[moved_alpha] = moved_layer[moved_alpha]
    assert np.array_equal(moved[np.roll(visible, 12, axis=1)], np.roll(source, 12, axis=1)[np.roll(visible, 12, axis=1)])
    assert np.all(moved[np.roll(higher, 12, axis=1)] == 255)


def test_actual_higher_color_bleed_is_removed_from_movable_lower_layer(underlay_engine):
    source = np.full((80, 120, 3), (220, 230, 240), dtype=np.uint8)
    semantic = np.zeros((80, 120), dtype=bool)
    semantic[10:70, 10:110] = True
    higher = np.zeros_like(semantic)
    higher[32:48, 52:68] = True
    bleed = cv2.dilate(higher.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    source[bleed] = (30, 80, 190)
    layer = underlay_engine.build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=source,
        ownership_mask=semantic & ~higher,
        semantic_mask=semantic,
        higher_layer_mask=higher,
        text_mask=np.zeros_like(higher),
    )
    visible_bleed = bleed & ~higher
    assert not np.any(layer["ownership_mask"] & visible_bleed)
    assert np.all(layer["generated_underlay_mask"][visible_bleed])
    assert np.max(np.abs(layer["rgb"][visible_bleed].astype(np.int16) - np.array([220, 230, 240]))) <= 8
    assert not np.any(layer["presentation_alpha_mask"] & higher)
