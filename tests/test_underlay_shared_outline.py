from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("directory", [
    "scripts", "skills/image-to-ppt/scripts", "skills/image-to-psd/scripts",
])
def test_shared_outline_color_is_not_higher_surface_bleed(directory):
    path = Path(__file__).resolve().parents[1] / directory / "component_underlay.py"
    spec = importlib.util.spec_from_file_location("shared_outline_underlay", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    source = np.full((80, 120, 3), (220, 230, 240), dtype=np.uint8)
    semantic = np.zeros((80, 120), dtype=bool)
    semantic[10:70, 10:110] = True
    higher = np.zeros_like(semantic)
    higher[32:48, 52:68] = True
    source[higher] = (30, 40, 50)
    source[34:46, 54:66] = (180, 40, 70)
    source[30:32, 46:74] = (30, 40, 50)
    visible = semantic & ~higher

    layer = engine.build_presentation_layer(
        source_rgb=source, text_clean_rgb=source,
        ownership_mask=visible, semantic_mask=semantic,
        higher_layer_mask=higher, text_mask=np.zeros_like(higher),
    )

    assert np.array_equal(layer["ownership_mask"], visible)
    assert np.array_equal(layer["rgb"][visible], source[visible])
    assert not np.any(layer["generated_underlay_mask"])
