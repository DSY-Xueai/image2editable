import hashlib

import numpy as np
import pytest
from PIL import Image

from image2editable import legacy


def test_text_refinement_survives_visual_component_protection(tmp_path):
    source = np.full((80, 120, 3), (224, 196, 155), dtype=np.uint8)
    ink = np.zeros((80, 120), dtype=bool)
    ink[30:49, 43:48] = True
    ink[30:35, 43:63] = True
    source[ink] = (40, 30, 20)
    visual = np.ones_like(ink)
    nodes = []
    for component_id, kind, mask in (
        ("text_0001", "text", ink),
        ("sign", "parent", visual),
    ):
        path = tmp_path / f"{component_id}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        nodes.append({
            "id": component_id, "kind": kind, "state": "frozen",
            "mask": path.name, "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    items = [{"text": "F", "box": [40, 27, 26, 25]}]
    _, mask_without_visual, clean_without_visual = legacy._effective_text_context(
        source=source, text_clean=source.copy(), text_mask=ink,
        text_items=items, graph={"nodes": nodes[:1]}, graph_dir=tmp_path,
        refine_cleanup_mask=True,
    )
    _, mask_with_visual, clean_with_visual = legacy._effective_text_context(
        source=source, text_clean=source.copy(), text_mask=ink,
        text_items=items, graph={"nodes": nodes}, graph_dir=tmp_path,
        refine_cleanup_mask=True,
    )

    assert np.array_equal(mask_with_visual, mask_without_visual)
    assert np.array_equal(clean_with_visual[ink], clean_without_visual[ink])
    assert np.max(np.abs(clean_with_visual[ink].astype(int) - [224, 196, 155])) <= 2
    assert np.array_equal(clean_with_visual[~mask_with_visual], source[~mask_with_visual])


def test_background_repair_margin_does_not_overwrite_neighbor_texture(tmp_path):
    yy, xx = np.indices((80, 120))
    source = np.stack((180 + xx % 23, 190 + yy % 19, 200 + (xx + yy) % 17), axis=2).astype(np.uint8)
    mask = np.zeros((80, 120), dtype=bool)
    mask[25:55, 40:75] = True
    source[mask] = (30, 50, 80)
    source_path = tmp_path / "source.png"
    text_path = tmp_path / "text.png"
    mask_path = tmp_path / "visual.png"
    output = tmp_path / "rebuilt.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    Image.new("L", (120, 80), 0).save(text_path)
    graph = {"nodes": [{
        "id": "visual", "kind": "parent", "state": "pending",
        "parent_id": None, "z_index": 0, "text_ids": [],
        "bbox": [40, 25, 75, 55], "mask": mask_path.name,
        "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
    }]}

    legacy._rebuild_canvas_background(
        source_path=source_path, current_background_path=source_path,
        restore_background_path=source_path, repair_requests=[({"visual"}, 0.05)],
        graph=graph, graph_dir=tmp_path, text_mask_path=text_path,
        output_path=output,
    )

    with Image.open(output) as image:
        rebuilt = np.asarray(image)
    assert np.array_equal(rebuilt[~mask], source[~mask])
    assert not np.array_equal(rebuilt[mask], source[mask])


@pytest.mark.parametrize("explicit_text_request", [False, True])
def test_background_rebuild_does_not_trust_stale_text_clean_pixels(tmp_path, explicit_text_request):
    source = np.full((80, 120, 3), (224, 196, 155), dtype=np.uint8)
    ink = np.zeros(source.shape[:2], dtype=bool)
    ink[30:49, 43:48] = True
    ink[30:35, 43:63] = True
    source[ink] = (40, 30, 20)
    source_path = tmp_path / "source.png"
    text_path = tmp_path / "text.png"
    output = tmp_path / "rebuilt.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(ink.astype(np.uint8) * 255).save(text_path)
    node = {
        "id": "text", "kind": "text", "state": "frozen",
        "parent_id": None, "z_index": 0, "text_ids": [],
        "bbox": [43, 30, 63, 49], "mask": text_path.name,
        "mask_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
    }

    legacy._rebuild_canvas_background(
        source_path=source_path, current_background_path=source_path,
        restore_background_path=source_path,
        repair_requests=[({"text"}, 0.01)] if explicit_text_request else [],
        graph={"nodes": [node] if explicit_text_request else []},
        graph_dir=tmp_path, text_mask_path=text_path,
        output_path=output,
    )

    with Image.open(output) as image:
        rebuilt = np.asarray(image)
    assert np.max(np.abs(rebuilt[ink].astype(int) - [224, 196, 155])) <= 3
    assert np.array_equal(rebuilt[~ink], source[~ink])
