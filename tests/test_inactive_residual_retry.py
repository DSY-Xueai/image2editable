import hashlib

import numpy as np
from PIL import Image

from image2editable.component_repair import (
    _page_residual_owner_ids,
    evaluate_component_quality_round,
)
from image2editable.store import RunStore
from scripts.visual_segment import execute_component_actions


def test_discarded_composite_unique_line_can_retry_and_pass_quality(tmp_path):
    shape = (120, 160)
    source = np.full((*shape, 3), 255, dtype=np.uint8)
    line = np.zeros(shape, dtype=bool)
    line[80:82, 30:130] = True
    icon = np.zeros(shape, dtype=bool)
    icon[20:40, 65:85] = True
    source[line | icon] = 30
    graph_dir = tmp_path / "input"
    (graph_dir / "masks").mkdir(parents=True)
    nodes = []
    for component_id, state, mask in (
        ("composite", "pending", line | icon),
        ("icon", "frozen", icon),
    ):
        path = graph_dir / "masks" / f"{component_id}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        ys, xs = np.nonzero(mask)
        nodes.append({
            "id": component_id, "kind": "parent", "parent_id": None,
            "state": state, "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "z_index": len(nodes), "text_ids": [],
        })

    def action(name, parameters):
        return {
            "action": name, "object_ids": ["composite"],
            "parameters": parameters, "confidence": 0.99,
            "evidence": ["The discarded composite contains a separate unique line."],
        }

    discarded_dir = tmp_path / "discarded"
    discarded = execute_component_actions(
        source, {"nodes": nodes}, [action("discard", {})],
        sam_runner=None, input_dir=graph_dir, output_dir=discarded_dir,
    )
    before = evaluate_component_quality_round(
        source, np.full_like(source, 255), source, discarded,
        graph_dir=discarded_dir, trusted_root=tmp_path,
        text_mask=np.zeros(shape, dtype=bool),
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
        initial_component_count=2, expected_component_ids=[],
        material_foreground=line | icon,
    )
    assert "unexplained_visual_residual" in before["violations"]
    assert before["visual_metrics"]["unexplained_visual_pixels"] == int(line.sum())
    residual_path = tmp_path / "residual.png"
    Image.fromarray(line.astype(np.uint8) * 255).save(residual_path)
    assert _page_residual_owner_ids(
        RunStore(tmp_path), graph=discarded, graph_root=discarded_dir,
        quality={"unexplained_mask_ref": {
            "path": residual_path.name,
            "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
        }},
    ) == set()

    output_dir = tmp_path / "retry"
    restored = execute_component_actions(
        source, discarded,
        [action("retry_with_box", {"box": [0.15, 0.6, 0.85, 0.75]})],
        sam_runner=lambda **_: line.copy(),
        input_dir=discarded_dir, output_dir=output_dir,
    )
    assert restored["nodes"][0]["state"] == "pending"
    assert restored["nodes"][1] == discarded["nodes"][1]
    recovered = np.asarray(Image.open(output_dir / restored["nodes"][0]["mask"])) > 0
    assert np.all(recovered[line])
    assert not np.any(recovered & icon)
    report = evaluate_component_quality_round(
        source, np.full_like(source, 255), source, restored,
        graph_dir=output_dir, trusted_root=tmp_path,
        text_mask=np.zeros(shape, dtype=bool),
        visual_metrics={"mae": 0.0, "p95": 0.0, "changed_ratio": 0.0},
        page_checks={"protected_native_overlap": "pass", "pptx_reopen": "pass"},
        initial_component_count=2, expected_component_ids=["composite"],
        material_foreground=line | icon,
    )
    assert report["accepted"], report["violations"]
    assert report["visual_metrics"]["unexplained_visual_pixels"] == 0
