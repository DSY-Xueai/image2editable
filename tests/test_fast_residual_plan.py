import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from image2editable import component_contracts, component_repair, host_agent, legacy
from image2editable.store import RunStore


def test_residual_owner_tie_matches_sorted_deterministic_actions(tmp_path):
    residual = np.zeros((30, 40), dtype=np.uint8)
    residual[10:20, 19:21] = 255
    residual_path = tmp_path / "residual.png"
    Image.fromarray(residual).save(residual_path)
    nodes = []
    for component_id, left, right, z_index in (("a", 10, 19, 0), ("b", 21, 30, 1)):
        mask = np.zeros_like(residual)
        mask[10:20, left:right] = 255
        path = tmp_path / f"{component_id}.png"
        Image.fromarray(mask).save(path)
        nodes.append({
            "id": component_id, "kind": "parent", "state": "pending",
            "bbox": [left, 10, right, 20], "z_index": z_index,
            "mask": path.name, "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    assert component_repair._page_residual_owner_ids(
        RunStore(tmp_path), graph={"nodes": nodes}, graph_root=tmp_path,
        quality={"unexplained_mask_ref": {
            "path": residual_path.name,
            "sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
        }},
    ) == {"a"}


def test_fast_plan_repairs_signed_residual_instead_of_only_accepting(tmp_path, monkeypatch):
    reconstruction = tmp_path / "pages/page_001/reconstruction"
    reconstruction.mkdir(parents=True)
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[8:22, 10:25] = 255
    residual = np.zeros_like(mask)
    residual[8:22, 25:27] = 255
    mask_path = reconstruction / "visual.png"
    residual_path = reconstruction / "residual.png"
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(residual).save(residual_path)

    def ref(path):
        return {"path": path.relative_to(tmp_path).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    quality_path = reconstruction / "quality.json"
    quality_path.write_text(json.dumps({
        "report": {"violations": ["unexplained_visual_residual"]},
        "unexplained_mask_ref": ref(residual_path),
    }), encoding="utf-8")
    request = {
        "repair_round": 2, "candidate_ids": ["visual"],
        "evidence": {"quality-report.json": {
            "path": quality_path.name, "sha256": ref(quality_path)["sha256"],
        }},
    }
    graph = {"nodes": [{
        "id": "visual", "kind": "parent", "state": "pending",
        "bbox": [10, 8, 25, 22], "z_index": 0,
        "mask": mask_path.name, "mask_sha256": ref(mask_path)["sha256"],
    }]}
    monkeypatch.setattr(legacy, "advance_component_repair", lambda *a, **kw: {})
    monkeypatch.setattr(component_repair, "load_component_agent_request", lambda path: request)
    monkeypatch.setattr(component_repair, "load_component_agent_graph", lambda path: graph)
    monkeypatch.setattr(host_agent, "_request_sha256", lambda request: "a" * 64)
    monkeypatch.setattr(host_agent, "_record_plan_reference", lambda *a, **kw: None)
    captured = {}
    monkeypatch.setattr(component_contracts, "validate_component_plan", lambda plan, **kw: captured.update(plan))

    legacy._record_deterministic_fast_plan(
        RunStore(tmp_path), "page_001", reconstruction / "request.json",
        reconstruction, _lease=object(),
    )

    repairs = [action for action in captured["actions"] if action["action"] == "absorb_residual"]
    assert [action["object_ids"] for action in repairs] == [["visual"]]
