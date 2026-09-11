"""Rebuild quality assets from a cached round without changing run state."""

from __future__ import annotations

import argparse
import json
import importlib
from pathlib import Path
import time

import numpy as np
from PIL import Image

from image2editable import component_repair, legacy
from image2editable.store import RunStore
from scripts.visual_segment import visual_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("page")
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--name", required=True)
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    store = RunStore.open(args.run.resolve())
    reconstruction = store.root / "pages" / args.page / "reconstruction"
    old_dir = reconstruction / f"execution-{args.round:02d}"
    old_quality = json.loads((old_dir / "component-quality.json").read_text(encoding="utf-8"))
    graph_path = old_dir / "component-graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    output = reconstruction / args.name
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    if args.repair:
        request_path = reconstruction / f"agent/round-{args.round:02d}/component_agent_request.json"
        request = component_repair.load_component_agent_request(request_path)
        graph = component_repair.load_component_agent_graph(request_path)
        quality_evidence = json.loads((request_path.parent / request["evidence"]["quality-report.json"]["path"]).read_text(encoding="utf-8"))
        owners = component_repair._page_residual_owner_ids(
            store, quality=quality_evidence,
            graph={"nodes": [node for node in graph["nodes"] if node["id"] in request["candidate_ids"]]},
            graph_root=request_path.parent,
        )
        actions = [{"action": "accept", "object_ids": [value], "parameters": {"preserve_mask": True}, "confidence": 1.0, "evidence": ["cached quality probe"]} for value in request["candidate_ids"]]
        actions.extend({"action": "absorb_residual", "object_ids": [value], "parameters": {}, "confidence": 1.0, "evidence": ["signed residual"]} for value in sorted(owners))
        graph_dir = output / "graph"
        source_path = legacy._state_artifact(store, old_quality["input_refs"]["source"])
        with Image.open(source_path) as image:
            source_pixels = np.asarray(image.convert("RGB")).copy()
        graph = component_repair.execute_component_action_round(
            source_pixels, graph, actions, input_dir=request_path.parent,
            output_dir=graph_dir,
        )
        old_dir = graph_dir
        module = importlib.import_module("image_to_ppt")
        prepared = module.load_component_layers(reconstruction / "initial/prepared_page.json")
        with Image.open(prepared["_text_clean_path"]) as image:
            clean = np.asarray(image.convert("RGB")).copy()
        with Image.open(prepared.get("_text_cleanup_mask_path", prepared["_text_mask_path"])) as image:
            mask = np.asarray(image.convert("L")) > 0
        _, mask, clean = legacy._effective_text_context(
            source=source_pixels, text_clean=clean, text_mask=mask,
            text_items=prepared["text_items"], graph=graph, graph_dir=graph_dir,
            refine_cleanup_mask="_text_cleanup_mask_path" in prepared,
        )
        clean_path, mask_path = output / "effective-clean.png", output / "effective-mask.png"
        Image.fromarray(clean).save(clean_path)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        legacy._rebuild_canvas_background(
            source_path=source_path, current_background_path=legacy._state_artifact(store, old_quality["input_refs"]["background"]),
            restore_background_path=clean_path,
            repair_requests=[(set(request["candidate_ids"]), 0.01)],
            graph=graph, graph_dir=graph_dir, text_mask_path=mask_path,
            output_path=output / "background-rebuilt.png", repair_all_active=True,
        )
    refs = legacy._quality_assets(
        store, args.page, graph, old_dir, output,
        previous_quality_refs=old_quality["input_refs"],
        background_rebuilt=args.repair,
    )
    def pixels(ref, mode):
        with Image.open(legacy._state_artifact(store, ref)) as image:
            return np.asarray(image.convert(mode)).copy()

    source = pixels(old_quality["input_refs"]["source"], "RGB")
    background = pixels(refs["background"], "RGB")
    reconstructed = pixels(refs["reconstructed"], "RGB")
    text_mask = pixels(refs["text_mask"], "L")
    native = json.loads(legacy._state_artifact(store, refs["native_check"]).read_text(encoding="utf-8"))
    manifest = json.loads(legacy._state_artifact(store, refs["presentation_manifest"]).read_text(encoding="utf-8"))
    report = component_repair.evaluate_component_quality_round(
        source, background, reconstructed, graph,
        graph_dir=old_dir, trusted_root=store.root,
        text_mask=text_mask, visual_metrics=visual_difference(source, reconstructed, text_mask),
        page_checks={
            "protected_native_overlap": native["protected_native_overlap"],
            "pptx_reopen": "unknown",
            "unowned_raster_text": component_repair._unowned_raster_text_check(
                native["initial_diagnostics"], native["text_items"],
            ),
        },
        initial_component_count=old_quality["initial_component_count"],
        expected_component_ids=[node["id"] for node in graph["nodes"] if node["kind"] != "text" and node["state"] in {"pending", "pending_gate"}],
        text_items=native["text_items"],
        presentation_layers=component_repair._iter_quality_presentation_layers(
            run_root=store.root, reconstruction=reconstruction, manifest=manifest,
            page_shape=source.shape[:2],
        ),
        material_foreground=pixels(refs["foreground_evidence"], "L") if "foreground_evidence" in refs else None,
        unexplained_output_path=output / "unexplained-mask.png",
    )
    report["duration_s"] = time.perf_counter() - started
    report["input_refs"] = refs
    (output / "probe-quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "duration_s": report["duration_s"], "violations": report.get("violations"), "page_quality": report.get("page_quality")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
