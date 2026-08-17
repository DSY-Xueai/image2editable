from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import pytest

from image2editable import local_agent, local_agent_worker
from image2editable.component_repair import (
    EVIDENCE_NAMES,
    build_component_agent_request,
    load_component_agent_request,
)


ROOT = Path(__file__).resolve().parents[1]


def test_local_prompt_requires_independently_movable_leaf_components() -> None:
    prompt = local_agent_worker.SYSTEM_PROMPT

    assert "semantic relationship does not justify merging" in prompt
    assert "independently moved" in prompt
    assert (
        "same physical entity: duplicate masks, edge fragments, shadows, "
        "or segmentation gaps"
    ) in prompt
    assert "semantic parent is grouping-only and non-rendering" in prompt
    assert "glyph-shaped transparent holes" in prompt
    assert "collapse_to_parent" in prompt
    assert "contained parent candidates" in prompt
    assert "Prefer preserving one complete parent" not in prompt
    assert "Default to accept" in prompt
    assert "two or more visibly disconnected" in prompt
    assert "Never use split merely because" in prompt
    assert "component-isolation.png" in local_agent_worker._IMAGE_EVIDENCE
    assert "without OCR text pixels" in prompt


def test_host_skill_limits_absorb_to_one_physical_entity() -> None:
    text = (ROOT / "skills/image-to-ppt/SKILL.md").read_text(encoding="utf-8")

    assert "同一物理实体" in text
    assert "重复掩码" in text
    assert "碎边" in text
    assert "阴影" in text
    assert "分割缺口" in text
    assert "语义父级只用于分组，不参与最终像素渲染" in text


def test_host_skill_requires_residual_driven_repairs() -> None:
    text = (ROOT / "skills/image-to-ppt/SKILL.md").read_text(encoding="utf-8")

    assert "unexplained_visual_residual" in text
    assert "unexplained-mask.png" in text
    assert "background_text_residual" in text


def _request_path(
    tmp_path: Path,
    *,
    repair_round: int = 1,
    provider: str = "local",
) -> Path:
    reconstruction = tmp_path / "pages" / "page_001" / "reconstruction"
    evidence_root = reconstruction / "evidence-source"
    masks = evidence_root / "masks"
    masks.mkdir(parents=True)
    mask = masks / "component_0001.png"
    Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(mask)
    graph = {
        "nodes": [
            {
                "id": "component_0001",
                "kind": "parent",
                "parent_id": None,
                "state": "pending",
                "mask": "masks/component_0001.png",
                "mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
                "bbox": [0, 0, 4, 4],
                "z_index": 0,
                "text_ids": [],
            }
        ]
    }
    evidence = {}
    for name in EVIDENCE_NAMES:
        path = evidence_root / name
        if name == "component-graph.json":
            path.write_text(json.dumps(graph), encoding="utf-8")
        elif name == "quality-report.json":
            quality = (
                {"report": {"violations": []}}
                if repair_round > 1 else {"violations": []}
            )
            path.write_text(json.dumps(quality), encoding="utf-8")
        elif name == "presentation-manifest.json":
            continue
        else:
            Image.fromarray(np.full((4, 4), 127, dtype=np.uint8)).save(path)
        evidence[name] = path
    assets = evidence_root / "presentation-assets"
    assets.mkdir()
    references = {}
    for name in (
        "rgba", "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask",
    ):
        path = assets / f"{name}.png"
        Image.fromarray(np.full((4, 4), 127, dtype=np.uint8)).save(path)
        references[name] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = evidence_root / "presentation-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "source_sha256": hashlib.sha256(evidence["source.png"].read_bytes()).hexdigest(),
        "graph_sha256": hashlib.sha256(
            evidence["component-graph.json"].read_bytes()
        ).hexdigest(),
        "components": [{
            "component_id": "component_0001", **references,
            "metrics": {"boundary_color_mae": 0.0},
        }],
    }), encoding="utf-8")
    evidence["presentation-manifest.json"] = manifest
    return build_component_agent_request(
        {
            "page_id": "page_001",
            "provider": provider,
            "reconstruction_dir": reconstruction,
            "evidence": evidence,
        },
        repair_round=repair_round,
    )


def _message_image_names(request_path: Path) -> list[str]:
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    messages = local_agent_worker._messages(
        request,
        json.loads(evidence["component-graph.json"].read_text(encoding="utf-8")),
        evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence,
        hashlib.sha256(request_path.read_bytes()).hexdigest(),
    )
    return [
        Path(item["image"]).name
        for item in messages[1]["content"]
        if item["type"] == "image"
    ]


def test_local_message_keeps_complete_request_and_graph_json(tmp_path: Path) -> None:
    request_path = _request_path(tmp_path, repair_round=2)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(evidence["component-graph.json"].read_text(encoding="utf-8"))

    messages = local_agent_worker._messages(
        request, graph, evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence, hashlib.sha256(request_path.read_bytes()).hexdigest(),
    )
    prompt = json.loads(messages[1]["content"][0]["text"].splitlines()[1])

    assert prompt["component_request"] == {
        "page_id": request["page_id"],
        "provider": request["provider"],
        "repair_round": request["repair_round"],
        "review_evidence": request["review_evidence"],
    }
    fields = prompt["component_graph"]["node_fields"]
    assert fields == [
        "id", "kind", "parent_id", "state", "bbox", "z_index", "text_ids",
    ]
    assert [dict(zip(fields, row, strict=True)) for row in prompt[
        "component_graph"
    ]["nodes"]] == [
        {field: node[field] for field in fields}
        for node in graph["nodes"]
    ]
    assert "action_codes" not in prompt
    assert "mask_sha256" not in messages[1]["content"][0]["text"]
    assert messages[0]["content"] == [
        {"type": "text", "text": local_agent_worker.SYSTEM_PROMPT}
    ]
    assert "exactly one field: actions" in local_agent_worker.SYSTEM_PROMPT


def test_component_action_scopes_keep_siblings_together() -> None:
    graph = {
        "nodes": [
            {
                "id": f"component_{index:04d}",
                "kind": "child",
                "parent_id": f"parent_{(index - 1) // 2:04d}",
            }
            for index in range(1, 11)
        ]
    }
    request = {
        "candidate_ids": sorted(node["id"] for node in graph["nodes"]),
    }

    scopes = local_agent_worker._component_action_scopes(
        request, graph, max_candidates=4
    )

    assert [len(scope) for scope in scopes] == [4, 4, 2]
    for index in range(0, 10, 2):
        siblings = {f"component_{index + 1:04d}", f"component_{index + 2:04d}"}
        assert sum(siblings <= scope for scope in scopes) == 1

    assert [len(scope) for scope in local_agent_worker._component_action_scopes(
        request, graph
    )] == [2, 2, 2, 2, 2]


def test_component_batch_keeps_only_linked_or_quality_referenced_context(
    tmp_path: Path,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    base = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )["nodes"][0]
    graph = {"nodes": [
        {
            **base, "id": "component_0001", "bbox": [0, 0, 20, 20],
            "text_ids": ["text_overlap"],
        },
        {**base, "id": "parent_unrelated", "state": "inactive", "bbox": [80, 80, 90, 90]},
        {
            **base, "id": "text_overlap", "kind": "text", "state": "frozen",
            "bbox": [5, 5, 10, 10], "z_index": 1,
        },
        {
            **base, "id": "text_quality", "kind": "text", "state": "frozen",
            "bbox": [50, 50, 60, 60], "z_index": 2,
        },
        {
            **base, "id": "text_unrelated", "kind": "text", "state": "frozen",
            "bbox": [70, 70, 75, 75], "z_index": 3,
        },
    ]}
    request = {
        **request,
        "candidate_ids": ["component_0001"],
        "frozen_ids": ["text_overlap", "text_quality", "text_unrelated"],
    }

    messages = local_agent_worker._messages(
        request,
        graph,
        json.dumps({
            "text_items": [
                {"id": "text_overlap", "text": "keep spatial"},
                {"_component_id": "text_quality", "text": "keep quality"},
                {"_component_id": "text_unrelated", "text": "drop unrelated"},
            ],
            "violations": [{"object_id": "text_quality"}],
            "report": {
                "component_reports": [
                    {
                        "component_id": "component_0001",
                        "violations": [],
                        "metrics": {"component_pixels": 5, "noise_l1": "drop"},
                    },
                    {"component_id": "parent_unrelated", "violations": ["empty"]},
                ],
                "violations": ["visual_difference"],
                "visual_metrics": {"mae": 1.0},
            },
        }),
        evidence,
        "1" * 64,
        action_scope={"component_0001"},
    )
    prompt = json.loads(messages[1]["content"][0]["text"].splitlines()[1])
    fields = prompt["component_graph"]["node_fields"]
    ids = {
        dict(zip(fields, row, strict=True))["id"]
        for row in prompt["component_graph"]["nodes"]
    }

    assert ids == {"component_0001", "text_overlap", "text_quality"}
    assert [
        item.get("id", item.get("_component_id"))
        for item in prompt["quality_report_untrusted"]["text_items"]
    ] == ["text_overlap", "text_quality"]
    assert prompt["quality_report_untrusted"]["report"] == {
        "component_reports": [
            {
                "component_id": "component_0001",
                "violations": [],
                "metrics": {"component_pixels": 5},
            }
        ],
        "violations": ["visual_difference"],
        "visual_metrics": {"mae": 1.0},
    }


def _replace_evidence_image(
    request: dict,
    evidence: dict[str, Path],
    name: str,
    image: Image.Image,
) -> None:
    image.save(evidence[name])
    request["evidence"][name]["sha256"] = hashlib.sha256(
        evidence[name].read_bytes()
    ).hexdigest()


def _crop_fixture(tmp_path: Path):
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    base = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )["nodes"][0]
    graph = {"nodes": [
        {**base, "id": "component_0001", "bbox": [40, 35, 50, 45]},
        {**base, "id": "component_0002", "bbox": [70, 55, 80, 65], "z_index": 1},
    ]}
    request = {
        **request,
        "candidate_ids": ["component_0001", "component_0002"],
    }
    pixels = np.arange(200 * 160, dtype=np.uint16).reshape(160, 200)
    pixels = (pixels % 256).astype(np.uint8)
    for name in request["review_evidence"]:
        if name.endswith(".png") and name != "component-isolation.png":
            _replace_evidence_image(
                request, evidence, name, Image.fromarray(pixels, mode="L")
            )
    isolation = Image.new("RGB", (640, 240), "red")
    isolation.paste("blue", (320, 0, 640, 240))
    _replace_evidence_image(
        request, evidence, "component-isolation.png", isolation
    )
    return request, evidence, graph, pixels


def test_component_batch_crops_same_coordinate_evidence_and_isolation_cells(
    tmp_path: Path,
) -> None:
    request, evidence, graph, pixels = _crop_fixture(tmp_path)

    images, max_pixels = local_agent_worker._batch_evidence_images(
        request,
        graph,
        {"violations": []},
        evidence,
        {"component_0001"},
    )
    try:
        assert max_pixels == 98_304
        page_images = {
            name: image
            for name, image in images.items()
            if name != "component-isolation.png"
        }
        assert {image.size for image in page_images.values()} == {(74, 74)}
        assert np.array_equal(
            np.asarray(page_images["source.png"]), pixels[3:77, 8:82]
        )
        isolation = images["component-isolation.png"]
        assert isolation.size == (320, 240)
        assert isolation.getpixel((10, 10)) == (255, 0, 0)
    finally:
        for image in images.values():
            if isinstance(image, Image.Image):
                image.close()


def test_component_batch_crop_covers_ancestors_linked_text_and_quality_nodes(
    tmp_path: Path,
) -> None:
    request, evidence, graph, _ = _crop_fixture(tmp_path)
    base = graph["nodes"][0]
    graph["nodes"] = [
        {**base, "id": "parent", "state": "inactive", "bbox": [5, 5, 10, 10]},
        {
            **base, "id": "component_0001", "parent_id": "parent",
            "bbox": [40, 35, 50, 45], "text_ids": ["text_linked"],
        },
        {**base, "id": "text_linked", "kind": "text", "state": "frozen", "bbox": [60, 5, 65, 10]},
        {**base, "id": "component_0002", "bbox": [70, 55, 80, 65]},
    ]
    quality = {"violations": [{"object_id": "component_0002"}]}

    assert local_agent_worker._included_node_ids(
        graph, {"component_0001"}, quality
    ) == {"parent", "component_0001", "text_linked", "component_0002"}


def test_component_batch_crop_includes_bound_unexplained_residual(
    tmp_path: Path,
) -> None:
    request, evidence, graph, _ = _crop_fixture(tmp_path)
    residual = Image.new("L", (200, 160), 0)
    residual.putpixel((190, 145), 255)
    _replace_evidence_image(request, evidence, "unexplained-mask.png", residual)

    images, max_pixels = local_agent_worker._batch_evidence_images(
        request,
        graph,
        {"report": {"violations": ["unexplained_visual_residual"]}},
        evidence,
        {"component_0001"},
    )
    try:
        assert max_pixels is None
        assert images["source.png"].size == (192, 157)
        assert images["unexplained-mask.png"].getpixel((182, 142)) == 255
    finally:
        for image in images.values():
            if isinstance(image, Image.Image):
                image.close()


def test_component_batch_crop_falls_back_to_full_evidence_on_size_mismatch(
    tmp_path: Path,
) -> None:
    request, evidence, graph, _ = _crop_fixture(tmp_path)
    _replace_evidence_image(
        request, evidence, "difference.png", Image.new("L", (199, 160), 0)
    )

    images, max_pixels = local_agent_worker._batch_evidence_images(
        request, graph, {"violations": []}, evidence, {"component_0001"}
    )

    assert images == {
        name: str(evidence[name])
        for name in request["review_evidence"]
        if name.endswith(".png")
    }
    assert max_pixels is None


def test_component_batch_crop_falls_back_on_out_of_page_node_geometry(
    tmp_path: Path,
) -> None:
    request, evidence, graph, _ = _crop_fixture(tmp_path)
    graph["nodes"][0]["bbox"] = [40, 35, 201, 45]

    images, max_pixels = local_agent_worker._batch_evidence_images(
        request, graph, {"violations": []}, evidence, {"component_0001"}
    )

    assert images["source.png"] == str(evidence["source.png"])
    assert max_pixels is None


def test_component_batch_crop_keeps_bound_evidence_tamper_fatal(
    tmp_path: Path,
) -> None:
    request, evidence, graph, _ = _crop_fixture(tmp_path)
    evidence["source.png"].write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="evidence hash mismatch"):
        local_agent_worker._batch_evidence_images(
            request, graph, {"violations": []}, evidence, {"component_0001"}
        )


@pytest.mark.parametrize("generation_fails", [False, True])
def test_component_batch_closes_cropped_images_after_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_fails: bool,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    cropped = {
        name: Image.new("RGB", (8, 8), "red")
        for name in request["review_evidence"]
        if name.endswith(".png")
    }
    monkeypatch.setattr(
        local_agent_worker, "_batch_evidence_images", lambda *args: (cropped, 8192)
    )
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda snapshot, processor_size: ("processor", "model"),
    )

    def generate(*args, **kwargs):
        assert kwargs["max_pixels"] == 8192
        if generation_fails:
            raise RuntimeError("inference failed")
        return json.dumps({
            "actions": [[
                "accept", [request["candidate_ids"][0]], {}, 0.95, 0,
            ]],
        })

    monkeypatch.setattr(local_agent_worker, "_generate_with_model", generate)

    if generation_fails:
        with pytest.raises(RuntimeError, match="inference failed"):
            local_agent_worker._generate_component_plan(
                request,
                graph,
                evidence["quality-report.json"].read_text(encoding="utf-8"),
                evidence,
                "1" * 64,
                Path("snapshot"),
                max_candidates=1,
            )
    else:
        local_agent_worker._generate_component_plan(
            request,
            graph,
            evidence["quality-report.json"].read_text(encoding="utf-8"),
            evidence,
            "1" * 64,
            Path("snapshot"),
            max_candidates=1,
        )
    for image in cropped.values():
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def _empty_component_report(component_id: str) -> dict[str, object]:
    return {
        "accepted": False,
        "component_id": component_id,
        "metrics": {"component_pixels": 0},
        "violations": ["empty_component"],
    }


def test_component_plan_skips_model_when_all_candidates_are_proven_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, evidence, graph, _ = _crop_fixture(tmp_path)
    quality = {
        "report": {
            "component_reports": [
                _empty_component_report(component_id)
                for component_id in request["candidate_ids"]
            ],
        },
    }
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda *args: pytest.fail("proven empty candidates must not load the model"),
    )

    plan = local_agent_worker._generate_component_plan(
        request,
        graph,
        json.dumps(quality),
        evidence,
        "1" * 64,
        Path("snapshot"),
    )

    assert plan["actions"] == [
        {
            "action": "discard",
            "object_ids": [component_id],
            "parameters": {},
            "confidence": 1.0,
            "evidence": ["quality-report.json"],
        }
        for component_id in request["candidate_ids"]
    ]


def test_component_plan_sends_only_ambiguous_candidates_to_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, evidence, graph, _ = _crop_fixture(tmp_path)
    quality = {
        "report": {
            "component_reports": [
                _empty_component_report("component_0001"),
                {
                    "accepted": False,
                    "component_id": "component_0002",
                    "metrics": {"component_pixels": 10},
                    "violations": ["background_text_residual"],
                },
            ],
        },
    }
    observed = []
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda *args: ("processor", "model"),
    )

    def generate(processor, model, messages, **kwargs):
        prompt = json.loads(messages[1]["content"][0]["text"].splitlines()[1])
        observed.append(prompt["action_scope"]["candidate_ids"])
        return json.dumps({
            "actions": [["accept", ["component_0002"], {}, 0.95, 0]],
        })

    monkeypatch.setattr(local_agent_worker, "_generate_with_model", generate)

    plan = local_agent_worker._generate_component_plan(
        request,
        graph,
        json.dumps(quality),
        evidence,
        "1" * 64,
        Path("snapshot"),
    )

    assert observed == [["component_0002"]]
    assert [(action["action"], action["object_ids"]) for action in plan["actions"]] == [
        ("discard", ["component_0001"]),
        ("accept", ["component_0002"]),
    ]


@pytest.mark.parametrize(
    "report",
    [
        {**_empty_component_report("component_0001"), "violations": ["empty_component", "duplicate_pixels"]},
        {**_empty_component_report("component_0001"), "metrics": {"component_pixels": 1}},
        {**_empty_component_report("component_0001"), "accepted": True},
    ],
)
def test_component_plan_does_not_deterministically_discard_ambiguous_reports(
    report: dict[str, object],
) -> None:
    assert local_agent_worker._deterministic_empty_actions(
        ["component_0001"], {"report": {"component_reports": [report]}}
    ) == []


def test_component_plan_deterministically_rebuilds_background_only_failure() -> None:
    graph = {"nodes": [{
        "id": "component_0001",
        "bbox": [10, 20, 110, 70],
    }]}
    quality = {"report": {"component_reports": [{
        "component_id": "component_0001",
        "accepted": False,
        "violations": ["background_text_residual"],
        "metrics": {"text_halo_px": 4},
    }]}}

    assert local_agent_worker._deterministic_background_actions(
        ["component_0001"], graph, quality
    ) == [{
        "action": "rebuild_background",
        "object_ids": ["component_0001"],
        "parameters": {"margin_ratio": 0.08},
        "confidence": 1.0,
        "evidence": ["quality-report.json"],
    }]


def test_component_plan_deterministically_handles_proven_quality_outcomes() -> None:
    graph = {"nodes": [
        {
            "id": "accepted", "kind": "parent", "parent_id": None,
            "bbox": [0, 0, 20, 20],
        },
        {
            "id": "incomplete", "kind": "child", "parent_id": "parent",
            "bbox": [20, 0, 40, 20],
        },
        {
            "id": "parent", "kind": "parent", "parent_id": None,
            "bbox": [20, 0, 40, 20],
        },
    ]}
    quality = {"report": {"component_reports": [
        {
            "component_id": "accepted", "accepted": True,
            "violations": [], "metrics": {},
        },
        {
            "component_id": "incomplete", "accepted": False,
            "violations": ["incomplete_child"], "metrics": {},
        },
    ]}}

    assert local_agent_worker._deterministic_quality_actions(
        ["accepted", "incomplete"], graph, quality
    ) == [
        {
            "action": "accept", "object_ids": ["accepted"],
            "parameters": {}, "confidence": 1.0,
            "evidence": ["quality-report.json"],
        },
        {
            "action": "absorb_into_parent",
            "object_ids": ["parent", "incomplete"],
            "parameters": {}, "confidence": 1.0,
            "evidence": ["quality-report.json"],
        },
    ]


@pytest.mark.parametrize(
    "report,node",
    [
        (
            {"component_id": "candidate", "accepted": True,
             "violations": ["visual_difference"], "metrics": {}},
            {"id": "candidate", "kind": "parent", "parent_id": None,
             "bbox": [0, 0, 20, 20]},
        ),
        (
            {"component_id": "candidate", "accepted": False,
             "violations": ["incomplete_child"], "metrics": {}},
            {"id": "candidate", "kind": "parent", "parent_id": None,
             "bbox": [0, 0, 20, 20]},
        ),
        (
            {"component_id": "candidate", "accepted": False,
             "violations": ["incomplete_child", "visual_difference"],
             "metrics": {}},
            {"id": "candidate", "kind": "child", "parent_id": "parent",
             "bbox": [0, 0, 20, 20]},
        ),
    ],
)
def test_component_plan_keeps_ambiguous_quality_outcomes_for_agent(
    report: dict[str, object], node: dict[str, object],
) -> None:
    assert local_agent_worker._deterministic_quality_actions(
        ["candidate"], {"nodes": [node]},
        {"report": {"component_reports": [report]}},
    ) == []


def test_component_plan_deterministically_detaches_underlay_seam_child() -> None:
    graph = {"nodes": [
        {
            "id": "component_0001", "kind": "child",
            "parent_id": "parent_0001", "bbox": [10, 20, 110, 70],
        },
        {
            "id": "parent_0001", "kind": "parent",
            "parent_id": None, "bbox": [10, 20, 110, 70],
        },
    ]}
    quality = {"report": {"component_reports": [{
        "component_id": "component_0001", "accepted": False,
        "violations": ["underlay_gradient_break", "underlay_seam"],
        "metrics": {},
    }]}}

    assert local_agent_worker._deterministic_quality_actions(
        ["component_0001"], graph, quality
    ) == [{
        "action": "accept",
        "object_ids": ["component_0001"],
        "parameters": {"independent": True},
        "confidence": 1.0,
        "evidence": ["quality-report.json"],
    }]
    assert local_agent_worker._deterministic_background_actions(
        ["component_0001"], graph, quality
    ) == []


def test_component_plan_deterministically_absorbs_owned_page_residual() -> None:
    graph = {"nodes": [{
        "id": "candidate", "kind": "parent", "parent_id": None,
        "bbox": [10, 20, 110, 70],
    }]}
    quality = {"report": {
        "accepted": False,
        "checks": {"visual_ownership": "fail"},
        "component_reports": [{
            "component_id": "candidate", "accepted": True,
            "violations": [], "metrics": {},
        }],
        "violations": [
            "pptx_reopen_unknown", "unexplained_visual_residual",
            "visual_difference",
        ],
        "visual_metrics": {"unexplained_visual_pixels": 23},
    }}

    assert local_agent_worker._deterministic_quality_actions(
        ["candidate"], graph, quality
    ) == [
        {
            "action": "accept", "object_ids": ["candidate"],
            "parameters": {}, "confidence": 1.0,
            "evidence": ["quality-report.json"],
        },
        {
            "action": "absorb_residual", "object_ids": ["candidate"],
            "parameters": {}, "confidence": 1.0,
            "evidence": ["quality-report.json"],
        },
    ]


def test_component_plan_skips_model_for_proven_accepted_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    quality = {"report": {"component_reports": [{
        "component_id": request["candidate_ids"][0],
        "accepted": True,
        "violations": [],
        "metrics": {},
    }]}}
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda *args: pytest.fail("accepted component must not load the model"),
    )

    plan = local_agent_worker._generate_component_plan(
        request, graph, json.dumps(quality), evidence, "1" * 64, Path("snapshot")
    )

    assert [action["action"] for action in plan["actions"]] == ["accept"]


@pytest.mark.parametrize(
    "report",
    [
        {
            "component_id": "component_0001", "accepted": False,
            "violations": ["background_text_residual", "duplicate_pixels"],
            "metrics": {"text_halo_px": 4},
        },
        {
            "component_id": "component_0001", "accepted": True,
            "violations": ["background_text_residual"],
            "metrics": {"text_halo_px": 4},
        },
        {
            "component_id": "component_0001", "accepted": False,
            "violations": ["background_text_residual"],
            "metrics": {"text_halo_px": 0},
        },
    ],
)
def test_component_plan_keeps_ambiguous_background_repairs_for_agent(
    report: dict[str, object],
) -> None:
    assert local_agent_worker._deterministic_background_actions(
        ["component_0001"],
        {"nodes": [{"id": "component_0001", "bbox": [0, 0, 100, 50]}]},
        {"report": {"component_reports": [report]}},
    ) == []


def test_component_plan_skips_model_for_proven_background_only_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    component_id = request["candidate_ids"][0]
    quality = {"report": {"component_reports": [{
        "component_id": component_id,
        "accepted": False,
        "violations": ["background_text_residual"],
        "metrics": {"text_halo_px": 1},
    }]}}
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda *args: pytest.fail("proven background repair must not load the model"),
    )

    plan = local_agent_worker._generate_component_plan(
        request, graph, json.dumps(quality), evidence, "1" * 64, Path("snapshot")
    )

    assert [action["action"] for action in plan["actions"]] == [
        "rebuild_background"
    ]


def test_component_batches_reuse_model_and_merge_only_scoped_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    base_node = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )["nodes"][0]
    graph = {
        "nodes": [
            {**base_node, "id": f"component_{index:04d}", "z_index": index - 1}
            for index in range(1, 4)
        ]
    }
    request = {**request, "candidate_ids": [node["id"] for node in graph["nodes"]]}
    loads = []
    prompts = []

    def load(snapshot, processor_size):
        loads.append((snapshot, processor_size))
        return "processor", "model"

    def generate(processor, model, messages, *, max_new_tokens, max_pixels):
        prompt = json.loads(messages[1]["content"][0]["text"].splitlines()[1])
        prompts.append((processor, model, prompt, max_new_tokens))
        return json.dumps({
            "actions": [
                ["accept", [component_id], {}, 0.95, 0]
                for component_id in prompt["action_scope"]["candidate_ids"]
            ],
        })

    monkeypatch.setattr(local_agent_worker, "_load_generator", load)
    monkeypatch.setattr(local_agent_worker, "_generate_with_model", generate)

    plan = local_agent_worker._generate_component_plan(
        request,
        graph,
        evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence,
        "1" * 64,
        Path("snapshot"),
        max_candidates=2,
    )

    assert len(loads) == 1
    assert len(prompts) == 2
    assert [prompt[2]["action_scope"]["candidate_ids"] for prompt in prompts] == [
        ["component_0001", "component_0002"],
        ["component_0003"],
    ]
    assert [prompt[2]["action_scope"]["maximum_actions"] for prompt in prompts] == [
        2,
        1,
    ]
    assert all(
        "Never return alternative actions" in prompt[2]["action_scope"]["rule"]
        for prompt in prompts
    )
    assert all(
        "Never reference IDs created by another action"
        in prompt[2]["action_scope"]["rule"]
        for prompt in prompts
    )
    assert [action["object_ids"] for action in plan["actions"]] == [
        ["component_0001"], ["component_0002"], ["component_0003"]
    ]


def test_component_batch_rejects_action_outside_its_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    base_node = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )["nodes"][0]
    graph = {
        "nodes": [
            {**base_node, "id": f"component_{index:04d}", "z_index": index - 1}
            for index in range(1, 3)
        ]
    }
    request = {**request, "candidate_ids": [node["id"] for node in graph["nodes"]]}
    monkeypatch.setattr(
        local_agent_worker, "_load_generator", lambda *args: (object(), object())
    )
    monkeypatch.setattr(
        local_agent_worker,
        "_generate_with_model",
        lambda *args, **kwargs: json.dumps({
            "actions": [["accept", ["component_0002"], {}, 0.95, 0]],
        }),
    )

    with pytest.raises(ValueError, match="outside its batch scope"):
        local_agent_worker._generate_component_plan(
            request,
            graph,
            evidence["quality-report.json"].read_text(encoding="utf-8"),
            evidence,
            "2" * 64,
            Path("snapshot"),
            max_candidates=1,
        )


def test_component_batch_rejects_more_actions_than_scoped_candidates(
    tmp_path: Path,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    action = {
        "action": "accept",
        "object_ids": [request["candidate_ids"][0]],
        "parameters": {},
        "confidence": 0.95,
        "evidence": ["complete visible component boundary"],
    }
    plan = {
        "schema_version": 1,
        "kind": "component_plan",
        "page_id": request["page_id"],
        "provider": request["provider"],
        "repair_round": request["repair_round"],
        "request_sha256": "1" * 64,
        "actions": [action, action],
    }

    with pytest.raises(ValueError, match="too many actions"):
        local_agent_worker._validate_batch_scope(
            plan,
            {request["candidate_ids"][0]},
            request,
            graph,
            allow_global_actions=True,
        )


def test_component_batches_adapt_once_and_reuse_the_narrower_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    base_node = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )["nodes"][0]
    graph = {"nodes": [
        {**base_node, "id": f"component_{index:04d}", "z_index": index - 1}
        for index in range(1, 9)
    ]}
    request = {**request, "candidate_ids": [node["id"] for node in graph["nodes"]]}
    loads = []
    attempts = []

    def load(snapshot, processor_size):
        loads.append((snapshot, processor_size))
        return "processor", "model"

    def generate(processor, model, messages, *, max_new_tokens, max_pixels):
        prompt = json.loads(messages[1]["content"][0]["text"].splitlines()[1])
        context_ids = [
            dict(zip(prompt["component_graph"]["node_fields"], row, strict=True))["id"]
            for row in prompt["component_graph"]["nodes"]
        ]
        action_ids = prompt["action_scope"]["candidate_ids"]
        attempts.append((action_ids, context_ids, max_new_tokens))
        if len(attempts) == 1:
            return "{"
        return json.dumps({
            "actions": [
                ["accept", [component_id], {}, 0.95, 0]
                for component_id in action_ids
            ],
        })

    monkeypatch.setattr(local_agent_worker, "_load_generator", load)
    monkeypatch.setattr(local_agent_worker, "_generate_with_model", generate)

    plan = local_agent_worker._generate_component_plan(
        request,
        graph,
        evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence,
        "1" * 64,
        Path("snapshot"),
        max_candidates=4,
    )

    assert len(loads) == 1
    assert [attempt[0] for attempt in attempts] == [
        ["component_0001", "component_0002", "component_0003", "component_0004"],
        ["component_0001", "component_0002"],
        ["component_0003", "component_0004"],
        ["component_0005", "component_0006"],
        ["component_0007", "component_0008"],
    ]
    assert [attempt[1] for attempt in attempts] == [
        ["component_0001", "component_0002", "component_0003", "component_0004"],
        ["component_0001", "component_0002"],
        ["component_0003", "component_0004"],
        ["component_0005", "component_0006"],
        ["component_0007", "component_0008"],
    ]
    assert [attempt[2] for attempt in attempts] == [384, 384, 384, 384, 384]
    assert [action["object_ids"] for action in plan["actions"]] == [
        [f"component_{index:04d}"] for index in range(1, 9)
    ]


def test_component_batch_keeps_invalid_single_candidate_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda snapshot, processor_size: ("processor", "model"),
    )
    attempts = []

    def generate(processor, model, messages, *, max_new_tokens, max_pixels):
        attempts.append((
            any(
                item["type"] == "image"
                for message in messages
                for item in message["content"]
            ),
            max_new_tokens,
        ))
        return "{"

    monkeypatch.setattr(local_agent_worker, "_generate_with_model", generate)

    with pytest.raises(json.JSONDecodeError):
        local_agent_worker._generate_component_plan(
            request,
            graph,
            evidence["quality-report.json"].read_text(encoding="utf-8"),
            evidence,
            "1" * 64,
            Path("snapshot"),
            max_candidates=1,
        )

    assert attempts == [(True, 128), (False, 128)]


def test_component_batch_repairs_singleton_tuple_format_without_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda *args: ("processor", "model"),
    )
    calls = []

    def generate(processor, model, messages, *, max_new_tokens, max_pixels):
        has_images = any(
            item["type"] == "image"
            for message in messages
            for item in message["content"]
        )
        calls.append((has_images, max_new_tokens, messages))
        if has_images:
            return '{"actions":[["split",["component_0001",],{"parts":2}]]}'
        return json.dumps({
            "action": "split",
            "object_ids": ["component_0001"],
            "parameters": {"parts": 2},
            "confidence": 0.95,
            "evidence_index": 0,
        })

    monkeypatch.setattr(local_agent_worker, "_generate_with_model", generate)

    plan = local_agent_worker._generate_component_plan(
        request,
        graph,
        evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence,
        "1" * 64,
        Path("snapshot"),
        max_candidates=1,
    )

    assert plan["actions"][0]["action"] == "split"
    assert [(has_images, tokens) for has_images, tokens, _ in calls] == [
        (True, 128),
        (False, 128),
    ]
    repair_text = "\n".join(
        item["text"]
        for message in calls[1][2]
        for item in message["content"]
    ).casefold()
    assert "preserve the intended action" in repair_text
    repair_request = json.loads(calls[1][2][1]["content"][0]["text"])
    assert set(repair_request) == {
        "ordered_candidates", "review_evidence", "invalid_response_untrusted",
    }
    assert "component_0001" in repair_text


def test_component_user_message_ends_with_named_action_response_contract(
    tmp_path: Path,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )

    messages = local_agent_worker._messages(
        request,
        graph,
        evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence,
        "1" * 64,
        action_scope={request["candidate_ids"][0]},
    )
    text = messages[1]["content"][0]["text"]

    assert text.endswith(
        "\nReturn only {\"actions\":[{\"action\":action,\"object_ids\":"
        "[object_id,...],\"parameters\":parameters,\"confidence\":confidence,"
        "\"evidence_index\":evidence_index},...]}; maximum_actions=1. Never exceed "
        "maximum_actions or append aggregate, alternative, or duplicate actions. Use "
        "exact action objects with only action, object_ids, parameters, confidence, "
        "and evidence_index. The "
        "ordered candidates are [\"component_0001\"]. Cover every ordered candidate "
        "exactly once across all object_ids and never repeat one as an alternative. "
        "Return one-line minified JSON without commentary."
    )


def test_component_batch_expands_named_action_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    evidence_index = request["review_evidence"].index("component-isolation.png")
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda snapshot, processor_size: ("processor", "model"),
    )
    monkeypatch.setattr(
        local_agent_worker,
        "_generate_with_model",
        lambda *args, **kwargs: json.dumps({
            "actions": [{
                "action": "accept",
                "object_ids": [request["candidate_ids"][0]],
                "parameters": {},
                "confidence": 0.95,
                "evidence_index": evidence_index,
            }],
        }),
    )

    plan = local_agent_worker._generate_component_plan(
        request,
        graph,
        evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence,
        "1" * 64,
        Path("snapshot"),
        max_candidates=1,
    )

    assert plan["actions"] == [{
        "action": "accept",
        "object_ids": [request["candidate_ids"][0]],
        "parameters": {},
        "confidence": 0.95,
        "evidence": ["component-isolation.png"],
    }]


@pytest.mark.parametrize(
    "action",
    [
        [True, ["component_0001"], {}, 0.95, 0],
        ["unknown", ["component_0001"], {}, 0.95, 0],
        ["accept", [True], {}, 0.95, 0],
        ["accept", [], {}, 0.95, 0],
        ["accept", ["component_0001"], {}, True, 0],
        ["accept", ["component_0001"], {}, -0.1, 0],
        ["accept", ["component_0001"], {}, 1.1, 0],
        ["accept", ["component_0001"], {}, 0.95, True],
        ["accept", ["component_0001"], {}, 0.95, 999],
        {
            "action": "accept",
            "object_ids": ["component_0001"],
            "parameters": {},
            "confidence": 0.95,
            "evidence": ["component-isolation.png"],
        },
    ],
)
def test_component_batch_rejects_invalid_named_compact_action_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: object,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda snapshot, processor_size: ("processor", "model"),
    )
    monkeypatch.setattr(
        local_agent_worker,
        "_generate_with_model",
        lambda *args, **kwargs: json.dumps({"actions": [action]}),
    )

    expected_error = (
        "format repair fields" if isinstance(action, dict)
        else "compact action tuple"
    )
    with pytest.raises(ValueError, match=expected_error):
        local_agent_worker._generate_component_plan(
            request,
            graph,
            evidence["quality-report.json"].read_text(encoding="utf-8"),
            evidence,
            "1" * 64,
            Path("snapshot"),
            max_candidates=1,
        )


def test_component_prompt_uses_explicit_graph_ids(
    tmp_path: Path,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    base = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )["nodes"][0]
    graph = {"nodes": [
        {**base, "id": f"component_{index:04d}", "z_index": index - 1}
        for index in range(1, 4)
    ]}
    request = {**request, "candidate_ids": [node["id"] for node in graph["nodes"]]}

    messages = local_agent_worker._messages(
        request,
        graph,
        json.dumps({"violations": []}),
        evidence,
        "1" * 64,
        action_scope={"component_0003"},
    )
    prompt = json.loads(messages[1]["content"][0]["text"].splitlines()[1])

    assert prompt["component_graph"]["node_fields"][0] == "id"
    assert prompt["component_graph"]["nodes"] == [[
        "component_0003",
        base["kind"],
        base["parent_id"],
        base["state"],
        base["bbox"],
        2,
        base["text_ids"],
    ]]
    assert "action_codes" not in prompt


def test_component_prompt_requires_named_action_objects() -> None:
    assert (
        '{"actions":[{"action":action,"object_ids":[object_id,...],'
        '"parameters":parameters,"confidence":confidence,'
        '"evidence_index":evidence_index},...]}'
        in local_agent_worker.SYSTEM_PROMPT
    )


def test_first_round_local_message_keeps_existing_image_order(tmp_path: Path) -> None:
    request_path = _request_path(tmp_path)

    assert _message_image_names(request_path) == [
        name for name in local_agent_worker._IMAGE_EVIDENCE
        if name in load_component_agent_request(request_path)["evidence"]
    ]


def test_later_round_local_message_only_sends_review_evidence_images(
    tmp_path: Path,
) -> None:
    request_path = _request_path(tmp_path, repair_round=2)

    assert _message_image_names(request_path) == [
        "source.png", "reconstructed.png", "difference.png", "round-review.png"
    ]


def test_later_round_local_telemetry_counts_only_review_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path, repair_round=2)
    request = load_component_agent_request(request_path)
    observed = []

    class Trace:
        def event(self, event, **fields):
            observed.append((event, fields))

    def invoke(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_plan(request_path)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)
    monkeypatch.setattr(local_agent.time, "perf_counter", lambda: 1.0)

    local_agent.run_local_agent(
        request_path,
        model_receipt=_receipt(tmp_path),
        performance_trace=Trace(),
    )

    image_paths = [
        request_path.parent / request["evidence"][name]["path"]
        for name in request["review_evidence"]
        if name.endswith(".png")
    ]
    assert observed == [("local_agent", {
        "image_count": len(image_paths),
        "total_bytes": sum(path.stat().st_size for path in image_paths),
        "duration_ms": 0,
        "status": "success",
    })]


def _receipt(tmp_path: Path) -> dict[str, object]:
    snapshot = tmp_path / "model-cache" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    return {
        "schema_version": 1,
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "requested_revision": "main",
        "resolved_revision": "a" * 40,
        "stability": "experimental",
        "snapshot_path": str(snapshot.resolve()),
        "files": [],
        "installed_at": "now",
    }


def _plan(request_path: Path, *, action: str = "accept") -> dict[str, object]:
    request = load_component_agent_request(request_path)
    return {
        "schema_version": 1,
        "kind": "component_plan",
        "page_id": request["page_id"],
        "provider": request["provider"],
        "repair_round": request["repair_round"],
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "actions": [
            {
                "action": action,
                "object_ids": ["component_0001"],
                "parameters": {},
                "confidence": 0.95,
                "evidence": ["complete visible component boundary"],
            }
        ],
    }


def _correction_context(request_path: Path) -> dict[str, object]:
    return {
        "instruction": (
            "The previous plan was rejected because an absorb_residual target had "
            "no containment or 3px adjacency with the signed residual. Modify or "
            "remove the related absorb_residual action; do not change request_sha256."
        ),
        "rejected_plan": _plan(request_path, action="absorb_residual"),
        "forbidden_action_pairs": [["absorb_residual", "component_0001"]],
    }


def _candidate(tmp_path: Path) -> dict[str, object]:
    image = tmp_path / "candidate.png"
    Image.new("RGB", (32, 18), "navy").save(image)
    return {
        "candidate_id": "candidate_001",
        "edge_gaps": {"bottom": 0.0, "left": 0.0, "right": 0.0, "top": 0.0},
        "image": "candidate_assets/candidate_001.png",
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "media_format": "PNG",
        "pixel_height": 18,
        "pixel_width": 32,
        "slide_coverage": 1.0,
        "source_object_sha256": "1" * 64,
        "source_shape_id": "2",
        "z_order": 0,
        "page_id": "page_001",
        "slide_index": 1,
        "native_object_counts": {"picture": 1},
        "image_path": str(image.resolve()),
        "allowed_decisions": ["ambiguous", "preserve", "replace"],
        "allowed_categories": [
            "decorative_asset", "full_slide_screenshot", "logo",
            "partial_slide_screenshot", "photo", "rasterized_chart",
            "rasterized_diagram", "unknown",
        ],
        "replace_confidence_threshold": 0.92,
    }


def _candidate_decision() -> dict[str, object]:
    return {
        "decision": "replace",
        "confidence": 0.99,
        "category": "full_slide_screenshot",
        "evidence": ["complete slide screenshot"],
    }


def test_local_candidate_agent_uses_offline_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(tmp_path)
    observed = {}

    def invoke(command, *, environment, timeout_seconds):
        observed["command"] = command
        request = Path(command[command.index("--candidate-request") + 1])
        observed["request"] = json.loads(request.read_text(encoding="utf-8"))
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_candidate_decision()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    assert local_agent.run_local_candidate_agent(
        candidate,
        model_receipt=_receipt(tmp_path),
    ) == _candidate_decision()
    assert observed["request"]["candidate"]["image_sha256"] == candidate[
        "image_sha256"
    ]
    assert "--request" not in observed["command"]


def test_local_service_candidate_agent_sends_image_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(tmp_path)
    observed = {}

    def complete(config, *, messages, timeout_seconds):
        observed["messages"] = messages
        return json.dumps(_candidate_decision())

    monkeypatch.setattr("image2editable.local_service.complete", complete)

    assert local_agent.run_local_service_candidate_agent(
        candidate,
        service_config=object(),
    ) == _candidate_decision()
    assert observed["messages"][1]["content"][1]["type"] == "image_url"
    prompt = json.loads(
        observed["messages"][1]["content"][0]["text"].splitlines()[1]
    )
    assert "image_path" not in prompt["candidate"]


def test_candidate_worker_loads_confirmed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(tmp_path)
    request = tmp_path / "candidate-request.json"
    request.write_text(
        json.dumps({"schema_version": 1, "candidate": candidate}),
        encoding="utf-8",
    )
    receipt = _receipt(tmp_path)
    calls = []

    class Inputs(dict):
        input_ids = [[1, 2, 3]]

        def to(self, device):
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor", str(path), kwargs))
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            return Inputs(input_ids=[[1, 2, 3]])

        def batch_decode(self, values, **kwargs):
            return [json.dumps(_candidate_decision())]

    class Model:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", str(path), kwargs))
            return cls()

        def generate(self, **kwargs):
            return [[1, 2, 3, 4]]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=Processor,
            AutoModelForImageTextToText=Model,
        ),
    )

    assert local_agent_worker.generate_candidate_decision(
        request,
        Path(receipt["snapshot_path"]),
    ) == _candidate_decision()
    assert calls[0][1] == receipt["snapshot_path"]
    assert calls[0][2]["size"] == {
        "shortest_edge": 4 * 32 * 32,
        "longest_edge": 512 * 32 * 32,
    }
    messages = local_agent_worker._candidate_messages(candidate)
    assert messages[0]["content"] == [
        {"type": "text", "text": local_agent_worker._CANDIDATE_PROMPT}
    ]
    assert "replace means convert" in local_agent_worker._CANDIDATE_PROMPT
    assert "preserve means keep" in local_agent_worker._CANDIDATE_PROMPT


def test_local_correction_context_is_added_to_actual_messages(tmp_path: Path) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )

    messages = local_agent_worker._messages(
        request,
        graph,
        evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence,
        hashlib.sha256(request_path.read_bytes()).hexdigest(),
        correction_context=_correction_context(request_path),
    )

    prompt = json.loads(messages[1]["content"][0]["text"].splitlines()[1])
    context = _correction_context(request_path)
    rejected_action = context["rejected_plan"]["actions"][0]
    assert prompt["correction_context"] == {
        "instruction": context["instruction"],
        "rule": (
            "Do not repeat a rejected action for any listed object ID; choose a "
            "different valid action for every listed candidate."
        ),
        "rejected_action_summaries": [{
            "action": rejected_action["action"],
            "object_ids": rejected_action["object_ids"],
            "parameters": rejected_action["parameters"],
        }],
        "forbidden_action_pairs": context["forbidden_action_pairs"],
    }
    assert (
        'Forbidden rejected action/object pairs: '
        '[["absorb_residual","component_0001"]].'
    ) in messages[1]["content"][0]["text"]
    assert (
        'Forbidden rejected action/object pairs: '
        '[["absorb_residual","component_0001"]].'
    ) in messages[0]["content"][0]["text"]


def test_local_correction_retries_box_when_model_repeats_rejected_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda snapshot, processor_size: ("processor", "model"),
    )
    monkeypatch.setattr(
        local_agent_worker,
        "_generate_with_model",
        lambda *args, **kwargs: json.dumps({
            "actions": [{
                "action": "absorb_residual",
                "object_ids": [request["candidate_ids"][0]],
                "parameters": {},
                "confidence": 0.95,
                "evidence_index": 0,
            }],
        }),
    )

    plan = local_agent_worker._generate_component_plan(
        request,
        graph,
        evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence,
        hashlib.sha256(request_path.read_bytes()).hexdigest(),
        Path("snapshot"),
        correction_context=_correction_context(request_path),
        max_candidates=1,
    )

    assert plan["actions"] == [{
        "action": "retry_with_box",
        "object_ids": [request["candidate_ids"][0]],
        "parameters": {"box": [0.0, 0.0, 1.0, 1.0]},
        "confidence": 1.0,
        "evidence": [request["review_evidence"][0]],
    }]


def test_component_batch_keeps_complete_correction_history() -> None:
    context = {
        "instruction": "instruction",
        "rejected_plan": {"actions": [
            {"action": "split", "object_ids": ["component_0001"]},
            {"action": "split", "object_ids": ["component_0002"]},
        ]},
        "forbidden_action_pairs": [
            ["absorb_residual", "component_0001"],
            ["split", "component_0001"],
            ["split", "component_0002"],
        ],
    }

    batch = local_agent_worker._batch_correction_context(
        context, {"component_0001"}
    )

    assert batch["forbidden_action_pairs"] == context["forbidden_action_pairs"]


def test_local_correction_reuses_non_rejected_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    second = {**graph["nodes"][0], "id": "component_0002", "z_index": 1}
    graph = {"nodes": [*graph["nodes"], second]}
    request = {**request, "candidate_ids": ["component_0001", "component_0002"]}
    rejected_plan = _plan(request_path)
    accepted = rejected_plan["actions"][0]
    rejected_plan["actions"] = [
        accepted,
        {
            **accepted,
            "action": "absorb_residual",
            "object_ids": ["component_0002"],
        },
    ]
    correction_context = {
        "instruction": _correction_context(request_path)["instruction"],
        "rejected_plan": rejected_plan,
        "forbidden_action_pairs": [["absorb_residual", "component_0002"]],
    }
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda snapshot, processor_size: ("processor", "model"),
    )
    calls = []

    def generate(processor, model, messages, **kwargs):
        calls.append(messages)
        return json.dumps({
            "actions": [{
                "action": "discard",
                "object_ids": ["component_0002"],
                "parameters": {},
                "confidence": 0.95,
                "evidence_index": 0,
            }],
        })

    monkeypatch.setattr(local_agent_worker, "_generate_with_model", generate)

    plan = local_agent_worker._generate_component_plan(
        request,
        graph,
        evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence,
        hashlib.sha256(request_path.read_bytes()).hexdigest(),
        Path("snapshot"),
        correction_context=correction_context,
        max_candidates=1,
    )

    assert [action["action"] for action in plan["actions"]] == ["accept", "discard"]
    assert len(calls) == 1
    assert 'ordered candidates are ["component_0002"]' in (
        calls[0][1]["content"][0]["text"]
    )


def test_local_split_correction_reuses_non_split_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    request = load_component_agent_request(request_path)
    evidence = {
        name: request_path.parent / record["path"]
        for name, record in request["evidence"].items()
    }
    graph = json.loads(
        evidence["component-graph.json"].read_text(encoding="utf-8")
    )
    second = {**graph["nodes"][0], "id": "component_0002", "z_index": 1}
    graph = {"nodes": [*graph["nodes"], second]}
    request = {**request, "candidate_ids": ["component_0001", "component_0002"]}
    rejected_plan = _plan(request_path)
    accepted = rejected_plan["actions"][0]
    rejected_plan["actions"] = [
        accepted,
        {
            **accepted,
            "action": "split",
            "object_ids": ["component_0002"],
            "parameters": {"parts": 2},
        },
    ]
    correction_context = {
        "instruction": (
            "The previous plan was rejected because a split target did not contain "
            "the exact requested number of connected proposals. Modify or remove "
            "the related split action; do not change request_sha256."
        ),
        "rejected_plan": rejected_plan,
        "forbidden_action_pairs": [
            ["absorb_residual", "component_0002"],
            ["split", "component_0002"],
        ],
    }
    monkeypatch.setattr(
        local_agent_worker,
        "_load_generator",
        lambda snapshot, processor_size: ("processor", "model"),
    )
    calls = []

    def generate(processor, model, messages, **kwargs):
        calls.append(messages)
        return json.dumps({
            "actions": [["discard", ["component_0002"], {}, 0.95, 0]],
        })

    monkeypatch.setattr(local_agent_worker, "_generate_with_model", generate)

    plan = local_agent_worker._generate_component_plan(
        request,
        graph,
        evidence["quality-report.json"].read_text(encoding="utf-8"),
        evidence,
        hashlib.sha256(request_path.read_bytes()).hexdigest(),
        Path("snapshot"),
        correction_context=correction_context,
        max_candidates=1,
    )

    assert [action["action"] for action in plan["actions"]] == ["accept", "discard"]
    assert len(calls) == 1


def test_local_agent_passes_correction_context_to_offline_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    observed = {}

    def invoke(command, **kwargs):
        correction = Path(command[command.index("--correction-context") + 1])
        observed["correction_context"] = json.loads(
            correction.read_text(encoding="utf-8")
        )
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_plan(request_path)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    result = local_agent.run_local_agent(
        request_path,
        model_receipt=_receipt(tmp_path),
        correction_context=_correction_context(request_path),
    )

    assert result == _plan(request_path)
    assert observed["correction_context"] == _correction_context(request_path)


def test_local_service_agent_passes_correction_context_to_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path, provider="local-service")
    observed = {}

    def complete(config, *, messages, timeout_seconds):
        observed["messages"] = messages
        return json.dumps(_plan(request_path))

    monkeypatch.setattr("image2editable.local_service.complete", complete)

    local_agent.run_local_service_agent(
        request_path,
        service_config=object(),
        correction_context=_correction_context(request_path),
    )

    prompt = json.loads(
        observed["messages"][1]["content"][0]["text"].splitlines()[1]
    )
    context = _correction_context(request_path)
    rejected_action = context["rejected_plan"]["actions"][0]
    assert prompt["correction_context"] == {
        "instruction": context["instruction"],
        "rule": (
            "Do not repeat a rejected action for any listed object ID; choose a "
            "different valid action for every listed candidate."
        ),
        "rejected_action_summaries": [{
            "action": rejected_action["action"],
            "object_ids": rejected_action["object_ids"],
            "parameters": rejected_action["parameters"],
        }],
        "forbidden_action_pairs": context["forbidden_action_pairs"],
    }


def test_local_agent_starts_one_worker_with_offline_bounded_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    observed: dict[str, object] = {}

    def invoke(command, *, environment, timeout_seconds):
        observed["command"] = command
        observed["environment"] = environment
        observed["timeout_seconds"] = timeout_seconds
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_plan(request_path)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    result = local_agent.run_local_agent(
        request_path,
        model_receipt=_receipt(tmp_path),
        resource_policy={
            "name": "safe-default",
            "cpu_threads": 3,
            "heavy_page_concurrency": 1,
            "sam_points_per_batch": 1,
        },
    )

    assert result == _plan(request_path)
    command = observed["command"]
    assert command[:3] == [sys.executable, "-m", "image2editable.local_agent_worker"]
    assert "--model-snapshot" in command
    environment = observed["environment"]
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["OMP_NUM_THREADS"] == "3"
    assert observed["timeout_seconds"] == 600


def test_local_agent_records_only_request_size_duration_and_worker_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    times = iter([4.0, 4.5])
    monkeypatch.setattr(local_agent.time, "perf_counter", lambda: next(times))
    observed = []

    class Trace:
        def event(self, event, **fields):
            observed.append((event, fields))

    def invoke(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_plan(request_path)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "response body", "")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    local_agent.run_local_agent(
        request_path,
        model_receipt=_receipt(tmp_path),
        performance_trace=Trace(),
    )

    image_paths = [
        request_path.parent / Path(*record["path"].split("/"))
        for record in load_component_agent_request(request_path)["evidence"].values()
        if record["path"].endswith(".png")
    ]
    assert observed == [
        (
            "local_agent",
            {
                "image_count": len(image_paths),
                "total_bytes": sum(path.stat().st_size for path in image_paths),
                "duration_ms": 500,
                "status": "success",
            },
        )
    ]


def test_local_worker_invocation_preserves_default_subprocess_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    monkeypatch.setattr(
        local_agent.subprocess,
        "run",
        lambda command, **kwargs: observed.append((command, kwargs))
        or subprocess.CompletedProcess(command, 0),
    )
    import scripts.worker_resources as worker_resources

    monkeypatch.setattr(
        worker_resources,
        "trim_parent_working_set_before_worker",
        lambda: pytest.fail("default Local Agent invocation must not trim parent"),
    )

    local_agent._invoke_worker(
        ["worker"], environment={"BASE": "kept"}, timeout_seconds=12
    )

    assert observed == [
        (
            ["worker"],
            {
                "env": {"BASE": "kept"},
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": 12,
            },
        )
    ]


def test_local_agent_keeps_success_result_when_trace_recording_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)

    def invoke(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_plan(request_path)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    class BrokenTrace:
        def event(self, event, **fields):
            raise OSError("trace unavailable")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    assert local_agent.run_local_agent(
        request_path,
        model_receipt=_receipt(tmp_path),
        performance_trace=BrokenTrace(),
    ) == _plan(request_path)


def test_local_agent_keeps_worker_timeout_when_trace_recording_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)

    class BrokenTrace:
        def event(self, event, **fields):
            raise OSError("trace unavailable")

    monkeypatch.setattr(
        local_agent,
        "_invoke_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["worker"], 600)
        ),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        local_agent.run_local_agent(
            request_path,
            model_receipt=_receipt(tmp_path),
            performance_trace=BrokenTrace(),
        )


def test_local_agent_records_failed_worker_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    times = iter([2.0, 2.5])
    monkeypatch.setattr(local_agent.time, "perf_counter", lambda: next(times))
    events = []

    class Trace:
        def event(self, event, **fields):
            events.append((event, fields))

    monkeypatch.setattr(
        local_agent,
        "_invoke_worker",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 7),
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        local_agent.run_local_agent(
            request_path,
            model_receipt=_receipt(tmp_path),
            performance_trace=Trace(),
        )

    assert events[0][1]["status"] == "failed"
    assert events[0][1]["duration_ms"] == 500


def test_local_agent_records_one_error_for_invalid_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    times = iter([2.0, 2.5])
    monkeypatch.setattr(local_agent.time, "perf_counter", lambda: next(times))
    events = []

    class Trace:
        def event(self, event, **fields):
            events.append((event, fields))

    def invoke(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text("not valid JSON", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    with pytest.raises(json.JSONDecodeError):
        local_agent.run_local_agent(
            request_path,
            model_receipt=_receipt(tmp_path),
            performance_trace=Trace(),
        )

    assert events == [
        (
            "local_agent",
            {
                "image_count": 8,
                "total_bytes": sum(
                    (request_path.parent / Path(*record["path"].split("/"))).stat().st_size
                    for record in load_component_agent_request(request_path)["evidence"].values()
                    if record["path"].endswith(".png")
                ),
                "duration_ms": 500,
                "status": "error",
            },
        )
    ]


def test_local_service_agent_records_content_free_performance_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path, provider="local-service")
    times = iter([5.0, 5.25])
    monkeypatch.setattr(local_agent.time, "perf_counter", lambda: next(times))
    events = []

    class Trace:
        def event(self, event, **fields):
            events.append((event, fields))

    monkeypatch.setattr(
        "image2editable.local_service.complete",
        lambda *args, **kwargs: json.dumps(_plan(request_path)),
    )

    assert local_agent.run_local_service_agent(
        request_path,
        service_config=object(),
        performance_trace=Trace(),
    ) == _plan(request_path)

    assert events[0][0] == "local_agent"
    assert set(events[0][1]) == {"image_count", "total_bytes", "duration_ms", "status"}
    image_paths = [
        request_path.parent / Path(*record["path"].split("/"))
        for record in load_component_agent_request(request_path)["evidence"].values()
        if record["path"].endswith(".png")
    ]
    assert events[0][1]["image_count"] == len(image_paths)
    assert events[0][1]["total_bytes"] == sum(path.stat().st_size for path in image_paths)
    assert events[0][1]["duration_ms"] == 250
    assert events[0][1]["status"] == "success"


def test_local_service_keeps_original_error_when_trace_recording_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path, provider="local-service")
    expected = RuntimeError("service unavailable")

    class BrokenTrace:
        def event(self, event, **fields):
            raise OSError("trace unavailable")

    monkeypatch.setattr(
        "image2editable.local_service.complete",
        lambda *args, **kwargs: (_ for _ in ()).throw(expected),
    )

    with pytest.raises(RuntimeError) as caught:
        local_agent.run_local_service_agent(
            request_path,
            service_config=object(),
            performance_trace=BrokenTrace(),
        )

    assert caught.value is expected


def test_local_service_keeps_success_result_when_trace_recording_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path, provider="local-service")

    class BrokenTrace:
        def event(self, event, **fields):
            raise OSError("trace unavailable")

    monkeypatch.setattr(
        "image2editable.local_service.complete",
        lambda *args, **kwargs: json.dumps(_plan(request_path)),
    )

    assert local_agent.run_local_service_agent(
        request_path,
        service_config=object(),
        performance_trace=BrokenTrace(),
    ) == _plan(request_path)
def test_local_service_agent_uses_the_user_configured_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image2editable.local_service import LocalServiceConfig

    request_path = _request_path(tmp_path, provider="local-service")
    observed: dict[str, object] = {}

    def complete(config, *, messages, timeout_seconds):
        observed["config"] = config
        observed["messages"] = messages
        observed["timeout_seconds"] = timeout_seconds
        return json.dumps(_plan(request_path))

    monkeypatch.setattr("image2editable.local_service.complete", complete)

    result = local_agent.run_local_service_agent(
        request_path,
        service_config=LocalServiceConfig("http://127.0.0.1:8000/v1", "my-vlm", None),
    )

    assert result == _plan(request_path)
    assert observed["config"].model == "my-vlm"
    assert observed["timeout_seconds"] == 600


def test_local_plan_passes_the_same_strict_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)

    def invoke(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(_plan(request_path, action="unknown")), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    with pytest.raises(ValueError, match="component action"):
        local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))


def test_local_worker_nonzero_exit_preserves_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    monkeypatch.setattr(
        local_agent,
        "_invoke_worker",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 7, "worker output", "CUDA out of memory"
        ),
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))

    diagnostic = request_path.parents[2] / "local-agent-diagnostics" / "round-01.json"
    saved = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert saved["status"] == "worker_failed"
    assert saved["returncode"] == 7
    assert "out of memory" in saved["stderr"]


def test_local_worker_process_boundary_releases_after_every_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    live = 0
    peak = 0

    def invoke(command, **kwargs):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_plan(request_path)), encoding="utf-8")
        live -= 1
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(local_agent, "_invoke_worker", invoke)

    local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))
    local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))

    assert peak == 1
    assert live == 0


def test_local_agent_rejects_insufficient_page_round_disk_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    invoked = False

    def unexpected_invoke(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("worker must not start with insufficient page budget")

    monkeypatch.setattr(local_agent, "_invoke_worker", unexpected_invoke)
    monkeypatch.setattr(
        local_agent.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(free=1),
    )

    with pytest.raises(RuntimeError, match="page repair disk budget"):
        local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))

    assert invoked is False


def test_unsafe_diagnostic_directory_does_not_mask_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    reconstruction = request_path.parents[2]
    outside = tmp_path / "outside"
    outside.mkdir()
    diagnostics = reconstruction / "local-agent-diagnostics"
    try:
        os.symlink(outside, diagnostics, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink is unavailable: {error}")
    monkeypatch.setattr(
        local_agent,
        "_invoke_worker",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 7, "worker output", "CUDA out of memory"
        ),
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        local_agent.run_local_agent(request_path, model_receipt=_receipt(tmp_path))

    assert list(outside.iterdir()) == []


def test_local_modules_do_not_import_model_runtime_at_parent_import() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import image2editable.local_agent; "
                "import image2editable.local_agent_worker; "
                "print(any(name in sys.modules for name in "
                "('torch', 'transformers')))"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_worker_loads_only_the_confirmed_local_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = _request_path(tmp_path)
    receipt = _receipt(tmp_path)
    calls: list[tuple[str, object, dict[str, object]]] = []

    class Inputs(dict):
        input_ids = [[1, 2, 3]]

        def to(self, device):
            calls.append(("inputs", str(device), {}))
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor", str(path), kwargs))
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls.append(("prompt", messages, kwargs))
            return Inputs(input_ids=[[1, 2, 3]])

        def batch_decode(self, values, **kwargs):
            request = load_component_agent_request(request_path)
            return [json.dumps({
                "actions": [[
                    "accept",
                    [request["candidate_ids"][0]],
                    {},
                    0.95,
                    request["review_evidence"].index("component-isolation.png"),
                ]],
            })]

    class Model:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", str(path), kwargs))
            return cls()

        def generate(self, **kwargs):
            return [[1, 2, 3, 4]]

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=Processor,
        AutoModelForImageTextToText=Model,
    )
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        empty_cache=lambda: calls.append(("empty_cache", "cuda", {})),
    ))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    result = local_agent_worker.generate_plan(
        request_path,
        Path(receipt["snapshot_path"]),
    )

    expected = _plan(request_path)
    expected["actions"][0]["evidence"] = ["component-isolation.png"]
    assert result == expected
    snapshot = receipt["snapshot_path"]
    assert (
        "processor",
        snapshot,
        {
            "local_files_only": True,
            "size": {
                "shortest_edge": 4 * 32 * 32,
                "longest_edge": 128 * 32 * 32,
            },
        },
    ) in calls
    assert calls.count(("empty_cache", "cuda", {})) == 1
    assert (
        "model",
        snapshot,
        {
            "local_files_only": True,
            "device_map": "auto",
            "torch_dtype": "auto",
        },
    ) in calls
    assert set(local_agent_worker.ALLOWED_ACTIONS) == {
        "accept",
        "discard",
        "merge",
        "split",
        "expand",
        "shrink",
        "retry_with_box",
        "retry_with_points",
        "attach_text",
        "suppress_text",
        "collapse_to_parent",
            "rebuild_background",
            "absorb_residual",
            "absorb_into_parent",
    }
    assert "untrusted" in local_agent_worker.SYSTEM_PROMPT.casefold()
    assert "JSON" in local_agent_worker.SYSTEM_PROMPT
    for required_rule in (
        'split parameters: {"parts"',
        'expand/shrink parameters: {"margin_ratio"',
        'rebuild_background parameters: {"margin_ratio"',
        "smallest margin that covers the visible residual",
        'absorb_into_parent parameters: {}',
        'retry_with_box parameters: {"box"',
        'retry_with_points parameters: {"positive"',
        "suppress_text only when visual evidence clearly proves",
        '"negative"',
        "normalized to 0..1",
        "unexplained_visual_residual",
        "unexplained-mask.png",
        "Do not accept, discard, or classify the region as background",
        "background_text_residual",
    ):
        assert required_rule in local_agent_worker.SYSTEM_PROMPT
    prompt_messages = next(value for kind, value, _ in calls if kind == "prompt")
    prompt_text = "\n".join(
        item["text"]
        for item in prompt_messages[1]["content"]
        if item["type"] == "text"
    )
    for evidence_name in local_agent_worker._IMAGE_EVIDENCE:
        assert evidence_name in prompt_text
