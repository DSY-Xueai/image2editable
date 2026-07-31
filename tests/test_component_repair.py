from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from image2editable.component_repair import (
    EVIDENCE_NAMES,
    build_component_agent_request,
    load_component_agent_request,
)


def _node(component_id: str, state: str, z_index: int) -> dict:
    return {
        "id": component_id,
        "kind": "parent",
        "parent_id": None,
        "state": state,
        "mask": f"masks/{component_id}.png",
        "mask_sha256": "a" * 64,
        "bbox": [0, 0, 2, 2],
        "z_index": z_index,
        "text_ids": [],
    }


@pytest.fixture
def page_session(tmp_path: Path) -> dict:
    reconstruction = tmp_path / "pages" / "page_001" / "reconstruction"
    evidence_root = reconstruction / "evidence-source"
    evidence_root.mkdir(parents=True)
    graph = {
        "nodes": [
            _node("candidate_b", "pending", 1),
            _node("frozen_a", "frozen", 0),
        ]
    }
    sources = {}
    for name in EVIDENCE_NAMES:
        path = evidence_root / name
        if name == "component-graph.json":
            path.write_text(json.dumps(graph), encoding="utf-8")
        elif name == "quality-report.json":
            path.write_text('{"valid": false}', encoding="utf-8")
        else:
            path.write_bytes((name + " data").encode())
        sources[name] = path
    return {
        "page_id": "page_001",
        "provider": "host",
        "reconstruction_dir": reconstruction,
        "evidence": sources,
    }


def test_build_request_hash_binds_every_evidence_file(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert set(request["evidence"]) == set(EVIDENCE_NAMES)
    assert all(len(record["sha256"]) == 64 for record in request["evidence"].values())
    assert request["candidate_ids"] == ["candidate_b"]
    assert request["frozen_ids"] == ["frozen_a"]
    assert request_path.parent.name == "round-01"


def test_validate_request_rejects_changed_overlay(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    overlay = request_path.parent / "ocr-overlay.png"
    overlay.write_bytes(overlay.read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match="evidence hash"):
        load_component_agent_request(request_path)


@pytest.mark.parametrize("repair_round", [0, 6, True, 1.0])
def test_build_request_rejects_round_outside_fixed_limit(
    page_session: dict,
    repair_round: object,
) -> None:
    with pytest.raises(ValueError, match="repair_round"):
        build_component_agent_request(page_session, repair_round=repair_round)


def test_published_request_cannot_be_overwritten(page_session: dict) -> None:
    first = build_component_agent_request(page_session, repair_round=1)
    before = first.read_bytes()

    with pytest.raises(RuntimeError, match="already published"):
        build_component_agent_request(page_session, repair_round=1)

    assert first.read_bytes() == before


def _publish(session: dict, ready: object, result: object) -> None:
    ready.wait(10)
    try:
        build_component_agent_request(session, repair_round=1)
    except RuntimeError:
        result.put("rejected")
    else:
        result.put("published")


def test_same_page_round_has_one_concurrent_publisher(page_session: dict) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    result = context.Queue()
    processes = [
        context.Process(target=_publish, args=(page_session, ready, result))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0

    assert sorted(result.get(timeout=2) for _ in processes) == [
        "published",
        "rejected",
    ]


def test_build_request_rejects_cross_page_evidence(page_session: dict, tmp_path: Path) -> None:
    outside = tmp_path / "pages" / "page_002" / "reconstruction" / "source.png"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"other page")
    page_session["evidence"]["source.png"] = outside

    with pytest.raises(RuntimeError, match="outside page reconstruction"):
        build_component_agent_request(page_session, repair_round=1)


def test_load_rejects_hard_linked_evidence(page_session: dict, tmp_path: Path) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    evidence = request_path.parent / "ownership.png"
    outside = tmp_path / "outside.png"
    evidence.replace(outside)
    try:
        os.link(outside, evidence)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="hard link"):
        load_component_agent_request(request_path)


def test_load_rejects_request_unknown_field(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["unexpected"] = True
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match="request fields"):
        load_component_agent_request(request_path)


def test_load_rejects_evidence_path_traversal(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["evidence"]["source.png"]["path"] = "../source.png"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence path"):
        load_component_agent_request(request_path)


def test_build_rejects_symlinked_evidence(page_session: dict, tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    source = page_session["evidence"]["source.png"]
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="link|safely"):
        build_component_agent_request(page_session, repair_round=1)


def test_load_rejects_candidate_ids_not_bound_to_graph(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["candidate_ids"] = []
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(RuntimeError, match="component ids"):
        load_component_agent_request(request_path)
