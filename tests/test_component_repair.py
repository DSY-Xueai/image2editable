from __future__ import annotations

import json
import hashlib
import hmac
import multiprocessing
import os
from pathlib import Path
import shutil

import pytest

from image2editable.component_repair import (
    EVIDENCE_NAMES,
    build_component_agent_request,
    load_component_agent_request,
)
import image2editable.component_repair as component_repair


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
    return _make_page_session(tmp_path, "page_001")


def _make_page_session(run_root: Path, page_id: str) -> dict:
    reconstruction = run_root / "pages" / page_id / "reconstruction"
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
        "page_id": page_id,
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


def _refresh_marker_for_contract_test(request_path: Path) -> None:
    marker_path = request_path.parent / "publication-marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["request_sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    key_path = (
        request_path.parents[5]
        / ".component-agent-integrity"
        / "key.bin"
    )
    fields = {key: value for key, value in marker.items() if key != "hmac_sha256"}
    canonical = json.dumps(
        fields,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    marker["hmac_sha256"] = hmac.new(
        key_path.read_bytes(), canonical, hashlib.sha256
    ).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


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
    _refresh_marker_for_contract_test(request_path)

    with pytest.raises(ValueError, match="request fields"):
        load_component_agent_request(request_path)


def test_load_rejects_evidence_path_traversal(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["evidence"]["source.png"]["path"] = "../source.png"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    _refresh_marker_for_contract_test(request_path)

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
    _refresh_marker_for_contract_test(request_path)

    with pytest.raises(RuntimeError, match="component ids"):
        load_component_agent_request(request_path)


def test_build_rejects_similar_but_non_pages_directory(tmp_path: Path) -> None:
    reconstruction = tmp_path / "not-pages" / "page_001" / "reconstruction"
    evidence_root = reconstruction / "evidence-source"
    evidence_root.mkdir(parents=True)
    graph = {"nodes": [_node("candidate", "pending", 0)]}
    evidence = {}
    for name in EVIDENCE_NAMES:
        path = evidence_root / name
        if name == "component-graph.json":
            path.write_text(json.dumps(graph), encoding="utf-8")
        elif name == "quality-report.json":
            path.write_text('{"valid": false}', encoding="utf-8")
        else:
            path.write_bytes(name.encode())
        evidence[name] = path

    with pytest.raises(RuntimeError, match="pages/.+reconstruction"):
        build_component_agent_request(
            {
                "page_id": "page_001",
                "provider": "host",
                "reconstruction_dir": reconstruction,
                "evidence": evidence,
            },
            repair_round=1,
        )


def test_load_rejects_request_moved_under_similar_fake_directory(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    pages_dir = Path(page_session["reconstruction_dir"]).parent.parent
    fake_dir = pages_dir.with_name("not-pages")
    pages_dir.rename(fake_dir)
    moved_request = fake_dir / request_path.relative_to(pages_dir)

    with pytest.raises(RuntimeError, match="pages/.+reconstruction"):
        load_component_agent_request(moved_request)


def test_load_requires_external_request_digest_marker(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    marker = request_path.parent / "publication-marker.json"
    marker.unlink()

    with pytest.raises(RuntimeError, match="marker"):
        load_component_agent_request(request_path)


def test_load_rejects_provider_changed_after_publish(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["provider"] = "local"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(RuntimeError, match="request hash"):
        load_component_agent_request(request_path)


def test_load_rejects_evidence_and_request_hash_changed_together(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    evidence_path = request_path.parent / "ocr-overlay.png"
    evidence_path.write_bytes(b"coordinated replacement")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["evidence"]["ocr-overlay.png"]["sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(RuntimeError, match="request hash"):
        load_component_agent_request(request_path)


def test_load_rejects_synchronized_request_evidence_and_marker_without_key(
    page_session: dict,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    evidence_path = request_path.parent / "ocr-overlay.png"
    evidence_path.write_bytes(b"coordinated replacement")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["evidence"]["ocr-overlay.png"]["sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    request_path.write_text(json.dumps(request), encoding="utf-8")
    marker_path = request_path.parent / "publication-marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["request_sha256"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    marker["hmac_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="signature"):
        load_component_agent_request(request_path)


def test_marker_write_failure_does_not_publish_round_and_can_retry(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = component_repair._write_exclusive
    failed = False

    def fail_marker(path: Path, payload: bytes, reconstruction: Path) -> None:
        nonlocal failed
        if path.name == "publication-marker.json" and not failed:
            failed = True
            raise OSError("simulated marker failure")
        real_write(path, payload, reconstruction)

    monkeypatch.setattr(component_repair, "_write_exclusive", fail_marker)
    with pytest.raises(OSError, match="marker failure"):
        build_component_agent_request(page_session, repair_round=1)

    round_dir = Path(page_session["reconstruction_dir"]) / "agent" / "round-01"
    assert not round_dir.exists()
    monkeypatch.setattr(component_repair, "_write_exclusive", real_write)
    assert build_component_agent_request(page_session, repair_round=1).is_file()


def test_round_rename_failure_leaves_no_publication_and_can_retry(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    component_repair._load_or_create_integrity_key(reconstruction)
    real_rename = Path.rename
    failed = False

    def fail_round_rename(path: Path, target: Path) -> Path:
        nonlocal failed
        if path.name.startswith(".round-01.tmp-") and not failed:
            failed = True
            raise OSError("simulated pre-publication crash")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_round_rename)
    with pytest.raises(RuntimeError, match="already published"):
        build_component_agent_request(page_session, repair_round=1)

    round_dir = reconstruction / "agent" / "round-01"
    assert not round_dir.exists()
    monkeypatch.setattr(Path, "rename", real_rename)
    assert build_component_agent_request(page_session, repair_round=1).is_file()


def test_damaged_integrity_key_fails_closed_without_rotation(page_session: dict) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    key_path = request_path.parents[5] / ".component-agent-integrity" / "key.bin"
    key_path.write_bytes(b"damaged")

    with pytest.raises(RuntimeError, match="integrity key"):
        load_component_agent_request(request_path)

    assert key_path.read_bytes() == b"damaged"


def test_two_pages_concurrently_reuse_one_complete_integrity_key(
    tmp_path: Path,
) -> None:
    first = _make_page_session(tmp_path, "page_001")
    second = _make_page_session(tmp_path, "page_002")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    result = context.Queue()
    processes = [
        context.Process(target=_publish, args=(session, ready, result))
        for session in (first, second)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0

    assert sorted(result.get(timeout=2) for _ in processes) == [
        "published",
        "published",
    ]
    key_path = tmp_path / ".component-agent-integrity" / "key.bin"
    assert len(key_path.read_bytes()) == 32
    assert key_path.stat().st_nlink == 1
    assert load_component_agent_request(
        tmp_path / "pages/page_001/reconstruction/agent/round-01/component_agent_request.json"
    )["page_id"] == "page_001"
    assert load_component_agent_request(
        tmp_path / "pages/page_002/reconstruction/agent/round-01/component_agent_request.json"
    )["page_id"] == "page_002"


def test_integrity_key_hard_link_fails_closed(page_session: dict, tmp_path: Path) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    key_path = request_path.parents[5] / ".component-agent-integrity" / "key.bin"
    outside = tmp_path / "stolen-key"
    key_path.replace(outside)
    try:
        os.link(outside, key_path)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="integrity key|hard link"):
        load_component_agent_request(request_path)


def test_integrity_key_directory_symlink_fails_closed(
    page_session: dict,
    tmp_path: Path,
) -> None:
    request_path = build_component_agent_request(page_session, repair_round=1)
    anchor = request_path.parents[5] / ".component-agent-integrity"
    outside = tmp_path / "outside-key-anchor"
    anchor.rename(outside)
    try:
        anchor.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        outside.rename(anchor)
        pytest.skip(f"directory symbolic links are unavailable: {error}")

    with pytest.raises(RuntimeError, match="integrity key|link|reparse"):
        load_component_agent_request(request_path)


def test_build_detects_parent_replaced_between_check_and_open(
    page_session: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = Path(page_session["evidence"]["component-graph.json"]).parent
    original_root = evidence_root.with_name("evidence-original")
    attacker_root = evidence_root.with_name("evidence-attacker")
    shutil.copytree(evidence_root, attacker_root)
    attacker_graph = attacker_root / "component-graph.json"
    attacker_graph.write_text(
        json.dumps({"nodes": [_node("attacker", "pending", 0)]}),
        encoding="utf-8",
    )
    real_open = os.open
    replaced = False

    def replace_parent_before_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if not replaced and Path(path) == evidence_root / "component-graph.json":
            evidence_root.rename(original_root)
            attacker_root.rename(evidence_root)
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(component_repair.os, "open", replace_parent_before_open)

    with pytest.raises(RuntimeError, match="directory identity"):
        build_component_agent_request(page_session, repair_round=1)

    assert replaced is True
    assert json.loads(
        (evidence_root / "component-graph.json").read_text(encoding="utf-8")
    )["nodes"][0]["id"] == "attacker"


def test_build_detects_parent_changed_to_symlink_before_open(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = Path(page_session["evidence"]["component-graph.json"]).parent
    original_root = evidence_root.with_name("evidence-original")
    attacker_root = evidence_root.with_name("evidence-attacker")
    shutil.copytree(evidence_root, attacker_root)
    try:
        probe = evidence_root.with_name("symlink-probe")
        probe.symlink_to(attacker_root, target_is_directory=True)
        probe.unlink()
    except OSError as error:
        pytest.skip(f"directory symbolic links are unavailable: {error}")
    real_open = os.open
    replaced = False

    def replace_with_symlink(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if not replaced and Path(path) == evidence_root / "component-graph.json":
            evidence_root.rename(original_root)
            evidence_root.symlink_to(attacker_root, target_is_directory=True)
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(component_repair.os, "open", replace_with_symlink)

    with pytest.raises(RuntimeError, match="directory identity"):
        build_component_agent_request(page_session, repair_round=1)


def test_marker_write_detects_agent_directory_replacement(
    page_session: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstruction = Path(page_session["reconstruction_dir"])
    agent_dir = reconstruction / "agent"
    original_agent = reconstruction / "agent-original"
    attacker_agent = reconstruction / "agent-attacker"
    real_open = os.open
    replaced = False

    def replace_agent_before_marker(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if not replaced and Path(path).name == "publication-marker.json":
            agent_dir.rename(original_agent)
            attacker_agent.mkdir()
            attacker_agent.rename(agent_dir)
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(component_repair.os, "open", replace_agent_before_marker)

    with pytest.raises(RuntimeError, match="directory identity"):
        build_component_agent_request(page_session, repair_round=1)

    assert replaced is True
