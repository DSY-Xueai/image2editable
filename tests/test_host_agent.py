from __future__ import annotations

import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import stat
import io

import pytest

from image2editable import host_agent
from image2editable.component_repair import (
    EVIDENCE_NAMES,
    advance_component_repair,
    build_component_agent_request,
    initialize_component_repair_state,
)


def test_host_skill_uses_request_review_evidence_without_skipping_quality() -> None:
    text = (Path(__file__).resolve().parents[1] / "skills/image-to-ppt/SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "review_evidence" in text
    assert "只查看并逐项核验" in text
    assert "完整 request" in text
    assert "质量门禁" in text
    assert "后续 `agent next` 返回当前组件请求及九项绝对证据路径" not in text
from image2editable.contracts import RunStatus
from image2editable.host_agent import next_host_agent_item, record_host_plan
from image2editable.inputs import prepare_image_job
from image2editable.store import RunStore


def _node(component_id: str, state: str, z_index: int) -> dict:
    return {"id": component_id, "kind": "parent", "parent_id": None,
            "state": state, "mask": f"masks/{component_id}.png",
            "mask_sha256": "a" * 64, "bbox": [0, 0, 2, 2],
            "z_index": z_index, "text_ids": []}


@pytest.fixture
def host_run(tmp_path: Path) -> Path:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_run(RunStatus.AWAITING_AGENT)
    return run_dir


def _publish_request(run_dir: Path) -> Path:
    reconstruction = run_dir / "pages/page_001/reconstruction"
    evidence_root = reconstruction / "evidence-source"
    evidence_root.mkdir(parents=True)
    graph = {"nodes": [_node("candidate_1", "pending", 0)]}
    masks = evidence_root / "masks"
    masks.mkdir()
    mask_path = masks / "candidate_1.png"
    mask_path.write_bytes(b"mask")
    graph["nodes"][0]["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    evidence = {}
    for name in EVIDENCE_NAMES:
        path = evidence_root / name
        if name == "component-graph.json":
            path.write_text(json.dumps(graph), encoding="utf-8")
        elif name == "quality-report.json":
            path.write_text('{"valid": false}', encoding="utf-8")
        elif name == "presentation-manifest.json":
            continue
        else:
            path.write_bytes(name.encode())
        evidence[name] = path
    assets = evidence_root / "presentation-assets"
    assets.mkdir()
    references = {}
    for name in (
        "rgba", "ownership_mask", "presentation_alpha_mask",
        "generated_underlay_mask",
    ):
        path = assets / f"{name}.png"
        path.write_bytes(name.encode())
        references[name] = {
            "path": path.relative_to(run_dir).as_posix(),
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
            "component_id": "candidate_1", **references,
            "metrics": {"boundary_color_mae": 0.0},
        }],
    }), encoding="utf-8")
    evidence["presentation-manifest.json"] = manifest
    request_path = build_component_agent_request({
        "page_id": "page_001", "provider": "host",
        "reconstruction_dir": reconstruction, "evidence": evidence,
    }, repair_round=1)
    store = RunStore.open(run_dir)
    initialize_component_repair_state(
        store, "page_001", request_path=request_path, initial_component_count=1,
    )
    advance_component_repair(store, "page_001")
    return request_path


def test_current_request_prefers_public_awaiting_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        root = tmp_path

        def read_json(self, name: str) -> dict:
            if name == "job_manifest.json":
                return {"pages": ["page_001", "page_002"]}
            if name == "page_jobs.json":
                return {"pages": {
                    "page_001": {"status": "awaiting_agent"},
                    "page_002": {"status": "processing"},
                }}
            page_id = Path(name).parts[1]
            request = requests[page_id]
            return {
                "provider": "host", "phase": "awaiting_plan",
                "current_round": {"request_ref": {
                    "path": (
                        f"pages/{page_id}/reconstruction/agent/"
                        f"round-{request['repair_round']:02d}/component_agent_request.json"
                    ),
                    "sha256": host_agent._request_sha256(request),
                }},
            }

    requests = {
        "page_001": {"page_id": "page_001", "provider": "host", "repair_round": 2},
        "page_002": {"page_id": "page_002", "provider": "host", "repair_round": 1},
    }
    for page_id in requests:
        state_path = tmp_path / f"pages/{page_id}/reconstruction/component_state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "image2editable.component_contracts.validate_component_repair_state",
        lambda state: state,
    )
    monkeypatch.setattr(
        host_agent,
        "load_component_agent_request",
        lambda path: requests[path.parents[3].name],
    )

    _, request = host_agent._current_request(FakeStore())

    assert request["page_id"] == "page_001"


def test_host_ignores_unreferenced_complete_round(host_run: Path, tmp_path: Path) -> None:
    request_path = _publish_request(host_run)
    reconstruction = host_run / "pages/page_001/reconstruction"
    evidence = {
        name: reconstruction / "evidence-source" / name for name in EVIDENCE_NAMES
    }
    orphan = build_component_agent_request({
        "page_id": "page_001", "provider": "host",
        "reconstruction_dir": reconstruction, "evidence": evidence,
    }, repair_round=2)
    handshake = next_host_agent_item(host_run)
    record_host_plan(
        host_run,
        _write_json(tmp_path / "capability.json", _capability_response(handshake)),
    )

    item = next_host_agent_item(host_run)

    assert item["request_path"] == str(request_path.resolve())
    assert item["request_path"] != str(orphan.resolve())


def _capability_response(item: dict) -> dict:
    return {"schema_version": 1, "kind": "host_capability_response",
            "challenge_id": item["challenge_id"],
            "observed": _observe_challenge(Path(item["image_path"]))}


def _observe_challenge(path: Path) -> dict:
    from PIL import Image
    image = Image.open(path).convert("RGB")
    for color_name in host_agent.CHALLENGE_COLORS:
        color = tuple(bytes.fromhex(color_name[1:]))
        components = _color_components(image, color)
        if components:
            first = components[0]
            if first["top_width"] == first["middle_width"] == first["bottom_width"]:
                shape = "square"
            elif first["top_width"] < first["middle_width"] < first["bottom_width"]:
                shape = "triangle"
            else:
                shape = "circle"
            return {"shape": shape, "color": color_name, "count": len(components)}
    raise AssertionError("challenge shape not found")


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_host_next_returns_random_hash_bound_visual_handshake_before_page(host_run: Path) -> None:
    item = next_host_agent_item(host_run)
    metadata = json.loads((host_run / "host-challenge/metadata.json").read_text(encoding="utf-8"))
    assert item["kind"] == "capability_handshake"
    assert Path(item["image_path"]).is_absolute()
    assert item["required_capabilities"] == [
        "vision", "local_file_read", "tool_use", "structured_json"]
    assert metadata["challenge_id"] == item["challenge_id"]
    assert set(metadata) == {
        "schema_version", "challenge_id", "image_path", "image_sha256"
    }
    assert metadata["image_sha256"] == hashlib.sha256(Path(item["image_path"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("shape", ["triangle", "circle", "square"])
def test_every_visual_shape_matches_label_color_count_and_square_bbox(
    shape: str,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    choices = iter([shape, "#d9485f", 4])
    monkeypatch.setattr(host_agent.secrets, "choice", lambda values: next(choices))
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_run(RunStatus.AWAITING_AGENT)
    item = next_host_agent_item(run_dir)
    metadata = json.loads((run_dir / "host-challenge/metadata.json").read_text(encoding="utf-8"))
    from PIL import Image
    image = Image.open(item["image_path"]).convert("RGB")
    observed = _observe_challenge(Path(item["image_path"]))
    color = tuple(bytes.fromhex(observed["color"][1:]))
    components = _color_components(image, color)
    assert "expected" not in metadata and "nonce" not in metadata
    assert observed == {"shape": shape, "color": "#d9485f", "count": 4}
    assert len(components) == 4
    assert all(component["width"] == component["height"] for component in components)
    if shape == "triangle":
        assert all(component["top_width"] < component["middle_width"] < component["bottom_width"] for component in components)
    elif shape == "circle":
        assert all(component["top_width"] < component["middle_width"] and component["bottom_width"] < component["middle_width"] for component in components)
    else:
        assert all(component["top_width"] == component["middle_width"] == component["bottom_width"] for component in components)


def test_same_answer_has_different_salted_png_hash_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    choices = iter(["circle", "#2b8a3e", 3, "circle", "#2b8a3e", 3])
    monkeypatch.setattr(host_agent.secrets, "choice", lambda values: next(choices))
    salts = iter([b"a" * 16, b"b" * 16])
    real_token_bytes = host_agent.secrets.token_bytes
    monkeypatch.setattr(
        host_agent.secrets,
        "token_bytes",
        lambda size: next(salts) if size == 16 else real_token_bytes(size),
    )
    items = []
    for index in range(2):
        source = tmp_path / f"source-{index}.png"
        source.write_bytes(b"image")
        run_dir = prepare_image_job(source, run_dir=tmp_path / f"run-{index}")
        store = RunStore.open(run_dir)
        store.transition_run(RunStatus.RUNNING)
        store.transition_run(RunStatus.AWAITING_AGENT)
        items.append(next_host_agent_item(run_dir))
    assert [_observe_challenge(Path(item["image_path"])) for item in items] == [
        {"shape": "circle", "color": "#2b8a3e", "count": 3},
        {"shape": "circle", "color": "#2b8a3e", "count": 3},
    ]
    assert len({hashlib.sha256(Path(item["image_path"]).read_bytes()).hexdigest() for item in items}) == 2


def test_salted_challenge_hash_is_not_in_public_36_image_dictionary(host_run: Path) -> None:
    item = next_host_agent_item(host_run)
    actual = hashlib.sha256(Path(item["image_path"]).read_bytes()).hexdigest()
    assert actual not in {
        hashlib.sha256(_render_unsalted(shape, color, count)).hexdigest()
        for shape in host_agent.CHALLENGE_SHAPES
        for color in host_agent.CHALLENGE_COLORS
        for count in host_agent.CHALLENGE_COUNTS
    }


def _render_unsalted(shape: str, color: str, count: int) -> bytes:
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    spacing = 220 // count
    for index in range(count):
        left = 10 + index * spacing + (spacing - 44) // 2
        top, right, bottom = 20, left + 44, 64
        if shape == "triangle":
            draw.polygon([(left + 22, top), (left, bottom), (right, bottom)], fill=color)
        elif shape == "circle":
            draw.ellipse((left, top, right, bottom), fill=color)
        else:
            draw.rectangle((left, top, right, bottom), fill=color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_observer_rejects_wrong_dimensions_before_convert(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image
    image = Image.new("RGB", (10000, 1), "white")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    converted = False
    real_convert = Image.Image.convert

    def tracked_convert(self: object, *args: object, **kwargs: object) -> object:
        nonlocal converted
        converted = True
        return real_convert(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", tracked_convert)
    with pytest.raises(RuntimeError, match="dimensions"):
        host_agent._observe_challenge_png(payload.getvalue())
    assert converted is False


def _color_components(image: object, color: tuple[int, int, int]) -> list[dict]:
    pixels = image.load()
    width, height = image.size
    remaining = {(x, y) for y in range(height) for x in range(width) if pixels[x, y] == color}
    components = []
    while remaining:
        pending = [remaining.pop()]
        points = []
        while pending:
            point = pending.pop()
            points.append(point)
            x, y = point
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        ys = [point[1] for point in points]
        top, bottom = min(ys), max(ys)
        middle = (top + bottom) // 2
        row_width = lambda row: sum(point[1] == row for point in points)
        components.append({"top_width": row_width(top), "middle_width": row_width(middle),
                           "bottom_width": row_width(bottom),
                           "width": max(point[0] for point in points) - min(point[0] for point in points) + 1,
                           "height": bottom - top + 1})
    return components


def test_host_module_does_not_import_or_download_local_models() -> None:
    source = Path(host_agent.__file__).read_text(encoding="utf-8")
    for forbidden in ("local_agent", "transformers", "torch", "huggingface"):
        assert forbidden not in source


def test_host_document_reader_rejects_windows_reparse_attribute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _write_json(tmp_path / "plan.json", {})
    real_lstat = Path.lstat
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    class ReparseStatus:
        def __init__(self, wrapped: object) -> None:
            self._wrapped = wrapped
            self.st_file_attributes = getattr(wrapped, "st_file_attributes", 0) | flag

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    def fake_lstat(path: Path) -> object:
        status = real_lstat(path)
        return ReparseStatus(status) if path == document else status

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(ValueError, match="unsafe"):
        host_agent._read_json_file(document)


def test_capability_must_match_exact_visual_answer(host_run: Path, tmp_path: Path) -> None:
    item = next_host_agent_item(host_run)
    response = _capability_response(item)
    response["observed"]["count"] = 2 if response["observed"]["count"] != 2 else 3
    with pytest.raises(ValueError, match="capability"):
        record_host_plan(host_run, _write_json(tmp_path / "bad.json", response))
    assert not (host_run / "host_capabilities.json").exists()


def test_hard_coded_old_capability_answer_is_rejected(
    host_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    choices = iter(["circle", "#2b8a3e", 4])
    monkeypatch.setattr(host_agent.secrets, "choice", lambda values: next(choices))
    item = next_host_agent_item(host_run)
    response = {"schema_version": 1, "kind": "host_capability_response",
                "challenge_id": item["challenge_id"],
                "observed": {"shape": "triangle", "color": "#2f6fed", "count": 3}}
    with pytest.raises(ValueError, match="capability"):
        record_host_plan(host_run, _write_json(tmp_path / "old.json", response))


def test_metadata_and_run_key_do_not_contain_or_derive_visual_answer(
    host_run: Path
) -> None:
    item = next_host_agent_item(host_run)
    metadata_path = host_run / "host-challenge/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    key = (host_run / ".component-agent-integrity/key.bin").read_bytes()
    assert set(metadata) == {"schema_version", "challenge_id", "image_path", "image_sha256"}
    assert all(value not in json.dumps(metadata) for value in host_agent.CHALLENGE_SHAPES)
    assert len(key) == 32
    assert _observe_challenge(Path(item["image_path"])) not in (metadata, key)


def test_bound_metadata_tampering_fails(host_run: Path) -> None:
    next_host_agent_item(host_run)
    metadata_path = host_run / "host-challenge/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["image_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata|expected|challenge"):
        next_host_agent_item(host_run)


def test_bound_png_and_challenge_id_tampering_fail_closed(host_run: Path) -> None:
    next_host_agent_item(host_run)
    image = host_run / "host-challenge/challenge.png"
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="hash"):
        next_host_agent_item(host_run)

def test_bound_challenge_id_tampering_fails_closed(host_run: Path) -> None:
    next_host_agent_item(host_run)
    metadata_path = host_run / "host-challenge/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["challenge_id"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ID"):
        next_host_agent_item(host_run)


def test_missing_or_changed_integrity_key_rejects_published_challenge(host_run: Path) -> None:
    next_host_agent_item(host_run)
    key = host_run / ".component-agent-integrity/key.bin"
    original = key.read_bytes()
    key.unlink()
    with pytest.raises(RuntimeError, match="integrity key"):
        next_host_agent_item(host_run)
    key.write_bytes(original)
    key.write_bytes(b"x" * 32)
    with pytest.raises(RuntimeError, match="challenge"):
        next_host_agent_item(host_run)


def test_challenge_metadata_failure_leaves_no_publication_and_can_retry(
    host_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = host_agent._write_challenge_metadata
    monkeypatch.setattr(host_agent, "_write_challenge_metadata", lambda *args: (_ for _ in ()).throw(OSError("metadata fail")))
    with pytest.raises(OSError, match="metadata fail"):
        next_host_agent_item(host_run)
    assert not (host_run / "host-challenge").exists()
    assert not list(host_run.glob(".host-challenge.tmp-*"))
    monkeypatch.setattr(host_agent, "_write_challenge_metadata", real_write)
    assert next_host_agent_item(host_run)["kind"] == "capability_handshake"


def test_challenge_rename_failure_cleans_staging_and_can_retry(
    host_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_rename = Path.rename
    failed = False

    def fail_once(path: Path, target: Path) -> Path:
        nonlocal failed
        if path.name.startswith(".host-challenge.tmp-") and not failed:
            failed = True
            raise OSError("rename fail")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_once)
    with pytest.raises(RuntimeError, match="publication"):
        next_host_agent_item(host_run)
    assert not (host_run / "host-challenge").exists()
    assert not list(host_run.glob(".host-challenge.tmp-*"))
    assert next_host_agent_item(host_run)["kind"] == "capability_handshake"


def test_concurrent_next_calls_load_one_complete_published_challenge(
    host_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_create = host_agent._create_challenge
    started = threading.Event()
    release = threading.Event()

    def slow_create(store: RunStore) -> dict:
        started.set()
        assert release.wait(10)
        return real_create(store)

    monkeypatch.setattr(host_agent, "_create_challenge", slow_create)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(next_host_agent_item, host_run)
        assert started.wait(10)
        second = executor.submit(next_host_agent_item, host_run)
        release.set()
        results = [first.result(timeout=10), second.result(timeout=10)]
    assert results[0]["challenge_id"] == results[1]["challenge_id"]
    assert results[0]["image_path"] == results[1]["image_path"]
    assert (host_run / "host-challenge/challenge.png").is_file()
    assert (host_run / "host-challenge/metadata.json").is_file()
    assert not list(host_run.glob(".host-challenge.tmp-*"))
    assert not (host_run / ".host-challenge-publication.lock").exists()


def test_next_retries_same_main_lock_after_posix_style_parent_conflict(
    host_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = []
    enters = 0
    sleeps = []

    class FakeLease:
        def __init__(self, path: Path, *, run_root: Path) -> None:
            attempts.append((Path(path), Path(run_root)))

        def __enter__(self) -> object:
            nonlocal enters
            enters += 1
            if enters == 1:
                raise RuntimeError("Run is already executing: simulated POSIX parent lock")
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(host_agent, "ExecutionLease", FakeLease)
    monkeypatch.setattr(host_agent.time, "sleep", sleeps.append)
    item = next_host_agent_item(host_run)
    assert item["kind"] == "capability_handshake"
    assert enters == 2
    assert all(path.name == "execution.lock" for path, _ in attempts)
    assert sleeps == [0.01]


def test_capability_success_exposes_bound_component_request(host_run: Path, tmp_path: Path) -> None:
    request_path = _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    item = next_host_agent_item(host_run)
    assert item["kind"] == "component_request"
    assert item["provider"] == "host"
    assert item["request_path"] == str(request_path.resolve())
    assert all(Path(path).is_absolute() for path in item["evidence_paths"])
    assert "untrusted" in item["instructions"].lower()


def test_record_plan_rejects_provider_round_hash_and_unknown_id(host_run: Path, tmp_path: Path) -> None:
    _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    request = next_host_agent_item(host_run)
    base = {"schema_version": 1, "kind": "component_plan", "page_id": request["page_id"],
            "provider": "host", "repair_round": request["repair_round"],
            "request_sha256": request["request_sha256"], "actions": []}
    variants = [({**base, "provider": "local"}, "provider"),
                ({**base, "repair_round": 2}, "repair_round"),
                ({**base, "request_sha256": "0" * 64}, "request_sha256"),
                ({**base, "actions": [{"action": "accept", "object_ids": ["unknown"],
                  "parameters": {}, "confidence": 0.9, "evidence": ["visible boundary"]}]}, "object")]
    for index, (plan, message) in enumerate(variants):
        with pytest.raises(ValueError, match=message):
            record_host_plan(host_run, _write_json(tmp_path / f"plan-{index}.json", plan))


@pytest.mark.parametrize(
    "action",
    [
        {"action": "retry_with_box", "object_ids": ["candidate_1"],
         "parameters": {"box": [-0.1, 0.1, 0.5, 0.5]},
         "confidence": 0.9, "evidence": ["edge"]},
        {"action": "accept", "object_ids": ["candidate_1"],
         "parameters": {}, "confidence": float("nan"), "evidence": ["edge"]},
        {"action": "accept", "object_ids": ["candidate_1"],
         "parameters": {}, "confidence": 0.9, "evidence": []},
    ],
)
def test_record_plan_rejects_invalid_coordinates_confidence_and_evidence(
    host_run: Path, tmp_path: Path, action: dict
) -> None:
    _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    request = next_host_agent_item(host_run)
    plan = {"schema_version": 1, "kind": "component_plan", "page_id": "page_001",
            "provider": "host", "repair_round": 1,
            "request_sha256": request["request_sha256"], "actions": [action]}
    with pytest.raises(ValueError):
        record_host_plan(host_run, _write_json(tmp_path / "plan.json", plan))


def test_record_valid_plan_is_atomic_resumable_and_not_repeatable(host_run: Path, tmp_path: Path) -> None:
    _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    request = next_host_agent_item(host_run)
    plan_path = _write_json(tmp_path / "plan.json", {
        "schema_version": 1, "kind": "component_plan", "page_id": "page_001",
        "provider": "host", "repair_round": 1,
        "request_sha256": request["request_sha256"], "actions": []})
    result = record_host_plan(host_run, plan_path)
    assert result["status"] == "recorded"
    assert RunStore.open(host_run).read_json("run_state.json")["status"] == "prepared"
    assert Path(result["plan_path"]).is_file()
    state = RunStore.open(host_run).read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    assert state["phase"] == "plan_recorded"
    assert state["plan_count"] == 1
    assert state["current_round"]["plan_ref"]["sha256"] == hashlib.sha256(
        Path(result["plan_path"]).read_bytes()
    ).hexdigest()
    outcome = advance_component_repair(RunStore.open(host_run), "page_001")
    assert outcome["status"] == "fallback_required"
    state = RunStore.open(host_run).read_json(
        "pages/page_001/reconstruction/component_state.json"
    )
    assert state["stop_reason"] == "empty_plan"
    with pytest.raises(RuntimeError, match="already recorded"):
        record_host_plan(host_run, plan_path)


def test_same_plan_recovers_state_transition_after_plan_publication(
    host_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    request = next_host_agent_item(host_run)
    plan = {"schema_version": 1, "kind": "component_plan", "page_id": "page_001",
            "provider": "host", "repair_round": 1,
            "request_sha256": request["request_sha256"], "actions": []}
    plan_path = _write_json(tmp_path / "plan.json", plan)
    real_transition = RunStore.transition_run
    failed = False

    def fail_once(store: RunStore, target: RunStatus) -> dict:
        nonlocal failed
        if target is RunStatus.PREPARED and not failed:
            failed = True
            raise OSError("simulated transition failure")
        return real_transition(store, target)

    monkeypatch.setattr(RunStore, "transition_run", fail_once)
    with pytest.raises(OSError, match="transition failure"):
        record_host_plan(host_run, plan_path)
    records = list(host_run.glob("host-component-plan-*.json"))
    assert len(records) == 1
    before = records[0].read_bytes()
    assert RunStore.open(host_run).read_json("run_state.json")["status"] == "awaiting_agent"

    result = record_host_plan(host_run, plan_path)

    assert result["status"] == "recorded"
    assert result["recovered"] is True
    assert records[0].read_bytes() == before
    assert RunStore.open(host_run).read_json("run_state.json")["status"] == "prepared"
    with pytest.raises(RuntimeError, match="already recorded"):
        record_host_plan(host_run, plan_path)


def test_different_plan_cannot_recover_half_committed_record(
    host_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    request = next_host_agent_item(host_run)
    plan = {"schema_version": 1, "kind": "component_plan", "page_id": "page_001",
            "provider": "host", "repair_round": 1,
            "request_sha256": request["request_sha256"], "actions": []}
    plan_path = _write_json(tmp_path / "plan.json", plan)
    real_transition = RunStore.transition_run
    monkeypatch.setattr(RunStore, "transition_run", lambda store, target: (_ for _ in ()).throw(OSError("fail")) if target is RunStatus.PREPARED else real_transition(store, target))
    with pytest.raises(OSError):
        record_host_plan(host_run, plan_path)
    monkeypatch.setattr(RunStore, "transition_run", real_transition)
    changed = {**plan, "actions": [{"action": "accept", "object_ids": ["candidate_1"],
                                     "parameters": {}, "confidence": 0.9,
                                     "evidence": ["visible boundary"]}]}
    with pytest.raises(RuntimeError, match="different|already recorded"):
        record_host_plan(host_run, _write_json(tmp_path / "changed.json", changed))
    assert RunStore.open(host_run).read_json("run_state.json")["status"] == "awaiting_agent"


def test_wrong_sha_preexisting_plan_cannot_recover_run(
    host_run: Path, tmp_path: Path
) -> None:
    _publish_request(host_run)
    handshake = next_host_agent_item(host_run)
    record_host_plan(host_run, _write_json(tmp_path / "capability.json", _capability_response(handshake)))
    request = next_host_agent_item(host_run)
    bad = {"schema_version": 1, "kind": "component_plan", "page_id": "page_001",
           "provider": "host", "repair_round": 1, "request_sha256": "0" * 64,
           "actions": []}
    destination = host_run / f"host-component-plan-page_001-01-{request['request_sha256']}.json"
    destination.write_text(json.dumps(bad), encoding="utf-8")
    plan_path = _write_json(tmp_path / "bad-plan.json", bad)
    with pytest.raises(ValueError, match="request_sha256"):
        record_host_plan(host_run, plan_path)
    assert RunStore.open(host_run).read_json("run_state.json")["status"] == "awaiting_agent"
