from __future__ import annotations

import copy
from datetime import datetime
import multiprocessing
import os
import re
import subprocess
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pptx import Presentation

from image2editable import legacy, runtime
from image2editable.contracts import PageStatus, RunStatus, SCHEMA_VERSION
from image2editable.execution import ExecutionLease
from image2editable.pptx_input import prepare_pptx_job
from image2editable.resources import safe_default_policy
from image2editable.store import RunStore


def _image(path: Path, color: tuple[int, int, int] = (1, 2, 3)) -> None:
    Image.new("RGB", (12, 8), color).save(path)


def _pptx(path: Path, slide_count: int = 2) -> None:
    presentation = Presentation()
    for _ in range(slide_count):
        presentation.slides.add_slide(presentation.slide_layouts[6])
    presentation.save(path)


def _run_synchronized(
    run_dir: str,
    barrier: object,
    release: object,
    results: object,
) -> None:
    real_open = runtime.RunStore.open
    first_state_read = True

    def synchronized_open(root: str | Path) -> RunStore:
        nonlocal first_state_read
        store = real_open(root)
        read_json = store.read_json

        def synchronized_read(relative: str | Path) -> dict[str, Any]:
            nonlocal first_state_read
            document = read_json(relative)
            if str(relative) == "run_state.json" and first_state_read:
                first_state_read = False
                barrier.wait(10)
            return document

        store.read_json = synchronized_read
        return store

    def execute(store: RunStore) -> dict[str, str]:
        release.wait(10)
        return {"pptx": str(store.root / "final" / "output.pptx")}

    runtime.RunStore.open = synchronized_open
    runtime.execute_legacy = execute
    try:
        results.put(("ok", runtime.run_job(run_dir)))
    except Exception as error:
        results.put(("error", type(error).__name__, str(error)))


def test_run_job_prepared_race_has_only_one_executor(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    release = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_run_synchronized,
            args=(str(run_dir), barrier, release, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    try:
        outcomes = [results.get(timeout=10)]
        release.set()
        outcomes.append(results.get(timeout=10))
    finally:
        release.set()
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(10)

    assert sorted(item[0] for item in outcomes) == ["error", "ok"]
    assert "already executing" in next(
        item[2] for item in outcomes if item[0] == "error"
    )
    assert all(process.exitcode == 0 for process in processes)


def test_run_job_writes_execution_metadata_while_lease_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    captured = {}

    def execute(store: RunStore) -> dict[str, str]:
        captured.update(store.read_json("execution.json"))
        with pytest.raises(RuntimeError, match="already executing"):
            with ExecutionLease(store.root / "execution.lock"):
                pass
        return {"pptx": str(store.root / "final" / "output.pptx")}

    monkeypatch.setattr(runtime, "execute_legacy", execute)

    runtime.run_job(run_dir)

    assert captured["schema_version"] == SCHEMA_VERSION
    assert re.fullmatch(r"[0-9a-f]{32}", captured["token"])
    assert captured["pid"] == os.getpid()
    assert datetime.fromisoformat(captured["started_at"].replace("Z", "+00:00")).tzinfo
    assert captured["input_type"] == "images"


def test_recover_orphaned_image_run_resets_pages_and_cleans_owned_dirs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_page("page_001", PageStatus.PROCESSING)
    for name in ("work", "final"):
        directory = run_dir / name
        directory.mkdir()
        (directory / "partial.bin").write_bytes(b"partial")

    status = runtime.recover_job(run_dir)

    assert status["run"]["status"] == "prepared"
    assert status["pages"]["pages"]["page_001"]["status"] == "pending"
    assert not (run_dir / "work").exists()
    assert not (run_dir / "final").exists()


def test_recover_finalizing_image_run_resets_validated_page(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_page("page_001", PageStatus.PROCESSING)
    store.transition_page("page_001", PageStatus.VALIDATED)
    store.transition_run(RunStatus.FINALIZING)

    status = runtime.recover_job(run_dir)

    assert status["run"]["status"] == "prepared"
    assert status["pages"]["pages"]["page_001"]["status"] == "pending"


def test_recover_orphaned_pptx_run_keeps_pages_analyzed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)

    status = runtime.recover_job(run_dir)

    assert status["run"]["status"] == "prepared"
    assert status["pages"]["pages"]["page_001"]["status"] == "analyzed"


def test_recover_rejects_active_execution_lease(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)

    with ExecutionLease(run_dir / "execution.lock"):
        with pytest.raises(RuntimeError, match="already executing"):
            runtime.recover_job(run_dir)


def test_recover_rejects_existing_external_output_without_state_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "external.pptx"
    _image(source)
    run_dir = runtime.prepare_job(
        source,
        run_dir=tmp_path / "run",
        output_path=output,
    )
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_page("page_001", PageStatus.PROCESSING)
    output.write_bytes(b"user")
    before = runtime.get_status(run_dir)

    with pytest.raises(RuntimeError, match="external output"):
        runtime.recover_job(run_dir)

    assert runtime.get_status(run_dir) == before
    assert output.read_bytes() == b"user"


def test_recover_treats_linked_external_parent_as_external(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    final = run_dir / "final"
    final.mkdir()
    external_parent = tmp_path / "external"
    try:
        external_parent.symlink_to(final, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"directory links are unavailable: {error}")
        junction = subprocess.run(
            [
                "cmd",
                "/c",
                "mklink",
                "/J",
                str(external_parent),
                str(final),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if junction.returncode:
            pytest.skip(f"directory links are unavailable: {error}")
    output = external_parent / "output.pptx"
    output.write_bytes(b"user")
    manifest = store.read_json("job_manifest.json")
    manifest["options"]["output_path"] = str(output.absolute())
    store.write_json("job_manifest.json", manifest)
    store.transition_run(RunStatus.RUNNING)
    store.transition_page("page_001", PageStatus.PROCESSING)
    before = runtime.get_status(run_dir)

    with pytest.raises(RuntimeError, match="external output"):
        runtime.recover_job(run_dir)

    assert runtime.get_status(run_dir) == before
    assert output.read_bytes() == b"user"


def test_recover_rejects_existing_external_single_image_variant(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "external.pptx"
    _image(source)
    run_dir = runtime.prepare_job(
        source,
        run_dir=tmp_path / "run",
        output_path=output,
        slide_size="both",
    )
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_page("page_001", PageStatus.PROCESSING)
    variant = tmp_path / "external_original.pptx"
    variant.write_bytes(b"user")
    before = runtime.get_status(run_dir)

    with pytest.raises(RuntimeError, match="external output"):
        runtime.recover_job(run_dir)

    assert runtime.get_status(run_dir) == before
    assert variant.read_bytes() == b"user"


def test_recover_rejects_existing_external_batch_variant_directory(
    tmp_path: Path,
) -> None:
    sources = [tmp_path / "first.png", tmp_path / "second.png"]
    for source in sources:
        _image(source)
    output = tmp_path / "external.pptx"
    run_dir = runtime.prepare_job(
        sources,
        run_dir=tmp_path / "run",
        output_path=output,
        slide_size="both",
    )
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    for page_id in ("page_001", "page_002"):
        store.transition_page(page_id, PageStatus.PROCESSING)
    variant_dir = tmp_path / "external_original"
    variant_dir.mkdir()
    sentinel = variant_dir / "first_original.pptx"
    sentinel.write_bytes(b"user")
    before = runtime.get_status(run_dir)

    with pytest.raises(RuntimeError, match="external output"):
        runtime.recover_job(run_dir)

    assert runtime.get_status(run_dir) == before
    assert sentinel.read_bytes() == b"user"


def test_recover_rejects_existing_external_original_batch_directory(
    tmp_path: Path,
) -> None:
    sources = [tmp_path / "first.png", tmp_path / "second.png"]
    for source in sources:
        _image(source)
    output = tmp_path / "external.pptx"
    run_dir = runtime.prepare_job(
        sources,
        run_dir=tmp_path / "run",
        output_path=output,
        slide_size="original",
    )
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    for page_id in ("page_001", "page_002"):
        store.transition_page(page_id, PageStatus.PROCESSING)
    variant_dir = tmp_path / "external_original"
    variant_dir.mkdir()
    sentinel = variant_dir / "first_original.pptx"
    sentinel.write_bytes(b"user")
    before = runtime.get_status(run_dir)

    with pytest.raises(RuntimeError, match="external output"):
        runtime.recover_job(run_dir)

    assert runtime.get_status(run_dir) == before
    assert sentinel.read_bytes() == b"user"


def test_recover_rejects_existing_pptx_output_without_state_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    output = run_dir / "final" / "output.pptx"
    output.parent.mkdir()
    output.write_bytes(b"unknown owner")
    before = runtime.get_status(run_dir)

    with pytest.raises(RuntimeError, match="PPTX.*output"):
        runtime.recover_job(run_dir)

    assert runtime.get_status(run_dir) == before
    assert output.read_bytes() == b"unknown owner"


@pytest.mark.parametrize("status", [RunStatus.PREPARED, RunStatus.FAILED])
def test_recover_rejects_non_orphan_status_without_state_change(
    tmp_path: Path, status: RunStatus
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / status.value)
    store = RunStore.open(run_dir)
    if status is RunStatus.FAILED:
        store.transition_run(status)
    before = runtime.get_status(run_dir)

    with pytest.raises(RuntimeError, match="running or finalizing"):
        runtime.recover_job(run_dir)

    assert runtime.get_status(run_dir) == before


def test_recover_rejects_linked_cleanup_path_without_state_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_page("page_001", PageStatus.PROCESSING)
    try:
        (run_dir / "work").symlink_to(external, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Cannot create symlink: {error}")
    before = runtime.get_status(run_dir)

    with pytest.raises(RuntimeError, match="work"):
        runtime.recover_job(run_dir)

    assert runtime.get_status(run_dir) == before
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_recover_rejects_non_directory_cleanup_path_without_state_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_page("page_001", PageStatus.PROCESSING)
    work = run_dir / "work"
    work.write_bytes(b"keep")
    before = runtime.get_status(run_dir)

    with pytest.raises(RuntimeError, match="work"):
        runtime.recover_job(run_dir)

    assert runtime.get_status(run_dir) == before
    assert work.read_bytes() == b"keep"


def test_recover_rejects_pptx_preserved_page_before_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_page("page_001", PageStatus.PRESERVED)
    work = run_dir / "work"
    work.mkdir()
    sentinel = work / "sentinel"
    sentinel.write_bytes(b"keep")
    before = runtime.get_status(run_dir)

    with pytest.raises(RuntimeError, match="PPTX.*blocked"):
        runtime.recover_job(run_dir)

    assert runtime.get_status(run_dir) == before
    assert sentinel.read_bytes() == b"keep"


def test_recover_job_is_exported() -> None:
    import image2editable

    assert image2editable.recover_job is runtime.recover_job


def test_run_job_does_not_acquire_lease_for_non_prepared_run(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)

    with ExecutionLease(run_dir / "execution.lock"):
        with pytest.raises(RuntimeError, match="current status is running"):
            runtime.run_job(run_dir)


def test_run_job_preserves_pptx_without_calling_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source)
    source_bytes = source.read_bytes()
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")

    def unexpected_legacy(store: RunStore) -> dict[str, Any]:
        raise AssertionError("PPTX run entered legacy execution")

    monkeypatch.setattr(runtime, "execute_legacy", unexpected_legacy)

    summary = runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    output = Path(summary["outputs"]["pptx"])

    assert summary["status"] == "completed"
    assert summary["pages"] == 2
    assert summary["resource_policy"] == safe_default_policy()
    assert "_output_identity" not in summary
    assert output.read_bytes() == source_bytes
    assert store.read_json("run_summary.json") == summary
    assert store.read_json("run_state.json")["status"] == "completed"
    assert {
        page["status"]
        for page in store.read_json("page_jobs.json")["pages"].values()
    } == {"preserved"}


def test_run_job_executes_agent_approved_shadow_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "slide.png"
    _image(image)
    source = tmp_path / "source.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.shapes.add_picture(
        str(image),
        0,
        0,
        presentation.slide_width,
        presentation.slide_height,
    )
    presentation.save(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    candidate = runtime.next_candidate(run_dir)["candidate"]
    runtime.record_decision(
        run_dir,
        page_id="page_001",
        object_id=candidate["source_shape_id"],
        decision="replace",
        confidence=0.99,
        category="full_slide_screenshot",
        evidence=["complete slide layout"],
    )
    calls = []

    def fake_execute(store: RunStore, plans) -> dict[str, Any]:
        calls.append(plans)
        output = store.root / "final/output.pptx"
        output.parent.mkdir(parents=True, exist_ok=True)
        input_path = store.root / "input/original.pptx"
        output.write_bytes(input_path.read_bytes())
        digest = runtime.sha256_file(output)
        status = output.lstat()
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "pages": 1,
            "preserved_objects": 0,
            "pending_candidates": 0,
            "replaced_pages": 1,
            "preserved_with_warning_pages": 0,
            "page_results": [
                {
                    "schema_version": SCHEMA_VERSION,
                    "page_id": "page_001",
                    "status": "replaced",
                }
            ],
            "warnings": [],
            "outputs": {"pptx": str(output)},
            "input_sha256": runtime.sha256_file(input_path),
            "output_sha256": digest,
            "_output_identity": {
                "version": 1,
                "path": str(output),
                "dev": status.st_dev,
                "ino": status.st_ino,
                "mode": status.st_mode,
                "size": status.st_size,
                "mtime_ns": status.st_mtime_ns,
                "sha256": digest,
            },
        }

    monkeypatch.setattr(
        runtime,
        "execute_pptx_shadow",
        fake_execute,
        raising=False,
    )

    summary = runtime.run_job(run_dir)

    assert len(calls) == 1
    assert calls[0][0]["page_id"] == "page_001"
    assert summary["replaced_pages"] == 1
    assert (
        RunStore.open(run_dir)
        .read_json("page_jobs.json")["pages"]["page_001"]["status"]
        == "replaced"
    )
    assert runtime.run_job(run_dir) == summary
    assert len(calls) == 1


def test_completed_pptx_run_is_idempotent_without_recopy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    completed = runtime.run_job(run_dir)
    output = Path(completed["outputs"]["pptx"])
    before = output.stat()

    def unexpected_execute(store: RunStore) -> dict[str, object]:
        raise AssertionError("completed PPTX run copied output again")

    monkeypatch.setattr(runtime, "execute_pptx_preserve", unexpected_execute)

    assert runtime.run_job(run_dir) == completed
    assert output.stat().st_mtime_ns == before.st_mtime_ns
    assert output.stat().st_size == before.st_size


@pytest.mark.parametrize(
    "damage",
    [
        "native_missing",
        "candidates_missing",
        "native_hash",
        "candidates_hash",
        "metadata",
        "objects",
        "candidates",
        "count",
        "manifest_record",
    ],
)
def test_completed_pptx_run_revalidates_bound_inventories_without_mutating_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    native_relative = "pages/page_001/native_objects.json"
    candidates_relative = "pages/page_001/screenshot_candidates.json"
    native_path = run_dir / native_relative
    candidates_path = run_dir / candidates_relative
    manifest = store.read_json("job_manifest.json")
    record = manifest["input"]["inventories"][0]

    if damage == "native_missing":
        native_path.unlink()
    elif damage == "candidates_missing":
        candidates_path.unlink()
    elif damage == "native_hash":
        native_path.write_bytes(native_path.read_bytes() + b" ")
    elif damage == "candidates_hash":
        candidates_path.write_bytes(candidates_path.read_bytes() + b" ")
    elif damage == "metadata":
        native = store.read_json(native_relative)
        native["slide_part"] = "ppt/slides/slide999.xml"
        store.write_json(native_relative, native)
        record["native_objects_sha256"] = runtime.sha256_file(native_path)
        store.write_json("job_manifest.json", manifest)
    elif damage == "objects":
        native = store.read_json(native_relative)
        native["objects"] = [{"action": "invalid"}]
        store.write_json(native_relative, native)
        record["native_objects_sha256"] = runtime.sha256_file(native_path)
        store.write_json("job_manifest.json", manifest)
    elif damage == "candidates":
        candidates = store.read_json(candidates_relative)
        candidates["candidates"] = [{"action": "candidate"}]
        store.write_json(candidates_relative, candidates)
        record["screenshot_candidates_sha256"] = runtime.sha256_file(
            candidates_path
        )
        store.write_json("job_manifest.json", manifest)
    elif damage == "count":
        native = store.read_json(native_relative)
        native["objects"] = [{"action": "preserve"}]
        store.write_json(native_relative, native)
        record["native_objects_sha256"] = runtime.sha256_file(native_path)
        store.write_json("job_manifest.json", manifest)
    else:
        manifest["input"]["inventories"].pop()
        store.write_json("job_manifest.json", manifest)

    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    before_summary = store.read_json("run_summary.json")

    def unexpected_execute(_store: RunStore) -> dict[str, object]:
        raise AssertionError("completed PPTX run executed again")

    monkeypatch.setattr(runtime, "execute_pptx_preserve", unexpected_execute)

    with pytest.raises((RuntimeError, ValueError), match="PPTX"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages
    assert store.read_json("run_summary.json") == before_summary


def test_pptx_run_revalidates_inventory_before_running_state_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    native_path = run_dir / "pages/page_001/native_objects.json"
    native_path.write_bytes(native_path.read_bytes() + b" ")
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")

    def unexpected_execute(_store: RunStore) -> dict[str, object]:
        raise AssertionError("invalid PPTX inventory executed")

    monkeypatch.setattr(runtime, "execute_pptx_preserve", unexpected_execute)

    with pytest.raises(RuntimeError, match="PPTX inventory hash"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages


def test_completed_pptx_inventory_validation_uses_runtime_manifest_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    native_path = run_dir / "pages/page_001/native_objects.json"
    native_path.write_bytes(native_path.read_bytes() + b" ")
    replacement = copy.deepcopy(manifest)
    replacement["input"]["inventories"][0][
        "native_objects_sha256"
    ] = runtime.sha256_file(native_path)
    store.write_json("job_manifest.json", replacement)
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    before_summary = store.read_json("run_summary.json")

    monkeypatch.setattr(
        runtime,
        "_manifest_input",
        lambda _store: (manifest, "pptx"),
    )

    with pytest.raises(RuntimeError, match="PPTX inventory hash"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages
    assert store.read_json("run_summary.json") == before_summary


def test_pptx_execution_reuses_pretransition_manifest_inventory_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    native_path = run_dir / "pages/page_001/native_objects.json"
    original_transition_run = RunStore.transition_run
    replaced = False

    def replace_manifest_after_running(
        self: RunStore, target: RunStatus
    ) -> dict[str, Any]:
        nonlocal replaced
        result = original_transition_run(self, target)
        if target is RunStatus.RUNNING and not replaced:
            replaced = True
            native_path.write_bytes(native_path.read_bytes() + b" ")
            replacement = copy.deepcopy(manifest)
            replacement["input"]["inventories"][0][
                "native_objects_sha256"
            ] = runtime.sha256_file(native_path)
            self.write_json("job_manifest.json", replacement)
        return result

    monkeypatch.setattr(RunStore, "transition_run", replace_manifest_after_running)

    with pytest.raises(RuntimeError, match="PPTX inventory hash"):
        runtime.run_job(run_dir)

    assert replaced is True
    assert store.read_json("run_state.json")["status"] == "failed"
    assert not (run_dir / "final/output.pptx").exists()


def test_completed_pptx_rejects_inventory_replaced_after_trusted_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    native_path = run_dir / "pages/page_001/native_objects.json"
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"tampered":true}')
    original_open = Path.open
    replaced = False

    class ReplaceAfterRead:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self.file.__enter__()

        def __exit__(self, *args):
            nonlocal replaced
            result = self.file.__exit__(*args)
            runtime.os.replace(replacement, native_path)
            replaced = True
            return result

    def replace_inventory(path: Path, *args, **kwargs):
        file = original_open(path, *args, **kwargs)
        if path == native_path and args == ("rb",):
            return ReplaceAfterRead(file)
        return file

    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    before_summary = store.read_json("run_summary.json")
    monkeypatch.setattr(Path, "open", replace_inventory)

    with pytest.raises(RuntimeError, match="changed during verification"):
        runtime.run_job(run_dir)

    assert replaced is True
    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages
    assert store.read_json("run_summary.json") == before_summary


@pytest.mark.parametrize("damage", ["missing", "bad_bytes", "directory"])
def test_completed_pptx_run_rejects_invalid_output_entry_without_mutating_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    completed = runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    output = Path(completed["outputs"]["pptx"])
    output.unlink()
    if damage == "bad_bytes":
        output.write_bytes(b"corrupt")
    elif damage == "directory":
        output.mkdir()
    before_run = store.read_json("run_state.json")
    before_summary = store.read_json("run_summary.json")

    def unexpected_execute(_store: RunStore) -> dict[str, object]:
        raise AssertionError("completed PPTX run executed again")

    monkeypatch.setattr(runtime, "execute_pptx_preserve", unexpected_execute)

    with pytest.raises(RuntimeError, match="PPTX completed output"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_run
    assert store.read_json("run_summary.json") == before_summary


def test_completed_pptx_run_rejects_output_symlink_without_mutating_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    completed = runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    output = Path(completed["outputs"]["pptx"])
    output.unlink()
    try:
        output.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    before_run = store.read_json("run_state.json")
    before_summary = store.read_json("run_summary.json")

    def unexpected_execute(_store: RunStore) -> dict[str, object]:
        raise AssertionError("completed PPTX run executed again")

    monkeypatch.setattr(runtime, "execute_pptx_preserve", unexpected_execute)

    with pytest.raises(RuntimeError, match="PPTX completed output"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_run
    assert store.read_json("run_summary.json") == before_summary


def test_completed_pptx_run_rejects_output_replaced_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    completed = runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    output = Path(completed["outputs"]["pptx"])
    replacement = tmp_path / "replacement.pptx"
    replacement.write_bytes(output.read_bytes())
    before_run = store.read_json("run_state.json")
    before_summary = store.read_json("run_summary.json")
    original_sha256_file = runtime.sha256_file

    def replace_after_hash(path: Path) -> str:
        digest = original_sha256_file(path)
        if Path(path) == output:
            runtime.os.replace(replacement, output)
        return digest

    def unexpected_execute(_store: RunStore) -> dict[str, object]:
        raise AssertionError("completed PPTX run executed again")

    monkeypatch.setattr(runtime, "sha256_file", replace_after_hash)
    monkeypatch.setattr(runtime, "execute_pptx_preserve", unexpected_execute)

    with pytest.raises(RuntimeError, match="changed during verification"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_run
    assert store.read_json("run_summary.json") == before_summary


def test_pptx_run_rejects_legacy_unbound_inventories_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    manifest["input"].pop("inventories", None)
    store.write_json("job_manifest.json", manifest)
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")

    def unexpected_execute(_store: RunStore) -> dict[str, object]:
        raise AssertionError("unbound PPTX run executed")

    monkeypatch.setattr(runtime, "execute_pptx_preserve", unexpected_execute)

    with pytest.raises(RuntimeError, match="inventor"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages


@pytest.mark.parametrize("invalid_slide_count", [True, 1.0, -1, 2])
def test_pptx_manifest_slide_count_is_validated_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_slide_count: object,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    manifest["input"]["slide_count"] = invalid_slide_count
    store.write_json("job_manifest.json", manifest)
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    called = False

    def malicious_execute(_store: RunStore) -> dict[str, object]:
        nonlocal called
        called = True
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "pages": invalid_slide_count,
        }

    monkeypatch.setattr(runtime, "execute_pptx_preserve", malicious_execute)

    with pytest.raises(RuntimeError, match="slide_count"):
        runtime.run_job(run_dir)

    assert called is False
    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("status", "failed"),
        ("pages", True),
        ("preserved_objects", 1),
        ("pending_candidates", 1),
        ("warnings", ["unexpected"]),
        ("outputs", {"pptx": "wrong"}),
        ("input_sha256", "0" * 64),
        ("output_sha256", "0" * 64),
        (
            "resource_policy",
            {**safe_default_policy(), "heavy_page_concurrency": True},
        ),
        ("_output_identity", {}),
        ("unknown_public", True),
    ],
)
def test_completed_pptx_summary_is_revalidated_against_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    completed = runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    summary = dict(completed)
    summary[field] = value
    store.write_json("run_summary.json", summary)
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")

    def unexpected_execute(_store: RunStore) -> dict[str, object]:
        raise AssertionError("completed PPTX run executed again")

    monkeypatch.setattr(runtime, "execute_pptx_preserve", unexpected_execute)

    with pytest.raises(RuntimeError, match="PPTX execution summary"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages
    assert store.read_json("run_summary.json") == summary


def test_pptx_execution_failure_records_analyzed_pages_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")

    def fail_execute(store: RunStore) -> dict[str, object]:
        raise RuntimeError("preserve failed")

    monkeypatch.setattr(runtime, "execute_pptx_preserve", fail_execute)

    with pytest.raises(RuntimeError, match="preserve failed"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "failed"
    assert {
        page["status"]
        for page in store.read_json("page_jobs.json")["pages"].values()
    } == {"failed"}
    assert store.read_json("run_summary.json") == {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": {"type": "RuntimeError", "message": "preserve failed"},
        "outputs": {},
    }


def test_pptx_run_never_overwrites_preexisting_output(tmp_path: Path) -> None:
    source = tmp_path / "source.pptx"
    output = tmp_path / "output.pptx"
    _pptx(source, slide_count=1)
    output.write_bytes(b"existing")
    run_dir = runtime.prepare_job(
        source,
        run_dir=tmp_path / "run",
        output_path=output,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        runtime.run_job(run_dir)

    assert output.read_bytes() == b"existing"
    store = RunStore.open(run_dir)
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    with pytest.raises(RuntimeError, match="blocked"):
        runtime.retry_page(run_dir, "page_001")
    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages
    assert output.read_bytes() == b"existing"


@pytest.mark.parametrize(
    "failure_point",
    ["finalizing", "summary", "completed"],
)
def test_pptx_post_publish_failure_compensates_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    source_bytes = source.read_bytes()
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_transition_run = RunStore.transition_run
    original_write_json = RunStore.write_json
    injected = False

    def fail_transition(
        self: RunStore, target: RunStatus
    ) -> dict[str, Any]:
        nonlocal injected
        should_fail = (
            failure_point == "finalizing" and target is RunStatus.FINALIZING
        ) or (
            failure_point == "completed" and target is RunStatus.COMPLETED
        )
        if should_fail and not injected:
            injected = True
            raise OSError(f"{failure_point} state write failed")
        return original_transition_run(self, target)

    def fail_summary_write(
        self: RunStore, relative: str | Path, document: dict[str, Any]
    ) -> None:
        nonlocal injected
        if (
            failure_point == "summary"
            and Path(relative) == Path("run_summary.json")
            and document.get("status") == "completed"
            and not injected
        ):
            injected = True
            raise OSError("summary write failed")
        original_write_json(self, relative, document)

    monkeypatch.setattr(RunStore, "transition_run", fail_transition)
    monkeypatch.setattr(RunStore, "write_json", fail_summary_write)

    with pytest.raises(OSError, match="write failed"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert not output.exists()
    assert store.read_json("run_state.json")["status"] == "failed"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == "failed"
    assert store.read_json("run_summary.json")["status"] == "failed"

    retried = runtime.retry_page(run_dir, "page_001")
    assert retried["run"]["status"] == "prepared"
    assert retried["pages"]["pages"]["page_001"]["status"] == "analyzed"
    assert runtime.run_job(run_dir)["status"] == "completed"
    assert output.read_bytes() == source_bytes


def test_pptx_compensation_does_not_delete_concurrent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_transition_run = RunStore.transition_run

    def replace_then_fail(
        self: RunStore, target: RunStatus
    ) -> dict[str, Any]:
        if target is RunStatus.FINALIZING:
            output.unlink()
            output.write_bytes(b"concurrent replacement")
            raise OSError("finalizing state write failed")
        return original_transition_run(self, target)

    monkeypatch.setattr(RunStore, "transition_run", replace_then_fail)

    with pytest.raises(OSError, match="finalizing") as error:
        runtime.run_job(run_dir)

    assert error.value.__cause__ is not None
    assert "safely" in str(error.value.__cause__)
    assert output.read_bytes() == b"concurrent replacement"
    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "failed"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == "preserved"
    with pytest.raises(RuntimeError, match="blocked"):
        runtime.retry_page(run_dir, "page_001")
    assert store.read_json("run_state.json")["status"] == "failed"
    assert output.read_bytes() == b"concurrent replacement"


def test_pptx_compensation_isolates_path_before_deleting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_transition_run = RunStore.transition_run
    original_replace = runtime.os.replace
    replacement_injected = False

    def replace_before_isolation(source_path, destination_path):
        nonlocal replacement_injected
        if Path(source_path) == output and not replacement_injected:
            replacement_injected = True
            output.unlink()
            output.write_bytes(b"last-moment replacement")
        return original_replace(source_path, destination_path)

    def fail_finalizing(
        self: RunStore, target: RunStatus
    ) -> dict[str, Any]:
        if target is RunStatus.FINALIZING:
            raise OSError("finalizing state write failed")
        return original_transition_run(self, target)

    monkeypatch.setattr(runtime.os, "replace", replace_before_isolation)
    monkeypatch.setattr(RunStore, "transition_run", fail_finalizing)

    with pytest.raises(OSError, match="finalizing"):
        runtime.run_job(run_dir)

    assert replacement_injected is True
    assert output.read_bytes() == b"last-moment replacement"
    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "failed"
    with pytest.raises(RuntimeError, match="blocked"):
        runtime.retry_page(run_dir, "page_001")
    assert store.read_json("run_state.json")["status"] == "failed"


def test_pptx_forged_summary_hash_cleans_token_owned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_execute = runtime.execute_pptx_preserve

    def return_wrong_output_hash(store: RunStore) -> dict[str, object]:
        summary = original_execute(store)
        summary["output_sha256"] = "0" * 64
        return summary

    monkeypatch.setattr(
        runtime, "execute_pptx_preserve", return_wrong_output_hash
    )

    with pytest.raises(RuntimeError, match="hash does not match"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert not output.exists()
    assert store.read_json("run_state.json")["status"] == "failed"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == "failed"
    assert store.read_json("run_summary.json")["status"] == "failed"
    assert "retry_blocked" not in store.read_json("run_summary.json")
    monkeypatch.setattr(runtime, "execute_pptx_preserve", original_execute)
    assert runtime.retry_page(run_dir, "page_001")["run"]["status"] == "prepared"
    assert runtime.run_job(run_dir)["status"] == "completed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("status", "failed"),
        ("pages", True),
        ("preserved_objects", True),
        ("pending_candidates", 0.0),
        ("warnings", ["unexpected"]),
        ("outputs", {"pptx": "wrong"}),
        ("input_sha256", "0" * 64),
    ],
)
def test_pptx_invalid_summary_is_cleaned_and_retryable_when_token_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_execute = runtime.execute_pptx_preserve

    def forge_summary(store: RunStore) -> dict[str, object]:
        summary = original_execute(store)
        summary[field] = value
        return summary

    monkeypatch.setattr(runtime, "execute_pptx_preserve", forge_summary)

    with pytest.raises(RuntimeError, match="PPTX execution summary"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert not output.exists()
    assert store.read_json("run_state.json")["status"] == "failed"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == "failed"
    assert "retry_blocked" not in store.read_json("run_summary.json")
    monkeypatch.setattr(runtime, "execute_pptx_preserve", original_execute)
    assert runtime.retry_page(run_dir, "page_001")["run"]["status"] == "prepared"
    assert runtime.run_job(run_dir)["status"] == "completed"


def test_pptx_self_consistent_wrong_bytes_are_rejected_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from image2editable.pptx_input import _publish_pptx_no_clobber

    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_execute = runtime.execute_pptx_preserve

    def publish_wrong_bytes(store: RunStore) -> dict[str, object]:
        manifest = store.read_json("job_manifest.json")
        temporary = output.parent / ".malicious.tmp"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(b"not-a-pptx")
        token = _publish_pptx_no_clobber(temporary, output)
        temporary.unlink()
        digest = runtime.sha256_file(output)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "pages": 1,
            "preserved_objects": manifest["input"]["object_count"],
            "pending_candidates": manifest["input"]["candidate_count"],
            "warnings": [],
            "outputs": {"pptx": str(output)},
            "input_sha256": digest,
            "output_sha256": digest,
            "_output_identity": token,
        }

    monkeypatch.setattr(runtime, "execute_pptx_preserve", publish_wrong_bytes)

    with pytest.raises(RuntimeError, match="manifest"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert not output.exists()
    assert "retry_blocked" not in store.read_json("run_summary.json")
    monkeypatch.setattr(runtime, "execute_pptx_preserve", original_execute)
    assert runtime.retry_page(run_dir, "page_001")["run"]["status"] == "prepared"
    assert runtime.run_job(run_dir)["status"] == "completed"


@pytest.mark.parametrize("invalid_sha256", [True, "A" * 64, "0" * 63])
def test_pptx_invalid_manifest_sha256_is_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_sha256: object,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    manifest["input"]["sha256"] = invalid_sha256
    store.write_json("job_manifest.json", manifest)
    before_state = store.read_json("run_state.json")

    def unexpected_execute(_store: RunStore) -> dict[str, object]:
        raise AssertionError("invalid manifest executed")

    monkeypatch.setattr(runtime, "execute_pptx_preserve", unexpected_execute)

    with pytest.raises(RuntimeError, match="manifest.*sha256"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_state
    assert not (run_dir / "final" / "output.pptx").exists()


def test_pptx_executor_cannot_claim_another_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    user_output = tmp_path / "user-owned.pptx"
    _pptx(source, slide_count=1)
    user_output.write_bytes(b"user")
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    status = user_output.lstat()

    def claim_user_output(_store: RunStore) -> dict[str, object]:
        digest = runtime.sha256_file(user_output)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "pages": 1,
            "outputs": {"pptx": str(user_output)},
            "output_sha256": digest,
            "_output_identity": {
                "version": 1,
                "path": str(user_output),
                "dev": status.st_dev,
                "ino": status.st_ino,
                "mode": status.st_mode,
                "size": status.st_size,
                "mtime_ns": status.st_mtime_ns,
                "sha256": digest,
            },
        }

    monkeypatch.setattr(runtime, "execute_pptx_preserve", claim_user_output)

    with pytest.raises(RuntimeError, match="expected output path"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert user_output.read_bytes() == b"user"
    assert store.read_json("run_summary.json")["retry_blocked"] is True
    with pytest.raises(RuntimeError, match="blocked"):
        runtime.retry_page(run_dir, "page_001")
    assert user_output.read_bytes() == b"user"


def test_pptx_executor_cannot_claim_preexisting_expected_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    output = tmp_path / "expected.pptx"
    _pptx(source, slide_count=1)
    output.write_bytes(b"user")
    run_dir = runtime.prepare_job(
        source, run_dir=tmp_path / "run", output_path=output
    )
    status = output.lstat()

    def claim_preexisting(_store: RunStore) -> dict[str, object]:
        digest = runtime.sha256_file(output)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "pages": 1,
            "outputs": {"pptx": str(output)},
            "output_sha256": digest,
            "_output_identity": {
                "version": 1,
                "path": str(output),
                "dev": status.st_dev,
                "ino": status.st_ino,
                "mode": status.st_mode,
                "size": status.st_size,
                "mtime_ns": status.st_mtime_ns,
                "sha256": digest,
            },
        }

    monkeypatch.setattr(runtime, "execute_pptx_preserve", claim_preexisting)

    with pytest.raises(RuntimeError, match="already existed"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert output.read_bytes() == b"user"
    assert store.read_json("run_summary.json")["retry_blocked"] is True


def test_pptx_same_bytes_replacement_before_return_is_not_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    replacement = tmp_path / "replacement.pptx"
    original_execute = runtime.execute_pptx_preserve

    def replace_before_return(store: RunStore) -> dict[str, object]:
        summary = original_execute(store)
        replacement.write_bytes(output.read_bytes())
        runtime.os.replace(replacement, output)
        return summary

    monkeypatch.setattr(
        runtime, "execute_pptx_preserve", replace_before_return
    )

    with pytest.raises(RuntimeError, match="identity token"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert output.read_bytes() == source.read_bytes()
    assert store.read_json("run_summary.json")["retry_blocked"] is True
    with pytest.raises(RuntimeError, match="blocked"):
        runtime.retry_page(run_dir, "page_001")
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    "mutation",
    [
        "path",
        "identity",
        "hash",
        "missing",
        "absent",
        "malformed",
        "version_bool",
        "version_float",
        "future",
    ],
)
def test_pptx_forged_or_unknown_identity_token_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_execute = runtime.execute_pptx_preserve

    def forge_token(store: RunStore) -> dict[str, object]:
        summary = original_execute(store)
        token = dict(summary["_output_identity"])
        if mutation == "path":
            token["path"] = str(tmp_path / "other.pptx")
        elif mutation == "identity":
            token["ino"] += 1
        elif mutation == "hash":
            token["sha256"] = "0" * 64
        elif mutation == "missing":
            token.pop("ino")
        elif mutation == "absent":
            summary.pop("_output_identity")
            return summary
        elif mutation == "malformed":
            summary["_output_identity"] = ["not", "an", "object"]
            return summary
        elif mutation == "version_bool":
            token["version"] = True
        elif mutation == "version_float":
            token["version"] = 1.0
        else:
            token["future"] = True
        summary["_output_identity"] = token
        return summary

    monkeypatch.setattr(runtime, "execute_pptx_preserve", forge_token)

    with pytest.raises(RuntimeError, match="identity token"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert output.is_file()
    assert store.read_json("run_summary.json")["retry_blocked"] is True


def test_pptx_execute_post_publish_error_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_execute = runtime.execute_pptx_preserve

    def publish_then_fail(store: RunStore) -> dict[str, object]:
        original_execute(store)
        raise OSError("post-publish cleanup failed")

    monkeypatch.setattr(
        runtime, "execute_pptx_preserve", publish_then_fail
    )

    with pytest.raises(OSError, match="post-publish"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert output.is_file()
    assert store.read_json("run_state.json")["status"] == "failed"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == "preserved"
    assert store.read_json("run_summary.json")["retry_blocked"] is True
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    with pytest.raises(RuntimeError, match="blocked"):
        runtime.retry_page(run_dir, "page_001")
    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages
    assert output.is_file()


def test_pptx_compensation_failure_blocks_retry_before_pages_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_transition_pages = runtime._transition_pages
    original_replace = runtime.os.replace
    replacement_injected = False

    def fail_preserved_transition(
        store: RunStore, page_ids: list[str], target: Any
    ) -> None:
        if target.value == "preserved":
            raise OSError("preserved page write failed")
        original_transition_pages(store, page_ids, target)

    def replace_before_isolation(source_path, destination_path):
        nonlocal replacement_injected
        if Path(source_path) == output and not replacement_injected:
            replacement_injected = True
            output.unlink()
            output.write_bytes(b"concurrent replacement")
        return original_replace(source_path, destination_path)

    monkeypatch.setattr(
        runtime, "_transition_pages", fail_preserved_transition
    )
    monkeypatch.setattr(runtime.os, "replace", replace_before_isolation)

    with pytest.raises(OSError, match="preserved page") as error:
        runtime.run_job(run_dir)

    assert error.value.__cause__ is not None
    assert output.read_bytes() == b"concurrent replacement"
    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "failed"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == "failed"
    assert store.read_json("run_summary.json")["retry_blocked"] is True
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    with pytest.raises(RuntimeError, match="blocked"):
        runtime.retry_page(run_dir, "page_001")
    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages


def test_pptx_compensation_recovers_one_shot_page_snapshot_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_transition_run = RunStore.transition_run
    original_write_json = RunStore.write_json
    finalizing_failed = False
    snapshot_failed = False

    def fail_finalizing_once(
        self: RunStore, target: RunStatus
    ) -> dict[str, Any]:
        nonlocal finalizing_failed
        if target is RunStatus.FINALIZING and not finalizing_failed:
            finalizing_failed = True
            raise OSError("finalizing state write failed")
        return original_transition_run(self, target)

    def fail_snapshot_once(
        self: RunStore, relative: str | Path, document: dict[str, Any]
    ) -> None:
        nonlocal snapshot_failed
        if (
            Path(relative) == Path("page_jobs.json")
            and {
                page["status"] for page in document["pages"].values()
            }
            == {"analyzed"}
            and not snapshot_failed
        ):
            snapshot_failed = True
            raise OSError("page snapshot write failed")
        original_write_json(self, relative, document)

    monkeypatch.setattr(RunStore, "transition_run", fail_finalizing_once)
    monkeypatch.setattr(RunStore, "write_json", fail_snapshot_once)

    with pytest.raises(OSError, match="finalizing") as error:
        runtime.run_job(run_dir)

    assert isinstance(error.value.__cause__, OSError)
    assert str(error.value.__cause__) == "page snapshot write failed"
    store = RunStore.open(run_dir)
    assert not output.exists()
    assert store.read_json("run_state.json")["status"] == "failed"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == "failed"
    assert store.read_json("run_summary.json")["status"] == "failed"

    retried = runtime.retry_page(run_dir, "page_001")
    assert retried["run"]["status"] == "prepared"
    assert retried["pages"]["pages"]["page_001"]["status"] == "analyzed"
    assert runtime.run_job(run_dir)["status"] == "completed"


def test_pptx_compensation_recovers_completed_post_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    output = run_dir / "final" / "output.pptx"
    original_transition_run = RunStore.transition_run
    completed_failed = False

    def persist_completed_then_fail(
        self: RunStore, target: RunStatus
    ) -> dict[str, Any]:
        nonlocal completed_failed
        result = original_transition_run(self, target)
        if target is RunStatus.COMPLETED and not completed_failed:
            completed_failed = True
            raise OSError("completed state post-write failure")
        return result

    monkeypatch.setattr(
        RunStore, "transition_run", persist_completed_then_fail
    )

    with pytest.raises(OSError, match="post-write"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert not output.exists()
    assert store.read_json("run_state.json")["status"] == "failed"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == "failed"
    assert store.read_json("run_summary.json")["status"] == "failed"

    retried = runtime.retry_page(run_dir, "page_001")
    assert retried["run"]["status"] == "prepared"
    assert retried["pages"]["pages"]["page_001"]["status"] == "analyzed"
    assert runtime.run_job(run_dir)["status"] == "completed"


@pytest.mark.parametrize("mutation", ["manifest_pages", "page_jobs", "status"])
def test_pptx_run_rejects_inconsistent_pages_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source)
    run_dir = prepare_pptx_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    if mutation == "manifest_pages":
        manifest = store.read_json("job_manifest.json")
        manifest["pages"] = ["page_001", "page_003"]
        store.write_json("job_manifest.json", manifest)
    else:
        page_jobs = store.read_json("page_jobs.json")
        if mutation == "page_jobs":
            page_jobs["pages"]["page_003"] = page_jobs["pages"].pop("page_002")
        else:
            page_jobs["pages"]["page_001"]["status"] = "pending"
        store.write_json("page_jobs.json", page_jobs)
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")

    def unexpected_execute(store: RunStore) -> dict[str, object]:
        raise AssertionError("inconsistent PPTX run executed")

    monkeypatch.setattr(runtime, "execute_pptx_preserve", unexpected_execute)

    with pytest.raises(RuntimeError, match="PPTX"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages
    assert not (run_dir / "final" / "output.pptx").exists()


def test_pptx_page_validation_accepts_sorted_json_order_for_1000_pages() -> None:
    page_ids = [f"page_{index:03d}" for index in range(1, 1001)]
    page_jobs = {
        "pages": {
            page_id: {"status": "analyzed"}
            for page_id in sorted(page_ids)
        }
    }

    assert runtime._pptx_page_ids(
        {"pages": page_ids},
        page_jobs,
        runtime.PageStatus.ANALYZED,
    ) == page_ids


def test_retry_pptx_run_restores_analyzed_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    original_execute = runtime.execute_pptx_preserve
    calls = 0

    def fail_once(store: RunStore) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("preserve failed")
        return original_execute(store)

    monkeypatch.setattr(runtime, "execute_pptx_preserve", fail_once)
    with pytest.raises(RuntimeError, match="preserve failed"):
        runtime.run_job(run_dir)
    work_root = run_dir / "work"
    work_root.mkdir()
    (work_root / "legacy.txt").write_text("stale", encoding="utf-8")

    status = runtime.retry_page(run_dir, "page_001")

    assert status["run"]["status"] == "prepared"
    assert status["pages"]["pages"]["page_001"]["status"] == "analyzed"
    assert not work_root.exists()
    assert runtime.run_job(run_dir)["status"] == "completed"
    assert calls == 2


def test_run_job_rejects_unknown_input_type_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    manifest["input"]["type"] = "unknown"
    store.write_json("job_manifest.json", manifest)

    def unexpected_legacy(store: RunStore) -> dict[str, Any]:
        raise AssertionError("unknown input entered legacy execution")

    monkeypatch.setattr(runtime, "execute_legacy", unexpected_legacy)

    with pytest.raises(RuntimeError, match="Unsupported input type"):
        runtime.run_job(run_dir)

    assert store.read_json("run_state.json")["status"] == "prepared"


def test_run_job_completes_and_writes_summary_and_page_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first)
    _image(second, (4, 5, 6))
    run_dir = runtime.prepare_job([first, second], run_dir=tmp_path / "run")
    outputs = {
        "16:9": str((tmp_path / "wide.pptx").resolve()),
        "original": [str((tmp_path / "first.pptx").resolve())],
    }
    monkeypatch.setattr(runtime, "execute_legacy", lambda store: outputs)

    summary = runtime.run_job(run_dir)
    store = RunStore.open(run_dir)

    assert summary == {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "pages": 2,
        "outputs": outputs,
        "resource_policy": safe_default_policy(),
    }
    assert store.read_json("run_summary.json") == summary
    assert store.read_json("run_state.json")["status"] == "completed"
    for page_id in ("page_001", "page_002"):
        assert store.read_json(f"pages/{page_id}/page_result.json") == {
            "schema_version": SCHEMA_VERSION,
            "page_id": page_id,
            "status": "validated",
            "outputs": outputs,
        }
        assert (
            store.read_json("page_jobs.json")["pages"][page_id]["status"]
            == "validated"
        )


def test_run_job_records_execution_failure_for_run_and_all_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first)
    _image(second)
    run_dir = runtime.prepare_job([first, second], run_dir=tmp_path / "run")

    def fail_execute(store: RunStore) -> dict[str, Any]:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(runtime, "execute_legacy", fail_execute)

    with pytest.raises(RuntimeError, match="model unavailable"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "failed"
    assert {
        page["status"]
        for page in store.read_json("page_jobs.json")["pages"].values()
    } == {"failed"}
    assert store.read_json("run_summary.json") == {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": {"type": "RuntimeError", "message": "model unavailable"},
        "outputs": {},
    }


def test_page_result_write_failure_records_failed_run_and_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first)
    _image(second)
    run_dir = runtime.prepare_job([first, second], run_dir=tmp_path / "run")
    monkeypatch.setattr(
        runtime,
        "execute_legacy",
        lambda store: {"16:9": str(tmp_path / "output.pptx")},
    )
    original_write_json = RunStore.write_json

    def fail_second_page_result(
        self: RunStore, relative: str | Path, document: dict[str, Any]
    ) -> None:
        if Path(relative) == Path("pages/page_002/page_result.json"):
            raise OSError("page result write failed")
        original_write_json(self, relative, document)

    monkeypatch.setattr(RunStore, "write_json", fail_second_page_result)

    with pytest.raises(OSError, match="page result write failed"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "failed"
    assert {
        page["status"]
        for page in store.read_json("page_jobs.json")["pages"].values()
    } == {"failed"}
    assert store.read_json("run_summary.json") == {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": {"type": "OSError", "message": "page result write failed"},
        "outputs": {},
    }


def test_completed_transition_failure_can_retry_the_entire_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    monkeypatch.setattr(
        runtime,
        "execute_legacy",
        lambda store: {"16:9": str(tmp_path / "output.pptx")},
    )
    original_transition_run = RunStore.transition_run

    def fail_completed(self: RunStore, target: RunStatus) -> dict[str, Any]:
        if target is RunStatus.COMPLETED:
            raise OSError("completed state write failed")
        return original_transition_run(self, target)

    monkeypatch.setattr(RunStore, "transition_run", fail_completed)

    with pytest.raises(OSError, match="completed state write failed"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "failed"
    assert (
        store.read_json("page_jobs.json")["pages"]["page_001"]["status"]
        == "failed"
    )
    assert store.read_json("run_summary.json") == {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": {"type": "OSError", "message": "completed state write failed"},
        "outputs": {},
    }

    retried = runtime.retry_page(run_dir, "page_001")
    assert retried["run"]["status"] == "prepared"
    assert retried["pages"]["pages"]["page_001"]["status"] == "pending"

    monkeypatch.setattr(RunStore, "transition_run", original_transition_run)
    completed = runtime.run_job(run_dir)

    assert completed["status"] == "completed"
    assert runtime.get_status(run_dir)["run"]["status"] == "completed"


def test_success_validates_pages_with_one_page_jobs_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first)
    _image(second)
    run_dir = runtime.prepare_job([first, second], run_dir=tmp_path / "run")
    monkeypatch.setattr(runtime, "execute_legacy", lambda store: {})
    original_write_json = RunStore.write_json
    validated_writes = 0

    def count_validated_write(
        self: RunStore, relative: str | Path, document: dict[str, Any]
    ) -> None:
        nonlocal validated_writes
        if (
            Path(relative) == Path("page_jobs.json")
            and "validated"
            in {
                page["status"] for page in document["pages"].values()
            }
        ):
            validated_writes += 1
        original_write_json(self, relative, document)

    monkeypatch.setattr(RunStore, "write_json", count_validated_write)

    runtime.run_job(run_dir)

    assert validated_writes == 1


def test_cleanup_error_is_cause_and_does_not_stop_later_failure_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    original_transition_pages = runtime._transition_pages

    def fail_execute(store: RunStore) -> dict[str, Any]:
        raise RuntimeError("execution failed")

    def fail_failed_pages(
        store: RunStore, page_ids: list[str], target: Any
    ) -> None:
        if target.value == "failed":
            raise OSError("page cleanup failed")
        original_transition_pages(store, page_ids, target)

    monkeypatch.setattr(runtime, "execute_legacy", fail_execute)
    monkeypatch.setattr(runtime, "_transition_pages", fail_failed_pages)

    with pytest.raises(RuntimeError, match="execution failed") as error:
        runtime.run_job(run_dir)

    assert isinstance(error.value.__cause__, OSError)
    assert str(error.value.__cause__) == "page cleanup failed"
    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "failed"
    assert store.read_json("run_summary.json")["status"] == "failed"
    assert (
        store.read_json("page_jobs.json")["pages"]["page_001"]["status"]
        == "processing"
    )

    monkeypatch.setattr(runtime, "_transition_pages", original_transition_pages)
    retried = runtime.retry_page(run_dir, "page_001")

    assert retried["run"]["status"] == "prepared"
    assert retried["pages"]["pages"]["page_001"]["status"] == "pending"


def test_retry_recovers_failed_batch_left_running_by_one_run_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    original_transition_run = RunStore.transition_run
    failed_transition = False
    execute_calls = 0

    def fail_execute_once(store: RunStore) -> dict[str, Any]:
        nonlocal execute_calls
        execute_calls += 1
        if execute_calls == 1:
            raise RuntimeError("execution failed")
        return {}

    def fail_run_failed_once(
        self: RunStore, target: RunStatus
    ) -> dict[str, Any]:
        nonlocal failed_transition
        if target is RunStatus.FAILED and not failed_transition:
            failed_transition = True
            raise OSError("run failed write failed")
        return original_transition_run(self, target)

    monkeypatch.setattr(runtime, "execute_legacy", fail_execute_once)
    monkeypatch.setattr(RunStore, "transition_run", fail_run_failed_once)

    with pytest.raises(RuntimeError, match="execution failed") as error:
        runtime.run_job(run_dir)

    assert isinstance(error.value.__cause__, OSError)
    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "running"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == "failed"
    assert store.read_json("run_summary.json")["status"] == "failed"

    retried = runtime.retry_page(run_dir, "page_001")
    assert retried["run"]["status"] == "prepared"
    assert retried["pages"]["pages"]["page_001"]["status"] == "pending"

    completed = runtime.run_job(run_dir)
    assert completed["status"] == "completed"
    assert execute_calls == 2


def test_retry_page_resets_the_entire_failed_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first)
    _image(second)
    run_dir = runtime.prepare_job([first, second], run_dir=tmp_path / "run")

    def fail_execute(store: RunStore) -> dict[str, Any]:
        raise RuntimeError("failed")

    monkeypatch.setattr(runtime, "execute_legacy", fail_execute)
    with pytest.raises(RuntimeError, match="failed"):
        runtime.run_job(run_dir)

    status = runtime.retry_page(run_dir, "page_001")

    assert status["run"]["status"] == "prepared"
    assert {
        page["status"] for page in status["pages"]["pages"].values()
    } == {"pending"}


def test_retry_page_removes_work_before_resetting_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    monkeypatch.setattr(
        runtime,
        "execute_legacy",
        lambda store: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError, match="failed"):
        runtime.run_job(run_dir)
    work_root = run_dir / "work"
    work_root.mkdir()
    (work_root / "diagnostic.txt").write_text("keep until retry", encoding="utf-8")

    status = runtime.retry_page(run_dir, "page_001")

    assert not work_root.exists()
    assert status["run"]["status"] == "prepared"
    assert status["pages"]["pages"]["page_001"]["status"] == "pending"


def test_retry_work_cleanup_failure_preserves_all_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    monkeypatch.setattr(
        runtime,
        "execute_legacy",
        lambda store: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError, match="failed"):
        runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    work_root = run_dir / "work"
    work_root.mkdir()
    (work_root / "diagnostic.txt").write_text("keep", encoding="utf-8")
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    before_summary = store.read_json("run_summary.json")

    def fail_cleanup(path: Path, expected_identity: tuple[int, int]) -> None:
        assert path == work_root.resolve()
        raise OSError("work cleanup failed")

    monkeypatch.setattr(runtime, "_safe_rmtree", fail_cleanup)

    with pytest.raises(OSError, match="work cleanup failed"):
        runtime.retry_page(run_dir, "page_001")

    assert work_root.is_dir()
    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages
    assert store.read_json("run_summary.json") == before_summary


def test_retry_rejects_work_symlink_without_deleting_external_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    monkeypatch.setattr(
        runtime,
        "execute_legacy",
        lambda store: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError, match="failed"):
        runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("outside", encoding="utf-8")
    work_root = run_dir / "work"
    try:
        work_root.symlink_to(external, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Cannot create symlink: {error}")
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    before_summary = store.read_json("run_summary.json")

    with pytest.raises(RuntimeError, match="work"):
        runtime.retry_page(run_dir, "page_001")

    assert sentinel.read_text(encoding="utf-8") == "outside"
    assert work_root.is_symlink()
    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages
    assert store.read_json("run_summary.json") == before_summary


@pytest.mark.parametrize("module", [legacy, runtime])
def test_work_safety_detects_windows_reparse_attribute(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reparse_flag = 0x400
    monkeypatch.setattr(
        module.stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        reparse_flag,
        raising=False,
    )
    directory_mode = module.stat.S_IFDIR

    assert module._is_link_or_reparse(
        types.SimpleNamespace(
            st_mode=directory_mode,
            st_file_attributes=reparse_flag,
        )
    )
    assert not module._is_link_or_reparse(
        types.SimpleNamespace(
            st_mode=directory_mode,
            st_file_attributes=0,
        )
    )


def test_retry_page_writes_page_jobs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first)
    _image(second)
    run_dir = runtime.prepare_job([first, second], run_dir=tmp_path / "run")

    def fail_execute(store: RunStore) -> dict[str, Any]:
        raise RuntimeError("failed")

    monkeypatch.setattr(runtime, "execute_legacy", fail_execute)
    with pytest.raises(RuntimeError, match="failed"):
        runtime.run_job(run_dir)

    original_write_json = RunStore.write_json
    page_jobs_writes = 0

    def count_page_jobs_write(
        self: RunStore, relative: str | Path, document: dict[str, Any]
    ) -> None:
        nonlocal page_jobs_writes
        if Path(relative) == Path("page_jobs.json"):
            page_jobs_writes += 1
        original_write_json(self, relative, document)

    monkeypatch.setattr(RunStore, "write_json", count_page_jobs_write)

    runtime.retry_page(run_dir, "page_001")

    assert page_jobs_writes == 1


def test_retry_page_write_failure_preserves_page_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first)
    _image(second)
    run_dir = runtime.prepare_job([first, second], run_dir=tmp_path / "run")

    def fail_execute(store: RunStore) -> dict[str, Any]:
        raise RuntimeError("failed")

    monkeypatch.setattr(runtime, "execute_legacy", fail_execute)
    with pytest.raises(RuntimeError, match="failed"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    original_page_jobs = store.read_json("page_jobs.json")
    original_write_json = RunStore.write_json
    page_jobs_writes = 0

    def fail_page_jobs_write(
        self: RunStore, relative: str | Path, document: dict[str, Any]
    ) -> None:
        nonlocal page_jobs_writes
        if Path(relative) == Path("page_jobs.json"):
            page_jobs_writes += 1
            raise OSError("page jobs write failed")
        original_write_json(self, relative, document)

    monkeypatch.setattr(RunStore, "write_json", fail_page_jobs_write)

    with pytest.raises(OSError, match="page jobs write failed"):
        runtime.retry_page(run_dir, "page_001")

    assert page_jobs_writes == 1
    assert store.read_json("page_jobs.json") == original_page_jobs
    assert store.read_json("run_state.json")["status"] == "failed"

    monkeypatch.setattr(RunStore, "write_json", original_write_json)

    retried = runtime.retry_page(run_dir, "page_001")
    repeated = runtime.retry_page(run_dir, "page_001")

    assert retried["run"]["status"] == "prepared"
    assert {
        page["status"] for page in retried["pages"]["pages"].values()
    } == {"pending"}
    assert repeated == retried


def test_pptx_retry_writes_pages_before_run_and_recovers_run_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    monkeypatch.setattr(
        runtime,
        "execute_pptx_preserve",
        lambda store: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError, match="failed"):
        runtime.run_job(run_dir)

    original_transition_run = RunStore.transition_run
    failed = False

    def fail_prepared_once(
        self: RunStore, target: RunStatus
    ) -> dict[str, Any]:
        nonlocal failed
        if target is RunStatus.PREPARED and not failed:
            failed = True
            raise OSError("prepared state write failed")
        return original_transition_run(self, target)

    monkeypatch.setattr(RunStore, "transition_run", fail_prepared_once)

    with pytest.raises(OSError, match="prepared state write failed"):
        runtime.retry_page(run_dir, "page_001")

    store = RunStore.open(run_dir)
    assert store.read_json("run_state.json")["status"] == "failed"
    assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == "analyzed"
    retried = runtime.retry_page(run_dir, "page_001")
    assert retried["run"]["status"] == "prepared"
    assert retried["pages"]["pages"]["page_001"]["status"] == "analyzed"


def test_pptx_retry_removes_stale_reconstruction_donor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    monkeypatch.setattr(
        runtime,
        "execute_pptx_preserve",
        lambda store: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError, match="failed"):
        runtime.run_job(run_dir)
    reconstruction = run_dir / "pages/page_001/reconstruction"
    reconstruction.mkdir()
    (reconstruction / "donor.pptx").write_bytes(b"stale")

    runtime.retry_page(run_dir, "page_001")

    assert not reconstruction.exists()


def test_pptx_recover_removes_stale_reconstruction_donor(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pptx"
    _pptx(source, slide_count=1)
    run_dir = runtime.prepare_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    store.transition_page("page_001", PageStatus.PROCESSING)
    reconstruction = run_dir / "pages/page_001/reconstruction"
    reconstruction.mkdir()
    (reconstruction / "donor.pptx").write_bytes(b"stale")

    runtime.recover_job(run_dir)

    assert not reconstruction.exists()


def test_retry_page_rejects_unknown_or_nonfailed_page(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")

    with pytest.raises(KeyError, match="Unknown page_id"):
        runtime.retry_page(run_dir, "missing")
    with pytest.raises(RuntimeError, match="not failed"):
        runtime.retry_page(run_dir, "page_001")


def test_run_job_rejects_non_prepared_run_without_changing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    store.transition_run(RunStatus.RUNNING)
    before_run = store.read_json("run_state.json")
    before_pages = store.read_json("page_jobs.json")
    called = False

    def fake_execute(store: RunStore) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(runtime, "execute_legacy", fake_execute)

    with pytest.raises(RuntimeError, match="must be prepared"):
        runtime.run_job(run_dir)

    assert not called
    assert store.read_json("run_state.json") == before_run
    assert store.read_json("page_jobs.json") == before_pages


def test_run_job_returns_existing_completed_summary(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    monkeypatch.setattr(runtime, "execute_legacy", lambda store: {})
    completed = runtime.run_job(run_dir)

    def unexpected_execute(store: RunStore) -> dict[str, Any]:
        raise AssertionError("completed run executed again")

    monkeypatch.setattr(runtime, "execute_legacy", unexpected_execute)

    assert runtime.run_job(run_dir) == completed


def test_run_job_validates_existing_completed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    monkeypatch.setattr(runtime, "execute_legacy", lambda store: {})
    runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    summary = store.read_json("run_summary.json")
    summary["schema_version"] = 2
    store.write_json("run_summary.json", summary)

    with pytest.raises(ValueError, match="Unsupported schema_version"):
        runtime.run_job(run_dir)


def test_retry_validates_existing_failed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")

    def fail(store: RunStore) -> dict[str, Any]:
        raise RuntimeError("failed")

    monkeypatch.setattr(runtime, "execute_legacy", fail)
    with pytest.raises(RuntimeError, match="failed"):
        runtime.run_job(run_dir)
    store = RunStore.open(run_dir)
    summary = store.read_json("run_summary.json")
    summary["schema_version"] = 2
    store.write_json("run_summary.json", summary)

    with pytest.raises(ValueError, match="Unsupported schema_version"):
        runtime.retry_page(run_dir, "page_001")


@pytest.mark.parametrize(
    ("image_count", "slide_size", "function_name", "expected_extra"),
    [
        (1, "both", "convert_variants", {}),
        (1, "original", "convert", {"slide_size": "original"}),
        (1, "16:9", "convert", {"slide_size": "16:9"}),
        (2, "both", "convert_batch_variants", {}),
        (
            2,
            "original",
            "convert_batch_variants",
            {"include_widescreen": False},
        ),
        (2, "16:9", "convert_batch", {}),
    ],
)
def test_execute_legacy_dispatches_with_real_signatures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    image_count: int,
    slide_size: str,
    function_name: str,
    expected_extra: dict[str, Any],
) -> None:
    sources = []
    for index in range(image_count):
        source = tmp_path / f"source-{index}.png"
        _image(source)
        sources.append(source)
    output_path = tmp_path / "chosen.pptx"
    run_dir = runtime.prepare_job(
        sources,
        run_dir=tmp_path / "run",
        output_path=output_path,
        slide_size=slide_size,
        lang="en",
    )
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def record(name: str):
        def converter(*args: Any, **kwargs: Any) -> Any:
            print("legacy progress")
            calls.append((name, args, kwargs))
            if name in {"convert_variants", "convert_batch_variants"}:
                return {"original": "original.pptx", "16:9": "wide.pptx"}
            return "single.pptx"

        return converter

    fake_module = types.SimpleNamespace(
        convert=record("convert"),
        convert_variants=record("convert_variants"),
        convert_batch=record("convert_batch"),
        convert_batch_variants=record("convert_batch_variants"),
    )
    monkeypatch.setattr(legacy.importlib, "import_module", lambda name: fake_module)

    result = legacy.execute_legacy(RunStore.open(run_dir))
    captured = capsys.readouterr()

    assert len(calls) == 1
    assert captured.out == ""
    assert "legacy progress" in captured.err
    name, args, kwargs = calls[0]
    assert name == function_name
    copied_sources = [
        (Path(run_dir) / "input" / f"{index:03d}_source-{index - 1}.png").resolve()
        for index in range(1, image_count + 1)
    ]
    assert args[0] == (copied_sources[0] if image_count == 1 else copied_sources)
    assert kwargs == {
        "output_path": str(output_path.resolve()),
        "lang": "en",
        "_work_root": (run_dir / "work").resolve(),
        "_resource_isolation": True,
        **expected_extra,
    }
    assert not (run_dir / "work").exists()
    assert set(result) == ({"original", "16:9"} if "variants" in name else {slide_size})


@pytest.mark.parametrize(
    ("input_document", "slide_size", "expected_name", "expected_extra"),
    [
        (
            {"type": "pdf", "page_ratios_equal": True, "page_aspect_ratio": 2.0},
            "both",
            "convert_batch_variants",
            {"combine_original": True, "original_aspect_ratio": 2.0},
        ),
        (
            {"type": "pdf", "page_ratios_equal": True, "page_aspect_ratio": 2.0},
            "original",
            "convert_batch_variants",
            {
                "include_widescreen": False,
                "combine_original": True,
                "original_aspect_ratio": 2.0,
            },
        ),
        (
            {"type": "pdf", "page_ratios_equal": True, "page_aspect_ratio": 2.0},
            "16:9",
            "convert_batch",
            {},
        ),
        (
            {"type": "pdf", "page_ratios_equal": False, "page_aspect_ratio": None},
            "both",
            "convert_batch_variants",
            {},
        ),
        (
            {"type": "pdf", "page_ratios_equal": True},
            "both",
            "convert_batch_variants",
            {"combine_original": True},
        ),
        ({"type": "images"}, "both", "convert_batch_variants", {}),
    ],
)
def test_execute_legacy_combines_only_equal_ratio_pdf_originals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_document: dict[str, Any],
    slide_size: str,
    expected_name: str,
    expected_extra: dict[str, Any],
) -> None:
    sources = [tmp_path / "first.png", tmp_path / "second.png"]
    for source in sources:
        _image(source)
    run_dir = runtime.prepare_job(
        sources,
        run_dir=tmp_path / "run",
        slide_size=slide_size,
    )
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    manifest["input"] = input_document
    store.write_json("job_manifest.json", manifest)
    calls: list[tuple[str, dict[str, Any]]] = []

    def record(name: str):
        def converter(*args: Any, **kwargs: Any) -> Any:
            calls.append((name, kwargs))
            return {"original": "original.pptx", "16:9": "wide.pptx"}

        return converter

    fake_module = types.SimpleNamespace(
        convert_batch=record("convert_batch"),
        convert_batch_variants=record("convert_batch_variants"),
    )
    monkeypatch.setattr(legacy.importlib, "import_module", lambda name: fake_module)

    legacy.execute_legacy(store)

    assert calls == [
        (
            expected_name,
            {
                    "output_path": str((run_dir / "final" / "output.pptx").resolve()),
                    "lang": "ch",
                    "_work_root": (run_dir / "work").resolve(),
                    "_resource_isolation": True,
                    **expected_extra,
            },
        )
    ]
    assert not (run_dir / "work").exists()


def test_execute_legacy_uses_default_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(
        [source], run_dir=tmp_path / "run", slide_size="16:9"
    )
    captured: dict[str, Any] = {}

    def convert_image(
        image_path: Path,
        output_path: str,
        lang: str,
        slide_size: str,
        _work_root: Path,
        _resource_isolation: bool,
    ) -> str:
        captured.update(
            output_path=output_path,
            lang=lang,
            slide_size=slide_size,
            _work_root=_work_root,
            _resource_isolation=_resource_isolation,
        )
        return output_path

    fake_module = types.SimpleNamespace(convert=convert_image)
    monkeypatch.setattr(legacy.importlib, "import_module", lambda name: fake_module)

    legacy.execute_legacy(RunStore.open(run_dir))

    assert captured == {
        "output_path": str((run_dir / "final" / "output.pptx").resolve()),
        "lang": "ch",
        "slide_size": "16:9",
        "_work_root": (run_dir / "work").resolve(),
        "_resource_isolation": True,
    }
    assert not (run_dir / "work").exists()


def test_execute_legacy_accepts_preexisting_empty_work_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(
        [source],
        run_dir=tmp_path / "run",
        slide_size="16:9",
    )
    work_root = run_dir / "work"
    work_root.mkdir()
    seen_roots = []

    def convert_image(*args: Any, _work_root: Path, **kwargs: Any) -> str:
        seen_roots.append(_work_root)
        return str(tmp_path / "output.pptx")

    monkeypatch.setattr(
        legacy.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(convert=convert_image),
    )

    legacy.execute_legacy(RunStore.open(run_dir))

    assert seen_roots == [work_root.resolve()]
    assert not work_root.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows directory handle semantics")
@pytest.mark.parametrize("replacement", ["root", "nested"])
def test_execute_legacy_cleanup_rejects_directory_replacement_during_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(
        [source],
        run_dir=tmp_path / "run",
        slide_size="16:9",
    )
    work_root = run_dir / "work"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("outside", encoding="utf-8")
    target = work_root if replacement == "root" else work_root / "nested"
    displaced = tmp_path / "displaced"
    converted = False

    def convert_image(*args: Any, _work_root: Path, **kwargs: Any) -> str:
        nonlocal converted
        target.mkdir(parents=True, exist_ok=True)
        (target / "owned.txt").write_text("owned", encoding="utf-8")
        converted = True
        return str(tmp_path / "output.pptx")

    monkeypatch.setattr(
        legacy.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(convert=convert_image),
    )
    original_entries = getattr(legacy, "_windows_entries", None)
    attempted = False

    def replace_directory_during_enumeration(
        kernel32: Any,
        handle: Any,
        path: Path,
        status: Any,
    ) -> Any:
        nonlocal attempted
        assert original_entries is not None
        entries = original_entries(kernel32, handle, path, status)
        if converted and not attempted and path == target:
            attempted = True
            target.rename(displaced)
            external.rename(target)
        return entries

    monkeypatch.setattr(
        legacy,
        "_windows_entries",
        replace_directory_during_enumeration,
        raising=False,
    )

    with pytest.raises(OSError):
        legacy.execute_legacy(RunStore.open(run_dir))

    assert attempted
    assert sentinel.read_text(encoding="utf-8") == "outside"
    assert external.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows directory handle semantics")
@pytest.mark.parametrize("entry_kind", ["directory", "file"])
def test_execute_legacy_cleanup_rejects_entry_replacement_after_parent_yield(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job(
        [source],
        run_dir=tmp_path / "run",
        slide_size="16:9",
    )
    work_root = run_dir / "work"
    target = work_root / ("nested" if entry_kind == "directory" else "owned.txt")
    displaced = tmp_path / "displaced"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("outside", encoding="utf-8")
    converted = False
    swapped = False

    def convert_image(*args: Any, _work_root: Path, **kwargs: Any) -> str:
        nonlocal converted
        work_root.mkdir(exist_ok=True)
        if entry_kind == "directory":
            target.mkdir()
            (target / "owned.txt").write_text("owned", encoding="utf-8")
        else:
            target.write_text("owned", encoding="utf-8")
        converted = True
        return str(tmp_path / "output.pptx")

    monkeypatch.setattr(
        legacy.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(convert=convert_image),
    )
    original_entries = getattr(legacy, "_windows_entries", None)

    def replace_after_parent_enumeration(
        kernel32: Any,
        handle: Any,
        path: Path,
        status: Any,
    ) -> Any:
        nonlocal swapped
        assert original_entries is not None
        entries = original_entries(kernel32, handle, path, status)
        if converted and not swapped and path == work_root:
            target.rename(displaced)
            if entry_kind == "directory":
                external.rename(target)
            else:
                sentinel.rename(target)
            swapped = True
        return entries

    monkeypatch.setattr(
        legacy,
        "_windows_entries",
        replace_after_parent_enumeration,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="changed"):
        legacy.execute_legacy(RunStore.open(run_dir))

    assert swapped
    if entry_kind == "directory":
        target.rename(external)
    else:
        target.rename(sentinel)
    assert sentinel.read_text(encoding="utf-8") == "outside"


@pytest.mark.parametrize("case", ["nonempty", "file"])
def test_execute_legacy_rejects_unsafe_existing_work_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    work_root = run_dir / "work"
    if case == "nonempty":
        work_root.mkdir()
        (work_root / "sentinel.txt").write_text("keep", encoding="utf-8")
    else:
        work_root.write_text("keep", encoding="utf-8")

    def unexpected_import(name: str) -> Any:
        raise AssertionError("legacy module imported before work validation")

    monkeypatch.setattr(legacy.importlib, "import_module", unexpected_import)

    with pytest.raises(RuntimeError, match="work"):
        legacy.execute_legacy(RunStore.open(run_dir))


def test_execute_legacy_rejects_work_symlink_before_import_without_external_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("outside", encoding="utf-8")
    work_root = run_dir / "work"
    try:
        work_root.symlink_to(external, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Cannot create symlink: {error}")

    def unexpected_import(name: str) -> Any:
        raise AssertionError("legacy module imported before work validation")

    monkeypatch.setattr(legacy.importlib, "import_module", unexpected_import)

    with pytest.raises(RuntimeError, match="work"):
        legacy.execute_legacy(RunStore.open(run_dir))

    assert sentinel.read_text(encoding="utf-8") == "outside"
    assert work_root.is_symlink()


def test_legacy_failure_retains_work_and_records_absolute_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    diagnostic_name = "failure.txt"

    def fail_conversion(*args: Any, _work_root: Path, **kwargs: Any) -> Any:
        page_root = _work_root / "page_001"
        page_root.mkdir()
        (page_root / diagnostic_name).write_text("details", encoding="utf-8")
        raise RuntimeError("conversion failed")

    fake_module = types.SimpleNamespace(convert_variants=fail_conversion)
    monkeypatch.setattr(
        legacy.importlib,
        "import_module",
        lambda name: fake_module,
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        runtime.run_job(run_dir)

    work_root = (run_dir / "work").resolve()
    assert (work_root / "page_001" / diagnostic_name).read_text(
        encoding="utf-8"
    ) == "details"
    assert RunStore.open(run_dir).read_json("run_summary.json") == {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": {"type": "RuntimeError", "message": "conversion failed"},
        "outputs": {},
        "diagnostics": str(work_root),
    }


def test_legacy_cleanup_failure_after_conversion_records_failed_run_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    diagnostic_name = "conversion.txt"
    converted = False

    def convert_image(*args: Any, _work_root: Path, **kwargs: Any) -> Any:
        nonlocal converted
        page_root = _work_root / "page_001"
        page_root.mkdir()
        (page_root / diagnostic_name).write_text("complete", encoding="utf-8")
        converted = True
        return {"16:9": str(tmp_path / "output.pptx")}

    def fail_cleanup(*args: Any, **kwargs: Any) -> None:
        assert converted
        raise OSError("work cleanup failed")

    monkeypatch.setattr(
        legacy.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(convert_variants=convert_image),
    )
    monkeypatch.setattr(legacy, "_safe_rmtree", fail_cleanup, raising=False)

    with pytest.raises(OSError, match="work cleanup failed"):
        runtime.run_job(run_dir)

    store = RunStore.open(run_dir)
    work_root = (run_dir / "work").resolve()
    assert store.read_json("run_state.json")["status"] == "failed"
    assert {
        page["status"]
        for page in store.read_json("page_jobs.json")["pages"].values()
    } == {"failed"}
    assert (work_root / "page_001" / diagnostic_name).read_text(
        encoding="utf-8"
    ) == "complete"
    assert store.read_json("run_summary.json") == {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": {"type": "OSError", "message": "work cleanup failed"},
        "outputs": {},
        "diagnostics": str(work_root),
    }


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("outside", "outside"),
        ("missing", "not a file"),
        ("directory", "not a file"),
        ("sha256", "sha256 mismatch"),
    ],
)
def test_execute_legacy_rejects_unsafe_or_changed_source_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    reason: str,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    request_path = Path("pages/page_001/page_request.json")
    request = store.read_json(request_path)
    if case == "outside":
        request["source"] = "../source.png"
    elif case == "missing":
        request["source"] = "input/missing.png"
    elif case == "directory":
        request["source"] = "input"
    else:
        request["sha256"] = "0" * 64
    store.write_json(request_path, request)

    def unexpected_import(name: str) -> Any:
        raise AssertionError("legacy module imported before source validation")

    monkeypatch.setattr(legacy.importlib, "import_module", unexpected_import)

    with pytest.raises(ValueError, match=rf"page_001.*{reason}"):
        legacy.execute_legacy(store)


@pytest.mark.parametrize("document_name", ["job_manifest", "page_request"])
def test_execute_legacy_validates_consumed_document_versions_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_name: str,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    run_dir = runtime.prepare_job([source], run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    relative = (
        Path("job_manifest.json")
        if document_name == "job_manifest"
        else Path("pages/page_001/page_request.json")
    )
    document = store.read_json(relative)
    document["schema_version"] = 2
    store.write_json(relative, document)

    def unexpected_import(name: str) -> Any:
        raise AssertionError("legacy module imported before schema validation")

    monkeypatch.setattr(legacy.importlib, "import_module", unexpected_import)

    with pytest.raises(ValueError, match="Unsupported schema_version"):
        legacy.execute_legacy(store)


def test_absolute_outputs_recurses_without_changing_other_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    value = {
        "single": "one.pptx",
        "nested": ["two.pptx", {"empty": None, "count": 2}],
    }

    assert legacy._absolute_outputs(value) == {
        "single": str((tmp_path / "one.pptx").resolve()),
        "nested": [
            str((tmp_path / "two.pptx").resolve()),
            {"empty": None, "count": 2},
        ],
    }


def test_convert_prepares_then_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []
    prepared = Path("prepared-run")

    def fake_prepare(inputs: list[str], **kwargs: Any) -> Path:
        calls.append(("prepare", (inputs, kwargs)))
        return prepared

    def fake_run(run_dir: Path) -> dict[str, Any]:
        calls.append(("run", run_dir))
        return {"status": "completed"}

    monkeypatch.setattr(runtime, "prepare_job", fake_prepare)
    monkeypatch.setattr(runtime, "run_job", fake_run)

    result = runtime.convert(
        ["source.png"],
        run_dir="run",
        output_path="output.pptx",
        slide_size="original",
        lang="en",
    )

    assert calls == [
        (
            "prepare",
            (
                ["source.png"],
                {
                    "run_dir": "run",
                    "output_path": "output.pptx",
                        "slide_size": "original",
                        "lang": "en",
                        "agent_provider": "host",
                },
            ),
        ),
        ("run", prepared),
    ]
    assert result == {"status": "completed"}


@pytest.mark.parametrize("as_string", [False, True])
def test_convert_accepts_one_path_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    as_string: bool,
) -> None:
    source = tmp_path / "source.png"
    _image(source)
    value = str(source) if as_string else source
    monkeypatch.setattr(runtime, "execute_legacy", lambda store: {})

    summary = runtime.convert(value, run_dir=tmp_path / "run")

    assert summary["status"] == "completed"
    assert summary["pages"] == 1
