from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image

from image2editable import runtime
from image2editable.inputs import prepare_image_job, resolve_image_inputs, sha256_file
from image2editable.resources import safe_default_policy
from image2editable.store import RunStore


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (12, 8), color).save(path)


def test_resolve_image_inputs_keeps_argument_order_and_sorts_directories(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "slides"
    folder.mkdir()
    _write_image(folder / "b.png", (0, 255, 0))
    _write_image(folder / "a.png", (255, 0, 0))
    final = tmp_path / "final.png"
    _write_image(final, (0, 0, 255))

    assert [path.name for path in resolve_image_inputs([folder, final])] == [
        "a.png",
        "b.png",
        "final.png",
    ]


@pytest.mark.parametrize("as_string", [False, True])
def test_resolve_image_inputs_accepts_one_path_directly(
    tmp_path: Path, as_string: bool
) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))
    value = str(source) if as_string else source

    assert resolve_image_inputs(value) == [source.resolve()]


def test_resolve_image_inputs_uses_name_as_casefold_sort_tiebreaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "slides"
    folder.mkdir()
    folded_first = folder / "ß.png"
    name_first = folder / "ss.png"
    _write_image(folded_first, (1, 2, 3))
    _write_image(name_first, (4, 5, 6))
    original_iterdir = type(folder).iterdir

    def reversed_casefold_tie(path: Path):
        if path == folder.resolve():
            return iter([folded_first, name_first])
        return original_iterdir(path)

    monkeypatch.setattr(type(folder), "iterdir", reversed_casefold_tie)

    assert [path.name for path in resolve_image_inputs([folder])] == [
        "ss.png",
        "ß.png",
    ]


def test_resolve_image_inputs_rejects_missing_explicit_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError) as error:
        resolve_image_inputs([missing])

    assert str(missing.resolve()) in str(error.value)


def test_resolve_image_inputs_rejects_unsupported_explicit_file(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"not an image")

    with pytest.raises(ValueError) as error:
        resolve_image_inputs([pdf])

    assert str(pdf.resolve()) in str(error.value)


def test_resolve_image_inputs_rejects_empty_directory(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()

    with pytest.raises(ValueError, match="No supported image inputs"):
        resolve_image_inputs([folder])


def test_resolve_image_inputs_skips_file_symlinks_in_directories(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "slides"
    folder.mkdir()
    regular = folder / "regular.png"
    outside = tmp_path / "outside.png"
    link = folder / "linked.png"
    _write_image(regular, (1, 2, 3))
    _write_image(outside, (4, 5, 6))
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Cannot create symlink: {error}")

    assert resolve_image_inputs([folder]) == [regular.resolve()]


def test_resolve_image_inputs_does_not_recurse_into_directories(tmp_path: Path) -> None:
    folder = tmp_path / "slides"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    direct = folder / "direct.png"
    _write_image(direct, (1, 2, 3))
    _write_image(nested / "nested.png", (4, 5, 6))

    assert resolve_image_inputs([folder]) == [direct.resolve()]


def test_prepare_image_job_copies_sources_and_writes_prepared_run(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))

    run_root = prepare_image_job(
        [source],
        run_dir=tmp_path / "run",
        output_path=tmp_path / "output.pptx",
        slide_size="16:9",
        lang="en",
    )
    copied = run_root / "input" / "001_source.png"
    manifest = json.loads((run_root / "job_manifest.json").read_text(encoding="utf-8"))
    request = json.loads(
        (run_root / "pages" / "page_001" / "page_request.json").read_text(
            encoding="utf-8"
        )
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    assert run_root.is_absolute()
    assert copied.read_bytes() == source.read_bytes()
    assert sha256_file(copied) == digest
    assert request == {
        "schema_version": 1,
        "page_id": "page_001",
        "source": "input/001_source.png",
        "sha256": digest,
    }
    assert manifest["input"] == {
        "type": "images",
        "items": [
            {
                "original_path": str(source.resolve()),
                "source": "input/001_source.png",
                "sha256": digest,
            }
        ],
    }
    assert manifest["output_format"] == "pptx"
    assert manifest["options"] == {
        "agent_provider": "host",
        "lang": "en",
        "slide_size": "16:9",
        "output_path": str((tmp_path / "output.pptx").resolve()),
        "resource_policy": safe_default_policy(),
    }
    assert manifest["pages"] == ["page_001"]
    assert RunStore.open(run_root).read_json("run_state.json")["status"] == "prepared"


@pytest.mark.parametrize("agent_provider", ["host", "local"])
def test_prepare_image_job_freezes_agent_provider(
    tmp_path: Path, agent_provider: str
) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))

    run_root = prepare_image_job(
        source,
        run_dir=tmp_path / agent_provider,
        agent_provider=agent_provider,
    )

    assert RunStore.open(run_root).read_json("job_manifest.json")["options"][
        "agent_provider"
    ] == agent_provider


@pytest.mark.parametrize("agent_provider", ["", "HOST", "remote", None])
def test_public_image_prepare_apis_reject_invalid_agent_provider(
    tmp_path: Path, agent_provider: object
) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))

    for prepare in (prepare_image_job, runtime.prepare_job):
        with pytest.raises(ValueError, match="agent_provider"):
            prepare(
                source,
                run_dir=tmp_path / f"run-{id(prepare)}",
                agent_provider=agent_provider,
            )


@pytest.mark.parametrize("as_string", [False, True])
def test_prepare_apis_accept_one_path_directly(
    tmp_path: Path, as_string: bool
) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))
    value = str(source) if as_string else source

    direct = prepare_image_job(value, run_dir=tmp_path / "direct")
    public = runtime.prepare_job(value, run_dir=tmp_path / "public")

    assert RunStore.open(direct).read_json("job_manifest.json")["pages"] == [
        "page_001"
    ]
    assert RunStore.open(public).read_json("job_manifest.json")["pages"] == [
        "page_001"
    ]


def test_prepare_image_job_rejects_invalid_slide_size(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))

    with pytest.raises(ValueError, match="slide_size"):
        prepare_image_job([source], run_dir=tmp_path / "run", slide_size="square")


def test_prepare_image_job_prefixes_duplicate_file_names(tmp_path: Path) -> None:
    first_folder = tmp_path / "first"
    second_folder = tmp_path / "second"
    first_folder.mkdir()
    second_folder.mkdir()
    first = first_folder / "slide.png"
    second = second_folder / "slide.png"
    _write_image(first, (1, 2, 3))
    _write_image(second, (4, 5, 6))

    run_root = prepare_image_job([first, second], run_dir=tmp_path / "run")

    assert (run_root / "input" / "001_slide.png").read_bytes() == first.read_bytes()
    assert (run_root / "input" / "002_slide.png").read_bytes() == second.read_bytes()


def test_prepare_image_job_does_not_overwrite_nonempty_run_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    keep = run_dir / "keep.txt"
    keep.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_image_job([source], run_dir=run_dir)

    assert keep.read_text(encoding="utf-8") == "user data"


@pytest.mark.parametrize("precreate_run_dir", [False, True])
def test_prepare_image_job_cleans_run_directory_after_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    precreate_run_dir: bool,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_image(first, (1, 2, 3))
    _write_image(second, (4, 5, 6))
    run_dir = tmp_path / "run"
    if precreate_run_dir:
        run_dir.mkdir()
    original_copy2 = shutil.copy2
    copy_count = 0

    def fail_second_copy(source: Path, destination: Path) -> Path:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            raise OSError("copy failed")
        return original_copy2(source, destination)

    monkeypatch.setattr("image2editable.inputs.shutil.copy2", fail_second_copy)

    with pytest.raises(OSError, match="copy failed"):
        prepare_image_job([first, second], run_dir=run_dir)

    assert run_dir.is_dir()
    assert not any(run_dir.iterdir())


def test_prepare_image_job_uses_cwd_runs_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))
    monkeypatch.chdir(tmp_path)

    run_root = prepare_image_job([source])

    assert run_root.parent == (tmp_path / "runs").resolve()


def test_prepare_image_job_records_utc_timestamp_and_uuid_job_id(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))

    run_root = prepare_image_job([source], run_dir=tmp_path / "run")
    job_id = RunStore.open(run_root).read_json("job_manifest.json")["job_id"]

    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", job_id)


def test_prepare_rejects_empty_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="output"):
        prepare_image_job([source], run_dir=tmp_path / "run", output_path="")


@pytest.mark.parametrize(
    "relative_output",
    [
        "run_state.json",
        "input/output.pptx",
        "pages/page_001/output.pptx",
        "job_manifest.pptx",
    ],
)
def test_prepare_rejects_output_in_run_internals(
    tmp_path: Path, relative_output: str
) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))
    run_dir = tmp_path / "run"
    output = run_dir / relative_output

    with pytest.raises(ValueError, match="output") as error:
        prepare_image_job([source], run_dir=run_dir, output_path=output)

    assert str(output.resolve()) in str(error.value)


def test_prepare_allows_output_under_run_final(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))
    run_dir = tmp_path / "run"
    output = run_dir / "final" / "custom.pptx"

    prepared = prepare_image_job(
        source, run_dir=run_dir, output_path=output, slide_size="both"
    )

    manifest = RunStore.open(prepared).read_json("job_manifest.json")
    assert manifest["options"]["output_path"] == str(output.resolve())


def test_prepare_rejects_non_pptx_output(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))
    output = tmp_path / "output.pdf"

    with pytest.raises(ValueError, match="output") as error:
        prepare_image_job([source], run_dir=tmp_path / "run", output_path=output)

    assert str(output.resolve()) in str(error.value)


def test_prepare_rejects_existing_directory_as_output(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, (1, 2, 3))
    output = tmp_path / "output.pptx"
    output.mkdir()

    with pytest.raises(ValueError, match="output") as error:
        prepare_image_job([source], run_dir=tmp_path / "run", output_path=output)

    assert str(output.resolve()) in str(error.value)


def test_prepare_rejects_output_that_overwrites_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pptx"
    source.write_bytes(b"source")
    monkeypatch.setattr(
        "image2editable.inputs.resolve_image_inputs",
        lambda inputs: [source.resolve()],
    )

    with pytest.raises(ValueError, match="output") as error:
        prepare_image_job(
            source,
            run_dir=tmp_path / "run",
            output_path=source,
        )

    assert str(source.resolve()) in str(error.value)
