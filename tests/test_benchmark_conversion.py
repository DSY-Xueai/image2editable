import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.benchmark_conversion as benchmark


CASE_SPECS = tuple(
    (f"image_{index}", "image", f"image-{index}.png", 1) for index in range(8)
) + (
    ("document", "pdf", "document.pdf", 3),
    ("mixed", "pptx", "mixed.pptx", 3),
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_manifest(root: Path) -> dict[str, object]:
    cases = []
    for identifier, kind, filename, pages in CASE_SPECS:
        payload = f"asset:{identifier}".encode()
        (root / filename).write_bytes(payload)
        cases.append(
            {
                "id": identifier,
                "kind": kind,
                "path": filename,
                "pages": pages,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    routes = [
        {"id": "images", "cases": [case[0] for case in CASE_SPECS[:8]], "pages": 8},
        {"id": "pdf", "cases": ["document"], "pages": 3},
        {"id": "mixed_pptx", "cases": ["mixed"], "pages": 3},
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "cases": cases,
        "routes": routes,
    }
    manifest["corpus_sha256"] = _canonical_sha256(manifest)
    return manifest


def _write_manifest(
    root: Path,
    manifest: dict[str, object],
    *,
    refresh_corpus_sha: bool = True,
) -> Path:
    if refresh_corpus_sha:
        manifest["corpus_sha256"] = _canonical_sha256(
            {
                "schema_version": manifest.get("schema_version"),
                "cases": manifest.get("cases"),
                "routes": manifest.get("routes"),
            }
        )
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _assert_invalid(path: Path) -> None:
    with pytest.raises(benchmark.BenchmarkError) as raised:
        benchmark.load_manifest(path)
    assert str(raised.value) == "invalid_corpus"


def test_module_exposes_benchmark_error() -> None:
    assert issubclass(benchmark.BenchmarkError, RuntimeError)


def test_loads_tracked_manifest_into_three_ordered_routes() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmark" / "corpus"

    manifest = benchmark.load_manifest(root / "manifest.json")

    assert manifest.root == root.resolve()
    assert manifest.corpus_sha256 == "5aec7fb2cf751e42fe0c1c51c49bac613af85fae1727ba66de517a7a0c1718a0"
    assert [route.identifier for route in manifest.routes] == [
        "images",
        "pdf",
        "mixed_pptx",
    ]
    assert [route.expected_pages for route in manifest.routes] == [8, 3, 3]
    assert sum(route.expected_pages for route in manifest.routes) == 14
    assert [path.name for path in manifest.routes[0].sources] == [
        "01-zh-courseware.png",
        "02-typography.png",
        "03-flowchart.png",
        "04-table-chart.png",
        "05-photo-overlay.png",
        "06-transparency-shadow.png",
        "07-compressed.jpg",
        "08-portrait.png",
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.update(extra=True),
        lambda manifest: manifest.update(schema_version=2),
        lambda manifest: manifest.update(schema_version=True),
        lambda manifest: manifest["cases"][0].update(extra=True),
        lambda manifest: manifest["cases"][0].update(id=""),
        lambda manifest: manifest["cases"][0].update(kind="video"),
        lambda manifest: manifest["cases"][0].update(pages=True),
        lambda manifest: manifest["cases"][0].update(pages=0),
        lambda manifest: manifest["cases"][0].update(bytes=True),
        lambda manifest: manifest["cases"][0].update(bytes=0),
        lambda manifest: manifest["cases"][0].update(sha256="A" * 64),
    ],
)
def test_rejects_invalid_exact_schema_and_strict_types(
    tmp_path: Path, mutation: object
) -> None:
    manifest = _valid_manifest(tmp_path)
    mutation(manifest)

    _assert_invalid(_write_manifest(tmp_path, manifest))


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../secret.png",
        "/absolute.png",
        "C:/absolute.png",
        "\\\\server\\share\\x.png",
        ".",
        "",
        "nested/file.png",
        "nested\\file.png",
    ],
)
def test_rejects_unsafe_asset_paths(tmp_path: Path, unsafe_path: str) -> None:
    manifest = _valid_manifest(tmp_path)
    manifest["cases"][0]["path"] = unsafe_path

    _assert_invalid(_write_manifest(tmp_path, manifest))


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "bad<name.png",
        "bad>name.png",
        'bad"name.png',
        "bad:name.png",
        "bad|name.png",
        "bad?name.png",
        "bad*name.png",
        "bad\x00name.png",
        "bad\x1fname.png",
        "bad\x7fname.png",
        "trailing.",
        "trailing ",
        "CON",
        "CONIN$",
        "CONOUT$",
        "CON .txt",
        "prn.txt",
        "AUX",
        "nul.png",
        "NUL .txt",
        "COM1",
        "COM1 .txt",
        "com9.log",
        "COM¹",
        "com².txt",
        "LPT1",
        "lpt9.txt",
        "LPT³.log",
    ],
)
def test_rejects_windows_unsafe_filename_before_reading_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_path: str,
) -> None:
    manifest = _valid_manifest(tmp_path)
    manifest["cases"][0]["path"] = unsafe_path
    manifest_path = _write_manifest(tmp_path, manifest)
    asset_reads: list[Path] = []
    original_read = benchmark._read_regular_file

    def tracked_read(path: Path, *args: object, **kwargs: object) -> bytes:
        if path != manifest_path:
            asset_reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(benchmark, "_read_regular_file", tracked_read)

    _assert_invalid(manifest_path)
    assert asset_reads == []


@pytest.mark.parametrize("field", ["id", "path"])
def test_rejects_duplicate_case_identity(tmp_path: Path, field: str) -> None:
    manifest = _valid_manifest(tmp_path)
    manifest["cases"][1][field] = manifest["cases"][0][field]

    _assert_invalid(_write_manifest(tmp_path, manifest))


def test_rejects_casefolded_duplicate_path_before_reading_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _valid_manifest(tmp_path)
    first, second = manifest["cases"][:2]
    second["path"] = first["path"].upper()
    second["bytes"] = first["bytes"]
    second["sha256"] = first["sha256"]
    manifest_path = _write_manifest(tmp_path, manifest)
    asset_reads: list[Path] = []
    original_read = benchmark._read_regular_file

    def tracked_read(path: Path, *args: object, **kwargs: object) -> bytes:
        if path != manifest_path:
            asset_reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(benchmark, "_read_regular_file", tracked_read)

    _assert_invalid(manifest_path)
    assert asset_reads == []


@pytest.mark.skipif(os.name != "nt", reason="requires Windows filesystem semantics")
def test_rejects_real_windows_case_alias(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    first, second = manifest["cases"][:2]
    alias = first["path"].upper()
    if not (tmp_path / alias).exists():
        pytest.skip("filesystem is case-sensitive")
    second["path"] = alias
    second["bytes"] = first["bytes"]
    second["sha256"] = first["sha256"]

    _assert_invalid(_write_manifest(tmp_path, manifest))


@pytest.mark.skipif(os.name != "nt", reason="requires NTFS alternate data streams")
def test_rejects_real_windows_alternate_data_stream(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    source = tmp_path / manifest["cases"][0]["path"]
    stream_name = f"{source.name}:asset"
    stream_path = tmp_path / stream_name
    payload = b"alternate-data"
    try:
        stream_path.write_bytes(payload)
    except OSError as error:
        pytest.skip(f"alternate data streams unavailable: {error.__class__.__name__}")
    second = manifest["cases"][1]
    second["path"] = stream_name
    second["bytes"] = len(payload)
    second["sha256"] = hashlib.sha256(payload).hexdigest()

    _assert_invalid(_write_manifest(tmp_path, manifest))


@pytest.mark.parametrize("problem", ["size", "sha256"])
def test_rejects_asset_size_or_hash_mismatch(tmp_path: Path, problem: str) -> None:
    manifest = _valid_manifest(tmp_path)
    if problem == "size":
        manifest["cases"][0]["bytes"] += 1
    else:
        manifest["cases"][0]["sha256"] = "0" * 64

    _assert_invalid(_write_manifest(tmp_path, manifest))


def test_rejects_asset_symlink_without_leaking_target_path(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    target = tmp_path / "private-target"
    target.write_bytes(b"secret")
    source = tmp_path / manifest["cases"][0]["path"]
    source.unlink()
    try:
        source.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error.__class__.__name__}")

    _assert_invalid(_write_manifest(tmp_path, manifest))


def test_rejects_multiply_linked_asset(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    source = tmp_path / manifest["cases"][0]["path"]
    os.link(source, tmp_path / "second-name.png")

    _assert_invalid(_write_manifest(tmp_path, manifest))


def test_file_identity_changes_when_link_count_changes(tmp_path: Path) -> None:
    source = tmp_path / "asset.png"
    source.write_bytes(b"asset")
    before = source.lstat()

    os.link(source, tmp_path / "second-name.png")

    assert benchmark._identity(source.lstat()) != benchmark._identity(before)


def test_rejects_manifest_symlink(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    real_manifest = _write_manifest(tmp_path, manifest)
    linked_manifest = tmp_path / "linked.json"
    try:
        linked_manifest.symlink_to(real_manifest)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error.__class__.__name__}")
    _assert_invalid(linked_manifest)


def test_rejects_oversize_manifest(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    real_manifest = _write_manifest(tmp_path, manifest)
    real_manifest.write_bytes(b" " * (64 * 1024 + 1))

    _assert_invalid(real_manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["routes"][0].update(extra=True),
        lambda manifest: manifest["routes"][0].update(id="pdf"),
        lambda manifest: manifest["routes"][0].update(pages=True),
        lambda manifest: manifest["routes"][0].update(pages=7),
        lambda manifest: manifest["routes"][0]["cases"].__setitem__(0, "missing"),
        lambda manifest: manifest["routes"][1]["cases"].append("image_0"),
        lambda manifest: manifest["routes"].reverse(),
    ],
)
def test_rejects_invalid_route_contract(tmp_path: Path, mutation: object) -> None:
    manifest = _valid_manifest(tmp_path)
    mutation(manifest)

    _assert_invalid(_write_manifest(tmp_path, manifest))


def test_rejects_wrong_corpus_sha256(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    manifest["corpus_sha256"] = "0" * 64

    _assert_invalid(_write_manifest(tmp_path, manifest, refresh_corpus_sha=False))


def test_rejects_more_than_twelve_mib_of_assets(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    payload = b"x" * (12 * 1024 * 1024)
    first = manifest["cases"][0]
    (tmp_path / first["path"]).write_bytes(payload)
    first["bytes"] = len(payload)
    first["sha256"] = hashlib.sha256(payload).hexdigest()

    _assert_invalid(_write_manifest(tmp_path, manifest))


def _completed_process(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _loaded_manifest(tmp_path: Path) -> benchmark.CorpusManifest:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return benchmark.load_manifest(_write_manifest(corpus, _valid_manifest(corpus)))


def test_run_cli_uses_isolated_interpreter_and_clean_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setenv("PYTHONPATH", "untrusted")

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return _completed_process(command, stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)

    completed = benchmark._run_cli(
        ["doctor", "--agent-local"], cwd=tmp_path, timeout=180.0
    )

    assert completed.stdout == "ok"
    assert len(calls) == 1
    command, options = calls[0]
    assert command[:5] == [sys.executable, "-I", "-B", "-m", "image2editable"]
    assert command[5:] == ["doctor", "--agent-local"]
    assert options["cwd"] == tmp_path
    assert "PYTHONPATH" not in options["env"]
    assert options["capture_output"] is True
    assert options["text"] is True
    assert options["check"] is False
    assert options["timeout"] == 180.0


@pytest.mark.parametrize(
    ("doctor_result", "raises_timeout"),
    [
        ({"returncode": 4, "stdout": '{"ready":true,"checks":{}}'}, False),
        ({"returncode": 0, "stdout": '{"ready":false,"checks":{}}'}, False),
        ({"returncode": 0, "stdout": "not-json"}, False),
        ({"returncode": 0, "stdout": "{}\n{}"}, False),
        ({"returncode": 0, "stdout": '{"ready":true,"checks":[]}'}, False),
        (
            {
                "returncode": 0,
                "stdout": '{"ready":true,"checks":{},"extra":true}',
            },
            False,
        ),
        (
            {
                "returncode": 0,
                "stdout": '{"ready":false,"ready":true,"checks":{}}',
            },
            False,
        ),
        (
            {
                "returncode": 0,
                "stdout": '{"ready":true,"checks":{"probe":NaN}}',
            },
            False,
        ),
        (
            {
                "returncode": 0,
                "stdout": '{"ready":true,"checks":{"probe":Infinity}}',
            },
            False,
        ),
        (
            {
                "returncode": 0,
                "stdout": '{"ready":true,"checks":{"probe":-Infinity}}',
            },
            False,
        ),
        ({}, True),
    ],
)
def test_doctor_failure_prevents_all_conversions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    doctor_result: dict[str, object],
    raises_timeout: bool,
) -> None:
    manifest = _loaded_manifest(tmp_path)
    result_root = tmp_path / "results"
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if raises_timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return _completed_process(command, **doctor_result)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(benchmark.BenchmarkError) as raised:
        benchmark.execute_routes(manifest, result_root)

    assert str(raised.value) == "doctor_not_ready"
    assert [command[5:] for command in commands] == [["doctor", "--agent-local"]]


def test_execute_routes_calls_doctor_once_then_three_exact_conversions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _loaded_manifest(tmp_path)
    result_root = tmp_path / "results"
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setenv("PYTHONPATH", "untrusted")

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if command[5] == "doctor":
            return _completed_process(
                command, stdout='{"ready":true,"checks":{"agent_local":"ok"}}\n'
            )
        run_path = Path(command[command.index("--run-dir") + 1])
        output_path = Path(command[command.index("--output") + 1])
        assert not run_path.exists()
        assert not output_path.exists()
        return _completed_process(command, stdout=f"converted:{command[6]}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    executions = benchmark.execute_routes(manifest, result_root)

    assert result_root.is_dir()
    assert len(calls) == 4
    assert [command[5] for command, _ in calls] == [
        "doctor",
        "convert",
        "convert",
        "convert",
    ]
    assert calls[0][0][5:] == ["doctor", "--agent-local"]
    assert calls[0][1]["timeout"] == 180.0
    for index, (route, execution) in enumerate(zip(manifest.routes, executions), start=1):
        command, options = calls[index]
        run_path = result_root.resolve() / route.identifier / "run"
        output_path = result_root.resolve() / route.identifier / "output.pptx"
        assert command == [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "image2editable",
            "convert",
            *(str(source) for source in route.sources),
            "--run-dir",
            str(run_path),
            "--output",
            str(output_path),
            "--slide-size",
            "16:9",
            "--agent-provider",
            "local",
        ]
        assert options["cwd"] == result_root.resolve()
        assert options["timeout"] is None
        assert "PYTHONPATH" not in options["env"]
        assert execution.route is route
        assert execution.run_root == run_path
        assert execution.output_path == output_path
        assert execution.returncode == 0
        assert execution.stdout.startswith("converted:")
        assert math.isfinite(execution.duration_seconds)
        assert execution.duration_seconds >= 0


def test_nonzero_conversion_is_recorded_and_does_not_stop_later_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _loaded_manifest(tmp_path)
    result_root = tmp_path / "results"
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[5] == "doctor":
            return _completed_process(command, stdout='{"ready":true,"checks":{}}')
        if len(calls) == 2:
            return _completed_process(
                command, returncode=9, stdout="failed-output", stderr="secret-stderr"
            )
        return _completed_process(command, stdout="later-output")

    monkeypatch.setattr(subprocess, "run", fake_run)

    executions = benchmark.execute_routes(manifest, result_root)

    assert [execution.route.identifier for execution in executions] == [
        "images",
        "pdf",
        "mixed_pptx",
    ]
    assert [execution.returncode for execution in executions] == [9, 0, 0]
    assert [execution.stdout for execution in executions] == [
        "failed-output",
        "later-output",
        "later-output",
    ]
    assert not hasattr(executions[0], "stderr")
    assert len(calls) == 4


def test_nonfinite_clock_duration_is_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _loaded_manifest(tmp_path)
    clock_values = iter([0.0, math.inf, 1.0, 2.0, 3.0, 4.0])

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[5] == "doctor":
            return _completed_process(command, stdout='{"ready":true,"checks":{}}')
        return _completed_process(command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock_values))

    executions = benchmark.execute_routes(manifest, tmp_path / "results")

    assert executions[0].duration_seconds == 0.0
    assert all(
        math.isfinite(execution.duration_seconds) for execution in executions
    )


def test_conversion_exception_propagates_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _loaded_manifest(tmp_path)
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[5] == "doctor":
            return _completed_process(command, stdout='{"ready":true,"checks":{}}')
        raise OSError("conversion failed")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OSError, match="conversion failed"):
        benchmark.execute_routes(manifest, tmp_path / "results")

    assert [command[5] for command in calls] == ["doctor", "convert"]


def test_existing_result_root_fails_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _loaded_manifest(tmp_path)
    result_root = tmp_path / "results"
    result_root.mkdir()

    def unexpected_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    with pytest.raises(benchmark.BenchmarkError) as raised:
        benchmark.execute_routes(manifest, result_root)

    assert str(raised.value) == "invalid_output"


def test_execute_revalidates_assets_before_output_or_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _loaded_manifest(tmp_path)
    source = manifest.routes[0].sources[0]
    original = source.read_bytes()
    replacement = b"x" * len(original)
    assert replacement != original
    source.write_bytes(replacement)
    result_root = tmp_path / "results"
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[5] == "doctor":
            return _completed_process(command, stdout='{"ready":true,"checks":{}}')
        return _completed_process(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(benchmark.BenchmarkError) as raised:
        benchmark.execute_routes(manifest, result_root)

    assert str(raised.value) == "invalid_corpus"
    assert calls == []
    assert not result_root.exists()


def test_dangling_result_entry_fails_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _loaded_manifest(tmp_path)
    result_root = tmp_path / "dangling-results"
    try:
        result_root.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error.__class__.__name__}")

    def unexpected_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    with pytest.raises(benchmark.BenchmarkError) as raised:
        benchmark.execute_routes(manifest, result_root)

    assert str(raised.value) == "invalid_output"
