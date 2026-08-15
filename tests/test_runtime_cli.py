from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pytest

from image2editable import cli
from image2editable import doctor
from image2editable.inputs import prepare_image_job
from image2editable.store import RunStore


def test_pyproject_exposes_complete_package_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"] == {
        "name": "image2editable",
        "version": "0.1.0",
        "description": "Local-first image to editable PPTX and layered PSD runtime",
        "readme": "README_EN.md",
        "requires-python": ">=3.10,<3.13",
        "license": {"file": "LICENSE"},
        "dynamic": ["dependencies"],
        "scripts": {"image2editable": "image2editable.cli:main"},
        "optional-dependencies": {
            "agent-local": [
                "huggingface-hub>=0.34.0",
                "torch>=2.5.1,<3",
                "transformers>=4.57,<5",
                "accelerate>=1.8,<2",
            ],
                "psd": ["aspose-psd>=26.5.0"],
                "render-qa": ["pywin32>=306; sys_platform == 'win32'"],
                "test": [
                "pytest",
                "pypdf>=5",
                "reportlab>=4",
                "tomli>=2; python_version < '3.11'",
            ],
        },
    }
    assert data["tool"]["setuptools"]["py-modules"] == [
        "image_to_ppt",
        "image_to_psd",
    ]
    assert data["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "image2editable*",
        "scripts*",
    ]
    assert data["tool"]["setuptools"]["package-data"] == {
        "image2editable": ["model_catalog.json", "runtime_model_catalog.json"]
    }
    assert data["tool"]["setuptools"]["dynamic"]["dependencies"] == {
        "file": ["requirements.txt"]
    }
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "aspose-psd" not in requirements.casefold()
    assert requirements.count("pypdfium2>=5.7.1,<6") == 1


def _doctor_probe_results(
    names: list[str],
    *,
    missing: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    missing = missing or set()
    return {
        name: (
            {"module": name, "ok": False, "error_type": "ModuleNotFoundError"}
            if name in missing
            else {"module": name, "ok": True}
        )
        for name in names
    }


def _stub_ready_doctor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing: set[str] | None = None,
    tesseract_ok: bool = True,
) -> None:
    monkeypatch.setattr(
        doctor,
        "_probe_modules",
        lambda names: _doctor_probe_results(names, missing=missing),
    )
    monkeypatch.setattr(
        doctor,
        "_tesseract_status",
        lambda: (
            {"module": "tesseract", "ok": True}
            if tesseract_ok
            else {"module": "tesseract", "ok": False, "error_type": "ProcessExit"}
        ),
    )
    monkeypatch.setattr(
        doctor,
        "runtime_model_status",
        lambda: {"installed": True, "valid": True},
    )
    monkeypatch.setattr(
        doctor,
        "model_status",
        lambda: {"installed": True, "valid": True},
    )
    monkeypatch.setattr(doctor.sys, "version_info", (3, 12, 0))


def _doctor_import_hook(body: str) -> str:
    indented = "\n".join(f"        {line}" for line in body.splitlines())
    return f"""
import importlib.abc
import importlib.util
import sys

class DoctorTestFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "doctor_test_probe":
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
{indented}

sys.meta_path.insert(0, DoctorTestFinder())
""" + doctor._IMPORT_SCRIPT


@pytest.mark.parametrize(
    ("missing_module", "expected_ready"),
    [
        ("numpy", False),
        ("pypdfium2", False),
        ("accelerate", False),
        ("aspose.psd", True),
    ],
)
def test_doctor_real_import_checks_control_ready(
    monkeypatch: pytest.MonkeyPatch,
    missing_module: str,
    expected_ready: bool,
) -> None:
    requested = []

    def fake_probe(names: list[str]) -> dict[str, dict[str, object]]:
        requested.extend(names)
        return _doctor_probe_results(names, missing={missing_module})

    monkeypatch.setattr(doctor, "_probe_modules", fake_probe)
    monkeypatch.setattr(
        doctor,
        "_tesseract_status",
        lambda: {
            "module": "tesseract",
            "ok": False,
            "error_type": "FileNotFoundError",
        },
    )
    monkeypatch.setattr(
        doctor,
        "runtime_model_status",
        lambda: {"installed": True, "valid": True},
    )
    monkeypatch.setattr(doctor.sys, "version_info", (3, 12, 0))

    report = doctor.check_environment()

    assert requested == [
        "pptx",
        "cv2",
        "PIL",
        "numpy",
        "pypdfium2",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "sam2",
        "paddleocr",
        "paddle",
        "pytesseract",
        "aspose.psd",
    ]
    assert report["ready"] is expected_ready
    assert report["checks"]["aspose-psd"]["required"] is False
    assert report["checks"]["numpy"]["required"] is True
    assert "lama" not in report["checks"]
    if missing_module == "aspose.psd":
        assert report["checks"]["aspose-psd"]["next_command"] == (
            'python -m pip install ".[psd]"'
        )


def test_doctor_batch_probe_really_imports_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)

    results = doctor._probe_modules(["json", "missing_doctor_probe_module"])

    assert results == {
        "json": {"module": "json", "ok": True},
        "missing_doctor_probe_module": {
            "module": "missing_doctor_probe_module",
            "ok": False,
            "error_type": "ModuleNotFoundError",
        },
    }


def test_doctor_batch_probe_handles_no_newline_noise_and_redacts_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doctor,
        "_IMPORT_SCRIPT",
        _doctor_import_hook(
            "print('third-party-noise', end='')\n"
            "raise RuntimeError(r'C:\\\\private\\\\model-cache')"
        ),
    )

    results = doctor._probe_modules(["json", "doctor_test_probe"])

    assert results == {
        "json": {"module": "json", "ok": True},
        "doctor_test_probe": {
            "module": "doctor_test_probe",
            "ok": False,
            "error_type": "RuntimeError",
        },
    }
    assert "private" not in json.dumps(results)


def test_doctor_batch_probe_normalizes_unsafe_exception_class_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doctor,
        "_IMPORT_SCRIPT",
        _doctor_import_hook(
            "SecretError = type('C_Users_private_model_cache', (Exception,), {})\n"
            "raise SecretError()"
        ),
    )

    results = doctor._probe_modules(["doctor_test_probe"])

    assert results == {
        "doctor_test_probe": {
            "module": "doctor_test_probe",
            "ok": False,
            "error_type": "ImportFailed",
        }
    }
    assert "private" not in json.dumps(results)


def test_doctor_batch_probe_timeout_marks_the_whole_batch_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def timeout(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise subprocess.TimeoutExpired("python", 1)

    monkeypatch.setattr(doctor.subprocess, "run", timeout)

    assert doctor._probe_modules(["torch", "transformers"]) == {
        "torch": {"module": "torch", "ok": False, "error_type": "ProbeFailed"},
        "transformers": {
            "module": "transformers",
            "ok": False,
            "error_type": "ProbeFailed",
        },
    }
    command = calls[0][0][0]
    assert command[1:3] == ["-I", "-B"]
    assert calls[0][1]["timeout"] > 0


def test_doctor_batch_probe_process_failure_marks_the_whole_batch_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doctor,
        "_IMPORT_SCRIPT",
        "import os\nos._exit(7)\n" + doctor._IMPORT_SCRIPT,
    )

    results = doctor._probe_modules(["json", "doctor_test_probe"])

    assert results == {
        "json": {"module": "json", "ok": False, "error_type": "ProbeFailed"},
        "doctor_test_probe": {
            "module": "doctor_test_probe",
            "ok": False,
            "error_type": "ProbeFailed",
        },
    }


@pytest.mark.parametrize(
    ("return_code", "expected"),
    [
        (0, {"module": "tesseract", "ok": True}),
        (
            2,
            {"module": "tesseract", "ok": False, "error_type": "ProcessExit"},
        ),
    ],
)
def test_doctor_tesseract_requires_version_exit_zero_without_output_leak(
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
    expected: dict[str, object],
) -> None:
    calls = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args[0], return_code, stdout=r"C:\private\tesseract", stderr="secret"
        )

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)

    assert doctor._tesseract_status() == expected
    assert calls[0][0][0] == ["tesseract", "--version"]
    assert calls[0][1]["timeout"] > 0


@pytest.mark.parametrize(
    ("ocr_modules", "tesseract_ok", "expected_ok"),
    [
        (set(), False, False),
        ({"paddleocr", "paddle"}, False, True),
        ({"pytesseract"}, True, True),
        ({"paddleocr"}, False, False),
        ({"paddle"}, False, False),
        ({"pytesseract"}, False, False),
        (set(), True, False),
    ],
)
def test_doctor_requires_one_complete_ocr_path(
    monkeypatch: pytest.MonkeyPatch,
    ocr_modules: set[str],
    tesseract_ok: bool,
    expected_ok: bool,
) -> None:
    missing = {
        name
        for name in {"paddleocr", "paddle", "pytesseract"}
        if name not in ocr_modules
    }
    _stub_ready_doctor(
        monkeypatch,
        missing=missing,
        tesseract_ok=tesseract_ok,
    )

    report = doctor.check_environment()
    ocr = report["checks"]["ocr"]

    assert ocr["required"] is True
    assert ocr["ok"] is expected_ok
    assert report["ready"] is expected_ok
    assert ocr["detail"]["paddle"]["ok"] is (
        {"paddleocr", "paddle"} <= ocr_modules
    )
    assert ocr["detail"]["tesseract"]["ok"] is (
        "pytesseract" in ocr_modules and tesseract_ok
    )
    if not expected_ok:
        assert ocr["next_command"] == "python -m pip install paddleocr paddlepaddle"


def test_doctor_requires_valid_runtime_models_without_installing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image2editable import models, runtime_models

    monkeypatch.setenv("IMAGE2EDITABLE_MODEL_CACHE", str(tmp_path))
    monkeypatch.setattr(
        doctor,
        "_probe_modules",
        lambda names: _doctor_probe_results(names),
    )
    monkeypatch.setattr(
        doctor,
        "_tesseract_status",
        lambda: {"module": "tesseract", "ok": True},
    )
    monkeypatch.setattr(
        runtime_models,
        "install_runtime_models",
        lambda **kwargs: pytest.fail("doctor must not install runtime models"),
    )
    monkeypatch.setattr(
        runtime_models,
        "download_file",
        lambda **kwargs: pytest.fail("doctor must not download runtime models"),
    )
    monkeypatch.setattr(
        runtime_models,
        "snapshot_download",
        lambda **kwargs: pytest.fail("doctor must not download snapshots"),
    )
    monkeypatch.setattr(
        models,
        "install_agent_model",
        lambda **kwargs: pytest.fail("doctor must not install agent models"),
    )
    monkeypatch.setattr(
        models,
        "snapshot_download",
        lambda **kwargs: pytest.fail("doctor must not download agent snapshots"),
    )

    report = doctor.check_environment()

    assert report["ready"] is False
    assert report["checks"]["runtime-models"] == {
        "ok": False,
        "required": True,
        "detail": {
            "module": "runtime-models",
            "ok": False,
            "error_type": "MissingOrInvalidReceipt",
        },
        "next_command": "image2editable models install runtime",
    }
    assert "agent-model" not in report["checks"]


def test_doctor_agent_local_requires_imports_and_receipt_without_leaking_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_ready_doctor(monkeypatch, missing={"huggingface_hub"})
    monkeypatch.setattr(
        doctor,
        "model_status",
        lambda: {
            "installed": True,
            "valid": False,
            "reason": r"invalid receipt at C:\private\qwen",
        },
    )

    report = doctor.check_environment(agent_local=True)

    assert report["ready"] is False
    assert report["checks"]["huggingface-hub"]["next_command"] == (
        'python -m pip install ".[agent-local]"'
    )
    assert report["checks"]["agent-model"]["next_command"] == (
        "image2editable models install agent"
    )
    assert "private" not in json.dumps(report)


def test_doctor_normalizes_unsafe_model_status_exception_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_error = type(r"C:\private\model-cache", (Exception,), {})

    def broken_status() -> dict[str, object]:
        raise secret_error()

    _stub_ready_doctor(monkeypatch)
    monkeypatch.setattr(doctor, "runtime_model_status", broken_status)

    report = doctor.check_environment()

    assert report["checks"]["runtime-models"]["detail"]["error_type"] == (
        "StatusFailed"
    )
    assert "private" not in json.dumps(report)


def test_doctor_uses_platform_appropriate_python_next_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_ready_doctor(monkeypatch)
    monkeypatch.setattr(doctor.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(doctor.sys, "platform", "linux")

    report = doctor.check_environment()

    assert report["checks"]["python"]["next_command"] == (
        "python3.12 -m image2editable doctor"
    )


def test_doctor_python_next_command_preserves_agent_local_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_ready_doctor(monkeypatch)
    monkeypatch.setattr(doctor.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(doctor.sys, "platform", "linux")

    report = doctor.check_environment(agent_local=True)

    assert report["checks"]["python"]["next_command"] == (
        "python3.12 -m image2editable doctor --agent-local"
    )


def test_cli_convert_forwards_all_image_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []

    def fake_convert(*args: object, **kwargs: object) -> dict[str, object]:
        print("conversion progress")
        calls.append((args, kwargs))
        return {"status": "completed", "outputs": {"16:9": "结果.pptx"}}

    monkeypatch.setattr(cli.runtime, "convert", fake_convert)

    exit_code = cli.main(
        [
            "convert",
            "一.png",
            "two.png",
            "--run-dir",
            "run",
            "--output",
            "out.pptx",
            "--slide-size",
            "16:9",
            "--lang",
            "en",
            "--agent-provider",
            "local",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert calls == [
        (
            (["一.png", "two.png"],),
            {
                "run_dir": "run",
                "output_path": "out.pptx",
                "slide_size": "16:9",
                "lang": "en",
                "agent_provider": "local",
            },
        )
    ]
    assert json.loads(captured.out) == {
        "status": "completed",
        "outputs": {"16:9": "结果.pptx"},
    }
    assert "conversion progress" in captured.err


@pytest.mark.parametrize("command", ["prepare", "convert"])
def test_cli_forwards_psd_output_format(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    calls = []

    def fake_method(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return Path("run") if command == "prepare" else {"status": "completed"}

    monkeypatch.setattr(
        cli.runtime,
        "prepare_job" if command == "prepare" else "convert",
        fake_method,
    )

    assert cli.main([command, "source.png", "--format", "psd"]) == 0
    capsys.readouterr()
    assert calls[0][1]["output_format"] == "psd"


def test_cli_json_keeps_non_ascii_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli.runtime,
        "convert",
        lambda *args, **kwargs: {"message": "转换完成"},
    )

    assert cli.main(["convert", "source.png"]) == 0

    output = capsys.readouterr().out
    assert "转换完成" in output
    assert "\\u8f6c" not in output


def test_cli_prepare_forwards_all_image_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    calls = []
    prepared = tmp_path / "任务"

    def fake_prepare(*args: object, **kwargs: object) -> Path:
        calls.append((args, kwargs))
        return prepared

    monkeypatch.setattr(cli.runtime, "prepare_job", fake_prepare)

    exit_code = cli.main(
        [
            "prepare",
            "source.png",
            "--run-dir",
            "run",
            "-o",
            "out.pptx",
            "--slide-size",
            "original",
            "--lang",
            "ch",
            "--agent-provider",
            "local",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            (["source.png"],),
            {
                "run_dir": "run",
                "output_path": "out.pptx",
                "slide_size": "original",
                "lang": "ch",
                "agent_provider": "local",
            },
        )
    ]
    assert json.loads(capsys.readouterr().out) == {
        "run_dir": str(prepared.resolve()),
        "status": "prepared",
    }


@pytest.mark.parametrize("source", ["document.pdf", "deck.pptx"])
@pytest.mark.parametrize("command", ["prepare", "convert"])
def test_cli_forwards_one_document_source(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: str,
    command: str,
) -> None:
    calls = []

    def fake_method(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        if command == "prepare":
            return Path("run")
        return {"status": "completed"}

    monkeypatch.setattr(cli.runtime, "prepare_job" if command == "prepare" else "convert", fake_method)

    assert cli.main([command, source]) == 0
    capsys.readouterr()
    assert calls == [
        (
            ([source],),
            {
                "run_dir": None,
                "output_path": None,
                "slide_size": "both",
                "lang": "ch",
                "agent_provider": "host",
            },
        )
    ]


def test_cli_source_argument_has_document_help() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["prepare", "deck.pptx"])
    help_text = parser._subparsers._group_actions[0].choices["prepare"].format_help()

    assert args.sources == ["deck.pptx"]
    assert "one PDF or one PPTX" in help_text


@pytest.mark.parametrize(
    ("argv", "method_name", "expected_args", "result"),
    [
        (
            ["run", "status", "run-dir"],
            "get_status",
            ("run-dir",),
            {"run": {"status": "prepared"}},
        ),
        (
            ["run", "execute", "run-dir"],
            "run_job",
            ("run-dir",),
            {"status": "completed"},
        ),
        (
            ["run", "recover", "run-dir"],
            "recover_job",
            ("run-dir",),
            {"run": {"status": "prepared"}},
        ),
        (
            ["run", "retry", "run-dir", "--page", "page_003"],
            "retry_page",
            ("run-dir", "page_003"),
            {"run": {"status": "prepared"}},
        ),
        (
            ["run", "render-detail", "run-dir", "--page", "page_001"],
            "rerender_pdf_page",
            ("run-dir", "page_001"),
            {"detail_used": True, "activated": True},
        ),
    ],
)
def test_cli_run_routes_forward_exact_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    method_name: str,
    expected_args: tuple[str, ...],
    result: dict[str, object],
) -> None:
    calls = []

    def fake_method(*args: object) -> dict[str, object]:
        print("runtime progress")
        calls.append(args)
        return result

    monkeypatch.setattr(cli.runtime, method_name, fake_method)

    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    assert calls == [expected_args]
    assert json.loads(captured.out) == result
    assert "runtime progress" in captured.err


@pytest.mark.parametrize(
    ("ready", "expected_exit"),
    [(True, 0), (False, 1)],
)
def test_cli_doctor_exit_code_follows_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ready: bool,
    expected_exit: int,
) -> None:
    report = {"ready": ready, "checks": {"python": {"ok": ready}}}
    calls = []

    def fake_check_environment(*, agent_local: bool = False) -> dict[str, object]:
        calls.append(agent_local)
        return report

    monkeypatch.setattr(cli, "check_environment", fake_check_environment)

    assert cli.main(["doctor"]) == expected_exit
    assert calls == [False]
    assert json.loads(capsys.readouterr().out) == report


def test_cli_doctor_agent_local_forwards_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []

    def fake_check_environment(*, agent_local: bool = False) -> dict[str, object]:
        calls.append(agent_local)
        return {"ready": True, "checks": {}}

    monkeypatch.setattr(cli, "check_environment", fake_check_environment)

    assert cli.main(["doctor", "--agent-local"]) == 0
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["ready"] is True


@pytest.mark.parametrize("provider", [None, "remote"])
def test_cli_status_rejects_missing_or_invalid_manifest_agent_provider(
    tmp_path: Path, provider: object
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    manifest_path = run_dir / "job_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if provider is None:
        del manifest["options"]["agent_provider"]
    else:
        manifest["options"]["agent_provider"] = provider
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest.*agent_provider"):
        cli.main(["run", "status", str(run_dir)])


def test_module_help_starts() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "image2editable", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "convert" in result.stdout
    assert "prepare" in result.stdout
    assert "run" in result.stdout
    assert "doctor" in result.stdout


def test_cli_warning_image_exits_nonzero_without_pptx(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    run_dir = prepare_image_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run_dir)
    reconstruction = run_dir / "pages/page_001/reconstruction"
    reconstruction.mkdir()
    store.write_json(
        "pages/page_001/reconstruction/component_state.json",
        {"status": "preserved_with_warning"},
    )
    page_jobs = store.read_json("page_jobs.json")
    page_jobs["pages"]["page_001"]["status"] = "preserved_with_warning"
    store.write_json("page_jobs.json", page_jobs)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "image2editable",
            "run",
            "execute",
            str(run_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "editable reconstruction incomplete" in result.stderr
    assert store.read_json("run_summary.json")["status"] == "failed"
    assert not (run_dir / "final/output_original.pptx").exists()
    assert not (run_dir / "final/output_16x9.pptx").exists()


def test_public_api_is_importable() -> None:
    from image2editable import (
        PageStatus,
        RunStatus,
        SCHEMA_VERSION,
        check_environment,
        convert,
        get_status,
        prepare_job,
        retry_page,
        rerender_pdf_page,
        run_job,
    )

    assert PageStatus
    assert RunStatus
    assert SCHEMA_VERSION
    assert callable(check_environment)
    assert callable(convert)
    assert callable(get_status)
    assert callable(prepare_job)
    assert callable(retry_page)
    assert callable(rerender_pdf_page)
    assert callable(run_job)


def test_cli_agent_next_and_record_emit_only_json_to_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli.runtime, "next_host_agent_item", lambda run: {"kind": "x"})
    monkeypatch.setattr(cli.runtime, "record_host_agent_plan", lambda run, value: {"status": "recorded"})
    assert cli.main(["agent", "next", "run"]) == 0
    assert json.loads(capsys.readouterr().out) == {"kind": "x"}
    assert cli.main(["agent", "record", "run", "--plan", str(plan)]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "recorded"}


def test_cli_models_recommend_json_is_routed_lazily(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from image2editable import models

    profile = models.HardwareProfile(8, 16, 20, True)
    expected = {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "compatible": True,
    }
    monkeypatch.setattr(models, "detect_hardware", lambda cache_dir=None: profile)
    monkeypatch.setattr(
        models,
        "recommend_agent_model",
        lambda hardware, cache_dir=None: expected,
    )

    assert cli.main(["models", "recommend", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_cli_models_install_yes_displays_plan_before_installing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from image2editable import models

    profile = models.HardwareProfile(8, 16, 20, True)
    plan = {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "main",
        "required_free_disk_gib": 8,
        "cache_dir": "cache",
        "stability": "experimental",
        "compatible": True,
    }
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(models, "detect_hardware", lambda cache_dir=None: profile)
    monkeypatch.setattr(
        models,
        "recommend_agent_model",
        lambda hardware, cache_dir=None: plan,
    )

    def install(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"model_id": plan["model_id"], "resolved_revision": "a" * 40}

    monkeypatch.setattr(models, "install_agent_model", install)

    assert cli.main(["models", "install", "agent", "--yes"]) == 0
    captured = capsys.readouterr()
    assert calls == [
        {
            "cache_dir": None,
            "confirmed": True,
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "main",
        }
    ]
    assert json.loads(captured.out)["resolved_revision"] == "a" * 40
    assert "Qwen/Qwen3-VL-2B-Instruct" in captured.err
    assert "experimental" in captured.err


def test_cli_models_install_cancelled_before_installer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from image2editable import models

    profile = models.HardwareProfile(8, 16, 20, True)
    monkeypatch.setattr(models, "detect_hardware", lambda cache_dir=None: profile)
    monkeypatch.setattr(
        models,
        "recommend_agent_model",
        lambda hardware, cache_dir=None: {
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "main",
            "required_free_disk_gib": 8,
            "cache_dir": "cache",
            "stability": "experimental",
            "compatible": True,
        },
    )
    monkeypatch.setattr(
        models,
        "install_agent_model",
        lambda **kwargs: pytest.fail("installer must not run after cancellation"),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "no")

    assert cli.main(["models", "install", "agent"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "cancelled"}
    assert "上述实验性模型" in captured.err


def test_cli_models_install_runtime_yes_prints_safe_plan_before_installing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from image2editable import runtime_models

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_models,
        "install_runtime_models",
        lambda **kwargs: calls.append(kwargs) or {"schema_version": 1},
    )
    secret_cache = "C:/Users/private-account/models"
    monkeypatch.setenv("IMAGE2EDITABLE_MODEL_CACHE", secret_cache)

    assert cli.main(["models", "install", "runtime", "--yes"]) == 0

    captured = capsys.readouterr()
    plan = json.loads(captured.err)
    assert calls == [{"cache_dir": None, "confirmed": True}]
    assert json.loads(captured.out) == {"schema_version": 1}
    assert plan == {
        "cache": "IMAGE2EDITABLE_MODEL_CACHE or the default user runtime cache",
        "estimated_download": {
            "additional": "Grounding DINO snapshot (size not declared in catalog)",
            "minimum_bytes": 1103887281,
        },
        "models": {
            "big_lama": {
                "sha256": (
                    "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
                ),
                "size": 205803670,
            },
            "grounding_dino": {
                "model_id": "IDEA-Research/grounding-dino-tiny",
                "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
            },
            "sam2_large": {
                "sha256": (
                    "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
                ),
                "size": 898083611,
            },
        },
        "target": "runtime",
    }
    assert secret_cache not in captured.err


@pytest.mark.parametrize("answer", ["no", "", None])
def test_cli_models_install_runtime_cancelled_without_network(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str | None,
) -> None:
    from image2editable import runtime_models

    def unexpected_call(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            f"cancelled runtime install attempted network: {args} {kwargs}"
        )

    monkeypatch.setattr(runtime_models, "install_runtime_models", unexpected_call)
    monkeypatch.setattr(runtime_models, "download_file", unexpected_call)
    monkeypatch.setattr(runtime_models, "snapshot_download", unexpected_call)
    def respond(_prompt: str) -> str:
        if answer is None:
            raise EOFError
        return answer

    monkeypatch.setattr("builtins.input", respond)

    assert cli.main(["models", "install", "runtime"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "cancelled"}
    assert "上述运行时模型" in captured.err
    assert "上述实验性模型" not in captured.err


def test_cli_models_status_prints_agent_and_runtime_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from image2editable import models
    from image2editable import runtime_models

    agent = {
        "installed": False,
        "valid": False,
        "install_command": "image2editable models install agent",
    }
    runtime = {
        "installed": False,
        "valid": False,
        "install_command": "image2editable models install runtime",
    }
    monkeypatch.setattr(models, "model_status", lambda cache_dir=None: agent)
    monkeypatch.setattr(
        runtime_models,
        "runtime_model_status",
        lambda cache_dir=None: runtime,
    )

    assert cli.main(["models", "status"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "agent": agent,
        "runtime": runtime,
    }
