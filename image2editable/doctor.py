from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Callable

_IMPORT_TIMEOUT_SECONDS = 120
_COMMAND_TIMEOUT_SECONDS = 10
_RESULT_PREFIX = "IMAGE2EDITABLE_DOCTOR_RESULT="
_SAFE_IMPORT_ERROR_TYPES = frozenset(
    {
        "AttributeError",
        "ImportError",
        "ImportFailed",
        "ModuleNotFoundError",
        "OSError",
        "RuntimeError",
        "SystemError",
        "TypeError",
        "ValueError",
    }
)
_IMPORT_SCRIPT = f"""
import importlib
import json
import sys

results = []
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
    except BaseException as error:
        error_type = type(error).__name__
        if error_type not in {_SAFE_IMPORT_ERROR_TYPES!r}:
            error_type = "ImportFailed"
        results.append({{"module": name, "ok": False, "error_type": error_type}})
    else:
        results.append({{"module": name, "ok": True}})
sys.__stdout__.write("\\n" + {_RESULT_PREFIX!r} + json.dumps(results) + "\\n")
sys.__stdout__.flush()
"""
_MODULES = (
    ("python-pptx", "pptx"),
    ("opencv", "cv2"),
    ("pillow", "PIL"),
    ("numpy", "numpy"),
    ("pdfium", "pypdfium2"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("sam2", "sam2"),
)
_OCR_MODULES = ("paddleocr", "paddle", "pytesseract")


def model_status() -> dict[str, object]:
    from image2editable.models import model_status as read_status

    return read_status()


def runtime_model_status() -> dict[str, object]:
    from image2editable.runtime_models import runtime_model_status as read_status

    return read_status()


def _failed_probe(names: list[str]) -> dict[str, dict[str, object]]:
    return {
        name: {"module": name, "ok": False, "error_type": "ProbeFailed"}
        for name in names
    }


def _probe_modules(names: list[str]) -> dict[str, dict[str, object]]:
    failed = _failed_probe(names)
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-B", "-c", _IMPORT_SCRIPT, json.dumps(names)],
            capture_output=True,
            text=True,
            check=False,
            timeout=_IMPORT_TIMEOUT_SECONDS,
        )
    except Exception:
        return failed
    if process.returncode != 0:
        return failed
    result_line = next(
        (
            line[len(_RESULT_PREFIX) :]
            for line in reversed(process.stdout.splitlines())
            if line.startswith(_RESULT_PREFIX)
        ),
        None,
    )
    try:
        parsed = json.loads(result_line) if result_line is not None else None
    except (TypeError, json.JSONDecodeError):
        return failed
    if not isinstance(parsed, list) or len(parsed) != len(names):
        return failed
    results: dict[str, dict[str, object]] = {}
    for name, item in zip(names, parsed, strict=True):
        if not isinstance(item, dict) or item.get("module") != name:
            return failed
        if item == {"module": name, "ok": True}:
            results[name] = item
            continue
        if (
            set(item) != {"module", "ok", "error_type"}
            or item.get("ok") is not False
            or not isinstance(item.get("error_type"), str)
        ):
            return failed
        if item["error_type"] not in _SAFE_IMPORT_ERROR_TYPES:
            item = {"module": name, "ok": False, "error_type": "ImportFailed"}
        results[name] = item
    return results


def _tesseract_status() -> dict[str, object]:
    try:
        process = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return {
            "module": "tesseract",
            "ok": False,
            "error_type": "FileNotFoundError",
        }
    except subprocess.TimeoutExpired:
        return {
            "module": "tesseract",
            "ok": False,
            "error_type": "TimeoutExpired",
        }
    except Exception:
        return {
            "module": "tesseract",
            "ok": False,
            "error_type": "CommandFailed",
        }
    if process.returncode != 0:
        return {
            "module": "tesseract",
            "ok": False,
            "error_type": "ProcessExit",
        }
    return {"module": "tesseract", "ok": True}


def _module_check(
    detail: dict[str, object],
    *,
    required: bool = True,
    next_command: str = "python -m pip install .",
) -> dict[str, Any]:
    check: dict[str, Any] = {
        "ok": detail["ok"],
        "required": required,
        "detail": detail,
    }
    if not detail["ok"]:
        check["next_command"] = next_command
    return check


def _ocr_check(results: dict[str, dict[str, object]]) -> dict[str, Any]:
    tesseract = _tesseract_status()
    paddle_ok = results["paddleocr"]["ok"] and results["paddle"]["ok"]
    tesseract_ok = results["pytesseract"]["ok"] and tesseract["ok"]
    check: dict[str, Any] = {
        "ok": paddle_ok or tesseract_ok,
        "required": True,
        "detail": {
            "paddle": {
                "ok": paddle_ok,
                "modules": [results["paddleocr"], results["paddle"]],
            },
            "tesseract": {
                "ok": tesseract_ok,
                "modules": [results["pytesseract"], tesseract],
            },
        },
    }
    if not check["ok"]:
        check["next_command"] = "python -m pip install paddleocr paddlepaddle"
    return check


def _model_check(
    name: str,
    status: Callable[[], dict[str, object]],
    install_command: str,
) -> dict[str, Any]:
    error_type = "MissingOrInvalidReceipt"
    try:
        ok = status().get("valid") is True
    except Exception:
        ok = False
        error_type = "StatusFailed"
    detail: dict[str, object] = {"module": name, "ok": ok}
    if not ok:
        detail["error_type"] = error_type
    return _module_check(detail, next_command=install_command)


def check_environment(*, agent_local: bool = False) -> dict[str, Any]:
    names = [module for _, module in _MODULES]
    names.extend(_OCR_MODULES)
    names.append("aspose.psd")
    if agent_local:
        names.append("huggingface_hub")
    results = _probe_modules(names)

    python_ok = (3, 10) <= sys.version_info[:2] < (3, 13)
    python_detail: dict[str, object] = {"module": "python", "ok": python_ok}
    if not python_ok:
        python_detail["error_type"] = "UnsupportedVersion"
    python_command = (
        "py -3.12 -m image2editable doctor"
        if sys.platform == "win32"
        else "python3.12 -m image2editable doctor"
    )
    if agent_local:
        python_command += " --agent-local"
    checks = {
        "python": _module_check(
            python_detail,
            next_command=python_command,
        ),
        **{
            check_name: _module_check(results[module_name])
            for check_name, module_name in _MODULES
        },
        "ocr": _ocr_check(results),
        "aspose-psd": _module_check(
            results["aspose.psd"],
            required=False,
            next_command='python -m pip install ".[psd]"',
        ),
        "runtime-models": _model_check(
            "runtime-models",
            runtime_model_status,
            "image2editable models install runtime",
        ),
    }
    if agent_local:
        checks["huggingface-hub"] = _module_check(
            results["huggingface_hub"],
            next_command='python -m pip install ".[agent-local]"',
        )
        checks["agent-model"] = _model_check(
            "agent-model",
            model_status,
            "image2editable models install agent",
        )
    return {
        "ready": all(
            check["ok"] for check in checks.values() if check["required"]
        ),
        "checks": checks,
    }
