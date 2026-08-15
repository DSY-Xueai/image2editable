from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
from importlib import metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
import sysconfig
from typing import Sequence


MODULES = ("image2editable", "scripts", "image_to_ppt")
CATALOGS = ("model_catalog.json", "runtime_model_catalog.json")
COMMAND_TIMEOUT_SECONDS = 180


class SmokeError(RuntimeError):
    pass


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolved_file(path: object) -> Path:
    if not isinstance(path, (str, Path)):
        raise SmokeError
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError, TypeError):
        raise SmokeError from None
    if not resolved.is_file():
        raise SmokeError
    return resolved


def _distribution():
    try:
        return metadata.distribution("image2editable")
    except metadata.PackageNotFoundError:
        raise SmokeError from None


def _declared_files(distribution) -> set[Path]:
    if not distribution.files:
        raise SmokeError
    declared_files = set()
    for relative_path in distribution.files:
        try:
            declared_files.add(
                Path(distribution.locate_file(relative_path)).resolve(strict=True)
            )
        except (OSError, RuntimeError, TypeError):
            continue
    return declared_files


def verify_imports(checkout: Path) -> tuple[str, ...]:
    try:
        checkout = Path(checkout).resolve(strict=True)
    except (OSError, RuntimeError):
        raise SmokeError from None
    if not checkout.is_dir():
        raise SmokeError
    distribution = _distribution()
    declared_files = _declared_files(distribution)

    install_roots = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if isinstance(value, str):
            install_roots.append(Path(value).resolve())
    if not install_roots:
        raise SmokeError

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            modules = {name: importlib.import_module(name) for name in MODULES}
        except Exception:
            raise SmokeError from None

    module_files = {}
    for name, module in modules.items():
        module_file = _resolved_file(getattr(module, "__file__", None))
        if (
            _is_within(module_file, checkout)
            or module_file not in declared_files
            or not any(_is_within(module_file, root) for root in install_roots)
        ):
            raise SmokeError
        module_files[name] = module_file

    package_root = module_files["image2editable"].parent
    for catalog_name in CATALOGS:
        catalog = _resolved_file(package_root / catalog_name)
        if (
            _is_within(catalog, checkout)
            or catalog not in declared_files
            or not any(_is_within(catalog, root) for root in install_roots)
        ):
            raise SmokeError
    return MODULES


def _verified_launcher(checkout: Path) -> Path:
    try:
        checkout = Path(checkout).resolve(strict=True)
    except (OSError, RuntimeError):
        raise SmokeError from None
    distribution = _distribution()
    entry_points = list(distribution.entry_points)
    if len(entry_points) != 1:
        raise SmokeError
    entry_point = entry_points[0]
    if (
        entry_point.group != "console_scripts"
        or entry_point.name != "image2editable"
        or entry_point.value != "image2editable.cli:main"
    ):
        raise SmokeError
    launcher_text = shutil.which("image2editable")
    if launcher_text is None:
        raise SmokeError
    launcher_path = Path(launcher_text)
    try:
        if launcher_path.is_symlink():
            raise SmokeError
        launcher = launcher_path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise SmokeError from None
    scripts_path = sysconfig.get_paths().get("scripts")
    if not isinstance(scripts_path, str):
        raise SmokeError
    scripts_root = Path(scripts_path).resolve()
    if (
        not launcher.is_file()
        or _is_within(launcher, checkout)
        or not _is_within(launcher, scripts_root)
        or launcher not in _declared_files(distribution)
    ):
        raise SmokeError
    return launcher


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except Exception:
        raise SmokeError from None


def _verify_commands(launcher: Path) -> None:
    command = str(launcher)
    if _run([command, "--help"]).returncode != 0:
        raise SmokeError
    doctor = _run([command, "doctor"])
    if doctor.returncode not in {0, 1}:
        raise SmokeError
    try:
        report = json.loads(doctor.stdout)
    except (TypeError, json.JSONDecodeError):
        raise SmokeError from None
    if (
        not isinstance(report, dict)
        or set(report) != {"ready", "checks"}
        or type(report["ready"]) is not bool
        or not isinstance(report["checks"], dict)
        or doctor.returncode != (0 if report["ready"] else 1)
    ):
        raise SmokeError


def _print_result(ok: bool, modules: tuple[str, ...]) -> None:
    print(json.dumps({"modules": list(modules), "ok": ok}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout-root", type=Path, required=True)
    args = parser.parse_args(argv)
    modules: tuple[str, ...] = ()
    try:
        modules = verify_imports(args.checkout_root)
        launcher = _verified_launcher(args.checkout_root)
        _verify_commands(launcher)
    except Exception:
        _print_result(False, modules)
        return 1
    _print_result(True, modules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
