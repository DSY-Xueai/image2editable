from __future__ import annotations

import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest


@pytest.fixture(scope="module", autouse=True)
def _require_installed_package_contract() -> None:
    if os.environ.get("IMAGE2EDITABLE_INSTALLED_PACKAGE_CONTRACT") != "1":
        pytest.skip("installed-package contract runs only in the wheel job")


def test_import_comes_from_the_installed_distribution() -> None:
    import image2editable

    distribution = metadata.distribution("image2editable")
    installed_root = Path(distribution.locate_file("")).resolve()
    imported = Path(image2editable.__file__).resolve()

    assert imported.is_relative_to(installed_root)
    assert "site-packages" in str(imported).casefold()


def test_packaged_catalogs_are_available() -> None:
    distribution = metadata.distribution("image2editable")
    files = {str(path).replace("\\", "/") for path in distribution.files or ()}

    assert "image2editable/model_catalog.json" in files
    assert "image2editable/runtime_model_catalog.json" in files


def test_module_cli_reports_the_installed_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "image2editable", "--version"],
        cwd=Path(os.environ.get("RUNNER_TEMP", Path.cwd())),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"image2editable {metadata.version('image2editable')}"
