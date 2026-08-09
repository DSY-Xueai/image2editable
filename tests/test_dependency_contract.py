import re
from pathlib import Path


SAM_PIN = "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"


def test_sam_ref_is_full_commit_sha() -> None:
    sam_line = next(
        line
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.startswith("SAM-2 @ git+")
    )

    assert re.fullmatch(r"[0-9a-f]{40}", sam_line.rsplit("@", 1)[1])


def test_requirements_contains_sam_pin() -> None:
    assert SAM_PIN in REQUIREMENTS.read_text(encoding="utf-8").splitlines()


def test_requirements_has_no_sam_main_reference() -> None:
    assert not any(
        line.startswith("SAM-2 @ git+") and line.endswith("@main")
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    )


def test_requirements_keeps_pillow_floor() -> None:
    assert "Pillow>=9.0.0" in REQUIREMENTS.read_text(encoding="utf-8").splitlines()


def test_pyproject_keeps_supported_python_range() -> None:
    assert 'requires-python = ">=3.10,<3.13"' in PYPROJECT.read_text(
        encoding="utf-8"
    ).splitlines()
