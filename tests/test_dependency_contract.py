import re
from pathlib import Path


SAM_PIN = "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
STANDALONE_REQUIREMENTS = ROOT / "skills" / "image-to-ppt" / "references" / "requirements.txt"
SKILL = ROOT / "skills" / "image-to-ppt" / "SKILL.md"
README = ROOT / "README.md"
README_EN = ROOT / "README_EN.md"
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


def test_standalone_sam_dependency_matches_product_pin() -> None:
    root_lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    standalone_lines = STANDALONE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()

    assert root_lines.count(SAM_PIN) == 1
    assert standalone_lines.count(SAM_PIN) == 1


def test_standalone_sam_dependency_does_not_follow_a_branch() -> None:
    sam_line = next(
        line
        for line in STANDALONE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.startswith("SAM-2 @ git+")
    )

    assert re.fullmatch(r"[0-9a-f]{40}", sam_line.rsplit("@", 1)[1])


def test_standalone_declares_accelerate_used_by_visual_segmentation() -> None:
    expected = "accelerate>=0.26.0"

    assert expected in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    assert expected in STANDALONE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()


def test_cross_platform_docs_prefer_the_verified_current_environment() -> None:
    skill_text = SKILL.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    readme_en_text = README_EN.read_text(encoding="utf-8")

    assert "优先使用 Linux/WSL" not in skill_text
    assert "优先使用当前平台" in skill_text
    assert "通过 `doctor`" in skill_text
    assert "优先使用当前平台" in readme_text
    assert "通过 `doctor`" in readme_text
    assert "prefer Linux/WSL" not in readme_en_text
    assert "current platform" in readme_en_text
    assert "passes `doctor`" in readme_en_text


def test_cross_platform_docs_keep_the_full_quality_model_on_cpu() -> None:
    skill_text = SKILL.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    readme_en_text = README_EN.read_text(encoding="utf-8")

    assert "SAM 2.1 large" in skill_text
    assert "CPU 仍运行完整模型" in skill_text
    assert "SAM 2.1 Large" in readme_text
    assert "CPU 仍运行完整模型" in readme_text
    assert "SAM 2.1 Large" in readme_en_text
    assert "CPU still runs the full model" in readme_en_text


def test_requirements_keeps_pillow_floor() -> None:
    assert "Pillow>=9.0.0" in REQUIREMENTS.read_text(encoding="utf-8").splitlines()


def test_pyproject_keeps_supported_python_range() -> None:
    assert 'requires-python = ">=3.10,<3.13"' in PYPROJECT.read_text(
        encoding="utf-8"
    ).splitlines()
