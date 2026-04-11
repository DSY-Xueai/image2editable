from pathlib import Path


def test_skill_scaffold_exists():
    root = Path("skills/pdf-image-to-editable-ppt")
    assert (root / "SKILL.md").exists()
    assert (root / "scripts" / "__init__.py").exists()
    assert (root / "references" / "README.md").exists()
