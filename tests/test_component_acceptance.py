import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "component_acceptance_manifest.json"
REQUIRED_CONTENT_TYPES = {
    "photo",
    "person",
    "poster",
    "card",
    "ui",
    "table",
    "chart",
    "flowchart",
    "scientific_figure",
    "formula",
    "dense_connectors",
    "map",
    "illustration",
    "icon_group",
    "low_contrast",
    "gradient",
    "transparency",
    "shadow",
    "antialiasing",
}
REQUIRED_INPUT_TYPES = {"image", "pdf", "image_pptx", "mixed_native_pptx"}


@pytest.mark.parametrize(
    "path",
    ["README.md", "README_EN.md", "skills/image-to-ppt/SKILL.md"],
)
def test_docs_describe_both_providers_and_safety_limits(path: str) -> None:
    text = (ROOT / path).read_text(encoding="utf-8").lower()

    assert "--agent-provider host" in text
    assert "--agent-provider local" in text
    assert "preserved_with_warning" in text
    assert "experimental" in text
    assert "5" in text or "five" in text or "五" in text


def test_acceptance_manifest_covers_generic_content_and_input_types() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert REQUIRED_CONTENT_TYPES <= set(manifest["content_types"])
    assert REQUIRED_INPUT_TYPES <= set(manifest["input_types"])
    assert manifest["semantic_decision_cache"] is False
    assert manifest["max_repair_rounds_per_page"] == 5


def test_skill_limits_absorb_to_one_physical_entity() -> None:
    text = (ROOT / "skills/image-to-ppt/SKILL.md").read_text(encoding="utf-8")

    assert "同一物理实体" in text
    assert "重复掩码" in text
    assert "碎边" in text
    assert "阴影" in text
    assert "分割缺口" in text
    assert "语义父级只用于分组，不参与最终像素渲染" in text
