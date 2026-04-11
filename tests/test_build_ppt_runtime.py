from pathlib import Path

import pytest

from conftest import load_skill_module


models = load_skill_module("models")
build_ppt = load_skill_module("build_ppt")
PagePlan = models.PagePlan


def test_build_presentation_rejects_missing_background(tmp_path):
    page = PagePlan(page_number=1, width_px=100, height_px=200, background_path="missing.png")
    output_path = tmp_path / "result.pptx"
    with pytest.raises(FileNotFoundError):
        build_ppt.build_presentation([page], output_path)
