from pathlib import Path

import pytest

from conftest import load_skill_module


convert_to_ppt = load_skill_module("convert_to_ppt").convert_to_ppt


def test_convert_to_ppt_rejects_missing_input(tmp_path):
    output_path = tmp_path / "result.pptx"
    with pytest.raises(FileNotFoundError):
        convert_to_ppt("missing.pdf", output_path)
