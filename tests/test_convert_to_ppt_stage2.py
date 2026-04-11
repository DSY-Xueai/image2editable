from pathlib import Path

from PIL import Image

from conftest import load_skill_module


convert_module = load_skill_module("convert_to_ppt")


def test_convert_to_ppt_supports_stage2_flag(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (50, 50), "white").save(image_path)
    output_path = tmp_path / "result.pptx"
    convert_module.convert_to_ppt(str(image_path), output_path, enable_stage2=True)
    assert output_path.exists()
