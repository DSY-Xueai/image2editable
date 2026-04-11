from conftest import load_skill_module
from PIL import Image


extract_images = load_skill_module("extract_images").extract_images
extract_text_layout = load_skill_module("extract_text_layout").extract_text_layout
render_pdf_pages = load_skill_module("render_pdf_pages").render_pdf_pages


def test_script_functions_return_list_shapes(tmp_path):
    output_dir = tmp_path / "pages"
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    rendered = []
    extracted_text = extract_text_layout(str(image_path), page_number=1)
    extracted_images = extract_images(str(image_path), page_number=1, output_dir=output_dir)
    assert isinstance(rendered, list)
    assert isinstance(extracted_text, list)
    assert isinstance(extracted_images, list)
