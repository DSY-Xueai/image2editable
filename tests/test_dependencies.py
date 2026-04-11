from conftest import load_skill_module


dependencies = load_skill_module("dependencies")


def test_probe_dependencies_reports_runtime_packages():
    result = dependencies.probe_dependencies()
    assert set(result) == {"pymupdf", "pillow", "paddleocr", "python_pptx"}
