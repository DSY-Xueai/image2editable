from pathlib import Path
from types import ModuleType, SimpleNamespace

from conftest import load_skill_module


models = load_skill_module("models")
build_ppt = load_skill_module("build_ppt")

PagePlan = models.PagePlan
TextBlock = models.TextBlock
ImageBlock = models.ImageBlock
build_presentation = build_ppt.build_presentation


def test_build_presentation_rejects_missing_backgrounds_when_python_pptx_is_available(
    monkeypatch, tmp_path
):
    fake_pptx = ModuleType("pptx")

    class FakeSlides:
        def __init__(self):
            self.added = []

        def add_slide(self, layout):
            slide = SimpleNamespace(shapes=SimpleNamespace(add_picture=lambda *args, **kwargs: None))
            self.added.append((layout, slide))
            return slide

    class FakePresentation:
        def __init__(self):
            self.slides = FakeSlides()
            self.slide_layouts = [object() for _ in range(7)]
            self.saved_to = None

        def save(self, output_path):
            self.saved_to = output_path

    fake_pptx.Presentation = FakePresentation

    monkeypatch.setattr(build_ppt.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(build_ppt.importlib, "import_module", lambda name: fake_pptx)

    page = PagePlan(
        page_number=1,
        width_px=100,
        height_px=200,
        background_path=str(tmp_path / "missing.png"),
    )

    try:
        build_presentation([page], tmp_path / "result.pptx")
    except FileNotFoundError as exc:
        assert "missing.png" in str(exc)
    else:
        raise AssertionError("Expected missing background validation to raise FileNotFoundError")


def test_build_presentation_uses_python_pptx_for_background_first_output(
    monkeypatch, tmp_path
):
    background = tmp_path / "page.png"
    background.write_bytes(b"fake image")

    call_order = []

    class FakeShapes:
        def add_picture(self, path, left, top, width=None, height=None):
            call_order.append(("picture", Path(path).name, left, top, width, height))

        def add_textbox(self, left, top, width, height):
            call_order.append(("textbox", left, top, width, height))
            return SimpleNamespace(text_frame=SimpleNamespace(text=""))

    class FakeSlide:
        def __init__(self):
            self.shapes = FakeShapes()

    class FakeSlides:
        def __init__(self):
            self.added = []

        def add_slide(self, layout):
            slide = FakeSlide()
            self.added.append((layout, slide))
            return slide

    class FakePresentation:
        def __init__(self):
            self.slides = FakeSlides()
            self.slide_layouts = [object() for _ in range(7)]
            self.saved_to = None

        def save(self, output_path):
            self.saved_to = output_path

    fake_pptx = ModuleType("pptx")
    fake_pptx.Presentation = FakePresentation

    page = PagePlan(
        page_number=1,
        width_px=400,
        height_px=300,
        background_path=str(background),
    )
    page.text_blocks.append(
        TextBlock("Hello", 10, 20, 30, 40, 12, "#000000", "left", 0.9)
    )
    page.image_blocks.append(ImageBlock(str(background), 50, 60, 70, 80, 0.9, True))

    monkeypatch.setattr(build_ppt.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(build_ppt.importlib, "import_module", lambda name: fake_pptx)

    build_presentation([page], tmp_path / "result.pptx")

    assert call_order[0][0] == "picture"
    assert call_order[1][0] == "textbox"
    assert call_order[2][0] == "picture"


def test_build_presentation_falls_back_to_minimal_writer_when_python_pptx_is_missing(
    monkeypatch, tmp_path
):
    background = tmp_path / "page.png"
    background.write_bytes(b"fake image")

    recorded = {}

    def fake_minimal_writer(output_path, slide_count):
        recorded["output_path"] = output_path
        recorded["slide_count"] = slide_count

    monkeypatch.setattr(build_ppt.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(build_ppt, "_build_minimal_pptx", fake_minimal_writer)

    page = PagePlan(
        page_number=1,
        width_px=100,
        height_px=200,
        background_path=str(background),
    )

    build_presentation([page], tmp_path / "result.pptx")

    assert recorded["slide_count"] == 1
    assert recorded["output_path"] == tmp_path / "result.pptx"
