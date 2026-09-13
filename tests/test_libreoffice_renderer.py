import json
import os
import sys

from PIL import Image
from pptx import Presentation
from reportlab.pdfgen import canvas
import pytest

from image2editable import libreoffice_renderer as module
from image2editable.powerpoint_renderer import RendererUnavailable


def test_portable_office_discovery_does_not_replace_python_path(tmp_path, monkeypatch):
    executable = tmp_path / 'portable office' / 'soffice.com'
    executable.parent.mkdir()
    executable.touch()
    original_path = os.environ.get('PATH')
    monkeypatch.setenv('IMAGE2EDITABLE_LIBREOFFICE', str(executable))
    assert module.LibreOfficeRenderer.discover().executable == str(executable)
    assert os.environ.get('PATH') == original_path
    monkeypatch.setenv('IMAGE2EDITABLE_LIBREOFFICE', 'relative/soffice.com')
    assert not module.LibreOfficeRenderer.discover().available()
    monkeypatch.setenv('IMAGE2EDITABLE_LIBREOFFICE', str(tmp_path / 'missing.com'))
    assert not module.LibreOfficeRenderer.discover().available()


def test_libreoffice_exports_selected_page_and_cleans_short_profile(tmp_path, monkeypatch):
    source = tmp_path / "input.pptx"
    Presentation().save(source)
    output = tmp_path / ("deep-output-" * 8) / "page.png"
    temporary = []

    def export(command, log_path):
        root = log_path.parent
        temporary.append(root)
        assert root.parent != output.parent
        selection = command[command.index("--convert-to") + 1].split(":", 2)[2]
        assert json.loads(selection)["PageRange"]["value"] == "2"
        assert (root / "input.pptx").read_bytes() == source.read_bytes()
        assert (root / "profile").as_uri() in command[1]
        pdf = canvas.Canvas(str(root / "input.pdf"), pagesize=(100, 60))
        pdf.setFillColorRGB(0, 0, 1)
        pdf.rect(0, 0, 100, 60, fill=1, stroke=0)
        pdf.save()

    monkeypatch.setattr(module, "_run_office", export)
    result = module.LibreOfficeRenderer("test-office").render_page(
        source, 2, output, width=300, height=180,
    )
    with Image.open(output) as image:
        assert image.size == (300, 180)
        assert image.getpixel((150, 90)) == (0, 0, 255)
    assert result["renderer"] == "libreoffice"
    assert all(not path.exists() for path in temporary)


def test_libreoffice_failure_never_publishes_a_render(tmp_path, monkeypatch):
    source, output = tmp_path / "input.pptx", tmp_path / "page.png"
    Presentation().save(source)
    temporary = []

    def no_pdf(command, log_path):
        temporary.append(log_path.parent)

    monkeypatch.setattr(module, "_run_office", no_pdf)
    with pytest.raises(RuntimeError, match="did not produce"):
        module.LibreOfficeRenderer("test-office").render_page(source, 1, output, width=100, height=60)
    assert not output.exists()
    assert all(not path.exists() for path in temporary)
    with pytest.raises(RendererUnavailable):
        module.LibreOfficeRenderer(None).render_page(source, 1, output, width=100, height=60)


def test_office_process_waits_for_its_child_after_launcher_exit(tmp_path):
    signal = tmp_path / "child-finished"
    child = "import pathlib,time,sys; time.sleep(1.0); pathlib.Path(sys.argv[1]).touch()"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(.4)"
    module._run_office([sys.executable, "-c", parent, child, str(signal)], tmp_path / "render.log")
    assert signal.is_file()
