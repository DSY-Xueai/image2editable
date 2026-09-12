from pathlib import Path

from PIL import ImageFont
import pytest
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont


@pytest.fixture(scope="session")
def art_font_path(tmp_path_factory):
    source = Path(__file__).resolve().parents[1] / "benchmarks/release/fonts/NotoSansSC[wght].ttf"
    path = tmp_path_factory.mktemp("art-font") / "NotoSansSC-Bold.ttf"
    with TTFont(source) as font:
        bold = instantiateVariableFont(font, {"wght": 700}, updateFontNames=True)
        bold.save(path)
    return path


@pytest.fixture
def art_font(monkeypatch, art_font_path):
    """Use the licensed repository font without relying on OS font installs."""
    from scripts import font_match
    from scripts import ppt_assemble

    path = art_font_path
    face = (ImageFont.truetype(str(path), 32).getname()[0], True, False, str(path), 0)
    monkeypatch.setattr(font_match, "installed_faces", lambda: (face,))
    font_match.resolve_font.cache_clear()
    ppt_assemble._ocr_measurement_font.cache_clear()
    yield lambda size: ImageFont.truetype(str(path), size)
    font_match.resolve_font.cache_clear()
    ppt_assemble._ocr_measurement_font.cache_clear()
