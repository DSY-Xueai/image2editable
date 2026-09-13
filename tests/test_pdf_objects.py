from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import FreeText
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
)
from reportlab.pdfgen import canvas

from image2editable.pdf_objects import (
    _normalize_font_name,
    analyze_pdf_document,
    analyze_pdf_page,
)


def _write_pdf(path: Path, draw, *, size=(200, 120)) -> None:
    document = canvas.Canvas(str(path), pagesize=size)
    draw(document)
    document.save()


@pytest.mark.parametrize(
    ("pdf_name", "office_name"),
    [
        ("MicrosoftYaHei", "Microsoft YaHei"),
        ("ArialMT", "Arial"),
        ("Calibri-Light", "Calibri Light"),
    ],
)
def test_pdf_postscript_font_names_map_to_office_families(
    pdf_name: str, office_name: str,
) -> None:
    assert _normalize_font_name(pdf_name) == office_name


def _wrap_page_with_powerpoint_operators(
    source: Path, output: Path, *, alpha: float = 1.0
) -> None:
    reader = PdfReader(source)
    writer = PdfWriter(clone_from=reader)
    page = writer.pages[0]
    resources = page[NameObject("/Resources")].get_object()
    resources[NameObject("/ExtGState")] = DictionaryObject({
        NameObject("/GSsafe"): DictionaryObject({
            NameObject("/Type"): NameObject("/ExtGState"),
            NameObject("/BM"): NameObject("/Normal"),
            NameObject("/ca"): FloatObject(alpha),
            NameObject("/CA"): FloatObject(alpha),
        })
    })
    original = page.get_contents().get_data()
    stream = DecodedStreamObject()
    stream.set_data(
        b"/Artifact <</Type /Pagination>> BDC EMC\n"
        b"/Artifact BMC q /GSsafe gs 0.000014305 0 200 120 re W* n\n"
        + original
        + b"\n0 j 10 M Q EMC\n"
    )
    page.replace_contents(stream)
    with output.open("wb") as target:
        writer.write(target)


def test_text_pdf_is_native_and_keeps_text_position_and_font(tmp_path: Path) -> None:
    source = tmp_path / "text.pdf"

    def draw(document):
        document.setFont("Helvetica", 14)
        document.drawString(20, 70, "Hello")

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "native"
    assert result["requires_visual"] is False
    text = next(item for item in result["objects"] if item["type"] == "text")
    assert text["text"] == "Hello"
    assert text["font_size"] == 14.0
    assert text["font"] == "Helvetica"
    assert text["bbox_pt"][0] == 20.0
    assert text["bbox_pt"][1] < 70.0 < text["bbox_pt"][3]
    assert 51.0 < text["bbox_pt"][2] < 52.0


def test_image_pdf_extracts_xobject_asset_and_transform(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    Image.new("RGB", (10, 20), "green").save(image)
    source = tmp_path / "image.pdf"

    def draw(document):
        document.drawImage(str(image), 40, 20, width=30, height=40)

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "native"
    picture = next(item for item in result["objects"] if item["type"] == "image")
    assert picture["bbox_pt"] == [40.0, 20.0, 70.0, 60.0]
    asset = Path(picture["asset_path"])
    assert asset.is_file()
    with Image.open(asset) as extracted:
        assert extracted.size == (10, 20)


def test_axis_aligned_image_clip_stays_native_with_crop_metadata(
    tmp_path: Path,
) -> None:
    image = tmp_path / "source.png"
    Image.new("RGB", (12, 8), "green").save(image)
    source = tmp_path / "clipped-image.pdf"

    def draw(document):
        document.saveState()
        path = document.beginPath()
        path.rect(40, 20, 25, 40)
        document.clipPath(path, stroke=0, fill=0)
        document.drawImage(str(image), 40, 20, width=30, height=40)
        document.restoreState()

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")
    picture = next(item for item in result["objects"] if item["type"] == "image")

    assert result["classification"] == "native"
    assert picture["bbox_pt"] == [40.0, 20.0, 65.0, 60.0]
    assert picture["crop"] == {
        "left": 0.0,
        "top": 0.0,
        "right": 1 / 6,
        "bottom": 0.0,
    }


def test_rotated_image_requires_visual_without_persisting_asset(
    tmp_path: Path,
) -> None:
    image = tmp_path / "source.png"
    Image.new("RGB", (10, 20), "green").save(image)
    source = tmp_path / "rotated-image.pdf"

    def draw(document):
        document.saveState()
        document.translate(80, 30)
        document.rotate(20)
        document.drawImage(str(image), 0, 0, width=30, height=40)
        document.restoreState()

    _write_pdf(source, draw)
    asset_dir = tmp_path / "assets"
    result = analyze_pdf_page(source, 0, asset_dir=asset_dir)

    assert result["classification"] == "mixed"
    assert result["requires_visual"] is True
    assert not asset_dir.exists()


def test_basic_vector_pdf_extracts_order_color_and_bbox(tmp_path: Path) -> None:
    source = tmp_path / "vector.pdf"

    def draw(document):
        document.setFillColorRGB(1, 0, 0)
        document.rect(30, 20, 50, 20, fill=1, stroke=0)
        document.setStrokeColorRGB(0, 0, 1)
        document.line(100, 20, 150, 80)

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "native"
    shapes = [item for item in result["objects"] if item["type"] == "shape"]
    assert [item["shape_type"] for item in shapes] == ["rectangle", "line"]
    assert shapes[0]["fill_rgb"] == [255, 0, 0]
    assert shapes[0]["bbox_pt"] == [30.0, 20.0, 80.0, 40.0]
    assert shapes[1]["line_start"] == [100.0, 20.0]
    assert shapes[1]["line_end"] == [150.0, 80.0]
    assert [item["z_index"] for item in shapes] == [1, 2]


def test_stroked_rectangle_keeps_fill_stroke_and_width(tmp_path: Path) -> None:
    source = tmp_path / "outlined.pdf"

    def draw(document):
        document.setFillColorRGB(1, 0, 0)
        document.setStrokeColorRGB(0, 0, 1)
        document.setLineWidth(2)
        document.rect(30, 20, 50, 20, fill=1, stroke=1)

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "native"
    shape = next(item for item in result["objects"] if item["type"] == "shape")
    assert shape["fill_rgb"] == [255, 0, 0]
    assert shape["stroke_rgb"] == [0, 0, 255]
    assert shape["line_width"] == 2.0


def test_nondefault_line_join_does_not_downgrade_fill_only_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fill-line-join.pdf"

    def draw(document):
        document._code.append("1 j 30 20 50 20 re f")

    _write_pdf(source, draw)

    result = analyze_pdf_page(source, 0)

    assert result["classification"] == "native"
    assert "line_join" not in result["unsupported_features"]


def test_nondefault_line_join_does_not_downgrade_single_line(
    tmp_path: Path,
) -> None:
    source = tmp_path / "single-line-join.pdf"

    def draw(document):
        document._code.append("1 j 30 20 m 80 40 l S")

    _write_pdf(source, draw)

    result = analyze_pdf_page(source, 0)

    assert result["classification"] == "native"
    assert "line_join" not in result["unsupported_features"]


def test_nondefault_line_join_keeps_stroked_corner_on_visual_route(
    tmp_path: Path,
) -> None:
    source = tmp_path / "outlined-line-join.pdf"

    def draw(document):
        document._code.append("1 j 30 20 50 20 re S")

    _write_pdf(source, draw)

    result = analyze_pdf_page(source, 0)

    assert result["requires_visual"] is True
    assert "line_join" in result["unsupported_features"]


def test_uniformly_scaled_stroke_scales_line_width(tmp_path: Path) -> None:
    source = tmp_path / "scaled-stroke.pdf"

    def draw(document):
        document.saveState()
        document.scale(2, 2)
        document.setLineWidth(3)
        document.rect(10, 10, 20, 10, fill=0, stroke=1)
        document.restoreState()

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "native"
    shape = next(item for item in result["objects"] if item["type"] == "shape")
    assert shape["line_width"] == 6.0


def test_nonuniformly_scaled_stroke_requires_visual_fallback(tmp_path: Path) -> None:
    source = tmp_path / "nonuniform-stroke.pdf"

    def draw(document):
        document.saveState()
        document.scale(2, 3)
        document.setLineWidth(3)
        document.rect(10, 10, 20, 10, fill=0, stroke=1)
        document.restoreState()

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["requires_visual"] is True
    assert "stroke_transform" in result["unsupported_features"]


def test_uniformly_scaled_text_scales_font_size(tmp_path: Path) -> None:
    source = tmp_path / "scaled-text.pdf"

    def draw(document):
        document.saveState()
        document.scale(2, 2)
        document.setFont("Helvetica", 12)
        document.drawString(10, 30, "Scaled")
        document.restoreState()

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "native"
    text = next(item for item in result["objects"] if item["type"] == "text")
    assert text["font_size"] == 24.0


def test_nonuniformly_scaled_text_requires_visual_fallback(tmp_path: Path) -> None:
    source = tmp_path / "nonuniform-text.pdf"

    def draw(document):
        document.saveState()
        document.scale(2, 3)
        document.drawString(10, 30, "Scaled")
        document.restoreState()

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["requires_visual"] is True
    assert "text_transform" in result["unsupported_features"]


def test_non_rectangular_four_point_path_becomes_local_patch(tmp_path: Path) -> None:
    source = tmp_path / "quadrilateral.pdf"

    def draw(document):
        path = document.beginPath()
        path.moveTo(10, 10)
        path.lineTo(60, 10)
        path.lineTo(70, 50)
        path.lineTo(20, 50)
        path.close()
        document.drawPath(path, fill=1, stroke=0)

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    patch = next(item for item in result["objects"] if item["type"] == "patch")
    assert result["classification"] == "hybrid"
    assert result["requires_visual"] is False
    assert result["unsupported_features"] == []
    assert result["localized_features"] == ["complex_path"]
    left, bottom, right, top = patch["bbox_pt"]
    assert (right - left) * (top - bottom) <= 200 * 120 * 0.35


def test_multiple_subpaths_in_one_paint_operation_require_visual(
    tmp_path: Path,
) -> None:
    source = tmp_path / "compound-path.pdf"

    def draw(document):
        document._code.append("10 10 20 10 re 40 10 20 10 re f")

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["requires_visual"] is True
    assert "complex_path" in result["unsupported_features"]


def test_full_page_scan_is_raster_and_requires_visual(tmp_path: Path) -> None:
    image = tmp_path / "scan.png"
    Image.new("RGB", (200, 120), "white").save(image)
    source = tmp_path / "scan.pdf"

    def draw(document):
        document.drawImage(str(image), 0, 0, width=200, height=120)

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "raster"
    assert result["requires_visual"] is True
    assert result["visual_regions"] == [[0.0, 0.0, 200.0, 120.0]]


def test_full_page_scan_does_not_write_unused_pdf_image_asset(
    tmp_path: Path,
) -> None:
    image = tmp_path / "scan.png"
    Image.new("RGB", (200, 120), "white").save(image)
    source = tmp_path / "scan.pdf"

    def draw(document):
        document.drawImage(str(image), 0, 0, width=200, height=120)

    _write_pdf(source, draw)
    asset_dir = tmp_path / "assets"

    result = analyze_pdf_page(source, 0, asset_dir=asset_dir)

    assert result["classification"] == "raster"
    assert not asset_dir.exists()


def test_full_page_scan_with_ocr_text_still_requires_visual(tmp_path: Path) -> None:
    image = tmp_path / "scan.png"
    Image.new("RGB", (200, 120), "white").save(image)
    source = tmp_path / "searchable-scan.pdf"

    def draw(document):
        document.drawImage(str(image), 0, 0, width=200, height=120)
        document.drawString(20, 70, "OCR layer")

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "raster"
    assert result["requires_visual"] is True


def test_visible_pdf_annotation_requires_visual_fallback(tmp_path: Path) -> None:
    plain = tmp_path / "plain.pdf"
    _write_pdf(plain, lambda document: document.drawString(10, 10, "base"))
    source = tmp_path / "annotated.pdf"
    reader = PdfReader(plain)
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.add_annotation(
        page_number=0,
        annotation=FreeText(
            text="Visible note",
            rect=(20, 20, 140, 60),
            font="Helvetica",
            font_size="12pt",
            font_color="ff0000",
        ),
    )
    with source.open("wb") as stream:
        writer.write(stream)

    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["requires_visual"] is True
    assert "annotations" in result["unsupported_features"]


def test_near_full_page_scan_with_small_margin_requires_visual(tmp_path: Path) -> None:
    image = tmp_path / "scan.png"
    Image.new("RGB", (198, 118), "white").save(image)
    source = tmp_path / "inset-scan.pdf"

    def draw(document):
        document.drawImage(str(image), 1, 1, width=198, height=118)
        document.drawString(20, 70, "OCR layer")

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "raster"
    assert result["requires_visual"] is True


def test_unsupported_shading_is_explicitly_recorded(tmp_path: Path) -> None:
    source = tmp_path / "shading.pdf"

    def draw(document):
        document.setFillColorRGB(0, 0, 0)
        document.rect(10, 10, 40, 40, fill=1, stroke=0)
        document._code.append("/Sh1 sh")

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "mixed"
    assert result["requires_visual"] is True
    assert "shading" in result["unsupported_features"]
    assert result["visual_regions"] == [[0.0, 0.0, 200.0, 120.0]]


def test_powerpoint_metadata_page_clip_and_identity_state_are_native(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain.pdf"
    wrapped = tmp_path / "wrapped.pdf"
    _write_pdf(plain, lambda document: document.drawString(20, 70, "Hello"))
    _wrap_page_with_powerpoint_operators(plain, wrapped)

    result = analyze_pdf_page(wrapped, 0)

    assert result["classification"] == "native"
    assert result["unsupported_features"] == []


def test_nonopaque_graphics_state_stays_visual(tmp_path: Path) -> None:
    plain = tmp_path / "plain.pdf"
    wrapped = tmp_path / "transparent.pdf"
    _write_pdf(plain, lambda document: document.rect(10, 10, 40, 40, fill=1))
    _wrap_page_with_powerpoint_operators(plain, wrapped, alpha=0.5)

    result = analyze_pdf_page(wrapped, 0)

    shape = next(item for item in result["objects"] if item["type"] == "shape")

    assert result["classification"] == "native"
    assert result["requires_visual"] is False
    assert result["unsupported_features"] == []
    assert shape["fill_opacity"] == pytest.approx(0.5)


def test_transparent_mirrored_image_stays_native(tmp_path: Path) -> None:
    image = tmp_path / "transparent.png"
    pixels = Image.new("RGBA", (12, 8), (255, 0, 0, 0))
    pixels.putpixel((0, 0), (0, 255, 0, 255))
    pixels.save(image)
    source = tmp_path / "mirrored.pdf"

    def draw(document):
        document.saveState()
        document.translate(80, 20)
        document.scale(-1, 1)
        document.drawImage(
            str(image), 0, 0, width=40, height=30, mask="auto"
        )
        document.restoreState()

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")
    picture = next(item for item in result["objects"] if item["type"] == "image")

    assert result["classification"] == "native"
    assert result["unsupported_features"] == []
    assert picture["flip_horizontal"] is True
    with Image.open(picture["asset_path"]) as extracted:
        assert extracted.mode == "RGBA"
        assert extracted.getpixel((extracted.width - 1, 0)) == (0, 255, 0, 255)


def test_text_form_requires_editable_reconstruction(tmp_path: Path) -> None:
    source = tmp_path / "formula-form.pdf"

    def draw(document):
        document.drawString(20, 95, "Editable heading")
        document.beginForm("formula", 0, 0, 80, 20)
        document.setFont("Times-Roman", 16)
        document.drawString(0, 2, "a / b = c")
        document.endForm()
        document.saveState()
        document.translate(60, 35)
        document.doForm("formula")
        document.restoreState()

    _write_pdf(source, draw, size=(200, 120))
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")
    assert result["classification"] == "native"
    assert result["requires_visual"] is False
    assert result["unsupported_features"] == []
    formula = next(item for item in result["objects"] if item.get("text") == "a / b = c")
    assert formula["matrix"][4:] == [60.0, 37.0]
    assert not any(item["type"] == "patch" for item in result["objects"])
    assert any(
        item["type"] == "text" and item["text"] == "Editable heading"
        for item in result["objects"]
    )


def test_near_rectangular_frame_form_becomes_native_bars(tmp_path: Path) -> None:
    source = tmp_path / "frame-form.pdf"

    def draw(document):
        document.beginForm("frame", 0, 0, 100, 100)
        document._code.append("1 .737 .051 rg")
        document._code.append(
            "0 100 m 0 0.08 l 100 0 l 100 100 l h "
            "3 3 m 3 97 l 97 97 l 97 3 l h f"
        )
        document.endForm()
        document.saveState()
        document.translate(20, 20)
        document.scale(6.25, 6.25)
        document.doForm("frame")
        document.restoreState()

    _write_pdf(source, draw, size=(1000, 700))
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    shapes = [item for item in result["objects"] if item["type"] == "shape"]
    assert result["classification"] == "native"
    assert result["requires_visual"] is False
    assert result["unsupported_features"] == []
    assert len(shapes) == 4
    assert all(item["shape_type"] == "rectangle" for item in shapes)


def test_clockwise_axis_aligned_clip_is_not_treated_as_complex(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    Image.new("RGB", (12, 8), "green").save(image)
    source = tmp_path / "clockwise-clip.pdf"

    def draw(document):
        document._code.append(
            "0 120 m 200 120 l 200 0 l 0 0 l W* n"
        )
        document.drawImage(str(image), 40, 20, width=30, height=40)

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "native"
    assert result["unsupported_features"] == []


def test_axis_aligned_clip_tolerates_pdf_coordinate_noise(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    Image.new("RGB", (12, 8), "green").save(image)
    source = tmp_path / "noisy-clip.pdf"

    def draw(document):
        document._code.append(
            "0 120 m 200 120 l 200 -0.00003 l -0.00006 0 l W* n"
        )
        document.drawImage(str(image), 40, 20, width=30, height=40)

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["classification"] == "native"
    assert result["unsupported_features"] == []


def test_character_spacing_is_preserved_as_native_metadata(tmp_path: Path) -> None:
    source = tmp_path / "spacing.pdf"

    def draw(document):
        document._code.append("BT /F1 12 Tf 2 Tc 1 0 0 1 20 70 Tm (Wide) Tj ET")

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    text = next(item for item in result["objects"] if item["type"] == "text")
    assert result["classification"] == "native"
    assert text["character_spacing_pt"] == 2.0


def test_nonzero_word_spacing_requires_visual_fallback(tmp_path: Path) -> None:
    source = tmp_path / "word-spacing.pdf"

    def draw(document):
        document._code.append("BT /F1 12 Tf 2 Tw 1 0 0 1 20 70 Tm (Wide words) Tj ET")

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["requires_visual"] is True
    assert "word_spacing" in result["unsupported_features"]


def test_unmappable_text_cannot_be_delivered_as_a_patch(tmp_path: Path) -> None:
    source = tmp_path / "undecodable.pdf"

    def draw(document):
        document._code.append("BT /F1 12 Tf 1 0 0 1 20 70 Tm <0000> Tj ET")

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    assert result["requires_visual"] is True
    assert "text_decode" in result["unsupported_features"]
    assert not any(item["type"] == "patch" for item in result["objects"])


def test_scaled_text_form_preserves_metrics_and_editability(tmp_path: Path) -> None:
    source = tmp_path / "scaled-form.pdf"

    def draw(document):
        document.beginForm("label", 0, 0, 80, 30)
        document.setFont("Helvetica", 12)
        document.drawString(2, 8, "Editable")
        document.endForm()
        document.translate(30, 20)
        document.scale(1.5, 1.5)
        document.doForm("label")

    _write_pdf(source, draw)
    result = analyze_pdf_page(source, 0)
    assert result["classification"] == "native"
    text = result["objects"][0]
    assert text["text"] == "Editable"
    assert text["font_size"] == 18
    assert text["matrix"][4:] == [33, 32]
    assert 32 < text["bbox_pt"][0] < 35
    assert 30 < text["bbox_pt"][1] < 34


def test_result_is_json_serializable_without_inline_binary(tmp_path: Path) -> None:
    source = tmp_path / "text.pdf"
    _write_pdf(source, lambda document: document.drawString(10, 10, "ok"))
    result = analyze_pdf_page(source, 0, asset_dir=tmp_path / "assets")

    json.dumps(result, ensure_ascii=False)
    assert all("data" not in item for item in result["objects"])


def test_document_analysis_opens_pdf_once(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "two-pages.pdf"
    document = canvas.Canvas(str(source), pagesize=(200, 120))
    document.drawString(10, 80, "one")
    document.showPage()
    document.drawString(10, 80, "two")
    document.save()

    import image2editable.pdf_objects as module

    original_reader = module.PdfReader
    calls = []

    def counted_reader(*args, **kwargs):
        calls.append(args[0])
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(module, "PdfReader", counted_reader)

    results = analyze_pdf_document(source, asset_root=tmp_path / "assets")

    assert len(results) == 2
    assert calls == [str(source.resolve())]


def test_text_only_analysis_does_not_create_empty_asset_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "text.pdf"
    _write_pdf(source, lambda document: document.drawString(10, 10, "ok"))
    asset_dir = tmp_path / "assets"

    analyze_pdf_page(source, 0, asset_dir=asset_dir)

    assert not asset_dir.exists()
