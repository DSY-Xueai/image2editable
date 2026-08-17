from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import re
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image, ImageChops, UnidentifiedImageError
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "benchmarks" / "release"
MANIFEST_PATH = RELEASE_ROOT / "manifest.json"

ROOT_FIELDS = {"schema_version", "cases", "categories"}
CASE_FIELDS = {
    "id",
    "kind",
    "path",
    "sha256",
    "source",
    "license",
    "page_count",
    "categories",
    "agent_provider",
    "pptx_mode",
    "expected_pages",
}
PAGE_FIELDS = {
    "page_id",
    "expected_status",
    "min_components",
    "min_text_boxes",
    "max_unexplained_pixels",
    "max_quality_violations",
}

CASE_SPECS = [
    (
        "image-bilingual-dashboard",
        "image",
        "inputs/01-bilingual-dashboard.png",
        1,
        "bilingual_dashboard",
        None,
        12,
        8,
    ),
    (
        "image-dense-parameter-comparison",
        "image",
        "inputs/02-dense-parameter-comparison.png",
        1,
        "dense_parameter_comparison",
        None,
        15,
        12,
    ),
    (
        "image-profile-cards",
        "image",
        "inputs/03-profile-cards.png",
        1,
        "profile_cards",
        None,
        10,
        8,
    ),
    (
        "image-four-stage-timeline",
        "image",
        "inputs/04-four-stage-timeline.png",
        1,
        "four_stage_timeline",
        None,
        10,
        6,
    ),
    (
        "image-combo-chart",
        "image",
        "inputs/05-combo-chart.png",
        1,
        "combo_chart",
        None,
        8,
        5,
    ),
    (
        "image-flowchart",
        "image",
        "inputs/06-flowchart.png",
        1,
        "flowchart",
        None,
        10,
        6,
    ),
    (
        "image-icon-matrix",
        "image",
        "inputs/07-icon-matrix.png",
        1,
        "icon_matrix",
        None,
        12,
        4,
    ),
    (
        "image-light-text-gradient",
        "image",
        "inputs/08-light-text-gradient.png",
        1,
        "light_text_gradient",
        None,
        4,
        3,
    ),
    (
        "image-thin-line-network",
        "image",
        "inputs/09-thin-line-network.png",
        1,
        "thin_line_network",
        None,
        12,
        3,
    ),
    (
        "image-tiny-element-table",
        "image",
        "inputs/10-tiny-element-table.png",
        1,
        "tiny_element_table",
        None,
        15,
        10,
    ),
    (
        "image-dark-poster",
        "image",
        "inputs/11-dark-poster.png",
        1,
        "dark_poster",
        None,
        5,
        4,
    ),
    (
        "image-non-16-9-infographic",
        "image",
        "inputs/12-non-16-9-infographic.png",
        1,
        "non_16_9_infographic",
        None,
        12,
        8,
    ),
    (
        "pdf-mixed-page-sizes",
        "pdf",
        "inputs/13-mixed-page-sizes.pdf",
        2,
        "pdf_mixed_page_sizes",
        None,
        6,
        4,
    ),
    (
        "pdf-rotated-page",
        "pdf",
        "inputs/14-rotated-page.pdf",
        2,
        "pdf_rotated_page",
        None,
        6,
        4,
    ),
    ("pdf-high-dpi", "pdf", "inputs/15-high-dpi.pdf", 2, "pdf_high_dpi", None, 6, 4),
    (
        "pptx-image-only",
        "pptx",
        "inputs/16-image-only.pptx",
        4,
        "pptx_image_only",
        "image_only",
        6,
        4,
    ),
    (
        "pptx-mixed-native",
        "pptx",
        "inputs/17-mixed-native.pptx",
        4,
        "pptx_mixed_native",
        "mixed_native",
        6,
        4,
    ),
    (
        "pptx-mixed-screenshot-candidates",
        "pptx",
        "inputs/18-mixed-screenshot-candidates.pptx",
        4,
        "pptx_mixed_screenshot_candidates",
        "mixed_screenshot_candidates",
        6,
        4,
    ),
]

EXPECTED_CATEGORIES = [spec[4] for spec in CASE_SPECS]
EXPECTED_README = """# 发布质量语料契约

本目录包含固定发布 benchmark 语料：18 个输入、30 页，包括 12 张图片、3 个双页 PDF、3 个四页 PPTX。

## 覆盖范围

图片固定覆盖：中英双语仪表盘、密集参数对比、人物信息卡、四段时间线、柱线组合图、流程图、图标矩阵、浅色文字渐变页、细线网络图、小元素表格、深色海报、非 16:9 信息图。

PDF 分别覆盖双页不同尺寸、旋转、高 DPI。PPTX 分别覆盖 `image_only`、`mixed_native`、`mixed_screenshot_candidates`。

## 来源与许可

输入由项目脚本公开生成，manifest 固定记录 `source=project-generated`、`license=CC0-1.0`。所有 case 默认使用已支持的 `agent_provider=host`。

## 字体来源与许可

生成器仅使用仓库内完整、未修改的 Google Fonts `Noto Sans SC` variable TTF；常规文本固定选择 `Regular`，既有粗体文本选择同一文件中的 `Bold` 实例，文件位于 `fonts/NotoSansSC[wght].ttf`。字体按 `fonts/OFL.txt` 中的 SIL Open Font License 1.1 分发，固定来源为 `google/fonts` commit `e1118da94a8cb00cf6d06cdac9ef13eb1e5c6ab7`；字体本身不是 CC0。字体渲染所得 benchmark 输入继续按 CC0-1.0 发布，manifest 中的许可字段不适用于捆绑字体文件。

## 阶段状态

当前目录已包含全部 18 个输入文件，仓库 canonical bytes 由 manifest 中的真实 `sha256` 绑定。可用 `python scripts/build_release_corpus.py <output-root>` 在不存在的目录重新生成语料；同一环境的两次 fresh generation 要求 PNG RGB 像素一致，fresh 与仓库 canonical 只比较格式、尺寸、页数和对象 inventory 等明确语义。跨平台同样只承诺这些语义等价，不承诺 PNG 像素或 PDF/PPTX 字节完全相同。本阶段不包含 runner。

## 通过标准

每页必须达到 manifest 中的最小组件数和文本框数、状态为 `validated`，并同时满足 0 warning、0 unexplained pixels、0 quality violations。后续 runner 对全部 30 页执行 `repeat=3`；runner 和 CI 接入不属于本阶段。
"""

EXPECTED_INPUT_NAMES = {Path(spec[2]).name for spec in CASE_SPECS}
EXPECTED_FONT_PATH = RELEASE_ROOT / "fonts" / "NotoSansSC[wght].ttf"
EXPECTED_FONT_SHA256 = "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da"
EXPECTED_IMAGE_SIZES = {
    "01-bilingual-dashboard.png": (1600, 900),
    "02-dense-parameter-comparison.png": (1600, 900),
    "03-profile-cards.png": (1600, 900),
    "04-four-stage-timeline.png": (1600, 900),
    "05-combo-chart.png": (1600, 900),
    "06-flowchart.png": (1600, 900),
    "07-icon-matrix.png": (1600, 900),
    "08-light-text-gradient.png": (1600, 900),
    "09-thin-line-network.png": (1600, 900),
    "10-tiny-element-table.png": (1600, 900),
    "11-dark-poster.png": (1600, 900),
    "12-non-16-9-infographic.png": (1000, 1400),
}


@pytest.fixture(scope="module")
def fresh_release_corpora(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    builder = importlib.import_module("scripts.build_release_corpus")
    parent = tmp_path_factory.mktemp("release-corpus")
    first = parent / "first"
    second = parent / "second"

    builder.build(first)
    builder.build(second)

    return first, second


def _pptx_inventory(path: Path) -> tuple[tuple[tuple[int, str], ...], ...]:
    deck = Presentation(path)
    return tuple(
        tuple((int(shape.shape_type), shape.text if shape.has_text_frame else "") for shape in slide.shapes)
        for slide in deck.slides
    )


def _pdf_inventory(path: Path) -> tuple[tuple[float, float, int], ...]:
    return tuple(
        (
            round(float(page.mediabox.width), 3),
            round(float(page.mediabox.height), 3),
            page.rotation,
        )
        for page in PdfReader(path).pages
    )


def _assert_same_rgb_pixels(first: Image.Image, second: Image.Image) -> None:
    assert ImageChops.difference(
        first.convert("RGB"), second.convert("RGB")
    ).getbbox() is None


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _manifest() -> dict[str, object]:
    return json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _assert_exact_fields(manifest: dict[str, object]) -> None:
    assert set(manifest) == ROOT_FIELDS
    for case in manifest["cases"]:
        assert set(case) == CASE_FIELDS
        for page in case["expected_pages"]:
            assert set(page) == PAGE_FIELDS


def _assert_numeric_contract(manifest: dict[str, object]) -> None:
    assert type(manifest["schema_version"]) is int
    assert manifest["schema_version"] == 1
    for case in manifest["cases"]:
        assert type(case["page_count"]) is int
        assert case["page_count"] > 0
        for page in case["expected_pages"]:
            for field in (
                "min_components",
                "min_text_boxes",
                "max_unexplained_pixels",
                "max_quality_violations",
            ):
                assert type(page[field]) is int
                assert page[field] >= 0


def _assert_release_input(case: dict[str, object], release_root: Path) -> None:
    path = release_root / case["path"]
    assert path.is_file(), path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert case["sha256"] == digest
    assert digest != "0" * 64

    if case["kind"] == "image":
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.n_frames == case["page_count"]
            image.verify()
    elif case["kind"] == "pdf":
        assert len(PdfReader(path).pages) == case["page_count"]
    elif case["kind"] == "pptx":
        assert len(Presentation(path).slides) == case["page_count"]
    else:
        raise AssertionError(f"unsupported release input kind: {case['kind']}")


def test_release_manifest_has_exact_root_schema_and_categories() -> None:
    manifest = _manifest()

    _assert_exact_fields(manifest)
    _assert_numeric_contract(manifest)
    assert manifest["schema_version"] == 1
    assert manifest["categories"] == EXPECTED_CATEGORIES
    assert len(manifest["cases"]) == 18
    assert len({case["id"] for case in manifest["cases"]}) == 18


def test_release_manifest_defines_eighteen_inputs_and_thirty_pages() -> None:
    manifest = _manifest()
    cases = manifest["cases"]

    assert [
        (
            case["id"],
            case["kind"],
            case["path"],
            case["page_count"],
            case["categories"][0],
            case["pptx_mode"],
        )
        for case in cases
    ] == [spec[:6] for spec in CASE_SPECS]
    assert {
        kind: sum(case["kind"] == kind for case in cases)
        for kind in ("image", "pdf", "pptx")
    } == {
        "image": 12,
        "pdf": 3,
        "pptx": 3,
    }
    assert sum(case["page_count"] for case in cases) == 30


def test_release_manifest_cases_and_pages_use_exact_strict_fields() -> None:
    cases = _manifest()["cases"]

    for case, spec in zip(cases, CASE_SPECS, strict=True):
        identifier, _, _, page_count, category, _, min_components, min_text_boxes = spec
        assert re.fullmatch(r"[0-9a-f]{64}", case["sha256"])
        assert case["source"] == "project-generated"
        assert case["license"] == "CC0-1.0"
        assert case["categories"] == [category]
        assert case["agent_provider"] == "host"
        assert len(case["expected_pages"]) == page_count

        for page_number, page in enumerate(case["expected_pages"], start=1):
            assert page == {
                "page_id": f"{identifier}-page-{page_number:03d}",
                "expected_status": "validated",
                "min_components": min_components,
                "min_text_boxes": min_text_boxes,
                "max_unexplained_pixels": 0,
                "max_quality_violations": 0,
            }


def test_release_manifest_pptx_modes_are_exact_and_only_apply_to_pptx() -> None:
    cases = _manifest()["cases"]
    pptx_cases = [case for case in cases if case["kind"] == "pptx"]

    assert {case["pptx_mode"] for case in pptx_cases} == {
        "image_only",
        "mixed_native",
        "mixed_screenshot_candidates",
    }
    assert all(case["pptx_mode"] is None for case in cases if case["kind"] != "pptx")


@pytest.mark.parametrize(
    ("needle", "duplicate"),
    [
        ('  "schema_version": 1,', '  "schema_version": 1,\n  "schema_version": 1,'),
        (
            '      "id": "image-bilingual-dashboard",',
            '      "id": "image-bilingual-dashboard",\n'
            '      "id": "image-bilingual-dashboard",',
        ),
        (
            '          "page_id": "image-bilingual-dashboard-page-001",',
            '          "page_id": "image-bilingual-dashboard-page-001",\n'
            '          "page_id": "image-bilingual-dashboard-page-001",',
        ),
    ],
)
def test_release_manifest_rejects_duplicate_keys_in_every_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    needle: str,
    duplicate: str,
) -> None:
    original = MANIFEST_PATH.read_text(encoding="utf-8")
    mutant = original.replace(needle, duplicate, 1)
    assert mutant != original
    mutant_path = tmp_path / "manifest.json"
    mutant_path.write_text(mutant, encoding="utf-8")
    monkeypatch.setitem(globals(), "MANIFEST_PATH", mutant_path)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        _manifest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("page_count", True),
        ("page_count", 0),
        ("min_components", True),
        ("min_components", -1),
        ("min_text_boxes", True),
        ("min_text_boxes", -1),
        ("max_unexplained_pixels", False),
        ("max_unexplained_pixels", -1),
        ("max_quality_violations", False),
        ("max_quality_violations", -1),
    ],
)
def test_release_manifest_rejects_bool_and_out_of_range_integers(
    field: str, value: object
) -> None:
    mutant = copy.deepcopy(_manifest())
    if field == "schema_version":
        mutant[field] = value
    elif field == "page_count":
        mutant["cases"][0][field] = value
    else:
        mutant["cases"][0]["expected_pages"][0][field] = value

    with pytest.raises(AssertionError):
        _assert_numeric_contract(mutant)


def test_release_manifest_object_order_is_not_schema() -> None:
    mutant = copy.deepcopy(_manifest())
    mutant["cases"][0]["expected_pages"][0] = dict(
        reversed(mutant["cases"][0]["expected_pages"][0].items())
    )
    mutant["cases"][0] = dict(reversed(mutant["cases"][0].items()))
    mutant = dict(reversed(mutant.items()))

    _assert_exact_fields(mutant)


def test_release_readme_documents_phase_one_and_future_strict_run() -> None:
    readme = (RELEASE_ROOT / "README.md").read_text(encoding="utf-8")

    assert readme == EXPECTED_README


def test_release_binary_inputs_have_git_attributes() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    ignores = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "benchmarks/release/inputs/*.png binary" in attributes
    assert "benchmarks/release/inputs/*.pdf binary" in attributes
    assert "benchmarks/release/inputs/*.pptx binary" in attributes
    assert "benchmarks/release/fonts/*.ttf binary" in attributes
    assert "!benchmarks/release/inputs/*.png" in ignores
    assert "!benchmarks/release/inputs/*.pdf" in ignores
    assert "!benchmarks/release/inputs/*.pptx" in ignores


def test_rgb_pixel_comparison_rejects_red_vs_blue_mutation() -> None:
    red = Image.new("RGB", (2, 2), "red")
    blue = Image.new("RGB", (2, 2), "blue")

    with pytest.raises(AssertionError):
        _assert_same_rgb_pixels(red, blue)


def test_release_builder_uses_only_bundled_regular_noto_sans_sc() -> None:
    builder = importlib.import_module("scripts.build_release_corpus")

    assert builder.FONT_PATH == EXPECTED_FONT_PATH
    assert builder.FONT_VARIATION == "Regular"
    assert builder.FONT_PATH.is_file()
    assert builder.FONT_PATH.with_name("OFL.txt").is_file()
    assert hashlib.sha256(builder.FONT_PATH.read_bytes()).hexdigest() == EXPECTED_FONT_SHA256
    source = inspect.getsource(builder._font)
    assert "FONT_PATH" in source
    assert "Windows\\Fonts" not in source
    assert "DejaVuSans" not in source
    assert "load_default" not in source


def test_release_builder_fails_fast_when_bundled_font_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = importlib.import_module("scripts.build_release_corpus")
    monkeypatch.setattr(builder, "FONT_PATH", tmp_path / "missing.ttf")

    with pytest.raises(FileNotFoundError):
        builder._font(24)


def test_bundled_font_has_distinct_cjk_glyphs_without_notdef() -> None:
    builder = importlib.import_module("scripts.build_release_corpus")
    font = builder._font(48)

    def fingerprint(character: str) -> tuple[tuple[int, int, int, int], bytes]:
        return font.getbbox(character), bytes(font.getmask(character))

    notdef = fingerprint("\U0010ffff")
    replacement = fingerprint("\ufffd")
    glyphs = [fingerprint(character) for character in "运营仪表盘"]

    assert all(glyph not in {notdef, replacement} for glyph in glyphs)
    assert len(set(glyphs)) == len(glyphs)


def test_release_builder_requires_a_new_output_root(tmp_path: Path) -> None:
    builder = importlib.import_module("scripts.build_release_corpus")
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        builder.build(existing)


def test_release_builder_main_accepts_one_positional_output(tmp_path: Path) -> None:
    builder = importlib.import_module("scripts.build_release_corpus")
    output = tmp_path / "from-main"

    assert builder.main([str(output)]) is None
    assert {path.name for path in output.iterdir()} == EXPECTED_INPUT_NAMES


def test_release_builder_creates_exact_eighteen_inputs_twice(
    fresh_release_corpora: tuple[Path, Path],
) -> None:
    first, second = fresh_release_corpora

    assert {path.name for path in first.iterdir()} == EXPECTED_INPUT_NAMES
    assert {path.name for path in second.iterdir()} == EXPECTED_INPUT_NAMES


def test_fresh_release_pngs_are_valid_and_pixel_equivalent(
    fresh_release_corpora: tuple[Path, Path],
) -> None:
    first, second = fresh_release_corpora

    for name, expected_size in EXPECTED_IMAGE_SIZES.items():
        with (
            Image.open(first / name) as first_image,
            Image.open(second / name) as second_image,
            Image.open(RELEASE_ROOT / "inputs" / name) as canonical_image,
        ):
            assert first_image.format == "PNG"
            assert second_image.format == "PNG"
            assert canonical_image.format == "PNG"
            assert first_image.size == expected_size
            assert second_image.size == expected_size
            assert canonical_image.size == expected_size
            _assert_same_rgb_pixels(first_image, second_image)


def test_fresh_release_pdfs_have_two_real_pages_and_equivalent_geometry(
    fresh_release_corpora: tuple[Path, Path],
) -> None:
    first, second = fresh_release_corpora

    for name in ("13-mixed-page-sizes.pdf", "14-rotated-page.pdf", "15-high-dpi.pdf"):
        assert _pdf_inventory(first / name) == _pdf_inventory(second / name)
        assert _pdf_inventory(first / name) == _pdf_inventory(
            RELEASE_ROOT / "inputs" / name
        )
        assert len(_pdf_inventory(first / name)) == 2

    mixed_sizes = _pdf_inventory(first / "13-mixed-page-sizes.pdf")
    assert mixed_sizes[0][:2] != mixed_sizes[1][:2]
    assert _pdf_inventory(first / "14-rotated-page.pdf")[1][2] == 90

    high_dpi_reader = PdfReader(first / "15-high-dpi.pdf")
    image_sizes = [
        (int(image.image.width), int(image.image.height))
        for page in high_dpi_reader.pages
        for image in page.images
    ]
    assert len(image_sizes) == 2
    assert all(max(size) >= 2400 for size in image_sizes)


def test_fresh_release_pptx_modes_have_expected_object_inventory(
    fresh_release_corpora: tuple[Path, Path],
) -> None:
    first, second = fresh_release_corpora
    names = (
        "16-image-only.pptx",
        "17-mixed-native.pptx",
        "18-mixed-screenshot-candidates.pptx",
    )

    for name in names:
        deck = Presentation(first / name)
        assert len(deck.slides) == 4
        assert deck.slide_width * 9 == deck.slide_height * 16
        assert deck.core_properties.created == datetime(2020, 1, 1)
        assert deck.core_properties.modified == datetime(2020, 1, 1)
        assert _pptx_inventory(first / name) == _pptx_inventory(second / name)
        assert _pptx_inventory(first / name) == _pptx_inventory(
            RELEASE_ROOT / "inputs" / name
        )
        with ZipFile(first / name) as archive:
            members = archive.infolist()
            assert [member.filename for member in members] == sorted(
                member.filename for member in members
            )
            assert all(member.date_time == (1980, 1, 1, 0, 0, 0) for member in members)
            assert all(member.compress_type == ZIP_DEFLATED for member in members)

    image_only = Presentation(first / "16-image-only.pptx")
    assert all(
        len(slide.shapes) == 1
        and slide.shapes[0].shape_type == MSO_SHAPE_TYPE.PICTURE
        for slide in image_only.slides
    )

    mixed_native = Presentation(first / "17-mixed-native.pptx")
    native_shapes = [shape for slide in mixed_native.slides for shape in slide.shapes]
    assert any(shape.shape_type == MSO_SHAPE_TYPE.PICTURE for shape in native_shapes)
    assert sum(shape.has_text_frame for shape in native_shapes) >= 8
    assert sum(shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE for shape in native_shapes) >= 8

    screenshot_candidates = Presentation(first / "18-mixed-screenshot-candidates.pptx")
    for slide in screenshot_candidates.slides:
        assert any(shape.shape_type == MSO_SHAPE_TYPE.PICTURE for shape in slide.shapes)
        assert any(shape.has_text_frame and shape.text for shape in slide.shapes)
        assert any(shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE for shape in slide.shapes)


def test_release_input_rejects_invalid_image_with_matching_hash(tmp_path: Path) -> None:
    case = copy.deepcopy(_manifest()["cases"][0])
    case["path"] = "invalid.png"
    path = tmp_path / case["path"]
    path.write_bytes(b"not an image")
    case["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(UnidentifiedImageError):
        _assert_release_input(case, tmp_path)


def test_release_input_rejects_valid_image_with_wrong_hash(tmp_path: Path) -> None:
    case = copy.deepcopy(_manifest()["cases"][0])
    case["path"] = "wrong-hash.png"
    path = tmp_path / case["path"]
    Image.new("RGB", (2, 2), "red").save(path)
    case["sha256"] = "f" * 64

    with pytest.raises(AssertionError):
        _assert_release_input(case, tmp_path)


def test_release_input_rejects_jpeg_renamed_as_png(tmp_path: Path) -> None:
    case = copy.deepcopy(_manifest()["cases"][0])
    case["path"] = "renamed.png"
    path = tmp_path / case["path"]
    Image.new("RGB", (2, 2), "red").save(path, format="JPEG")
    case["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        assert image.format == "JPEG"

    with pytest.raises(AssertionError):
        _assert_release_input(case, tmp_path)


def test_release_input_rejects_multiframe_apng(tmp_path: Path) -> None:
    case = copy.deepcopy(_manifest()["cases"][0])
    case["path"] = "two-frame.png"
    path = tmp_path / case["path"]
    first = Image.new("RGBA", (2, 2), "red")
    second = Image.new("RGBA", (2, 2), "blue")
    first.save(
        path,
        format="PNG",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    case["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.n_frames == 2

    with pytest.raises(AssertionError):
        _assert_release_input(case, tmp_path)


def test_release_input_rejects_image_that_opens_but_fails_verify(
    tmp_path: Path,
) -> None:
    case = copy.deepcopy(_manifest()["cases"][0])
    case["path"] = "truncated.png"
    path = tmp_path / case["path"]
    Image.new("RGB", (2, 2), "red").save(path)
    path.write_bytes(path.read_bytes()[:-10])
    case["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises((OSError, SyntaxError)):
        _assert_release_input(case, tmp_path)


def test_release_input_rejects_wrong_pdf_page_count_with_matching_hash(
    tmp_path: Path,
) -> None:
    case = copy.deepcopy(_manifest()["cases"][12])
    case["path"] = "one-page.pdf"
    path = tmp_path / case["path"]
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as stream:
        writer.write(stream)
    case["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(AssertionError):
        _assert_release_input(case, tmp_path)


def test_release_input_rejects_wrong_pptx_slide_count_with_matching_hash(
    tmp_path: Path,
) -> None:
    case = copy.deepcopy(_manifest()["cases"][15])
    case["path"] = "one-slide.pptx"
    path = tmp_path / case["path"]
    deck = Presentation()
    deck.slides.add_slide(deck.slide_layouts[6])
    deck.save(path)
    case["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(AssertionError):
        _assert_release_input(case, tmp_path)


def test_release_inputs_exist_and_match_manifest_sha256() -> None:
    cases = _manifest()["cases"]

    for case in cases:
        _assert_release_input(case, RELEASE_ROOT)
