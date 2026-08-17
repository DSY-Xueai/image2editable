from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError
from pptx import Presentation
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

本目录定义真实发布 benchmark 的第一阶段契约：18 个输入、30 页，包括 12 张图片、3 个双页 PDF、3 个四页 PPTX。

## 覆盖范围

图片固定覆盖：中英双语仪表盘、密集参数对比、人物信息卡、四段时间线、柱线组合图、流程图、图标矩阵、浅色文字渐变页、细线网络图、小元素表格、深色海报、非 16:9 信息图。

PDF 分别覆盖双页不同尺寸、旋转、高 DPI。PPTX 分别覆盖 `image_only`、`mixed_native`、`mixed_screenshot_candidates`。

## 来源与许可

后续输入由项目公开生成，manifest 固定记录 `source=project-generated`、`license=CC0-1.0`。所有 case 默认使用已支持的 `agent_provider=host`。

## 阶段状态

当前阶段不包含输入文件，也不包含 runner。当前 `sha256` 是满足 schema 的 64 位占位值；SHA-256 为 64 位占位值不代表语料已完成。Task 2 必须生成全部输入文件，并把占位值替换为文件的真实 SHA-256；测试中的严格 xfail 保留这项 RED 契约。

## 通过标准

每页必须达到 manifest 中的最小组件数和文本框数、状态为 `validated`，并同时满足 0 warning、0 unexplained pixels、0 quality violations。后续 runner 对全部 30 页执行 `repeat=3`；runner、输入生成和 CI 接入不属于本阶段。
"""


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


@pytest.mark.xfail(
    strict=True,
    reason="Task 2 must generate every release input and replace placeholder hashes",
)
def test_release_inputs_exist_and_match_manifest_sha256() -> None:
    cases = _manifest()["cases"]

    for case in cases:
        _assert_release_input(case, RELEASE_ROOT)
