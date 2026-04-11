# PDF/Image To Editable PPT Runtime Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current skill scaffold into a working PDF/image-to-PPT pipeline with real PDF rendering, native text extraction, OCR fallback, image extraction, PPT coordinate mapping, and conservative visual fallback behavior.

**Architecture:** The pipeline stays page-based. Each input is normalized into `PagePlan` objects containing a guaranteed background layer plus optional editable text/image blocks. `PyMuPDF` handles PDF rendering and native object extraction, `PaddleOCR` handles OCR fallback for images and scanned pages, and `python-pptx` writes the final presentation while preserving page proportions and block coordinates.

**Tech Stack:** Python, PyMuPDF, Pillow, PaddleOCR, python-pptx, pytest

---

### Task 1: Add runtime dependency documentation and availability checks

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/references/README.md`
- Modify: `skills/pdf-image-to-editable-ppt/SKILL.md`
- Create: `skills/pdf-image-to-editable-ppt/scripts/dependencies.py`
- Test: `tests/test_dependencies.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


dependencies = load_skill_module("dependencies")


def test_dependency_probe_reports_known_keys():
    result = dependencies.probe_dependencies()
    assert "pymupdf" in result
    assert "pillow" in result
    assert "paddleocr" in result
    assert "python_pptx" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dependencies.py::test_dependency_probe_reports_known_keys -v --basetemp .tmp-pytest/task1`
Expected: FAIL because `dependencies.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `skills/pdf-image-to-editable-ppt/scripts/dependencies.py`:

```python
from importlib.util import find_spec


def probe_dependencies() -> dict[str, bool]:
    return {
        "pymupdf": find_spec("fitz") is not None,
        "pillow": find_spec("PIL") is not None,
        "paddleocr": find_spec("paddleocr") is not None,
        "python_pptx": find_spec("pptx") is not None,
    }
```

Update `references/README.md` with concrete dependency notes and fallback behavior.

Update `SKILL.md` with a short “runtime expectations” section:

```markdown
## Runtime expectations
- Prefer PyMuPDF, Pillow, PaddleOCR, and python-pptx when available.
- If OCR dependencies are unavailable, still generate a fidelity-first PPT with background layers.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dependencies.py::test_dependency_probe_reports_known_keys -v --basetemp .tmp-pytest/task1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_dependencies.py skills/pdf-image-to-editable-ppt/scripts/dependencies.py skills/pdf-image-to-editable-ppt/references/README.md skills/pdf-image-to-editable-ppt/SKILL.md
git commit -m "feat: add runtime dependency probing"
```

### Task 2: Extend page models for real extraction pipelines

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/models.py`
- Test: `tests/test_models_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


models = load_skill_module("models")
PagePlan = models.PagePlan


def test_page_plan_tracks_page_dimensions_and_source_type():
    plan = PagePlan(
        page_number=2,
        width_px=1200,
        height_px=1600,
        background_path="page-2.png",
        source_type="pdf",
        page_width_points=595.0,
        page_height_points=842.0,
    )
    assert plan.source_type == "pdf"
    assert plan.page_width_points == 595.0
    assert plan.page_height_points == 842.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_runtime.py::test_page_plan_tracks_page_dimensions_and_source_type -v --basetemp .tmp-pytest/task2`
Expected: FAIL because the new fields are missing from `PagePlan`.

- [ ] **Step 3: Write minimal implementation**

Update `PagePlan` so it includes:

```python
source_type: str = "image"
page_width_points: float | None = None
page_height_points: float | None = None
```

Keep existing defaults intact so current tests remain valid.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py tests/test_models_runtime.py -v --basetemp .tmp-pytest/task2`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_models_runtime.py skills/pdf-image-to-editable-ppt/scripts/models.py
git commit -m "feat: extend page models for runtime pipeline"
```

### Task 3: Implement real PDF page rendering with PyMuPDF

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/render_pdf_pages.py`
- Test: `tests/test_render_pdf_pages.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from conftest import load_skill_module


render_pdf_pages = load_skill_module("render_pdf_pages").render_pdf_pages


def test_render_pdf_pages_rejects_missing_input(tmp_path):
    output_dir = tmp_path / "pages"
    try:
        render_pdf_pages("missing.pdf", output_dir)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_render_pdf_pages.py::test_render_pdf_pages_rejects_missing_input -v --basetemp .tmp-pytest/task3`
Expected: FAIL because the current implementation silently returns `[]`.

- [ ] **Step 3: Write minimal implementation**

Update `render_pdf_pages.py`:

```python
from pathlib import Path


def render_pdf_pages(input_path: str, output_dir: Path) -> list[str]:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    return []
```

Then add optional PyMuPDF rendering:

```python
try:
    import fitz
except ImportError:
    fitz = None

if fitz is None:
    return []
```

When `fitz` is available, iterate pages, render each page to PNG, and return the written file paths.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_render_pdf_pages.py::test_render_pdf_pages_rejects_missing_input -v --basetemp .tmp-pytest/task3`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_render_pdf_pages.py skills/pdf-image-to-editable-ppt/scripts/render_pdf_pages.py
git commit -m "feat: add real pdf rendering entry point"
```

### Task 4: Implement native text extraction with OCR fallback entry points

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/extract_text_layout.py`
- Test: `tests/test_extract_text_layout.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from conftest import load_skill_module


extract_text_layout = load_skill_module("extract_text_layout").extract_text_layout


def test_extract_text_layout_rejects_missing_input():
    try:
        extract_text_layout("missing.pdf", page_number=1)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extract_text_layout.py::test_extract_text_layout_rejects_missing_input -v --basetemp .tmp-pytest/task4`
Expected: FAIL because the current implementation always returns `[]`.

- [ ] **Step 3: Write minimal implementation**

Update `extract_text_layout.py` to:

```python
from pathlib import Path


def extract_text_layout(input_path: str, *, page_number: int) -> list[dict]:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    return []
```

Then implement page-type branching:

- If input is an image, try OCR.
- If input is a PDF page with native text, return normalized text blocks from PyMuPDF.
- If native text is absent or poor, try OCR.

Normalize output dictionaries to include:

```python
{
    "text": "Title",
    "left": 10.0,
    "top": 20.0,
    "width": 100.0,
    "height": 30.0,
    "font_size": 24.0,
    "color": "#000000",
    "alignment": "left",
    "confidence": 0.95,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extract_text_layout.py::test_extract_text_layout_rejects_missing_input -v --basetemp .tmp-pytest/task4`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_extract_text_layout.py skills/pdf-image-to-editable-ppt/scripts/extract_text_layout.py
git commit -m "feat: add native text and ocr extraction flow"
```

### Task 5: Implement native and cropped image extraction

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/extract_images.py`
- Test: `tests/test_extract_images.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from conftest import load_skill_module


extract_images = load_skill_module("extract_images").extract_images


def test_extract_images_rejects_missing_input(tmp_path):
    try:
        extract_images("missing.pdf", page_number=1, output_dir=tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extract_images.py::test_extract_images_rejects_missing_input -v --basetemp .tmp-pytest/task5`
Expected: FAIL because the current implementation always returns `[]`.

- [ ] **Step 3: Write minimal implementation**

Update `extract_images.py` to:

```python
from pathlib import Path


def extract_images(input_path: str, *, page_number: int, output_dir: Path) -> list[dict]:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    return []
```

Then implement:

- PDF native image extraction via PyMuPDF
- optional high-confidence crop extraction from rendered pages

Normalize each image block to:

```python
{
    "path": "page-1-img-0.png",
    "left": 10.0,
    "top": 20.0,
    "width": 200.0,
    "height": 120.0,
    "confidence": 0.95,
    "extractable": True,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extract_images.py::test_extract_images_rejects_missing_input -v --basetemp .tmp-pytest/task5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_extract_images.py skills/pdf-image-to-editable-ppt/scripts/extract_images.py
git commit -m "feat: add native and cropped image extraction"
```

### Task 6: Build page plans from real extraction outputs

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/page_planner.py`
- Test: `tests/test_page_planner.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


page_planner = load_skill_module("page_planner")


def test_page_planner_builds_page_plan_with_blocks():
    plan = page_planner.build_page_plan(
        page_number=1,
        width_px=1000,
        height_px=1500,
        background_path="page.png",
        text_items=[{"text": "Hello", "left": 0, "top": 0, "width": 10, "height": 10, "font_size": 12, "color": "#000000", "alignment": "left", "confidence": 0.9}],
        image_items=[{"path": "img.png", "left": 0, "top": 0, "width": 10, "height": 10, "confidence": 0.9, "extractable": True}],
    )
    assert len(plan.text_blocks) == 1
    assert len(plan.image_blocks) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_page_planner.py::test_page_planner_builds_page_plan_with_blocks -v --basetemp .tmp-pytest/task6`
Expected: FAIL because `page_planner.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `page_planner.py`:

```python
from .models import ImageBlock, PagePlan, TextBlock


def build_page_plan(*, page_number, width_px, height_px, background_path, text_items, image_items, source_type="image", page_width_points=None, page_height_points=None):
    plan = PagePlan(
        page_number=page_number,
        width_px=width_px,
        height_px=height_px,
        background_path=background_path,
        source_type=source_type,
        page_width_points=page_width_points,
        page_height_points=page_height_points,
    )
    plan.text_blocks = [TextBlock(**item) for item in text_items]
    plan.image_blocks = [ImageBlock(**item) for item in image_items]
    return plan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_page_planner.py::test_page_planner_builds_page_plan_with_blocks -v --basetemp .tmp-pytest/task6`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_page_planner.py skills/pdf-image-to-editable-ppt/scripts/page_planner.py
git commit -m "feat: add runtime page planning"
```

### Task 7: Replace minimal PPT writer with python-pptx background-first output

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/build_ppt.py`
- Test: `tests/test_build_ppt_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from conftest import load_skill_module


models = load_skill_module("models")
build_ppt = load_skill_module("build_ppt")
PagePlan = models.PagePlan


def test_build_presentation_rejects_missing_background(tmp_path):
    page = PagePlan(page_number=1, width_px=100, height_px=200, background_path="missing.png")
    output_path = tmp_path / "result.pptx"
    try:
        build_ppt.build_presentation([page], output_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_ppt_runtime.py::test_build_presentation_rejects_missing_background -v --basetemp .tmp-pytest/task7`
Expected: FAIL because the current implementation ignores backgrounds entirely.

- [ ] **Step 3: Write minimal implementation**

Update `build_ppt.py`:

- validate `background_path` exists for every page
- if `python-pptx` is unavailable, fall back to the current minimal `.pptx` writer
- if `python-pptx` is available:
  - set slide size from the first page ratio
  - add one blank slide per page
  - insert the background image as a full-slide picture
  - insert text blocks as text boxes
  - insert image blocks as pictures

Use a helper signature like:

```python
def build_presentation(page_plans, output_path: Path) -> None:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_ppt.py tests/test_build_ppt_runtime.py -v --basetemp .tmp-pytest/task7`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_ppt_runtime.py skills/pdf-image-to-editable-ppt/scripts/build_ppt.py
git commit -m "feat: add background-first ppt generation"
```

### Task 8: Add an end-to-end orchestration entry point

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/convert_to_ppt.py`
- Test: `tests/test_convert_to_ppt.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from conftest import load_skill_module


convert_to_ppt = load_skill_module("convert_to_ppt").convert_to_ppt


def test_convert_to_ppt_rejects_missing_input(tmp_path):
    output_path = tmp_path / "result.pptx"
    try:
        convert_to_ppt("missing.pdf", output_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convert_to_ppt.py::test_convert_to_ppt_rejects_missing_input -v --basetemp .tmp-pytest/task8`
Expected: FAIL because `convert_to_ppt.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `convert_to_ppt.py`:

```python
from pathlib import Path


def convert_to_ppt(input_path: str, output_path: Path) -> None:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
```

Then wire it to:

- render pages when input is PDF
- build text/image candidates per page
- build page plans
- filter editable blocks
- call `build_presentation`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convert_to_ppt.py::test_convert_to_ppt_rejects_missing_input -v --basetemp .tmp-pytest/task8`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_convert_to_ppt.py skills/pdf-image-to-editable-ppt/scripts/convert_to_ppt.py
git commit -m "feat: add end-to-end conversion entry point"
```

### Task 9: Update tests and docs for runtime pipeline behavior

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/SKILL.md`
- Modify: `skills/pdf-image-to-editable-ppt/references/README.md`
- Modify: `Course.md`
- Test: `tests/test_skill_content.py`
- Test: `tests/test_references_readme.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_skill_mentions_runtime_pipeline():
    text = Path("skills/pdf-image-to-editable-ppt/SKILL.md").read_text(encoding="utf-8")
    assert "PyMuPDF" in text
    assert "PaddleOCR" in text
    assert "convert_to_ppt.py" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_content.py::test_skill_mentions_runtime_pipeline -v --basetemp .tmp-pytest/task9`
Expected: FAIL because the current skill file only describes placeholders.

- [ ] **Step 3: Write minimal implementation**

Update `SKILL.md` to describe:

- native-text-first behavior
- OCR fallback behavior
- background-first PPT output
- `convert_to_ppt.py` as the orchestration entry point

Update `references/README.md` to describe:

- dependency probing
- optional vs required runtime libraries
- current stage-one limitations

Update `Course.md` so it reflects the new runtime-upgrade plan status.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skill_content.py tests/test_references_readme.py -v --basetemp .tmp-pytest/task9`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add skills/pdf-image-to-editable-ppt/SKILL.md skills/pdf-image-to-editable-ppt/references/README.md Course.md tests/test_skill_content.py tests/test_references_readme.py
git commit -m "docs: describe runtime conversion pipeline"
```

## Self-Review

- Spec coverage:
  - 真实依赖接入由 Task 1 覆盖。
  - 页面模型扩展与坐标信息由 Task 2 覆盖。
  - PDF 渲染由 Task 3 覆盖。
  - 原生文本与 OCR 入口由 Task 4 覆盖。
  - 原生图片与裁切图片提取由 Task 5 覆盖。
  - 页面计划组装由 Task 6 覆盖。
  - PPT 背景层与对象写入由 Task 7 覆盖。
  - 端到端转换入口由 Task 8 覆盖。
  - 文档与状态同步由 Task 9 覆盖。
- Placeholder scan:
  - 所有任务都包含明确文件、测试、命令和最小实现内容。
  - 没有 `TODO`、`TBD`、或“后续自行处理”这类占位描述。
- Type consistency:
  - `PagePlan` 的新增字段先在 Task 2 定义，再在后续任务消费。
  - `probe_dependencies`、`render_pdf_pages`、`extract_text_layout`、`extract_images`、`build_page_plan`、`build_presentation`、`convert_to_ppt` 名称前后一致。
