# Image To Canvas To PPT Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the conversion pipeline to process image input into a Canvas scene model and generate usable PPT output with visual-first fallback and PaddleOCR text editability.

**Architecture:** The new pipeline is `ingest -> OCR -> segmentation -> CanvasPage -> PPT`. `CanvasPage/CanvasNode` becomes the only intermediate contract, and PPT generation reads only Canvas models. Failure is fail-soft: per-node and per-page fallback always produces an openable PPT.

**Tech Stack:** Python, Pillow, PaddleOCR, python-pptx, pytest

---

### Task 1: 建立 Canvas 数据模型与接口契约

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/canvas_models.py`
- Test: `tests/test_canvas_models.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


models = load_skill_module("canvas_models")
CanvasNode = models.CanvasNode
CanvasPage = models.CanvasPage


def test_canvas_page_contains_nodes():
    page = CanvasPage(page_w_px=1000, page_h_px=500, bg_color=None)
    page.nodes.append(
        CanvasNode(type="text", x=10, y=20, w=100, h=30, z=5, payload={"text": "Hi"})
    )
    assert page.page_w_px == 1000
    assert page.page_h_px == 500
    assert page.nodes[0].type == "text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_canvas_models.py::test_canvas_page_contains_nodes -v --basetemp .tmp-pytest/canvas-models`
Expected: FAIL because `canvas_models.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `skills/pdf-image-to-editable-ppt/scripts/canvas_models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CanvasNodeType = Literal["text", "image", "shape"]


@dataclass(slots=True)
class CanvasNode:
    type: CanvasNodeType
    x: float
    y: float
    w: float
    h: float
    z: int
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class CanvasPage:
    page_w_px: int
    page_h_px: int
    bg_color: tuple[int, int, int] | None = None
    nodes: list[CanvasNode] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_canvas_models.py -v --basetemp .tmp-pytest/canvas-models`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_canvas_models.py skills/pdf-image-to-editable-ppt/scripts/canvas_models.py
git commit -m "feat: add canvas scene data models"
```

### Task 2: 图片输入标准化（ingest）

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/ingest_image.py`
- Test: `tests/test_ingest_image.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from PIL import Image
from conftest import load_skill_module


ingest = load_skill_module("ingest_image")


def test_ingest_image_returns_size(tmp_path):
    src = tmp_path / "sample.png"
    Image.new("RGB", (320, 180), "white").save(src)
    result = ingest.ingest_image(src)
    assert result["width_px"] == 320
    assert result["height_px"] == 180
    assert Path(result["image_path"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_image.py::test_ingest_image_returns_size -v --basetemp .tmp-pytest/ingest`
Expected: FAIL because `ingest_image.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `skills/pdf-image-to-editable-ppt/scripts/ingest_image.py`:

```python
from __future__ import annotations

from pathlib import Path

from PIL import Image


ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def ingest_image(input_path: str | Path) -> dict:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported image type: {source.suffix}")

    with Image.open(source) as img:
        if img.mode not in {"RGB", "RGBA"}:
            img = img.convert("RGB")
            img.save(source)
        width_px, height_px = img.size

    return {
        "width_px": int(width_px),
        "height_px": int(height_px),
        "image_path": str(source),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_image.py -v --basetemp .tmp-pytest/ingest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_ingest_image.py skills/pdf-image-to-editable-ppt/scripts/ingest_image.py
git commit -m "feat: add image ingest normalization"
```

### Task 3: PaddleOCR 抽取与标准化

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/ocr_paddle.py`
- Test: `tests/test_ocr_paddle.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


ocr_paddle = load_skill_module("ocr_paddle")


def test_normalize_paddle_line_to_textbox():
    line = [
        [[10, 20], [110, 20], [110, 50], [10, 50]],
        ("标题", 0.93),
    ]
    item = ocr_paddle.normalize_paddle_line(line)
    assert item["text"] == "标题"
    assert item["left"] == 10.0
    assert item["top"] == 20.0
    assert item["width"] == 100.0
    assert item["height"] == 30.0
    assert item["score"] == 0.93
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ocr_paddle.py::test_normalize_paddle_line_to_textbox -v --basetemp .tmp-pytest/ocr`
Expected: FAIL because `ocr_paddle.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `skills/pdf-image-to-editable-ppt/scripts/ocr_paddle.py`:

```python
from __future__ import annotations

from pathlib import Path


def normalize_paddle_line(line) -> dict:
    bbox, payload = line
    text, score = payload
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    left = float(min(xs))
    top = float(min(ys))
    right = float(max(xs))
    bottom = float(max(ys))
    return {
        "text": str(text),
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
        "score": float(score),
        "angle": 0.0,
    }


def extract_text_boxes(image_path: str | Path) -> list[dict]:
    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(source)
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return []

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    raw = ocr.ocr(str(source))
    items: list[dict] = []
    for group in raw or []:
        for line in group:
            items.append(normalize_paddle_line(line))
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ocr_paddle.py -v --basetemp .tmp-pytest/ocr`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_ocr_paddle.py skills/pdf-image-to-editable-ppt/scripts/ocr_paddle.py
git commit -m "feat: add paddle ocr normalization module"
```

### Task 4: 非文本区域分块（矩形版）

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/scene_segmentation.py`
- Test: `tests/test_scene_segmentation.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


seg = load_skill_module("scene_segmentation")


def test_inverse_text_regions_returns_image_chunks():
    text_boxes = [{"left": 100, "top": 100, "width": 200, "height": 100}]
    chunks = seg.inverse_text_regions(page_w=800, page_h=600, text_boxes=text_boxes)
    assert len(chunks) >= 1
    assert all(c["width"] > 0 and c["height"] > 0 for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scene_segmentation.py::test_inverse_text_regions_returns_image_chunks -v --basetemp .tmp-pytest/seg`
Expected: FAIL because `scene_segmentation.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `skills/pdf-image-to-editable-ppt/scripts/scene_segmentation.py`:

```python
from __future__ import annotations


def inverse_text_regions(*, page_w: int, page_h: int, text_boxes: list[dict]) -> list[dict]:
    if page_w <= 0 or page_h <= 0:
        return []
    if not text_boxes:
        return [{"left": 0.0, "top": 0.0, "width": float(page_w), "height": float(page_h)}]

    top = min(max(float(b["top"]), 0.0) for b in text_boxes)
    bottom = max(min(float(b["top"] + b["height"]), float(page_h)) for b in text_boxes)

    chunks: list[dict] = []
    if top > 0:
        chunks.append({"left": 0.0, "top": 0.0, "width": float(page_w), "height": top})
    if bottom < page_h:
        chunks.append(
            {"left": 0.0, "top": bottom, "width": float(page_w), "height": float(page_h) - bottom}
        )
    return [c for c in chunks if c["width"] > 1 and c["height"] > 1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scene_segmentation.py -v --basetemp .tmp-pytest/seg`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_scene_segmentation.py skills/pdf-image-to-editable-ppt/scripts/scene_segmentation.py
git commit -m "feat: add phase1 non-text segmentation"
```

### Task 5: 组装 Canvas 场景（统一中间层）

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/build_canvas_scene.py`
- Create: `skills/pdf-image-to-editable-ppt/scripts/fallback_policy.py`
- Test: `tests/test_build_canvas_scene.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


builder = load_skill_module("build_canvas_scene")


def test_build_canvas_scene_includes_background_and_text():
    page = builder.build_canvas_scene(
        image_source={"width_px": 800, "height_px": 600, "image_path": "a.png"},
        ocr_items=[{"text": "Hello", "left": 10, "top": 20, "width": 100, "height": 30, "score": 0.95}],
        ocr_min_score=0.78,
    )
    assert page.nodes[0].type == "image"
    assert any(n.type == "text" for n in page.nodes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_canvas_scene.py::test_build_canvas_scene_includes_background_and_text -v --basetemp .tmp-pytest/scene-build`
Expected: FAIL because `build_canvas_scene.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `skills/pdf-image-to-editable-ppt/scripts/fallback_policy.py`:

```python
from __future__ import annotations


def should_keep_text(item: dict, min_score: float) -> bool:
    return float(item.get("score", 0.0)) >= float(min_score)
```

Create `skills/pdf-image-to-editable-ppt/scripts/build_canvas_scene.py`:

```python
from __future__ import annotations

from .canvas_models import CanvasNode, CanvasPage
from .fallback_policy import should_keep_text


def build_canvas_scene(*, image_source: dict, ocr_items: list[dict], ocr_min_score: float = 0.78) -> CanvasPage:
    page = CanvasPage(
        page_w_px=int(image_source["width_px"]),
        page_h_px=int(image_source["height_px"]),
        bg_color=None,
    )
    page.nodes.append(
        CanvasNode(
            type="image",
            x=0.0,
            y=0.0,
            w=float(page.page_w_px),
            h=float(page.page_h_px),
            z=0,
            payload={"image_path": image_source["image_path"], "crop_source": "full", "confidence": 1.0},
        )
    )

    z = 100
    for item in ocr_items:
        if not should_keep_text(item, ocr_min_score):
            continue
        page.nodes.append(
            CanvasNode(
                type="text",
                x=float(item["left"]),
                y=float(item["top"]),
                w=float(item["width"]),
                h=float(item["height"]),
                z=z,
                payload={
                    "text": item["text"],
                    "score": float(item["score"]),
                    "font_size_est": float(item["height"]),
                    "color_est": "#000000",
                    "align": "left",
                },
            )
        )
        z += 1

    page.nodes.sort(key=lambda n: n.z)
    return page
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_canvas_scene.py -v --basetemp .tmp-pytest/scene-build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_canvas_scene.py skills/pdf-image-to-editable-ppt/scripts/build_canvas_scene.py skills/pdf-image-to-editable-ppt/scripts/fallback_policy.py
git commit -m "feat: add canvas scene builder and fallback policy"
```

### Task 6: Canvas 到 PPT 映射输出

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/canvas_to_ppt.py`
- Test: `tests/test_canvas_to_ppt.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from conftest import load_skill_module


models = load_skill_module("canvas_models")
writer = load_skill_module("canvas_to_ppt")


def test_canvas_to_ppt_writes_file(tmp_path):
    png = tmp_path / "bg.png"
    from PIL import Image
    Image.new("RGB", (320, 180), "white").save(png)

    page = models.CanvasPage(page_w_px=320, page_h_px=180)
    page.nodes.append(models.CanvasNode(type="image", x=0, y=0, w=320, h=180, z=0, payload={"image_path": str(png)}))

    out = tmp_path / "result.pptx"
    writer.canvas_pages_to_ppt([page], out)
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_canvas_to_ppt.py::test_canvas_to_ppt_writes_file -v --basetemp .tmp-pytest/canvas-ppt`
Expected: FAIL because `canvas_to_ppt.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `skills/pdf-image-to-editable-ppt/scripts/canvas_to_ppt.py`:

```python
from __future__ import annotations

from pathlib import Path


def _emu(v: float) -> int:
    return int(round(v))


def canvas_pages_to_ppt(canvas_pages, output_path: Path) -> None:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Emu, Pt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    while prs.slides:
        prs.slides._sldIdLst.remove(prs.slides._sldIdLst[0])

    first = canvas_pages[0]
    slide_w = 12192000
    slide_h = int(round(slide_w * (first.page_h_px / max(first.page_w_px, 1))))
    prs.slide_width = slide_w
    prs.slide_height = slide_h

    align_map = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT, "justify": PP_ALIGN.JUSTIFY}

    for page in canvas_pages:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        sx = prs.slide_width / max(page.page_w_px, 1)
        sy = prs.slide_height / max(page.page_h_px, 1)
        for node in sorted(page.nodes, key=lambda n: n.z):
            x = Emu(_emu(node.x * sx))
            y = Emu(_emu(node.y * sy))
            w = Emu(_emu(node.w * sx))
            h = Emu(_emu(node.h * sy))
            if node.type == "image":
                image_path = node.payload.get("image_path")
                if image_path and Path(image_path).exists():
                    slide.shapes.add_picture(str(image_path), x, y, width=w, height=h)
            elif node.type == "text":
                tb = slide.shapes.add_textbox(x, y, w, h)
                p = tb.text_frame.paragraphs[0]
                p.text = str(node.payload.get("text", ""))
                p.alignment = align_map.get(node.payload.get("align", "left"), PP_ALIGN.LEFT)
                run = p.runs[0]
                run.font.size = Pt(float(node.payload.get("font_size_est", 12.0)))
                run.font.color.rgb = RGBColor.from_string(str(node.payload.get("color_est", "#000000")).lstrip("#"))

    prs.save(output_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_canvas_to_ppt.py -v --basetemp .tmp-pytest/canvas-ppt`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_canvas_to_ppt.py skills/pdf-image-to-editable-ppt/scripts/canvas_to_ppt.py
git commit -m "feat: add canvas to ppt writer"
```

### Task 7: 重写主入口为 image->canvas->ppt 并输出 debug 产物

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/convert_to_ppt.py`
- Test: `tests/test_convert_to_ppt.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest
from PIL import Image
from conftest import load_skill_module


convert_to_ppt = load_skill_module("convert_to_ppt").convert_to_ppt


def test_convert_to_ppt_rejects_pdf_input(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(ValueError):
        convert_to_ppt(str(pdf), tmp_path / "out.pptx")


def test_convert_to_ppt_image_generates_debug_outputs(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (200, 120), "white").save(img)
    out = tmp_path / "out.pptx"
    convert_to_ppt(str(img), out)
    assert out.exists()
    assert (tmp_path / "out_debug" / "canvas_scene.json").exists()
    assert (tmp_path / "out_debug" / "ocr.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convert_to_ppt.py::test_convert_to_ppt_rejects_pdf_input tests/test_convert_to_ppt.py::test_convert_to_ppt_image_generates_debug_outputs -v --basetemp .tmp-pytest/entry`
Expected: FAIL because current `convert_to_ppt.py` still supports PDF and has no new debug outputs.

- [ ] **Step 3: Write minimal implementation**

Update `skills/pdf-image-to-editable-ppt/scripts/convert_to_ppt.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .build_canvas_scene import build_canvas_scene
from .canvas_to_ppt import canvas_pages_to_ppt
from .ingest_image import ingest_image
from .ocr_paddle import extract_text_boxes


def _write_debug(output_path: Path, canvas_page, ocr_items: list[dict], image_path: str) -> None:
    debug_dir = output_path.parent / f"{output_path.stem}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    scene = {
        "page_w_px": canvas_page.page_w_px,
        "page_h_px": canvas_page.page_h_px,
        "nodes": [
            {
                "type": n.type,
                "x": n.x,
                "y": n.y,
                "w": n.w,
                "h": n.h,
                "z": n.z,
                "payload": n.payload,
            }
            for n in canvas_page.nodes
        ],
    }
    (debug_dir / "canvas_scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    (debug_dir / "ocr.json").write_text(json.dumps(ocr_items, ensure_ascii=False, indent=2), encoding="utf-8")

    with Image.open(image_path) as img:
        vis = img.convert("RGB")
        draw = ImageDraw.Draw(vis)
        for item in ocr_items:
            x = item["left"]
            y = item["top"]
            w = item["width"]
            h = item["height"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=1)
        vis.save(debug_dir / "segmentation.png")


def convert_to_ppt(input_path: str, output_path: Path, enable_stage2: bool = False) -> None:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Phase 1 only supports image input")

    image_source = ingest_image(source)
    ocr_items = extract_text_boxes(source)
    canvas_page = build_canvas_scene(image_source=image_source, ocr_items=ocr_items, ocr_min_score=0.78)
    canvas_pages_to_ppt([canvas_page], output_path)
    _write_debug(output_path, canvas_page, ocr_items, str(source))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convert_to_ppt.py -v --basetemp .tmp-pytest/entry`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_convert_to_ppt.py skills/pdf-image-to-editable-ppt/scripts/convert_to_ppt.py
git commit -m "refactor: switch entry to image-canvas-ppt pipeline"
```

### Task 8: 文档同步（SKILL/README/Course）

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/SKILL.md`
- Modify: `skills/pdf-image-to-editable-ppt/references/README.md`
- Modify: `Course.md`
- Test: `tests/test_skill_content.py`
- Test: `tests/test_references_readme.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_skill_mentions_canvas_pipeline():
    text = Path("skills/pdf-image-to-editable-ppt/SKILL.md").read_text(encoding="utf-8")
    assert "Canvas" in text
    assert "PaddleOCR" in text
    assert "convert_to_ppt.py" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_content.py::test_skill_mentions_canvas_pipeline -v --basetemp .tmp-pytest/docs`
Expected: FAIL because docs still describe old hybrid PDF flow.

- [ ] **Step 3: Write minimal implementation**

Update docs to reflect:

```markdown
- Phase 1 input: png/jpg/webp only
- Pipeline: image -> Canvas scene -> PPT
- OCR engine: PaddleOCR (text editability)
- Visual-first fallback for complex regions
- Debug outputs: canvas_scene.json / ocr.json / segmentation.png
```

Update `Course.md` 的本轮内容为“Canvas 重构计划已立项并开始实施”，并列出新入口与关键文件。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skill_content.py tests/test_references_readme.py -v --basetemp .tmp-pytest/docs`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add skills/pdf-image-to-editable-ppt/SKILL.md skills/pdf-image-to-editable-ppt/references/README.md Course.md tests/test_skill_content.py tests/test_references_readme.py
git commit -m "docs: align docs with image-canvas-ppt phase1"
```

## Self-Review

- Spec coverage:
  - Canvas 中间层与类型边界由 Task 1、Task 5 覆盖。
  - 图片单入口策略由 Task 2、Task 7 覆盖。
  - PaddleOCR 主引擎与文本可编辑由 Task 3、Task 6、Task 7 覆盖。
  - 视觉优先降级策略由 Task 4、Task 5 覆盖。
  - Debug 产物与逐张验收支持由 Task 7 覆盖。
  - 文档与课程状态同步由 Task 8 覆盖。
- Placeholder scan:
  - 无占位式描述（例如“后续补充”“实现时再定”）。
  - 每个任务均包含明确文件、测试、命令、最小实现代码。
- Type consistency:
  - `CanvasPage`/`CanvasNode` 字段在 Task 1 定义，Task 5/6/7 一致使用。
  - `convert_to_ppt` 签名保持 `convert_to_ppt(input_path, output_path, enable_stage2=False)`，兼容现有调用方。
