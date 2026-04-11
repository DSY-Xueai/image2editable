# PDF/Image To Editable PPT Stage 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stage 2 enhancements that improve text-editability only when layout can remain fully identical and improve common visual effects only when PPT can reproduce them safely.

**Architecture:** Stage 2 extends the stage 1 runtime pipeline with two conservative subsystems: a strict text-fitting validator and a common-effect mapping layer. Both subsystems operate as optional enhancement passes after background generation and before final PPT writing, and both must fail closed back to the background layer or stage 1 output.

**Tech Stack:** Python, Pillow, PyMuPDF, python-pptx, pytest

---

### Task 1: Add stage 2 data structures for text fitting and effects

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/models.py`
- Test: `tests/test_models_stage2.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


models = load_skill_module("models")
TextBlock = models.TextBlock
EffectBlock = models.EffectBlock


def test_text_block_supports_fit_validation_flags():
    block = TextBlock(
        text="Title",
        left=10.0,
        top=20.0,
        width=200.0,
        height=40.0,
        font_size=24.0,
        color="#112233",
        alignment="center",
        confidence=0.95,
        font_name="Arial",
        line_height=28.0,
        fit_verified=True,
    )
    assert block.font_name == "Arial"
    assert block.line_height == 28.0
    assert block.fit_verified is True


def test_effect_block_tracks_effect_kind_and_confidence():
    block = EffectBlock(
        effect_type="shadow",
        left=0.0,
        top=0.0,
        width=100.0,
        height=40.0,
        confidence=0.9,
        payload={"blur": 2.0},
    )
    assert block.effect_type == "shadow"
    assert block.payload["blur"] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_stage2.py -v --basetemp .tmp-pytest/stage2-task1`
Expected: FAIL because the extra text-fitting fields and `EffectBlock` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Update `models.py`:

```python
@dataclass(slots=True)
class TextBlock:
    ...
    font_name: str | None = None
    line_height: float | None = None
    fit_verified: bool = False


@dataclass(slots=True)
class EffectBlock:
    effect_type: str
    left: float
    top: float
    width: float
    height: float
    confidence: float
    payload: dict = field(default_factory=dict)
```

Also extend `PagePlan` with:

```python
effect_blocks: list[EffectBlock] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py tests/test_models_stage2.py -v --basetemp .tmp-pytest/stage2-task1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_models_stage2.py skills/pdf-image-to-editable-ppt/scripts/models.py
git commit -m "feat: add stage2 models for text fitting and effects"
```

### Task 2: Implement strict text fitting validation

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/text_fitting.py`
- Test: `tests/test_text_fitting.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


text_fitting = load_skill_module("text_fitting")


def test_validate_text_block_fit_rejects_unverified_wrap():
    candidate = {
        "text": "A long title",
        "font_name": "Arial",
        "font_size": 24.0,
        "line_height": 28.0,
        "left": 10.0,
        "top": 20.0,
        "width": 200.0,
        "height": 40.0,
        "alignment": "center",
        "expected_lines": 1,
        "predicted_lines": 2,
    }
    result = text_fitting.validate_text_block_fit(candidate)
    assert result is False


def test_validate_text_block_fit_accepts_exact_layout_match():
    candidate = {
        "text": "Title",
        "font_name": "Arial",
        "font_size": 24.0,
        "line_height": 28.0,
        "left": 10.0,
        "top": 20.0,
        "width": 200.0,
        "height": 40.0,
        "alignment": "center",
        "expected_lines": 1,
        "predicted_lines": 1,
        "position_delta": 0.0,
    }
    result = text_fitting.validate_text_block_fit(candidate)
    assert result is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_text_fitting.py -v --basetemp .tmp-pytest/stage2-task2`
Expected: FAIL because `text_fitting.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `text_fitting.py`:

```python
def validate_text_block_fit(candidate: dict) -> bool:
    if candidate.get("expected_lines") != candidate.get("predicted_lines"):
        return False
    if float(candidate.get("position_delta", 0.0)) != 0.0:
        return False
    if not candidate.get("font_name"):
        return False
    return True
```

Also add:

```python
def mark_fit_verified(text_block):
    text_block.fit_verified = True
    return text_block
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_text_fitting.py -v --basetemp .tmp-pytest/stage2-task2`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_text_fitting.py skills/pdf-image-to-editable-ppt/scripts/text_fitting.py
git commit -m "feat: add strict text fitting validator"
```

### Task 3: Implement common-effect capability mapping

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/effect_mapping.py`
- Test: `tests/test_effect_mapping.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


effect_mapping = load_skill_module("effect_mapping")


def test_map_effect_type_accepts_supported_shadow():
    result = effect_mapping.map_effect_type(
        {"effect_type": "shadow", "confidence": 0.95, "complexity": "simple"}
    )
    assert result["mapped"] is True
    assert result["effect_type"] == "shadow"


def test_map_effect_type_rejects_unsupported_blend_mode():
    result = effect_mapping.map_effect_type(
        {"effect_type": "blend_mode", "confidence": 0.95, "complexity": "complex"}
    )
    assert result["mapped"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_effect_mapping.py -v --basetemp .tmp-pytest/stage2-task3`
Expected: FAIL because `effect_mapping.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `effect_mapping.py`:

```python
DIRECT_EFFECTS = {"shadow", "stroke", "glow", "opacity", "gradient"}


def map_effect_type(effect: dict) -> dict:
    effect_type = effect.get("effect_type")
    confidence = float(effect.get("confidence", 0.0))
    complexity = effect.get("complexity", "simple")
    if effect_type in DIRECT_EFFECTS and confidence >= 0.9 and complexity == "simple":
        return {"mapped": True, "effect_type": effect_type}
    return {"mapped": False, "effect_type": effect_type}
```

Also add:

```python
def filter_mappable_effects(effects: list[dict]) -> list[dict]:
    return [effect for effect in effects if map_effect_type(effect)["mapped"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_effect_mapping.py -v --basetemp .tmp-pytest/stage2-task3`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_effect_mapping.py skills/pdf-image-to-editable-ppt/scripts/effect_mapping.py
git commit -m "feat: add common effect capability mapping"
```

### Task 4: Build a stage 2 enhancement pass for page plans

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/stage2_enhance.py`
- Test: `tests/test_stage2_enhance.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


models = load_skill_module("models")
stage2 = load_skill_module("stage2_enhance")

PagePlan = models.PagePlan
TextBlock = models.TextBlock


def test_stage2_enhance_removes_unverified_text_blocks():
    plan = PagePlan(page_number=1, width_px=100, height_px=100, background_path="page.png")
    plan.text_blocks.append(
        TextBlock(
            text="Title",
            left=0,
            top=0,
            width=10,
            height=10,
            font_size=12,
            color="#000000",
            alignment="left",
            confidence=0.95,
            font_name="Arial",
            fit_verified=False,
        )
    )
    enhanced = stage2.apply_stage2_enhancements(plan)
    assert enhanced.text_blocks == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage2_enhance.py -v --basetemp .tmp-pytest/stage2-task4`
Expected: FAIL because `stage2_enhance.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `stage2_enhance.py`:

```python
from .models import PagePlan


def apply_stage2_enhancements(plan: PagePlan) -> PagePlan:
    enhanced = PagePlan(
        page_number=plan.page_number,
        width_px=plan.width_px,
        height_px=plan.height_px,
        background_path=plan.background_path,
        source_type=plan.source_type,
        page_width_points=plan.page_width_points,
        page_height_points=plan.page_height_points,
    )
    enhanced.text_blocks = [block for block in plan.text_blocks if block.fit_verified]
    enhanced.image_blocks = list(plan.image_blocks)
    enhanced.effect_blocks = list(plan.effect_blocks)
    return enhanced
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage2_enhance.py -v --basetemp .tmp-pytest/stage2-task4`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_stage2_enhance.py skills/pdf-image-to-editable-ppt/scripts/stage2_enhance.py
git commit -m "feat: add stage2 page enhancement pass"
```

### Task 5: Extend PPT output for verified text and mapped effects

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/build_ppt.py`
- Test: `tests/test_build_ppt_stage2.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from PIL import Image
from conftest import load_skill_module


models = load_skill_module("models")
build_ppt = load_skill_module("build_ppt")

PagePlan = models.PagePlan
TextBlock = models.TextBlock


def test_build_presentation_accepts_fit_verified_text_blocks(tmp_path):
    background = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(background)
    plan = PagePlan(page_number=1, width_px=100, height_px=100, background_path=str(background))
    plan.text_blocks.append(
        TextBlock(
            text="Title",
            left=10,
            top=10,
            width=50,
            height=20,
            font_size=16,
            color="#000000",
            alignment="left",
            confidence=0.95,
            font_name="Arial",
            fit_verified=True,
        )
    )
    output = tmp_path / "stage2.pptx"
    build_ppt.build_presentation([plan], output)
    assert output.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_ppt_stage2.py -v --basetemp .tmp-pytest/stage2-task5`
Expected: FAIL because the stage 2 text fields are not exercised by tests yet.

- [ ] **Step 3: Write minimal implementation**

Update `build_ppt.py` so that:

- verified text blocks (`fit_verified=True`) are treated as normal text blocks
- effect blocks are ignored if unmapped or unsupported
- mapped direct effects can be attached where `python-pptx` has native support

For the minimal step, it is enough to keep writing verified text blocks without regression and safely ignore unsupported effects.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_ppt.py tests/test_build_ppt_runtime.py tests/test_build_ppt_stage2.py -v --basetemp .tmp-pytest/stage2-task5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_ppt_stage2.py skills/pdf-image-to-editable-ppt/scripts/build_ppt.py
git commit -m "feat: support stage2 verified text output"
```

### Task 6: Wire stage 2 into the runtime entry point

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/convert_to_ppt.py`
- Test: `tests/test_convert_to_ppt_stage2.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from PIL import Image
from conftest import load_skill_module


convert_module = load_skill_module("convert_to_ppt")


def test_convert_to_ppt_supports_stage2_flag(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (50, 50), "white").save(image_path)
    output_path = tmp_path / "result.pptx"
    convert_module.convert_to_ppt(str(image_path), output_path, enable_stage2=True)
    assert output_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convert_to_ppt_stage2.py -v --basetemp .tmp-pytest/stage2-task6`
Expected: FAIL because `convert_to_ppt` does not support a stage 2 flag yet.

- [ ] **Step 3: Write minimal implementation**

Update `convert_to_ppt.py`:

```python
def convert_to_ppt(input_path: str, output_path: Path, enable_stage2: bool = False) -> None:
    ...
```

When `enable_stage2` is true:

- run the stage 1 planning flow
- optionally validate text fitting
- optionally apply stage 2 enhancement pass
- then build the presentation

If stage 2 enhancement cannot verify a block, it must fall back to the stage 1 output.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convert_to_ppt.py tests/test_convert_to_ppt_stage2.py -v --basetemp .tmp-pytest/stage2-task6`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_convert_to_ppt_stage2.py skills/pdf-image-to-editable-ppt/scripts/convert_to_ppt.py
git commit -m "feat: wire stage2 enhancement flow"
```

### Task 7: Update documentation for stage 2 behavior and limits

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/SKILL.md`
- Modify: `skills/pdf-image-to-editable-ppt/references/README.md`
- Modify: `Course.md`
- Test: `tests/test_skill_content.py`
- Test: `tests/test_references_readme.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_skill_mentions_stage2_rules():
    text = Path("skills/pdf-image-to-editable-ppt/SKILL.md").read_text(encoding="utf-8")
    assert "fully identical" in text
    assert "stage 2" in text
    assert "effects" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_content.py::test_skill_mentions_stage2_rules -v --basetemp .tmp-pytest/stage2-task7`
Expected: FAIL because stage 2 rules are not documented yet.

- [ ] **Step 3: Write minimal implementation**

Update `SKILL.md` to describe:

- fully-identical-only text promotion
- stage 2 effects enhancement pass
- fail-closed fallback to the background layer

Update `references/README.md` to describe:

- stage 2 scope
- 2A vs 2B boundary
- current limitations around effects and exact text reproduction

Update `Course.md` so it reflects stage 2 plan status and current branch state.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skill_content.py tests/test_references_readme.py -v --basetemp .tmp-pytest/stage2-task7`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add skills/pdf-image-to-editable-ppt/SKILL.md skills/pdf-image-to-editable-ppt/references/README.md Course.md tests/test_skill_content.py tests/test_references_readme.py
git commit -m "docs: describe stage2 text and effect rules"
```

## Self-Review

- Spec coverage:
  - 文字严格拟合链由 Task 1、Task 2、Task 4、Task 6 覆盖。
  - 常见效果映射链由 Task 1、Task 3、Task 4、Task 5 覆盖。
  - 2A 的 fail-closed 回退策略由 Task 4、Task 6、Task 7 覆盖。
  - 2B 只作为本轮边界记录在文档中，没有被错误纳入实现任务。
- Placeholder scan:
  - 所有任务都包含明确文件、测试、命令和最小实现内容。
  - 没有 `TODO`、`TBD`、或“后续自行处理”式占位描述。
- Type consistency:
  - `EffectBlock`、`validate_text_block_fit`、`map_effect_type`、`apply_stage2_enhancements`、`enable_stage2` 在各任务中命名一致。
