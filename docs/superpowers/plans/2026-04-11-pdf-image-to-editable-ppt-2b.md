# PDF/Image To Editable PPT 2B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 2B-level page layering and vector candidate handling while keeping all complex or unreliable content fail-closed back to the background layer.

**Architecture:** 2B first introduces page-layer decomposition so later vector and blend reconstruction work has stable object boundaries and layer order. Then it adds a conservative vector-mapping path that only emits shape instructions for simple, independent vector candidates. Blend-heavy content remains grouped and explicitly marked for fallback unless a later phase proves it can be reconstructed safely.

**Tech Stack:** Python, Pillow, PyMuPDF, pytest

---

### Task 1: Add 2B data structures for layered objects and vector candidates

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/models.py`
- Test: `tests/test_models_2b.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


models = load_skill_module("models")
LayeredObject = models.LayeredObject
VectorInstruction = models.VectorInstruction


def test_layered_object_tracks_layer_metadata():
    obj = LayeredObject(
        object_type="vector",
        left=10.0,
        top=20.0,
        width=30.0,
        height=40.0,
        z_index=3,
        rebuildable=True,
        must_fallback=False,
    )
    assert obj.object_type == "vector"
    assert obj.z_index == 3


def test_vector_instruction_tracks_shape_payload():
    instruction = VectorInstruction(
        shape_type="freeform",
        left=1.0,
        top=2.0,
        width=3.0,
        height=4.0,
        payload={"points": [(0, 0), (1, 1)]},
    )
    assert instruction.shape_type == "freeform"
    assert instruction.payload["points"][1] == (1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_2b.py -v --basetemp .tmp-pytest/2b-task1`
Expected: FAIL because the new 2B dataclasses do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Update `models.py`:

```python
@dataclass(slots=True)
class LayeredObject:
    object_type: str
    left: float
    top: float
    width: float
    height: float
    z_index: int
    rebuildable: bool
    must_fallback: bool


@dataclass(slots=True)
class VectorInstruction:
    shape_type: str
    left: float
    top: float
    width: float
    height: float
    payload: dict = field(default_factory=dict)
```

Also extend `PagePlan` with:

```python
layered_objects: list[LayeredObject] = field(default_factory=list)
vector_instructions: list[VectorInstruction] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models_stage2.py tests/test_models_2b.py -v --basetemp .tmp-pytest/2b-task1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_models_2b.py skills/pdf-image-to-editable-ppt/scripts/models.py
git commit -m "feat: add 2b layering and vector models"
```

### Task 2: Implement page layering decomposition

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/layout_layers.py`
- Test: `tests/test_layout_layers.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


layout_layers = load_skill_module("layout_layers")


def test_decompose_page_layers_marks_large_background_region():
    result = layout_layers.decompose_page_layers(
        page_width=1000,
        page_height=1500,
        objects=[
            {"object_type": "background", "left": 0, "top": 0, "width": 1000, "height": 1500},
            {"object_type": "vector", "left": 20, "top": 20, "width": 100, "height": 50},
        ],
    )
    assert len(result) == 2
    assert result[0].object_type == "background"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout_layers.py -v --basetemp .tmp-pytest/2b-task2`
Expected: FAIL because `layout_layers.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `layout_layers.py`:

```python
from .models import LayeredObject


def decompose_page_layers(*, page_width: int, page_height: int, objects: list[dict]) -> list[LayeredObject]:
    layered = []
    for index, obj in enumerate(objects):
        layered.append(
            LayeredObject(
                object_type=obj["object_type"],
                left=float(obj["left"]),
                top=float(obj["top"]),
                width=float(obj["width"]),
                height=float(obj["height"]),
                z_index=index,
                rebuildable=obj["object_type"] != "background",
                must_fallback=obj["object_type"] == "complex_blend",
            )
        )
    return layered
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_layout_layers.py -v --basetemp .tmp-pytest/2b-task2`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_layout_layers.py skills/pdf-image-to-editable-ppt/scripts/layout_layers.py
git commit -m "feat: add 2b page layer decomposition"
```

### Task 3: Implement conservative vector mapping

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/vector_mapping.py`
- Test: `tests/test_vector_mapping.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


vector_mapping = load_skill_module("vector_mapping")


def test_map_vector_candidate_accepts_simple_independent_shape():
    result = vector_mapping.map_vector_candidate(
        {"object_type": "vector", "complexity": "simple", "rebuildable": True, "must_fallback": False}
    )
    assert result is not None
    assert result.shape_type == "freeform"


def test_map_vector_candidate_rejects_complex_fragmented_shape():
    result = vector_mapping.map_vector_candidate(
        {"object_type": "vector", "complexity": "fragmented", "rebuildable": True, "must_fallback": False}
    )
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vector_mapping.py -v --basetemp .tmp-pytest/2b-task3`
Expected: FAIL because `vector_mapping.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `vector_mapping.py`:

```python
from .models import VectorInstruction


def map_vector_candidate(candidate: dict):
    if candidate.get("object_type") != "vector":
        return None
    if candidate.get("complexity") != "simple":
        return None
    if not candidate.get("rebuildable", False):
        return None
    if candidate.get("must_fallback", False):
        return None
    return VectorInstruction(
        shape_type="freeform",
        left=float(candidate.get("left", 0.0)),
        top=float(candidate.get("top", 0.0)),
        width=float(candidate.get("width", 0.0)),
        height=float(candidate.get("height", 0.0)),
        payload={"source": "vector_candidate"},
    )
```

Also add:

```python
def map_vector_candidates(candidates: list[dict]) -> list[VectorInstruction]:
    return [instruction for candidate in candidates if (instruction := map_vector_candidate(candidate)) is not None]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vector_mapping.py -v --basetemp .tmp-pytest/2b-task3`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_vector_mapping.py skills/pdf-image-to-editable-ppt/scripts/vector_mapping.py
git commit -m "feat: add conservative vector mapping"
```

### Task 4: Add explicit fallback grouping for blend-heavy content

**Files:**
- Create: `skills/pdf-image-to-editable-ppt/scripts/blend_mapping.py`
- Test: `tests/test_blend_mapping.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


blend_mapping = load_skill_module("blend_mapping")


def test_group_blend_candidates_marks_complex_group_for_fallback():
    groups = blend_mapping.group_blend_candidates(
        [
            {"group_id": "g1", "effect_type": "opacity", "must_fallback": False},
            {"group_id": "g1", "effect_type": "blend_mode", "must_fallback": True},
        ]
    )
    assert groups[0]["must_fallback"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_blend_mapping.py -v --basetemp .tmp-pytest/2b-task4`
Expected: FAIL because `blend_mapping.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `blend_mapping.py`:

```python
def group_blend_candidates(candidates: list[dict]) -> list[dict]:
    grouped = {}
    for candidate in candidates:
        group_id = candidate.get("group_id", "default")
        state = grouped.setdefault(group_id, {"group_id": group_id, "items": [], "must_fallback": False})
        state["items"].append(candidate)
        if candidate.get("must_fallback", False):
            state["must_fallback"] = True
    return list(grouped.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_blend_mapping.py -v --basetemp .tmp-pytest/2b-task4`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_blend_mapping.py skills/pdf-image-to-editable-ppt/scripts/blend_mapping.py
git commit -m "feat: add blend fallback grouping"
```

### Task 5: Wire 2B layering and vector candidates into page plans

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/scripts/page_planner.py`
- Test: `tests/test_page_planner_2b.py`

- [ ] **Step 1: Write the failing test**

```python
from conftest import load_skill_module


page_planner = load_skill_module("page_planner")


def test_page_planner_accepts_layered_objects_and_vectors():
    plan = page_planner.build_page_plan(
        page_number=1,
        width_px=1000,
        height_px=1500,
        background_path="page.png",
        text_items=[],
        image_items=[],
        layered_objects=[{"object_type": "vector", "left": 0, "top": 0, "width": 10, "height": 10, "z_index": 0, "rebuildable": True, "must_fallback": False}],
        vector_instructions=[{"shape_type": "freeform", "left": 0, "top": 0, "width": 10, "height": 10, "payload": {"source": "vector_candidate"}}],
    )
    assert len(plan.layered_objects) == 1
    assert len(plan.vector_instructions) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_page_planner_2b.py -v --basetemp .tmp-pytest/2b-task5`
Expected: FAIL because `build_page_plan` does not accept 2B arguments yet.

- [ ] **Step 3: Write minimal implementation**

Update `page_planner.py`:

```python
from .models import ImageBlock, LayeredObject, PagePlan, TextBlock, VectorInstruction


def build_page_plan(..., layered_objects=None, vector_instructions=None):
    ...
    plan.layered_objects = [LayeredObject(**item) for item in (layered_objects or [])]
    plan.vector_instructions = [VectorInstruction(**item) for item in (vector_instructions or [])]
    return plan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_page_planner.py tests/test_page_planner_2b.py -v --basetemp .tmp-pytest/2b-task5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_page_planner_2b.py skills/pdf-image-to-editable-ppt/scripts/page_planner.py
git commit -m "feat: carry 2b layer and vector data in page plans"
```

### Task 6: Document 2B boundaries and current fail-closed behavior

**Files:**
- Modify: `skills/pdf-image-to-editable-ppt/SKILL.md`
- Modify: `skills/pdf-image-to-editable-ppt/references/README.md`
- Modify: `Course.md`
- Test: `tests/test_skill_content.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_skill_mentions_2b_layering_and_vector_boundaries():
    text = Path("skills/pdf-image-to-editable-ppt/SKILL.md").read_text(encoding="utf-8")
    assert "2B" in text
    assert "layering" in text
    assert "vector" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_content.py::test_skill_mentions_2b_layering_and_vector_boundaries -v --basetemp .tmp-pytest/2b-task6`
Expected: FAIL because 2B behavior is not documented yet.

- [ ] **Step 3: Write minimal implementation**

Update `SKILL.md` to describe:

- 2B layering before vector and blend reconstruction
- simple vector candidates only
- fail-closed fallback for blend-heavy content

Update `references/README.md` to describe:

- 2B-1 / 2B-2 / 2B-3 scope
- current implementation only lays groundwork for blend-heavy content

Update `Course.md` to reflect 2B plan status and current limitations.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skill_content.py tests/test_references_readme.py -v --basetemp .tmp-pytest/2b-task6`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add skills/pdf-image-to-editable-ppt/SKILL.md skills/pdf-image-to-editable-ppt/references/README.md Course.md tests/test_skill_content.py tests/test_references_readme.py
git commit -m "docs: describe 2b layering and vector boundaries"
```

## Self-Review

- Spec coverage:
  - 2B-1 页面分层由 Task 1、Task 2、Task 5 覆盖。
  - 2B-2 矢量重建候选由 Task 1、Task 3、Task 5 覆盖。
  - 2B-3 混合效果整组回退边界由 Task 4、Task 6 记录，但没有被错误扩大为完整重建实现。
- Placeholder scan:
  - 所有任务都包含明确文件、测试、命令和最小实现内容。
  - 没有 `TODO`、`TBD`、或“后续自行处理”式占位描述。
- Type consistency:
  - `LayeredObject`、`VectorInstruction`、`decompose_page_layers`、`map_vector_candidate`、`group_blend_candidates` 命名在计划中保持一致。
