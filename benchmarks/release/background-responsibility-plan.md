# Background Responsibility Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Every code task must also follow superpowers:test-driven-development. Work in the current release worktree and do not push.

**Goal:** Keep the SHA-bound pixel responsibility artifact fail-closed while extending it to bounded horizontal and vertical chart grid segments whose 3 px cores survive erosion.

**Architecture:** Background reconstruction creates an optional binary responsibility mask from actual retained pixels. A shared pure geometry helper recognizes the existing eroded edge plus bounded axis-aligned long segments; generation and quality each rebuild their candidate mask from separately validated inputs before calling it. Execution records keep carrying the artifact by hash, and existing executions without it retain their current semantics.

**Tech Stack:** Python 3.12, NumPy, OpenCV, Pillow, pytest, existing component repair artifact contracts.

---

Tasks 1–3 below record the completed removal of the page-wide boolean and the first
pixel responsibility implementation. Task 4 is the original acceptance task; its
strict replay did not complete and Task 7 supersedes it. Execute only Tasks 5–7 next.

### Task 1: Remove the page-wide authorization

**Files:**
- Modify: `image2editable/component_repair.py`
- Modify: `tests/test_component_quality.py`

- [ ] **Step 1: Replace the permissive test with an attack regression**

Create a large retained raster outside a small valid component and assert that the
quality report keeps `visual_ownership == "fail"` even when a rebuild action exists.

- [ ] **Step 2: Run the attack test and verify RED**

Run:
`E:\i2e-release-py312\Scripts\python.exe -m pytest tests/test_component_quality.py -k "background_rebuild and flattened" -q`

Expected: FAIL because `background_rebuild_approved=True` currently grants ownership.

- [ ] **Step 3: Delete the boolean path**

Remove the argument and the plan-derived call-site expression:

```python
background_rebuild_approved=(
    plan is not None
    and any(action["action"] == "rebuild_background" for action in plan["actions"])
)
```

Remove the corresponding whole-page `background_responsibility` calculation.

- [ ] **Step 4: Run the attack test and verify GREEN**

Expected: PASS with `visual_ownership == "fail"`.

### Task 2: Generate a bounded responsibility mask

**Files:**
- Modify: `image2editable/legacy.py`
- Modify: `tests/test_runtime_execution.py`

- [ ] **Step 1: Add failing rebuild tests**

Cover a 1–2 px grid line, a broad raster block, text overlap, active-component overlap,
and a page whose eligible pixels exceed 5%.

- [ ] **Step 2: Verify RED**

Run:
`E:\i2e-release-py312\Scripts\python.exe -m pytest tests/test_runtime_execution.py -k "background_responsibility" -q`

Expected: FAIL because no artifact is produced.

- [ ] **Step 3: Implement the smallest generator**

Use the exact masks already loaded by `_rebuild_canvas_background`:

```python
candidate = (
    foreground_evidence
    & ~text_repair
    & ~repairable_visual
    & np.all(rebuilt == source, axis=2)
)
thin = candidate & ~(
    cv2.erode(candidate.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
)
responsibility = thin if thin.mean() <= 0.05 else np.zeros_like(thin)
```

Write the result as an 8-bit binary PNG. When a previous bound mask exists, validate
its dimensions and union it before applying the same final constraints.

- [ ] **Step 4: Verify GREEN**

Expected: grid-line test passes; broad/text/component/over-budget tests remain empty.

### Task 3: Bind and independently validate the artifact

**Files:**
- Modify: `image2editable/component_contracts.py`
- Modify: `image2editable/component_repair.py`
- Modify: `image2editable/legacy.py`
- Modify: `tests/test_component_repair.py`
- Modify: `tests/test_component_quality.py`

- [ ] **Step 1: Add contract and tamper RED tests**

Test optional exact ref fields, SHA mismatch, malformed/non-binary PNG, wrong dimensions,
text overlap, pixels outside foreground evidence, a 3×3 solid core, and area over 5%.

- [ ] **Step 2: Verify RED**

Run:
`E:\i2e-release-py312\Scripts\python.exe -m pytest tests/test_component_repair.py tests/test_component_quality.py -k "background_responsibility" -q`

Expected: FAIL because the ref and validator do not exist.

- [ ] **Step 3: Extend the exact artifact contract**

Allow only these quality-ref sets:

```python
legacy
legacy | {"foreground_evidence"}
legacy | {"foreground_evidence", "background_responsibility"}
```

Decode the optional payload and pass a boolean mask to
`evaluate_component_quality_round(background_responsibility=...)`.

- [ ] **Step 4: Revalidate at the quality boundary**

Require binary values, matching dimensions, at most 5% coverage, no text overlap,
subset of refined foreground evidence, and no 3×3 solid core. Only then add the mask
to `generated_underlay_masks`.

- [ ] **Step 5: Verify GREEN and the surrounding suites**

Run the focused command, then:

`E:\i2e-release-py312\Scripts\python.exe -m pytest tests/test_component_repair.py tests/test_component_quality.py tests/test_runtime_execution.py tests/test_release_benchmark.py -q`

Expected: zero failures.

### Task 4: Original combo replay (superseded; do not execute)

**Files:**
- Modify: `Course.md` (ignored; never stage)
- Add or update: `benchmarks/release/plans/image-combo-chart--component-round-*.json`

- [ ] **Step 1: Reinstall the candidate package**

Run:
`E:\i2e-release-py312\Scripts\python.exe -m pip install --no-deps --force-reinstall --no-build-isolation .`

- [ ] **Step 2: Replay the combo-chart case from a fresh short E-drive run root**

Require every request/graph hash to match its fixed plan, `run_summary=completed`, and
`page_result=validated`. If a request changes, author a new evidence-based plan; never
weaken warning or ownership gates.

- [ ] **Step 3: Perform visual QA**

Render the PPTX, inspect the single slide, and run `slides_test.py`. Require intact CJK,
bars, line, legend and grid lines with no overflow.

- [ ] **Step 4: Request independent code review**

Critical and Important findings must be zero before commit.

- [ ] **Step 5: Run final verification and commit**

Run focused tests, Ruff, `py_compile`, `git diff --check`, and verify `Course.md` is
  ignored. Stage only production/tests/three fixed plans, then commit with a Chinese
  message. Do not push.

### Task 5: Add a linear-time axis-aligned grid geometry helper

**Files:**
- Modify: `image2editable/component_quality.py`
- Modify: `tests/test_component_quality.py`

- [ ] **Step 1: Write direct geometry RED tests**

Import the new private helper only in this focused test module. Build 900×1600 boolean
masks and assert the exact accepted pixels for:

```python
@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_background_geometry_accepts_three_pixel_grid_segments(orientation):
    candidate = np.zeros((900, 1600), dtype=bool)
    if orientation == "horizontal":
        candidate[300:303, 120:520] = True
    else:
        candidate[120:520, 300:303] = True
    assert np.array_equal(
        _background_responsibility_geometry(candidate), candidate
    )
```

Add separate negative cases for a 3 px line shorter than `min_length`, a 6 px thick
bar, a diagonal line, a curved line, and a broad rectangle. For the broad cases,
assert only the pre-existing `candidate & ~erode(candidate, 3×3)` edge remains and
the core is rejected. Add a crossed horizontal/vertical grid and assert its center is
accepted.

- [ ] **Step 2: Run the geometry tests and verify RED**

Run:
`E:\i2e-release-py312\Scripts\python.exe -m pytest tests/test_component_quality.py -k "background_geometry" -q`

Expected: collection or import FAIL because
`_background_responsibility_geometry` does not exist.

- [ ] **Step 3: Implement the pure helper**

Add this focused helper beside the existing component-quality morphology helpers:

```python
def _background_responsibility_geometry(candidate: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=bool)
    if candidate.ndim != 2:
        raise ValueError("background responsibility candidate is invalid")
    height, width = candidate.shape
    short_side = min(height, width)
    max_thickness = max(3, (short_side + 150) // 300)
    min_length = max(32, (short_side + 5) // 10)
    pixels = candidate.astype(np.uint8)
    core = cv2.erode(pixels, np.ones((3, 3), np.uint8)) > 0
    accepted = candidate & ~core

    for kernel_shape, horizontal in (
        ((1, min_length), True),
        ((min_length, 1), False),
    ):
        opened = cv2.morphologyEx(
            pixels, cv2.MORPH_OPEN, np.ones(kernel_shape, np.uint8)
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        keep = np.zeros(count, dtype=bool)
        for label in range(1, count):
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            major = component_width if horizontal else component_height
            minor = component_height if horizontal else component_width
            keep[label] = (
                major >= min_length
                and minor <= max_thickness
                and major >= 20 * minor
            )
        accepted |= candidate & keep[labels]
    return accepted
```

Keep this helper model-free and allocation-bounded. Do not add configuration, new
dependencies, rotation handling, or a general line detector.

- [ ] **Step 4: Add and pass the complexity regression**

Create thousands of isolated noise pixels, monkeypatch
`cv2.connectedComponentsWithStats` with a counting delegate, call the helper once,
and assert the delegate is called exactly twice. This locks two whole-page label
passes and prevents a per-component whole-page loop.

Run the focused command again. Expected: all `background_geometry` tests PASS.

- [ ] **Step 5: Commit the helper and direct tests**

```powershell
git add -- image2editable/component_quality.py tests/test_component_quality.py
git commit -m "质量：识别受约束的长直网格线"
```

### Task 6: Use the same geometry with independently reconstructed candidates

**Files:**
- Modify: `image2editable/legacy.py`
- Modify: `image2editable/component_repair.py`
- Modify: `tests/test_runtime_execution.py`
- Modify: `tests/test_component_quality.py`
- Modify: `Course.md` (ignored; never stage)

- [ ] **Step 1: Write generator and quality RED tests**

In `tests/test_runtime_execution.py`, add a real `_rebuild_canvas_background` case
containing a 3 px horizontal grid, a 3 px vertical grid, text overlap, active-component
overlap, a short segment, a diagonal stroke and a broad raster. Assert that the saved
PNG is exactly the helper result after text and active masks are removed.

In `tests/test_component_quality.py`, submit responsibility masks containing each
forbidden region and assert `ValueError("background responsibility mask is invalid")`.
Add an attack case with a broad source-backed raster containing long edges and assert
`visual_ownership == "fail"` because its interior remains unexplained.

- [ ] **Step 2: Run the integration tests and verify RED**

Run:
`E:\i2e-release-py312\Scripts\python.exe -m pytest tests/test_runtime_execution.py tests/test_component_quality.py -k "background_responsibility" -q`

Expected: the 3 px long grid assertions FAIL because generation and quality still
require an empty 3×3 eroded core.

- [ ] **Step 3: Replace generator-only erosion with the helper**

In `_rebuild_canvas_background`, retain the existing candidate construction and use:

```python
from image2editable.component_quality import (
    _background_responsibility_geometry,
    calibrate_page,
    refine_material_foreground,
)

responsibility = _background_responsibility_geometry(candidate)
if float(responsibility.mean()) > 0.05:
    responsibility = None
```

Keep the current rule that an over-budget mask is not written and no ref is published.

- [ ] **Step 4: Reconstruct and validate the quality candidate independently**

In `evaluate_component_quality_round`, first build `active_ownership` from the bound
component masks, then reconstruct the allowed candidate from quality-side inputs:

```python
allowed_candidate = (
    material_foreground
    & ~(np.asarray(text_mask) > 0)
    & ~active_ownership
    & np.all(np.asarray(source) == np.asarray(background), axis=2)
)
allowed_responsibility = _background_responsibility_geometry(allowed_candidate)
if np.any(responsibility & ~allowed_responsibility):
    raise ValueError("background responsibility mask is invalid")
```

Retain the existing dtype, shape, 5%, foreground, text, exact source/background and
component-overlap checks. Delete only the final blanket `np.any(core)` rejection that
the new exact allowed-mask comparison replaces.

- [ ] **Step 5: Verify focused and surrounding suites**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests/test_component_quality.py tests/test_component_repair.py tests/test_runtime_execution.py -q
E:\i2e-release-py312\Scripts\python.exe -m pytest tests/test_release_benchmark.py tests/test_benchmark_conversion.py -q
E:\i2e-release-py312\Scripts\python.exe -m ruff check image2editable/component_quality.py image2editable/component_repair.py image2editable/legacy.py tests/test_component_quality.py tests/test_runtime_execution.py
E:\i2e-release-py312\Scripts\python.exe -m py_compile image2editable/component_quality.py image2editable/component_repair.py image2editable/legacy.py
git diff --check
```

Expected: zero failures. Update ignored `Course.md` with the new geometry, test totals,
remaining strict-replay status and the unchanged 5%/SHA/model boundaries.

- [ ] **Step 6: Request code review and commit**

Require zero Critical and Important findings. Stage only the three production files
and two test files; verify `Course.md` remains ignored.

```powershell
git add -- image2editable/component_quality.py image2editable/component_repair.py image2editable/legacy.py tests/test_component_quality.py tests/test_runtime_execution.py
git commit -m "修复：保留受约束的三像素网格线"
```

### Task 7: Re-author and strictly replay the combo-chart plans

**Files:**
- Add or modify: `benchmarks/release/plans/image-combo-chart--component-round-*.json`
- Modify: `Course.md` (ignored; never stage)

- [ ] **Step 1: Rebuild and reinstall the current wheel**

Build from the current commit and force-reinstall it into
`E:\i2e-release-py312`. Verify installed production-file hashes match the checkout,
`pip check` passes, `IMAGE2EDITABLE_MODEL_CACHE=E:\image2editable-model-cache`, and
`image2editable doctor --agent-local` reports ready. Do not download or add models.

- [ ] **Step 2: Author plans only from fresh evidence**

Use a new short E-drive run root. Record the Host capability handshake, inspect every
component request and its visual evidence, and write actions for the exact
`request_sha256` and `graph_sha256`. Never reuse the stale round-03 hash, infer actions
from IDs alone, call Local Agent, or accept/discard everything as a shortcut.

- [ ] **Step 3: Replay from another fresh run root**

Run the same case using only the fixed Host plans. Require exact hash matches at every
round, `run_summary.status == "completed"`, `page_result.status == "validated"`, no
`preserved_with_warning`, no fallback, and an actual PPTX output. Any changed request
invalidates the plan and returns to Step 2.

- [ ] **Step 4: Perform visual and structural QA**

Render the output slide, compare it with the source, and verify the bars, conversion
line, legend, CJK text and full 3 px grid are intact. Run `slides_test.py` and the
release benchmark evaluator; both must pass without warnings.

- [ ] **Step 5: Run final verification and commit the case plans**

Run the focused component/runtime/release suites, Ruff, `py_compile`,
`git diff --check`, plan-schema tests and a candidate-wheel checkout-outside smoke.
Request final independent review and require zero Critical/Important findings. Update
ignored `Course.md`, stage only the validated fixed plans, and commit with a Chinese
message. Do not push.
