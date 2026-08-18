# Background Responsibility Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development task-by-task. Execute inline in the current release worktree; do not push.

**Goal:** Replace the unsafe page-wide rebuild authorization with a bounded, SHA-bound pixel responsibility artifact that preserves thin chart structure without accepting flattened raster output.

**Architecture:** Background reconstruction creates an optional binary responsibility mask from actual retained pixels. Execution records carry that artifact by hash, and the quality gate independently revalidates the mask before counting it as generated-underlay responsibility. Existing executions without the artifact retain their current semantics.

**Tech Stack:** Python 3.12, NumPy, OpenCV, Pillow, pytest, existing component repair artifact contracts.

---

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

### Task 4: Rebuild, replay, review, and commit case 5

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
