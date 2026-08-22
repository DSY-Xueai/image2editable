# Mixed Technical Separator OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve reliable OCR lines that combine whitespace-delimited semantic separators with embedded technical punctuation, so the mixed-page-size benchmark footer becomes native text without weakening strict quality gates.

**Architecture:** Keep the change inside the existing `_is_spaced_semantic_separator_text()` predicate. Split the line on whitespace-delimited `/` or `-`, validate each segment with a conservative character grammar, and leave OCR inference, confidence filtering, style estimation, component repair, and benchmark thresholds unchanged.

**Tech Stack:** Python 3.12, `re`, NumPy, pytest, setuptools wheel, image2editable release benchmark runner

---

### Task 1: Lock the mixed technical footer contract with failing tests

**Files:**
- Modify: `tests/test_ocr_isolation.py:382-440`

- [ ] **Step 1: Add the real footer to the accepted separator cases**

Add the footer string to `test_filter_noise_keeps_spaced_semantic_separators`:

```python
labels = [
    "LETTER PORTRAIT / MIXED SIZE",
    "ALPHA - BETA",
    "ALPHA / BETA / GAMMA",
    "842 x 595 pt / project-generated / CC0-1.0",
]
```

- [ ] **Step 2: Add punctuation boundary counterexamples**

Extend `test_filter_noise_rejects_ambiguous_spaced_separators` with:

```python
"ALPHA / BETA. GAMMA",
"ALPHA / beta-.gamma",
```

These prove that a period at a segment boundary and consecutive embedded punctuation remain invalid.

- [ ] **Step 3: Add a final-result regression for the exact OCR payload**

Add this test after `test_build_text_result_preserves_internal_semantic_separator`:

```python
def test_build_text_result_preserves_mixed_technical_separator(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        text_detect,
        "_estimate_style",
        lambda *_: {"font_size": 12.6, "color": "#3f576f", "bold": False},
    )
    image = np.full((80, 700, 3), 255, dtype=np.uint8)
    raw_boxes = [
        {
            "box": (10, 10, 581, 42),
            "text": "842 x 595 pt / project-generated / CC0-1.0",
            "confidence": 0.9892293214797974,
        }
    ]

    text_items, _ = text_detect._build_text_result(image, raw_boxes, 0.7, 2)

    assert [item["text"] for item in text_items] == [
        "842 x 595 pt / project-generated / CC0-1.0"
    ]
```

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_ocr_isolation.py -k "spaced_semantic_separators or mixed_technical_separator" -q
```

Expected: the accepted-footer filter test and exact final-result test fail because the current predicate rejects `.` anywhere in a spaced-separator line; the ambiguity test continues to pass.

### Task 2: Implement the segment grammar and synchronize the standalone skill

**Files:**
- Modify: `scripts/text_detect.py:594-612`
- Modify: `skills/image-to-ppt/scripts/text_detect.py:594-612`
- Modify: `Course.md:12-16`
- Test: `tests/test_ocr_isolation.py`

- [ ] **Step 1: Replace the broad punctuation rejection with segment validation**

Implement `_is_spaced_semantic_separator_text()` as:

```python
def _is_spaced_semantic_separator_text(text: str) -> bool:
    parts = re.split(r"\s+[/\-]\s+", text)
    if len(parts) == 1:
        return False

    for part in parts:
        meaningful = 0
        for index, char in enumerate(part):
            if (
                char.isalnum()
                or "\u4e00" <= char <= "\u9fff"
                or "\u3400" <= char <= "\u4dbf"
            ):
                meaningful += 1
                continue
            if char.isspace():
                continue
            if (
                char in ".-"
                and index > 0
                and index + 1 < len(part)
                and part[index - 1].isalnum()
                and part[index + 1].isalnum()
            ):
                continue
            return False
        if meaningful < 2:
            return False
    return True
```

Do not change `_filter_noise()`, confidence thresholds, font-size thresholds, or OCR model settings.

- [ ] **Step 2: Apply the identical implementation to the standalone skill mirror**

Make the same function-level edit in `skills/image-to-ppt/scripts/text_detect.py`, then verify byte identity:

```powershell
$rootHash = (Get-FileHash scripts/text_detect.py -Algorithm SHA256).Hash
$skillHash = (Get-FileHash skills/image-to-ppt/scripts/text_detect.py -Algorithm SHA256).Hash
if ($rootHash -ne $skillHash) { throw "text_detect.py mirrors differ" }
```

Expected: command exits zero and both hashes are identical.

- [ ] **Step 3: Run focused tests and verify GREEN**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_ocr_isolation.py -k "technical_labels or semantic_separator or malformed_or_low_confidence" -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Run the full related regression suite**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_ocr_isolation.py tests/test_regressions.py -q
D:\python\python.exe -m py_compile scripts/text_detect.py skills/image-to-ppt/scripts/text_detect.py
D:\python\Scripts\ruff.exe check --ignore E402 scripts/text_detect.py skills/image-to-ppt/scripts/text_detect.py tests/test_ocr_isolation.py
```

Expected: pytest, compilation, and lint all exit zero; no existing noise-filter regression changes result.

- [ ] **Step 5: Update the current project context**

Replace the stale separator paragraph in `Course.md` with evidence that embedded `.` and `-` are accepted only between alphanumeric characters, the real footer regression passes, and case 13 still requires a fresh installed-wheel author/replay cycle.

- [ ] **Step 6: Commit the behavior change**

Run:

```powershell
git add scripts/text_detect.py skills/image-to-ppt/scripts/text_detect.py tests/test_ocr_isolation.py
git diff --cached --check
git commit -m "修复：保留混合技术分隔文本"
```

Expected: one focused commit; ignored `Course.md` remains local project context and old untracked benchmark plans are not included.

### Task 3: Prove the installed Python 3.12 path and strict benchmark behavior

**Files:**
- Build artifact: `E:\rpdf13n-wheel\image2editable-0.2.0-py3-none-any.whl`
- Runtime evidence: a new directory under `E:\` distinct from previous failed/author runs
- Replace after validation: `benchmarks/release/plans/pdf-mixed-page-sizes--component-page*-round-*-gpu.json`

- [ ] **Step 1: Build and inspect a fresh wheel**

Run:

```powershell
D:\python\python.exe -m build --wheel --outdir E:\rpdf13n-wheel .
E:\i2e-release-py312\Scripts\python.exe -m zipfile -l E:\rpdf13n-wheel\image2editable-0.2.0-py3-none-any.whl
```

Expected: wheel build exits zero, includes the updated product script, and includes no `.pt`, `.pth`, `.bin`, `.safetensors`, or `.onnx` model payload.

- [ ] **Step 2: Reinstall into the E-drive Python 3.12 validation environment**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pip install --force-reinstall --no-deps E:\rpdf13n-wheel\image2editable-0.2.0-py3-none-any.whl
E:\i2e-release-py312\Scripts\python.exe -m pip check
E:\i2e-release-py312\Scripts\image2editable.exe doctor --agent-local
```

Expected: install and `pip check` exit zero; doctor reports ready with caches/models on E and no required dependency failure.

- [ ] **Step 3: Verify the installed package outside the checkout**

From a directory outside the repository, call installed `_build_text_result()` with the exact footer payload and assert it returns one native text item containing the unchanged string.

Expected: assertion succeeds without importing the checkout copy.

- [ ] **Step 4: Start a completely fresh host-author run for case 13**

Use the fixed input `benchmarks/release/inputs/13-mixed-page-sizes.pdf`, `--agent-provider host`, `--slide-size original`, the E-drive cache/temp variables, and a new run root. Complete the bounded capability handshake and every component repair round from fresh evidence.

Expected on both pages: native footer/title text is present where reliably recognized, `unexplained_visual_pixels=0`, no quality violation, warning, or fallback, and each page reaches `validated` without relaxing the manifest.

- [ ] **Step 5: Perform PPTX visual and structural QA**

Render the two-slide PPTX, inspect both rendered slides, and run the repository slide overflow/structure checker.

Expected: mixed source aspect ratios use the first-page canvas with contain placement on page 2; no cropping, stretching, overflow, duplicated text, or rasterized reliable OCR text.

- [ ] **Step 6: Replace stale plans and run three fresh strict replays**

Only after the author run validates, replace the six stale untracked plans with all newly required page/round plans and their current request/graph hashes. Run the manifest subset with fixed `repeat=3` in three independent workspaces.

Expected: formal evaluator reports all three repeats passed, both pages meet `min_components=6`, `min_text_boxes=4`, `max_unexplained_pixels=0`, and `max_quality_violations=0`, with no warning or fallback.

- [ ] **Step 7: Commit only validated benchmark plans and evidence metadata**

Inspect the worktree, ensure no model/cache/private run artifact is staged, update `Course.md` with measured timings and exact gate evidence, then commit the validated case-13 plans. Do not push.
