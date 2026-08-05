# Image-to-PSD Agent Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route image-to-PSD conversions through the existing Agent reconstruction and quality-gate pipeline, then assemble the accepted layers as PSD files.

**Architecture:** Image jobs record an explicit `output_format`. All OCR, segmentation, Agent repair and quality validation remain shared; only `assemble_legacy_results` selects the final PPTX or PSD writer. The PSD skill becomes a thin guide to the installed shared runtime and retains only PSD-specific assembly assets.

**Tech Stack:** Python 3.10–3.12, Pillow, NumPy, PaddleOCR/Tesseract, SAM 2.1, Aspose.PSD, pytest.

---

### Task 1: Add the image-job PSD output contract

**Files:**
- Modify: `image2editable/inputs.py`
- Modify: `image2editable/runtime.py`
- Modify: `image2editable/cli.py`
- Test: `tests/test_runtime_inputs.py`
- Test: `tests/test_runtime_cli.py`

- [ ] **Step 1: Write failing input-contract tests**

Add tests that call `prepare_image_job(..., output_format="psd")` and assert:

```python
assert manifest["output_format"] == "psd"
assert manifest["options"]["output_path"] == str(output.resolve())
```

Also assert `.pptx` is rejected for PSD, `.psd` is rejected for PPTX, PDF/PPTX inputs reject `output_format="psd"`, and an unsupported format raises `ValueError`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_runtime_inputs.py tests/test_runtime_cli.py -q
```

Expected: failures because `output_format` and `--format` do not exist.

- [ ] **Step 3: Implement the minimal contract**

Add `output_format: str = "pptx"` to `prepare_job`, `prepare_image_job`, and `convert`. Validate it against `{"pptx", "psd"}`. For PSD, require image input and validate a single-image target as `.psd`; for multiple images treat the target as an output directory. Store the normalized value in `job_manifest.json`.

Add CLI option:

```python
parser.add_argument("--format", choices=("pptx", "psd"), default="pptx")
```

Forward it as `output_format=args.format`. Keep all existing PPTX defaults unchanged; `slide_size` remains accepted but has no effect on PSD pixel dimensions.

- [ ] **Step 4: Run the focused tests**

Run the command from Step 2. Expected: all tests pass.

### Task 2: Assemble accepted Agent layers as PSD

**Files:**
- Modify: `scripts/psd_assemble.py`
- Modify: `image2editable/legacy.py`
- Test: `tests/test_psd_runtime.py`

- [ ] **Step 1: Write failing PSD assembly tests**

Create tests with a fake licensed PSD writer that verify:

```python
assert call["background_path"] == slide["background_original_path"]
assert call["components"] == slide["components"]
assert call["text_items"] == slide["text_items"]
assert call["img_width"] == slide["img_width"]
assert call["img_height"] == slide["img_height"]
```

Cover one image, multiple images, output collision, atomic staging cleanup, and rejection when any page is `preserved_with_warning`.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
python -m pytest tests/test_psd_runtime.py -q
```

Expected: failures because Runtime always invokes the PPTX assembler.

- [ ] **Step 3: Add license preflight**

Expose a small `preflight_psd_runtime()` in `scripts/psd_assemble.py` that calls `ensure_aspose_psd_license()` before any OCR/SAM work. Invoke it while preparing an image job whose output format is PSD. Tests monkeypatch this boundary and assert it runs before `RunStore.create`.

- [ ] **Step 4: Add PSD final assembly routing**

In `assemble_legacy_results`, retain the existing accepted-slide loading. When `manifest["output_format"] == "psd"`, stage one PSD per page and call:

```python
assemble_psd(
    background_path=slide["background_original_path"],
    components=slide["components"],
    text_items=slide["text_items"],
    img_width=slide["img_width"],
    img_height=slide["img_height"],
    output_path=staging,
)
```

Publish with the existing no-overwrite discipline, return authenticated output records in the run summary, and remove staging files on failure. Do not alter PPTX assembly.

- [ ] **Step 5: Run PSD and existing assembly regressions**

Run:

```bash
python -m pytest tests/test_psd_runtime.py tests/test_runtime_execution.py -q
```

Expected: all tests pass.

### Task 3: Replace the stale PSD skill pipeline

**Files:**
- Modify: `skills/image-to-psd/SKILL.md`
- Modify: `skills/image-to-psd/references/requirements.txt`
- Modify: `skills/image-to-psd/scripts/image_to_psd.py`
- Keep synchronized: `skills/image-to-psd/scripts/psd_assemble.py`
- Test: `tests/test_psd_skill.py`

- [ ] **Step 1: Write failing skill consistency tests**

Assert the skill no longer imports or instructs use of `build_background`, `extract_foreground_mask`, or `split_components`; assert it documents Host/Local Agent, five repair batches, image-only input, Aspose preflight, and `--format psd`.

- [ ] **Step 2: Verify the consistency tests fail**

Run:

```bash
python -m pytest tests/test_psd_skill.py -q
```

Expected: failures against the old standalone CV workflow.

- [ ] **Step 3: Make the skill a shared-runtime entry**

Replace the stale converter with a thin compatibility launcher that imports `image2editable.cli` from the installed project and adds `--format psd`. If the package is unavailable, fail with a direct installation message instead of silently using old CV code. Keep `psd_assemble.py` byte-identical to the root PSD assembler.

Update `SKILL.md` with these commands:

```bash
image2editable convert input.png -o output.psd --format psd --agent-provider local
image2editable prepare input.png -o output.psd --run-dir runs/psd-job --format psd --agent-provider host
image2editable run execute runs/psd-job
image2editable agent next runs/psd-job
image2editable agent record runs/psd-job --plan response.json
```

Document that only images are accepted, quality failure produces no PSD, and Aspose.PSD remains a licensed dependency.

- [ ] **Step 4: Run skill tests and an import smoke test**

Run:

```bash
python -m pytest tests/test_psd_skill.py -q
python skills/image-to-psd/scripts/image_to_psd.py --help
```

Expected: tests pass and help exits with status 0 without loading OCR/SAM.

### Task 4: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `Course.md`

- [ ] **Step 1: Update user documentation**

Describe the PSD image-only Agent workflow, output mapping, license preflight, Host/Local behavior, and lack of PDF/PPTX PSD input. Remove claims that PSD uses the old independent traditional-CV pipeline.

- [ ] **Step 2: Run targeted and full regression suites**

Run:

```bash
python -m pytest tests/test_psd_runtime.py tests/test_psd_skill.py tests/test_runtime_inputs.py tests/test_runtime_cli.py tests/test_runtime_execution.py -q
python -m pytest -q
```

Expected: all tests pass. If the ignored historical `test_agent_decision.py` helper is unavailable, run the full suite with its known local test directory temporarily added to `PYTHONPATH`; do not commit that ignored file.

- [ ] **Step 3: Check packaging and source consistency**

Run:

```bash
python -m build
git diff --check
```

Confirm the wheel contains `image2editable`, root `scripts/psd_assemble.py`, and both skill documents. Confirm root and PSD-skill `psd_assemble.py` hashes match.

- [ ] **Step 4: Update Course.md with verified facts**

Record the new PSD Agent path, changed files, commands, test counts, license limitation, and whether a real licensed PSD open test was available.
