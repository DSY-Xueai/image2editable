# Adaptive Fast Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce image and PDF conversion latency to a measured seconds/minutes target while preserving or improving the current Strict visual and editable-quality contract.

**Architecture:** Keep the current Strict pipeline as the reference implementation and add a deterministic local page router that chooses direct reconstruction, local prompted segmentation, or Strict processing. Fast output must pass the same render/quality gates; failures trigger local escalation and never flatten a whole page. Models move from disposable per-call workers to task-scoped resident workers, while native PDF objects bypass raster OCR where they are reliably extractable.

**Tech Stack:** Python 3.10-3.12, NumPy/OpenCV, PaddleOCR, Grounding DINO, SAM 2.1, Big-LaMa, pypdf/pypdfium2, python-pptx/Aspose assembly, pytest.

---

## Scope and File Map

The work is one coordinated conversion-speed project, delivered in independently testable phases:

- Create `image2editable/page_routing.py` for pure, explainable page signals and policy selection.
- Create `image2editable/worker_pool.py` for task-scoped resident OCR/DINO/SAM/LaMa workers.
- Create `image2editable/pdf_objects.py` for native PDF object extraction and support classification.
- Modify `scripts/performance_trace.py` and its synchronized copy under `skills/image-to-ppt/scripts/` for timing and route events.
- Modify `image_to_ppt.py` for policy-driven lazy segmentation, page-level LaMa, local escalation, and output artifact retention.
- Modify `image2editable/pdf_input.py`, `image2editable/legacy.py`, and `image2editable/runtime.py` for PDF routing, policy propagation, independent page scheduling, and cleanup.
- Modify `image2editable/inputs.py` and `image2editable/component_contracts.py` only where the new pipeline policy is persisted and validated.
- Add focused tests under `tests/` and extend `scripts/release_benchmark.py` with Fast/Strict comparison data.
- Update root `Course.md` after functional changes, as required by the repository instructions.

### Task 1: Establish a no-behavior-change performance and quality baseline

**Files:**
- Modify: `scripts/performance_trace.py`
- Modify: `skills/image-to-ppt/scripts/performance_trace.py`
- Modify: `image2editable/legacy.py:587-616`
- Modify: `image2editable/pdf_input.py:63-96,682-718`
- Modify: `image_to_ppt.py:1322-1483,2336-2667,3925-4335`
- Test: `tests/test_conversion_performance.py`
- Test: `tests/test_pdf_input.py`
- Test: `tests/test_runtime_execution.py`

- [ ] **Step 1: Add red tests for stage records and a page summary.**

```python
def test_page_summary_contains_only_safe_metrics(tmp_path):
    trace = _load_performance_trace().PerformanceTrace(tmp_path / "performance.jsonl")
    trace.event(
        "page_summary", page_id="page_001", route="strict",
        duration_ms=1200, sam_calls=2, lama_calls=1, worker_starts=4,
        host_wait_ms=0, token_count=0,
    )
    event = json.loads((tmp_path / "performance.jsonl").read_text().splitlines()[0])
    assert event["route"] == "strict"
    assert "source_path" not in event
    assert "prompt" not in event
```

- [ ] **Step 2: Run the focused test and verify it fails because `page_summary` is not an allowed event.**

Run: `pytest tests/test_conversion_performance.py::test_page_summary_contains_only_safe_metrics -q`

Expected: FAIL with `ValueError: unknown performance event`.

- [ ] **Step 3: Add a bounded `page_summary` event to both performance trace copies.** Define its required fields as `page_id`, `route`, `duration_ms`, `sam_calls`, `lama_calls`, `worker_starts`, `host_wait_ms`, and `token_count`; accept only non-negative integers and route identifiers matching the existing identifier validator. Keep schema version `1` so existing trace readers remain compatible.

- [ ] **Step 4: Wrap existing PDF render, `prepare_component_layers`, visual processing, quality finalization, and Agent plan calls with the existing `PerformanceTrace.span()` API.** Do not change stage order or model parameters. In `legacy.py`, pass the existing page trace through the preparation boundary; in `pdf_input.py`, add an optional trace parameter used only by runtime callers.

- [ ] **Step 5: Emit one page summary at the terminal page boundary using counts accumulated from stage spans.** Do not include source paths, image bytes, OCR text, prompts, masks, or model responses.

- [ ] **Step 6: Run the existing performance, PDF, runtime, and full contract tests.**

Run: `pytest tests/test_conversion_performance.py tests/test_pdf_input.py tests/test_runtime_execution.py -q`

Expected: all existing tests pass, and a real trace contains separate render/OCR/DINO/SAM/LaMa/quality durations without changing output files.

- [ ] **Step 7: Commit the baseline instrumentation.**

```bash
git add scripts/performance_trace.py skills/image-to-ppt/scripts/performance_trace.py image2editable/legacy.py image2editable/pdf_input.py image_to_ppt.py tests/test_conversion_performance.py tests/test_pdf_input.py tests/test_runtime_execution.py
git commit -m "性能：补充转换阶段基线埋点"
```

### Task 2: Add deterministic page understanding and policy selection

**Files:**
- Create: `image2editable/page_routing.py`
- Modify: `image_to_ppt.py:3925-4335`
- Modify: `image2editable/inputs.py:160-235`
- Modify: `image2editable/runtime.py:576-610,818-835`
- Modify: `image2editable/component_contracts.py`
- Test: `tests/test_page_routing.py`
- Test: `tests/test_runtime_cli.py`

- [ ] **Step 1: Write pure routing tests covering native PDF, direct layout, local refinement, and strict pages.**

```python
def test_high_confidence_regular_page_uses_direct_policy():
    result = classify_page(PageSignals(
        source_kind="image", ocr_items=14, ocr_mean_confidence=0.96,
        text_coverage=0.31, regular_geometry_ratio=0.82,
        overlap_ratio=0.01, transparency_ratio=0.0,
        edge_density=0.18, scan_noise=0.02,
    ))
    assert result.route == "direct"
    assert result.automatic_sam is False
    assert result.max_residual_rounds == 0

def test_low_confidence_page_uses_strict_policy_without_agent():
    result = classify_page(PageSignals(
        source_kind="image", ocr_items=0, ocr_mean_confidence=0.0,
        text_coverage=0.0, regular_geometry_ratio=0.08,
        overlap_ratio=0.42, transparency_ratio=0.35,
        edge_density=0.81, scan_noise=0.34,
    ))
    assert result.route == "strict"
    assert result.host_agent_allowed is False
```

- [ ] **Step 2: Run the routing tests and verify they fail because the module and policy types do not exist.**

Run: `pytest tests/test_page_routing.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `PageSignals`, `PagePolicy`, and `classify_page()` as deterministic dataclasses and pure functions.** `PagePolicy` must expose `route`, `automatic_sam`, `max_residual_rounds`, `hole_recheck`, `max_lama_calls`, and `host_agent_allowed`. The classifier may only use numeric signals and `source_kind`; it must return reasons and a confidence value in `[0, 1]`.

- [ ] **Step 4: Add a persisted `pipeline_mode` option with values `fast` and `strict`, defaulting to `strict` until benchmark approval.** Validate it in `inputs.py`, runtime manifest validation, and CLI contract tests. `fast` selects the router; `strict` constructs a policy that preserves current behavior.

- [ ] **Step 5: Compute signals from existing OCR results, page dimensions, candidate geometry, and low-resolution image statistics inside `prepare_component_layers()`.** Do not call Host Agent, DINO, SAM, or LaMa to compute signals. Persist only route, confidence, and bounded reason identifiers in `prepared_page.json`.

- [ ] **Step 6: Run routing and runtime contract tests.**

Run: `pytest tests/test_page_routing.py tests/test_runtime_cli.py tests/test_component_contracts.py -q`

Expected: all pass; existing calls without `pipeline_mode` remain strict-compatible.

- [ ] **Step 7: Commit the deterministic router and policy contract.**

```bash
git add image2editable/page_routing.py image_to_ppt.py image2editable/inputs.py image2editable/runtime.py image2editable/component_contracts.py tests/test_page_routing.py tests/test_runtime_cli.py tests/test_component_contracts.py
git commit -m "架构：增加页面理解路由策略"
```

### Task 3: Implement quality-gated lazy segmentation and page-level background repair

**Files:**
- Modify: `image_to_ppt.py:1008-1085,2336-2667,3925-4335`
- Modify: `scripts/visual_segment.py:1567-1765,1957-1990`
- Modify: `scripts/bg_model.py:489-590`
- Modify: `image2editable/legacy.py:504-584,587-616`
- Test: `tests/test_image_to_ppt.py`
- Test: `tests/test_visual_segment.py`
- Test: `tests/test_bg_model.py`
- Test: `tests/test_runtime_execution.py`

- [ ] **Step 1: Add tests proving strict policy retains the current automatic-SAM, three-round, and hole-recheck behavior, while fast policy disables those defaults.** Mock candidate generators and assert call counts rather than running model weights.

```python
def test_fast_policy_skips_automatic_sam_and_extra_residual_rounds(monkeypatch, fast_policy):
    calls = {"automatic": 0, "residual": 0, "holes": 0}
    monkeypatch.setattr("image_to_ppt.generate_mask_candidates", lambda *a, **k: calls.__setitem__("automatic", calls["automatic"] + 1) or [])
    monkeypatch.setattr("image_to_ppt.recheck_visual_element_holes", lambda *a, **k: calls.__setitem__("holes", calls["holes"] + 1))
    result = _run_process_image(policy=fast_policy, monkeypatch=monkeypatch)
    assert result is not None
    assert calls == {"automatic": 0, "residual": 0, "holes": 0}
```

- [ ] **Step 2: Run the focused tests and verify the policy argument is currently unsupported.**

Run: `pytest tests/test_image_to_ppt.py::test_fast_policy_skips_automatic_sam_and_extra_residual_rounds -q`

Expected: FAIL with a missing policy argument or unexpected call count.

- [ ] **Step 3: Thread `PagePolicy` through `prepare_component_layers()` and `_process_image()`.** Preserve the current strict defaults when no policy is supplied. In fast mode: execute geometry/direct components first, call prompted SAM only for unresolved regions, skip automatic SAM, set residual rounds from `max_residual_rounds`, and run hole recheck only when `hole_recheck` is true.

- [ ] **Step 4: Replace fixed `range(3)` residual processing with a bounded policy loop that stops immediately when the quality improvement metric is zero or negative.** Keep candidate validation, ownership, and existing quality gates unchanged.

- [ ] **Step 5: Change background repair to collect all page masks before invoking the large inpainter.** Keep OpenCV/local fill for small regions; invoke LaMa at most `max_lama_calls` times per page, cache the repaired image by source and mask digest, and reuse it for final and 16:9 assembly.

- [ ] **Step 6: Add a local escalation loop after rendering the candidate PPTX.** If any existing hard quality gate fails, discard only the candidate component assets, recompute the failed regions with strict local prompts, and rerun the same gate. Never suppress a failure by flattening the page.

- [ ] **Step 7: Run all visual and runtime tests, then run the existing full suite.**

Run: `pytest tests/test_image_to_ppt.py tests/test_visual_segment.py tests/test_bg_model.py tests/test_runtime_execution.py -q`

Expected: strict mode produces the same accepted/rejected decisions as before; fast mode passes synthetic tests and escalates any injected quality failure.

- [ ] **Step 8: Commit the policy-driven lazy pipeline.**

```bash
git add image_to_ppt.py scripts/visual_segment.py scripts/bg_model.py image2editable/legacy.py tests/test_image_to_ppt.py tests/test_visual_segment.py tests/test_bg_model.py tests/test_runtime_execution.py
git commit -m "性能：按需分割并合并页面背景修复"
```

### Task 4: Replace disposable model workers with task-scoped resident workers

**Files:**
- Create: `image2editable/worker_pool.py`
- Modify: `image_to_ppt.py:1294-1320,1538-2035,2148-2248`
- Modify: `scripts/object_worker.py`
- Modify: `scripts/sam_worker.py`
- Modify: `scripts/lama_worker.py`
- Modify: `scripts/text_detect.py`
- Modify: `image2editable/legacy.py:504-584`
- Test: `tests/test_worker_pool.py`
- Test: `tests/test_worker_resources.py`
- Test: `tests/test_runtime_execution.py`

- [ ] **Step 1: Write worker-pool tests for one model load, ordered requests, failure propagation, cancellation, and shutdown.** Use fake worker factories; no model weights are required.

```python
def test_task_pool_loads_model_once_and_reuses_worker():
    factory = FakeModelFactory()
    with ModelWorkerPool(factory, max_pending=2) as pool:
        assert pool.submit({"op": "ocr", "page_id": "page_001"}).result() == "ok"
        assert pool.submit({"op": "ocr", "page_id": "page_002"}).result() == "ok"
    assert factory.load_count == 1
    assert factory.request_count == 2
```

- [ ] **Step 2: Run the tests and verify the pool module is missing.**

Run: `pytest tests/test_worker_pool.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ModelWorkerPool` with a single child process per model, a bounded queue, request IDs, result/error envelopes, deadline-aware cancellation, and explicit `close()`/`terminate()` paths.** Load the model in the child before accepting requests. Keep the existing file-boundary validation for large binary masks; use shared temporary files only for payloads exceeding the pipe limit.

- [ ] **Step 4: Adapt OCR, DINO, SAM, and LaMa adapters to submit multiple page operations to the corresponding pool instead of spawning a new subprocess per call.** Preserve the existing worker CLI as a compatibility entry point for standalone use and failure recovery.

- [ ] **Step 5: Start one pool set per `execute_legacy()` task, pass it through page processing, and close it in a `finally` block.** Keep GPU concurrency at one for SAM Large on the 4060 8GB profile; allow CPU OCR and PDF object extraction to overlap with queued GPU work.

- [ ] **Step 6: Record `model_load_start/finish`, worker reuse, queue wait, and cancellation metrics in the performance trace.** Never record image content or request payloads.

- [ ] **Step 7: Run worker, runtime, and full tests.**

Run: `pytest tests/test_worker_pool.py tests/test_worker_resources.py tests/test_runtime_execution.py -q`

Expected: fake workers load once, requests remain ordered, a failed request does not corrupt later requests, and existing one-shot worker tests remain green.

- [ ] **Step 8: Commit resident workers.**

```bash
git add image2editable/worker_pool.py image_to_ppt.py scripts/object_worker.py scripts/sam_worker.py scripts/lama_worker.py scripts/text_detect.py image2editable/legacy.py tests/test_worker_pool.py tests/test_worker_resources.py tests/test_runtime_execution.py
git commit -m "性能：复用任务级视觉模型进程"
```

### Task 5: Add native PDF object extraction and page-independent scheduling

**Files:**
- Create: `image2editable/pdf_objects.py`
- Modify: `image2editable/pdf_input.py:63-180,682-718`
- Modify: `image2editable/legacy.py:504-584,3032-3200`
- Modify: `image2editable/runtime.py:671-710,1520-1690`
- Test: `tests/test_pdf_objects.py`
- Test: `tests/test_pdf_input.py`
- Test: `tests/test_runtime_execution.py`

- [ ] **Step 1: Add PDF fixture tests for text-only, image-only, vector, scanned, and mixed pages.** Assert that extraction returns bounded text spans, font metadata when present, image rectangles, path metadata, transforms, display order, and an explicit `unsupported_features` list.

```python
def test_native_text_page_is_classified_without_full_visual_reconstruction(tmp_path):
    pdf = _write_text_fixture(tmp_path / "text.pdf")
    page = inspect_pdf_page(pdf, 0)
    assert page.route == "native"
    assert page.text_runs[0].text == "native text"
    assert page.requires_raster_visuals is False
```

- [ ] **Step 2: Run the PDF object tests and verify the extractor is missing.**

Run: `pytest tests/test_pdf_objects.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `inspect_pdf_page()` using pypdf resource dictionaries and content operations.** Support text positioning/font lookup, image XObjects, basic path operators, page rotation/media/crop boxes, and display order. Mark transparency groups, filters, shadings, clipping paths, and unsupported operators instead of guessing.

- [ ] **Step 4: Update `prepare_pdf_job()` to inspect every page before rendering.** Native pages receive `native_objects.json` and a small preview only for quality comparison; scanned or unsupported pages receive the existing standard raster. Mixed pages keep native objects and rasterize only unsupported regions.

- [ ] **Step 5: Add native assembly in `legacy.py` for supported text, images, and basic paths.** Convert PDF coordinates to slide coordinates, preserve z-order, and send unsupported regions through the policy-driven local visual path. Do not silently convert an unsupported full page into one image.

- [ ] **Step 6: Change runtime page advancement so pages can prepare and process independently, report per-page progress, and continue other pages when one page is retrying.** Preserve durable state transitions and ensure cancellation closes worker pools.

- [ ] **Step 7: Run PDF, runtime, and assembly tests.**

Run: `pytest tests/test_pdf_objects.py tests/test_pdf_input.py tests/test_runtime_execution.py -q`

Expected: native fixtures bypass full OCR/SAM, mixed fixtures preserve native objects, scanned fixtures retain current visual quality gates, and page progress remains durable after restart.

- [ ] **Step 8: Commit native PDF routing and independent scheduling.**

```bash
git add image2editable/pdf_objects.py image2editable/pdf_input.py image2editable/legacy.py image2editable/runtime.py tests/test_pdf_objects.py tests/test_pdf_input.py tests/test_runtime_execution.py
git commit -m "性能：增加原生 PDF 快速通道"
```

### Task 6: Minimize successful-run artifacts without weakening recovery

**Files:**
- Modify: `image2editable/legacy.py:504-584,3032-3200`
- Modify: `image2editable/runtime.py:1988-2105,2210-2245`
- Modify: `image2editable/store.py`
- Modify: `image2editable/cli.py`
- Test: `tests/test_artifact_retention.py`
- Test: `tests/test_runtime_execution.py`

- [ ] **Step 1: Write retention tests for successful, failed, canceled, and debug runs.**

```python
def test_successful_run_keeps_output_and_minimal_summary(tmp_path):
    result = finish_fake_run(tmp_path, debug=False)
    assert result.output_path.is_file()
    assert (result.run_dir / "run_summary.json").is_file()
    assert not list((result.run_dir / "work").glob("**/*.png"))

def test_failed_run_keeps_minimal_diagnostics(tmp_path):
    result = finish_fake_run(tmp_path, debug=False, failed_page="page_001")
    assert (result.run_dir / "pages/page_001/reconstruction/quality-report.json").is_file()
    assert not (result.run_dir / "pages/page_001/reconstruction/residual-round-1").is_dir()
```

- [ ] **Step 2: Run retention tests and verify no retention policy exists.**

Run: `pytest tests/test_artifact_retention.py -q`

Expected: FAIL because successful runs still expose the full work tree.

- [ ] **Step 3: Implement a retention policy with `debug=False` by default.** Keep final PPTX, `run_summary.json`, page status, hashes, route/quality metrics, and the minimum artifacts required for durable recovery. Put masks, worker envelopes, intermediate PNGs, residual diagnostics, and full evidence under the disposable work root.

- [ ] **Step 4: Prune the disposable work root only after final output is atomically committed and reopen-validated.** On failure or cancellation, retain the failed page's source, quality report, one difference image, and route summary; debug mode retains the existing full evidence tree.

- [ ] **Step 5: Add a CLI `--debug-artifacts` switch and persist it in the manifest.** The default user path must not print or expose internal diagnostic file lists, while errors still include the run directory and concise failure reason.

- [ ] **Step 6: Run retention and runtime recovery tests.**

Run: `pytest tests/test_artifact_retention.py tests/test_runtime_execution.py -q`

Expected: successful output contains only final artifacts and minimum state; failed/canceled runs remain recoverable and contain no OCR text or prompt payloads in performance traces.

- [ ] **Step 7: Commit artifact retention.**

```bash
git add image2editable/legacy.py image2editable/runtime.py image2editable/store.py image2editable/cli.py tests/test_artifact_retention.py tests/test_runtime_execution.py
git commit -m "运行：精简成功任务中间产物"
```

### Task 7: Validate quality and speed before changing defaults

**Files:**
- Modify: `scripts/release_benchmark.py`
- Modify: `Course.md`
- Test: `tests/test_release_benchmark.py`
- Test: `tests/test_conversion_performance.py`
- Test: `tests/test_pdf_input.py`
- Test: `tests/test_runtime_execution.py`

- [ ] **Step 1: Add benchmark assertions for Fast/Strict pairs.** The report must include warm/cold timing, p50/p95 page time, 15-page aggregate time, model-load time, queue wait, SAM/LaMa counts, Host wait, token count, OCR accuracy, component coverage, render MAE/p95, PPTX reopen status, and failed-page count.

- [ ] **Step 2: Run the benchmark against the repository corpus plus one native PDF, one scanned PDF, one mixed PDF, one simple image, and one complex transparent/shadow image.** Store only aggregate metrics in the repository report; keep source images and full diagnostics outside Git.

- [ ] **Step 3: Require Fast output to match Strict on hard gates and stay within the same quality tolerances before enabling Fast by default.** A speed improvement alone cannot pass this gate. If any quality metric regresses, keep the default strict and adjust the router or local escalation policy.

- [ ] **Step 4: Run the complete fixed-environment suite.**

Run: `local-resources\\venv\\release-py312\\Scripts\\python.exe -m pytest -q`

Expected: all existing tests plus new routing/worker/PDF/retention tests pass; benchmark report contains no source content or model prompts.

- [ ] **Step 5: Update `Course.md` with the accepted default, route policy, worker lifecycle, artifact retention behavior, benchmark results, and remaining hardware/model validation notes.** Remove stale statements that imply every page always runs the full visual pipeline.

- [ ] **Step 6: Commit the benchmark gate and documentation.**

```bash
git add scripts/release_benchmark.py tests/test_release_benchmark.py tests/test_conversion_performance.py tests/test_pdf_input.py tests/test_runtime_execution.py Course.md
git commit -m "验证：建立快速转换质量门禁"
```

## Plan Self-Review

- Spec coverage: routing, selective SAM, page-level LaMa, resident workers, native PDF, Host Agent exclusion from the hot path, time budgets, artifact retention, and Fast/Strict quality comparison each have an implementation task.
- Quality safety: strict behavior remains the reference; every fast result is render-gated; no task permits whole-page image fallback.
- Token safety: Agent use is excluded from normal preparation and only existing structured diagnostics may be submitted when an explicit anomaly path is later added.
- Placeholder scan: no TODO/TBD steps; every code step names concrete files, interfaces, tests, and commands.
- Type consistency: `PageSignals` feeds `classify_page()` and returns `PagePolicy`; `pipeline_mode` selects strict or routed policy; worker pools expose submit/result/close; all are referenced consistently.
- Scope check: each task leaves a testable subsystem and can be reverted independently before the next task changes the default.

