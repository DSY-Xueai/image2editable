# Dual-Mask Component Underlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve unique source-pixel ownership while exporting complete movable components with deterministic, artifact-free generated underlays.

**Architecture:** Keep the component graph mask as the immutable ownership mask. Derive hash-bound presentation RGBA assets per round from ownership, parent semantic support, higher-z occluders, frozen text, and the page-local text-clean RGB; quality and final assembly consume those exact assets rather than reconstructing different layers later.

**Tech Stack:** Python 3, NumPy, OpenCV, Pillow, pytest, existing component graph/RunStore contracts, python-pptx only for project runtime delivery verification.

---

## File map

- Create `scripts/component_underlay.py`: deterministic underlay derivation, dual inpaint candidate selection, metrics and manifest helpers.
- Create `skills/image-to-ppt/scripts/component_underlay.py`: byte-for-byte runtime mirror.
- Modify `image2editable/legacy.py`: derive per-round presentation assets, render Agent evidence from them, bind them into quality inputs, and assemble the accepted assets.
- Modify `image2editable/component_contracts.py` and `skills/image-to-ppt/scripts/component_contracts.py`: require the presentation manifest in evidence and quality refs.
- Modify `image2editable/component_repair.py`: verify presentation refs and evaluate the exact presentation assets.
- Modify `image2editable/component_quality.py` and `skills/image-to-ppt/scripts/component_quality.py`: add generated-underlay metrics and hard gates without weakening ownership checks.
- Modify `tests/test_component_repair.py`, `tests/test_component_quality.py`, `tests/test_runtime_execution.py`, and `tests/test_skill_sync.py`: TDD, integration, tamper, freeze, delivery, and mirror parity coverage.
- Modify `Course.md`: record implemented behavior, verification facts, entry points and remaining real-file acceptance.

### Task 1: Deterministic presentation-layer engine

**Files:**
- Create: `scripts/component_underlay.py`
- Create: `skills/image-to-ppt/scripts/component_underlay.py`
- Test: `tests/test_component_repair.py`

- [ ] **Step 1: Write failing tests for bounded underlay derivation**

Add tests that build a 96×64 horizontal/vertical gradient, a semantic rounded rectangle, a unique ownership mask with a central child hole, and a higher-z occluder. Assert that `build_presentation_layer()` returns exactly these fields and never expands beyond semantic support:

```python
layer = build_presentation_layer(
    source_rgb=source,
    text_clean_rgb=text_clean,
    ownership_mask=ownership,
    semantic_mask=semantic,
    higher_layer_mask=child,
    text_mask=text,
)
assert set(layer) == {
    "rgb", "ownership_mask", "presentation_alpha_mask",
    "generated_underlay_mask", "metrics",
}
assert np.array_equal(layer["ownership_mask"], ownership)
assert not np.any(layer["presentation_alpha_mask"] & ~semantic)
assert np.array_equal(
    layer["generated_underlay_mask"],
    semantic & ~ownership & (child | text),
)
assert np.all(layer["presentation_alpha_mask"][layer["generated_underlay_mask"]])
```

Add a regression asserting the old nearest-donor fill produces a visible seam while the new result stays below the synthetic gradient tolerance:

```python
boundary = cv2.dilate(repair.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
boundary &= ~repair
assert layer["metrics"]["boundary_color_mae"] <= 3.0
assert layer["metrics"]["gradient_jump_p95"] <= 6.0
```

- [ ] **Step 2: Run the RED tests**

Run:

```powershell
python -m pytest -q tests/test_component_repair.py -k "presentation_layer or underlay_gradient"
```

Expected: collection/import failure because `scripts.component_underlay` does not exist.

- [ ] **Step 3: Implement the minimal page-local engine**

Implement shape/dtype validation and this public API in `scripts/component_underlay.py`:

```python
def build_presentation_layer(
    *,
    source_rgb: np.ndarray,
    text_clean_rgb: np.ndarray,
    ownership_mask: np.ndarray,
    semantic_mask: np.ndarray,
    higher_layer_mask: np.ndarray,
    text_mask: np.ndarray,
) -> dict:
    ownership = np.asarray(ownership_mask, dtype=bool)
    semantic = np.asarray(semantic_mask, dtype=bool)
    text_hole = semantic & ~ownership & np.asarray(text_mask, dtype=bool)
    visual_hole = (
        semantic & ~ownership
        & np.asarray(higher_layer_mask, dtype=bool)
        & ~text_hole
    )
    generated = text_hole | visual_hole
    rgb = np.asarray(text_clean_rgb, dtype=np.uint8).copy()
    if np.any(visual_hole):
        rgb, metrics = _choose_visual_fill(rgb, visual_hole, semantic)
    else:
        metrics = _empty_metrics()
    return {
        "rgb": rgb,
        "ownership_mask": ownership,
        "presentation_alpha_mask": ownership | generated,
        "generated_underlay_mask": generated,
        "metrics": metrics,
    }
```

`_choose_visual_fill()` must crop to the semantic bbox plus eight pixels, run `cv2.inpaint(..., 3, cv2.INPAINT_TELEA)` and `cv2.inpaint(..., 3, cv2.INPAINT_NS)`, score both with the tuple `(boundary_color_mae, gradient_jump_p95, added_high_frequency_pixels)`, and choose Telea on an exact tie. It must return finite numeric metrics and never modify pixels outside `visual_hole`.

Copy the completed file byte-for-byte to `skills/image-to-ppt/scripts/component_underlay.py` using `Copy-Item` only as a mechanical mirror step.

- [ ] **Step 4: Run GREEN and mirror tests**

Run:

```powershell
python -m pytest -q tests/test_component_repair.py -k "presentation_layer or underlay_gradient"
python -m pytest -q tests/test_skill_sync.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/component_underlay.py skills/image-to-ppt/scripts/component_underlay.py tests/test_component_repair.py
git commit -m "新增确定性组件底图重建"
```

### Task 2: Bind presentation assets into every Agent round

**Files:**
- Modify: `image2editable/legacy.py:463-584,657-788,1055-1148,1250-1325`
- Modify: `image2editable/component_contracts.py:10-24,248-255`
- Modify: `skills/image-to-ppt/scripts/component_contracts.py:10-24,248-255`
- Test: `tests/test_runtime_execution.py`
- Test: `tests/test_component_repair.py`

- [ ] **Step 1: Write failing request/evidence and tamper tests**

Extend the existing host E2E fixture so round 1 and round 2 must contain `presentation-manifest.json`. Assert its source and graph hashes match the request and each active component entry contains hash-bound RGBA, ownership, alpha, and generated-underlay assets:

```python
manifest = json.loads((round_dir / "presentation-manifest.json").read_text())
assert manifest["source_sha256"] == request["source_sha256"]
assert manifest["graph_sha256"] == request["graph_sha256"]
assert set(manifest["components"][0]) == {
    "component_id", "rgba", "ownership_mask",
    "presentation_alpha_mask", "generated_underlay_mask", "metrics",
}
assert "presentation-manifest.json" in request["evidence"]
```

Add a negative test that changes one manifest asset byte after publication and expects the next state transition to fail with a presentation asset hash mismatch.

- [ ] **Step 2: Run the RED tests**

```powershell
python -m pytest -q tests/test_runtime_execution.py tests/test_component_repair.py -k "presentation_manifest or presentation_asset_tamper"
```

Expected: evidence-set validation fails because the manifest is not yet produced or accepted.

- [ ] **Step 3: Derive and publish exact presentation assets**

Add `_build_presentation_assets()` to `legacy.py`. It must:

```python
resolved = resolve_visual_mask_ownership(active_nodes, active_masks)
for index, (node, ownership) in enumerate(zip(active_nodes, resolved, strict=True)):
    semantic = parent_masks.get(node.get("parent_id"), ownership)
    higher = np.logical_or.reduce([
        other_mask for other, other_mask in zip(active_nodes, resolved, strict=True)
        if other["z_index"] > node["z_index"]
    ], initial=np.zeros_like(ownership))
    layer = build_presentation_layer(
        source_rgb=source,
        text_clean_rgb=text_clean,
        ownership_mask=ownership,
        semantic_mask=semantic,
        higher_layer_mask=higher,
        text_mask=text_mask,
    )
```

Write each layer into a new per-round `presentation-assets/` directory and write `presentation-manifest.json` with relative POSIX paths and SHA-256 values. Use exclusive/new-directory publication; do not overwrite prior rounds or frozen assets. Initial and later Agent rounds must call the same helper.

Change `_render_component_evidence()` to load the manifest-bound RGBA for `component-isolation.png`; use the presentation assets for `reconstructed.png`, while `ownership.png` continues to show only source ownership. Add `presentation-manifest.json` to `EVIDENCE_NAMES` in both contract copies and to quality input refs.

- [ ] **Step 4: Run GREEN and contract regressions**

```powershell
python -m pytest -q tests/test_runtime_execution.py tests/test_component_repair.py tests/test_component_contracts.py -k "presentation or component_request or quality_input"
python -m pytest -q tests/test_skill_sync.py
```

Expected: all selected tests pass and mirror parity remains exact.

- [ ] **Step 5: Commit**

```powershell
git add image2editable/legacy.py image2editable/component_contracts.py skills/image-to-ppt/scripts/component_contracts.py tests/test_runtime_execution.py tests/test_component_repair.py tests/test_component_contracts.py
git commit -m "接入逐轮双遮罩组件证据"
```

### Task 3: Evaluate ownership and generated underlay separately

**Files:**
- Modify: `image2editable/component_quality.py:108-172,650-840`
- Modify: `skills/image-to-ppt/scripts/component_quality.py:108-172,650-840`
- Modify: `image2editable/component_repair.py:741-1000`
- Test: `tests/test_component_quality.py`
- Test: `tests/test_component_repair.py`

- [ ] **Step 1: Write failing quality-gate tests**

Add tests proving:

1. Real ownership overlap still yields `component_overlap`.
2. Presentation alpha overlap is allowed only where both layers declare the pixel as generated underlay.
3. Underlay outside the parent semantic mask yields `underlay_out_of_bounds`.
4. A deliberately striped or glyph-shaped fill yields `underlay_seam` or `component_text_residual`.
5. A clean gradient underlay passes without weakening existing duplicate-shadow checks.

Use strict expected metrics:

```python
assert set(report["metrics"]) >= {
    "generated_underlay_pixels",
    "underlay_out_of_bounds_pixels",
    "underlay_boundary_color_mae",
    "underlay_gradient_jump_p95",
    "underlay_added_high_frequency_pixels",
}
```

- [ ] **Step 2: Run the RED tests**

```powershell
python -m pytest -q tests/test_component_quality.py tests/test_component_repair.py -k "underlay or presentation_overlap"
```

Expected: missing metric/schema or incorrect overlap failure.

- [ ] **Step 3: Add the hard gates without changing confidence semantics**

Extend `evaluate_component()` with keyword-only `presentation_alpha_mask`, `generated_underlay_mask`, and `underlay_metrics`. Keep all missing/duplicate/text computations on `component_mask` (ownership). Validate:

```python
underlay_outside = generated_underlay_mask & ~parent_mask
if np.any(underlay_outside):
    violations.append("underlay_out_of_bounds")
if underlay_metrics["boundary_color_mae"] > calibration.hard_pixel_tolerance * 2:
    violations.append("underlay_seam")
if underlay_metrics["gradient_jump_p95"] > calibration.hard_pixel_tolerance * 4:
    violations.append("underlay_gradient_break")
high_frequency_limit = max(4, round(np.count_nonzero(generated_underlay_mask) * 0.005))
if underlay_metrics["added_high_frequency_pixels"] > high_frequency_limit:
    violations.append("underlay_patch")
```

Update quality report validation and improvement fields. In `component_repair._recompute_quality_artifact()`, verify and decode the manifest plus every bound asset, then pass exact ownership/presentation/underlay data into quality evaluation. Agent confidence must remain unable to remove these violations.

Mirror `component_quality.py` byte-for-byte.

- [ ] **Step 4: Run GREEN and all quality regressions**

```powershell
python -m pytest -q tests/test_component_quality.py tests/test_component_repair.py
python -m pytest -q tests/test_skill_sync.py
```

Expected: all tests pass; no existing ownership, text residual, frozen-state, or five-round tests regress.

- [ ] **Step 5: Commit**

```powershell
git add image2editable/component_quality.py skills/image-to-ppt/scripts/component_quality.py image2editable/component_repair.py tests/test_component_quality.py tests/test_component_repair.py
git commit -m "新增组件底图连续性质量门禁"
```

### Task 4: Assemble the exact accepted presentation assets

**Files:**
- Modify: `image2editable/legacy.py:1543-1628`
- Test: `tests/test_runtime_execution.py`
- Test: `tests/test_component_repair.py`

- [ ] **Step 1: Write failing assembly and freeze tests**

Add an E2E test with one base component, one higher-z child and one editable text box. After quality acceptance, assert the PPTX assembly input uses the manifest RGBA bytes rather than rebuilding alpha from graph ownership:

```python
accepted = legacy._accepted_slide_data(store, reconstruction, prepared, result)
base = next(item for item in accepted["components"] if item["component_id"] == "base")
with Image.open(base["path"]) as image:
    rgba = np.asarray(image.convert("RGBA"))
assert np.all(rgba[child_hole, 3] == 255)
assert not np.array_equal(rgba[child_hole, :3], source[child_hole])
```

Add tamper tests for a changed manifest, changed RGBA, mismatched component ID, and a later round attempting to replace a frozen component asset.

- [ ] **Step 2: Run the RED tests**

```powershell
python -m pytest -q tests/test_runtime_execution.py tests/test_component_repair.py -k "accepted_presentation or frozen_presentation or presentation_tamper"
```

Expected: `_accepted_slide_data()` still rebuilds alpha from ownership and fails the full-alpha assertion.

- [ ] **Step 3: Load accepted RGBA assets directly**

Replace the final per-node `np.dstack((reconstructed_image, mask * 255))` path with strict manifest loading:

```python
entry = presentation_by_id[component_id]
rgba_payload = _load_legacy_ref(store, entry["rgba"])[1]
with Image.open(io.BytesIO(rgba_payload)) as image:
    rgba = np.asarray(image.convert("RGBA")).copy()
if hashlib.sha256(rgba_payload).hexdigest() != entry["rgba"]["sha256"]:
    raise ValueError("accepted presentation RGBA hash mismatch")
```

Verify exact component ID order, bbox, z-index, source hash, graph hash, and frozen asset hashes before creating assembly files. Do not recompute or mutate accepted RGB/alpha. Preserve existing PowerPoint reopen and atomic publication checks.

- [ ] **Step 4: Run GREEN and runtime E2E regressions**

```powershell
python -m pytest -q tests/test_runtime_execution.py tests/test_component_repair.py tests/test_task10_runtime_e2e.py
```

Expected: all tests pass, including image/PDF/PPTX routing and native-object preservation coverage.

- [ ] **Step 5: Commit**

```powershell
git add image2editable/legacy.py tests/test_runtime_execution.py tests/test_component_repair.py
git commit -m "使用验收绑定的双遮罩组件导出"
```

### Task 5: Full verification, documentation and fresh real-file acceptance

**Files:**
- Modify: `Course.md`
- Runtime artifacts only: `tmp/task13-host-test1-r7/`

- [ ] **Step 1: Run formatting and focused suites**

```powershell
git diff --check
python -m pytest -q tests/test_component_quality.py tests/test_component_repair.py tests/test_runtime_execution.py tests/test_skill_sync.py
```

Expected: zero failures and no mirror diff.

- [ ] **Step 2: Run the full automated suite**

```powershell
python -m pytest -q
```

Expected: zero failures; record the exact pass/skip counts in `Course.md`.

- [ ] **Step 3: Run fresh `test1.pptx` acceptance**

Prepare a new `tmp/task13-host-test1-r7` Host Run with `--slide-size both`, record both full-slide background decisions, and execute one heavy page at a time. Do not resume r6. At every Agent round inspect `component-isolation.png`, `ownership.png`, `reconstructed.png`, `difference.png`, and `presentation-manifest.json`; stop after at most five real repair batches per page.

Acceptance requires editable text, minimal complete visual units, no text in visual components, no white holes/color bands/gray ghosts, and no `preserved_with_warning` on a page claimed successful.

- [ ] **Step 4: PowerPoint-native delivery verification**

Open both output variants with PowerPoint COM, confirm two slides and expected editable text/picture counts, export every slide to PNG, then create a copy with all text shapes deleted, save, close, reopen and export again. The pictures-only render must retain complete components while showing no raster text outlines or fill patches.

- [ ] **Step 5: Update `Course.md` and commit verified implementation**

Record current status, implemented dual-mask behavior, changed files, commands, exact test facts, r7 result, resource peak and remaining acceptance files. Then run:

```powershell
git add Course.md
git commit -m "验收双遮罩组件底图重建"
```

Do not push or merge to `main`.
