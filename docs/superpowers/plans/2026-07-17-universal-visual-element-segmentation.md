# Universal Visual Element Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace connected-component foreground grouping with general-purpose visual instance segmentation, export independent text-free RGBA elements, reconstruct a clean background, and always assemble a 16:9 PPT.

**Architecture:** SAM 2.1 Large generates multi-scale masks; OpenCV contributes only geometry candidates. Candidate normalization removes duplicates, preserves parent/child instances, and assigns each visible pixel to exactly one element. Existing OCR output is consumed but its detection, merging, font, and style logic remain unchanged.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, Pillow, python-pptx, PyTorch, official SAM 2.1, pytest.

---

## Working conventions

- Work only in `E:\My_project\Change_PPT\.worktrees\visual-segmentation` on branch `codex/visual-segmentation`.
- The project intentionally ignores `tests/` and `Course.md`. Local regression tests are required but must not be staged.
- Every modified main PPT script must be copied byte-for-byte to `skills/image-to-ppt/scripts/`.
- Do not modify `scripts/text_detect.py` or its skill copy.
- Do not restore the dense-layout flattened-success path.
- Before adding the first new test, add `import cv2`, `import json`, and
  `from scripts import fg_extract, ppt_assemble, visual_segment` to
  `tests/test_regressions.py`; retain its existing `bg_model` and `text_detect`
  imports.

## File map

- Create `scripts/visual_segment.py`: checkpoint resolution, SAM generator creation, multi-scale candidates, geometry candidates, candidate normalization, ownership, validation, and diagnostics.
- Modify `scripts/fg_extract.py`: export one supplied semantic mask as one text-free RGBA component.
- Modify `scripts/bg_model.py`: clean background from the union of final element masks and OCR text.
- Modify `scripts/ppt_assemble.py`: fixed 16:9 page and shared contain transform.
- Modify `image_to_ppt.py`: shared single-image processing pipeline and strict quality failure.
- Modify `requirements.txt`, `.gitignore`, `README.md`, and skill dependency documentation.
- Create `THIRD_PARTY_NOTICES.md`, `third_party/licenses/SAM2-APACHE-2.0.txt`, and `CITATION.cff`.
- Test in local ignored `tests/test_regressions.py`.

### Task 1: SAM 2.1 runtime adapter and checkpoint cache

**Files:**
- Create: `scripts/visual_segment.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Test: `tests/test_regressions.py`

- [ ] **Step 1: Write failing cache and lazy-import tests**

```python
from pathlib import Path

import pytest

from scripts import visual_segment


def test_resolve_sam_checkpoint_downloads_once(tmp_path: Path) -> None:
    calls = []

    def fake_download(url: str, target: str) -> None:
        calls.append((url, target))
        Path(target).write_bytes(b"checkpoint")

    first = visual_segment.resolve_sam_checkpoint(tmp_path, fake_download)
    second = visual_segment.resolve_sam_checkpoint(tmp_path, fake_download)

    assert first == second
    assert first.name == "sam2.1_hiera_large.pt"
    assert len(calls) == 1


def test_create_sam_generator_reports_missing_dependency(monkeypatch) -> None:
    real_import = visual_segment.importlib.import_module

    def fake_import(name: str):
        if name.startswith("sam2"):
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(visual_segment.importlib, "import_module", fake_import)

    with pytest.raises(
        visual_segment.VisualSegmentationError,
        match="SAM 2.1 is required",
    ):
        visual_segment.create_sam_generator(Path("missing.pt"), device="cpu")
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```powershell
python -m pytest -q tests/test_regressions.py -k "resolve_sam_checkpoint or create_sam_generator"
```

Expected: collection fails because `scripts.visual_segment` does not exist.

- [ ] **Step 3: Implement the runtime adapter**

Create `scripts/visual_segment.py` with these public definitions:

```python
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import urlretrieve

import numpy as np

SAM21_LARGE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_large.pt"
)
SAM21_LARGE_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class VisualSegmentationError(RuntimeError):
    pass


@dataclass
class MaskCandidate:
    mask: np.ndarray
    score: float
    source: str


@dataclass
class VisualElement:
    mask: np.ndarray
    z_index: int
    score: float
    source: str


def resolve_sam_checkpoint(
    cache_dir: str | Path | None = None,
    downloader: Callable[[str, str], object] = urlretrieve,
) -> Path:
    root = Path(
        cache_dir
        or os.environ.get(
            "IMAGE2EDITABLE_MODEL_CACHE",
            Path.home() / ".cache" / "image2editable",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "sam2.1_hiera_large.pt"
    if checkpoint.exists() and checkpoint.stat().st_size > 0:
        return checkpoint

    partial = checkpoint.with_suffix(".pt.part")
    try:
        downloader(SAM21_LARGE_URL, str(partial))
        if not partial.exists() or partial.stat().st_size == 0:
            raise VisualSegmentationError("SAM 2.1 checkpoint download was empty")
        partial.replace(checkpoint)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise VisualSegmentationError(
            f"Unable to download SAM 2.1 checkpoint: {exc}"
        ) from exc
    return checkpoint


def create_sam_generator(
    checkpoint_path: str | Path,
    device: str | None = None,
):
    try:
        torch = importlib.import_module("torch")
        build_module = importlib.import_module("sam2.build_sam")
        generator_module = importlib.import_module("sam2.automatic_mask_generator")
    except ModuleNotFoundError as exc:
        raise VisualSegmentationError(
            "SAM 2.1 is required. Install project segmentation dependencies."
        ) from exc

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_module.build_sam2(
        SAM21_LARGE_CONFIG,
        str(checkpoint_path),
        device=selected_device,
        apply_postprocessing=False,
    )
    return generator_module.SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,
        points_per_batch=16 if selected_device == "cuda" else 4,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=20,
    )
```

- [ ] **Step 4: Declare the dependency and cache exclusion**

Append to `requirements.txt`:

```text
# General-purpose visual instance segmentation
SAM-2 @ git+https://github.com/facebookresearch/sam2.git@main
```

Append to `.gitignore`:

```gitignore
# Downloaded visual-segmentation checkpoints
*.pt
.model-cache/
```

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m pytest -q tests/test_regressions.py -k "resolve_sam_checkpoint or create_sam_generator"
python -m pytest -q
```

Expected: focused tests pass; full suite reports at least 44 passing tests.

- [ ] **Step 6: Commit runtime adapter**

```powershell
git add scripts/visual_segment.py requirements.txt .gitignore
git commit -m "feat: add SAM 2.1 segmentation runtime"
```

### Task 2: Multi-scale and geometry candidate generation

**Files:**
- Modify: `scripts/visual_segment.py`
- Test: `tests/test_regressions.py`

- [ ] **Step 1: Write failing candidate-generation tests**

```python
def test_generate_candidates_maps_crop_masks_to_full_image() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)

    class FakeGenerator:
        def generate(self, crop):
            mask = np.zeros(crop.shape[:2], dtype=bool)
            mask[5:15, 7:17] = True
            return [{
                "segmentation": mask,
                "predicted_iou": 0.9,
                "stability_score": 0.95,
            }]

    candidates = visual_segment.generate_mask_candidates(
        image,
        FakeGenerator(),
        crop_size=64,
        overlap=16,
        include_geometry=False,
    )

    assert candidates
    assert all(item.mask.shape == image.shape[:2] for item in candidates)
    assert any(np.any(item.mask[:, 64:]) for item in candidates)


def test_geometry_candidates_include_closed_flat_shape() -> None:
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (55, 55), (20, 80, 180), thickness=-1)

    candidates = visual_segment.generate_geometry_candidates(image, min_area=40)

    assert any(np.count_nonzero(item.mask[25:50, 25:50]) > 400 for item in candidates)
```

- [ ] **Step 2: Run the tests and confirm RED**

```powershell
python -m pytest -q tests/test_regressions.py -k "generate_candidates or geometry_candidates"
```

Expected: FAIL because candidate-generation functions are absent.

- [ ] **Step 3: Implement deterministic crop iteration and SAM mapping**

Add:

```python
def _crop_origins(length: int, crop_size: int, overlap: int) -> list[int]:
    if length <= crop_size:
        return [0]
    step = max(crop_size - overlap, 1)
    origins = list(range(0, max(length - crop_size, 0) + 1, step))
    last = length - crop_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def generate_mask_candidates(
    image: np.ndarray,
    generator,
    crop_size: int = 768,
    overlap: int = 128,
    include_geometry: bool = True,
) -> list[MaskCandidate]:
    height, width = image.shape[:2]
    candidates: list[MaskCandidate] = []

    crops = [(0, 0, width, height)]
    if width > crop_size or height > crop_size:
        for y in _crop_origins(height, min(crop_size, height), overlap):
            for x in _crop_origins(width, min(crop_size, width), overlap):
                crops.append((
                    x,
                    y,
                    min(x + crop_size, width),
                    min(y + crop_size, height),
                ))

    seen_crops = set()
    for x1, y1, x2, y2 in crops:
        crop_key = (x1, y1, x2, y2)
        if crop_key in seen_crops:
            continue
        seen_crops.add(crop_key)
        crop = image[y1:y2, x1:x2]
        for record in generator.generate(crop):
            local = np.asarray(record["segmentation"], dtype=bool)
            full = np.zeros((height, width), dtype=bool)
            full[y1:y2, x1:x2] = local
            score = min(
                float(record.get("predicted_iou", 0.0)),
                float(record.get("stability_score", 0.0)),
            )
            candidates.append(MaskCandidate(full, score, "sam"))

    if include_geometry:
        candidates.extend(generate_geometry_candidates(image))
    return candidates
```

- [ ] **Step 4: Implement geometry candidates as supplements**

Add:

```python
def generate_geometry_candidates(
    image: np.ndarray,
    min_area: int = 20,
) -> list[MaskCandidate]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[MaskCandidate] = []
    total_area = image.shape[0] * image.shape[1]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > total_area * 0.90:
            continue
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        candidates.append(MaskCandidate(mask > 0, 0.70, "geometry"))
    return candidates
```

Add `import cv2` to `scripts/visual_segment.py`.

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m pytest -q tests/test_regressions.py -k "generate_candidates or geometry_candidates"
python -m pytest -q
```

Expected: both new tests and all prior tests pass.

- [ ] **Step 6: Commit candidate generation**

```powershell
git add scripts/visual_segment.py
git commit -m "feat: generate multi-scale visual masks"
```

### Task 3: Duplicate removal, parent-child separation, and unique ownership

**Files:**
- Modify: `scripts/visual_segment.py`
- Test: `tests/test_regressions.py`

- [ ] **Step 1: Write failing ownership tests**

```python
def test_resolve_elements_keeps_touching_instances_separate() -> None:
    left = np.zeros((30, 30), dtype=bool)
    right = np.zeros((30, 30), dtype=bool)
    left[5:25, 2:15] = True
    right[5:25, 15:28] = True

    elements = visual_segment.resolve_visual_elements([
        visual_segment.MaskCandidate(left, 0.9, "sam"),
        visual_segment.MaskCandidate(right, 0.9, "sam"),
    ])

    assert len(elements) == 2


def test_resolve_elements_removes_duplicate_masks() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:25, 5:25] = True

    elements = visual_segment.resolve_visual_elements([
        visual_segment.MaskCandidate(mask, 0.8, "geometry"),
        visual_segment.MaskCandidate(mask.copy(), 0.95, "sam"),
    ])

    assert len(elements) == 1
    assert elements[0].source == "sam"


def test_resolve_elements_assigns_each_pixel_once() -> None:
    parent = np.zeros((40, 40), dtype=bool)
    child = np.zeros((40, 40), dtype=bool)
    parent[4:36, 4:36] = True
    child[12:28, 12:28] = True

    elements = visual_segment.resolve_visual_elements([
        visual_segment.MaskCandidate(parent, 0.9, "sam"),
        visual_segment.MaskCandidate(child, 0.95, "sam"),
    ])
    ownership = np.sum([item.mask.astype(np.uint8) for item in elements], axis=0)

    assert len(elements) == 2
    assert int(ownership.max()) == 1
    assert np.count_nonzero(elements[-1].mask) == np.count_nonzero(child)
```

- [ ] **Step 2: Run the tests and confirm RED**

```powershell
python -m pytest -q tests/test_regressions.py -k "resolve_elements"
```

Expected: FAIL because `resolve_visual_elements` is absent.

- [ ] **Step 3: Implement duplicate filtering and ownership**

Add:

```python
def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / max(union, 1)


def resolve_visual_elements(
    candidates: list[MaskCandidate],
    min_area: int = 20,
    duplicate_iou: float = 0.92,
) -> list[VisualElement]:
    valid = [
        item
        for item in candidates
        if item.mask.dtype == bool
        and int(np.count_nonzero(item.mask)) >= min_area
        and np.count_nonzero(item.mask) / item.mask.size < 0.95
    ]
    valid.sort(key=lambda item: item.score, reverse=True)

    deduplicated: list[MaskCandidate] = []
    for item in valid:
        if any(_mask_iou(item.mask, kept.mask) >= duplicate_iou for kept in deduplicated):
            continue
        deduplicated.append(item)

    front_to_back = sorted(
        deduplicated,
        key=lambda item: (np.count_nonzero(item.mask), -item.score),
    )
    if not front_to_back:
        return []

    claimed = np.zeros(front_to_back[0].mask.shape, dtype=bool)
    visible_front_to_back: list[VisualElement] = []
    for item in front_to_back:
        visible = item.mask & ~claimed
        if np.count_nonzero(visible) < min_area:
            continue
        visible_front_to_back.append(
            VisualElement(visible, 0, item.score, item.source)
        )
        claimed |= visible

    back_to_front = list(reversed(visible_front_to_back))
    for index, item in enumerate(back_to_front):
        item.z_index = index
    return back_to_front
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest -q tests/test_regressions.py -k "resolve_elements"
python -m pytest -q
```

Expected: new ownership tests and full suite pass.

- [ ] **Step 5: Commit ownership logic**

```powershell
git add scripts/visual_segment.py
git commit -m "feat: assign visual pixels to independent elements"
```

### Task 4: Export text-free transparent components

**Files:**
- Modify: `scripts/fg_extract.py`
- Test: `tests/test_regressions.py`

- [ ] **Step 1: Write failing export tests**

```python
def test_export_visual_components_preserves_independent_masks(tmp_path: Path) -> None:
    image = np.full((40, 60, 3), 200, dtype=np.uint8)
    first = np.zeros((40, 60), dtype=bool)
    second = np.zeros((40, 60), dtype=bool)
    first[5:25, 5:25] = True
    second[10:35, 30:55] = True

    components = fg_extract.export_visual_components(
        image,
        [first, second],
        tmp_path,
        text_mask=np.zeros((40, 60), dtype=np.uint8),
    )

    assert len(components) == 2
    for component in components:
        rgba = np.asarray(Image.open(component["path"]).convert("RGBA"))
        assert np.any(rgba[:, :, 3] == 0)
        assert np.any(rgba[:, :, 3] > 0)
        assert int(rgba[0, 0, 3]) == 0


def test_export_visual_components_repairs_text_pixels(tmp_path: Path) -> None:
    image = np.full((40, 60, 3), 60, dtype=np.uint8)
    image[15:20, 20:40] = 250
    element = np.zeros((40, 60), dtype=bool)
    element[5:35, 5:55] = True
    text_mask = np.zeros((40, 60), dtype=np.uint8)
    text_mask[12:23, 17:43] = 255

    component = fg_extract.export_visual_components(
        image,
        [element],
        tmp_path,
        text_mask=text_mask,
    )[0]
    rgba = np.asarray(Image.open(component["path"]).convert("RGBA"))

    assert float(np.mean(rgba[10:20, 12:38, :3])) < 180
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
python -m pytest -q tests/test_regressions.py -k "export_visual_components"
```

Expected: FAIL because `export_visual_components` is absent.

- [ ] **Step 3: Implement one-mask-per-component export**

Add to `scripts/fg_extract.py`:

```python
def _refine_visual_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    if np.count_nonzero(binary) < 20:
        return binary > 0

    dilated = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    trimap = np.full(binary.shape, cv2.GC_BGD, dtype=np.uint8)
    trimap[dilated > 0] = cv2.GC_PR_BGD
    trimap[binary > 0] = cv2.GC_PR_FGD
    trimap[eroded > 0] = cv2.GC_FGD
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            img,
            trimap,
            None,
            bg_model,
            fg_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return binary > 0
    return (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)


def _soft_alpha(mask: np.ndarray) -> np.ndarray:
    hard = mask.astype(np.uint8) * 255
    eroded = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=1)
    feathered = cv2.GaussianBlur(hard, (3, 3), 0)
    feathered[eroded == 255] = 255
    feathered[~mask] = 0
    return feathered


def export_visual_components(
    img: np.ndarray,
    element_masks: list[np.ndarray],
    output_dir: str | Path,
    text_mask: np.ndarray,
    padding: int = 3,
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_ink = _build_text_ink_mask(img, text_mask)
    height, width = img.shape[:2]
    components = []

    for index, element_mask in enumerate(element_masks, start=1):
        mask = _refine_visual_mask(img, np.asarray(element_mask, dtype=bool))
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x1 = max(int(xs.min()) - padding, 0)
        y1 = max(int(ys.min()) - padding, 0)
        x2 = min(int(xs.max()) + 1 + padding, width)
        y2 = min(int(ys.max()) + 1 + padding, height)

        local_mask = mask[y1:y2, x1:x2]
        local_text = text_ink[y1:y2, x1:x2]
        local_text = np.where(local_mask, local_text, 0).astype(np.uint8)
        rgb = _repair_component_rgb(img[y1:y2, x1:x2], local_text)
        alpha = _soft_alpha(local_mask)
        rgba = np.dstack([rgb, alpha])

        path = output_dir / f"component_{index:04d}.png"
        Image.fromarray(rgba).save(path)
        components.append({
            "path": str(path),
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "area": int(np.count_nonzero(mask)),
            "z_index": index - 1,
        })
    return components
```

- [ ] **Step 4: Run focused and full tests**

```powershell
python -m pytest -q tests/test_regressions.py -k "export_visual_components"
python -m pytest -q
```

Expected: transparent component and text-repair tests pass; no regression.

- [ ] **Step 5: Commit component export**

```powershell
git add scripts/fg_extract.py
git commit -m "feat: export independent transparent visual elements"
```

### Task 5: Clean background and strict visual QA

**Files:**
- Modify: `scripts/bg_model.py`
- Modify: `scripts/visual_segment.py`
- Test: `tests/test_regressions.py`

- [ ] **Step 1: Write failing clean-background and QA tests**

```python
def test_build_clean_background_removes_elements_and_text() -> None:
    image = np.full((80, 100, 3), (20, 40, 60), dtype=np.uint8)
    image[20:55, 15:45] = (220, 30, 30)
    image[10:18, 50:85] = 255
    element = np.zeros((80, 100), dtype=bool)
    element[20:55, 15:45] = True
    text = np.zeros((80, 100), dtype=np.uint8)
    text[10:18, 50:85] = 255

    clean = bg_model.build_clean_background(image, [element], text)

    assert np.mean(np.abs(clean[25:50, 20:40].astype(int) - (20, 40, 60))) < 30
    assert np.mean(clean[11:17, 52:82]) < 120


def test_validate_visual_result_rejects_overlapping_ownership() -> None:
    first = np.zeros((20, 20), dtype=bool)
    second = np.zeros((20, 20), dtype=bool)
    first[2:15, 2:15] = True
    second[10:18, 10:18] = True

    with pytest.raises(
        visual_segment.VisualSegmentationError,
        match="overlapping visual ownership",
    ):
        visual_segment.validate_visual_masks([first, second])
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
python -m pytest -q tests/test_regressions.py -k "clean_background or validate_visual_result"
```

Expected: FAIL because the new APIs are absent.

- [ ] **Step 3: Implement clean background**

Add to `scripts/bg_model.py`:

```python
def build_clean_background(
    img: np.ndarray,
    element_masks: list[np.ndarray],
    text_mask: np.ndarray,
) -> np.ndarray:
    removal = (text_mask > 0).astype(np.uint8) * 255
    for mask in element_masks:
        removal[np.asarray(mask, dtype=bool)] = 255
    removal = cv2.dilate(removal, np.ones((5, 5), np.uint8), iterations=1)
    return _inpaint(img, removal)
```

- [ ] **Step 4: Implement ownership and reconstruction validation**

Add to `scripts/visual_segment.py`:

```python
def validate_visual_masks(element_masks: list[np.ndarray]) -> None:
    if not element_masks:
        return
    ownership = np.sum(
        [np.asarray(mask, dtype=np.uint8) for mask in element_masks],
        axis=0,
    )
    if int(ownership.max()) > 1:
        raise VisualSegmentationError("overlapping visual ownership detected")


def visual_difference(
    source: np.ndarray,
    reconstructed: np.ndarray,
    text_mask: np.ndarray,
) -> dict:
    valid = text_mask == 0
    if not np.any(valid):
        return {"mae": 0.0, "p95": 0.0}
    delta = np.mean(
        np.abs(source.astype(np.float32) - reconstructed.astype(np.float32)),
        axis=2,
    )[valid]
    return {
        "mae": float(np.mean(delta)),
        "p95": float(np.percentile(delta, 95)),
    }


def require_visual_quality(metrics: dict) -> None:
    if metrics["mae"] > 12.0 or metrics["p95"] > 48.0:
        raise VisualSegmentationError(
            "visual reconstruction did not meet the quality threshold"
        )
```

- [ ] **Step 5: Run tests**

```powershell
python -m pytest -q tests/test_regressions.py -k "clean_background or validate_visual_result"
python -m pytest -q
```

Expected: new tests and full regression suite pass.

- [ ] **Step 6: Commit background and QA**

```powershell
git add scripts/bg_model.py scripts/visual_segment.py
git commit -m "feat: reconstruct clean background and validate masks"
```

### Task 6: Fixed 16:9 contain transform

**Files:**
- Modify: `scripts/bg_model.py`
- Modify: `scripts/ppt_assemble.py`
- Test: `tests/test_regressions.py`

- [ ] **Step 1: Write failing 16:9 and coordinate tests**

```python
def test_compute_contain_transform_centers_portrait_image() -> None:
    transform = ppt_assemble.compute_contain_transform(1122, 1402)

    assert transform.slide_width == pytest.approx(13.333)
    assert transform.slide_height == pytest.approx(7.5)
    assert transform.offset_x > 0
    assert transform.offset_y == pytest.approx(0)


def test_assemble_pptx_is_widescreen_for_portrait_input(tmp_path: Path) -> None:
    background = np.zeros((1080, 1920, 3), dtype=np.uint8)
    background_path = tmp_path / "background.png"
    Image.fromarray(background).save(background_path)
    output = tmp_path / "portrait.pptx"

    assemble_pptx(background_path, [], [], 1122, 1402, output)
    prs = Presentation(output)

    assert prs.slide_width / prs.slide_height == pytest.approx(16 / 9, rel=1e-4)
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
python -m pytest -q tests/test_regressions.py -k "contain_transform or widescreen"
```

Expected: FAIL because portrait slides still preserve source ratio.

- [ ] **Step 3: Add the shared transform**

Add to `scripts/ppt_assemble.py`:

```python
from dataclasses import dataclass

SLIDE_WIDTH_INCHES = 13.333
SLIDE_HEIGHT_INCHES = 7.5


@dataclass(frozen=True)
class ContainTransform:
    slide_width: float
    slide_height: float
    content_width: float
    content_height: float
    offset_x: float
    offset_y: float


def compute_contain_transform(img_width: int, img_height: int) -> ContainTransform:
    scale = min(
        SLIDE_WIDTH_INCHES / img_width,
        SLIDE_HEIGHT_INCHES / img_height,
    )
    content_width = img_width * scale
    content_height = img_height * scale
    return ContainTransform(
        SLIDE_WIDTH_INCHES,
        SLIDE_HEIGHT_INCHES,
        content_width,
        content_height,
        (SLIDE_WIDTH_INCHES - content_width) / 2,
        (SLIDE_HEIGHT_INCHES - content_height) / 2,
    )
```

Set both `assemble_pptx` and `assemble_pptx_multi` to:

```python
prs.slide_width = Inches(SLIDE_WIDTH_INCHES)
prs.slide_height = Inches(SLIDE_HEIGHT_INCHES)
```

Map components with:

```python
transform = compute_contain_transform(img_width, img_height)
left = Inches(transform.offset_x + comp["x"] / img_width * transform.content_width)
top = Inches(transform.offset_y + comp["y"] / img_height * transform.content_height)
width = Inches(comp["w"] / img_width * transform.content_width)
height = Inches(comp["h"] / img_height * transform.content_height)
```

Pass the same transform into `_add_textbox`; only replace its coordinate mapping. Do not change text, font, size, weight, color, alignment, margins, wrapping, or vertical anchor code.

For an optional reference slide, place the widescreen background across the
full page first, then place the original image at `offset_x`, `offset_y` with
`content_width`, `content_height`. This preserves the same contain transform
without stretching or cropping the reference image.

- [ ] **Step 4: Add deterministic widescreen background extension**

Add to `scripts/bg_model.py`:

```python
def extend_background_to_widescreen(
    background: np.ndarray,
    canvas_width: int = 1920,
    canvas_height: int = 1080,
) -> np.ndarray:
    height, width = background.shape[:2]
    cover_scale = max(canvas_width / width, canvas_height / height)
    cover_size = (
        max(1, round(width * cover_scale)),
        max(1, round(height * cover_scale)),
    )
    cover = cv2.resize(background, cover_size, interpolation=cv2.INTER_LINEAR)
    x1 = (cover.shape[1] - canvas_width) // 2
    y1 = (cover.shape[0] - canvas_height) // 2
    canvas = cover[y1:y1 + canvas_height, x1:x1 + canvas_width]
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=24, sigmaY=24)

    contain_scale = min(canvas_width / width, canvas_height / height)
    content_size = (
        max(1, round(width * contain_scale)),
        max(1, round(height * contain_scale)),
    )
    content = cv2.resize(background, content_size, interpolation=cv2.INTER_AREA)
    x = (canvas_width - content.shape[1]) // 2
    y = (canvas_height - content.shape[0]) // 2
    canvas[y:y + content.shape[0], x:x + content.shape[1]] = content
    return canvas
```

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m pytest -q tests/test_regressions.py -k "contain_transform or widescreen"
python -m pytest -q
```

Expected: exact 16:9 and coordinate tests pass; existing text tests remain unchanged.

- [ ] **Step 6: Commit widescreen assembly**

```powershell
git add scripts/bg_model.py scripts/ppt_assemble.py
git commit -m "feat: assemble all slides in widescreen format"
```

### Task 7: Integrate strict semantic pipeline for single and batch conversion

**Files:**
- Modify: `image_to_ppt.py`
- Modify: `scripts/visual_segment.py`
- Test: `tests/test_regressions.py`

- [ ] **Step 1: Write failing integration tests with an injected generator**

```python
def test_process_image_uses_semantic_elements_without_flattening(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = np.full((60, 80, 3), 40, dtype=np.uint8)
    first = np.zeros((60, 80), dtype=bool)
    second = np.zeros((60, 80), dtype=bool)
    first[10:35, 5:30] = True
    second[10:35, 30:55] = True

    monkeypatch.setattr(
        image_to_ppt,
        "detect_text",
        lambda *args, **kwargs: ([], np.zeros((60, 80), dtype=np.uint8)),
    )
    monkeypatch.setattr(image_to_ppt, "_load_rgb", lambda path: image)

    class FakeGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, crop):
            self.calls += 1
            if self.calls > 1:
                return []
            records = []
            for mask in (first, second):
                records.append({
                    "segmentation": mask[:crop.shape[0], :crop.shape[1]],
                    "predicted_iou": 0.99,
                    "stability_score": 0.99,
                })
            return records

    data = image_to_ppt._process_image(
        tmp_path / "input.png",
        tmp_path,
        FakeGenerator(),
        lang="ch",
    )

    assert len(data["components"]) == 2
    assert data["img_width"] == 80
    assert data["img_height"] == 60


def test_process_image_recovers_residual_visual_element(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = np.full((60, 80, 3), 40, dtype=np.uint8)
    first = np.zeros((60, 80), dtype=bool)
    second = np.zeros((60, 80), dtype=bool)
    first[8:25, 8:25] = True
    second[32:52, 45:70] = True
    monkeypatch.setattr(image_to_ppt, "_load_rgb", lambda path: image)
    monkeypatch.setattr(
        image_to_ppt,
        "detect_text",
        lambda *args, **kwargs: ([], np.zeros((60, 80), dtype=np.uint8)),
    )

    class SequenceGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, crop):
            self.calls += 1
            selected = first if self.calls == 1 else second if self.calls == 2 else None
            if selected is None:
                return []
            return [{
                "segmentation": selected,
                "predicted_iou": 0.99,
                "stability_score": 0.99,
            }]

    data = image_to_ppt._process_image(
        tmp_path / "input.png",
        tmp_path,
        SequenceGenerator(),
        lang="ch",
    )

    assert len(data["components"]) == 2


def test_process_image_propagates_quality_failure(tmp_path: Path, monkeypatch) -> None:
    image = np.full((20, 20, 3), 40, dtype=np.uint8)
    monkeypatch.setattr(image_to_ppt, "_load_rgb", lambda path: image)
    monkeypatch.setattr(
        image_to_ppt,
        "detect_text",
        lambda *args, **kwargs: ([], np.zeros((20, 20), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "require_visual_quality",
        lambda metrics: (_ for _ in ()).throw(
            visual_segment.VisualSegmentationError("quality failure")
        ),
    )

    class EmptyGenerator:
        def generate(self, crop):
            return []

    with pytest.raises(visual_segment.VisualSegmentationError, match="quality failure"):
        image_to_ppt._process_image(
            tmp_path / "input.png",
            tmp_path,
            EmptyGenerator(),
            lang="ch",
        )
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
python -m pytest -q tests/test_regressions.py -k "process_image"
```

Expected: FAIL because the shared pipeline does not exist.

- [ ] **Step 3: Implement reconstruction from owned pixels**

Add to `scripts/visual_segment.py`:

```python
def compose_visual_result(
    clean_background: np.ndarray,
    source: np.ndarray,
    element_masks: list[np.ndarray],
    text_mask: np.ndarray,
) -> np.ndarray:
    composed = clean_background.copy()
    for mask in element_masks:
        visible = np.asarray(mask, dtype=bool) & (text_mask == 0)
        composed[visible] = source[visible]
    return composed
```

- [ ] **Step 4: Implement a shared image-processing function**

Replace duplicated single/batch internals with:

```python
def _process_image(
    image_path: Path,
    work_dir: Path,
    mask_generator,
    lang: str,
) -> dict:
    img = _load_rgb(image_path)
    img_h, img_w = img.shape[:2]
    text_items, text_mask = detect_text(str(image_path), lang=lang)

    candidates = generate_mask_candidates(img, mask_generator)
    for pass_index in range(2):
        elements = resolve_visual_elements(candidates)
        element_masks = [item.mask for item in elements]
        validate_visual_masks(element_masks)
        clean_background = build_clean_background(img, element_masks, text_mask)
        residual = [
            item
            for item in generate_mask_candidates(clean_background, mask_generator)
            if item.score >= 0.90
        ]
        if not residual:
            break
        candidates.extend(
            MaskCandidate(item.mask, item.score, "residual")
            for item in residual
        )
    else:
        raise VisualSegmentationError(
            "clean background still contains independent visual elements"
        )

    components = export_visual_components(
        img,
        element_masks,
        work_dir / "components",
        text_mask,
    )
    widescreen_background = extend_background_to_widescreen(clean_background)
    background_path = work_dir / "background.png"
    _save_rgb(str(background_path), widescreen_background)

    reconstructed = compose_visual_result(
        clean_background,
        img,
        element_masks,
        text_mask,
    )
    metrics = visual_difference(img, reconstructed, text_mask)
    require_visual_quality(metrics)

    return {
        "background_path": str(background_path),
        "components": components,
        "text_items": text_items,
        "img_width": img_w,
        "img_height": img_h,
        "original_image_path": str(image_path),
        "quality": metrics,
    }
```

- [ ] **Step 5: Reuse one model instance and remove flattened success**

At the start of `convert` and once before the loop in `convert_batch`:

```python
checkpoint = resolve_sam_checkpoint()
mask_generator = create_sam_generator(checkpoint)
```

Call `_process_image` for each input. Remove calls to `_should_use_text_only_fallback`, remove the fallback function, and remove all success messages that describe a flattened background.

- [ ] **Step 6: Add one-pass raster-text residue verification**

Save the visual-only reconstruction in the work directory and invoke the existing `detect_text` once on it. If any text item is returned, raise:

```python
raise VisualSegmentationError(
    "visual components still contain raster text after cleanup"
)
```

Do not change `detect_text`; only call it with the generated visual-only image.

- [ ] **Step 7: Run focused and full tests**

```powershell
python -m pytest -q tests/test_regressions.py -k "process_image"
python -m pytest -q
```

Expected: shared-pipeline tests pass; full suite passes; no test expects flattened success.

- [ ] **Step 8: Commit pipeline integration**

```powershell
git add image_to_ppt.py scripts/visual_segment.py
git commit -m "feat: use strict semantic visual decomposition"
```

### Task 8: Diagnostics, licensing, documentation, and skill synchronization

**Files:**
- Modify: `scripts/visual_segment.py`
- Modify: `README.md`
- Modify: `skills/image-to-ppt/SKILL.md`
- Modify: `skills/image-to-ppt/references/requirements.txt`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `third_party/licenses/SAM2-APACHE-2.0.txt`
- Create: `CITATION.cff`
- Create: `skills/image-to-ppt/scripts/visual_segment.py`
- Modify: relevant `skills/image-to-ppt/scripts/*.py`
- Test: `tests/test_regressions.py`

- [ ] **Step 1: Write a failing diagnostics test**

```python
def test_write_segmentation_diagnostics_creates_report(tmp_path: Path) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:20] = True

    visual_segment.write_segmentation_diagnostics(
        tmp_path,
        source=image,
        masks=[mask],
        reconstructed=image,
        metrics={"mae": 0.0, "p95": 0.0},
    )

    assert (tmp_path / "source.png").exists()
    assert (tmp_path / "ownership.png").exists()
    assert (tmp_path / "reconstructed.png").exists()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["component_count"] == 1
```

- [ ] **Step 2: Implement deterministic diagnostics**

Add `json` and `Image` imports, then:

```python
def write_segmentation_diagnostics(
    output_dir: str | Path,
    source: np.ndarray,
    masks: list[np.ndarray],
    reconstructed: np.ndarray,
    metrics: dict,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(source).save(output / "source.png")
    Image.fromarray(reconstructed).save(output / "reconstructed.png")

    ownership = np.zeros(source.shape[:2], dtype=np.uint16)
    for index, mask in enumerate(masks, start=1):
        ownership[np.asarray(mask, dtype=bool)] = index
    normalized = np.where(
        ownership > 0,
        (ownership * 37) % 255,
        0,
    ).astype(np.uint8)
    Image.fromarray(normalized).save(output / "ownership.png")

    report = dict(metrics)
    report["component_count"] = len(masks)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

Call this function before raising a quality error.

- [ ] **Step 3: Add third-party notice and project citation**

Create `THIRD_PARTY_NOTICES.md`:

```markdown
# Third-Party Notices

## Segment Anything Model 2.1

This project integrates Segment Anything Model 2.1 through its official Python API.

- Copyright: Meta Platforms, Inc. and affiliates
- License: Apache License 2.0
- Source: https://github.com/facebookresearch/sam2
- Checkpoints: https://dl.fbaipublicfiles.com/segment_anything_2/092824/

SAM 2.1 source code and model checkpoints are not stored in this repository.
The checkpoint is downloaded to the user's local cache when first needed.
```

Create `CITATION.cff`:

```yaml
cff-version: 1.2.0
message: "If you use image2editable, please cite this software and SAM 2."
title: "image2editable"
type: software
authors:
  - name: "DengShouYang"
repository-code: "https://github.com/DSY-Xueai/image2editable"
license: MIT
```

Use the exact Apache 2.0 text from the official SAM 2 repository at `https://github.com/facebookresearch/sam2/blob/main/LICENSE` for `third_party/licenses/SAM2-APACHE-2.0.txt`. Do not alter that text or replace the project root MIT license.

- [ ] **Step 4: Document installation, runtime, and citation**

Update README with:

```markdown
### SAM 2.1 visual segmentation

Visual-element decomposition uses the official SAM 2.1 Large checkpoint.
The checkpoint is downloaded to the local model cache on first use and is not
included in this repository. CUDA is used automatically when available; CPU
execution remains supported but is slower.

SAM 2.1 is licensed under Apache 2.0. See
`THIRD_PARTY_NOTICES.md` and `third_party/licenses/SAM2-APACHE-2.0.txt`.
```

Add the official SAM 2 BibTeX from the upstream README without changing author names, title, URL, or year.

- [ ] **Step 5: Update the distributable skill**

Copy these files byte-for-byte:

```powershell
Copy-Item image_to_ppt.py skills/image-to-ppt/scripts/image_to_ppt.py
Copy-Item scripts/bg_model.py skills/image-to-ppt/scripts/bg_model.py
Copy-Item scripts/fg_extract.py skills/image-to-ppt/scripts/fg_extract.py
Copy-Item scripts/ppt_assemble.py skills/image-to-ppt/scripts/ppt_assemble.py
Copy-Item scripts/visual_segment.py skills/image-to-ppt/scripts/visual_segment.py
```

Add SAM 2.1 and PyTorch installation requirements to `skills/image-to-ppt/references/requirements.txt`, and update `skills/image-to-ppt/SKILL.md` to state that the first run downloads the official checkpoint into the local cache.

- [ ] **Step 6: Verify synchronization and tests**

```powershell
python -m pytest -q
git diff --check
$pairs = @(
  @('image_to_ppt.py','skills/image-to-ppt/scripts/image_to_ppt.py'),
  @('scripts/bg_model.py','skills/image-to-ppt/scripts/bg_model.py'),
  @('scripts/fg_extract.py','skills/image-to-ppt/scripts/fg_extract.py'),
  @('scripts/ppt_assemble.py','skills/image-to-ppt/scripts/ppt_assemble.py'),
  @('scripts/visual_segment.py','skills/image-to-ppt/scripts/visual_segment.py')
)
foreach ($pair in $pairs) {
  if ((Get-FileHash $pair[0]).Hash -ne (Get-FileHash $pair[1]).Hash) {
    throw "Unsynchronized script: $($pair[0])"
  }
}
```

Expected: full tests pass, diff check is clean, and all hashes match.

- [ ] **Step 7: Commit diagnostics, docs, license, and skill**

```powershell
git add README.md THIRD_PARTY_NOTICES.md CITATION.cff third_party requirements.txt .gitignore
git add scripts/visual_segment.py skills/image-to-ppt
git commit -m "docs: document SAM 2.1 integration and licensing"
```

Do not stage `tests/`, `Course.md`, model checkpoints, caches, generated components, or PPT output files.

### Task 9: End-to-end test, render, and human visual review

**Files:**
- Test input: `E:\My_project\Change_PPT\test-image\test.png`
- Output: `E:\My_project\Change_PPT\test_output_files\test_semantic_16x9.pptx`
- Diagnostics: `E:\My_project\Change_PPT\test_output_files\test_semantic_16x9_diagnostics`

- [ ] **Step 1: Verify the runtime before the expensive test**

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import sam2; print('sam2 import ok')"
```

Expected: both commands succeed. CUDA should be used when the installed PyTorch build supports it; otherwise record that the run is CPU-only.

- [ ] **Step 2: Run the real conversion**

```powershell
python image_to_ppt.py "E:\My_project\Change_PPT\test-image\test.png" -o "E:\My_project\Change_PPT\test_output_files\test_semantic_16x9.pptx"
```

Expected: conversion succeeds without a flattened-layout message and reports at least one independent visual component.

- [ ] **Step 3: Inspect PPT structure**

```powershell
@'
from pptx import Presentation

path = r"E:\My_project\Change_PPT\test_output_files\test_semantic_16x9.pptx"
prs = Presentation(path)
assert len(prs.slides) == 1
assert abs(prs.slide_width / prs.slide_height - 16 / 9) < 1e-4
picture_count = sum(
    1
    for shape in prs.slides[0].shapes
    if shape.shape_type == 13
)
text_count = sum(
    1
    for shape in prs.slides[0].shapes
    if getattr(shape, "has_text_frame", False)
)
assert picture_count > 1
assert text_count > 0
print(f"pictures={picture_count} textboxes={text_count}")
'@ | python -
```

Expected: one 16:9 slide, one background plus multiple picture components, and editable text boxes.

- [ ] **Step 4: Render and run overflow checks**

```powershell
$skill="C:\Users\d's'y\.codex\plugins\cache\openai-primary-runtime\presentations\26.715.12143\skills\presentations"
$py="C:\Users\d's'y\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py "$skill\container_tools\render_slides.py" "E:\My_project\Change_PPT\test_output_files\test_semantic_16x9.pptx" --output_dir "E:\My_project\Change_PPT\test_output_files\test_semantic_16x9_render"
& $py "$skill\container_tools\slides_test.py" "E:\My_project\Change_PPT\test_output_files\test_semantic_16x9.pptx"
```

Expected: render succeeds and `slides_test.py` reports no overflow.

- [ ] **Step 5: Perform human visual review**

Inspect the rendered slide and a contact sheet of all component PNGs. Confirm:

- no component includes visible text;
- touching objects are not merged into one movable layer;
- no component has a rectangular opaque background;
- no duplicated pixels create ghosts;
- edges have no obvious white or black halo;
- the composed slide matches the source content without stretching or cropping;
- portrait content is centered on a 16:9 canvas with clean background extension.

If any item fails, add a minimal local regression test for that specific root cause before changing code.

- [ ] **Step 6: Run final verification**

```powershell
python -m pytest -q
python -m ruff check --ignore F541,F841 image_to_ppt.py scripts skills/image-to-ppt/scripts
git diff --check
git status --short
```

Expected: all tests and lint checks pass; only intended tracked changes and ignored local artifacts remain.

- [ ] **Step 7: Request code review before merge**

Use the `requesting-code-review` skill. Resolve Critical and Important findings with failing tests first, repeat full verification, and then present the branch for merge.
