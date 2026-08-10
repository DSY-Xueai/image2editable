# Residual-Driven Editable Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 image2editable 重建流程内补齐残差驱动闭环，使项目支持的图片、PDF、图片版 PPTX 和混合 PPTX 截图区域只有在文字可编辑、显著视觉元素有独立 ownership、背景无残影且 PowerPoint 实际渲染通过时才算成功。

**Architecture:** 保留现有 Runtime、组件图、Agent、Router、PPTX/PSD Adapter 和 Render QA。修改现有页面准备、组件质量和 repair state machine：初始提取阶段扩大 residual candidate 覆盖，质量阶段持久化 material foreground evidence 并计算 unexplained ownership，页面级违反进入下一修复轮而不是立即回退，Legacy 背景阶段聚合并执行全部 `rebuild_background` 动作。

**Tech Stack:** Python 3.12、NumPy、OpenCV、Pillow、SAM 2.1、python-pptx、pytest、Microsoft PowerPoint COM Render QA。

---

## 文件结构与职责

- `image2editable/component_repair.py`：组件修复状态机、页面级修复轮次、质量制品重算与单调收敛。
- `image2editable/component_quality.py`：material foreground ownership 指标、unexplained 门禁和报告验证。
- `image2editable/component_contracts.py`：evidence 与 quality input refs 契约。
- `image2editable/legacy.py`：Legacy Run 接线、背景动作聚合、evidence 发布和最终交付状态。
- `image2editable/local_agent.py`：Local Agent 对页面级违反的动作约束。
- `image_to_ppt.py`：现有初始 OCR、SAM、残差候选、背景和 prepared page 资产。
- `scripts/visual_segment.py`：通用 residual candidate 协调函数和现有组件动作执行。
- `skills/image-to-ppt/scripts/image_to_ppt.py`、`skills/image-to-ppt/scripts/visual_segment.py`：运行脚本的字节镜像。
- `skills/image-to-ppt/SKILL.md`：Host Agent 采用同一页面级修复规则。
- `tests/test_component_repair.py`：状态机、evidence、安全绑定和背景动作回归。
- `tests/test_component_quality.py`：ownership 指标与硬门禁。
- `tests/test_regressions.py`、`tests/test_ocr_isolation.py`：初始残差候选与 prepared page 兼容。
- `tests/test_runtime_input_dispatch.py`、`tests/test_task10_runtime_e2e.py`、`tests/test_task10_mixed_native.py`：跨输入与真实组装契约。
- `Course.md`：同步最终项目状态、测试事实和未完成风险。

### Task 1: 页面级质量违反继续进入修复轮

**Files:**
- Modify: `image2editable/component_repair.py:44-145`
- Modify: `image2editable/component_repair.py:590-735`
- Test: `tests/test_component_repair.py`

- [ ] **Step 1: 写页面级违反的失败测试**

修改现有 `test_last_pending_discard_records_page_quality_without_rechecking_frozen` 的最后断言。该测试已经建立了 `page_session`、真实 `RunStore`、空 `candidate_ids` 和失败的页面级质量，不新增 fixture：

```python
next_round = advance_component_repair(store, "page_001")
assert next_round == {
        "status": "needs_next_round",
        "page_id": "page_001",
        "repair_round": 2,
        "candidate_ids": [],
        "page_violations": ["background_text_residual"],
}
```

在现有五轮上限测试的 synthetic quality 中把 `failed_ids` 设为空，并让 `_strict_quality_report` 返回 `background_text_residual`，断言第 5 轮进入 `fallback_required` 且不会创建 `round-06`。沿用该测试已有的 `bind_synthetic_quality`，不增加生产侧测试后门。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_component_repair.py -k "page_only_background_residual or page_only_violation_stops" -vv
```

Expected: 第一个测试收到 `preserved_with_warning` 而不是 `needs_next_round`；第二个测试保持终止行为。

- [ ] **Step 3: 最小修改状态机**

在 `component_repair.py` 增加：

```python
_REPAIRABLE_PAGE_VIOLATIONS = frozenset({
    "background_text_residual",
    "unexplained_visual_residual",
})


def _repairable_page_quality_violations(store, state: dict) -> list[str]:
    return sorted(
        _blocking_page_quality_violations(store, state)
        & _REPAIRABLE_PAGE_VIOLATIONS
    )
```

把 `freeze_committed` 分支调整为：

```python
if state["phase"] == "freeze_committed":
    page_violations = _repairable_page_quality_violations(store, state)
    if state["failed_ids"] or page_violations:
        if state["repair_round"] >= MAX_REPAIR_ROUNDS:
            return _commit_fallback_required(
                store, state, page_id, "round_limit"
            )
        return {
            "status": "needs_next_round",
            "page_id": page_id,
            "repair_round": state["repair_round"] + 1,
            "candidate_ids": list(state["failed_ids"]),
            "page_violations": page_violations,
        }
    blocking_violations = _blocking_page_quality_violations(store, state)
    if blocking_violations:
        updated = dict(state)
        updated["stop_reason"] = (
            "unowned_raster_text"
            if "unowned_raster_text" in blocking_violations
            else "page_quality_failed"
        )
        return _commit_preserved_warning(store, updated, page_id)
```

`record_next_component_request` 允许 `failed_ids == []` 且存在 repairable page violation 的 `freeze_committed` 状态发布下一轮；request 的 `candidate_ids` 仍必须等于空列表，不改变现有候选身份约束。

- [ ] **Step 4: 运行状态机测试**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_component_repair.py -k "page_only or next_component_request or round_limit" -vv
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add image2editable/component_repair.py tests/test_component_repair.py
git commit -m "修复：页面级质量失败继续进入修复轮"
```

### Task 2: 聚合并验证全部背景修复动作

**Files:**
- Modify: `image2editable/legacy.py:1339-1413`
- Modify: `image2editable/legacy.py:1930-2054`
- Test: `tests/test_runtime_execution.py`
- Test: `tests/test_component_repair.py`

- [ ] **Step 1: 写多动作与文字残影失败测试**

增加直接测试 `_rebuild_canvas_background` 的合成页面：两个相距较远的文字/组件区域分别由两条动作请求修复，并断言两个区域均改变、请求外像素保持逐像素相同。

```python
def test_rebuild_canvas_background_consumes_every_repair_request(tmp_path):
    shape = (80, 120)
    source_pixels = np.full((*shape, 3), 240, dtype=np.uint8)
    source_pixels[8:24, 8:36] = 20
    source_pixels[40:64, 80:112] = 40
    source = tmp_path / "source.png"
    current = tmp_path / "current.png"
    Image.fromarray(source_pixels).save(source)
    Image.fromarray(source_pixels).save(current)
    graph_dir = tmp_path / "graph"
    (graph_dir / "masks").mkdir(parents=True)
    graph = _write_background_action_graph(
        graph_dir,
        {
            "text_left": _box_mask(shape, (8, 8, 36, 24)),
            "component_right": _box_mask(shape, (80, 40, 112, 64)),
        },
    )
    Image.fromarray(
        _box_mask(shape, (8, 8, 36, 24)), mode="L"
    ).save(tmp_path / "text-mask.png")
    output = tmp_path / "rebuilt.png"

    legacy._rebuild_canvas_background(
        source_path=source,
        current_background_path=current,
        component_ids=None,
        repair_requests=[
            ({"text_left"}, 0.01),
            ({"component_right"}, 0.02),
        ],
        graph=graph,
        graph_dir=graph_dir,
        text_mask_path=tmp_path / "text-mask.png",
        output_path=output,
    )

    actual = np.asarray(Image.open(output).convert("RGB"))
    assert not np.array_equal(actual[8:24, 8:36], source_pixels[8:24, 8:36])
    assert not np.array_equal(actual[40:64, 80:112], source_pixels[40:64, 80:112])
    assert np.array_equal(actual[:4, :4], source_pixels[:4, :4])
```

测试内新增以下 helper；`_box_mask` 复用 `tests/test_component_repair.py` 已有定义：

```python
def _write_background_action_graph(graph_dir: Path, masks: dict[str, np.ndarray]) -> dict:
    nodes = []
    for z_index, (component_id, mask) in enumerate(masks.items()):
        path = graph_dir / "masks" / f"{component_id}.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)
        ys, xs = np.where(mask)
        nodes.append({
            "id": component_id,
            "kind": "text" if component_id.startswith("text_") else "parent",
            "parent_id": None,
            "state": "frozen",
            "mask": f"masks/{component_id}.png",
            "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bbox": [
                int(xs.min()), int(ys.min()),
                int(xs.max()) + 1, int(ys.max()) + 1,
            ],
            "z_index": z_index,
            "text_ids": [],
        })
    return {"nodes": nodes}
```

再增加 `_execute_legacy_round` 回归，给出两条 `rebuild_background` action，断言传给背景 helper 的 request 有两项，而不是只消费 `rebuild_actions[0]`。

- [ ] **Step 2: 运行并确认 RED**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_runtime_execution.py tests/test_component_repair.py -k "every_repair_request or aggregates_background" -vv
```

Expected: 当前签名不接受 `repair_requests`，并且 Legacy 只传第一条 action。

- [ ] **Step 3: 实现有界 repair mask 聚合**

把 `_rebuild_canvas_background` 的 `margin_ratio` 改为 `repair_requests: list[tuple[set[str], float]]`。对每条请求分别读取目标 node mask、按该请求 margin 膨胀，再合并：

```python
repair = np.zeros(text_repair.shape, dtype=bool)
by_id = {node["id"]: node for node in graph["nodes"]}
for object_ids, margin_ratio in repair_requests:
    radius = max(1, round(min(text_repair.shape) * margin_ratio))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    request_mask = np.zeros(text_repair.shape, dtype=bool)
    for object_id in object_ids:
        node = by_id[object_id]
        mask_path = (graph_dir / Path(node["mask"])).resolve()
        if not mask_path.is_relative_to(graph_dir.resolve()):
            raise ValueError("background rebuild mask is outside graph directory")
        if sha256_file(mask_path) != node["mask_sha256"]:
            raise ValueError("background rebuild mask sha256 mismatch")
        with Image.open(mask_path) as image:
            request_mask |= np.asarray(image.convert("L")) > 0
    repair |= cv2.dilate(request_mask.astype(np.uint8), kernel) > 0
```

当请求目标包含 text node 时，不再因 `node["kind"] == "text"` 跳过。以 `restored` 或 `current` 为 seed，继续调用现有 `_choose_visual_fill`；未选中的区域不改变。

`_execute_legacy_round` 构造：

```python
repair_requests = [
    (set(action["object_ids"]), action["parameters"]["margin_ratio"])
    for action in rebuild_actions
]
```

并一次传入 helper。mask 动作执行器仍只负责 graph/mask，不移动背景职责。

- [ ] **Step 4: 验证背景与完整相关测试**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_runtime_execution.py tests/test_component_repair.py tests/test_regressions.py -k "background or repair or text_cleanup" -vv
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add image2editable/legacy.py tests/test_runtime_execution.py tests/test_component_repair.py
git commit -m "修复：聚合执行页面背景修复动作"
```

### Task 3: 扩大初始残差候选覆盖

**Files:**
- Modify: `scripts/visual_segment.py:1253-1394`
- Modify: `image_to_ppt.py:1730-1865`
- Modify: `skills/image-to-ppt/scripts/visual_segment.py`
- Modify: `skills/image-to-ppt/scripts/image_to_ppt.py`
- Test: `tests/test_regressions.py`
- Test: `tests/test_ocr_isolation.py`

- [ ] **Step 1: 写 detector 漏检但 automatic SAM 命中的失败测试**

先直接测试新的协调函数。构造两个独立矩形，prompted 为空，automatic SAM 候选覆盖右图，clean background 在右图区域仍等于 source：

```python
def test_combine_residual_candidates_keeps_prompt_free_object():
    source = np.full((80, 120, 3), 240, dtype=np.uint8)
    right = np.zeros((80, 120), dtype=bool)
    right[24:56, 72:104] = True
    source[right] = 20
    clean_background = np.full_like(source, 240)
    clean_background[right] = source[right]
    automatic = [MaskCandidate(right, 0.96, "sam")]

    residual, attached = combine_residual_candidates(
        source=source,
        clean_background=clean_background,
        prompted=[],
        prompt_free=automatic,
        existing=[],
        text_mask=np.zeros(right.shape, dtype=np.uint8),
    )

    assert attached == 0
    assert len(residual) == 1
    assert np.array_equal(residual[0].mask, right)
```

新增 `test_process_image_uses_prompt_free_sam_when_detector_misses`，复用现有 `test_process_image_restores_raw_text_mask_after_each_component_inpaint` 的 `_process_image` 依赖桩。把 `generate_mask_candidates` 设为按调用次数返回首轮左图、残差轮右图，并断言 `resolve_visual_elements` 的最后一次输入包含两个候选。这里调用真实 `_process_image`，不引用不存在的高层 API。

测试名称使用文件现有公开准备函数的真实名称；不通过 mock 直接伪造最终 slide_data。

- [ ] **Step 2: 运行并确认 RED**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_regressions.py -k "prompt_free_sam_when_detector_misses" -vv
```

Expected: 右图没有进入最终 element masks。

- [ ] **Step 3: 增加通用 residual candidate 合并函数**

在 `visual_segment.py` 增加：

```python
def combine_residual_candidates(
    *,
    source: np.ndarray,
    clean_background: np.ndarray,
    prompted: list[MaskCandidate],
    prompt_free: list[MaskCandidate],
    existing: list[MaskCandidate],
    text_mask: np.ndarray,
) -> tuple[list[MaskCandidate], int]:
    automatic = filter_prompt_free_candidates(
        prompt_free,
        prompted,
        text_mask,
    )
    residual = filter_unchanged_residual_candidates(
        source,
        clean_background,
        [*prompted, *automatic],
        text_mask,
    )
    return reconcile_residual_candidates(residual, existing, source.shape[:2])
```

在每个现有 residual round 中，除 prompted residual 外，再对 `clean_background` 调用已有 automatic SAM 路径：主进程使用 `generate_mask_candidates(..., include_geometry=True, min_score=0.90)`；资源隔离使用现有 `_generate_sam_candidates_isolated(..., mode="automatic")` 并补上 geometry candidates。两条路径都调用 `combine_residual_candidates`。

不得降低首轮 `min_score`，不得加入内容关键词、固定坐标或文件名判断。

- [ ] **Step 4: 同步镜像并验证**

Run:

```powershell
Copy-Item scripts\visual_segment.py skills\image-to-ppt\scripts\visual_segment.py -Force
Copy-Item image_to_ppt.py skills\image-to-ppt\scripts\image_to_ppt.py -Force
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_regressions.py tests/test_ocr_isolation.py -k "residual or prompt_free or resource_isolation" -vv
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_dependency_contract.py -k "mirror" -vv
```

Expected: PASS，两个镜像文件逐字节一致。

- [ ] **Step 5: 提交**

```powershell
git add image_to_ppt.py scripts/visual_segment.py skills/image-to-ppt/scripts/image_to_ppt.py skills/image-to-ppt/scripts/visual_segment.py tests/test_regressions.py tests/test_ocr_isolation.py
git commit -m "功能：补充自动分割残差候选"
```

### Task 4: 持久化 material foreground 并增加 ownership 门禁

**Files:**
- Modify: `image_to_ppt.py:1890-1990`
- Modify: `image_to_ppt.py:1961-2620`
- Modify: `skills/image-to-ppt/scripts/image_to_ppt.py`
- Modify: `image2editable/component_contracts.py`
- Modify: `image2editable/component_quality.py`
- Modify: `image2editable/component_repair.py`
- Modify: `image2editable/legacy.py`
- Test: `tests/test_component_quality.py`
- Test: `tests/test_component_repair.py`
- Test: `tests/test_ocr_isolation.py`

- [ ] **Step 1: 写显著候选被丢回背景的失败测试**

在 `tests/test_component_quality.py` 增加：

```python
def test_material_foreground_without_owner_fails_page_gate():
    shape = (96, 160)
    evidence = np.zeros(shape, dtype=bool)
    evidence[24:56, 72:104] = True
    owned = np.zeros(shape, dtype=bool)
    calibration = PageCalibration(1.0, 20.0, 2, 3, 20)

    metrics, unexplained = material_ownership_metrics(
        evidence,
        [owned],
        np.zeros(shape, dtype=bool),
        calibration,
    )

    assert metrics["largest_unexplained_region_pixels"] == 32 * 32
    assert metrics["visual_ownership_coverage"] == 0.0
    assert np.array_equal(unexplained, evidence)
```

再给 `evaluate_page_quality` 增加 `page_checks={"visual_ownership": "fail"}`，断言违反项包含 `unexplained_visual_residual`，即使 MAE/SSIM 对应指标全部通过也不能接受。

- [ ] **Step 2: 运行并确认 RED**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_component_quality.py -k "material_foreground_without_owner or visual_ownership" -vv
```

Expected: `material_ownership_metrics` 不存在，page gate 不认识 `visual_ownership`。

- [ ] **Step 3: 实现 ownership 指标**

在 `component_quality.py` 增加：

```python
def material_ownership_metrics(
    material_foreground: np.ndarray,
    component_masks: Iterable[np.ndarray],
    text_mask: np.ndarray,
    calibration: PageCalibration,
) -> tuple[dict, np.ndarray]:
    material = np.asarray(material_foreground, dtype=bool) & ~np.asarray(
        text_mask, dtype=bool
    )
    owned = np.zeros(material.shape, dtype=bool)
    for mask in component_masks:
        projected, _ = _project_component_mask(mask, material.shape)
        owned |= projected
    unexplained = material & ~owned
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        unexplained.astype(np.uint8), 8
    )
    keep = np.zeros(material.shape, dtype=bool)
    largest = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < calibration.min_component_pixels:
            continue
        keep |= labels == label
        largest = max(largest, area)
    material_pixels = int(np.count_nonzero(material))
    unexplained_pixels = int(np.count_nonzero(keep))
    owned_pixels = int(np.count_nonzero(material & owned))
    return {
        "material_foreground_pixels": material_pixels,
        "owned_visual_pixels": owned_pixels,
        "unexplained_visual_pixels": unexplained_pixels,
        "largest_unexplained_region_pixels": largest,
        "visual_ownership_coverage": (
            owned_pixels / material_pixels if material_pixels else 1.0
        ),
    }, keep
```

`evaluate_page_quality` 将 `visual_ownership=fail` 映射为 `unexplained_visual_residual`。指标加入现有 `visual_metrics`，不新增第二种质量报告。

- [ ] **Step 4: 把 evidence 资产接入 prepared page 和质量输入**

在初始提取完成后保存所有最终 semantic masks 的并集：

```python
material_foreground = (
    np.logical_or.reduce(semantic_masks)
    if semantic_masks
    else np.zeros(img.shape[:2], dtype=bool)
)
foreground_evidence_path = work_dir / "foreground-evidence-mask.png"
Image.fromarray(material_foreground.astype(np.uint8) * 255, mode="L").save(
    foreground_evidence_path
)
slide_data["_foreground_evidence_mask_path"] = str(foreground_evidence_path)
```

prepared page schema 升到 5，并在 assets 中加入 `foreground_evidence_mask`。`component_contracts._validate_quality_input_refs` 增加 `foreground_evidence`；`legacy._quality_assets` 和 `component_repair.evaluate_component_quality_round` 加载该 hash-bound mask，调用 `material_ownership_metrics`，保存 `unexplained-mask.png`，并把 `visual_ownership` 写入 page checks。

旧 prepared schema 1-4 继续加载，但兼容路径不运行也不写入新的 `visual_ownership` check，保持旧 Run 的既有状态语义；它们不能被记为通过本版本新增的视觉闭环验收。schema 5 的新 Run 必须提供 evidence，缺失时拒绝质量评估。

- [ ] **Step 5: 验证 schema、质量与镜像**

Run:

```powershell
Copy-Item image_to_ppt.py skills\image-to-ppt\scripts\image_to_ppt.py -Force
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_component_quality.py tests/test_component_repair.py tests/test_ocr_isolation.py -k "ownership or prepared or quality_input_refs or unexplained" -vv
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_dependency_contract.py -k "mirror" -vv
```

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add image_to_ppt.py skills/image-to-ppt/scripts/image_to_ppt.py image2editable/component_contracts.py image2editable/component_quality.py image2editable/component_repair.py image2editable/legacy.py tests/test_component_quality.py tests/test_component_repair.py tests/test_ocr_isolation.py
git commit -m "功能：增加显著视觉归属门禁"
```

### Task 5: 让 Agent 对页面残差执行可验证动作

**Files:**
- Modify: `image2editable/component_contracts.py:299-534`
- Modify: `scripts/visual_segment.py:114-330`
- Modify: `skills/image-to-ppt/scripts/visual_segment.py`
- Modify: `image2editable/local_agent.py`
- Modify: `skills/image-to-ppt/SKILL.md`
- Modify: `image2editable/legacy.py:2057-2185`
- Test: `tests/test_component_contracts.py`
- Test: `tests/test_component_repair.py`
- Test: `tests/test_local_agent.py`
- Test: `tests/test_psd_skill.py`

- [ ] **Step 1: 写 inactive 候选重试和页面 evidence 失败测试**

增加契约/执行测试：`retry_with_box`、`retry_with_points` 可以针对 inactive visual node，并在 SAM 返回有效 mask 后把它恢复为 `pending`；其他动作仍不能任意恢复 inactive node。

```python
def test_retry_with_box_can_reconsider_inactive_visual(tmp_path):
    image, graph, input_dir = _action_case(tmp_path)
    left = next(node for node in graph["nodes"] if node["id"] == "left")
    left["state"] = "inactive"

    result = execute_component_actions(
        image,
        graph,
        [_action("retry_with_box", ["left"], {
            "box": [0.05, 0.05, 0.5, 0.8],
            "independent": True,
        })],
        sam_runner=lambda **_: np.pad(
            np.ones((4, 4), dtype=bool), ((2, 6), (2, 10))
        ),
        input_dir=input_dir,
        output_dir=tmp_path / "reconsidered",
    )

    node = next(value for value in result["nodes"] if value["id"] == "left")
    assert node["state"] == "pending"
```

增加 request evidence 测试，repairable page violation 的下一轮必须包含 `unexplained-mask.png`；文件 hash 不匹配时拒绝 request。

- [ ] **Step 2: 运行并确认 RED**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_component_contracts.py tests/test_component_repair.py -k "reconsider_inactive or unexplained_mask" -vv
```

Expected: inactive retry 被拒绝，evidence 契约不认识 unexplained mask。

- [ ] **Step 3: 扩展现有动作状态，不新增动作类型**

在 `execute_component_actions` 的状态检查中，仅对 `retry_with_box` 和 `retry_with_points` 允许 `pending` 或 `inactive` visual node。成功生成 mask 后显式设置：

```python
nodes[component_id]["state"] = "pending"
```

text node、frozen node 和被其他冻结 ownership 完全覆盖的 inactive node仍由 graph transition/quality gate保护。同步 skill 镜像。

把 `unexplained-mask.png` 加入 `COMPONENT_EVIDENCE_NAMES`。初始化 round-01 request 时由 schema 5 的 `foreground_evidence_mask` 生成首份 mask；后续 `legacy._publish_next_legacy_request` 从上一 quality artifact 同目录复制 hash-bound mask 到 evidence round。旧 schema 1-4 沿用旧 evidence 名单，不伪造新版证据。

`execute_component_actions` 只收集成功执行 `retry_with_box` / `retry_with_points` 的 inactive visual ID 为 `reactivated_ids`，并把它传给 `validate_graph_transition`。契约仅允许这些 ID 从 `inactive` 变为 `pending`；普通动作、text node、frozen node 和未命中的重试仍不得恢复。这样重试权限来自已校验动作及实际 SAM 结果，不是放宽全局状态转换。

- [ ] **Step 4: 收紧 Local/Host Agent 规则**

Local message 和 `skills/image-to-ppt/SKILL.md` 使用同一规则：

```text
When quality-report.json contains unexplained_visual_residual, inspect
unexplained-mask.png. Every material region must be covered by an active
visual owner or repaired with retry_with_box/retry_with_points on the closest
inactive visual candidate. Do not accept, discard, or classify the region as
background merely to reduce violations. When background_text_residual is the
only blocking violation, issue rebuild_background for the affected frozen
text/visual IDs with evidence from the residual diagnostics.
```

Agent 输出仍必须经过现有 plan schema 与质量门，prompt 不获得放宽权限。

- [ ] **Step 5: 验证 Agent、镜像和 skill**

Run:

```powershell
Copy-Item scripts\visual_segment.py skills\image-to-ppt\scripts\visual_segment.py -Force
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_component_contracts.py tests/test_component_repair.py tests/test_local_agent.py tests/test_psd_skill.py -k "retry_with or unexplained or background_text_residual or skill" -vv
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_dependency_contract.py -k "mirror" -vv
```

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add image2editable/component_contracts.py image2editable/local_agent.py image2editable/legacy.py scripts/visual_segment.py skills/image-to-ppt/scripts/visual_segment.py skills/image-to-ppt/SKILL.md tests/test_component_contracts.py tests/test_component_repair.py tests/test_local_agent.py tests/test_psd_skill.py
git commit -m "功能：闭合页面残差修复动作"
```

### Task 6: 阻止无进步修复与伪成功交付

**Files:**
- Modify: `image2editable/component_repair.py`
- Modify: `image2editable/legacy.py`
- Test: `tests/test_component_repair.py`
- Test: `tests/test_runtime_execution.py`
- Test: `tests/test_runtime_cli.py`

- [ ] **Step 1: 写无进步与整页回退交付失败测试**

增加两个行为测试：

```python
def test_page_progress_key_does_not_treat_equal_quality_as_progress():
    quality = {
        "violations": ["background_text_residual"],
        "visual_metrics": {
            "largest_unexplained_region_pixels": 120,
            "unexplained_visual_pixels": 120,
            "mae": 1.0,
            "p95": 2.0,
        },
    }
    assert component_repair._page_progress_key(quality) == (
        component_repair._page_progress_key(copy.deepcopy(quality))
    )


with pytest.raises(RuntimeError, match="editable reconstruction incomplete"):
    legacy.assemble_legacy_results(store)
assert not (run_dir / "final/output_16x9.pptx").exists()
```

把该断言直接放入现有 `test_warning_page_assembly_preserves_full_source_and_records_warning`，复用它已经创建的 `store`、`run_dir`、source 和 monkeypatch。删除旧的“输出存在”断言。混合 PPTX 测试继续验证原文件恢复资产可以生成，但 delivery/job status 必须是 `preserved_with_warning`，不能写 `ready_for_assembly` 或成功转换计数。

- [ ] **Step 2: 运行并确认 RED**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_component_repair.py tests/test_runtime_execution.py tests/test_runtime_cli.py -k "without_metric_improvement or whole_page_fallback" -vv
```

Expected: 当前状态机只比较 plan hash，图片任务仍会发布整页 PPTX。

- [ ] **Step 3: 实现字典序进步比较**

在 `component_repair.py` 增加：

```python
def _page_progress_key(quality: dict) -> tuple[float, ...]:
    visual = quality["visual_metrics"]
    violations = set(quality["violations"])
    return (
        float(visual.get("largest_unexplained_region_pixels", 0)),
        float(visual.get("unexplained_visual_pixels", 0)),
        float("background_text_residual" in violations),
        float("component_text_residual" in violations),
        float("duplicate_pixels" in violations),
        float("over_merged_component" in violations),
        float(visual.get("mae", 0.0)),
        float(visual.get("p95", 0.0)),
    )
```

页面级下一轮发布时比较上一轮和当前轮 key。只有 key 字典序减小才允许把相同类别动作视为进步；没有改善时，下一 request 的证据明确禁止复用上一标准化计划。仍保留五次 Agent 上限，不引入无限重试。

- [ ] **Step 4: 阻止图片/PDF伪成功**

`assemble_legacy_results` 在输入类型为图片、目录或 PDF 且任一页 `preserved_with_warning` 时，不创建 final PPTX并抛出明确的 incomplete 错误。混合 PPTX 继续生成保护性恢复文件，但 delivery status 保持 warning，不能计入成功页。现有成功路径返回值不变。

- [ ] **Step 5: 验证状态和 CLI**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_component_repair.py tests/test_runtime_execution.py tests/test_runtime_cli.py -k "progress or preserved or fallback or incomplete" -vv
```

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add image2editable/component_repair.py image2editable/legacy.py tests/test_component_repair.py tests/test_runtime_execution.py tests/test_runtime_cli.py
git commit -m "修复：拒绝无进步修复和伪成功交付"
```

### Task 7: 跨输入、真实 PowerPoint 验收与文档收口

**Files:**
- Modify: `tests/test_runtime_input_dispatch.py`
- Modify: `tests/test_task10_runtime_e2e.py`
- Modify: `tests/test_task10_mixed_native.py`
- Modify outside feature worktree: `E:\My_project\Change_PPT\Course.md`（当前被 Git 忽略）

- [ ] **Step 1: 增加跨输入等价性质测试**

用同一参数化页面分别生成 PNG、单页 PDF 和图片版 PPTX，走公开 Runtime prepare/execute 接口，比较：

```python
text_counts = [len(result["text_items"]) for result in results]
component_counts = [len(result["components"]) for result in results]
normalized_boxes = [_normalized_component_boxes(result) for result in results]
assert len(set(text_counts)) == 1
assert len(set(component_counts)) == 1
assert normalized_boxes[0] == pytest.approx(normalized_boxes[1], abs=0.01)
assert normalized_boxes[0] == pytest.approx(normalized_boxes[2], abs=0.01)
assert all(
    result["visual_metrics"]["unexplained_visual_pixels"] == 0
    for result in results
)
```

`results` 是三个公开 Runtime 路径各自读取的 `component_result.json` 与 accepted slide data 合并字典。测试文件内增加：

```python
def _normalized_component_boxes(result: dict) -> list[tuple[float, ...]]:
    width = result["img_width"]
    height = result["img_height"]
    return sorted(
        (
            component["x"] / width,
            component["y"] / height,
            component["w"] / width,
            component["h"] / height,
        )
        for component in result["components"]
    )
```

测试页面至少包含中文文字、英文文字、两个独立图标、一个复杂局部 Raster 对象和渐变背景；fixture 由测试代码参数化生成，不读取当前两个真实样本。

- [ ] **Step 2: 增加混合 PPTX 不变性测试**

构造带原生文字、形状、表格、备注、z-order 和一个获准替换截图对象的混合 PPTX。转换后断言未命中对象的 XML identity、文字、位置、备注和 z-order 不变，截图区域通过 ownership gate。

- [ ] **Step 3: 运行跨输入与混合测试**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest tests/test_runtime_input_dispatch.py tests/test_task10_runtime_e2e.py tests/test_task10_mixed_native.py -vv
```

Expected: PASS；缺少真实 Office 时只允许已有标记的 PowerPoint 集成测试 skip，纯 Python 跨输入测试不能 skip。

- [ ] **Step 4: 运行真实样本验收**

使用全新 Run 目录执行：

```powershell
E:\v\i2e-rr\Scripts\image2editable.exe convert "wsl和虚拟机对比.png" -o ".superpowers\closure-validation\image-output.pptx" --slide-size original --agent-provider host --run-dir ".superpowers\closure-validation\image-run"
E:\v\i2e-rr\Scripts\image2editable.exe prepare "test1.pptx" --run-dir ".superpowers\closure-validation\pptx-run" --agent-provider host
```

按 Host request逐轮提交符合契约的计划，直到页面 ready 或暴露新的通用违反。不得复用旧 Run、旧 plan 或按样本坐标编写生产逻辑。

对成功输出执行 PowerPoint COM 原生打开与渲染，并检查：

- 可靠 OCR 文字是 TextBox；
- 独立图标/卡片是可移动组件；
- 删除全部 TextBox 后无栅格文字；
- 移动组件后背景无对象残留、空洞或重复；
- `unexplained_visual_pixels == 0`；
- 输出不是单一整页图片，也不与输入 PPTX 字节相同。

若任一真实样本仍只产生回退，本计划不得标记完成；根据 quality evidence 新增通用回归测试后返回对应 Task，不写样本特判。

- [ ] **Step 5: 全量验证**

Run:

```powershell
E:\v\i2e-rr\Scripts\python.exe -m pytest -q -p no:faulthandler
git diff --check
git status --short --branch
```

Expected: 全量退出码 0；功能 worktree 干净；PowerPoint 进程最终退出。主工作区的本地 `Course.md` 在下一步单独同步。

- [ ] **Step 6: 更新 Course.md**

在主工作区 `E:\My_project\Change_PPT\Course.md` 同步：

- 当前闭环行为与成功定义；
- 本轮新增 ownership evidence、页面修复轮和背景动作聚合；
- 关键修改文件；
- 图片/PDF/PPTX 运行入口；
- 全量测试与真实 PowerPoint 验收数字；
- 仍未通过的输入类别或风险，不把回退写成成功。

- [ ] **Step 7: 提交验收与文档**

```powershell
git add tests/test_runtime_input_dispatch.py tests/test_task10_runtime_e2e.py tests/test_task10_mixed_native.py
git commit -m "测试：建立可编辑闭环跨输入验收"
```

`Course.md` 位于主工作区且按当前仓库规则本地忽略时，只同步本地文件，不使用 `git add -f` 把它意外纳入仓库。

## 计划自检清单

- 设计中的页面级修复轮由 Task 1 覆盖。
- 全部背景动作和文字区域修补由 Task 2 覆盖。
- 首轮模型漏检后的 residual discovery 由 Task 3 覆盖。
- ownership ledger、unexplained mask 和硬门禁由 Task 4 覆盖。
- Agent 对页面残差的可执行动作与 evidence 由 Task 5 覆盖。
- 单调收敛和禁止伪成功由 Task 6 覆盖。
- 图片、PDF、图片版/混合 PPTX 与 PowerPoint 实测由 Task 7 覆盖。
- 没有新增第二套 Runtime、Router、Scene Graph 或组装器。
- 生产代码中不出现真实样本文件名、固定对象数、坐标、主题词或颜色特判。
