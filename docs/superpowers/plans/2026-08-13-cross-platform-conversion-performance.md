# Cross-Platform Conversion Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不更换模型、不减少候选、不放宽质量门禁的前提下，减少单页转换中的重模型冷启动、整页重复推理和 Agent 重复视觉输入。

**Architecture:** 保留现有隔离 worker 和完整质量闭环，在隔离边界内批量执行同阶段 SAM 操作；所有增量路径均以 hash、依赖闭包和完整性条件约束，不能证明安全时调用原有完整路径。首轮 Agent 使用完整证据，后续轮使用由当前节点和 violation 生成的无损 review atlas，同时保留完整 evidence hash 供验证器校验。

**Tech Stack:** Python 3.10–3.12、PyTorch、SAM 2.1 Large、Grounding DINO tiny、PaddleOCR/Tesseract、OpenCV、Pillow、pytest、python-pptx。

---

## 文件结构

- Create: `scripts/performance_trace.py` — 无内容数据的 JSONL 性能事件和设备摘要。
- Create: `tests/test_conversion_performance.py` — 性能事件、批处理调用次数和跨平台探测契约。
- Modify: `scripts/sam_worker.py` — 候选批处理和组件 prompt 批处理协议。
- Modify: `scripts/visual_segment.py` — 批量 SAM 结果进入现有 action/候选逻辑，局部 crop 坐标映射。
- Modify: `image_to_ppt.py` — 候选批处理调用、OCR 增量依赖闭包、残差 crop 调度。
- Modify: `image2editable/legacy.py` — 同轮多个组件 retry 一次提交给 SAM worker。
- Modify: `image2editable/component_contracts.py` — Agent review evidence 的严格 schema。
- Modify: `image2editable/component_repair.py` — 后续轮 review atlas、完整 evidence hash 和发布绑定。
- Modify: `image2editable/local_agent_worker.py` — 只把 request 指定的本轮视觉证据交给模型。
- Modify: `image2editable/local_agent.py` — 记录 Local Agent 输入规模和执行耗时。
- Modify: `scripts/worker_resources.py` — worker 统一性能事件边界。
- Mirror: `skills/image-to-ppt/scripts/` 下所有同名运行文件。
- Modify: `skills/image-to-ppt/SKILL.md`、`skills/image-to-ppt/references/requirements.txt`、`requirements.txt`、`README.md`、`README_EN.md` — 跨平台说明和固定依赖。
- Modify: `tests/test_component_repair.py`、`tests/test_dependency_contract.py`、`tests/test_local_agent.py`、`tests/test_regressions.py`、`tests/test_targeted_ocr.py`、`tests/test_worker_resources.py` — 对应回归。
- Update: `Course.md` — 当前状态、关键文件、入口、测试和剩余限制；该文件被仓库忽略，合并后同步到主工作区本地文件。

## Task 1: 建立无内容性能记录和跨平台设备摘要

**Files:**
- Create: `scripts/performance_trace.py`
- Create: `tests/test_conversion_performance.py`
- Modify: `scripts/worker_resources.py`
- Modify: `image2editable/local_agent.py`
- Mirror: `skills/image-to-ppt/scripts/performance_trace.py`
- Mirror: `skills/image-to-ppt/scripts/worker_resources.py`

- [ ] **Step 1: 写性能记录的失败测试**

测试使用临时目录和受控 `perf_counter`，要求只写允许字段，不出现 source path、OCR text、prompt 或 response：

```python
def test_performance_trace_records_duration_without_content(tmp_path):
    clock = iter((10.0, 12.5)).__next__
    trace = PerformanceTrace(tmp_path / "performance.jsonl", clock=clock)
    with trace.span("inference", page_id="page_001", model="sam", operation_count=2):
        pass
    event = json.loads((tmp_path / "performance.jsonl").read_text().strip())
    assert event["duration_ms"] == 2500
    assert set(event) == ALLOWED_PERFORMANCE_FIELDS
    assert "path" not in json.dumps(event).lower()
    assert "text" not in json.dumps(event).lower()
```

再添加参数化设备探测测试：Windows/Linux/macOS 字符串只影响 `platform` 字段；CUDA/MPS 探测异常时返回 `device="unknown"`，不得抛错或自行选择新后端。

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest tests/test_conversion_performance.py -q`

Expected: FAIL，`scripts.performance_trace` 尚不存在。

- [ ] **Step 3: 实现最小记录器**

公开接口固定为：

```python
class PerformanceTrace:
    def __init__(self, path: str | Path, *, clock=time.perf_counter): ...
    def event(self, event: str, **fields: object) -> None: ...
    @contextmanager
    def span(self, stage: str, **fields: object): ...

def device_summary(torch_module=None, *, platform_name=platform.system()) -> dict: ...
```

写入采用单行 JSON、UTF-8、`flush()`；字段 allowlist 固定，额外字段直接 `ValueError`。`run_isolated_worker()` 接受可选 `performance_trace/stage/model/operation_count`，不传时行为完全不变。Local Agent 调用只记录请求图片数量、总字节数、耗时和 worker 状态。

- [ ] **Step 4: 运行 GREEN 和现有 worker 测试**

Run: `python -m pytest tests/test_conversion_performance.py tests/test_worker_resources.py tests/test_local_agent.py -q`

Expected: PASS。

- [ ] **Step 5: 同步 skill 镜像并提交**

Run: `python -c "from pathlib import Path; pairs=[('scripts/performance_trace.py','skills/image-to-ppt/scripts/performance_trace.py'),('scripts/worker_resources.py','skills/image-to-ppt/scripts/worker_resources.py')]; assert all(Path(a).read_bytes()==Path(b).read_bytes() for a,b in pairs)"`

```powershell
git add scripts/performance_trace.py scripts/worker_resources.py image2editable/local_agent.py skills/image-to-ppt/scripts/performance_trace.py skills/image-to-ppt/scripts/worker_resources.py tests/test_conversion_performance.py tests/test_worker_resources.py tests/test_local_agent.py
git commit -m "性能：增加转换阶段与设备记录"
```

## Task 2: 一次 SAM 加载完成同阶段候选生成

**Files:**
- Modify: `scripts/sam_worker.py`
- Modify: `image_to_ppt.py`
- Modify: `tests/test_regressions.py`
- Mirror: `skills/image-to-ppt/scripts/sam_worker.py`
- Mirror: `skills/image-to-ppt/scripts/image_to_ppt.py`

- [ ] **Step 1: 写候选批处理 RED 测试**

在 `tests/test_regressions.py` 添加：

```python
def test_isolated_candidate_batch_loads_sam_once(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(sam_worker, "create_sam_generator", lambda *a, **k: loads.append(1) or FakeGenerator())
    result = sam_worker.execute_batch_request(
        image=np.zeros((32, 48, 3), dtype=np.uint8),
        request={"schema_version": 1, "operations": [prompted_operation(), automatic_operation()]},
    )
    assert len(loads) == 1
    assert [item["id"] for item in result] == ["prompted", "automatic"]
```

增加结果项缺失、ID 重复、顺序变化、mask shape 错误时整批拒绝的测试。

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest tests/test_regressions.py -k "candidate_batch" -q`

Expected: FAIL，batch API 尚不存在。

- [ ] **Step 3: 实现候选 batch 协议**

增加 `--mode batch --request REQUEST_JSON_PATH`；request 精确字段为 `schema_version/operations`，operation 精确字段为 `id/kind`，kind 仅允许 `prompted/automatic/recheck/components`，其余输入继续通过当前受控临时文件传入。worker 创建一次 generator，执行全部 operation，将结果先写临时文件，完整验证后 `os.replace()`。

调用端新增：

```python
def _generate_sam_candidate_batch_isolated(
    image, text_mask, proposals, work_dir
) -> tuple[list[MaskCandidate], list[MaskCandidate]]: ...
```

初始页面和每轮 residual 的 prompted+automatic 改为一个 batch；模型、`points_per_side=16`、阈值和过滤顺序不变。recheck 只有依赖已准备好时才加入同一批，否则继续独立运行。

- [ ] **Step 4: 运行 GREEN 和分割回归**

Run: `python -m pytest tests/test_regressions.py -k "sam or segmentation or candidate_batch" -q`

Expected: PASS；测试断言原单项结果与 batch 结果 mask/score/source 顺序一致。

- [ ] **Step 5: 镜像、diff 检查并提交**

Run: `python -c "from pathlib import Path; assert Path('scripts/sam_worker.py').read_bytes()==Path('skills/image-to-ppt/scripts/sam_worker.py').read_bytes(); assert Path('image_to_ppt.py').read_bytes()==Path('skills/image-to-ppt/scripts/image_to_ppt.py').read_bytes()"`

```powershell
git add scripts/sam_worker.py image_to_ppt.py skills/image-to-ppt/scripts/sam_worker.py skills/image-to-ppt/scripts/image_to_ppt.py tests/test_regressions.py
git commit -m "性能：批量执行同阶段 SAM 候选生成"
```

## Task 3: 同轮组件 retry 共享一次 SAM 加载

**Files:**
- Modify: `scripts/visual_segment.py`
- Modify: `scripts/sam_worker.py`
- Modify: `image2editable/component_repair.py`
- Modify: `image2editable/legacy.py`
- Modify: `tests/test_component_repair.py`
- Modify: `tests/test_regressions.py`
- Mirror corresponding product scripts under `skills/image-to-ppt/scripts/`

- [ ] **Step 1: 写同轮多 retry 的 RED 测试**

```python
def test_retry_actions_are_batched_before_graph_mutation(tmp_path):
    calls = []
    def batch_runner(*, image, prompts):
        calls.append(prompts)
        return [mask_left(), mask_right()]
    result = execute_component_actions(
        image(), graph_with_two_pending(), [retry_box("left"), retry_points("right")],
        sam_batch_runner=batch_runner, input_dir=input_dir, output_dir=tmp_path / "out",
    )
    assert len(calls) == 1
    assert [item["component_id"] for item in calls[0]] == ["left", "right"]
    assert result["nodes"] == expected_nodes()
```

另测第二个 mask 无效时输出目录不存在、输入 graph 未修改，证明没有半轮发布。

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest tests/test_component_repair.py -k "retry_actions_are_batched" -q`

Expected: FAIL，`sam_batch_runner` 尚不受支持。

- [ ] **Step 3: 预计算并验证整批 mask**

`execute_component_actions()` 在任何 graph mutation 前收集 retry actions，映射坐标，调用一次 batch runner，验证数量、顺序、shape 和非空 mask；验证通过后才进入原 action 顺序应用结果。保留单项 `sam_runner` 参数用于兼容测试和非 runtime 调用，但 runtime `legacy.py` 必须使用 batch runner。

`sam_worker.py` 增加 `run_component_prompt_batch_worker(image, prompts, work_dir)`，一次加载 SAM 并复用同一 source image，返回与 prompts 严格等长的 mask 列表。

- [ ] **Step 4: 运行 GREEN**

Run: `python -m pytest tests/test_component_repair.py tests/test_regressions.py -k "retry_with or retry_actions or component_prompt" -q`

Expected: PASS。

- [ ] **Step 5: 镜像并提交**

```powershell
git add scripts/visual_segment.py scripts/sam_worker.py image2editable/component_repair.py image2editable/legacy.py skills/image-to-ppt/scripts/visual_segment.py skills/image-to-ppt/scripts/sam_worker.py tests/test_component_repair.py tests/test_regressions.py
git commit -m "性能：合并组件修复轮 SAM 推理"
```

## Task 4: OCR 补回后的增量依赖闭包和完整回退

**Files:**
- Modify: `image_to_ppt.py`
- Modify: `tests/test_targeted_ocr.py`
- Modify: `tests/test_regressions.py`
- Mirror: `skills/image-to-ppt/scripts/image_to_ppt.py`

- [ ] **Step 1: 写依赖闭包 RED 测试**

新增纯函数契约：

```python
def test_text_delta_reopens_intersections_parents_children_and_neighbors():
    scope = _text_delta_recompute_scope(
        old_mask=np.zeros((80, 100), np.uint8),
        new_mask=mask_at(30, 20, 10, 8),
        graph=graph_with_parent_child_and_touching_neighbor(),
        graph_dir=fixture_graph_dir,
        source_sha256="a" * 64,
        cache_identity=matching_identity(),
    )
    assert scope == {"child", "parent", "touching_neighbor"}
```

参数化测试要求 source/text mask/model protocol hash 不匹配、mask 不能读取、节点关系不完整、OCR 文本变化但差集为空时返回 `None`，含义是完整回退。

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest tests/test_targeted_ocr.py -k "text_delta" -q`

Expected: FAIL，scope 函数尚不存在。

- [ ] **Step 3: 持久化首 pass 推理资产并实现增量路径**

首 pass 复用现有 prepared-page 原子清单保存最终 component、element/semantic mask、关系和逐资产 hash，另存最小 cache identity：source hash、旧 text cleanup mask hash、SAM/DINO protocol version。新增文字后使用 cleanup mask 差集（该 mask 已包含文本清理安全边距）计算受影响节点，再加入父子、重叠和 3px 邻接依赖。

仅当 OCR 文字确实新增、cleanup 差集非空、`scope == set()`，且全部缓存、关系、hash、协议和 shape 完整时，零 DINO/SAM 复用首 pass 视觉资产，并重新执行 text-clean、背景、removal、foreground/ownership 和原质量证据路径。`scope` 非空、本身为 `None` 或输出无法通过完整性校验时调用原来的第二次 `_process_image_isolated()`。本任务不在全局视觉流水线上伪实现非空 scope 的局部 SAM；该能力需未来先独立拆分视觉流水线。不吞掉原完整路径异常，不输出原图成功。

- [ ] **Step 4: 验证增量与完整路径等价**

Run: `python -m pytest tests/test_targeted_ocr.py tests/test_regressions.py -k "targeted or text_delta or visual_pass" -q`

Expected: PASS；夹具验证安全空 scope 不启动第二个视觉 worker，并比较其与完整重算的 active IDs、mask union、text items 和 quality violations；普通复杂输入安全执行完整回退。

- [ ] **Step 5: 镜像并提交**

```powershell
git add image_to_ppt.py skills/image-to-ppt/scripts/image_to_ppt.py tests/test_targeted_ocr.py tests/test_regressions.py docs/superpowers/specs/2026-08-13-cross-platform-conversion-performance-design.md docs/superpowers/plans/2026-08-13-cross-platform-conversion-performance.md
git commit -m "性能：复用未受 OCR 影响的视觉资产"
```

## Task 5: 残差 connected-component crop 调度

**Files:**
- Modify: `scripts/visual_segment.py`
- Modify: `image_to_ppt.py`
- Modify: `tests/test_regressions.py`
- Mirror corresponding files under `skills/image-to-ppt/scripts/`

- [ ] **Step 1: 写 crop 规划和回退 RED 测试**

```python
@pytest.mark.parametrize("case", ["touches_crop_edge", "fragmented", "bad_mapping"])
def test_residual_crop_unsafe_cases_require_full_page(case):
    plan = plan_residual_crops(mask_for(case), page_shape=(720, 1280))
    assert plan.mode == "full_page"

def test_residual_crop_round_trip_preserves_mask_pixels():
    plan = plan_residual_crops(two_regions(), page_shape=(720, 1280))
    restored = restore_crop_masks(plan, crop_masks_for(plan))
    assert np.array_equal(restored & two_regions(), two_regions())
```

再测相邻 crop 合并、lossless 原分辨率、padding 不越页面、总 crop 面积无收益时整页回退。

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest tests/test_regressions.py -k "residual_crop" -q`

Expected: FAIL，crop planner 尚不存在。

- [ ] **Step 3: 实现保守局部调度**

新增不可配置的内部 planner，输入当前确定性 residual mask 和页面 shape，输出 `local` crops 或 `full_page`。crop 使用原像素 PNG、上下文边距和重叠合并；任何 SAM mask 触碰非页面 crop 边界即丢弃整批局部结果并执行原整页 DINO/SAM。回映射后重新运行现有 `combine_residual_candidates()`、ownership 和 quality gate。

不修改 residual 阈值，不以 crop 外区域为背景，不把局部处理失败转成 warning 成功。

- [ ] **Step 4: 运行 GREEN 与页面质量测试**

Run: `python -m pytest tests/test_regressions.py tests/test_component_quality.py -k "residual or ownership or unexplained" -q`

Expected: PASS；局部与整页夹具输出的显著 residual coverage 一致。

- [ ] **Step 5: 镜像并提交**

```powershell
git add scripts/visual_segment.py image_to_ppt.py skills/image-to-ppt/scripts/visual_segment.py skills/image-to-ppt/scripts/image_to_ppt.py tests/test_regressions.py tests/test_component_quality.py
git commit -m "性能：局部调度页面残差重建"
```

## Task 6: 后续 Agent 轮使用完整的增量 review evidence

**Files:**
- Modify: `image2editable/component_contracts.py`
- Modify: `image2editable/component_repair.py`
- Modify: `image2editable/local_agent_worker.py`
- Modify: `skills/image-to-ppt/SKILL.md`
- Modify: `tests/test_component_contracts.py`
- Modify: `tests/test_component_repair.py`
- Modify: `tests/test_local_agent.py`

- [ ] **Step 1: 写 request schema 和 atlas 的 RED 测试**

request 新增精确字段 `review_evidence`，值为 evidence name 的有序无重复列表。测试要求第一轮列出全部视觉 evidence；后续轮至少包含 `source.png/reconstructed.png/difference.png/quality-report.json/round-review.png`，存在 unexplained violation 时必须包含 `unexplained-mask.png`。

```python
def test_round_two_review_contains_every_failed_node_and_dependency_neighbor(page_session):
    request = json.loads(build_component_agent_request(page_session, repair_round=2).read_text())
    assert request["review_evidence"] == expected_review_names
    atlas = Image.open(request_path.parent / "round-review.png")
    assert atlas.info["component_ids"].split(",") == ["failed", "parent", "neighbor"]
```

添加 atlas 写入失败时回退为完整 evidence 列表的测试。

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest tests/test_component_contracts.py tests/test_component_repair.py tests/test_local_agent.py -k "review_evidence or round_review" -q`

Expected: FAIL，新字段和 atlas 尚不存在。

- [ ] **Step 3: 实现 strict review evidence**

完整 `evidence` map 继续包含并校验现有文件 hash；后续轮新增 `round-review.png`。atlas 对每个 failed/reopened 节点及其父子、contained pair、重叠和残差邻接节点，使用同坐标无损 crop 展示 source/isolation/ownership/reconstructed/difference/residual，并在图内渲染稳定 ID。

`local_agent_worker._messages()` 只把 `review_evidence` 指定的图片加入消息，但仍从完整 graph 验证输出 plan。Host skill 改为严格查看 request 的 `review_evidence`，不得根据固定文件清单重复打开不相关图片。首轮保持完整证据行为。

- [ ] **Step 4: 运行 GREEN**

Run: `python -m pytest tests/test_component_contracts.py tests/test_component_repair.py tests/test_local_agent.py tests/test_host_agent.py -q`

Expected: PASS；现有 request tamper、hash、HMAC 和 frozen node 测试继续通过。

- [ ] **Step 5: 提交**

```powershell
git add image2editable/component_contracts.py image2editable/component_repair.py image2editable/local_agent_worker.py skills/image-to-ppt/SKILL.md tests/test_component_contracts.py tests/test_component_repair.py tests/test_local_agent.py tests/test_host_agent.py
git commit -m "性能：减少 Agent 修复轮重复视觉证据"
```

## Task 7: 固定 standalone 依赖并修正跨平台说明

**Files:**
- Modify: `skills/image-to-ppt/references/requirements.txt`
- Modify: `requirements.txt`
- Modify: `skills/image-to-ppt/SKILL.md`
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `tests/test_dependency_contract.py`

- [ ] **Step 1: 写依赖和文档 RED 测试**

```python
def test_standalone_sam_dependency_matches_product_pin():
    expected = "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
    assert expected in Path("requirements.txt").read_text(encoding="utf-8")
    assert expected in Path("skills/image-to-ppt/references/requirements.txt").read_text(encoding="utf-8")

def test_skill_does_not_recommend_wsl_unconditionally():
    text = Path("skills/image-to-ppt/SKILL.md").read_text(encoding="utf-8")
    assert "优先使用 Linux/WSL" not in text
    assert "通过 doctor" in text
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest tests/test_dependency_contract.py -q`

Expected: FAIL，standalone 仍使用 `@main` 且 skill 仍统一推荐 WSL。

- [ ] **Step 3: 修正文档和依赖**

固定 commit；文档说明优先当前平台已经正确安装且通过 `doctor` 的硬件加速环境。Windows/Linux CUDA/ROCm 沿 PyTorch 设备接口，macOS 在无真实回归前不把 MPS 设为新默认，CPU 保持完整模型但提示性能较慢。不得写某一型号 GPU 的产品承诺。

- [ ] **Step 4: 运行 GREEN 和镜像契约**

Run: `python -m pytest tests/test_dependency_contract.py tests/test_runtime_resources.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add requirements.txt skills/image-to-ppt/references/requirements.txt skills/image-to-ppt/SKILL.md README.md README_EN.md tests/test_dependency_contract.py tests/test_runtime_resources.py
git commit -m "文档：固定 SAM 依赖并修正跨平台建议"
```

## Task 8: 集成性能汇总并同步 Course.md

**Files:**
- Modify: `image2editable/runtime.py`
- Modify: `image2editable/legacy.py`
- Modify: `tests/test_runtime_execution.py`
- Update ignored local file: `Course.md`

- [ ] **Step 1: 写 run summary RED 测试**

要求 summary 只包含页级聚合，不含原图路径或 OCR 内容：

```python
def test_run_summary_reports_performance_counts_without_content(run_store):
    summary = execute_run(run_store)
    perf = summary["performance"]
    assert perf["sam_model_loads"] == 1
    assert perf["agent_image_count"] >= 0
    assert "path" not in json.dumps(perf).lower()
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest tests/test_runtime_execution.py -k "performance_counts" -q`

Expected: FAIL，summary 尚无 performance 聚合。

- [ ] **Step 3: 实现只读聚合**

runtime 在完成页边界时读取该页 JSONL，按 event/model/stage 聚合 count 和 duration；坏行跳过并记录日志，不改变页面状态。同步更新 `Course.md`：当前状态、本轮变更、关键文件、入口、测试事实和 CPU-only/Mac 真机验证限制。

- [ ] **Step 4: 运行 GREEN**

Run: `python -m pytest tests/test_runtime_execution.py tests/test_runtime_contracts.py -k "performance or summary" -q`

Expected: PASS。

- [ ] **Step 5: 提交 tracked 文件**

```powershell
git add image2editable/runtime.py image2editable/legacy.py tests/test_runtime_execution.py tests/test_runtime_contracts.py
git commit -m "性能：汇总转换模型与 Agent 开销"
```

`Course.md` 被仓库忽略，不强制纳入提交；保留更新内容，并在功能分支合并后同步到主工作区现有 `Course.md`。

## Task 9: 完整回归、真实转换和代码审查

**Files:**
- No production changes unless verification exposes a regression; any fix must start a new RED test.

- [ ] **Step 1: 静态和镜像验证**

Run: `git diff --check main...HEAD`

Run: `python -m pytest tests/test_dependency_contract.py -q`

Expected: PASS，所有产品/skill 同名运行文件字节一致，SAM commit 固定一致。

- [ ] **Step 2: 全量回归**

Run: `python -m pytest -q`

Expected: exit 0；通过数量不得低于基线 `1763 passed, 22 skipped`。记录既有 pytest-asyncio warning 和 Windows PowerPoint COM 退出期诊断，确认没有新增 warning 类型。

- [ ] **Step 3: 真实图片和 PPTX 转换**

使用仓库现有真实输入执行统一 runtime，而非旧单脚本兼容入口：

```powershell
image2editable prepare "E:\My_project\Change_PPT\.worktrees\reconstruction-router-render-qa\wsl和虚拟机对比.png" --run-dir ".superpowers\performance-validation\image-run" --agent-provider host
image2editable run execute ".superpowers\performance-validation\image-run"
image2editable prepare "E:\My_project\Change_PPT\.worktrees\reconstruction-router-render-qa\test1.pptx" --run-dir ".superpowers\performance-validation\pptx-run" --agent-provider host
image2editable run execute ".superpowers\performance-validation\pptx-run"
```

按 Host request 实际查看证据并完成计划。验收：无整页原图伪装、TextBox 可编辑、组件独立、PPTX reopen 通过、`unexplained_visual_pixels=0` 或由真实未解决 violation 阻止成功。

- [ ] **Step 4: 对比性能和质量**

对同一输入记录：SAM/DINO/OCR/Agent model load count、各阶段 duration、Agent image count/bytes、组件数、TextBox 数、quality violations。性能改善不能以组件或文字减少、warning 增加、门禁放宽为代价。

- [ ] **Step 5: 请求代码审查并修复 Critical/Important**

审查范围为设计 commit 的父提交到当前 HEAD，逐条核对本计划“不做”清单、跨平台路径、原子发布、fallback 和隐私字段。任何 Critical/Important 修复必须补 RED 测试并重新运行相关套件。

- [ ] **Step 6: 最终验证后提交收尾**

Run: `git status -sb`

Run: `git log --oneline main..HEAD`

确认只有本轮文件；不要包含主工作区 `.gitignore` 或既有未跟踪文档。

若审查产生修复，回到暴露问题的对应 Task，补 RED 测试、运行该 Task 的明确验证命令，并将修复纳入该 Task 的文件列表与中文提交；没有 tracked 收尾变更时不创建空提交。
