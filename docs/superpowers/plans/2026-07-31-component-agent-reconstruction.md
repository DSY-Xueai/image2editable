# P2.3 通用组件 Agent 重建 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `image2editable:subagent-driven-development` (recommended) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用统一的 Host/Local 视觉 Agent 契约替换当前整页 `component_text_overlap -> text-only fallback`，在图片、PDF、图片版 PPTX 和混合原生对象 PPTX 中保留完整、独立、可移动的视觉组件，并通过最多 5 个页面级批量重修轮次消除文字重影、背景残影、残缺边缘和错误拆分。

**Architecture:** `image_to_ppt.py` 只负责可恢复的初始 OCR/CV/SAM 分层和最终组装；`image2editable/component_repair.py` 持久化页面组件树、证据包和有界重修状态；Host 与 Local Provider 只生成同一份严格 JSON 计划；`scripts/visual_segment.py` 和 `scripts/fg_extract.py` 精确执行掩码动作；确定性门禁负责唯一像素所有权和最终是否交付。PPTX 继续以原文件为底稿，只替换 Agent 已批准的大面积截图对象，未命中原生对象、备注和其他页面保持不变。

**Tech Stack:** Python 3.10–3.12、NumPy、OpenCV、Pillow、SAM 2.1、Transformers、python-pptx、OOXML、pytest；Local Provider 额外使用 Hugging Face Hub 和隔离的视觉语言模型子进程。

---

## 实施边界与固定决策

- 每页只执行一次初始分割；之后最多 5 个批量重修轮次，轮次作用于该页全部未通过组件。
- Agent 只输出 `accept`、`merge`、`split`、`expand`、`shrink`、`retry_with_box`、`retry_with_points`、`attach_text`、`collapse_to_parent`。
- Agent 不能写掩码、修改质量阈值、直接组装 PPTX 或批准绕过门禁。
- 已通过组件立即冻结，后续计划不能引用或修改它。
- 父组件和子组件不能同时成为活动渲染对象；子组件失败时折叠成完整父组件，而不是烘焙进背景。
- 初始组件非零而最终组件为零时，只允许页面显式进入 `preserved_with_warning`。
- `agent_provider` 在创建 Run 时写入 `job_manifest.json`；同一 Run 不允许从 `host` 切换到 `local`，也不自动回退。
- Host 模式不下载、不导入本地 VLM；Local 模式不读取宿主会话或 Host 决策。
- “缓存”只包括模型文件和当前 Run 的不可变证据/计划；不跨图片缓存组件判断、语义结果或掩码，每张图片独立分析。
- 两种 Provider 在通过同一真实验收集前均按实际状态标记；不能仅凭 mock 测试宣称稳定。

## 文件职责

### 新增文件

- `image2editable/component_contracts.py`：Provider、动作、请求、计划、组件树和严格字段校验。
- `image2editable/component_quality.py`：页面自适应标定、逐组件质量门禁、唯一所有权和最终页面门禁。
- `image2editable/component_repair.py`：页面重建状态机、证据包、冻结规则、5 轮限制和父组件折叠。
- `image2editable/host_agent.py`：Host 视觉能力握手、`agent next` 和计划接收。
- `image2editable/local_agent.py`：Local Provider 调度与子进程边界；模块顶层不得导入 VLM 依赖。
- `image2editable/local_agent_worker.py`：单轮加载模型、读取证据、生成严格 JSON、退出释放显存/内存。
- `image2editable/models.py`：硬件探测、版本化目录、推荐、显式安装和状态查询。
- `image2editable/model_catalog.json`：经项目声明的本地模型兼容目录；首个 Qwen 条目保持 `experimental`，直到真实验收通过。
- `tests/test_component_contracts.py`、`tests/test_component_quality.py`、`tests/test_component_repair.py`、`tests/test_host_agent.py`、`tests/test_local_agent.py`、`tests/test_models.py`：新功能的单元和集成测试。

### 修改文件

- `image2editable/contracts.py`：增加 `awaiting_agent` Run/Page 状态与合法转换。
- `image2editable/inputs.py`、`image2editable/pdf_input.py`、`image2editable/pptx_input.py`：在创建 Run 时冻结 Provider。
- `image2editable/runtime.py`：推进或暂停页面重建，返回可机器读取的 `awaiting_agent`，完成后才组装/发布。
- `image2editable/agent.py`：保留现有 PPTX 截图候选路由；不混入组件重修协议。
- `image2editable/cli.py`：增加 `--agent-provider`、`agent next/record` 和 `models recommend/install/status`。
- `image2editable/legacy.py`：图片/PDF 从持久化的已验收页面资产组装，不重复执行视觉模型。
- `image2editable/pptx_reconstruct.py`：从已验收组件资产生成 donor，不在 patch 阶段重新跑分层。
- `image2editable/pptx_shadow_run.py`：只消费已完成 donor，并保留现有单页安全回退。
- `image_to_ppt.py`：拆开“初始分层”和“最终质量/组装”，Agent 管理路径不再整页清空组件。
- `scripts/visual_segment.py`：执行合并、拆分、边界调整和 SAM box/point 重试。
- `scripts/fg_extract.py`：按组件树导出活动对象并保留完整父组件资产。
- `scripts/ppt_assemble.py`：拒绝父子同时输出，写入稳定组件名称和父级元数据。
- `pyproject.toml`：增加 `agent-local` 可选依赖。
- `README.md`、`README_EN.md`、`skills/image-to-ppt/SKILL.md`：双 Provider、视觉能力、隐私、模型和 5 轮流程。
- `skills/image-to-ppt/scripts/`：同步所有被 Skill 直接调用的传统视觉脚本镜像。
- `Course.md`：同步当前状态、行为、关键文件、入口、验收和注意事项。

## Task 1：冻结 Provider 并扩展可暂停状态机

**Files:**

- Create: `image2editable/component_contracts.py`
- Modify: `image2editable/contracts.py`
- Modify: `image2editable/inputs.py`
- Modify: `image2editable/pdf_input.py`
- Modify: `image2editable/pptx_input.py`
- Modify: `image2editable/runtime.py`
- Modify: `image2editable/cli.py`
- Test: `tests/test_component_contracts.py`
- Test: `tests/test_runtime_contracts.py`
- Test: `tests/test_runtime_cli.py`
- Test: `tests/test_pdf_input.py`
- Test: `tests/test_pptx_input.py`

- [ ] **Step 1：先写 Provider 和状态转换失败测试**

```python
def test_prepare_freezes_host_provider_in_manifest(tmp_path, sample_png):
    run = prepare_job(sample_png, run_dir=tmp_path / "run", agent_provider="host")
    manifest = RunStore.open(run).read_json("job_manifest.json")
    assert manifest["options"]["agent_provider"] == "host"


def test_awaiting_agent_can_resume_only_through_prepared():
    document = {"schema_version": 1, "status": "running", "updated_at": utc_now()}
    waiting = transition_run_document(document, RunStatus.AWAITING_AGENT)
    assert transition_run_document(waiting, RunStatus.PREPARED)["status"] == "prepared"
    with pytest.raises(ValueError, match="awaiting_agent -> completed"):
        transition_run_document(waiting, RunStatus.COMPLETED)


@pytest.mark.parametrize("provider", ["", "HOST", "remote", None])
def test_prepare_rejects_unknown_provider(tmp_path, sample_png, provider):
    with pytest.raises(ValueError, match="agent_provider"):
        prepare_job(sample_png, run_dir=tmp_path / "run", agent_provider=provider)
```

- [ ] **Step 2：运行测试并确认先失败**

Run:

```powershell
python -m pytest tests/test_component_contracts.py tests/test_runtime_contracts.py tests/test_runtime_cli.py -q
```

Expected: FAIL，提示缺少 `agent_provider` 或 `AWAITING_AGENT`。

- [ ] **Step 3：实现最小 Provider 校验和状态转换**

```python
# image2editable/component_contracts.py
AGENT_PROVIDERS = frozenset({"host", "local"})
MAX_REPAIR_ROUNDS = 5


def validate_agent_provider(value: object) -> str:
    if not isinstance(value, str) or value not in AGENT_PROVIDERS:
        raise ValueError("agent_provider must be 'host' or 'local'")
    return value
```

在 `RunStatus` 和 `PageStatus` 中增加 `AWAITING_AGENT = "awaiting_agent"`，只开放以下新增路径：

```text
Run:  running -> awaiting_agent -> prepared
Page: processing -> awaiting_agent -> processing
```

给 `prepare_job`、三种输入 prepare 函数和 CLI `prepare/convert` 增加 `agent_provider="host"`，在清单的 `options.agent_provider` 中冻结。`runtime._manifest_input` 每次执行都严格校验该值，不能通过后续 CLI 参数覆盖。

- [ ] **Step 4：补齐 CLI 与三种输入测试并运行**

Run:

```powershell
python -m pytest tests/test_component_contracts.py tests/test_runtime_contracts.py tests/test_runtime_cli.py tests/test_runtime_inputs.py tests/test_pdf_input.py tests/test_pptx_input.py -q
```

Expected: PASS；`--agent-provider host|local` 正确写入清单，旧 Run 缺少字段时给出明确的版本错误而不是静默猜测。

- [ ] **Step 5：提交本任务**

```powershell
git add image2editable/component_contracts.py image2editable/contracts.py image2editable/inputs.py image2editable/pdf_input.py image2editable/pptx_input.py image2editable/runtime.py image2editable/cli.py
git add -f tests/test_component_contracts.py tests/test_runtime_contracts.py tests/test_runtime_cli.py tests/test_pdf_input.py tests/test_pptx_input.py
git commit -m "功能：冻结组件Agent提供方并增加等待状态"
```

## Task 2：把初始视觉分层变成可恢复的页面资产

**Files:**

- Modify: `image_to_ppt.py`
- Modify: `scripts/visual_worker.py`
- Modify: `scripts/visual_segment.py`
- Modify: `scripts/fg_extract.py`
- Test: `tests/test_regressions.py`
- Test: `tests/test_ocr_isolation.py`
- Test: `tests/test_runtime_execution.py`

- [ ] **Step 1：写出“初始分层不触发整页清空”的失败测试**

```python
def test_prepare_component_layers_persists_nonzero_candidates_before_quality(
    tmp_path, monkeypatch
):
    # 使用合成的卡片、线条、阴影和文字交叠图；传统分层返回两个候选。
    prepared = image_to_ppt.prepare_component_layers(
        SOURCE,
        tmp_path / "reconstruction",
        lang="ch",
        resource_isolation=True,
    )
    assert prepared["initial_component_count"] == 2
    assert len(prepared["components"]) == 2
    assert prepared["phase"] == "initial_layers"
    assert "quality_fallback" not in prepared
    assert Path(prepared["state_path"]).is_file()
```

同时增加恢复测试：关闭进程后重新读取 `prepared_page.json`，所有路径都必须位于页面 `reconstruction` 目录，且源图、OCR mask、组件 mask、RGBA 和背景均通过 SHA-256 校验。

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_regressions.py -k "prepare_component_layers or agent_managed" -q
python -m pytest tests/test_ocr_isolation.py -k "visual_processing or product_and_skill" -q
```

Expected: FAIL，当前只有一次性 `_prepare_multiple_images`，且最终阶段会调用 `_apply_text_only_fallback`。

- [ ] **Step 3：实现分阶段内部 API**

在 `image_to_ppt.py` 增加以下边界：

```python
def prepare_component_layers(
    image_path: str | Path,
    work_dir: str | Path,
    *,
    lang: str,
    resource_isolation: bool,
) -> dict:
    """Run OCR/CV/SAM once and persist hash-bound, unfinalized page layers."""


def load_component_layers(state_path: str | Path) -> dict:
    """Reload only files owned by the page reconstruction directory."""


def finalize_component_layers(prepared: dict, accepted: dict, *, lang: str) -> dict:
    """Run final deterministic QA without invoking page-level text-only fallback."""
```

具体要求：

- 复用现有 `_process_image(image_path, work_dir, object_detector, mask_generator, lang, text_analysis=text_analysis, defer_quality=True, _resource_isolation=resource_isolation)`；初始 OCR/CV/SAM 仍只运行一次。
- 将 `_work_dir`、`_text_mask_path`、`_element_mask_paths` 改成可序列化的相对路径记录；每个文件保存 SHA-256。
- 保留 `initial_component_count`，后续任何普通成功路径都不能把非零值变成零。
- `finalize_component_layers` 可以返回门禁失败报告，但不得调用 `_apply_text_only_fallback`；兼容入口原有行为暂不改变。
- `visual_worker.py` 继续作为重型模型释放边界，返回 JSON 中不得出现工作目录外路径。

- [ ] **Step 4：验证资源释放和旧兼容入口**

Run:

```powershell
python -m pytest tests/test_ocr_isolation.py tests/test_regressions.py -k "component_layers or visual_processing or clear_alpha or text_only_fallback or product_and_skill" -q
```

Expected: PASS；分阶段 API 可恢复，现有直接 `image_to_ppt` 测试不回归，Skill 镜像哈希暂时仍一致。

- [ ] **Step 5：同步传统视觉镜像并提交**

先使用 `apply_patch` 将四个产品文件的同一改动同步到对应 Skill 镜像，再运行：

```powershell
python -m pytest tests/test_ocr_isolation.py -k product_and_skill -q
git add image_to_ppt.py scripts/visual_worker.py scripts/visual_segment.py scripts/fg_extract.py skills/image-to-ppt/scripts/image_to_ppt.py skills/image-to-ppt/scripts/visual_worker.py skills/image-to-ppt/scripts/visual_segment.py skills/image-to-ppt/scripts/fg_extract.py
git add -f tests/test_regressions.py tests/test_ocr_isolation.py tests/test_runtime_execution.py
git commit -m "重构：持久化可恢复的初始组件分层"
```

## Task 3：建立通用组件树和唯一像素所有权

**Files:**

- Modify: `image2editable/component_contracts.py`
- Create: `image2editable/component_quality.py`
- Modify: `scripts/visual_segment.py`
- Modify: `scripts/fg_extract.py`
- Test: `tests/test_component_contracts.py`
- Test: `tests/test_component_quality.py`
- Test: `tests/test_regressions.py`

- [ ] **Step 1：先写父子互斥、冻结和所有权失败测试**

```python
def test_parent_and_children_cannot_render_together():
    graph = component_graph(parent_active=True, child_active=True)
    with pytest.raises(ValueError, match="parent and child"):
        validate_component_graph(graph)


def test_each_foreground_pixel_has_one_active_owner():
    first = mask_at(2, 2, 8, 8)
    second = mask_at(6, 6, 12, 12)
    report = validate_pixel_ownership([first, second], text_mask=zeros(), shape=first.shape)
    assert report["valid"] is False
    assert report["duplicate_pixels"] == 4


def test_frozen_component_hash_cannot_change():
    with pytest.raises(ValueError, match="frozen"):
        validate_graph_transition(before=FROZEN_GRAPH, after=MUTATED_GRAPH)
```

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_component_contracts.py tests/test_component_quality.py -q
```

Expected: FAIL，缺少组件树与所有权实现。

- [ ] **Step 3：实现严格组件树结构**

组件节点固定字段如下，未知字段一律拒绝：

```python
COMPONENT_STATES = frozenset({"pending", "failed", "frozen", "inactive"})
COMPONENT_KINDS = frozenset({"parent", "child", "text"})

# component-graph.json 中的视觉节点
{
    "id": "component_0001",
    "kind": "child",
    "parent_id": "parent_0001",
    "state": "pending",
    "mask": "masks/component_0001.png",
    "mask_sha256": sha256_file(mask_path),
    "bbox": [x1, y1, x2, y2],
    "z_index": 0,
    "text_ids": [],
}
```

实现规则：

- 初始每个语义实例都保存完整 `parent` 掩码；可拆区域作为它的 `child`。
- 父组件活动时所有后代为 `inactive`；子组件活动时父组件只保留元数据。
- `validate_graph_transition` 比较冻结节点的 mask hash、bbox、z-index、parent 和 text 归属。
- `validate_pixel_ownership` 分别报告 duplicate、missing、text duplicate 和越界像素，不自动修正。
- Alpha 半透明允许正常合成，但源组件证据、阴影和抗锯齿边缘只能属于一个活动组件。

- [ ] **Step 4：运行组件树、传统分割回归测试**

Run:

```powershell
python -m pytest tests/test_component_contracts.py tests/test_component_quality.py tests/test_regressions.py -k "component or ownership or visual_element" -q
```

Expected: PASS；任意类别名称不参与父子判断，测试样例使用卡片、人物轮廓、密集线条和渐变对象。

- [ ] **Step 5：同步镜像并提交**

先使用 `apply_patch` 将两个产品文件的同一改动同步到对应 Skill 镜像，再运行：

```powershell
git add image2editable/component_contracts.py image2editable/component_quality.py scripts/visual_segment.py scripts/fg_extract.py skills/image-to-ppt/scripts/visual_segment.py skills/image-to-ppt/scripts/fg_extract.py
git add -f tests/test_component_contracts.py tests/test_component_quality.py tests/test_regressions.py
git commit -m "功能：建立组件层级与唯一像素所有权"
```

## Task 4：生成不可变、哈希绑定的 Agent 证据包

**Files:**

- Create: `image2editable/component_repair.py`
- Modify: `image2editable/component_contracts.py`
- Test: `tests/test_component_repair.py`
- Test: `tests/test_component_contracts.py`

- [ ] **Step 1：写证据完整性和篡改失败测试**

```python
EXPECTED_EVIDENCE = {
    "source.png",
    "numbered-masks.png",
    "ocr-overlay.png",
    "ownership.png",
    "reconstructed.png",
    "difference.png",
    "component-graph.json",
    "quality-report.json",
}


def test_build_request_hash_binds_every_evidence_file(page_session):
    request_path = build_component_agent_request(page_session, repair_round=1)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert set(request["evidence"]) == EXPECTED_EVIDENCE
    assert all(len(record["sha256"]) == 64 for record in request["evidence"].values())


def test_validate_request_rejects_changed_overlay(page_session):
    request_path = build_component_agent_request(page_session, repair_round=1)
    overlay = request_path.parent / "ocr-overlay.png"
    overlay.write_bytes(overlay.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="evidence hash"):
        load_component_agent_request(request_path)
```

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_component_repair.py tests/test_component_contracts.py -k "request or evidence" -q
```

Expected: FAIL，证据构建器尚不存在。

- [ ] **Step 3：实现轮次目录和原子写入**

每轮固定写入：

```text
pages/<page_id>/reconstruction/agent/round-01/
pages/<page_id>/reconstruction/agent/round-02/
pages/<page_id>/reconstruction/agent/round-03/
pages/<page_id>/reconstruction/agent/round-04/
pages/<page_id>/reconstruction/agent/round-05/
```

`component_agent_request.json` 严格包含：

```python
{
    "schema_version": 1,
    "page_id": page_id,
    "provider": provider,
    "repair_round": repair_round,
    "source_sha256": source_sha256,
    "graph_sha256": graph_sha256,
    "candidate_ids": sorted(pending_ids),
    "frozen_ids": sorted(frozen_ids),
    "evidence": {name: {"path": relative_path, "sha256": digest}},
}
```

文件先写 `.tmp` 再原子替换；请求发布后不得覆盖。所有相对路径必须解析在该轮目录或页面重建目录内，拒绝符号链接、reparse point 和跨页路径。

- [ ] **Step 4：运行证据与并发安全测试**

Run:

```powershell
python -m pytest tests/test_component_repair.py tests/test_component_contracts.py tests/test_execution_lease.py -q
```

Expected: PASS；同一页同一轮不能发布两个不同请求，篡改任一证据会在 Agent 调用前失败。

- [ ] **Step 5：提交本任务**

```powershell
git add image2editable/component_repair.py image2editable/component_contracts.py
git add -f tests/test_component_repair.py tests/test_component_contracts.py
git commit -m "功能：生成哈希绑定的组件Agent证据包"
```

## Task 5：实现 Host 视觉能力握手和计划接收

**Files:**

- Create: `image2editable/host_agent.py`
- Modify: `image2editable/component_contracts.py`
- Modify: `image2editable/runtime.py`
- Modify: `image2editable/cli.py`
- Test: `tests/test_host_agent.py`
- Test: `tests/test_runtime_cli.py`
- Test: `tests/test_runtime_execution.py`

- [ ] **Step 1：写 Host 不加载本地模型、握手优先和严格计划测试**

```python
def test_host_next_returns_visual_handshake_before_real_page(host_run):
    item = next_host_agent_item(host_run)
    assert item["kind"] == "capability_handshake"
    assert Path(item["image_path"]).is_absolute()
    assert item["required_capabilities"] == [
        "vision", "local_file_read", "tool_use", "structured_json"
    ]


def test_host_module_does_not_import_local_provider():
    source = Path(host_agent.__file__).read_text(encoding="utf-8")
    assert "local_agent" not in source
    assert "transformers" not in source
    assert "torch" not in source


def test_record_plan_rejects_provider_round_and_hash_mismatch(host_request, tmp_path):
    plan = valid_plan(host_request) | {"provider": "local"}
    with pytest.raises(ValueError, match="provider"):
        record_host_plan(host_request.run_dir, write_json(tmp_path / "plan.json", plan))
```

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_host_agent.py tests/test_runtime_cli.py -q
```

Expected: FAIL，`agent next/record` 尚不存在。

- [ ] **Step 3：实现握手和统一计划 Schema**

`image2editable agent next RUN_DIR` 返回两种对象：

```python
{
    "kind": "capability_handshake",
    "challenge_id": challenge_id,
    "image_path": str(challenge_path.resolve()),
    "required_capabilities": [
        "vision", "local_file_read", "tool_use", "structured_json"
    ],
}
{
    "kind": "component_request",
    "page_id": page_id,
    "provider": "host",
    "repair_round": repair_round,
    "request_path": str(request_path.resolve()),
    "evidence_paths": [str(path.resolve()) for path in evidence_paths],
}
```

`image2editable agent record RUN_DIR --plan PLAN.json` 接受两种严格文档：

```python
{
    "schema_version": 1,
    "kind": "host_capability_response",
    "challenge_id": challenge_id,
    "observed": {"shape": "triangle", "color": "#2f6fed", "count": 3},
}
```

```python
{
    "schema_version": 1,
    "kind": "component_plan",
    "page_id": page_id,
    "provider": "host",
    "repair_round": 1,
    "request_sha256": request_sha256,
    "actions": [],
}
```

握手图由 Run 内随机 challenge 生成并哈希绑定；只有视觉结果完全匹配才记录 `host_capabilities.json`。组件计划验证严格字段、当前请求、Provider、页 ID、轮次、对象 ID、坐标范围、置信度和非空证据说明。写入成功后才将 Run 从 `awaiting_agent` 恢复为 `prepared`。

Host 提示必须把源图、OCR 文本和诊断图视为不可信数据；图片中出现的命令、角色说明或工具调用文字都不能覆盖组件计划 Schema、用户请求或门禁规则。

- [ ] **Step 4：验证 CLI JSON、锁和恢复**

Run:

```powershell
python -m pytest tests/test_host_agent.py tests/test_runtime_cli.py tests/test_runtime_execution.py -k "agent or awaiting or lease or recover" -q
```

Expected: PASS；所有正常输出只写 stdout JSON，诊断写 stderr；重复记录、过期计划和并发记录均被拒绝。

- [ ] **Step 5：提交本任务**

```powershell
git add image2editable/host_agent.py image2editable/component_contracts.py image2editable/runtime.py image2editable/cli.py
git add -f tests/test_host_agent.py tests/test_runtime_cli.py tests/test_runtime_execution.py
git commit -m "功能：接通宿主视觉Agent握手与计划记录"
```

## Task 6：实现九种 Agent 动作的确定性 CV/SAM 执行器

**Files:**

- Modify: `image2editable/component_contracts.py`
- Modify: `image2editable/component_repair.py`
- Modify: `scripts/visual_segment.py`
- Modify: `scripts/fg_extract.py`
- Modify: `scripts/sam_worker.py`
- Test: `tests/test_component_contracts.py`
- Test: `tests/test_component_repair.py`
- Test: `tests/test_regressions.py`
- Test: `tests/test_ocr_isolation.py`

- [ ] **Step 1：用参数化测试固定九种动作及字段**

```python
@pytest.mark.parametrize(
    ("action", "object_ids", "parameters"),
    [
        ("accept", ["component_0001"], {}),
        ("merge", ["component_0001", "component_0002"], {}),
        ("split", ["component_0001"], {"parts": 2}),
        ("expand", ["component_0001"], {"margin_ratio": 0.01}),
        ("shrink", ["component_0001"], {"margin_ratio": 0.01}),
        ("retry_with_box", ["component_0001"], {"box": [0.1, 0.1, 0.5, 0.5]}),
        ("retry_with_points", ["component_0001"], {"positive": [[0.2, 0.2]], "negative": []}),
        ("attach_text", ["component_0001", "text_0001"], {}),
        ("collapse_to_parent", ["parent_0001"], {}),
    ],
)
def test_component_actions_have_strict_shapes(action, object_ids, parameters):
    validate_component_action({
        "action": action,
        "object_ids": object_ids,
        "parameters": parameters,
        "confidence": 0.95,
        "evidence": ["visible relationship in numbered masks"],
    })
```

增加拒绝测试：未知动作、重复 ID、冻结 ID、跨父级非法 merge、归一化坐标越界、`split.parts < 2`、空 evidence、NaN confidence、一次计划对同一对象执行冲突动作。

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_component_contracts.py tests/test_component_repair.py -k "action or merge or split or retry" -q
```

Expected: FAIL，动作校验与执行器尚未实现。

- [ ] **Step 3：实现动作执行但不自动验收**

在 `scripts/visual_segment.py` 增加纯执行入口：

```python
def execute_component_actions(
    image: np.ndarray,
    graph: dict,
    actions: list[dict],
    *,
    sam_runner,
) -> dict:
    """Return a new graph; never mutate frozen nodes or decide gate outcomes."""
```

执行约束：

- `merge` 用语义掩码并集创建新父级，原子节点转 `inactive`。
- `split` 先用连通/边缘/已有 proposal 生成指定数量候选，不足时保持失败，不能硬切矩形。
- `expand`/`shrink` 的像素半径由页面短边乘归一化比例得到，并限制在原父级语义支持附近。
- `retry_with_box`/`retry_with_points` 通过独立 `sam_worker.py` 调用 SAM；坐标从 0..1 映射到实际页面。
- `attach_text` 只改变文字归属，不把文字像素并入组件。
- `accept` 仅把对象标记为待门禁；只有门禁通过后才能冻结。
- `collapse_to_parent` 激活完整父掩码并停用后代。
- 每轮输出新 mask 文件和新 graph；不覆盖前一轮证据。

- [ ] **Step 4：验证动作执行、SAM 进程释放和镜像**

Run:

```powershell
python -m pytest tests/test_component_contracts.py tests/test_component_repair.py tests/test_regressions.py tests/test_ocr_isolation.py -k "component_action or sam_worker or mask" -q
```

Expected: PASS；动作结果可复现，冻结对象 hash 不变，SAM 子进程结束后可继续下一轮。

- [ ] **Step 5：同步镜像并提交**

先使用 `apply_patch` 将三个产品文件的同一改动同步到对应 Skill 镜像，再运行：

```powershell
git add image2editable/component_contracts.py image2editable/component_repair.py scripts/visual_segment.py scripts/fg_extract.py scripts/sam_worker.py skills/image-to-ppt/scripts/visual_segment.py skills/image-to-ppt/scripts/fg_extract.py skills/image-to-ppt/scripts/sam_worker.py
git add -f tests/test_component_contracts.py tests/test_component_repair.py tests/test_regressions.py tests/test_ocr_isolation.py
git commit -m "功能：实现组件Agent动作执行器"
```

## Task 7：实现页面自适应质量门禁

**Files:**

- Modify: `image2editable/component_quality.py`
- Modify: `image2editable/component_repair.py`
- Modify: `image_to_ppt.py`
- Modify: `scripts/visual_segment.py`
- Test: `tests/test_component_quality.py`
- Test: `tests/test_regressions.py`

- [ ] **Step 1：写分辨率、对比度和缺陷类型不变性测试**

```python
@pytest.mark.parametrize("scale", [1, 2, 4])
def test_gate_decision_is_stable_across_resolution(scale):
    case = render_synthetic_component_case(scale=scale, contrast=0.35)
    assert evaluate_component(case)["accepted"] is True


@pytest.mark.parametrize(
    "defect",
    ["duplicate_shadow", "missing_edge", "text_ghost", "alpha_halo", "parent_child_double"],
)
def test_hard_safety_defects_never_pass(defect):
    case = render_synthetic_component_case(defect=defect)
    report = evaluate_component(case)
    assert report["accepted"] is False
    assert defect in report["violations"]
```

增加“同一固定阈值不适用于高噪照片和纯色 UI”的对照测试，要求校准结果随页面噪声、局部对比度、边缘宽度和字号改变，但最终缺陷判定保持一致。

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_component_quality.py -q
```

Expected: FAIL，当前门禁只有页级固定指标。

- [ ] **Step 3：实现标定和分层门禁**

```python
@dataclass(frozen=True)
class PageCalibration:
    noise_l1: float
    local_contrast: float
    edge_width_px: int
    text_halo_px: int
    min_component_pixels: int


def calibrate_page(source: np.ndarray, text_mask: np.ndarray) -> PageCalibration:
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    median = cv2.medianBlur(gray, 3)
    noise_l1 = float(np.median(np.abs(gray.astype(np.float32) - median)))
    lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    local_mean = cv2.blur(lab, (9, 9))
    local_contrast = float(np.median(np.linalg.norm(lab - local_mean, axis=2)))
    text = np.asarray(text_mask) > 0
    distance = cv2.distanceTransform(text.astype(np.uint8), cv2.DIST_L2, 5)
    text_halo_px = max(1, int(round(float(np.percentile(distance[text], 50))))) if np.any(text) else 1
    edge_width_px = max(1, int(round(max(noise_l1, 1.0) ** 0.5)))
    min_component_pixels = max(20, int(round(source.shape[0] * source.shape[1] * 1e-5)))
    return PageCalibration(
        noise_l1=noise_l1,
        local_contrast=local_contrast,
        edge_width_px=edge_width_px,
        text_halo_px=text_halo_px,
        min_component_pixels=min_component_pixels,
    )


def evaluate_component(
    source: np.ndarray,
    background: np.ndarray,
    reconstructed: np.ndarray,
    node: dict,
    graph: dict,
    calibration: PageCalibration,
) -> dict:
    metrics = component_metrics(
        source, background, reconstructed, node, graph, calibration
    )
    violations = [
        *hard_safety_violations(metrics, node, graph),
        *adaptive_quality_violations(metrics, calibration),
    ]
    return {
        "component_id": node["id"],
        "accepted": not violations,
        "metrics": metrics,
        "violations": sorted(set(violations)),
    }
```

实现要求：

- 用页面采样的 MAD 噪声、局部 Lab 对比度、距离变换估计边缘宽度和 OCR 尺度进行归一化。
- 每个组件报告 missing、duplicate、edge、shadow、alpha、text 和 ownership 指标以及相对上一轮改善。
- 大面积重复/缺失、文字重复、父子双渲染、越界、背景组件副本、受保护原生对象遮挡和 PPTX 无法重开是不可放宽的硬门禁。
- Agent 提供的 confidence 只用于报告和动作冲突排序，不能改变门禁阈值。
- 最终页级 `visual_difference` 保留作为总门禁，但不再触发 `components=[]`。

- [ ] **Step 4：运行合成缺陷和现有视觉回归测试**

Run:

```powershell
python -m pytest tests/test_component_quality.py tests/test_regressions.py -k "quality or residual or shadow or alpha or component_text_overlap" -q
```

Expected: PASS；当前 `component_text_overlap` 用例转为组件级失败报告，不再触发整页 text-only 成功。

- [ ] **Step 5：同步镜像并提交**

先使用 `apply_patch` 将两个产品文件的同一改动同步到对应 Skill 镜像，再运行：

```powershell
git add image2editable/component_quality.py image2editable/component_repair.py image_to_ppt.py scripts/visual_segment.py skills/image-to-ppt/scripts/image_to_ppt.py skills/image-to-ppt/scripts/visual_segment.py
git add -f tests/test_component_quality.py tests/test_regressions.py
git commit -m "功能：增加页面自适应组件质量门禁"
```

## Task 8：实现最多 5 个批量重修轮次和父组件回退

**Files:**

- Modify: `image2editable/component_repair.py`
- Modify: `image2editable/component_contracts.py`
- Modify: `image2editable/runtime.py`
- Test: `tests/test_component_repair.py`
- Test: `tests/test_runtime_execution.py`
- Test: `tests/test_execution_lease.py`

- [ ] **Step 1：写轮次上限、冻结、提前停止和父级回退测试**

```python
def test_repair_round_is_page_batch_not_per_component(session):
    for expected_round in range(1, 6):
        outcome = advance_component_repair(session)
        assert outcome["repair_round"] == expected_round
        assert set(outcome["candidate_ids"]) == set(session.failed_ids)
    assert session.plan_count == 5


def test_passed_component_freezes_while_failed_sibling_retries(session):
    first = advance_with_plan(session, plan_accepting("component_0001"))
    frozen_hash = first.graph.node("component_0001")["mask_sha256"]
    second = advance_with_plan(session, plan_repairing("component_0002"))
    assert second.graph.node("component_0001")["mask_sha256"] == frozen_hash


def test_round_five_collapses_failed_children_to_intact_parent(session):
    outcome = exhaust_five_rounds(session)
    assert outcome["parent_states"] == {"parent_0001": "parent_preserved"}
    assert outcome["active_ids"] == ["parent_0001"]
```

还要测试：空动作提前停止、连续两次相同规范化计划提前停止、无可执行动作提前停止、父组件门禁失败后页面 `preserved_with_warning`、初始非零组件不能普通成功为零、中断恢复保持当前轮次和 Provider。

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_component_repair.py tests/test_runtime_execution.py -k "five_round or freeze or collapse or awaiting_agent" -q
```

Expected: FAIL，页面重修推进器尚未完成。

- [ ] **Step 3：实现单步推进器**

```python
def advance_component_repair(store: RunStore, page_id: str) -> dict:
    """Advance exactly one durable state boundary under ExecutionLease."""
```

单步推进规则：

1. 无初始状态：运行一次初始分层、建树、校验、生成第一轮证据。
2. Host 且无当前计划：返回 `awaiting_agent`，不持有模型和文件句柄。
3. 有计划：验证请求和计划 hash，执行本轮全部动作，运行逐组件门禁并冻结通过对象。
4. 仍有失败且轮次小于 5：生成下一轮不可变证据。
5. 空计划、重复计划或无可执行动作：提前折叠失败子树。
6. 第 5 轮后：折叠失败子树并验证完整父组件。
7. 父组件通过：`parent_preserved`；父组件失败：页面 `preserved_with_warning`。
8. 所有活动组件和页面门禁通过：写 `component_result.json`，状态可进入最终组装。

每一步都先写新文件再原子更新 `component_state.json`。恢复只读取当前状态指向的完整轮次；孤立的 `.tmp` 或未引用轮次不参与决策。

- [ ] **Step 4：验证并发、恢复和上限**

Run:

```powershell
python -m pytest tests/test_component_repair.py tests/test_runtime_execution.py tests/test_execution_lease.py -k "component or awaiting_agent or recover or concurrent" -q
```

Expected: PASS；任何页面都不可能出现第 6 轮，两个进程不能领取同一轮，重启后不会重新跑已完成的初始分割。

- [ ] **Step 5：提交本任务**

```powershell
git add image2editable/component_repair.py image2editable/component_contracts.py image2editable/runtime.py
git add -f tests/test_component_repair.py tests/test_runtime_execution.py tests/test_execution_lease.py
git commit -m "功能：实现五轮组件重修与父组件回退"
```

## Task 9：把图片和 PDF 接入共同重建状态机

**Files:**

- Modify: `image2editable/runtime.py`
- Modify: `image2editable/legacy.py`
- Modify: `image_to_ppt.py`
- Modify: `scripts/ppt_assemble.py`
- Test: `tests/test_runtime_execution.py`
- Test: `tests/test_pdf_input.py`
- Test: `tests/test_runtime_inputs.py`
- Test: `tests/test_regressions.py`

- [ ] **Step 1：写图片/PDF Host 暂停与最终组装测试**

```python
@pytest.mark.parametrize("input_kind", ["images", "pdf"])
def test_host_run_pauses_and_resumes_each_page_without_reprocessing_initial_layers(
    prepared_run, input_kind, monkeypatch
):
    first = run_job(prepared_run(input_kind, provider="host"))
    assert first["status"] == "awaiting_agent"
    record_all_host_plans(first["run_dir"])
    completed = execute_until_complete(first["run_dir"])
    assert completed["status"] == "completed"
    assert initial_segmentation_call_count() == completed["pages"]


def test_pdf_and_image_use_same_component_gate(prepared_image_run, prepared_pdf_run):
    image_report = run_recorded_provider(prepared_image_run)
    pdf_report = run_recorded_provider(prepared_pdf_run)
    assert image_report["quality_gate_version"] == pdf_report["quality_gate_version"]
```

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_runtime_execution.py tests/test_pdf_input.py -k "component_agent or awaiting_agent or prepared_layers" -q
```

Expected: FAIL，当前 `execute_legacy` 一次性运行完整转换。

- [ ] **Step 3：实现逐页推进和只组装一次**

- `runtime._run_job` 对图片/PDF 逐页调用 `advance_component_repair`；遇到 Host 等待立即安全退出并写 `run_summary.json` 的临时等待摘要，不进入 `COMPLETED`。
- 页面完成后，`legacy.execute_legacy` 只读取 `component_result.json` 和已验收资产，调用 `assemble_pptx`/`assemble_pptx_multi`，不能再次调用 OCR、DINO、SAM、LaMa 或 Agent。
- 多页 PDF 仍串行；所有页面完成后一次性发布输出。
- `convert --agent-provider host` 首次调用可以返回 `awaiting_agent` 和 `run_dir`，此时尚无最终输出；Skill 负责循环恢复。`convert --agent-provider local` 在模型已安装且页面可通过时保持无人值守完成。
- `run status` 返回当前页、当前轮次、已冻结组件数、待处理组件数、Provider 和绝对诊断目录。
- 组装前再次验证所有资产 hash 和唯一所有权；任一页 `preserved_with_warning` 时，图片/PDF 输出按该页源图整体保留并记录明确警告，不伪装成完整可编辑成功。

- [ ] **Step 4：验证图片/PDF、输出防覆盖和资源策略**

Run:

```powershell
python -m pytest tests/test_runtime_execution.py tests/test_runtime_inputs.py tests/test_pdf_input.py tests/test_runtime_resources.py tests/test_regressions.py -q
```

Expected: PASS；旧输出不覆盖，图片/PDF 都使用同一组件门禁，重型页并发仍为 1。

- [ ] **Step 5：同步组装镜像并提交**

先使用 `apply_patch` 将两个产品文件的同一改动同步到对应 Skill 镜像，再运行：

```powershell
git add image2editable/runtime.py image2editable/legacy.py image_to_ppt.py scripts/ppt_assemble.py skills/image-to-ppt/scripts/image_to_ppt.py skills/image-to-ppt/scripts/ppt_assemble.py
git add -f tests/test_runtime_execution.py tests/test_runtime_inputs.py tests/test_pdf_input.py tests/test_runtime_resources.py tests/test_regressions.py
git commit -m "功能：统一图片与PDF组件Agent重建流程"
```

## Task 10：把截图型 PPTX 接入共同重建并保护原生对象

**Files:**

- Modify: `image2editable/runtime.py`
- Modify: `image2editable/pptx_reconstruct.py`
- Modify: `image2editable/pptx_shadow_run.py`
- Modify: `image2editable/pptx_input.py`
- Modify: `scripts/ppt_assemble.py`
- Test: `tests/test_pptx_reconstruct.py`
- Test: `tests/test_pptx_shadow_run.py`
- Test: `tests/test_pptx_shadow.py`
- Test: `tests/test_pptx_input.py`
- Test: `tests/test_runtime_execution.py`

- [ ] **Step 1：写 donor 复用、组件非零和原生对象保护测试**

```python
def test_donor_uses_accepted_assets_without_rerunning_cv(reconstruction, monkeypatch):
    monkeypatch.setattr(image_to_ppt, "prepare_component_layers", unexpected_call)
    manifest = build_reconstruction_donor_from_result(reconstruction.result_path)
    assert manifest["components"] > 0


def test_mixed_pptx_preserves_native_objects_while_replacing_screenshot(mixed_pptx_run):
    before = scan_pptx(mixed_pptx_run.source)
    after = scan_pptx(execute_recorded_provider(mixed_pptx_run))
    assert protected_object_hashes(after) == protected_object_hashes(before)
    assert replaced_screenshot_ids(after).isdisjoint(replaced_screenshot_ids(before))


def test_nonzero_initial_components_cannot_publish_zero_component_donor(run):
    corrupt_final_result_to_zero_components(run)
    result = execute_pptx_shadow(run)
    assert result["page_results"][0]["status"] == "preserved_with_warning"
```

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_pptx_reconstruct.py tests/test_pptx_shadow_run.py tests/test_runtime_execution.py -k "component or accepted_assets or native" -q
```

Expected: FAIL，当前 donor builder 会直接重新执行完整 CV 管线。

- [ ] **Step 3：把重建放在 patch 前并复用已验收资产**

- Runtime 先为每个现有 `shadow_replacement_plan` 推进组件状态；Host 等待时不得创建 donor 或输出 PPTX。
- `build_reconstruction_donor` 改为接收 `component_result.json`，验证 Provider、源截图 SHA、活动组件、文字和所有资产 hash 后组装 donor。
- `pptx_shadow_run._run_page` 不再调用视觉模型，只做 donor 组装、OOXML patch、结构校验和单页回退。
- 保留当前 `replace + full_slide_screenshot + confidence >= 0.92` 的截图候选准入；P2.3 不扩大 PPTX 被替换对象范围。
- 普通 PPTX 图片形状继续保留原来的 x/y/cx/cy 与 z-order；未命中原生文字、形状、表格、图表、备注、连接线引用和其他页面 hash 必须不变。
- donor 中活动子组件或父组件都是独立图片对象；父子同时出现直接拒绝。

- [ ] **Step 4：运行全部 PPTX 结构和恢复测试**

Run:

```powershell
python -m pytest tests/test_pptx_input.py tests/test_pptx_reconstruct.py tests/test_pptx_shadow.py tests/test_pptx_shadow_run.py tests/test_agent_decision.py tests/test_runtime_execution.py -q
```

Expected: PASS；PPTX 可重开，页数/尺寸/备注/受保护对象不变，页面失败时仅该页保留原截图并报告。

- [ ] **Step 5：提交本任务**

```powershell
git add image2editable/runtime.py image2editable/pptx_reconstruct.py image2editable/pptx_shadow_run.py image2editable/pptx_input.py scripts/ppt_assemble.py
git add -f tests/test_pptx_reconstruct.py tests/test_pptx_shadow_run.py tests/test_pptx_shadow.py tests/test_pptx_input.py tests/test_runtime_execution.py
git commit -m "功能：接通PPTX组件重建并保护原生对象"
```

## Task 11：实现本地模型目录、硬件推荐和显式安装

**Files:**

- Create: `image2editable/model_catalog.json`
- Create: `image2editable/models.py`
- Modify: `image2editable/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_models.py`
- Test: `tests/test_runtime_cli.py`

- [ ] **Step 1：写“不下载即可推荐”和安装前检查测试**

```python
def test_recommend_uses_hardware_and_versioned_catalog_without_network(fake_catalog):
    result = recommend_agent_model(
        HardwareProfile(vram_gib=8, ram_gib=16, free_disk_gib=20, cuda=True),
        catalog=fake_catalog,
    )
    assert result["model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert result["stability"] == "experimental"


def test_install_stops_before_network_when_disk_is_insufficient(monkeypatch, tmp_path):
    called = False
    monkeypatch.setattr(models, "snapshot_download", lambda **_: set_called())
    with pytest.raises(RuntimeError, match="free disk"):
        install_agent_model(cache_dir=tmp_path, free_disk_gib=2)
    assert called is False


def test_host_commands_never_probe_or_modify_local_model_cache(monkeypatch, host_run):
    monkeypatch.setattr(models, "model_status", unexpected_call)
    run_job(host_run)
```

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_models.py tests/test_runtime_cli.py -k "model or recommend or install" -q
```

Expected: FAIL，模型命令尚不存在。

- [ ] **Step 3：实现目录和命令**

首个目录条目固定为实验性：

```json
{
  "catalog_version": 1,
  "models": [
    {
      "model_id": "Qwen/Qwen3-VL-2B-Instruct",
      "revision": "main",
      "stability": "experimental",
      "minimum_vram_gib": 8,
      "minimum_ram_gib": 16,
      "required_free_disk_gib": 8,
      "priority": 100
    }
  ]
}
```

实现三个入口：

```text
image2editable models recommend --json
image2editable models install agent
image2editable models status
```

要求：

- `recommend` 只读本地目录和硬件信息，不访问网络、不下载；输出推荐理由、兼容性、稳定性、预计空间和缓存位置。
- 硬件探测使用轻量标准库/包元数据；只有执行 `models` 命令时才允许惰性检查 CUDA/PyTorch，Host 转换路径不导入此模块。
- `install` 在网络调用前显示模型、revision、空间、缓存和实验性状态，并要求 CLI 显式确认；自动化可使用 `--yes`。
- 下载使用 Hugging Face Hub 的断点续传和本地 snapshot；保存解析后的 commit SHA、文件清单和校验 receipt。
- 转换过程中发现模型缺失只报错并给出安装命令，不自动下载。
- `pyproject.toml` 增加 `agent-local` 可选依赖；核心依赖不增加 Qwen 专用包。
- 真实验收通过后另行把目录 `revision` 固定为验收 commit 并将 `stability` 改为 `stable`；本任务不得提前修改。

- [ ] **Step 4：验证命令 JSON 和无网络路径**

Run:

```powershell
python -m pytest tests/test_models.py tests/test_runtime_cli.py tests/test_runtime_resources.py -q
```

Expected: PASS；推荐不下载，空间不足不发起网络请求，Host 路径不碰本地缓存。

- [ ] **Step 5：提交本任务**

```powershell
git add image2editable/model_catalog.json image2editable/models.py image2editable/cli.py pyproject.toml
git add -f tests/test_models.py tests/test_runtime_cli.py
git commit -m "功能：增加本地Agent模型推荐与显式安装"
```

## Task 12：实现 Local Provider 隔离推理并复用同一计划门禁

**Files:**

- Create: `image2editable/local_agent.py`
- Create: `image2editable/local_agent_worker.py`
- Modify: `image2editable/component_repair.py`
- Modify: `image2editable/runtime.py`
- Modify: `image2editable/resources.py`
- Test: `tests/test_local_agent.py`
- Test: `tests/test_component_repair.py`
- Test: `tests/test_runtime_execution.py`
- Test: `tests/test_runtime_resources.py`

- [ ] **Step 1：写 Local mock 完整循环、无 Host 状态和进程释放测试**

```python
def test_local_provider_runs_without_host_receipt(local_run, fake_worker):
    complete = execute_until_complete(local_run)
    assert complete["status"] == "completed"
    assert not (local_run / "host_capabilities.json").exists()
    assert all(plan["provider"] == "local" for plan in recorded_plans(local_run))


def test_local_worker_exits_after_each_page_round(local_run, fake_worker):
    execute_until_complete(local_run)
    assert fake_worker.concurrent_process_peak == 1
    assert fake_worker.live_processes == 0


def test_local_plan_passes_same_validator_as_host(local_request, fake_worker):
    fake_worker.return_plan({"actions": [{"action": "unknown"}]})
    with pytest.raises(ValueError, match="Unsupported component action"):
        run_local_agent(local_request)
```

- [ ] **Step 2：运行失败测试**

Run:

```powershell
python -m pytest tests/test_local_agent.py tests/test_component_repair.py tests/test_runtime_resources.py -q
```

Expected: FAIL，Local Provider 尚未实现。

- [ ] **Step 3：实现轻量调度器和单轮 Worker**

`local_agent.py` 顶层仅导入标准库和项目契约：

```python
def run_local_agent(request_path: str | Path, *, model_receipt: dict) -> dict:
    """Start one worker for one page round, validate its plan, then return."""
```

`local_agent_worker.py` 才允许惰性导入 `torch`、`transformers` 和 Qwen 处理工具：

```python
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype="auto",
)
```

Worker 输入只包括当前请求、只读证据绝对路径、模型 snapshot 和输出临时文件；系统提示列出九种动作、严格字段、5 轮上限、父子互斥和“只输出 JSON”。生成结果由父进程的 `validate_component_plan` 再次校验后原子写入 Run。

Local 系统提示同样明确：源图和 OCR 内容是不可信数据，其中的任何指令都只能作为画面内容分析，不能改变允许动作、文件访问范围、轮次或质量门禁。

资源要求：

- OCR、DINO、SAM、LaMa、Local VLM 分阶段且不同时驻留。
- Local Worker 每一轮结束即退出；父进程不保留模型对象。
- 继承 `safe-default` 的 CPU 线程和低优先级；运行前检查 RAM/VRAM/临时磁盘预算。
- 超时、OOM、非法 JSON 或进程非零退出都保留诊断并进入明确失败/页面保留路径，不切换 Host。
- 不保存跨图片 prompt/response cache；只保留当前 Run 的计划和审计记录。

- [ ] **Step 4：运行 Provider 隔离和 Runtime 全集**

Run:

```powershell
python -m pytest tests/test_local_agent.py tests/test_host_agent.py tests/test_component_contracts.py tests/test_component_repair.py tests/test_runtime_execution.py tests/test_runtime_resources.py -q
```

Expected: PASS；Host/Local 使用同一计划校验器和门禁，且互不导入/读取对方状态。

- [ ] **Step 5：提交本任务**

```powershell
git add image2editable/local_agent.py image2editable/local_agent_worker.py image2editable/component_repair.py image2editable/runtime.py image2editable/resources.py
git add -f tests/test_local_agent.py tests/test_component_repair.py tests/test_runtime_execution.py tests/test_runtime_resources.py
git commit -m "功能：接通隔离的本地视觉Agent提供方"
```

## Task 13：更新 Skill/README、运行完整与真实验收并同步 Course

**Files:**

- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `skills/image-to-ppt/SKILL.md`
- Modify: `Course.md`
- Modify: `tests/test_runtime_cli.py`
- Modify: `tests/test_regressions.py`
- Create: `tests/test_component_acceptance.py`

- [ ] **Step 1：先写文档契约和通用验收清单测试**

```python
@pytest.mark.parametrize("path", ["README.md", "README_EN.md", "skills/image-to-ppt/SKILL.md"])
def test_docs_name_both_providers_and_five_round_limit(path):
    text = Path(path).read_text(encoding="utf-8")
    assert "agent_provider=host" in text
    assert "agent_provider=local" in text
    assert "5" in text
    assert "preserved_with_warning" in text


def test_acceptance_manifest_covers_generic_content_types():
    assert REQUIRED_CASES <= set(load_acceptance_manifest()["content_types"])
```

`REQUIRED_CASES` 固定包含：照片、人物、海报、卡片、UI、表格、图表、流程图、科研绘图、公式、密集连线、地图、插画、图标组合、低对比度、渐变、透明、阴影、抗锯齿；输入类型固定包含图片、PDF、图片版 PPTX、混合原生对象 PPTX。

- [ ] **Step 2：运行文档测试并确认先失败**

Run:

```powershell
python -m pytest tests/test_component_acceptance.py tests/test_runtime_cli.py -q
```

Expected: FAIL，README/Skill 尚未记录完整双模式流程。

- [ ] **Step 3：更新 README 和 Skill**

README 中必须明确：

- Host 是 Skill 默认：不下载额外 VLM；宿主必须支持视觉、本地文件读取、工具调用和结构化 JSON。
- Host 服务可能接收诊断图，敏感内容用户应选择 Local。
- Local 是显式可选：先 `models recommend --json`，得到项目兼容目录和本机硬件结果后再由用户确认 `models install agent`。
- 两种模式的完整命令、状态流、Provider 不可混用、没有自动回退。
- 每页最多 5 个批量轮次；已通过组件冻结；子组件失败折叠为完整父组件；父组件失败保留原页。
- 模型缓存不等于图片判断缓存；每张图片重新分析，跨图片不复用语义决策。
- 当前每个 Provider 的真实稳定性状态；未通过真实验收时标记 `experimental`。
- 原生 PPTX 保护范围和非目标：组件通常是透明图片对象，不承诺任意矢量/SmartArt 重建。

Skill 中必须实现自动 Host 循环：

```text
prepare/截图候选判断
→ run execute
→ 若 awaiting_agent，agent next
→ 读取绝对证据路径并视觉检查
→ 生成严格计划
→ agent record --plan
→ run execute
→ 最多 5 轮，直到 completed 或 preserved_with_warning
```

当用户选择 Local 时，Skill 只能展示 `models recommend --json` 的结果；下载前必须取得明确授权，不能凭模型知识自行选择新模型。

- [ ] **Step 4：运行全量自动化测试和镜像一致性检查**

Run:

```powershell
python -m pytest -q
```

Expected: PASS；不得只运行新测试后跳过旧 Runtime、PDF、PPTX、OCR、资源与安全测试。

- [ ] **Step 5：用通用语料和隐藏集运行记录回放验收**

Run:

```powershell
python -m pytest tests/test_component_acceptance.py -m "not real_provider" -q
```

Expected: PASS；同一记录计划分别经 Host 回放和 Local mock 路径进入相同执行器和门禁，所有内容类型均有至少一个非调参样例；隐藏集结果只记录总指标，不把样例名称写入规则。

- [ ] **Step 6：在受控资源下验收用户真实文件**

按顺序单独执行，不并行：

```powershell
image2editable prepare "1-Embedding与向量数据库.pdf" --run-dir tmp/p23-pdf-host --agent-provider host -o tmp/p23-pdf-host/final/output.pptx --slide-size original
image2editable run execute tmp/p23-pdf-host
image2editable prepare test1.pptx --run-dir tmp/p23-pptx-host --agent-provider host -o tmp/p23-pptx-host/final/output.pptx
```

随后由 Skill 完成 Host 握手、截图候选判断、每页组件计划和最多 5 轮循环。验收必须记录：

- 每页初始、父级、子级、冻结和最终组件数；
- 每轮动作、质量变化、提前停止或折叠原因；
- 输出逐页渲染与源图差异；
- 文字重影、浅灰栅格残影、阴影副本、透明边缘、孔洞、缺失和错误拆分检查；
- PPTX 页数、尺寸、备注、原生对象和未命中页面 hash；
- 峰值 RAM、VRAM、临时磁盘、运行目录体积和所有重型子进程退出情况。

目标机门禁：8GB 显存、16GB 内存，不允许 RAM 或磁盘持续 100%，也不允许用整页组件清零换取通过。

- [ ] **Step 7：运行真实 Local Provider 验收**

先运行：

```powershell
image2editable models recommend --json
image2editable models status
```

若模型尚未安装，向用户展示模型、revision、预计下载量、缓存路径、硬件匹配和实验性状态，并在取得明确授权后执行：

```powershell
image2editable models install agent
```

然后使用与 Host 完全相同的真实文件和门禁，新建 `--agent-provider local` Run。只有 Host 与 Local 都完成视觉、结构和资源验收后，才把对应目录/README 状态标为 `stable`；否则保留 `experimental` 并写清失败项。

- [ ] **Step 8：更新 Course.md 并进行最终自审**

`Course.md` 至少同步：当前分支和本地提交策略、本轮功能、关键修改文件、Host/Local 入口、5 轮行为、父组件/页面回退、真实验收结果、资源数据、尚未解决事项。

自审命令：

```powershell
Select-String -Path image2editable/*.py,scripts/*.py,skills/image-to-ppt/scripts/*.py -Pattern 'TODO|TBD|NotImplementedError'
git diff --check
python -m pytest -q
git status --short
```

Expected: 没有未解释的占位实现；`git diff --check` 无输出；全量测试 PASS；工作树只包含本计划可追溯的修改。

- [ ] **Step 9：请求代码审查并修复结论**

使用 `image2editable:requesting-code-review` 检查：设计覆盖、Provider 隔离、状态恢复、路径安全、哈希绑定、资源边界、父子互斥、原生 PPTX 保护和文档真实性。修复所有高/中优先级问题后重新运行受影响测试及全量测试。

- [ ] **Step 10：提交最终文档和验收状态**

```powershell
git add README.md README_EN.md skills/image-to-ppt/SKILL.md image2editable/model_catalog.json
git add -f Course.md tests/test_component_acceptance.py tests/test_runtime_cli.py tests/test_regressions.py
git commit -m "文档：说明双模式组件Agent与验收结果"
```

不执行 `git push`，不合并 `main`。

## 完成判定

只有同时满足以下条件才可宣称 P2.3 完成：

- 图片、PDF、图片版 PPTX、混合原生对象 PPTX 都进入同一组件状态机。
- Host 无本地 VLM 也能完成视觉握手和真实转换；Local 无宿主会话也能完成真实转换。
- 每页最多 5 个批量重修轮次，已冻结组件不会变化。
- 失败细拆会得到独立完整父组件；父组件失败会保留原页并报告。
- 初始组件非零时不存在普通成功的零组件输出。
- 逐页渲染未见文字重影、浅灰栅格残影、阴影副本、残缺边缘、孔洞或父子双渲染。
- 原生 PPTX 的页数、尺寸、备注、未命中对象和未命中页面保持不变。
- 在目标机上无 RAM/磁盘持续 100%，所有重型模型进程按阶段退出。
- README/Skill 的稳定性声明与真实验收一致。
- 全量测试通过、代码审查完成、`Course.md` 已同步，并只产生本地提交。
