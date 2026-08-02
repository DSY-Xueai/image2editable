# Leaf Components and Text Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 阻止 Agent 把多个可独立移动对象烘焙成一个父图片，并保证最终视觉组件不携带栅格文字或浅灰文字残影。

**Architecture:** 保留现有 OCR/CV/SAM、Host/Local 双 Provider、每页最多五批和页面级降级状态机。Agent 负责叶组件规划，确定性质量门复核 `absorb_into_parent` 的候选来源和最终组件/背景/文字三层隔离；两项新违反都复用现有重修循环，第五批仍失败则仅保留该原页。

**Tech Stack:** Python 3.12+、NumPy、OpenCV、Pillow、python-pptx、pytest、PowerPoint COM（真实 PPTX 验收）

---

## Task 1: 叶组件优先规则与过度合并硬门禁

**Files:**
- Modify: `image2editable/local_agent_worker.py`
- Modify: `image2editable/component_quality.py`
- Modify: `image2editable/component_repair.py`
- Modify: `skills/image-to-ppt/SKILL.md`
- Modify: `skills/image-to-ppt/scripts/component_quality.py`
- Test: `tests/test_local_agent.py`
- Test: `tests/test_component_acceptance.py`
- Test: `tests/test_component_quality.py`
- Test: `tests/test_component_repair.py`

- [ ] **Step 1: 先写失败测试，固定“语义相关不等于合并”**

  在 `tests/test_local_agent.py` 断言 Local `SYSTEM_PROMPT` 包含以下通用约束：

  ```python
  assert "semantic relationship does not justify merging" in SYSTEM_PROMPT
  assert "independently moved" in SYSTEM_PROMPT
  assert "Prefer preserving one complete parent" not in SYSTEM_PROMPT
  ```

  在 `tests/test_component_acceptance.py` 扩展现有文档契约，读取 `skills/image-to-ppt/SKILL.md`，断言 `absorb_into_parent` 只允许同一物理实体的重复掩码、碎边、阴影或分割缺口；语义父级不参与最终像素渲染。

- [ ] **Step 2: 写过度合并门禁的失败测试**

  在 `tests/test_component_quality.py` 增加三个完整的纯合成掩码用例：`test_absorbed_disjoint_leaf_clusters_are_over_merged`、`test_overlapping_duplicate_masks_remain_one_leaf_cluster`、`test_tiny_noise_fragment_does_not_create_a_leaf_cluster`。

  在 `tests/test_component_repair.py` 增加集成用例：请求中的 `absorb_into_parent` 把两个有效、空间独立候选吸进父组件时，质量报告必须给父组件加入 `over_merged_component`，且不能冻结；同一实体的重叠/包含重复掩码仍能通过。

- [ ] **Step 3: 运行定向测试，确认红灯原因准确**

  Run:

  ```powershell
  python -m pytest tests/test_local_agent.py tests/test_component_acceptance.py tests/test_component_quality.py tests/test_component_repair.py -q
  ```

  Expected: 新测试仅因缺少叶组件规则和 `over_merged_component` 门禁失败。

- [ ] **Step 4: 实现最小确定性检测并接入现有质量报告**

  在 `image2editable/component_quality.py` 增加一个纯函数，输入被吸收候选掩码与现有 `PageCalibration`：

  ```python
  def absorbed_leaf_cluster_count(masks, calibration) -> int:
      """忽略自适应小碎片，按显著重叠/包含关系计算独立候选簇。"""
  ```

  规则只看几何证据：目的父掩码不参与聚类；低于 `calibration.min_component_pixels` 的碎片忽略；候选之间具有显著像素重叠或相互包含时归为同一实体簇。有效簇数大于 1 即为硬失败。

  在 `image2editable/component_repair.py::_recompute_quality_artifact` 中读取当前请求绑定的输入 `component-graph.json`、已验证计划及原始掩码，只检查本批 `absorb_into_parent`。把失败目标 ID 传给 `evaluate_component_quality_round()`，最终由 `evaluate_component()` 将 `over_merged_component` 加入 violations。不要增加轮数、旁路验证器或改变冻结语义。

- [ ] **Step 5: 改写 Host/Local Agent 规则**

  外科式修改 `image2editable/local_agent_worker.py::SYSTEM_PROMPT` 与 `skills/image-to-ppt/SKILL.md`：

  - 使用“对象单独移动后是否仍完整”的反事实判断；
  - 卡片、箭头、连线、容器等按可独立移动的最小完整视觉单元处理，不写任何内容类别特例；
  - `absorb_into_parent` 仅合并同一物理实体的重复/缺口证据；
  - 父级只保留分组关系，不作为多个叶对象的合成图片。

  同步 `skills/image-to-ppt/scripts/component_quality.py`，不改无关脚本。

- [ ] **Step 6: 跑定向测试并提交 Task 1**

  Run:

  ```powershell
  python -m pytest tests/test_local_agent.py tests/test_component_acceptance.py tests/test_component_quality.py tests/test_component_repair.py -q
  git diff --check
  ```

  Expected: 全部通过。

  Commit:

  ```powershell
  git add image2editable/local_agent_worker.py image2editable/component_quality.py image2editable/component_repair.py skills/image-to-ppt/SKILL.md skills/image-to-ppt/scripts/component_quality.py tests/test_local_agent.py tests/test_component_acceptance.py tests/test_component_quality.py tests/test_component_repair.py
  git commit -m "修复：阻止组件过度合并"
  ```

## Task 2: 组件/背景/文字隔离门禁与通用验收

**Files:**
- Modify: `image2editable/component_contracts.py`
- Modify: `image2editable/component_quality.py`
- Modify: `image2editable/component_repair.py`
- Modify: `image2editable/legacy.py`
- Modify: `image2editable/local_agent_worker.py`
- Modify: `skills/image-to-ppt/scripts/component_contracts.py`
- Modify: `skills/image-to-ppt/scripts/component_quality.py`
- Modify: `skills/image-to-ppt/SKILL.md`
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `Course.md`
- Test: `tests/test_component_contracts.py`
- Test: `tests/test_component_quality.py`
- Test: `tests/test_component_repair.py`
- Test: `tests/test_runtime_execution.py`
- Test: `tests/test_task10_runtime_e2e.py`
- Test: `tests/test_component_acceptance.py`

- [ ] **Step 1: 写三层隔离与证据契约的失败测试**

  在 `tests/test_component_quality.py` 增加四个不依赖具体语言或颜色的完整合成用例：`test_component_only_view_rejects_raster_text_ink`、`test_background_only_view_rejects_text_residual`、`test_editable_text_over_clean_layers_appears_once`、`test_clean_fill_and_lines_crossing_ocr_box_do_not_false_fail`。

  在 `tests/test_component_contracts.py`、`tests/test_component_repair.py` 和 `tests/test_local_agent.py` 固定新增 `component-isolation.png` 证据；在 `tests/test_runtime_execution.py` 验证第二至第五批也重新生成、重新哈希该证据，不能跨页面复用。

  在 `tests/test_task10_runtime_e2e.py` 增加同一门禁覆盖 PNG、PDF、图片版 PPTX 和获批截图型混合 PPTX 的参数化验收，并验证未命中的原生 PPTX 文字/形状仍按原对象保留。

- [ ] **Step 2: 运行定向测试，确认红灯原因准确**

  Run:

  ```powershell
  python -m pytest tests/test_component_contracts.py tests/test_component_quality.py tests/test_component_repair.py tests/test_runtime_execution.py tests/test_task10_runtime_e2e.py tests/test_component_acceptance.py -q
  ```

  Expected: 新测试仅因缺少隔离证据和 `component_text_residual` 门禁失败。

- [ ] **Step 3: 生成逐组件隔离证据并扩展严格契约**

  在 `image2editable/legacy.py` 复用现有 `text-clean` RGB、完整组件 alpha 和组件 ID，生成一张有编号的 `component-isolation.png` 联系表。每格只显示一个候选组件的透明图层，不叠加 OCR 文字；不新增模型推理。

  将该文件加入 `COMPONENT_EVIDENCE_NAMES`、Local `_IMAGE_EVIDENCE` 与描述表。保持证据路径、哈希、目录边界和每批不可覆盖规则不变；同步 `skills/image-to-ppt/scripts/component_contracts.py`。

- [ ] **Step 4: 实现三层隔离硬门禁**

  在 `image2editable/component_quality.py` 基于已有页面缓存和校准增加最小指标：

  ```python
  "component_text_residual_ratio"
  "background_text_residual_ratio"
  ```

  - 组件视图：在完整组件 alpha 与 OCR 文字邻域内，检测重建像素是否仍匹配源字形；超过自适应像素下限时加入 `component_text_residual`。
  - 背景视图：在不含组件和文字的背景上，检测相对局部底色仍可见的字形连通区；把失败归因到相邻候选或页面级失败。
  - 合成视图：确认可靠 OCR 项均有可编辑文字对象，并且对应字形只由文字层贡献一次。
  - 继续保留现有 `text_ghost` 兼容报告，但它不能代替新的组件资产发布前门禁。

  `component_text_residual` 必须沿用现有失败 ID、下一批请求和五批上限；第五批仍失败时状态为 `preserved_with_warning`，不得发布该页组件 donor。不要加入栅格文字兜底。

- [ ] **Step 5: 更新文档与当前项目状态**

  在 `README.md`、`README_EN.md` 和 `skills/image-to-ppt/SKILL.md` 写清：最小完整视觉单元、视觉组件无文字、所有可靠文字可编辑、每页最多五批、失败仅保留该原页。删除“优先完整父组件”等已过时描述。

  同步 `Course.md` 的当前状态、本轮改动、关键文件、运行入口、真实验收结果与注意事项；删除 `test1.pptx` “3 个完整父组件即成功”的滞后记录。

- [ ] **Step 6: 自动化回归、真实文件验收与资源检查**

  先运行定向与全量测试：

  ```powershell
  python -m pytest tests/test_component_contracts.py tests/test_component_quality.py tests/test_component_repair.py tests/test_runtime_execution.py tests/test_task10_runtime_e2e.py tests/test_component_acceptance.py -q
  python -m pytest -q
  git diff --check
  ```

  再按 `Course.md` 现有真实验收入口依次运行，保持重型页面串行：

  - `wsl和虚拟机对比.png`
  - `research_layout_demo_3pages.pdf`
  - `test1.pptx`
  - `混合.pptx`

  验收必须检查：组件隔离联系表、背景隔离图、最终渲染、可编辑文字对象数量、组件对象数量、`preserved_with_warning` 明细、PowerPoint COM 原生重开，以及混合 PPTX 未命中原生对象不变。`test1.pptx` 的流程图必须输出多个叶组件，不能再以 3 个整块父组件作为成功标准。验收结束确认没有残留 Python/OCR/SAM 子进程，记录峰值内存和临时目录大小。

- [ ] **Step 7: 同步镜像一致性并提交 Task 2**

  Run:

  ```powershell
  python -c "from pathlib import Path; import hashlib; pairs=[('image2editable/component_contracts.py','skills/image-to-ppt/scripts/component_contracts.py'),('image2editable/component_quality.py','skills/image-to-ppt/scripts/component_quality.py')]; assert all(hashlib.sha256(Path(a).read_bytes()).digest()==hashlib.sha256(Path(b).read_bytes()).digest() for a,b in pairs)"
  git status --short
  ```

  Commit:

  ```powershell
  git add image2editable/component_contracts.py image2editable/component_quality.py image2editable/component_repair.py image2editable/legacy.py image2editable/local_agent_worker.py skills/image-to-ppt/scripts/component_contracts.py skills/image-to-ppt/scripts/component_quality.py skills/image-to-ppt/SKILL.md README.md README_EN.md Course.md tests/test_component_contracts.py tests/test_component_quality.py tests/test_component_repair.py tests/test_runtime_execution.py tests/test_task10_runtime_e2e.py tests/test_component_acceptance.py
  git commit -m "修复：隔离组件文字并完成通用验收"
  ```
