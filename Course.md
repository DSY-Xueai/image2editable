# 项目接手说明

## 当前状态

- 项目提供图片、PDF、图片版 PPTX 与混合 PPTX 到分层可编辑 PowerPoint 的统一 Runtime。
- 图片、PDF、图片版 PPTX 和获批的混合 PPTX 整页截图共用组件重建与硬质量门禁；混合 PPTX 中未命中的原生文字、形状、表格、图表、备注、z-order 和图片保持不变。
- Agent Provider 在 Run 创建时冻结为 `host` 或 `local`。两者共享相同的严格请求/证据契约、十二类动作、确定性执行、最多五批修复和质量门禁；Host 不读取本地模型状态，Local 不读取 Host 握手状态。
- 本地模型只在用户明确安装后离线使用；转换过程不下载模型。模型权重可缓存，但图片语义、组件计划和证据不得跨图片、跨页或跨批复用。
- 每页以可独立移动的最小完整视觉单元为组件。已通过组件冻结；五批后完整父组件仍失败时仅该页进入 `preserved_with_warning`，不发布 donor，不允许通过清空组件或保留栅格文字伪装成功。

## 本轮变更

- 严格 evidence 集新增逐轮生成、逐文件 SHA-256 绑定的 `component-isolation.png`。联系表每格只展示一个带编号候选，使用最终完整 alpha 与 text-clean RGB，不叠加 OCR 文字。
- 组件质量新增 `component_text_residual_ratio` 与 `component_text_residual`：在最终会发布的完整组件 alpha 内，匹配源字形的连通残留超过页面自适应像素下限即硬失败。
- 背景质量新增 `background_text_residual_ratio` 与页级 `background_text_clean`：无组件、无文字的背景在 OCR 区仍保留相对局部底色可见的源字形时硬失败；页级检查避免通过清空组件绕过。
- 合成质量新增 `editable_text_once`：存在可靠文字掩码时必须存在可编辑文字对象，且栅格层不得保留第二份字形。已删除 `raster_text_preserved` 成功分支，结果不再以栅格文字兜底。
- 组件文字、背景文字和合成检查复用 `_PageQualityContext` 的页级 RGB、差异、亮度与 text-ink 缓存；逐组件仅处理当前 mask/邻域和连通残留，不重复创建整页 float/blur/distance 数组。
- 保留既有 `text_ghost` 兼容报告、`over_merged_component` 跨轮 sticky、merge/collapse 非活动来源 provenance、紧 bbox 低内存与 TOCTOU 门禁；Agent confidence 不能放宽任何硬失败。

## 关键文件

- 统一入口与调度：`image2editable/cli.py`、`image2editable/runtime.py`、`image2editable/agent.py`
- 输入与 PPTX 原位替换：`image2editable/inputs.py`、`image2editable/pptx_reconstruct.py`
- 组件证据与旧管线适配：`image2editable/legacy.py`
- 组件契约、状态机与质量：`image2editable/component_contracts.py`、`image2editable/component_repair.py`、`image2editable/component_quality.py`
- Agent：`image2editable/host_agent.py`、`image2editable/local_agent.py`、`image2editable/local_agent_worker.py`
- Skill 镜像：`skills/image-to-ppt/SKILL.md`、`skills/image-to-ppt/scripts/component_contracts.py`、`skills/image-to-ppt/scripts/component_quality.py`
- 主要测试：`tests/test_component_contracts.py`、`tests/test_component_quality.py`、`tests/test_component_repair.py`、`tests/test_runtime_execution.py`、`tests/test_task10_runtime_e2e.py`、`tests/test_local_agent.py`

## 运行入口

```bash
image2editable convert input.png -o output.pptx --agent-provider host
image2editable convert input.pdf -o output.pptx --agent-provider local
image2editable prepare input.pptx --run-dir runs/pptx-job --agent-provider host
image2editable run execute runs/pptx-job
image2editable agent next runs/pptx-job
image2editable agent record runs/pptx-job --plan response.json
```

Local 使用前检查：

```bash
image2editable models recommend --json
image2editable models status
```

## 验证事实

- 本轮 TDD 红灯：新增门禁前，证据契约、隔离图参数、两个残影指标与 `editable_text_once` 共 6 项按预期失败；实现后对应定向测试通过。
- 本轮六个目标测试文件最终结果：`481 passed, 9 skipped`；全量结果：`1407 passed, 20 skipped`。
- 真实文件与 PowerPoint COM reopen 尚未在本轮新门禁下完成，不能沿用旧门禁的输出宣称本轮真实验收已通过。
- `wsl和虚拟机对比.png` 已用 `image2editable prepare` 后串行执行 `image2editable run execute`；约 182.4 秒后安全停在第 1 批 `awaiting_plan` Host 决策边界。当前 25 个候选、29 个可靠文字项、0 个冻结组件；`component-isolation.png` 为 960×2160，SHA-256 `7c034d18651d1512ca2e19f19343326dd90aea621c65a29212f426121b1696a1`。运行目录 `tmp/task2-isolation-real-image-r1` 约 25.2 MiB，相关进程总工作集峰值约 2.45 GiB，停止后无残留 Python/OCR/SAM 进程。尚未形成最终渲染或 donor，不能写作验收通过。

## 当前注意事项

- 真实文件必须串行、一次一个文件/重型页面；监控内存和磁盘，结束后确认无残留 Python/OCR/SAM 进程。禁止下载模型。
- 待复验文件：`wsl和虚拟机对比.png`、`research_layout_demo_3pages.pdf`、`test1.pptx`、`混合.pptx`。验收需检查隔离联系表、background-only、最终渲染、可编辑文字、叶组件数、warning、PowerPoint COM reopen，以及混合 PPTX 未命中原生对象不变。
- `test1.pptx` 不得再把“每页 3 个整块父组件”视为成功条件；必须通过最小完整视觉单元和三层文字隔离门禁。
- `tests/test_component_acceptance.py` 是 ignored 的本地历史文件，不得 force-add。
