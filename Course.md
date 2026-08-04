# 项目接手说明

## 当前状态

- 项目把图片、PDF、图片版 PPTX 和混合 PPTX 转成分层可编辑 PowerPoint；混合 PPTX 中未命中的原生文字、形状、表格、图表、备注和 z-order 保持不变。
- Agent Provider 支持 `host` 与 `local`。两者共用证据、动作和质量门契约；每页最多 5 个重修批次，失败时保留原页并给出 warning，不以清空组件或保留栅格文字冒充成功。
- 输出契约为：文字单独可编辑；视觉组件不携带文字像素；背景不保留文字或视觉残影；Agent 决定组件拆分、合并、重试、独立性和背景重建，传统 OCR/SAM/CV 负责确定性执行。
- 本地模式只使用用户明确安装的模型，转换时不自动下载；Host 模式可使用 Codex、Claude Code 等宿主视觉模型，两种模式互不读取对方状态。

## 本轮变更

- prepared-page 升级到 schema v4，认证保存精炼后的文字清理蒙版；旧 v1/v2/v3 仍可读取。
- presentation 层改用精炼字形蒙版而非 OCR 整框，视觉 ownership 明确排除文字；源文字像素不会被重新带回组件。
- 新增 OCR 前缀/包含关系去重，保留完整文本行，避免同一区域重复可编辑文字。
- 背景和组件去字洞会在低纹理平滑面上使用受质量阈值约束的局部平面重建；复杂纹理继续使用原候选策略，避免星芒、浅灰矩形和字形残影。
- Agent 接受组件时会修复高实心度蒙版的小孔与边缘缺口；在平滑背景上还会恢复与现有蒙版相连的高对比视觉像素，用于补全稀疏线图中的节点，同时不填满线图内部空白。
- 组件补全只吸收与原组件颜色兼容的像素，保留真实透明孔洞和相邻异色对象；定向 OCR 同时校验位置与文本相似度，prepared-page 每次重新生成并认证本轮文字清理蒙版，不复用旧文件。
- 资源回收继续覆盖 OCR、LaMa、SAM、视觉和重修子进程；真实 `test1.pptx` 首次分层峰值约 1.68 GiB，冻结识别结果后的最终重新组装约 32 秒。

## 关键文件

- 入口与调度：`image2editable/cli.py`、`image2editable/runtime.py`、`image2editable/agent.py`
- 输入与原生 PPTX 保留：`image2editable/inputs.py`、`image2editable/pptx_reconstruct.py`
- Agent 重修与质量：`image2editable/legacy.py`、`image2editable/component_contracts.py`、`image2editable/component_repair.py`、`image2editable/component_quality.py`
- OCR、分割和背景：`image_to_ppt.py`、`scripts/visual_segment.py`、`scripts/component_underlay.py`、`scripts/fg_extract.py`、`scripts/worker_resources.py`
- Skill 镜像：`skills/image-to-ppt/SKILL.md`、`skills/image-to-ppt/scripts/`
- 主要回归：`tests/test_component_repair.py`、`tests/test_component_quality.py`、`tests/test_runtime_execution.py`、`tests/test_targeted_ocr.py`、`tests/test_regressions.py`

## 运行入口

```bash
image2editable convert input.png -o output.pptx --agent-provider host
image2editable convert input.pdf -o output.pptx --agent-provider local
image2editable prepare input.pptx --run-dir runs/pptx-job --agent-provider host
image2editable run execute runs/pptx-job
image2editable agent next runs/pptx-job
image2editable agent record runs/pptx-job --plan response.json
```

本地模型检查：

```bash
image2editable models recommend --json
image2editable models status
```

## 验证事实

- 全量自动化：`1651 passed, 20 skipped`。
- `test1.pptx` 双页真实验收输出：`tmp/task13-host-test1-r16/acceptance-final-r11/output.pptx`。
- 第 1 页为 10 个独立视觉组件、1 个底图、35 个可编辑文本框；第 2 页为 13 个独立视觉组件、1 个底图、26 个可编辑文本框。不存在重叠的 OCR 前缀重复文本。
- PowerPoint 渲染和 `slides_test.py` 通过；第 1 页文档图标、关系图节点无缺口/星芒/虚线残影，第 2 页无重复文字或浅灰文字印子。

## 当前注意事项

- 真实重型文件仍应串行运行并监控内存、磁盘和残留 Python/OCR/SAM 进程；禁止在转换过程中自动下载模型。
- 后续仍需用 `research_layout_demo_3pages.pdf`、`混合.pptx` 等继续覆盖科研图、表格和原生对象不变性；不要把针对单一测试文件的坐标或类别规则写入实现。
- `scripts/` 与 `skills/image-to-ppt/scripts/` 的同名运行脚本必须保持一致。
- `tests/test_component_acceptance.py` 是 ignored 的本地历史文件，不要 force-add。
