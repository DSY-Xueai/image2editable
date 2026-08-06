# 项目接手说明

## 当前状态

- 项目把图片、PDF、图片版 PPTX 和混合 PPTX 转成分层可编辑 PowerPoint，也可把图片转成分层 PSD；混合 PPTX 中未命中的原生文字、形状、表格、图表、备注和 z-order 保持不变。
- Agent Provider 支持 `host` 与 `local`。两者共用证据、动作和质量门契约；每页最多 5 个重修批次，失败时保留原页并给出 warning，不以清空组件或保留栅格文字冒充成功。
- 输出契约为：文字单独可编辑；视觉组件不携带文字像素；背景不保留文字或视觉残影；Agent 决定组件拆分、合并、重试、独立性和背景重建，传统 OCR/SAM/CV 负责确定性执行。
- 本地模式调用用户自行部署的 OpenAI 兼容视觉模型服务；优先读取项目根目录 `.env` 中的 `IMAGE2EDITABLE_LOCAL_BASE_URL`、`IMAGE2EDITABLE_LOCAL_MODEL` 和可选 `IMAGE2EDITABLE_LOCAL_API_KEY`，同名环境变量可临时覆盖。不下载、不推荐也不默认绑定模型。Host 模式可使用 Codex、Claude Code 等宿主视觉模型。

## 本轮变更

- Local Agent 改为请求用户配置的本地 OpenAI 兼容 Chat Completions 服务，并将诊断图以 data URL 发送；不再在 Local 运行路径中探测、下载或加载固定 Hugging Face 模型。新增 `image2editable/local_service.py`。
- `skills/image-to-ppt/SKILL.md` 在 OCR 缺失时要求用户在 PaddleOCR 与 Tesseract 间明确选择；确认后才安装、执行 `doctor` 并继续转换。
- README 中 Local CLI 改为本地服务环境变量配置，并说明 OCR 选择与安装流程；`tests/` 不再被忽略，以便新测试与源码一起上传。
- PSD 不再使用独立的固定阈值 CV 管线；`--format psd` 复用图片转 PPTX 的 Host/Local Agent 决策、OCR、组件 ownership、背景修复和每页最多 5 批的硬质量门。
- PSD 当前只接受图片文件或图片目录。单图输出一个 PSD，多图输出每图一个 PSD；PDF/PPTX 请求会在创建任务前拒绝，Aspose.PSD 包或授权缺失也会提前失败，不留下半成品 run。
- PSD 只组装最终通过质量门的背景、视觉组件和可编辑文字；`preserved_with_warning` 页面不会伪装成成功 PSD。根入口与 `skills/image-to-psd` 兼容入口都转发到统一 Runtime，skill 内旧 OCR/CV 脚本已删除。
- prepared-page 升级到 schema v4，认证保存精炼后的文字清理蒙版；旧 v1/v2/v3 仍可读取。
- presentation 层改用精炼字形蒙版而非 OCR 整框，视觉 ownership 明确排除文字；源文字像素不会被重新带回组件。
- 新增 OCR 前缀/包含关系去重，保留完整文本行，避免同一区域重复可编辑文字。
- 背景和组件去字洞会在低纹理平滑面上使用受质量阈值约束的局部平面重建；复杂纹理继续使用原候选策略，避免星芒、浅灰矩形和字形残影。
- Agent 接受组件时会修复高实心度蒙版的小孔与边缘缺口；在平滑背景上还会恢复与现有蒙版相连的高对比视觉像素，用于补全稀疏线图中的节点，同时不填满线图内部空白。
- 组件补全只吸收与原组件颜色兼容的像素，保留真实透明孔洞和相邻异色对象；定向 OCR 同时校验位置与文本相似度，prepared-page 每次重新生成并认证本轮文字清理蒙版，不复用旧文件。
- 资源回收继续覆盖 OCR、LaMa、SAM、视觉和重修子进程；真实 `test1.pptx` 首次分层峰值约 1.68 GiB，冻结识别结果后的最终重新组装约 32 秒。

## 关键文件

- 入口与调度：`image2editable/cli.py`、`image2editable/runtime.py`、`image2editable/agent.py`、`image2editable/local_service.py`
- 输入与原生 PPTX 保留：`image2editable/inputs.py`、`image2editable/pptx_reconstruct.py`
- Agent 重修与质量：`image2editable/legacy.py`、`image2editable/component_contracts.py`、`image2editable/component_repair.py`、`image2editable/component_quality.py`
- OCR、分割和背景：`image_to_ppt.py`、`scripts/visual_segment.py`、`scripts/component_underlay.py`、`scripts/fg_extract.py`、`scripts/worker_resources.py`
- Skill 镜像：`skills/image-to-ppt/SKILL.md`、`skills/image-to-ppt/scripts/`、`skills/image-to-psd/SKILL.md`
- 主要回归：`tests/test_component_repair.py`、`tests/test_component_quality.py`、`tests/test_runtime_execution.py`、`tests/test_targeted_ocr.py`、`tests/test_regressions.py`

## 运行入口

```bash
image2editable convert input.png -o output.pptx --agent-provider host
image2editable convert input.png -o output.psd --format psd --agent-provider host
$env:IMAGE2EDITABLE_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
$env:IMAGE2EDITABLE_LOCAL_MODEL = "my-local-vision-model"
image2editable convert input.pdf -o output.pptx --agent-provider local
image2editable prepare input.pptx --run-dir runs/pptx-job --agent-provider host
image2editable run execute runs/pptx-job
image2editable agent next runs/pptx-job
image2editable agent record runs/pptx-job --plan response.json
```

## 验证事实

- 本轮全量自动化：`1596 passed, 21 skipped`；PSD skill 官方结构校验通过；Python 3.12 下 wheel 构建通过。
- `test1.pptx` 双页真实验收输出：`tmp/task13-host-test1-r16/acceptance-final-r11/output.pptx`。
- 第 1 页为 10 个独立视觉组件、1 个底图、35 个可编辑文本框；第 2 页为 13 个独立视觉组件、1 个底图、26 个可编辑文本框。不存在重叠的 OCR 前缀重复文本。
- PowerPoint 渲染和 `slides_test.py` 通过；第 1 页文档图标、关系图节点无缺口/星芒/虚线残影，第 2 页无重复文字或浅灰文字印子。

## 当前注意事项

- 本地服务必须支持图像输入与 OpenAI 兼容 Chat Completions；服务模型名由 `IMAGE2EDITABLE_LOCAL_MODEL` 指定，转换期间不会下载模型。
- PSD 生成依赖 `pip install .[psd]` 和有效的 `ASPOSE_PSD_LICENSE`；未配置授权的环境只能完成自动化模拟验证，不能生成真实 PSD。
- 后续仍需用 `research_layout_demo_3pages.pdf`、`混合.pptx` 等继续覆盖科研图、表格和原生对象不变性；不要把针对单一测试文件的坐标或类别规则写入实现。
- `scripts/` 与 `skills/image-to-ppt/scripts/` 的同名运行脚本必须保持一致。
- `tests/test_component_acceptance.py` 是 ignored 的本地历史文件，不要 force-add。
