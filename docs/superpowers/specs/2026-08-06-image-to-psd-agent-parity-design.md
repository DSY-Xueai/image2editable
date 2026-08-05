# Image-to-PSD Agent 能力同步设计

## 目标

让 `image-to-psd` 在仅接收图片的前提下，复用当前 `image2editable` 的 OCR、组件候选、Host/Local Agent、每页最多 5 个重修批次和确定性质量门，最终输出分层 PSD。

不增加 PDF 或 PPTX 输入，不改变 PPTX 输出行为，也不复制一套独立的 Agent 判断逻辑。

## 架构

共享管线负责生成通过质量门的最终页面层：

1. OCR 生成可编辑文字和精炼文字清理蒙版。
2. CV/SAM 生成背景及视觉组件候选。
3. Host 或 Local Agent 按现有契约决定拆分、合并、抑制误 OCR 和背景重建。
4. 确定性质量门检查文字重影、组件缺损、重复像素、背景残影和组件独立性。
5. 根据输出类型选择 PPTX 或 PSD 装配器。

PSD 装配器只消费最终背景、最终组件及最终文字，不参与组件判断。

## 接口与行为

- `image-to-psd` 只接受现有图片扩展名；单图输出一个 PSD，批量输入每图输出一个 PSD。
- PSD skill 提供 Host 和 Local 两种 Agent 使用说明，模型选择、离线规则、最多 5 批重修和证据检查规则与 PPT skill 相同。
- PSD 包含一个背景图层、按 z-order 排列的透明视觉组件图层和可编辑文字图层。
- 视觉组件不得携带 OCR 文字像素；可靠文字必须且只能由可编辑文字图层贡献一次。
- 转换开始前检查 Aspose.PSD 授权。授权缺失或无效时立即失败，不启动 OCR、SAM 或 Agent。
- 页面未通过质量门时不输出伪成功 PSD，保留诊断目录并返回明确错误。
- 旧参数可保留 CLI 兼容性，但不允许其绕过统一质量门。

## 代码边界

- 共享 Runtime 增加最小的输出类型分流和 PSD 最终装配调用。
- PSD 装配逻辑保留在独立模块，仅负责把已确认层写入 Aspose.PSD。
- `skills/image-to-psd/` 不再携带旧版传统 CV 管线副本；入口调用共享 Runtime，skill 文档说明完整依赖和执行流程。
- `skills/image-to-ppt/` 行为保持不变。

## 验证

- 单元测试覆盖输出类型校验、授权前置失败、最终层到 PSD 图层的映射和批量输出路径。
- 回归测试确认 PPTX 默认路径不变，Host/Local 的五批上限及质量门不因 PSD 输出而放宽。
- 在可用 Aspose.PSD 授权的环境中执行真实 PSD 打开验证；无授权环境使用受控替身验证装配调用，不伪造真实授权验收结论。
- 检查 PSD skill 不再引用旧版 `build_background`、`extract_foreground_mask` 和 `split_components` 管线。
