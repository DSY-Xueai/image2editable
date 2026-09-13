# image2editable v0.3.0

将图片、PDF 和图片版 PPTX 转换为分层可编辑的 PowerPoint，保留混合 PPTX 中已有的原生对象。

- 按页面内容选择处理路径，复用已验证的 OCR、分割和背景缓存，减少重复推理。
- 改进漏字恢复、可变字体粗细、艺术字字符样式，以及背景残影和独立连接线的处理。
- 质量检查未通过时执行针对性修复，检测重复状态和处理停滞，保留有效结果用于继续转换。
- 同步图片转 PPT 和图片转 PSD Skills 的脚本与运行说明。

支持 Python 3.10–3.12。安装与模型准备见 [使用说明](https://github.com/DSY-Xueai/image2editable#readme)。原生 PDF 的最终渲染校验需要 PowerPoint 或 LibreOffice。实际耗时取决于页面内容、页数和硬件配置。

附件提供安装包、验收报告和 SHA-256 校验文件。
