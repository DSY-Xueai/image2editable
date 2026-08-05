---
name: image-to-psd
description: 将一张或多张图片通过共享的 OCR、SAM、Host/Local 视觉 Agent、最多五批组件重修和严格质量门转换为分层 PSD；输出独立透明视觉组件、修复背景和可编辑 Photoshop 文字图层。用于截图、科研绘图、设计稿或幻灯片图片的 PSD 分层重建；仅支持图片输入，不用于 PDF 或 PPTX。
---

# Image to PSD

使用项目统一 Runtime 完成图片分层判断，只把通过质量门的最终层写入 PSD。不要调用旧版 `build_background`、`extract_foreground_mask` 或 `split_components` 管线。

## 环境

从完整仓库安装 Runtime 和 PSD 依赖：

```bash
pip install -e ".[psd]"
```

PSD 文字图层依赖已授权的 Aspose.PSD。转换前设置：

```powershell
$env:ASPOSE_PSD_LICENSE="C:\path\to\Aspose.PSD.lic"
```

Linux/macOS 使用：

```bash
export ASPOSE_PSD_LICENSE=/path/to/Aspose.PSD.lic
```

授权缺失或无效时必须在创建 Run、加载 OCR/SAM 或调用 Agent 前失败。不得把文字降级为图片图层。

## 输入与输出

- 仅支持图片：PNG、JPEG、BMP、TIFF、WebP。
- 单图输出一个 `.psd`；多图输出到目录，同名源文件自动增加序号。
- 每个 PSD 包含修复背景、按 z-order 排列的透明视觉组件以及可编辑文字图层。
- 不接受 PDF 或 PPTX。需要这些输入时使用 `image-to-ppt`。

## Provider

优先使用当前宿主视觉模型：

```bash
image2editable prepare input.png -o output.psd \
  --run-dir runs/psd-job --format psd --agent-provider host
image2editable run execute runs/psd-job
image2editable agent next runs/psd-job
image2editable agent record runs/psd-job --plan response.json
image2editable run execute runs/psd-job
```

Host 必须支持视觉识别、本地文件读取、工具调用和结构化 JSON。实际查看 `agent next` 返回的全部证据后再记录计划，不从文件名或 metadata 猜测。

只有用户明确要求离线/自托管时才使用 Local：

```bash
image2editable models recommend --json
image2editable models status
image2editable convert input.png -o output.psd \
  --format psd --agent-provider local
```

只使用用户明确安装且 `installed=true`、`valid=true` 的模型；转换期间不自动下载，也不自动切换 Provider。

## 质量契约

- 每页最多 5 个重修批次；已通过组件立即冻结。
- 文字必须且只能由可编辑文字图层贡献一次，视觉组件和背景不得保留文字像素。
- 每个视觉组件应是可独立移动的最小完整单元，不得残缺、重叠、带阴影残片或吸收相邻对象。
- `rebuild_background.margin_ratio` 根据当前证据选择能覆盖残影且不触及相邻结构的最小值，不固定写死。
- 页面最终成为 `preserved_with_warning` 时不生成伪分层 PSD；保留诊断目录并明确报告质量门失败。

兼容入口 `scripts/image_to_psd.py` 只负责把旧命令形式转发给共享 Runtime：

```bash
python scripts/image_to_psd.py input.png -o output.psd --agent-provider local
```
