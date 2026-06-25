---
name: image-to-psd
description: 将一张或多张图片转换为分层 PSD 文件。自动完成 OCR 文本识别、背景建模修复、前景组件拆分，并使用 Aspose.PSD 生成 Photoshop 文本图层。当用户需要把截图、设计稿、幻灯片图片还原为可在 Photoshop 中编辑的分层 PSD 时使用此 skill。
version: 1.0.0
---

# Image to PSD

将图片转换为分层 PSD 文件。每张图片生成一个 PSD，包含三类图层：修复后的背景、独立前景图层、Photoshop 文本图层。

## 环境准备

```bash
pip install -r references/requirements.txt
```

PSD 文本图层导出依赖 Aspose.PSD 授权。配置方式：

```bash
set ASPOSE_PSD_LICENSE=C:\path\to\Aspose.PSD.lic
```

macOS/Linux:

```bash
export ASPOSE_PSD_LICENSE=/path/to/Aspose.PSD.lic
```

## 使用方式

所有脚本位于 `scripts/` 目录下。

```bash
# 单张图片转换
python scripts/image_to_psd.py input.png

# 指定输出 PSD
python scripts/image_to_psd.py input.png -o output.psd

# 多张图片，每张图片输出一个 PSD
python scripts/image_to_psd.py img1.png img2.png -o psd_output_dir

# 传入目录，目录第一层每张图片输出一个 PSD
python scripts/image_to_psd.py ./my_slides/ -o psd_output_dir

# 调整参数
python scripts/image_to_psd.py input.png --lang en --diff-threshold 15 --min-area 30
```

## Python API

```python
import sys
sys.path.insert(0, "path/to/skills/image-to-psd/scripts")

from image_to_psd import convert, convert_batch

result = convert("input.png", output_path="output.psd")

results = convert_batch(
    ["img1.png", "img2.png"],
    output_path="psd_output_dir",
)
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| images | 路径 | （必填） | 图片文件、多个图片文件、或包含图片的目录 |
| -o / --output | 路径 | 单图同名 `.psd`；多图同目录同名 `.psd` | 单图可传 PSD 文件；多图应传输出目录 |
| --lang | 字符串 | ch | OCR 语言代码（ch=中文, en=英文） |
| --period | 整数 | 32 | 背景建模瓦片周期 |
| --diff-threshold | 浮点数 | 20.0 | 前景检测灵敏度（越小越敏感） |
| --min-area | 整数 | 20 | 最小组件面积（像素），过滤噪点 |

## 输出结构

1. `Background` — 修复后的背景图层
2. `Foreground 001...` — 独立前景像素图层
3. `Text 001...` — Photoshop 文本图层

## 注意

- 本 skill 不依赖 `image-to-ppt` skill。
- 项目代码以 MIT 发布；Aspose.PSD 是第三方商业依赖，受其官方 EULA 和授权约束。
- 未配置 `ASPOSE_PSD_LICENSE` 时，脚本会明确失败，不会把文本降级为图片图层。
