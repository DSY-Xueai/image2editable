---
name: image-to-ppt
description: 将一张或多张图片转换为可编辑的分层 PowerPoint 演示文稿。自动完成 OCR 文本识别、背景建模修复、前景组件拆分和 PPTX 分层组装。当用户需要把截图、设计稿、幻灯片图片还原为可编辑 PPT 时使用此 skill。适用场景包括：图片转 PPT、截图还原为可编辑演示文稿、从图片中提取文字和元素重建 PowerPoint、批量图片生成多页 PPT。
---

# Image to PPT

将图片转换为分层可编辑的 PowerPoint 文件。每张图片生成一页 slide，包含三层：背景图层、前景组件图层（独立可移动）、可编辑文本框图层。

使用 SAM 2.1 分割视觉元素。首次运行时将 large checkpoint 下载到用户本地缓存；自动使用可用的 CUDA，CPU 推理较慢。严格视觉质量校验失败时停止并保留 diagnostics，不要将页面 flatten 为单张图片。

## 工作原理

管线分四步执行：

1. **OCR 文本检测**（`text_detect.py`）— 使用 PaddleOCR（优先）或 pytesseract 识别图片中的文字，同时估计字号、颜色、粗体、对齐方式
2. **背景建模与修复**（`bg_model.py`）— 两轮处理：第一轮用平滑背景做初始前景检测，第二轮用原图 + inpainting 精修前景/文字区域，非前景区域像素级保留原图
3. **前景提取与组件拆分**（`fg_extract.py`）— 基于 diff + Canny 边缘检测 + 连通域分析，每个组件输出为独立透明 PNG
4. **PPTX 分层组装**（`ppt_assemble.py`）— 将背景、前景组件、文本框按层级组装为 PowerPoint 文件

## 环境准备

### 安装依赖

```bash
pip install -r references/requirements.txt
```

核心依赖：
- `python-pptx` — PPTX 生成
- `opencv-python` — 图像处理
- `Pillow` — 图像 I/O
- `numpy` — 数值计算

OCR 引擎（至少安装一个）：
- `paddleocr` + `paddlepaddle`（推荐，精度更高）
- `pytesseract` + Tesseract 引擎（回退方案）

## 使用方式

### 命令行

所有脚本位于 `scripts/` 目录下。

```bash
# 单张图片转换
python scripts/image_to_ppt.py input.png

# 指定输出路径
python scripts/image_to_ppt.py input.png -o output.pptx

# 多张图片合并为一个多页 PPTX
python scripts/image_to_ppt.py img1.png img2.png img3.png -o slides.pptx

# 传入目录（自动扫描目录下所有图片文件）
python scripts/image_to_ppt.py ./my_slides/ -o presentation.pptx

# 调整参数
python scripts/image_to_ppt.py input.png --lang en --diff-threshold 15 --min-area 30

# 每页后附加原图参考页
python scripts/image_to_ppt.py img1.png img2.png --reference
```

### Python API

```python
import sys
sys.path.insert(0, "path/to/skills/image-to-ppt/scripts")

from image_to_ppt import convert, convert_batch

# 单张图片
result = convert("input.png", output_path="output.pptx")

# 多张图片 → 一个多页 PPTX
result = convert_batch(
    ["img1.png", "img2.png", "img3.png"],
    output_path="slides.pptx",
)
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| images | 路径 | （必填） | 图片文件、多个图片文件、或包含图片的目录 |
| -o / --output | 路径 | 首张图片同名.pptx | 输出 PPTX 文件路径 |
| --lang | 字符串 | ch | OCR 语言代码（ch=中文, en=英文） |
| --period | 整数 | 32 | 背景建模瓦片周期 |
| --diff-threshold | 浮点数 | 20.0 | 前景检测灵敏度（越小越敏感） |
| --min-area | 整数 | 20 | 最小组件面积（像素），过滤噪点 |
| --reference | 标志 | 不启用 | 启用后每页内容 slide 后附加原图参考 slide |

## 支持的图片格式

PNG, JPG/JPEG, BMP, TIFF/TIF, WebP

## 输出结构

生成的 PPTX 每页 slide 包含三层（从底到顶）：

1. **背景层** — 修复后的完整背景图（全幅覆盖）
2. **前景组件层** — 各个独立的透明 PNG 元素（可自由移动、缩放、删除）
3. **文本框层** — 可编辑的文本框，保留原始字号、颜色、粗体和对齐方式

## 已知限制

- OCR 精度依赖 PaddleOCR 模型，复杂排版或手写体可能有遗漏
- 背景修复对复杂纹理/渐变区域效果有限（使用 OpenCV inpainting）
- 文本颜色基于采样估计，深色背景上的浅色文字精度较高
- 多图模式下 slide 尺寸以第一张图片的宽高比为基准
- Windows 环境下 PaddleOCR 的 OneDNN/mkldnn bug 已内置 patch 处理
