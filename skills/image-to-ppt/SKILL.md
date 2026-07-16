---
name: image-to-ppt
description: 将一张或多张图片转换为严格校验、分层可编辑的 PowerPoint。用于把截图、设计稿或幻灯片图片重建为固定 16:9 PPTX，包含 clean background、独立透明视觉组件和可编辑文本框。
---

# Image to PPT

把输入图片重建为分层 PPTX。保持严格语义拆分；质量校验失败时停止，不要将整页 flatten 为单张图片。

## 环境

- 使用 Python 3.10 或更高版本。
- 安装 `torch>=2.5.1`、`torchvision>=0.20.1` 和 SAM 2.1。运行 `pip install -r references/requirements.txt`。
- 安装 PaddleOCR 或 Tesseract 作为 OCR 引擎。
- 优先使用 Linux/WSL；SAM 官方建议 Windows 用户使用 WSL。
- 自动使用可用的 CUDA；CPU 也受支持，但推理较慢。

首次运行时把 SAM 2.1 large checkpoint 下载到用户本地 cache。源码和权重不存放在此 skill 中。

## 命令行

从 skill 根目录执行 module，不要直接运行 `scripts/image_to_ppt.py`：

```bash
cd skills/image-to-ppt
python -m scripts.image_to_ppt input.png
python -m scripts.image_to_ppt img1.png img2.png -o slides.pptx
python -m scripts.image_to_ppt ./my_slides/ -o presentation.pptx
python -m scripts.image_to_ppt input.png --lang en --reference
```

`--period`、`--diff-threshold` 和 `--min-area` 仅为兼容保留，strict SAM 管线会忽略它们。

## Python API

从 skill 根目录导入：

```python
from scripts.image_to_ppt import convert, convert_batch

convert("input.png", output_path="output.pptx")
convert_batch(["img1.png", "img2.png"], output_path="slides.pptx")
```

## 严格管线

1. 使用现有 OCR 逻辑检测文字并生成文字遮罩。
2. 使用 SAM 2.1 生成多尺度候选，并用 OpenCV 几何候选补漏。
3. 对候选去重，解析父子关系，为每个像素建立唯一 ownership。
4. 导出不含文字的透明组件，并重建 clean background。
5. 按实际导出的 RGBA 图层重建页面，执行严格视觉质量 QA。
6. 以固定 16:9 画布组装背景、组件和可编辑文本框。

## 输出与失败

每页从底到顶包含 clean background、可独立移动的透明组件和可编辑文本框。

命令会在处理每张图片前打印绝对 work directory。严格视觉质量失败时，异常包含 `mae`、`p95` 和 diagnostics 绝对路径；检查其中的 `source.png`、`ownership.png`、`reconstructed.png` 和 `report.json`。更早的分割或 OCR 失败仍可通过已打印的 work directory 定位资产。不要在失败后回退为整页图片。
