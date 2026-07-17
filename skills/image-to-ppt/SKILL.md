---
name: image-to-ppt
description: 将一张或多张图片转换为严格质量校验、分层可编辑的 PowerPoint。用于把截图、设计稿或幻灯片图片重建为原比例与 16:9 PPTX，包含 clean background、无文字且相互独立的透明视觉组件和可编辑文本框。
---

# Image to PPT

把输入图片重建为分层 PPTX。保持严格语义拆分；质量校验失败时停止，不要将整页 flatten 为单张图片。

## 环境

- 使用 Python 3.10 或更高版本。
- 安装 `torch>=2.5.1`、`torchvision>=0.20.1`、Transformers 和 SAM 2.1。运行 `pip install -r references/requirements.txt`。
- 安装 `simple-lama-inpainting==0.1.2`。`LAMA_MODEL` 可指向本地 TorchScript 模型；未设置时，wrapper 首次运行可把模型下载到本地 cache。
- 安装 PaddleOCR 或 Tesseract 作为 OCR 引擎。
- 优先使用 Linux/WSL；SAM 官方建议 Windows 用户使用 WSL。
- 自动使用可用的 CUDA；CPU 也受支持，但推理较慢。

首次运行时把 Grounding DINO tiny、SAM 2.1 large 和默认 LaMa 模型下载到用户本地 cache。源码和权重不存放在此 skill 中。大/深遮罩需要 LaMa；依赖缺失或初始化失败时明确失败，不降级到容易产生条带拖影的 OpenCV 修复。

## 命令行

从 skill 根目录执行 module，不要直接运行 `scripts/image_to_ppt.py`：

```bash
cd skills/image-to-ppt
python -m scripts.image_to_ppt input.png
python -m scripts.image_to_ppt input.png --slide-size original
python -m scripts.image_to_ppt input.png --slide-size 16:9
python -m scripts.image_to_ppt input.png --slide-size both
python -m scripts.image_to_ppt img1.png img2.png -o slides.pptx --slide-size both
python -m scripts.image_to_ppt input.png --lang en --reference
```

CLI 默认 `--slide-size both`。单图输出 `<stem>_original.pptx` 和 `<stem>_16x9.pptx`；批量输出 `<base>_16x9.pptx`，并在 `<base>_original/` 中为每张输入生成原比例单页 PPTX。`--period`、`--diff-threshold` 和 `--min-area` 仅为兼容保留，strict SAM 管线会忽略它们。

## Python API

从 skill 根目录导入：

```python
from scripts.image_to_ppt import (
    convert,
    convert_batch,
    convert_batch_variants,
    convert_variants,
)

convert("input.png", output_path="output.pptx")
convert_variants("input.png")
convert_batch(["img1.png", "img2.png"], output_path="slides.pptx")
convert_batch_variants(["img1.png", "img2.png"], output_path="slides.pptx")
```

旧 `convert()` 保持兼容：默认返回单个 16:9 PPTX 路径字符串；CLI 默认输出两种尺寸。

## 严格管线

1. 使用现有 OCR 逻辑检测文字并生成文字遮罩。
2. 使用 Grounding DINO 生成整图与重叠分块语义候选，再用 SAM 2.1 生成对象掩膜，并以无提示 SAM 候选覆盖词表外对象。
3. 对候选去重，解析父子关系，为每个像素建立唯一 ownership；结合语义支撑和定向 SAM 复查修补组件内部破洞。
4. 导出不含文字的独立透明组件；小/窄遮罩用 OpenCV 修复背景，大/深遮罩用 LaMa。
5. 按实际导出的 RGBA 图层重建页面，执行严格视觉质量 QA。
6. 组装原比例或 16:9 画布；16:9 使用 contain 居中和四角/边缘颜色渐变，不使用模糊放大的原图副本。

## 输出与失败

每页从底到顶包含 clean background、可独立移动的透明组件和可编辑文本框。

命令会在处理每张图片前打印绝对 work directory。严格视觉质量失败时，异常包含 `mae`、`p95` 和 diagnostics 绝对路径；检查其中的 `source.png`、`ownership.png`、`reconstructed.png` 和 `report.json`。更早的分割、OCR 或 LaMa 失败仍可通过已打印的 work directory 定位资产。不要在失败后回退为整页图片。
