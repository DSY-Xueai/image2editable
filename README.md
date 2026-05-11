<div align="center">

# any2ppt

**图片 → 可编辑 PowerPoint，一键还原分层结构**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

</div>

> 拿到一张 PPT 截图或设计稿，想还原成可编辑的 PowerPoint？手动重做太慢，OCR 只能提文字。
> **any2ppt** 通过计算机视觉 + OCR，自动将图片拆解为背景、前景组件、文本三层，生成真正可编辑的 PPTX。

---

## 效果演示

```
┌─────────────┐         ┌──────────────────────────────────────────┐
│             │         │  可编辑 PPTX                              │
│  输入图片    │  ────►  │  ├── 背景图层（修复后的完整背景）          │
│  (截图/设计稿)│         │  ├── 前景组件层（独立可移动的透明 PNG）    │
│             │         │  └── 文本框层（可编辑文字，保留样式）       │
└─────────────┘         └──────────────────────────────────────────┘
```

### 输入 vs 输出

>输入图片 | 也可输入多张
<img width="2154" height="1127" alt="image" src="https://github.com/user-attachments/assets/867e95ba-a7ba-4966-8fd4-a3208a5fc924" />


> 输出的 PPTX 中，每个前景元素都是独立图层，可自由拖动、缩放、删除；文本框可直接编辑文字内容。
<img width="2022" height="1058" alt="image" src="https://github.com/user-attachments/assets/cf86c0dc-515e-4d86-a6fb-a42f084518fd" />

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **智能背景修复** | 两轮迭代建模：平滑背景初检 + 原图 inpainting 精修，非前景区域像素级保留 |
| **前景组件拆分** | 差分 + Canny 边缘检测 + 连通域分析，每个元素输出为独立透明 PNG |
| **OCR 文本重建** | PaddleOCR / Tesseract 双引擎，自动估计字号、颜色、粗体、对齐方式 |
| **分层 PPTX 组装** | 背景层 + 前景组件层 + 文本框层，完全可编辑 |
| **批量处理** | 多张图片或整个目录一次性转换为多页 PPTX |
| **Skill 包** | 可作为独立 skill 分发，别人拿到即可直接使用 |

---

## 快速开始

### 环境要求

- Python 3.9+
- OCR 引擎至少安装一个（见下方）

### 安装

```bash
git clone https://github.com/DSY-Xueai/any2ppt.git
cd any2ppt
pip install -r requirements.txt
```

### OCR 引擎

**方式 A：PaddleOCR（推荐，中文识别精度更高）**

```bash
pip install paddleocr paddlepaddle
```

**方式 B：Tesseract（更轻量）**

```bash
pip install pytesseract
# 系统安装 Tesseract：
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# macOS:   brew install tesseract
# Ubuntu:  sudo apt install tesseract-ocr tesseract-ocr-chi-sim
```

---

## 使用方法

### 命令行

```bash
# 单张图片
python image_to_ppt.py input.png

# 指定输出
python image_to_ppt.py input.png -o output.pptx

# 多张图片 → 一个多页 PPTX
python image_to_ppt.py img1.png img2.png img3.png -o slides.pptx

# 传入目录（自动扫描所有图片）
python image_to_ppt.py ./my_slides/ -o presentation.pptx

# 调整参数
python image_to_ppt.py input.png --lang en --diff-threshold 15 --min-area 30
```

### Python API

```python
from image_to_ppt import convert, convert_batch

# 单张图片
result = convert("input.png", output_path="output.pptx")

# 多张图片
result = convert_batch(
    ["img1.png", "img2.png", "img3.png"],
    output_path="slides.pptx",
)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `images` | （必填） | 图片文件、多个文件、或目录路径 |
| `-o, --output` | 首张图片同名.pptx | 输出路径 |
| `--lang` | `ch` | OCR 语言（ch / en） |
| `--period` | `32` | 背景建模周期 |
| `--diff-threshold` | `20.0` | 前景检测灵敏度（越小越敏感） |
| `--min-area` | `20` | 最小组件面积（像素） |
| `--reference` | 不启用 | 每页后附加原图参考 slide |

---

## 工作原理

```
输入图片
  │
  ▼
┌─────────────────┐
│  OCR 文本检测    │  PaddleOCR / Tesseract
│  字号·颜色·粗体  │  自动样式估计
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  背景建模（两轮） │  第一轮：平滑背景 → 初始前景检测
│                  │  第二轮：原图 + inpainting 精修
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  前景提取与拆分   │  diff + Canny 边缘 + 连通域分析
│                  │  每个组件 → 独立透明 PNG
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PPTX 分层组装   │  背景层 + 组件层 + 文本框层
└────────┬────────┘
         │
         ▼
    输出 .pptx
```

---

## 项目结构

```
any2ppt/
├── image_to_ppt.py        # 主入口（CLI + 管线编排）
├── text_detect.py         # 文本检测（OCR + 样式估计）
├── bg_model.py            # 背景建模（平滑初检 + inpainting 精修）
├── fg_extract.py          # 前景提取（差分 + 边缘检测 + 连通域拆分）
├── ppt_assemble.py        # PPTX 组装（分层构建）
├── requirements.txt       # 依赖清单
├── test-image/            # 测试图片
└── skills/
    └── image-to-ppt/      # 可分发的 Skill 包
        ├── SKILL.md
        ├── scripts/       # 完整源码副本
        └── references/    # 依赖说明
```

---

## Skill 包

项目已封装为独立可分发的 Skill，位于 `skills/image-to-ppt/`。

---

## 技术栈

| 领域 | 技术 |
|------|------|
| 图像处理 | OpenCV, Pillow, NumPy |
| OCR | PaddleOCR, Tesseract |
| PPT 生成 | python-pptx |
| 背景修复 | OpenCV Inpainting (Telea / NS) |
| 前景检测 | 差分阈值 + Canny 边缘 + 形态学运算 |

---

## 已知限制

- 复杂纹理/渐变背景的修复质量有待提升
- 文本颜色基于采样估计，极端配色场景可能偏差
- 目前主要针对中英文优化，其他语言需额外测试
- 多图模式下 slide 尺寸以首张图片宽高比为基准

---

## 支持的图片格式

PNG · JPG / JPEG · BMP · TIFF / TIF · WebP

---

## Star History

<a href="https://www.star-history.com/#DSY-Xueai/any2ppt&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DSY-Xueai/any2ppt&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DSY-Xueai/any2ppt&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=DSY-Xueai/any2ppt&type=Date" />
  </picture>
</a>
