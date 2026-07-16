<div align="center">

# image2editable

**图片 → 可编辑 PPTX / 分层 PSD**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

</div>

输入 PPT 截图、页面截图或设计稿图片，自动拆成背景、前景组件和文本层，并导出为可编辑 PPTX 或分层 PSD。

---

## 效果演示

```
输入图片
  ├── 背景修复
  ├── 前景组件拆分
  └── OCR 文本重建
        ├── 输出 .pptx：背景层 + 前景图片层 + 可编辑文本框
        └── 输出 .psd：背景层 + 前景像素层 + Photoshop 文本图层
```

> 输入图片 | 也可输入多张
<img width="2154" height="1127" alt="image" src="https://github.com/user-attachments/assets/867e95ba-a7ba-4966-8fd4-a3208a5fc924" />

> PPTX 输出中，前景元素可移动，文本框可编辑。
<img width="2022" height="1058" alt="image" src="https://github.com/user-attachments/assets/cf86c0dc-515e-4d86-a6fb-a42f084518fd" />

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 背景修复 | 为 PPTX 生成 clean background；PSD 使用两轮背景建模与 inpainting 修复 |
| 前景拆分 | PPTX 使用 SAM 2.1 多尺度候选与 OpenCV 几何补漏；PSD 使用差分、边缘和连通域 |
| OCR 文本重建 | 识别文本并估计字号、颜色、粗体、对齐方式 |
| PPTX 导出 | 生成可编辑 PowerPoint：背景层、前景组件层、文本框层 |
| PSD 导出 | 生成分层 PSD：背景层、前景像素层、Photoshop 文本图层 |
| 批量处理 | 多张图片或目录输入；PPTX 合并为多页，PSD 每图一个文件 |

---

## 快速开始

### 环境要求

- Python 3.10+
- `torch>=2.5.1`、`torchvision>=0.20.1`
- SAM 官方推荐 Linux/WSL；Windows 建议使用 WSL
- OCR 引擎至少安装一个
- PSD 导出需要 Aspose.PSD 授权，并设置 `ASPOSE_PSD_LICENSE`

### 安装

```bash
git clone https://github.com/DSY-Xueai/image2editable.git
cd image2editable
pip install -r requirements.txt
```

### SAM 2.1 视觉分割

PPTX 前景视觉元素通过 SAM 2.1 官方 Python API 和 large checkpoint 分割。首次运行会将 checkpoint 下载到用户本地缓存；仓库不包含模型权重。程序会自动使用可用的 CUDA，CPU 也可运行但速度较慢。第三方说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，SAM 2 代码与 checkpoint 采用 [Apache License 2.0](third_party/licenses/SAM2-APACHE-2.0.txt)。

### PPTX 严格管线

1. 使用现有 OCR 逻辑检测文字并生成文字遮罩。
2. 使用 SAM 2.1 生成多尺度候选，并用 OpenCV 几何候选补漏。
3. 对候选去重，解析父子关系，为每个像素建立唯一 ownership。
4. 导出不含文字的透明视觉组件，并重建 clean background。
5. 按实际导出的 RGBA 图层重建页面，执行严格视觉质量 QA。
6. 以固定 16:9 画布组装背景、组件和可编辑文本框。

引用 SAM 2：

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

### OCR 引擎

**方式 A：PaddleOCR（中文识别精度更高）**

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

PSD 导出授权：

```bash
# Windows
set ASPOSE_PSD_LICENSE=C:\path\to\Aspose.PSD.lic

# macOS/Linux
export ASPOSE_PSD_LICENSE=/path/to/Aspose.PSD.lic
```

项目代码以 MIT 发布；Aspose.PSD 是第三方商业依赖，受其官方 EULA 和授权约束。SAM 2.1 的来源、缓存和许可信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [Apache License 2.0 副本](third_party/licenses/SAM2-APACHE-2.0.txt)。

---

## 使用方法

### Skill 安装

项目提供两个互相独立的 Skill：

- `skills/image-to-ppt/`：图片转可编辑 PPTX
- `skills/image-to-psd/`：图片转分层 PSD

**方式一：使用 skills CLI**

```bash
npx skills add DSY-Xueai/image2editable --skill image-to-ppt
npx skills add DSY-Xueai/image2editable --skill image-to-psd
```

**方式二：让 Agent 自动安装**

```text
请从 https://github.com/DSY-Xueai/image2editable 安装 image-to-ppt skill。
请从 https://github.com/DSY-Xueai/image2editable 安装 image-to-psd skill。
```

**方式三：Claude Code plugin**

```bash
claude plugin marketplace add https://github.com/DSY-Xueai/image2editable
claude plugin install image2editable@image2editable --scope user
```

**方式四：手动安装**

```bash
git clone https://github.com/DSY-Xueai/image2editable.git
mkdir -p ~/.claude/skills
cp -R image2editable/skills/image-to-ppt ~/.claude/skills/image-to-ppt
cp -R image2editable/skills/image-to-psd ~/.claude/skills/image-to-psd
```

### 命令行

```bash
# 单张图片 → PPTX
python image_to_ppt.py input.png

# 多张图片 → 一个多页 PPTX
python image_to_ppt.py img1.png img2.png img3.png -o slides.pptx

# 传入目录 → 一个多页 PPTX
python image_to_ppt.py ./my_slides/ -o presentation.pptx

# 每页后附加原图参考页
python image_to_ppt.py img1.png img2.png --reference

# 单张图片 → PSD
python image_to_psd.py input.png

# 多张图片 → 每张图片一个 PSD
python image_to_psd.py img1.png img2.png -o psd_output_dir

# 传入目录 → 每张图片一个 PSD
python image_to_psd.py ./my_slides/ -o psd_output_dir

# 调整 PSD 参数
python image_to_psd.py input.png --lang en --diff-threshold 15 --min-area 30
```

### Python API

```python
from image_to_ppt import convert as convert_to_ppt
from image_to_ppt import convert_batch as convert_batch_to_ppt
from image_to_psd import convert as convert_to_psd
from image_to_psd import convert_batch as convert_batch_to_psd

convert_to_ppt("input.png", output_path="output.pptx")
convert_batch_to_ppt(["img1.png", "img2.png"], output_path="slides.pptx")

convert_to_psd("input.png", output_path="output.psd")
convert_batch_to_psd(["img1.png", "img2.png"], output_path="psd_output_dir")
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `images` | （必填） | 图片文件、多个图片文件、或目录路径；目录只扫描第一层图片 |
| `-o, --output` | 输入同名输出 | PPTX 为文件路径；PSD 单图可为文件路径，多图为输出目录 |
| `--lang` | `ch` | OCR 语言，常用 `ch` / `en` |
| `--period` | `32` | PPTX：仅为兼容保留，strict SAM 管线忽略；PSD：背景建模瓦片周期 |
| `--diff-threshold` | `20.0` | PPTX：仅为兼容保留，strict SAM 管线忽略；PSD：前景检测阈值 |
| `--min-area` | `20` | PPTX：仅为兼容保留，strict SAM 管线忽略；PSD：最小组件面积 |
| `--reference` | 不启用 | 仅 PPTX：每页内容后附加原图参考页 |
| `--no-reference` | 默认行为 | 仅 PPTX：显式关闭原图参考页 |

---

## 项目结构

```
image2editable/
├── .claude-plugin/
│   └── plugin.json        # Claude Code plugin 配置，暴露两个独立 skill
├── image_to_ppt.py        # 图片转 PPTX 入口（CLI + Python API）
├── image_to_psd.py        # 图片转 PSD 入口（CLI + Python API）
├── scripts/               # 核心处理与导出模块
│   ├── text_detect.py     # OCR 文本识别与样式估计
│   ├── bg_model.py        # 背景建模与修复
│   ├── fg_extract.py      # 前景组件提取与拆分
│   ├── ppt_assemble.py    # PPTX 分层组装
│   ├── psd_assemble.py    # PSD 分层组装（Aspose.PSD）
│   └── visual_compare_qa.py # 手动视觉对比 QA 工具
├── skills/                # 可分发 Agent skill
│   ├── image-to-ppt/      # 图片转 PPTX skill
│   └── image-to-psd/      # 图片转 PSD skill
└── requirements.txt       # Python 依赖
```

---

## 技术栈

| 领域 | 技术 |
|------|------|
| 图像处理 | OpenCV, Pillow, NumPy |
| OCR | PaddleOCR, Tesseract |
| PPTX 生成 | python-pptx |
| PSD 生成 | Aspose.PSD |
| 背景修复 | OpenCV Inpainting |
| PPTX 视觉分割 | SAM 2.1 多尺度候选 + OpenCV 几何补漏 + 唯一 ownership |
| PSD 前景检测 | 差分阈值 + Canny 边缘 + 形态学运算 |

---

## 适用场景

- PPT 截图、课程页面、设计稿预览图转可编辑 PPTX
- 截图或设计稿转 Photoshop 分层 PSD
- 背景相对规整、文字清晰的图片效果更好
- 支持中文和英文内容

---

## 支持的图片格式

PNG · JPG / JPEG · BMP · TIFF / TIF · WebP
