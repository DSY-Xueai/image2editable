<div align="center">

# image2editable

中文 | [English](README_EN.md)

**图片 → 可编辑 PPTX / 分层 PSD**

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

</div>

输入 PPT 截图、页面截图或设计稿图片，自动拆成背景、前景组件和文本层，并导出为可编辑 PPTX 或分层 PSD。

---

## 效果演示

> 输入图片 | 也可输入多张
<img width="2154" height="1127" alt="image" src="https://github.com/user-attachments/assets/867e95ba-a7ba-4966-8fd4-a3208a5fc924" />

> PPTX 输出中，前景元素可移动，文本框可编辑。
>
> 为获得最佳的 16:9 PPT 视觉效果，建议输入图片采用 16:9 比例。
<img width="2022" height="1058" alt="image" src="https://github.com/user-attachments/assets/cf86c0dc-515e-4d86-a6fb-a42f084518fd" />

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 背景修复 | PPTX 对小/窄遮罩使用 OpenCV、对大/深遮罩使用 LaMa；PSD 使用两轮背景建模与 inpainting 修复 |
| 前景拆分 | PPTX 使用 Grounding DINO 语义候选与 SAM 2.1 分割；PSD 使用差分、边缘和连通域 |
| OCR 文本重建 | 识别文本并估计字号、颜色、粗体、对齐方式 |
| PPTX 导出 | 生成背景、独立透明组件和可编辑文本框，默认同时输出原图比例与 16:9 版本 |
| PSD 导出 | 生成分层 PSD：背景层、前景像素层、Photoshop 文本图层 |
| 批量处理 | 多张图片或目录输入；PPTX 合并为多页，PSD 每图一个文件 |

---

## 快速开始

### 环境要求

- Python 3.10–3.12（上限来自 `simple-lama-inpainting 0.1.2` 的 NumPy/Pillow 依赖约束）
- `torch>=2.5.1`、`torchvision>=0.20.1`、`transformers>=4.40.0`、`simple-lama-inpainting==0.1.2`
- SAM 官方推荐 Linux/WSL；Windows 建议使用 WSL
- OCR 至少配置一条完整路径：`paddleocr` + `paddlepaddle`，或 `pytesseract` + 系统 Tesseract 可执行文件；`doctor` 以此判断环境是否 ready
- PSD 导出额外需要 Aspose.PSD 包及授权，并设置 `ASPOSE_PSD_LICENSE`

### 安装

```bash
git clone https://github.com/DSY-Xueai/image2editable.git
cd image2editable
pip install .

# 需要 PSD 导出时安装可选依赖
pip install .[psd]
```

### 模型与首次运行

PPTX 转换依赖 Grounding DINO、SAM 2.1 和 LaMa。首次运行会自动下载所需模型到本地缓存，本仓库不包含模型权重。运行时会优先使用 CUDA，也支持 CPU；CPU 模式速度会明显慢一些。已有本地 LaMa TorchScript 模型时，可通过 `LAMA_MODEL` 指定模型路径。

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

Windows PowerShell：

```powershell
$env:ASPOSE_PSD_LICENSE = "C:\path\to\Aspose.PSD.lic"
```

macOS/Linux：

```bash
export ASPOSE_PSD_LICENSE=/path/to/Aspose.PSD.lic
```

Aspose.PSD 是商业组件，使用前请确认已获得符合官方 EULA 的授权。其他第三方依赖及许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，论文引用信息见 [CITATION.cff](CITATION.cff)。

---

## 使用方法

### Skill 安装

项目提供两个互相独立的 Skill：

- `skills/image-to-ppt/`：图片转可编辑 PPTX
- `skills/image-to-psd/`：图片转分层 PSD

**方式一：使用 skills CLI**

```bash
npx skills add DSY-Xueai/image2editable --skill <skill_name>
```
把 <skill_name> 换成要安装的 skill 目录名，例如 image-to-ppt。

**方式二：让 Agent 自动安装**

```text
请从 https://github.com/DSY-Xueai/image2editable 安装 <skill_name> skill。
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
cp -R image2editable/skills/image-to-ppt ~/.claude/skills/<skill_name>
```

### 命令行运行

#### 统一 Runtime（P1）

```bash
# 图片或目录继续使用现有可编辑 PPTX 管线
image2editable convert input.png -o output.pptx --slide-size 16:9

# PDF 可直接转换，也可准备任务后按页请求一次细节重渲染再执行
image2editable convert input.pdf -o output.pptx --slide-size 16:9
image2editable prepare input.pdf --run-dir runs/pdf-job
image2editable run render-detail runs/pdf-job --page page_001
image2editable run execute runs/pdf-job

# PPTX 在 P1 中无损保留
image2editable convert input.pptx -o preserved.pptx

# 检查本地依赖
image2editable doctor
```

Unified CLI 的位置参数概念为 `sources`：可输入图片文件/目录、单个 PDF 或单个 PPTX。文档输入不能与其他来源混用，也不能重复传入；原有 `python image_to_ppt.py` 和 `python image_to_psd.py` 图片入口继续兼容。

PDF 会先自适应渲染，再复用现有“图像转可编辑 PPTX”管线。标准目标为 200 DPI；小页会提高到短边至少 1200 px，但不超过 300 DPI；所有渲染均限制长边不超过 6000 px、总像素不超过 24 MP。Agent 或用户可对每页调用一次 `render-detail`，以 300 DPI 目标重新渲染。物理宽高比相同的 PDF 页面可合并为保持该比例的多页 PPTX；混合宽高比时，`original` 输出为逐页文件，同时仍可生成统一 16:9 版本。所有布局均等比放置，不做非均匀拉伸。

PPTX 输入只读扫描 OOXML 原生对象、备注、图片关系和稳定指纹，仅将覆盖幻灯片至少 80% 且结构安全的大图标记为候选。P1 执行输出与输入 PPTX byte-identical，因此已有可编辑文字、形状、表格、图表等原生对象全部保留，不经过 CV，也不会自动拆图或替换。Agent 最终截图分类、OCR/重建与 OOXML 原位替换属于 P2，尚未实现。

正式支持 Python 3.10–3.12；`doctor` 现在也检查 PDFium。

#### 兼容入口

```bash
# 单张图片 → 默认同时生成 input_original.pptx 和 input_16x9.pptx
python image_to_ppt.py input.png

# 只生成一种尺寸
python image_to_ppt.py input.png --slide-size original
python image_to_ppt.py input.png --slide-size 16:9

# 多张图片 → 默认生成 16:9 多页 PPTX，并在 *_original 目录生成原比例单页 PPTX
python image_to_ppt.py img1.png img2.png img3.png -o slides.pptx

# 传入目录 → 同样支持 original、16:9 或 both
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


### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `images` | （必填） | 图片文件、多个图片文件、或目录路径；目录只扫描第一层图片 |
| `-o, --output` | 输入同名输出 | PPTX：`original` / `16:9` 单模式为文件路径，默认 `both` 时为输出基名；PSD 单图可为文件路径，多图为输出目录 |
| `--lang` | `ch` | OCR 语言，常用 `ch` / `en` |
| `--period` | `32` | PPTX：仅为兼容保留、不生效；PSD：背景建模瓦片周期 |
| `--diff-threshold` | `20.0` | PPTX：仅为兼容保留、不生效；PSD：前景检测阈值 |
| `--min-area` | `20` | PPTX：仅为兼容保留、不生效；PSD：最小组件面积 |
| `--reference` | 不启用 | 仅 PPTX：每页内容后附加原图参考页 |
| `--no-reference` | 默认行为 | 仅 PPTX：显式关闭原图参考页 |
| `--slide-size` | `both` | 仅 PPTX：`original` 保持输入比例；`16:9` 输出宽屏；`both` 同时输出两种尺寸 |

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
| 背景修复 | OpenCV Inpainting（小/窄遮罩）+ LaMa（大/深遮罩） |
| PPTX 视觉分割 | Grounding DINO 语义候选 + SAM 2.1 掩膜 + 唯一 ownership |
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

## LICENES

MIT
