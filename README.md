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

> 通过硬门禁的 PPTX 输出中，视觉组件可移动且不含文字像素，所有可靠 OCR 文字只由一个原生可编辑文本框贡献；质量不足时仅保留该页原始内容并给出 warning。
>
> 为获得最佳的 16:9 PPT 视觉效果，建议输入图片采用 16:9 比例。
<img width="2022" height="1058" alt="image" src="https://github.com/user-attachments/assets/cf86c0dc-515e-4d86-a6fb-a42f084518fd" />

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 背景修复 | PPTX 与 PSD 共用 Agent 质量管线：小/窄遮罩使用 OpenCV，大/深遮罩使用 LaMa |
| 前景拆分 | PPTX 与 PSD 共用 Grounding DINO 语义候选、SAM 2.1 分割和唯一 ownership |
| OCR 文本重建 | 先做全页识别，再对未被文字遮罩覆盖的小型视觉候选做双视图有界复查；一致高置信结果估计字号、颜色、粗体和对齐方式后回灌为可编辑文字 |
| PPTX 导出 | 通过门禁时生成背景、最小完整透明视觉组件和可编辑文本框；任一页五批后仍失败时仅原样保留该页；默认同时输出原图比例与 16:9 版本 |
| 资源保护 | 重型页面串行，OCR、LaMa、DINO、SAM 和整页视觉阶段使用顺序子进程；SAM2.1 Large 默认单批推理 |
| PSD 导出 | 仅接受图片输入；复用与 PPTX 相同的 Agent 分层和质量门，生成背景、视觉组件及 Photoshop 文本图层 |
| 批量处理 | 多张图片或目录输入；PPTX 合并为多页，PSD 每图一个文件 |
| Agent 截图路由 | 对 PPTX 中覆盖页面至少 80% 且结构安全的大图生成可审计候选；只有 Agent 高置信判定为完整幻灯片截图时才进入 shadow-run 队列 |

---

## 快速开始

### 环境要求

- Python 3.10–3.12（上限来自 `simple-lama-inpainting 0.1.2` 的 NumPy/Pillow 依赖约束）
- `torch>=2.5.1`、`torchvision>=0.20.1`、`transformers>=4.40.0`、`accelerate>=0.26.0`、`simple-lama-inpainting==0.1.2`
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

图片分层转换依赖 Grounding DINO、SAM 2.1 和 LaMa。首次运行会自动下载所需模型到本地缓存，本仓库不包含模型权重。运行时会优先使用 CUDA，也支持 CPU；CPU 模式速度会明显慢一些。已有本地 LaMa TorchScript 模型时，可通过 `LAMA_MODEL` 指定模型路径。

### 资源与质量策略

默认策略限制重型页面并发为 1、数值计算线程最多为 8、SAM `points_per_batch=1`。OCR 检测/识别、LaMa、DINO、SAM prompted/automatic/最终孔洞复检和整页视觉处理按阶段运行在顺序子进程中，以进程退出作为资源释放边界。候选 OCR 每页最多检查 24 个候选，每候选使用最长边分别不超过 512 与 448 像素的两个确定性视图，总 crop 像素不超过 6 MiPixel；大候选先按 bbox 和 alpha 摘要跳过，已知文字在逐项结果阶段去重。该策略仍使用 SAM2.1 Large，不降低 PDF 的 200 DPI 基线、SAM 采样点或候选阈值；代价是转换时间会增加。

最终质量门禁同时检查组件、无组件背景和合成结果：最终完整 alpha 内不得残留源字形，背景文字区不得留下相对局部底色可见的字印，可靠 OCR 文字必须且只能由可编辑文本框贡献一次。同候选的多项文字按页坐标一一匹配，一致项逐条回灌并重建全部资产；高置信冲突按确定顺序最多记录 96 条绑定 source SHA-256、稳定 `candidate_id` 和文字 bbox 的 `unowned_raster_text`，超出部分截断但该页仍保持硬失败。已通过叶组件继续冻结；若没有真实失败组件则立即 `preserved_with_warning`，否则最多完成五批真实修复后保留原始内容。不允许用栅格文字兜底。

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

#### 统一 Runtime

组件重建提供两种互斥的 Agent Provider；Provider 在创建 Run 时冻结，不会中途自动切换：

- `host`（默认）：直接使用当前 Codex、Claude Code 等宿主 AI，不下载额外的组件决策模型。宿主必须支持视觉识别、本地文件读取、工具调用和结构化 JSON；Runtime 会通过一次视觉能力握手后，逐轮请求组件计划。
- `local`（实验性）：使用用户显式安装的本地视觉模型，Runtime 在隔离子进程中自动完成每页最多五轮计划与质量门禁。转换期间严格离线，不会自动下载模型。

两种 Provider 当前均为 `experimental`，尚未完成真实文件双模式验收。Host 可能把诊断图交给宿主服务处理；敏感内容应选择完全离线的 Local。模型文件缓存只减少重复下载，不缓存图片语义判断；每张图片、每一页都重新分析，不跨图片复用拆分决策。

Local 模式不要硬编码模型名。先让 Skill/Agent 读取当前电脑配置和版本化模型目录：

```bash
image2editable models recommend --json
image2editable models status
```

只有推荐结果兼容、状态为已安装且有效时才能直接使用 Local。未安装时必须先向用户说明模型、revision、实验性状态和资源要求，并取得明确授权后执行 `image2editable models install agent`；Host 模式不执行模型探测或下载。两种模式共享相同的组件动作、最多五轮限制、确定性执行和质量门禁，互不读取对方的握手、计划或模型状态。

Host 由 Skill 循环执行 `run execute → agent next → 查看九项诊断证据（含 component-isolation.png）→ agent record → run execute`，直到完成或每页达到 5 轮。Local 查看相同证据并受同一硬门禁约束；两者的 confidence 都不能放宽门禁。已通过组件立即冻结；Agent 可将同一物理实体的重复碎片吸收到完整父组件，并在外缘画布足够一致时显式重建残影背景。文字区使用去字后的组件 RGB 和最终完整 alpha，不挖透明矩形；父组件仍失败时保留原页并标记 `preserved_with_warning`，不会用清空组件或栅格文字换取成功。重建组件通常是可移动的透明图片对象；项目不承诺把任意图形转换为原生矢量或 SmartArt。

```bash
# 图片、PDF 和图片版 PPTX 均可选择 Host 或 Local
image2editable convert input.png -o output.pptx --slide-size 16:9 --agent-provider host
image2editable convert input.pdf -o output.pptx --slide-size 16:9 --agent-provider local

# PDF 可直接转换，也可准备任务后按页请求一次细节重渲染再执行
image2editable convert input.pdf -o output.pptx --slide-size 16:9
image2editable prepare input.pdf --run-dir runs/pdf-job
image2editable run render-detail runs/pdf-job --page page_001
image2editable run execute runs/pdf-job
image2editable run recover runs/pdf-job

# PPTX：保留原生文字、形状、表格、图表和未命中图片；仅处理高置信整页截图候选
image2editable prepare input.pptx --run-dir runs/pptx-job --agent-provider host
image2editable run next runs/pptx-job
image2editable decision record runs/pptx-job \
  --page page_001 --object 7 \
  --decision replace --confidence 0.96 \
  --category full_slide_screenshot \
  --evidence "complete slide layout"

# Agent 确认并且组件质量门禁通过后，原位替换命中的截图页；失败页面保留原截图并给出 warning
image2editable run execute runs/pptx-job

# 检查本地依赖
image2editable doctor
```

Unified CLI 的位置参数概念为 `sources`：PPTX 输出可输入图片文件/目录、单个 PDF 或单个 PPTX；PSD 输出仅接受图片文件或图片目录。文档输入不能与其他来源混用，也不能重复传入；原有 `python image_to_ppt.py` 和 `python image_to_psd.py` 图片入口继续兼容。

`run recover` 只恢复执行锁已经消失的孤儿任务，不会终止仍在运行的转换进程。

PDF 会先自适应渲染，再复用现有“图像转可编辑 PPTX”管线。标准目标为 200 DPI；小页会提高到短边至少 1200 px，但不超过 300 DPI；所有渲染均限制长边不超过 6000 px、总像素不超过 24 MP。Agent 或用户可对每页调用一次 `render-detail`，以 300 DPI 目标重新渲染。物理宽高比相同的 PDF 页面可合并为保持该比例的多页 PPTX；混合宽高比时，`original` 输出为逐页文件，同时仍可生成统一 16:9 版本。所有布局均等比放置，不做非均匀拉伸。

PPTX 输入只读扫描 OOXML 原生对象、备注、图片关系和稳定指纹，仅将覆盖幻灯片至少 80% 且结构安全的大图标记为候选。Runtime 只允许 `replace + full_slide_screenshot + confidence >= 0.92` 进入组件重建；照片、Logo、装饰图和不确定项全部保留。候选页通过组件质量门禁和 PPTX reopen 后才进行 OOXML 原位替换，未通过时保留原截图并记录 warning；现有可编辑文字、形状、表格、图表、备注、z-order、其他页面和未命中图片保持原生对象。

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

# 使用本地视觉 Agent；Host Agent 为默认值
python image_to_psd.py input.png --lang en --agent-provider local
```


### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sources` | （必填） | 图片文件、多个图片文件、或目录路径；目录只扫描第一层图片 |
| `-o, --output` | 输入同名输出 | PPTX：`original` / `16:9` 单模式为文件路径，默认 `both` 时为输出基名；PSD 单图可为文件路径，多图为输出目录 |
| `--lang` | `ch` | OCR 语言，常用 `ch` / `en` |
| `--format` | `pptx` | Unified CLI 输出格式；PSD 使用 `--format psd`，且仅支持图片输入 |
| `--agent-provider` | `host` | 使用宿主视觉 Agent，或选择已安装模型的 `local` 模式 |
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
| PSD 分层 | 与 PPTX 共用 Agent 决策、Grounding DINO、SAM 2.1、OCR、背景修复和硬质量门 |

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
