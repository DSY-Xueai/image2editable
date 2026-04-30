# Image-to-PPT Converter

将图片转换为可编辑的 PowerPoint 演示文稿。通过计算机视觉和 OCR 技术，自动分离背景、前景组件和文本，生成分层可编辑的 PPTX 文件。

## 功能特性

- **背景建模与修复**：两轮迭代——平滑背景初检 + 原图 inpainting 精修，非前景区域像素级保留原图
- **前景提取与拆分**：基于差分 + Canny 边缘检测的前景 mask，连通域拆分为独立透明 PNG 组件
- **OCR 文本重建**：支持 PaddleOCR（推荐，中文精度更高）和 Tesseract 双引擎，自动估计字号、颜色、粗体
- **分层 PPTX 组装**：背景层 + 前景组件层 + 文本框层，每个元素独立可编辑

## 效果示意

```
输入图片 ──→ [OCR 文本检测] ──→ [背景建模] ──→ [前景提取] ──→ 可编辑 PPTX
                                                                  ├── 背景图层
                                                                  ├── 前景组件（独立 PNG）
                                                                  └── 文本框（可编辑文字）
```

## 快速开始

### 环境要求

- Python 3.9+
- OCR 引擎至少安装一个（见下方说明）

### 安装依赖

```bash
pip install -r requirements.txt
```

### OCR 引擎配置

**方式 A：PaddleOCR（推荐，中文识别更准）**

```bash
pip install paddleocr paddlepaddle
```

**方式 B：Tesseract（更轻量，需系统级安装）**

```bash
pip install pytesseract
```

系统安装 Tesseract：
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-chi-sim`

### 使用方法

```bash
# 基本用法
python image_to_ppt.py input.png

# 指定输出路径
python image_to_ppt.py input.png -o output.pptx

# 调整参数
python image_to_ppt.py input.png --lang ch --period 32 --diff-threshold 20 --min-area 20

# 添加参考页（第二页放原图对照）
python image_to_ppt.py input.png --reference
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `image` | (必填) | 输入图片路径（PNG、JPG 等） |
| `-o, --output` | 与输入同名 .pptx | 输出 PPTX 路径 |
| `--lang` | `ch` | OCR 语言代码 |
| `--period` | `32` | 背景建模瓦片周期 |
| `--diff-threshold` | `20.0` | 前景检测灵敏度 |
| `--min-area` | `20` | 最小组件面积（像素） |
| `--reference` | 关闭 | 添加原图参考页 |

## 项目结构

```
├── image_to_ppt.py     # 主入口（CLI + 管线编排）
├── text_detect.py      # 文本检测（PaddleOCR / Tesseract + 样式估计）
├── bg_model.py         # 背景建模（平滑初检 + inpainting 精修）
├── fg_extract.py       # 前景提取（差分 + 边缘检测 + 连通域拆分）
├── ppt_assemble.py     # PPTX 组装（背景 + 组件 + 文本框分层）
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量配置示例
└── docs/               # 设计文档与开发计划
```

## 技术栈

- **图像处理**：OpenCV、Pillow、NumPy
- **OCR**：PaddleOCR / Tesseract
- **PPT 生成**：python-pptx

## 已知限制

- 复杂纹理/渐变背景区域的修复质量有待提升
- 文本颜色采样在深色背景上可能不够精确
- 目前主要针对中英文文本优化，其他语言需要额外测试

## License

MIT
