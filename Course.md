# Course

## 当前项目状态
- 核心功能：图片→可编辑 PPT 转换（背景建模/修复 + 前景拆分 + 文本重建）。
- 管线：读图 → OCR 文本检测 → 自适应背景建模 → 前景提取/组件拆分 → PPTX 分层组装。
- OCR：PaddleOCR 优先（已 patch OneDNN mkldnn bug），pytesseract 回退。
- 自动估计字号（非线性校正，±7% 偏差）、颜色（采样）、粗体（ink ratio）。
- `skills/ppt-master/` 保留为 SVG→PPTX 基础设施（本管线未使用但不冲突）。

## 本轮变更
- **架构重建**：从 AI 视觉/OCR+SVG 方案切换到纯本地 CV 管线。
- 删除 `faithful_convert.py`、`vision_analyzer.py`、`auto_convert.py`、`demo.py`（旧方案）。
- 新增 5 个模块化文件：
  - `image_to_ppt.py` — 主入口 + CLI + 管线编排
  - `text_detect.py` — OCR 文本检测 + 样式估计
  - `bg_model.py` — 自适应背景建模 + inpainting 双 pass 修复
  - `fg_extract.py` — 前景提取 + 连通域组件拆分
  - `ppt_assemble.py` — PPTX 分层组装（背景 + 前景组件 + 文本框）
- 更新 `requirements.txt`：新增 `opencv-python`，整理依赖说明。
- **关键修复**：
  1. PaddleOCR 可用（patch 掉 PaddlePaddle 3.x OneDNN mkldnn bug）→ 中文识别正确
  2. 前景组件提取完整（测试图 146 个组件，超过参考 PPTX 的 104 个）
  3. 背景 inpainting 双 pass 修复（TELEA + NS，更大 radius，边界模糊平滑）
  4. 默认不输出参考页（可通过 `--reference` 开启）
  5. OCR 噪声过滤（小字号 < 8pt、纯符号行、乱码大写字母串）
  6. 字号非线性校正（大字号校正更多，偏差控制在 ±7%）
  7. 粗体检测（Otsu + ink ratio > 0.20）和居中对齐正确

## 关键文件
- `image_to_ppt.py` — 主入口（CLI + 管线编排）
- `text_detect.py` — 文本检测模块（PaddleOCR / pytesseract + 样式估计 + mkldnn patch）
- `bg_model.py` — 背景建模模块（自适应边缘采样 + 瓦片中值 + 双 pass inpainting）
- `fg_extract.py` — 前景提取模块（差分阈值 20 + 连通域拆分 + alpha feathering）
- `ppt_assemble.py` — PPTX 组装模块（背景层 + 前景组件层 + 全幅居中文本框层）
- `skills/ppt-master/` — ppt-master 基础设施（SVG→PPTX，保留未使用）

## 运行入口
```bash
# 基本用法（默认不加参考页）
python image_to_ppt.py input.png

# 指定输出路径
python image_to_ppt.py input.png -o output.pptx

# 调整参数
python image_to_ppt.py input.png --lang ch --period 32 --diff-threshold 20 --min-area 20

# 加参考页（第二页放原图对照）
python image_to_ppt.py input.png --reference
```

## 运行时依赖
- 见 `requirements.txt`（核心：python-pptx、opencv-python、Pillow、numpy）
- OCR 引擎至少需要一个：PaddleOCR（推荐，已内置 mkldnn patch）或 Tesseract

## 关键目录
- `skills/ppt-master/` — ppt-master skill（SVG→PPTX 基础设施，保留）
- `archive/` — 旧 skill 归档

## 当前注意事项
- PaddleOCR v3.5.0 + PaddlePaddle 3.3.1 的 OneDNN mkldnn bug 已通过 `_patch_paddle_mkldnn()` 修复（强制 `run_mode='paddle'`）。
- Windows 环境需先 import torch 再 import paddle 以避免 DLL 路径污染（patch 中已处理）。
- 粗体检测基于墨水密度（Otsu + ink ratio），对大多数中英文文本有效。
- 背景建模为自适应方案（边缘采样 + 瓦片中值 + 双 pass inpainting），支持亮色/暗色背景。
- 前景组件按连通域拆分（diff_threshold=20, min_area=20），每个组件为独立透明 PNG。
- 待优化：背景修复质量（复杂纹理/渐变区域）、文本颜色精度、更多图片类型验证。
