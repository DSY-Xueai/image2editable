---
name: image-to-psd
description: 将一张或多张图片转换为经过严格质量校验的分层 PSD；可独立运行，也可使用 image2editable 的 Host、Local 或本地服务 Agent。输出修复背景、独立透明视觉组件和可编辑 Photoshop 文字图层。仅支持图片输入，不用于 PDF 或 PPTX。
---

# Image to PSD

把图片重建为分层 PSD。文字只由可编辑文字图层贡献一次；视觉组件和背景不得残留文字像素。质量检查失败时停止，不把整页图片伪装成分层结果。

## 输入与授权

- 仅支持 PNG、JPEG、BMP、TIFF 和 WebP。
- 单图输出一个 `.psd`；多图输出到目录，同名文件使用稳定序号区分。
- 每个 PSD 包含修复背景、按 z-order 排列的透明视觉组件和可编辑文字图层。
- PSD 写入依赖已授权的 Aspose.PSD。模型推理前必须设置 `ASPOSE_PSD_LICENSE`；授权缺失或无效时立即停止。

Windows PowerShell：

```powershell
$env:ASPOSE_PSD_LICENSE="C:\path\to\Aspose.PSD.lic"
```

Linux/macOS：

```bash
export ASPOSE_PSD_LICENSE=/path/to/Aspose.PSD.lic
```

授权文件、模型权重、OCR 缓存和运行产物都不存放在此 skill 中。

## 独立运行

独立模式不需要安装 `image2editable` 产品包。使用 Python 3.10-3.12，并从 skill 根目录安装依赖：

```bash
python -m pip install -r references/requirements.txt
```

若 OCR 尚未准备好，先让用户选择 PaddleOCR 或 Tesseract；未经确认不要安装。PaddleOCR 更适合中文、英文和复杂版面，Tesseract 较轻量但还需要系统程序。

```bash
# PaddleOCR
python -m pip install "paddleocr==3.7.0" "paddlepaddle==3.3.1" "PaddleX==3.7.2" "PyYAML==6.0.2"

# Tesseract Python adapter
python -m pip install pytesseract
```

开始转换前，把三个模型配置为绝对本地路径：`SAM2_MODEL` 和 `LAMA_MODEL` 指向文件，`GROUNDING_DINO_MODEL` 指向目录。独立模式不读取产品 receipt，也不运行 `image2editable doctor`。

```bash
python -c "import os; from pathlib import Path; names=('SAM2_MODEL','LAMA_MODEL','GROUNDING_DINO_MODEL'); raw={name: os.environ.get(name, '') for name in names}; paths={name: Path(value) for name, value in raw.items()}; assert all(raw.values()) and all(path.is_absolute() for path in paths.values()) and paths['SAM2_MODEL'].is_file() and paths['LAMA_MODEL'].is_file() and paths['GROUNDING_DINO_MODEL'].is_dir(); print('runtime model paths: ok')"
```

推理不会下载模型或回退 Hugging Face cache。SAM 和 LaMa 文件必须匹配固定身份；DINO 目录视为用户明确提供的本地 override。LaMa 缺失或初始化失败时停止，不降级到容易产生条带或拖影的 OpenCV 修复。

检查当前设备后再运行：

```bash
python -c "import sys, torch; print({'platform': sys.platform, 'cuda': torch.cuda.is_available(), 'rocm': torch.version.hip})"
```

CPU 仍使用完整模型和相同质量门，速度会明显慢于 GPU。macOS 在真实 Apple Silicon 回归完成前不自动把 MPS 设为默认。

从 skill 根目录运行 module，不要直接执行脚本文件：

```bash
cd skills/image-to-psd
python -m scripts.image_to_psd input.png
python -m scripts.image_to_psd input.png -o output.psd
python -m scripts.image_to_psd img1.png img2.png -o psd-output
python -m scripts.image_to_psd images/ -o psd-output --lang en
```

standalone CLI 只负责图片重建，不接受 `--agent-provider`。它先完成全部页面的严格准备，再发布 PSD；任一页面失败时不会留下部分输出。

## 产品 Runtime

完整仓库或已安装的 `image2editable` 支持 `host`、`local` 和 `local-service`。三种 Provider 使用同一组件动作、最多 5 批修复和相同质量门，运行开始后不能切换。

先安装 PSD 依赖。OCR 就绪并获得用户同意后，安装并校验固定的 SAM、LaMa 和 DINO runtime：

```bash
python -m pip install -e ".[psd]"
image2editable models install runtime
image2editable doctor
```

`host` 直接使用当前支持视觉、本地文件读取、工具调用和结构化 JSON 的宿主，不探测或下载本地组件决策模型。敏感文件应使用用户已经准备好的 `local` 或 `local-service`。

`local` 使用用户明确安装的内置 Qwen Agent。安装模型前必须再次确认：

```bash
python -m pip install -e ".[psd,agent-local]"
image2editable models install agent
image2editable doctor --agent-local
image2editable convert input.png -o output.psd --format psd --agent-provider local
```

`local-service` 用于用户已经部署的 OpenAI-compatible 视觉模型服务。它必须支持图片输入、JSON 输出和 Chat Completions。优先读取项目根目录 `.env` 中的 `IMAGE2EDITABLE_LOCAL_BASE_URL`、`IMAGE2EDITABLE_LOCAL_MODEL` 和可选 `IMAGE2EDITABLE_LOCAL_API_KEY`；同名环境变量可以临时覆盖。缺少地址或模型名时停止，不猜测模型名、不下载模型，也不回退到其他 Provider。

```bash
image2editable convert input.png -o output.psd \
  --format psd --agent-provider local-service
```

Host 模式先准备 Run，再推进到 `awaiting_agent`：

```bash
image2editable prepare input.png -o output.psd \
  --run-dir runs/psd-job --format psd --agent-provider host
image2editable run execute runs/psd-job
image2editable agent next runs/psd-job
image2editable agent record runs/psd-job --plan response.json
image2editable run execute runs/psd-job
```

第一次 `agent next` 返回视觉能力 challenge。必须实际查看 `image_path`，记录观察到的 `shape`、`color` 和 `count`，不能从文件名或 metadata 猜测。之后每轮只查看 request 中按顺序列出的 `review_evidence`，同时核验完整 request、hash、组件图、候选和冻结状态；`quality-report.json` 作为质量证据读取，不能当图片发送。

计划必须绑定当前 `request_sha256`。每个 action 只使用请求组件图中的 ID，并限定为现有十四类动作：`accept`、`discard`、`merge`、`split`、`expand`、`shrink`、`retry_with_box`、`retry_with_points`、`attach_text`、`suppress_text`、`collapse_to_parent`、`rebuild_background`、`absorb_residual`、`absorb_into_parent`。Agent confidence 不能放宽硬失败。

## 质量与失败

- 每张图片独立判断，不能跨图片套用拆分结果。
- 每个视觉组件应是可独立移动的最小完整单元，不得残缺、重叠、吸收相邻对象或只保留阴影碎片。
- 已通过组件立即冻结；质量没有改善时提前停止，不为耗尽轮数重复执行。
- `rebuild_background.margin_ratio` 使用能覆盖残影且不触及相邻结构的最小值，不固定写死。
- `unexplained_visual_residual` 必须由 active visual owner 覆盖；不能用 `accept`、`discard` 或归为背景来消除违规。
- 可靠 OCR 文字必须全部写为可编辑文字图层，并且只能出现一次。
- 产品 Runtime 页面最终成为 `preserved_with_warning` 时不生成伪分层 PSD；保留诊断目录并明确报告失败。
- standalone 严格质量检查失败时异常包含指标和诊断路径；检查 `source.png`、`ownership.png`、`reconstructed.png` 和 `report.json`，不要回退为整页图片。
