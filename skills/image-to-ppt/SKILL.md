---
name: image-to-ppt
description: 将图片、PDF、图片版 PPTX 或含原生对象的混合 PPTX 转换为严格质量校验、分层可编辑的 PowerPoint；保留既有原生文字、形状、表格和图表，并支持宿主视觉 Agent 或显式安装的本地视觉 Agent。用于截图、设计稿、科研图和幻灯片页面的组件重建、残影检查与可编辑输出。
---

# Image to PPT

把输入图片重建为分层 PPTX。保持严格语义拆分；质量校验失败时停止，不要将整页 flatten 为单张图片。

## 环境

- 使用 Python 3.10–3.12；上限来自 `simple-lama-inpainting 0.1.2` 的 NumPy/Pillow 依赖约束。
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

## 统一 Runtime Agent 模式

在完整仓库环境中，先按运行环境选择 Provider。Provider 写入 Run 后不可切换，两种模式共享同一套严格组件动作、最多五轮修复和质量门禁。

优先选择 `host`：当前 Codex、Claude Code 等宿主必须支持视觉识别、本地文件读取、工具调用和结构化 JSON；该模式直接使用当前 AI，不探测、加载或下载本地组件决策模型。

Host 可能把诊断图交给宿主服务处理，敏感内容应选择完全离线的 Local。两种 Provider 当前都保持 `experimental`，直到使用相同真实文件完成视觉、结构和资源验收。

只有用户要求离线/自托管 Agent 时才选择 `local`。不要硬编码模型；先读取当前电脑配置和版本化目录：

```bash
image2editable models recommend --json
image2editable models status
```

推荐结果必须 `compatible=true`，状态必须 `installed=true` 且 `valid=true`。若未安装，先向用户说明推荐模型、revision、`experimental/stable` 状态、内存/显存和磁盘结论；只有取得明确下载授权后才运行 `image2editable models install agent`。转换期间不会自动下载，也不自动回退到 Host。

模型缓存只复用已下载权重，不缓存图片语义判断。每张图片、每一页都必须重新查看证据并独立决策，不能跨图片套用拆分结果。

Local 运行由 Runtime 内部串行完成：

```bash
image2editable convert input.pdf -o output.pptx --agent-provider local
image2editable prepare input.pptx --run-dir runs/pptx-job --agent-provider local
image2editable run execute runs/pptx-job
```

Host 运行先准备并推进到 `awaiting_agent`：

```bash
image2editable prepare input.pptx --run-dir runs/pptx-job --agent-provider host
image2editable run execute runs/pptx-job
image2editable agent next runs/pptx-job
image2editable agent record runs/pptx-job --plan response.json
image2editable run execute runs/pptx-job
```

第一次 `agent next` 返回视觉 challenge。必须实际查看 `image_path`，把观察到的 `shape/color/count` 写入 `host_capability_response` 后记录；不能从 metadata 或文件名猜答案。后续 `agent next` 返回当前组件请求及八项绝对证据路径。逐张查看源图、编号掩码、OCR overlay、ownership、当前重建和差异图，再生成绑定当前 `request_sha256` 的严格 `component_plan`；记录并继续执行，直到完成或 Runtime 安全回退。

每页最多 5 个重修批次。已通过组件冻结；失败子组件折叠为完整父组件，父组件仍失败时保留原页并报告 `preserved_with_warning`。不得用清空组件换取成功。重建组件通常是透明图片对象，不承诺把任意图形转换为原生矢量或 SmartArt。

```json
{
  "schema_version": 1,
  "kind": "host_capability_response",
  "challenge_id": "agent next 返回的值",
  "observed": {"shape": "circle", "color": "#2b8a3e", "count": 3}
}
```

组件计划固定包含 `schema_version/kind/page_id/provider/repair_round/request_sha256/actions`；每个 action 固定包含 `action/object_ids/parameters/confidence/evidence`。只使用请求组件图中的候选 ID；`collapse_to_parent` 可使用候选子组件关联的父 ID。既定十类动作包括 `accept/discard/merge/split/expand/shrink/retry_with_box/retry_with_points/attach_text/collapse_to_parent`，不添加未知字段。

PPTX 的整页截图候选先使用决策路由：

```bash
image2editable run next runs/pptx-job
image2editable decision record runs/pptx-job \
  --page page_001 --object 7 \
  --decision replace --confidence 0.96 \
  --category full_slide_screenshot \
  --evidence "complete slide layout"
```

每次先查看 `run next` 返回的绝对 `image_path`。只有图片覆盖大部分页面、包含标题/多个文字区/图表或卡片等完整页面结构，且明显不是照片、Logo、头像或装饰素材时，才记录 `replace + full_slide_screenshot`。证据冲突或不确定时记录 `preserve` 或 `ambiguous`；不要为了提高拆分数量抬高置信度。

组件计划必须以视觉整体为单位：科研图、表格只是示例，不得按固定类型写死；不要把本属一个整体的组件无依据拆碎。对被更完整组件覆盖、没有独立编辑价值的重复候选使用 `discard`，页面级质量门仍会检查丢弃后是否缺失内容。OCR 文字以冻结的 `text_XXXX` 只读节点出现，只能作为 `attach_text` 的第二对象。任何动作仍需通过确定性重建、独占像素、残影/重影/缺损和 PPTX reopen 门禁；Agent 置信度不能放宽硬失败。

Runtime 只有在 `confidence >= 0.92` 时才重建完整截图。通过门禁后只原位替换命中的截图对象；既有原生文字、形状、表格、图表、备注、z-order、其他页面和未命中图片保持原生。未通过页面保留原截图并给出 warning，不伪装成可编辑组件。

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
