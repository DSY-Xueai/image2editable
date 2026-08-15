---
name: image-to-ppt
description: 将图片、PDF、图片版 PPTX 或含原生对象的混合 PPTX 转换为严格质量校验、分层可编辑的 PowerPoint；保留既有原生文字、形状、表格和图表，并支持宿主视觉 Agent 或显式安装的本地视觉 Agent。用于截图、设计稿、科研图和幻灯片页面的组件重建、残影检查与可编辑输出。
---

# Image to PPT

把输入图片重建为分层 PPTX。保持严格语义拆分；质量校验失败时停止，不要将整页 flatten 为单张图片。

## 环境

- 使用 Python 3.10–3.12；该范围与当前项目测试和分发契约一致。
- 安装 `torch>=2.5.1`、`torchvision>=0.20.1`、Transformers 和 SAM 2.1。运行 `pip install -r references/requirements.txt`。
- LaMa 由内置的本地 TorchScript adapter 调用，依赖随 `references/requirements.txt` 中的 `torch>=2.5.1,<3` 安装。产品安装默认从已验证的 runtime receipt 解析模型；独立 skill 必须通过绝对路径设置 `LAMA_MODEL`，且文件须匹配固定 Big-LaMa 身份。
- 已安装 `image2editable` 产品包时，开始转换前运行 `image2editable doctor`；独立 skill 不假设该包存在，改用下列设备预检与三个显式模型路径。若 OCR 不可用，先让我选择：PaddleOCR（中文、英文和复杂版面识别通常更好，执行 `pip install paddleocr paddlepaddle`）或 Tesseract（较轻量，但还要安装系统 Tesseract，执行 `pip install pytesseract`）。**未经我确认，不要安装任何 OCR。** 我确认后安装所选项；产品环境再次运行 `doctor`，独立 skill 再次执行依赖和设备预检，通过后继续转换。
- 优先使用当前平台已正确安装、且通过 `doctor` 与下列设备预检的硬件加速环境，不要仅为 WSL 建议离开已经可用的环境：

  ```bash
  python -c "import sys, torch; print({'platform': sys.platform, 'cuda': torch.cuda.is_available(), 'rocm': torch.version.hip})"
  ```

- Windows/Linux 沿用 PyTorch 的设备接口：PyTorch 报告 CUDA 可用时使用 CUDA，ROCm 环境使用 PyTorch 提供的兼容设备接口。
- macOS 保持当前受支持的设备选择；在完成真实 Apple Silicon 回归前，不把 MPS 自动设为新默认。
- CPU 仍运行完整模型和相同质量门禁，包括 SAM 2.1 large，不替换为轻量分割模型，但推理会显著较慢。

推理不会下载模型或回退 Hugging Face cache。产品环境先安装并验证 runtime 模型；独立 skill 必须把 `SAM2_MODEL`、`LAMA_MODEL` 和 `GROUNDING_DINO_MODEL` 设置为绝对本地路径，其中前两者校验固定文件身份，DINO 目录视为操作者显式信任的 override。源码和权重不存放在此 skill 中。大/深遮罩需要 LaMa；依赖缺失或初始化失败时明确失败，不降级到容易产生条带拖影的 OpenCV 修复。

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

在完整仓库环境中，先按运行环境选择 Provider。Provider 写入 Run 后不可切换，两种模式共享同一套严格组件动作、最多五轮修复和质量门禁；质量没有改善时会提前停止，不会为了耗尽轮数重复执行。

优先选择 `host`：当前 Codex、Claude Code 等宿主必须支持视觉识别、本地文件读取、工具调用和结构化 JSON；该模式直接使用当前 AI，不探测、加载或下载本地组件决策模型。

Host 可能把诊断图交给宿主服务处理，敏感内容应选择完全离线的 Local。两种 Provider 当前都保持 `experimental`，直到使用相同真实文件完成视觉、结构和资源验收。

只有我已经部署本地视觉模型服务时才选择 `local`。该服务必须支持图像输入、JSON 输出和 OpenAI 兼容的 Chat Completions 接口；项目不下载、推荐或默认绑定模型。优先读取项目根目录 `.env` 中的 `IMAGE2EDITABLE_LOCAL_BASE_URL`、`IMAGE2EDITABLE_LOCAL_MODEL` 和可选 `IMAGE2EDITABLE_LOCAL_API_KEY`；同名环境变量可临时覆盖 `.env`。缺少地址或模型名时，说明缺少的配置并停止，不要猜测模型名、下载模型或回退到 Host。

每张图片、每一页都必须重新查看证据并独立决策，不能跨图片套用拆分决策。

Local 运行由 Runtime 内部串行完成：

```bash
image2editable convert input.pdf -o output.pptx --agent-provider local
image2editable prepare input.pptx --run-dir runs/pptx-job --agent-provider local
image2editable run execute runs/pptx-job
```

Host 运行先准备并推进到 `awaiting_agent`：

对 PPTX，`prepare` 后循环调用 `run next`。每个非 `null` 的 `candidate` 都必须查看 `image_path`，按 `--page candidate.page_id --object candidate.source_shape_id` 执行 `decision record`，然后继续 `run next`。仅当返回对象的 `candidate` 字段为 `null` 时才退出路由循环，随后首次执行 `run execute`。进入 `awaiting_agent` 后，再循环执行 `agent next`、`agent record` 和 `run execute`。

```bash
image2editable prepare input.pptx --run-dir runs/pptx-job --agent-provider host
image2editable run next runs/pptx-job
# candidate 非 null：查看 image_path；--page candidate.page_id --object candidate.source_shape_id
image2editable decision record runs/pptx-job \
  --page candidate.page_id --object candidate.source_shape_id \
  --decision replace --confidence 0.96 \
  --category full_slide_screenshot \
  --evidence "complete slide layout"
# 对每个后续非 null candidate，重复 decision record，然后继续 run next
image2editable run next runs/pptx-job  # 响应对象的 candidate 字段为 null，退出路由循环
image2editable run execute runs/pptx-job
image2editable agent next runs/pptx-job
image2editable agent record runs/pptx-job --plan response.json
image2editable run execute runs/pptx-job
```

第一次 `agent next` 返回视觉 challenge。必须实际查看 `image_path`，把观察到的 `shape/color/count` 写入 `host_capability_response` 后记录；不能从 metadata 或文件名猜答案。后续 `agent next` 返回当前组件请求及绝对证据路径。必须先验证并遵循完整 request、组件图、evidence map、全部 hash、候选和冻结状态，只查看并逐项核验 request 的有序 `review_evidence`，不得再按固定文件清单重复打开未列入本轮审查的图片。首轮 `review_evidence` 仍包含全部视觉证据；后续轮的 `round-review.png` 以相同坐标提供本轮失败或重开节点及依赖邻居的 source、isolation、ownership、reconstructed、difference 和 residual 无损视图。若 request 回退为完整 `review_evidence`，必须逐项查看；`quality-report.json` 仍作为完整质量证据读取，不能当图片发送，也不得跳过任何质量门禁。再生成绑定当前 `request_sha256` 的严格 `component_plan`。Host 与 Local 使用同一门禁，Agent confidence 不能放宽硬失败。

每页最多 5 个重修批次。已通过组件冻结；失败子组件折叠为完整父组件，父组件仍失败时只保留该页并报告 `preserved_with_warning`。不得用清空组件或栅格文字换取成功。可靠 OCR 文字必须全部由原生可编辑文本框贡献且仅出现一次；视觉组件和背景不得残留文字像素。重建组件通常是透明图片对象，不承诺把任意图形转换为原生矢量或 SmartArt。

prepare 会在全页 OCR 和首轮视觉候选完成后，对小型候选做两个最长边分别不超过 512 与 448 像素的确定性视图串行 OCR。比较文本只做 NFKC、casefold 和去空白，不删除语义标点；同候选多项文字按页坐标一一匹配，已知文字逐项去重，一致项逐条回灌并从 source 重建全部资产。高置信冲突按确定顺序最多写入 96 条绑定 source SHA-256、稳定 `candidate_id` 和文字 bbox 的 `unowned_raster_text`；超出部分截断但页面仍硬失败。后续 native-check 必须与初始哈希证据中的 diagnostics 结构和内容完全一致。页级硬失败不解冻已通过叶组件；没有真实失败组件时立即 `preserved_with_warning`，否则最多执行五批真实修复。不要根据文件名、语言或具体标签添加特例。

```json
{
  "schema_version": 1,
  "kind": "host_capability_response",
  "challenge_id": "agent next 返回的值",
  "observed": {"shape": "circle", "color": "#2b8a3e", "count": 3}
}
```

组件计划固定包含 `schema_version/kind/page_id/provider/repair_round/request_sha256/actions`；每个 action 固定包含 `action/object_ids/parameters/confidence/evidence`。只使用请求组件图中的候选 ID；`collapse_to_parent` 和 `absorb_into_parent` 可使用候选子组件关联的父 ID。既定十四类动作包括 `accept/discard/merge/split/expand/shrink/retry_with_box/retry_with_points/attach_text/suppress_text/collapse_to_parent/rebuild_background/absorb_residual/absorb_into_parent`，不添加未知字段。`accept`、`retry_with_box` 和 `retry_with_points` 仅在视觉证据确认对象是可单独移动的视觉元素、而非当前语义父级的子组件时，才可在既有参数中额外写入 `"independent": true`；不得按固定面积自动解除父关系。`rebuild_background` 可把已冻结视觉组件列为仅清理背景重复像素的对象，不得改变其冻结资产。

当 `quality-report.json` 包含 `unexplained_visual_residual` 时，必须查看 `unexplained-mask.png`。每个显著区域都必须由 active visual owner 覆盖；若区域是候选边界框内经验证的结构碎片，使用 `absorb_residual` 将绑定残差精确并入最小包含候选；否则对最接近的 inactive visual candidate 执行 `retry_with_box` / `retry_with_points`。不得用 accept、discard 或将其归为背景来消除违规。当 `background_text_residual` 是唯一阻断项时，对诊断命中的冻结文字或视觉 ID 执行 `rebuild_background`。

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

组件计划必须以可独立移动的最小完整视觉单元为单位。使用反事实标准：单独移动一个单元后，该单元及其余视觉单元是否仍各自完整；语义相关不构成合并理由。`component-isolation.png` 中沿 OCR 字形出现的透明孔洞，或本应连续的底色、填充、线条缺失，都属于残缺分割而不是成功去字；若失活父组件能恢复同一完整视觉单元，应使用 `collapse_to_parent`，同时保留可独立移动的高层组件且不得恢复源字形。质量报告出现 `contained_parent_review` 时，必须使用质量证据中的精确 `contained_parent_pairs` 对照两个隔离单元：若一个只是重复子集，选择唯一像素所有者并丢弃重复层；若两者确实都是可独立移动的视觉单元，双方必须分别使用 `accept`、置信度不低于 `0.92`，且每条 evidence 都把该 pair 的两个精确 ID 作为两个独立字符串列出，才允许共同保留，否则门禁继续硬失败。对被更完整组件覆盖、没有独立编辑价值的重复候选使用 `discard`。`absorb_into_parent` 只允许合并同一物理实体的重复掩码、碎边、阴影或分割缺口证据，禁止把多个可独立移动对象烘焙为一张父图；语义父级只用于分组，不参与最终像素渲染。仅当外缘画布颜色一致且证据显示背景残影时使用 `rebuild_background`，`margin_ratio` 必须在 `(0, 0.1]`。OCR 文字以冻结的 `text_XXXX` 节点出现，可作为 `attach_text` 的第二对象；只有视觉证据足以明确证明 OCR 候选实际为非文字时，才可对该文字节点使用 `suppress_text`，不确定或真实文字不得抑制。被抑制文字会从后续可编辑文字、文字蒙版、质量检查和 PPTX 中移除，并以该 OCR 边界框执行同质量 SAM，生成必须继续通过质量门禁的独立视觉候选；文字区域不能挖透明文字框，也不能只留在背景。任何动作仍需通过确定性重建、独占像素、残影/重影/缺损和 PPTX reopen 门禁；Agent 置信度不能放宽硬失败。

Runtime 只有在 `confidence >= 0.92` 时才重建完整截图。通过门禁后只原位替换命中的截图对象；既有原生文字、形状、表格、图表、备注、z-order、其他页面和未命中图片保持原生。未通过页面保留原截图并给出 warning，不伪装成可编辑组件。

`rebuild_background.margin_ratio` 必须由 Agent 根据当前残影和抗锯齿范围自适应选择：使用能完整覆盖残影、又不触及相邻结构线的最小值，禁止固定使用同一数值。

## 严格管线

1. 使用现有 OCR 逻辑检测文字并生成文字遮罩。
2. 使用 Grounding DINO 与 SAM 生成首轮视觉候选；对未被文字遮罩覆盖的小候选做资源有界的双视图 OCR，安全恢复后从源图重跑完整视觉准备。
3. 使用 Grounding DINO 生成整图与重叠分块语义候选，再用 SAM 2.1 生成对象掩膜，并以无提示 SAM 候选覆盖词表外对象。
4. 对候选去重，解析父子关系，为每个像素建立唯一 ownership；结合语义支撑和定向 SAM 复查修补组件内部破洞。
5. 导出不含文字的独立透明组件；小/窄遮罩用 OpenCV 修复背景，大/深遮罩用 LaMa。
6. 按实际导出的 RGBA 图层重建页面，执行严格视觉质量 QA。
7. 组装原比例或 16:9 画布；16:9 使用 contain 居中和四角/边缘颜色渐变，不使用模糊放大的原图副本。

## 输出与失败

每页从底到顶包含 clean background、可独立移动的透明组件和可编辑文本框。

命令会在处理每张图片前打印绝对 work directory。严格视觉质量失败时，异常包含 `mae`、`p95` 和 diagnostics 绝对路径；检查其中的 `source.png`、`ownership.png`、`reconstructed.png` 和 `report.json`。更早的分割、OCR 或 LaMa 失败仍可通过已打印的 work directory 定位资产。不要在失败后回退为整页图片。
