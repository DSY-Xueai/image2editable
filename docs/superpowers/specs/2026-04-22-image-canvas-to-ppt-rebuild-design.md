# Image To Canvas To PPT Rebuild Design (Phase 1)

## 背景与目标

当前项目在 `PDF/图片 -> PPT` 方案上已验证存在可用性问题。本次重构改为新主链路：

`image (png/jpg/webp) -> Canvas scene model -> pptx`

第一阶段目标：

1. 仅支持图片输入，逐张验收，不做批量吞吐优化。
2. 视觉优先（必要时降级为图片块），确保输出 PPT 可用。
3. 通过 PaddleOCR 提升文字可编辑覆盖率，但不强求 100% 文本识别。
4. 保持页面与对象尺寸/位置映射稳定，满足 PPT 坐标规范。

## 关键决策

### 1) 总体策略

- 采用“语义 Canvas 场景模型（Hybrid）”而不是纯 OCR 叠字或全量矢量化。
- Canvas 在这里定义为“统一中间场景抽象”，不是最终单层位图。
- PPT 生成器只消费 Canvas 模型，不直接消费 OCR 原始结果或输入图片。

### 2) 优先级

- 视觉优先（用户已确认）：
  - 当视觉保真与可编辑冲突时，优先保真。
  - 无法可靠语义化的复杂区域降级为图片节点。

### 3) 输入与 OCR

- 第一阶段只接收 `png/jpg/webp`。
- OCR 主引擎固定为 PaddleOCR。
- OCR 仅用于可编辑文本节点生成；低置信内容不强行文本化。

## 架构设计

### 模块划分

- `ingest_image.py`
  - 读取图片，标准化模式，记录原始宽高。
  - 输出：`ImageSource(width_px, height_px, image_path)`。

- `ocr_paddle.py`
  - 调用 PaddleOCR。
  - 标准化输出：`OCRTextBox(text, bbox, score, angle)`。

- `scene_segmentation.py`
  - 依据 OCR 框反推非文本区域。
  - 产出基础矩形图片分块（Phase 1 不做复杂矢量追踪）。

- `canvas_models.py`
  - 定义 `CanvasPage`、`CanvasNode` 数据结构。
  - `CanvasNode.type` 仅保留 `text | image | shape` 三类。

- `build_canvas_scene.py`
  - 编排主流程：聚合 OCR 与分块结果，构建 `CanvasPage`。
  - 维护统一 z-index 顺序：
    - `background image`
    - `non-text image nodes`
    - `text nodes`

- `canvas_to_ppt.py`
  - 负责 Canvas 到 PPT 的唯一映射。
  - 文本节点 -> 文本框；图片节点 -> 图片对象；基础形状 -> 形状对象。

- `fallback_policy.py`
  - 定义块级/页级降级策略与错误容忍规则。

## 数据模型

### CanvasPage

- `page_w_px: int`
- `page_h_px: int`
- `bg_color: tuple[int, int, int] | None`
- `nodes: list[CanvasNode]`

### CanvasNode

- `type: Literal["text", "image", "shape"]`
- `x, y, w, h: float`（像素坐标）
- `z: int`
- `payload: dict`（按节点类型解释）

### 文本节点 payload（Phase 1 最小集）

- `text`
- `score`
- `font_size_est`
- `color_est`
- `align`

### 图片节点 payload（Phase 1 最小集）

- `image_path`
- `crop_source`
- `confidence`

## 坐标与尺寸策略

- 使用图片像素坐标作为场景唯一几何坐标。
- 输出 PPT 页面尺寸按原图比例设置。
- 坐标映射采用线性换算：
  - `ppt_x = x_px * (slide_w_emu / page_w_px)`
  - `ppt_y = y_px * (slide_h_emu / page_h_px)`
  - `ppt_w = w_px * (slide_w_emu / page_w_px)`
  - `ppt_h = h_px * (slide_h_emu / page_h_px)`
- EMU 写入统一用 `round`，避免累计截断误差。

## 失败与降级策略

- OCR 失败：不终止，输出整页背景图并记录错误。
- 文本拟合失败：该文本节点回退为图片覆盖，不写错误文本。
- 单节点写入失败：跳过该节点，页面继续生成。
- 任意异常都不允许导致“整批崩溃”；单张模式下至少输出可打开 PPT。

## 运行产物与调试文件

每次单图转换输出：

- `outputs/<name>.pptx`
- `outputs/<name>_debug/canvas_scene.json`
- `outputs/<name>_debug/ocr.json`
- `outputs/<name>_debug/segmentation.png`

目的：支持逐张验收与快速定位（识别错误/层级错误/坐标偏差）。

## 参数（Phase 1）

保持参数最小化，避免过度抽象：

- `ocr_min_score`（默认 `0.78`）
- `text_expand_px`（默认 `2`）
- `min_text_area_px`（默认 `64`）
- `slide_long_edge_in`（默认 `13.333`）
- `debug_dump`（默认 `true`）

## 验收标准（逐张）

- 几何一致性：映射回像素后，关键对象位置偏差 <= 2px。
- 视觉一致性：主要结构无明显错位、串层、比例异常。
- 可编辑性：
  - 文本节点可选中并编辑。
  - 图片节点可选中并移动/缩放/替换。
- 稳定性：转换失败不崩，始终有可打开输出。

## 测试策略

- 单元测试：
  - 坐标映射
  - EMU round
  - 节点排序
  - fallback 分支
- 集成测试：单张样图跑全链路并断言产物完整。
- 人工回归：按用户逐张样本清单记录问题并迭代参数。

## 里程碑

- `M1`：最小链路跑通（整图入 PPT + Canvas 落盘）。
- `M2`：PaddleOCR 文本节点接入并可编辑。
- `M3`：非文本区域分块，减少文本重影。
- `M4`：稳定性修正与参数收敛（基于逐张反馈）。
- `M5`：冻结第一阶段版本，准备进入批量验证。

## 非目标（Phase 1）

- 复杂矢量路径的高保真可编辑重建。
- 混合模式、复杂透明、特效 1:1 映射。
- 任意字体下的精确排版复刻。

以上内容在第一阶段允许降级为图片节点，不阻塞主链路可用性。
