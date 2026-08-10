# 残差驱动的可编辑重建闭环设计

## 背景

GitHub 当前对外目标是把图片、PDF、图片版 PPTX 和混合 PPTX 中获准重建的截图区域转换为可继续修改的 PowerPoint：可靠文字成为原生文本框，可独立处理的视觉元素成为可移动或可替换的独立组件，混合 PPTX 中未参与重建的原生对象保持不变。

当前优化分支已经具备统一 Runtime、组件图、Agent 动作、重建路由、PPTX 组装、PowerPoint 权威渲染和质量门，但真实页面会在质量失败后保留整页栅格内容。这只能防止损坏，不能满足可编辑交付目标。

已确认的通用根因是闭环缺少有效反馈动作：

- `rebuild_background` 在 mask 动作执行器中按职责保持图状态不变，Legacy 执行阶段会另行重建背景；但当前接线只消费第一条背景动作，质量违反也不会确定性地产生修复请求。
- 质量门能检测 `background_text_residual`，却不能把诊断区域自动转成有界、可验证的背景修复。
- 初始候选筛选可能丢弃低分、小面积、重叠或未被首轮模型覆盖的视觉区域。
- 系统没有页面级“显著视觉区域必须有 ownership”的约束；对象留在背景中时，最终合成图仍可能获得很低的视觉误差。
- 相同组件计划可以被重复提交，最终以 `repeated_plan`、`empty_plan` 或轮次上限进入整页回退。

## 目标

在现有项目和现有重建流程内补齐残差驱动闭环，使项目声明支持的所有输入类型共享同一行为：

1. 可靠 OCR 文字只由可编辑 TextBox 表达。
2. 可独立移动的显著视觉实体尽量成为独立 Raster 或 Native 组件。
3. 复杂照片、插画和纹理对象可以保留为局部 Raster Component，但不得因难以细分而直接烘焙进整页背景。
4. 质量失败必须产生可执行的局部修复或新候选，不得只重复相同计划。
5. 只有视觉、ownership、文字隔离、PowerPoint 重开和混合 PPTX 不变性全部通过，页面才进入 `ready_for_assembly`。

## 非目标

- 不建立第二套 Runtime、Router、Scene Graph、Agent 或 PPTX 组装入口。
- 不按文件名、页面主题、固定对象数量或真实验收样本坐标写规则。
- 不要求照片和复杂插画内部全部矢量化或转换成原生 PowerPoint 形状。
- 不在本轮增加 Table、SVG、公式、SmartArt 或通用 Native Shape 类型。
- 不引入转换时下载的新模型，也不让 Agent 直接生成、修改或修补像素。
- 不放宽现有文字残影、组件重复、过度合并、原生对象保护和 PowerPoint Render QA 门禁。

## 设计原则

### 增量修改

本轮复用现有输入适配、OCR、Grounded detection、SAM、组件图、Agent Provider、背景模型、组件质量、Router、Assembler 和 PowerPoint Renderer。新增信息作为现有组件图和质量报告的派生数据存在，不创建平行主流程。

### 可编辑性优先

整页视觉相似不等于重建成功。成功判断先验证文字与视觉对象 ownership，再验证渲染相似度。整页图片、原页复制或输入输出字节相同只能作为内部容灾，不计为可编辑成功。

### 背景需要证据

背景不是“未被提取的剩余像素”。一个显著区域只有在没有文字、组件、几何、检测或显著结构证据，并且局部连续性验证支持其属于环境纹理时，才可归入背景。

### 局部修复与单调收敛

通过门禁的组件保持冻结。失败只重做诊断区域。每轮必须减少未解释区域、文字残影、重复/过度合并或权威渲染误差中的至少一项，且不得使更高优先级指标退化。

## 现有流程内的数据流

```text
输入页面
  -> OCR text ownership
  -> Grounded / prompt-free SAM / geometry 初始候选
  -> 现有组件图去重、拆分和 ownership 分配
  -> 按 text + component removal mask 重建背景
  -> 组装 background + components + editable text
  -> 计算文字、组件、背景和页面级质量
  -> 生成 unexplained mask 与局部诊断
  -> 对 unexplained region 生成 box / point / multi-scale SAM 候选
  -> Agent 仅裁决仍有歧义的候选关系
  -> 执行局部 mask 动作和 rebuild_background
  -> 重新质量检查并冻结通过对象
  -> 收敛后进入现有 Router / PPTX Adapter / PowerPoint Render QA
```

图片、PDF 渲染页、图片版 PPTX 页面和混合 PPTX 中获准替换的截图对象都从同一个页面源图进入上述流程。输入 Adapter 只负责准备页面和保护原生对象，不改变提取算法。

## 页面 ownership ledger

ownership ledger 由现有 OCR mask、组件图 mask、presentation alpha、候选证据和质量上下文逐轮确定性派生，不另建持久数据库。每个页面像素具有以下职责之一：

- `text`：属于可靠 OCR 项，由一个冻结 TextBox 负责最终渲染。
- `component`：属于一个活动或冻结视觉组件的真实源像素。
- `generated_underlay`：为组件可移动性而生成的内部底图像素，不冒充真实 ownership。
- `background`：没有前景证据且通过局部背景连续性判断的像素。
- `unexplained`：包含显著前景证据但没有文字或组件 ownership 的像素。

真实源像素不能同时属于两个组件。`generated_underlay` 可以在契约允许的父子遮挡区域与更高层组件重叠，但必须继续使用现有 underlay 边界、接缝和越界门禁。

### 显著前景证据

显著前景证据采用页面自适应标定，不使用特定颜色、语言、页面比例或固定绝对像素阈值。它是以下证据的并集：

- Grounded detection 和 prompt-free SAM 候选；
- 现有 geometry candidates；
- OCR mask 之外，相对低频背景模型具有稳定颜色或梯度差异的连通区域；
- 闭合边缘、重复结构和局部显著性支持的区域；
- 移除现有文字或组件后暴露出的结构残差；
- 上一轮组件边界外仍与源对象连续的孤立像素。

低于现有 `calibrate_page` 自适应噪声下限的孤立碎片不作为独立实体。达到有效面积、边缘长度或重复结构条件的连通区域必须进入 `component`、`text` 或 `unexplained`，不能静默归入背景。

### 背景判定

一个歧义区域只有同时满足以下条件才可归入背景：

1. 不与可靠 OCR、检测、SAM 或几何候选形成有效支持；
2. 与其父区域或页面边界保持颜色、梯度和纹理连续；
3. 将其作为独立对象移除后，局部边界接缝和高频误差不会改善；
4. Agent 若参与判断，只能在确定性候选中的 `background` 与 `component` 解释之间选择，并提交对应证据。

## 残差驱动候选闭环

### 初始候选

继续使用现有 Grounded、prompt-free SAM、geometry 和 residual candidate 生成器。现有去重逻辑保留，但任何因重叠或低置信度被过滤的候选，如果覆盖显著 `unexplained` 区域，必须保留到残差协调阶段，不能直接丢弃。

### 残差候选

对每个达到自适应显著性下限的 `unexplained` 连通区域：

1. 生成紧 bbox，并以区域内部稳定点作为 positive prompts、邻近已归属对象和文字作为 negative prompts。
2. 调用现有 SAM predictor 生成局部候选，不重新加载模型或重复整页 embedding。
3. 在原尺度和一个扩展上下文尺度上比较候选，拒绝跨越已冻结 ownership 的 mask。
4. 使用现有重复、包含、父子和叶组件规则协调新旧候选。
5. 若候选包含多个空间独立实体，保留叶组件并把父级仅作为语义关系，不把父级烘焙为最终组件。
6. 若无法可靠细分复杂对象，保留覆盖该对象的最小完整 Raster Component；其 presentation alpha 和移除后的背景仍必须通过质量门。

### 覆盖门禁

质量报告新增页面级派生指标：

- `material_foreground_pixels`
- `owned_visual_pixels`
- `unexplained_visual_pixels`
- `largest_unexplained_region_pixels`
- `visual_ownership_coverage`

只要存在达到自适应显著性下限的 `unexplained` 连通区域，页面违反 `unexplained_visual_residual`，不能进入 `ready_for_assembly`。coverage 比例用于诊断和比较轮次，不使用单一百分比掩盖一个显著漏失对象。

## 有效的 `rebuild_background`

### 动作语义

现有 `rebuild_background` 继续引用组件图中的文字或视觉对象 ID，并使用现有 `margin_ratio` 形成有界 removal mask。动作不改变组件 ownership，只请求在指定区域重新生成背景。

mask 动作执行器继续保持图状态不变，不承担背景模型职责。Legacy 执行阶段聚合本轮全部 `rebuild_background` 动作，输出与本轮 graph hash、source hash 和 action hash 绑定的 repair mask，并在当前背景上执行局部修复。后续质量检查和 evidence 必须消费修复后的背景，避免 Agent 看到的证据与最终 PPTX 不一致。

### 修复策略

- 窄小或边界上下文充分的区域使用现有 OpenCV Telea + Navier-Stokes 路径。
- 大面积、内部深度超过局部 inpaint 能力的区域使用 `scripts/bg_model.py` 已有 large-mask inpaint 路径。
- 修复范围限制在 removal mask 与页面自适应邻域内，不改动无关区域。
- 候选背景按边界色差、边界梯度突变、残留边缘、高频补丁和文字残影排序。
- 修复未改善任何相关指标或引入更严重接缝时，拒绝该背景候选，并保留失败诊断供下一轮改变 mask，而不是重复相同动作。

### 文字背景

可靠 OCR mask 在抗锯齿笔画和自适应 halo 扩展后进入 removal mask。修复后必须同时通过：

- 背景隔离图不存在可识别文字残影；
- 组件隔离图不存在对应栅格文字；
- 合成图中该文字恰好由一个可编辑 TextBox 贡献。

## Agent 职责

Host 与 Local Agent 使用同一契约和证据。Agent 只处理以下歧义：

- 重叠候选是否属于同一物体；
- 一个大候选是否需要拆成多个可独立移动的叶组件；
- 歧义区域更符合独立对象还是背景纹理；
- 应使用现有 `split`、`retry_with_box`、`retry_with_points`、`accept`、`merge` 或 `rebuild_background` 动作。

Agent 不得：

- 绕过 ownership、文字残影、重复、过度合并或 PowerPoint Render QA 硬门禁；
- 因对象复杂而把显著 `unexplained` 区域直接归入背景；
- 使用整页父组件代替多个可分离对象；
- suppress 可靠 OCR 文字以通过质量门；
- 重复没有带来指标提升的标准化计划；
- 直接生成或修改 mask、背景或组件像素。

Agent evidence 增加 `unexplained-mask.png` 和对应区域表。只向 Agent提供失败区域及必要上下文，不在每个确定性修复轮次重复提交完整页面证据，从而控制 Token 和等待时间。

## 收敛与状态

页面质量按以下字典序验证，较低优先级改善不能抵消较高优先级退化：

1. 显著 `unexplained` 区域数量和最大面积；
2. `background_text_residual`、`component_text_residual` 和文字唯一性；
3. 缺失、重复、over-merged、orphan 和 ownership 冲突；
4. PowerPoint 权威渲染的结构与视觉误差。

现有最多五次 Agent 语义决策保持不变。每次 Agent 决策内部可以执行多轮低成本、确定性的残差发现和背景修复，直到本轮不再产生新候选或质量不再提升。确定性内循环设置基于状态变化的终止条件，不使用任意 sleep 或无界重试。

以下行为不计为进步：

- 标准化计划与上一轮相同；
- `rebuild_background` 没有改变背景资产或相关质量指标；
- 只通过保留更多源图像素提高 SSIM；
- 通过停用可靠文字、清空组件或扩大整页父组件减少违反项。

通过的文字和视觉组件继续冻结；背景修复不得改写其真实 ownership。未通过闭环的页面不能进入 `ready_for_assembly`。整页保留只作为内部容灾：任务必须明确记录未完成，不能把该 PPTX 计作可编辑转换成功。

## 输入类型一致性

- 图片和图片目录：每张图片直接作为页面源图进入闭环。
- PDF：按现有等比例页面渲染生成源图；页序、页面比例和多页合并保持不变。
- 图片版 PPTX：全页图片候选作为待重建源图，不把原图本身当作已提取组件。
- 混合 PPTX：只处理显式获准替换的截图对象；未命中的原生文字、形状、表格、图表、备注、对象 ID 和 z-order 保持不变。
- PSD：继续消费 Raster/Text 能力，但本轮不要求 PowerPoint Render QA；PPTX 与 PSD 仍消费同一最终组件 ownership 结果。

## 测试设计

### TDD 顺序

每项行为先建立失败测试，再做最小实现：

1. `rebuild_background` 当前无效的回归测试。
2. 显著未归属对象仍留在背景时，页面必须失败的 ownership 测试。
3. 残差区域生成新 SAM box/point 候选并减少 unexplained 的测试。
4. 无法细分的复杂局部对象保留为最小 Raster Component，而不是整页回退的测试。
5. 无指标提升的背景动作不能被视为有效修复的测试。
6. 跨 PNG、PDF、图片版 PPTX 和混合 PPTX 的一致性测试。

### 参数化通用页面

测试构造使用参数化布局，不依赖真实样本坐标：

- 纯色、横纵渐变、纹理和照片背景；
- 中文、英文、深浅色和抗锯齿文字；
- 卡片、箭头、图标、照片、嵌套对象、透明区域和阴影；
- 相邻对象、重叠对象、重复检测、父子遮挡和小型显著对象；
- 多种页面比例、分辨率和多页组合。

必须验证：

- 可靠 OCR 项与冻结 TextBox 一一对应；
- 显著前景区域具有唯一文字或组件 ownership；
- 删除全部 TextBox 后，背景和视觉组件中不出现对应文字；
- 删除或移动任意视觉组件后，其他组件不缺损，暴露背景无空洞、重影或源对象残留；
- 无整页 Raster Component 冒充多个可分离对象；
- 同一页面包装为 PNG、PDF 和图片版 PPTX 后得到等价的文字与组件语义；
- 混合 PPTX 未命中原生对象保持不变。

### 真实验收

真实验收语料覆盖：

- 仓库 README 的三组演示页面；
- 当前复杂图片和两页图片版 PPTX；
- 至少一个真实多页 PDF；
- 至少一个同时包含原生对象和全页截图候选的混合 PPTX。

真实样本只作为验收语料，不在生产代码中出现文件名、固定对象数量、坐标、颜色或内容关键词。验收使用 PowerPoint 原生打开、文字编辑、组件移动、重新渲染和对象结构检查，不以单独 SSIM 作为成功依据。

## 预期修改范围

- `scripts/visual_segment.py`：执行现有背景动作，扩展 residual candidates 与协调逻辑。
- `skills/image-to-ppt/scripts/visual_segment.py`：与运行脚本保持字节一致。
- `scripts/bg_model.py`：仅在现有局部/large-mask 修补接口不能返回所需诊断时做最小扩展。
- `skills/image-to-ppt/scripts/bg_model.py`：与运行脚本保持字节一致。
- `image2editable/component_quality.py`：派生 ownership ledger、unexplained mask 和页面门禁。
- `image2editable/component_contracts.py`：补充现有 evidence 与动作结果所需的最小契约。
- `image2editable/component_repair.py`：让质量违反驱动有效局部修复与残差候选，并验证单调收敛。
- `image2editable/legacy.py` 或 `image2editable/runtime.py`：仅在现有调用链不能传递修复后背景时做最小接线。
- 对应测试文件和 `Course.md`。

不预先修改 Router、IR、PowerPoint Renderer、PPTX Adapter 或 Native Shape 实现。只有测试证明最终 ownership 没有被现有消费者正确使用时，才对对应消费者做可追溯的最小修复。

## 发布阻断条件

出现以下任一情况，本轮优化不能发布：

- 图片或 PDF 通过整页图片回退被标记为成功；
- 可靠文字仍存在于背景或视觉组件中；
- 达到显著性下限的可分离对象仍留在背景；
- 组件移动后暴露明显残影、空洞或重复对象；
- 相同输入在 PNG、PDF、图片版 PPTX 路径产生不一致的核心 ownership；
- 混合 PPTX 未命中原生对象、备注或 z-order 被改变；
- 新路线让旧路线已经合格的仓库演示页面发生退化；
- 全量自动化、PowerPoint 原生重开或真实验收未通过。

## 完成标准

本设计完成不以“新增框架”或“测试数量增加”为判断依据。只有项目声明支持的输入路径共享同一闭环、通用性质测试通过、真实页面产生可编辑文字与可移动局部组件、无显著元素被偷留在背景，并且 PowerPoint 实测保持视觉和结构正确，才算本轮优化成功。
