# 真实验收数据链修复设计

## 背景

Task 13 首个真实 Host PNG 在第 1 轮暴露了三个数据链问题：Prepared Page 只保存可见组件掩码，语义父掩码在进入 Runtime 前丢失；组件背景修复会再次污染已经由 OCR 清理的文字区域；首轮 `reconstructed.png` 只显示背景，没有合成当前活跃组件。结果是所有候选都被标记为独立 parent，Agent 看不到真实交付画面，并且最终文字下方可能出现浅灰残影。

本设计修复通用数据契约，不按图片类型、文件名或当前样本设置阈值。

## 目标

- 从传统视觉分割一直保留“完整语义父组件 + 可编辑可见子组件”的层级。
- 组件背景修复不能重新污染已通过文字清理的区域。
- Host 与 Local Agent 看到的重建证据必须等同于当前确定性交付语义。
- 保持每页最多 5 个重修批次、冻结规则、父级/原页回退和 Provider 隔离不变。
- 新 Prepared Page 可恢复，旧版本仍可读取并采用原有安全行为。

## 非目标

- 不新增图片类别判断规则。
- 不缓存跨图片语义决策。
- 不新增 `retry_text_cleanup` 等由 Agent 猜测确定性清理错误的动作。
- 不承诺把组件转换为原生矢量或 SmartArt。
- 本轮不下载 Local 模型。

## 方案

### 1. Prepared Page v2 保留父子掩码

视觉阶段已有每个元素的 `semantic_mask` 和 `mask`。v2 将两组掩码分别作为 hash-bound 资产保存：

- `semantic_masks`：完整语义父掩码；
- `element_masks`：当前可编辑子掩码。

两组数量必须与组件数量相等；每个子掩码必须非空、尺寸一致且完全位于对应父掩码内。加载器继续支持 v1：v1 没有语义父掩码时沿用旧的单层 parent 图，不伪造层级。

### 2. Runtime 构建真实父子图

对 v2 的每一对掩码创建：

- `parent_XXXX`：`kind=parent`、`state=inactive`、无父级，保存完整语义掩码；
- `component_XXXX`：`kind=child`、`state=pending`、`parent_id=parent_XXXX`，保存可编辑子掩码。

初始待修数量仍按可见子组件计数，inactive 父组件只作为最多五轮失败后的完整回退资产，不参与普通渲染和质量计数。OCR 文字继续作为 frozen `text_XXXX` 节点。

### 3. 文字区域使用确定性清理结果

视觉组件移除完成后，将 `text_cleanup_mask` 覆盖区域从已经生成的 `text_clean_image` 恢复到 clean background。该操作发生在每次用于残余检测或最终交付的背景生成之后，防止大面积组件 inpaint 再次生成文字轮廓、浅灰栅格或阴影副本。

若当前页面没有可靠 `text_clean_image`，保持既有背景行为，不用未经验证的像素替换。

### 4. 首轮证据按交付语义合成

首轮 `reconstructed.png` 从 clean background 开始，只合成状态为 pending、pending_gate 或 frozen 的视觉节点，并排除文字遮罩区域；inactive 父组件和 text 节点不渲染。`difference.png` 基于该结果生成。

后续轮次继续使用执行器产出的重建图。Host 和 Local 因此看到同一份真实状态，不需要 Provider 专用逻辑。

## 失败与兼容行为

- v2 父子资产数量、尺寸、包含关系或 SHA-256 不一致时，在发布 Agent 证据前 fail closed。
- v1 状态可继续恢复，但不会宣称具有不存在的父子层级。
- 子组件五轮后仍失败时恢复 hash-bound 完整父掩码；父组件门禁仍失败时保留原页并报告 `preserved_with_warning`。
- 修复不改变 Host/Local 状态、计划或模型 provenance 的隔离边界。

## 测试

按 TDD 分三组推进：

1. 背景测试：模拟组件 inpaint 在文字区域产生灰色污染，证明恢复后该区域逐像素等于 `text_clean_image`。
2. Prepared Page 测试：v2 往返保存父子掩码、拒绝数量/包含关系错误，并验证 v1 兼容读取。
3. Runtime 测试：初始图包含 inactive parent、pending child 和 frozen text；首轮重建实际合成 child、排除 text、忽略 inactive parent。

自动测试通过后，废弃当前未记录计划的真实 Run，使用新目录重新串行验收 PNG，再继续 PDF、图片版 PPTX 和混合原生对象 PPTX。验收继续记录视觉残影、组件层级、RAM/VRAM、临时磁盘和重型子进程退出情况。

## 完成判定

- 新真实 PNG 的首轮图具有真实父子关系，而不是所有节点均为独立 parent。
- `reconstructed.png` 与实际装配语义一致。
- OCR 文字区域没有由组件背景修复重新引入的浅灰残影。
- 既有 v1 Run 可安全读取；全部自动化测试通过。
- 未针对任何示例文件添加路径、类别或尺寸特例。
