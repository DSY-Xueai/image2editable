# 组件最终轮闭环设计

## 背景

真实 PPTX 转换暴露了四个会破坏“高质量可编辑重建”的闭环问题：

1. 同一视觉对象需要先用 `accept` 提交独立可编辑性证据，再用 `absorb_residual` 吸收已绑定残差；旧协议把这两个动作误判为冲突。
2. `accept` 的不透明区域补全可能侵入其他活动组件，制造新的重叠。
3. 无关 residual target 属于可纠正的 Agent 计划错误；旧 runtime 会把整个转换标为失败。
4. OCR 可能把图标识别成文字。仅停用文字会把图标留在背景中，不能形成独立可编辑对象。

## 目标与边界

- 只允许同一 pending visual 按顺序执行一次 `accept`，随后执行一次 `absorb_residual`；反向顺序、重复动作和其他动作堆叠继续拒绝。
- residual 仍只来自当前签名请求绑定的 `unexplained-mask.png`，并继续受包含、3px 邻接和分区约束。每个 `absorb_residual` target 必须获得至少一个合法分区；与所有 absorb target 无关的区域保留给同轮 retry 或后续质量门禁，不得强制归属。
- `accept` 保留原 mask，只裁掉补全增量中属于其他 pending、pending_gate 或 frozen visual 的像素。
- 只有在 graph mutation 和发布前发现的无关 residual target 才可恢复到同一轮；SAM、I/O、hash、资产和发布异常继续失败。
- Host 与 Local Agent 都必须收到固定纠错说明和被拒计划；request、evidence、hash、repair round 和 `plan_count` 不变。
- `suppress_text` 只用于有证据证明为非文字的 OCR 节点。执行器以该 OCR bbox 调用现有同质量 SAM 批处理，创建新的 pending visual 组件；不得把对象留在背景中。
- 不降低模型、候选、阈值或质量门禁，不为测试文件增加专用规则。

## 实现

### 动作协议

`component_contracts.validate_component_plan` 与 `scripts.visual_segment.execute_component_actions` 使用相同的窄规则：记录每个对象已出现的动作，只有历史为 `accept` 且当前为 `absorb_residual` 时允许第二次触碰。动作仍按计划顺序执行。

### 可恢复计划拒绝

执行器对没有获得任何合法 residual 分区的 target 抛出专用 `RecoverableComponentPlanError`。runtime 在同一 `ExecutionLease` 内先按下一 component-state revision 写入不可覆盖、hash-bound 的 rejection record，再把状态从 `plan_recorded` 精确回滚到 `awaiting_plan`。共享 loader 由当前 revision 精确定位记录，并验证 page、round、request、被拒计划、摘要和 graph；不使用 glob 或 mtime 推断。连续拒绝保留全部旧记录。Host `agent next` 与 Local prompt 只在该专用状态附加 `correction_context`。

### OCR 图标重建

`suppress_text` 与其他 SAM retry 一样在任何 graph mutation 前收集 prompt、整批推理并严格验证数量、顺序、类型、shape 和非空 mask。成功后原 text 节点转为 inactive，并以新 ID 发布独立 visual 节点。批处理不可用时仍走原等质量隔离 worker，不使用轻量模型。

### 质量进度

页面质量进度除原页面指标外，还把“新增通过质量门禁的组件 ID”视为严格进展。这样不会因为 residual 短暂上升而提前触发 `no_quality_improvement` 回退。

### 重试发布

父组件 fallback 每次使用 run-owned 的唯一目录。旧的部分目录保持不可覆盖，不再因固定 `parent-fallback` 路径冲突而阻断重试。

## 验收

- 合法 `accept` → `absorb_residual` 通过；其他冲突组合失败。
- contained parent 证据与 residual 合并可在同轮共存。
- `accept` 不新增跨活动组件重叠，原 mask 像素不丢失。
- 无关 residual 计划返回同轮且 Host/Local 都获得纠错上下文；普通执行错误继续传播。
- 错误 OCR 图标经 `suppress_text` 生成独立视觉组件，不作为背景残留。
- 新组件通过质量门禁时继续下一轮，不误触发无进展回退。
- 父组件 fallback 遇到旧目录时创建新目录且不覆盖旧内容。
- root 与 standalone 镜像字节一致，相关回归和全量测试通过。
- 真实 `test1.pptx` 两页均为 `replaced`，无 warning、无 preserved；最终 PPTX 可重新打开、无溢出、无整页语义截图对象。
