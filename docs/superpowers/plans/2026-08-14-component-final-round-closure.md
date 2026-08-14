# 组件最终轮闭环实施计划

**目标：** 在不降低模型、候选或质量门禁的前提下，消除最后一轮协议死锁、错误 OCR 背景残留、误判无进展和重试路径冲突。

## Task 1：动作协议

- [x] 用 RED 锁定唯一合法组合 `accept` → `absorb_residual`。
- [x] 在计划验证器和执行器实现相同窄规则。
- [x] 保持 contained-pair 证据与 signed residual 分区语义。
- [x] 同步 standalone 镜像。

## Task 2：Host PPTX 路由文档

- [x] 锁定 `prepare` → `run next` / `decision record` 循环 → `run next` 返回 null → `run execute`。
- [x] 锁定后续 `agent next` / `agent record` → `run execute`。
- [x] 测试剥离 shell 注释，只验证真实命令顺序。

## Task 3：accept 重叠保护

- [x] RED 复现不透明补全侵入其他 active visual。
- [x] 仅从 completion delta 中扣除其他活动 mask，保留原 mask。
- [x] 运行组件修复回归并同步镜像。

## Task 4：可恢复计划拒绝

- [x] 用专用异常区分无关 residual target 与普通执行失败。
- [x] 在同一 lease 中写 durable rejection pointer 并精确回滚同轮状态。
- [x] 为 Host 和 Local Agent 提供同一纠错上下文。
- [x] 覆盖崩溃、半提交、幂等、旧文件不覆盖和普通错误传播。

## Task 5：真实转换暴露的质量缺陷

- [x] RED 复现新增可接受组件仍被误判无进展。
- [x] 把新 accepted component ID 纳入质量进度判定。
- [x] RED 复现 `suppress_text` 仅停用错误 OCR、未生成视觉组件。
- [x] 使用现有同质量 SAM bbox 批处理重建独立 visual。
- [x] RED 复现固定 `parent-fallback` 目录阻断重试。
- [x] 改为唯一 run-owned fallback 目录且不覆盖旧内容。

## Task 6：真实 `test1.pptx` 验收

- [x] 从原始 PPTX 建立干净 Host run，完成两页全部路由和组件轮次。
- [x] 最终状态：2 页 replaced，0 warning，0 preserved，0 pending。
- [x] 第 2 页错误 OCR“自”被移除并重建为独立剪贴板 visual。
- [x] 渲染最终 PPTX，运行 `slides_test.py`，确认无溢出。
- [x] 用 `python-pptx` 检查 shape/picture/text，确认无整页图片。
- [x] 运行最终相关回归、全量 pytest 和镜像校验；`git diff --check` 在提交前执行。
- [x] 更新 `Course.md`，完成代码审查、`git diff --check` 并提交中文 commit。
