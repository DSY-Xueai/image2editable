# Course

## 当前项目状态

- 当前分支：`codex/agent-runtime-foundation`；只保留本地提交，不推送、不合并 `main`。
- Unified Runtime 已支持图片/图片目录、PDF 和 PPTX 输入。
- 图片与 PDF 进入同一套 OCR、视觉分层、背景修复和 PPTX 组装流程。
- PPTX 先只读扫描原生对象；只有 Agent 高置信确认的整页截图候选进入重建，其余文字、形状、表格、图表、备注和未命中页面保持原生。
- P2.2 已接通：Agent 决策 → 串行 CV 重建 → OOXML 原位替换 → 结构校验 → 单页安全回退。
- P2.3 Task 2 已完成：初始 OCR/CV/SAM 分层可持久化并跨进程恢复，Agent 管理路径拥有禁止整页 text-only fallback 的最终质量入口；组件动作与 Agent 状态机尚未接入。

## P2.2 既有行为

- 支持替换两类整页截图：幻灯片背景图片、铺满页面的普通图片形状。
- 普通图片形状按原始 `x/y/cx/cy` 矩形映射并保持原 z-order；被连接线或动画引用的图片安全回退，不留下悬空 shape ID。
- OOXML 以原 PPTX 为底稿，只移除命中的截图对象并导入重建对象；保持页数、页面尺寸、备注、其他页面和受保护原生对象不变。
- 单页重建失败时状态为 `preserved_with_warning`，不影响其他页面，输出不覆盖源文件或已有文件。
- `reconstruction` 工作目录固定在对应页面目录内，计划层和执行层都会拒绝符号链接、目录联接及越界路径。
- PPTX 失败重试或中断恢复会先安全清理旧 donor 和重建清单，避免旧产物让新一轮误回退。
- 文字清理扩大到抗锯齿/模糊边缘，并保护邻近表格线、卡片边框和长图形线。
- 彩色、渐变和浅色底采用局部插值；纯色底采用局部颜色平面，修改严格限制在清理掩码内。
- 多色 OCR 文本框可同时清理彩色前缀和黑色正文；低置信、短小、近方形 OCR 候选按图标保留。
- 图标候选被过滤时会同步从 OCR 掩码中扣除，避免图标仍被文字清理流程擦除。
- 低对比度浅灰源对象使用更敏感的残影检测阈值，不再只检测深色文字或组件。
- 组件掩码与可编辑文字区域发生实质重叠时，自动降级为“干净底图 + 可编辑文字”，避免透明组件和底图留下浅灰栅格残影。
- 资源策略保持 `safe-default`：重型页面串行、数值线程最多 8、SAM `points_per_batch=1`。

## P2.3 Task 1 本轮变更

- `convert` 与 `prepare` 支持 `--agent-provider host|local`，默认 `host`；Provider 在创建 Run 时写入清单，后续 CLI 子命令不能覆盖。
- Run/Page 状态机增加 `awaiting_agent` 暂停状态，仅允许 `running → awaiting_agent → prepared` 和 `processing → awaiting_agent → processing` 的新增路径。

## P2.3 Task 2 本轮变更

- `prepare_component_layers` 将源图、OCR mask、text-clean、element masks、component RGBA 与背景/差异资产写入页面工作目录，并原子生成带相对路径和逐文件 SHA-256 的 `prepared_page.json`；sidecar 直接哈希同一份待发布 JSON bytes，不回读已发布路径。
- `load_component_layers` 严格校验 sidecar、schema、路径所有权、链接/reparse/hardlink 与资产哈希；state 由单一句柄读取，前后 `fstat`/`lstat` 校验身份、大小和 mtime，哈希与 JSON 解析只消费同一份 bytes。
- `finalize_component_layers` 只接受与 fresh prepared state 完全一致的当前 components 与 element masks；质量检查前用 exclusive/no-follow 句柄从已校验 bytes 建立 staging 快照，并核对 fd/path 身份与单链接属性；质量流程不再打开原资产路径，要求 fallback 时仍明确失败且不清空初始组件。
- 普通 `convert`/`convert_batch`/variants 入口仍沿用既有最终质量与 text-only fallback 行为。

## 关键修改文件

- Agent 契约与运行时：`image2editable/component_contracts.py`、`image2editable/agent.py`、`image2editable/runtime.py`
- PPTX 扫描与执行：`image2editable/pptx_input.py`
- CV 重建：`image2editable/pptx_reconstruct.py`
- OOXML 替换：`image2editable/pptx_shadow.py`
- 串行替换与回退：`image2editable/pptx_shadow_run.py`
- 共享图片/PDF 清理：`image_to_ppt.py`
- 背景、前景与质量门禁：`scripts/bg_model.py`、`scripts/fg_extract.py`、`scripts/visual_segment.py`
- Skill 镜像：`skills/image-to-ppt/scripts/`

## 运行入口

```bash
image2editable doctor
image2editable convert input.pdf -o output.pptx --slide-size original --agent-provider host
image2editable convert images/ -o output.pptx --slide-size 16:9
image2editable prepare input.pptx --run-dir runs/pptx-job
image2editable run next runs/pptx-job
image2editable decision record runs/pptx-job --page page_001 --object background --decision replace --confidence 0.99 --category full_slide_screenshot --evidence "complete slide layout"
image2editable run execute runs/pptx-job
```

## 真实文件验收

- `test1.pptx`：两页均由 Agent 高置信替换，无回退、无告警。
  - 第 1 页：1 张干净底图 + 35 个可编辑文字框。
  - 第 2 页：1 张干净底图 + 26 个可编辑文字框。
  - 输出：`tmp/p2-agent-test1-v2/final/output.pptx`
- `1-Embedding与向量数据库.pdf` 第 2 页单页验收：
  - 1 张干净底图 + 18 个可编辑文字框；27 个与文字冲突的浅灰组件被自动降级移除。
  - 输出：`tmp/p2-agent-pdf-page2-v3/final/accepted.pptx`
- 两份输出均可重新打开、无文字溢出；逐页渲染未见浅灰栅格残影或重复文字。
- `test1.pptx` 源文件 SHA-256 仍为 `03415ac5973a91e5b0d462a796690f618267ff1c05b4eb00d5f7ab20fa92ae80`。

## 当前注意事项

- P2.3 通用组件 Agent 重建设计已确认并写入 `docs/superpowers/specs/2026-07-31-component-agent-reconstruction-design.md`；Task 1/2 已完成 Provider 契约与可恢复初始视觉资产，尚未把这些资产接入组件决策、修复、五轮循环或 Host/Local 调用。
- Agent 只自动执行 `replace + full_slide_screenshot + confidence >= 0.92`；不确定候选继续保留。
- 每页最多记录一个自动替换决策；旧运行若存在同页双批准，会按单页 `preserved_with_warning` 回退。
- 当前优先保证视觉稳定和文字可编辑；与文字冲突的独立视觉组件会保留在净化底图中，不强行拆成透明对象。
- OCR 未识别的符号会完整保留在底图中，而不是冒险生成错误文字对象。
- 实测单页 PDF 转换的 Python 工作集合计约 2.2 GiB；`test1.pptx` 两页运行目录约 35.6 MiB，未出现内存或磁盘 100%。
- 主脚本与 `skills/image-to-ppt/scripts/` 镜像必须保持 SHA-256 一致。
