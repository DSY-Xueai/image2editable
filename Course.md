# Course

## 当前项目状态

- 当前分支：`codex/agent-runtime-foundation`；只保留本地提交，不推送、不合并 `main`。
- Unified Runtime 已支持图片/图片目录、PDF 和 PPTX 输入。
- 图片与 PDF 进入同一套 OCR、视觉分层、背景修复和 PPTX 组装流程。
- PPTX 先只读扫描原生对象；只有 Agent 高置信确认的整页截图候选进入重建，其余文字、形状、表格、图表、备注和未命中页面保持原生。
- P2.2 已接通：Agent 决策 → 串行 CV 重建 → OOXML 原位替换 → 结构校验 → 单页安全回退。
- P2.3 Task 8 已实现：Host 计划进入每页最多五轮的组件重修状态机，通过组件立即冻结；提前停止或五轮仍失败时折叠到完整父组件。

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
- 组件掩码与可编辑文字区域发生实质重叠时按组件报告失败，不再清空整页组件或把 text-only fallback 当作成功结果。
- 资源策略保持 `safe-default`：重型页面串行、数值线程最多 8、SAM `points_per_batch=1`。

## P2.3 Task 1 本轮变更

- `convert` 与 `prepare` 支持 `--agent-provider host|local`，默认 `host`；Provider 在创建 Run 时写入清单，后续 CLI 子命令不能覆盖。
- Run/Page 状态机增加 `awaiting_agent` 暂停状态，仅允许 `running → awaiting_agent → prepared` 和 `processing → awaiting_agent → processing` 的新增路径。

## P2.3 Task 2 本轮变更

- `prepare_component_layers` 原子生成带逐文件 SHA-256 的可恢复资产；isolated text-clean 写盘后会在启动 visual worker 前释放大数组，OCR/visual/gc cleanup 不遮蔽主异常。
- `load_component_layers` 只读校验已存在的工作目录，不会为缺失 state 创建目录；state 与资产继续使用单句柄 bytes、sidecar/hash、路径身份和链接属性校验。
- `finalize_component_layers` 只接受与 fresh state 完全一致的 components 与 element masks；quality 单次加载 staging source/masks 并执行严格 overlap 检查，成功返回的组件继续由存活 staging 承载，失败则完整清理。
- 普通 `convert`/`convert_batch`/variants 入口继续沿用最终质量检查，但不再以 text-only fallback 清空整页组件作为成功结果。

## P2.3 Task 3 本轮变更

- 组件节点固定为 `id/kind/parent_id/state/mask/mask_sha256/bbox/z_index/text_ids`；未知字段拒绝，冻结节点的掩码、位置、层级和文字归属不可修改或删除。
- `pending/frozen` 是仅有的活动渲染状态，`failed/inactive` 不参与输出；父子节点不能同时渲染；活动节点 `z_index` 唯一并按其确定导出顺序。
- 初始语义实例同时保存完整父掩码和可拆子掩码；常规导出只消费活动节点，完整父资产留作后续折叠回退。
- 唯一像素所有权分别报告组件重复、显式前景缺失、文字重复和越界像素；半透明、阴影和抗锯齿的任意非零证据也只能归属于一个活动组件，不自动修正。
- graph 声明的每个 mask 都会以同一文件句柄校验 SHA-256，并重算 bbox；缺失、额外、符号链接、重解析点和硬链接资产都会拒绝。
- 组件树与组件 PNG 先写入同级 staging，全部通过后整目录发布；失败完整清理，已有输出目录不会被逐文件覆盖。
- 原始组件与文字交叠不再提前误杀；只有显式提供可靠 `text_items` 和已验证 `text_clean_image` 才进入清字导出，避免整块 OCR 框擦断线条；missing 仅依据显式非文字前景证据判定。
- mask 校验、union 和像素所有权改为流式 bool 累计，文字洞修复只保留一张 owner map，无文字时不分配 repair map；不再构造随组件数增长的 `N×H×W` 堆叠或全页 repair 列表。Skill 同步携带组件契约与质量模块，可脱离仓库导入。

## P2.3 Task 4 本轮变更

- 每轮固定发布 `source.png`、编号掩码、OCR/所有权叠加图、重建图、差异图、组件图和质量报告八项证据；请求逐项记录 SHA-256，并绑定源图、组件图、Provider、页面、轮次及待修/冻结组件 ID。
- 轮目录固定为 `reconstruction/agent/round-01` 至 `round-05`；先写唯一 staging 再整目录发布，已有轮次不可覆盖，同页同轮并发发布只有一个成功。
- 构建和读取均限制在当前页面 reconstruction 内；证据采用同一文件句柄校验身份并读取，拒绝路径穿越、跨页、符号链接、重解析点、硬链接和读取中变化，任何证据篡改都会在 Agent 调用前失败。
- Loader 进一步强制 `pages/<page_id>/reconstruction/agent/round-XX` 完整目录拓扑；签名 marker 与请求及八项证据在同一 staging 中一次发布，写入或 rename 中断不会留下 round，可安全重试。
- Run 根目录原子创建并复用单一 32 字节 integrity key；marker 以 HMAC-SHA256 绑定请求原始 bytes、Provider、页面和轮次，key 缺失、损坏、链接或验签失败均 fail closed，不自动轮换。
- 文件读写会在打开前快照从 `pages` 到目标父目录的完整身份，并在打开后及 I/O 后复核；父目录在检查与打开之间被替换时拒绝结果，marker 写入也使用相同保护。
- 威胁边界：防止 Run/reconstruction 内请求、证据和 marker 被同步改写后伪装为合法发布；若同一 OS 账户已失陷并主动读取 integrity key，则不属于 Task 4 的防护范围。
- 证据构建和复核使用 1 MiB 分块复制/增量 SHA-256，普通图片不设内容大小上限且不会八项同时常驻；仅结构化 JSON 限制为组件图 16 MiB、请求 4 MiB、marker 64 KiB，超限在解析前失败。
- Run 已有任意发布轮次时，integrity key 缺失或损坏会拒绝继续构建，禁止自动生成新 key 使旧轮失效；固定 `pages/<page>/reconstruction/agent/round-*` 扫描同样拒绝链接与重解析点。
- 同一 Run 的证据发布由安全的跨进程 OS lease 串行化，锁覆盖 key 生命周期、签名和 round 最终确认；这是控制资源峰值并防止在途发布与 key 恢复交错的既定策略。
- 失败 staging 或身份不匹配的 round 先原子移入唯一 quarantine 释放固定名称；可证明属于本次 staging 的已知平面文件逐项校验清理，未知替代目录不递归删除并保留隔离，后续轮次仍可安全重试。

## P2.3 Task 5 本轮变更

- `image2editable agent next RUN_DIR` 在真实页面前先返回 Run 内随机、SHA-256 绑定的轻量视觉 challenge；从安全随机源独立选择严格白名单内的形状、颜色和数量，三角、圆和方形均使用等宽高边界。Host 路径不导入本地模型模块、不下载模型。
- challenge metadata 仅保存 Schema、图片路径、PNG 哈希和 challenge ID，不保存答案、nonce 或可复现布局的随机状态。形状、颜色和数量只在生成时从安全随机源选择；另有 128-bit 高熵视觉盐只写入 PNG 底部保留像素，不单独持久化且被答案观察器明确忽略，使公开 36 种无盐模板无法按 SHA 枚举。验证端先用 Task 4 Run integrity key 认证 PNG 哈希/ID，先拒绝非 240×120 图片，再独立解析已绑定 PNG 像素得到答案。即使同权限 Agent 读取 metadata 与 key，也不能脱离视觉图像推导答案；已有 challenge 时 key 缺失、损坏或被替换均 fail closed。PNG 与 metadata 在唯一 staging 目录内完整验证后整目录发布，写入/rename 故障清理自有 staging 后可重试，并发 next 复用同一完整发布。
- capability response 必须严格匹配当前 Run challenge 的形状、颜色和数量，成功后原子记录 challenge ID、图片哈希和能力集合；图片、OCR 与诊断内容均明确视为不可信数据，不能覆盖 Schema、用户请求或质量门禁。
- 握手通过后只使用 Task 4 的 HMAC 安全 loader 读取当前组件请求，并返回绝对请求/证据路径；请求继续绑定 Provider、页面、轮次、请求哈希和当前组件 ID。
- `image2editable agent record RUN_DIR --plan PLAN.json` 在首次写入和半提交恢复前均先校验当前请求 SHA；同时读取 Task 4 已认证组件图，严格校验动作对象数量与真实 kind/parent 角色。`attach_text` 只接受 visual→text，`collapse_to_parent` 只接受 parent，child merge 只能同父级；过期哈希、错误 Provider/轮次/页面、未知或冻结对象、跨角色/跨父级及冲突动作均拒绝。
- Agent next/record 与执行、恢复共用同一把 Run OS lease；`next` 最多有界等待 30 秒并在单一临界区内读取或发布 challenge，跨平台并发调用只会加载同一个完整结果，超时明确失败；`record` 仍非阻塞拒绝并发。计划以临时文件加排他链接原子发布，重复或并发记录不能覆盖。若计划已发布但状态切换中断，仅同一份且重新严格验证通过的计划可补完 `awaiting_agent → prepared`，不同计划和已恢复后的重复提交仍拒绝。

## P2.3 Task 6 本轮变更

- 新增九类严格组件动作执行：接受、合并、真实连通域拆分、按页面短边比例扩张/收缩、SAM 框/点提示重试、文字归属以及折叠到父组件；执行器只做确定性变换，不自行通过质量门禁。
- `accept` 只把 `pending` 转为仍参与渲染的 `pending_gate`；后续质量门禁决定 `frozen/failed`。冻结节点的结构与掩码保持不变，合并源和折叠后的后代转为 `inactive`。
- 每轮写入新的组件图与逐文件哈希掩码目录，以原子 no-replace 方式发布，已存在输出不覆盖；失败不发布正式轮次，异常 staging 不再按可替换路径移动或递归删除。SAM 提示通过最长 600 秒的独立 worker 子进程执行，并校验返回掩码尺寸与非空性。
- 主脚本与 Skill 镜像已同步 `component_contracts.py`、`visual_segment.py`、`fg_extract.py` 和 `sam_worker.py`。

## P2.3 Task 7 本轮变更

- 新增页面自适应校准与组件级质量报告，按源图、清理背景、重建图、已认证组件掩码和文字掩码检测缺失、重复、边缘、阴影、透明边缘、文字重影、像素归属及相对上轮改进；噪声和 Agent 置信度不能放宽硬缺陷。
- 文字重影只归因到与组件及其 halo 相交的文字区域；掩码外残留只有与组件边界直接连通且归属唯一时才归入该组件，多组件歧义证据作为页面级 `orphan_residual` 硬失败。
- 每轮只计算一次全页 RGB、差异图、亮度图和外环归属计数，逐组件评估复用只读上下文，避免组件数增加时重复分配整页中间数组。
- 页面视觉差异只作为总门禁，不能触发 `components=[]` 或 text-only 成功；组件/文字重叠改为组件级失败报告，组件继续保留给后续重修。
- `protected_native_overlap` 与 `pptx_reopen` 采用严格 `pass/fail/unknown`，缺失或 `unknown` 均失败；Task 8/最终组装负责接入真实检查事实，本阶段不伪造通过结果。
- 质量门禁读取组件图声明的掩码并复核哈希、目录身份和链接属性，拒绝越界、`..` 语义路径、链接祖先、读取中替换及组件 ID/数量不一致。

## P2.3 Task 8 本轮变更

- 每页按完整失败候选批次最多重修五轮；空计划、连续相同计划或零个可执行动作会提前停止，绝不创建第六轮。
- 通过质量门禁的组件立即写入不可变冻结图并从后续候选中移除；执行数量必须与计划动作及组件图实际变化一致。执行产物先绑定背景、重建图、文字掩码和原生对象重叠检查的逐文件哈希，质量阶段只读取这些状态引用并内部重算 Task 7 组件指标与页面视觉差异，不接受 Agent 自报指标。
- 重修停止后只使用初始请求中逐文件哈希认证的完整父组件资源；父组件通过门禁时状态为 `ready_for_assembly/parent_preserved`，失败时为 `preserved_with_warning`。
- 每次推进只提交一个持久化边界，先写产物再写带 SHA-256 引用的状态；恢复只读取状态引用，忽略临时目录和未引用轮次，并使用 Run 级执行租约防止重复领取。
- Task 8 终态保留 `delivery_checks.pptx_reopen=unknown`，真实 PPTX 组装与重新打开检查由后续 Task 9/10 完成；缺少组件状态时只返回 `needs_initialization`。

## 关键修改文件

- Agent 契约、Host 握手、证据、组件质量与运行时：`image2editable/component_contracts.py`、`image2editable/host_agent.py`、`image2editable/component_repair.py`、`image2editable/component_quality.py`、`image2editable/agent.py`、`image2editable/runtime.py`
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
image2editable agent next runs/pptx-job
image2editable agent record runs/pptx-job --plan plan.json
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

- P2.3 通用组件 Agent 重建设计已确认并写入 `docs/superpowers/specs/2026-07-31-component-agent-reconstruction-design.md`；Task 1–8 已完成 Provider、可恢复视觉资产、组件树/所有权、五轮证据包、Host 握手/严格计划、确定性动作执行、组件级质量门禁和最多五轮的修复状态机。Task 9 尚需把初始分层和逐轮执行接入统一运行时。
- Agent 只自动执行 `replace + full_slide_screenshot + confidence >= 0.92`；不确定候选继续保留。
- 每页最多记录一个自动替换决策；旧运行若存在同页双批准，会按单页 `preserved_with_warning` 回退。
- 普通与 Agent 转换均不再把整页清空组件作为成功结果；不稳定组件按组件失败，由 Task 8 重修或折叠到完整父组件。
- OCR 未识别的符号会完整保留在底图中，而不是冒险生成错误文字对象。
- 实测单页 PDF 转换的 Python 工作集合计约 2.2 GiB；`test1.pptx` 两页运行目录约 35.6 MiB，未出现内存或磁盘 100%。
- 主脚本与 `skills/image-to-ppt/scripts/` 镜像必须保持 SHA-256 一致。
