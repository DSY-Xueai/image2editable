# Course

## 叶组件优先与过度合并门禁（本轮）

- 当前状态：组件计划按可独立移动的最小完整视觉单元处理；本轮仅完成规则与确定性质量门禁，未运行真实 OCR、SAM、模型重建或 `test1.pptx` 验收。
- 本轮变更：`absorb_into_parent` 仅允许同一物理实体的重复掩码、碎边、阴影或分割缺口；质量重算从当前哈希绑定输出图复核 active 组件覆盖的 inactive 源 masks，空间独立的多个叶簇会给 absorb/merge/collapse 及后续未改变组件持续加入硬失败 `over_merged_component`，只有不再覆盖多个独立源叶的新 mask 才能解除。
- 几何与资源：叶簇使用紧 bbox crop、预存面积和 bbox 快速跳过；宽容器不能传递桥接独立叶，显著且不相交的完整 masks 默认独立，gap 仅在明显更小且细长的碎边强证据下非传递聚合，偏移阴影与小缺口仍可按页面校准聚合。小页面不再受固定 20 像素噪声下限影响，mask 读取后复核完整祖先目录身份并拒绝 symlink/junction/reparse 替换。
- 执行来源约束：`record_component_execution` 要求 after 图中 newly inactive 与 retained inactive 来源保留 ID，且除 `state` 外的节点身份字段（含 kind、parent_id、mask 路径/hash/bbox）及实际绑定 bytes 与请求 input graph 一致；合法重新激活的 parent 仍可更新自身 mask，不新增状态 schema。
- 关键文件：`image2editable/component_quality.py`、`image2editable/component_repair.py`、`image2editable/local_agent_worker.py`、`skills/image-to-ppt/SKILL.md`、`skills/image-to-ppt/scripts/component_quality.py` 及对应测试。
- 运行入口保持不变：Host 使用 `image2editable agent next/record`，Local 使用 `--agent-provider local`；每页仍最多五批并复用既有父组件/页面降级流程。
- 注意：未实现计划中的后续 Task 2；此前真实文件验收记录属于旧门禁的历史结果，不能视为本轮叶组件规则的真实验收。
- 契约覆盖：已跟踪的 Local Agent 测试同时锁定 Local 提示词与 Host Skill 的同一物理实体、语义父级只分组不渲染规则。

## P2.3 Task 9 已完成

- 图片/PDF 页面已接入统一组件状态机：每页只初始化一次，Host 等待时安全退出，恢复时不重复初始分层；多页 PDF 严格串行。
- 最终组装只读取状态机验收后的背景、重建图、文字掩码、组件图和逐文件 SHA-256；组件 RGB 使用已去字的 `text-clean` 像素，alpha 保留完整组件掩码，避免文字框挖洞和原文字重影。
- `preserved_with_warning` 明确使用完整源图且不输出伪可编辑组件或文字，并写入降级警告。
- PPTX 先写同目录临时文件，实际重新打开并核对页数后再以 no-replace 方式发布；交付记录写入 `pptx_reopen=pass`。
- 组装 accepted assets 与 component masks 使用已验签 bytes 快照；多 variant 先全部 staging/reopen 通过，再 no-replace 发布，发布异常会清理本轮已发布目标。
- 本轮新增真实图片/PDF Host plan E2E：覆盖等待/恢复、真实动作执行与质量门禁、冻结/下一轮/父组件回退、同 gate version、单次最终组装、warning 和 no-overwrite/reopen。
- Task9 指定回归集已验证：`524 passed, 7 skipped`，已本地提交。

## P2.3 Task 10 已完成

- 截图型 PPTX 的获批页面先进入共同组件状态机；Host 等待期间不创建 donor、不发布 PPTX，也不会提前启动 OCR/CV worker。
- 获批的整页截图与图片/PDF 共用真实 OCR、CV 和语义父子组件初始化；Agent 在最多五轮内按每页证据决定拆分、修复、合并或回退，不再用单一整页父组件冒充语义分层。
- `ready_for_assembly` 只允许从已验签 `component_result.json` 读取组件图、掩码、重建图、文字掩码和文字对象；donor 组装不重新执行 CV/OCR/Agent。
- 每个冻结视觉节点生成独立图片对象；可靠 `text_items` 生成文字框。没有可靠文字时明确记录 `raster_text_preserved`，不删除原图文字或伪装成可编辑文字。
- result、graph、source、mask 和 accepted assets 均校验 Run 内相对路径、SHA-256、普通文件身份、硬链接/符号链接/重解析点与读取中变化；donor 组件、manifest 和 PPTX 使用 no-clobber 发布。
- `preserved_with_warning` 页面直接保留原截图，不会再次进入旧 CV donor；PPTX patch 失败只回退受影响页面，原生对象、备注、z-order 和其他页面保持不变。
- PPTX 初始化异常会写入失败状态并可恢复，不再把 Run 留在 `running`；第二轮及以后始终复用首轮绑定的 source 快照，避免 PNG 重编码导致 hash 漂移。
- 最终全量验证：`1270 passed, 18 skipped`；Ruff 与 `git diff --check` 通过，测试后无 visual/OCR/SAM worker 残留。

## P2.3 Task 11 已完成

- 新增版本化本地模型目录；首个 `Qwen/Qwen3-VL-2B-Instruct@main` 条目保持 `experimental`，真实 Local 验收前不标记 stable、不伪造固定 commit。
- `models recommend --json` 只读取本地目录并探测 CUDA、显存、内存和缓存所在磁盘，不访问网络、不创建模型缓存；结果包含兼容性、原因、预计空间和缓存路径。
- `models install agent` 在联网前显示模型、revision、实验性状态、空间和硬件结论；必须交互确认或显式传入 `--yes`，磁盘不足或未确认时下载调用次数为零。
- 下载使用 Hugging Face 本地 snapshot 缓存；完成后记录解析出的 commit SHA、逐文件大小/SHA-256 和 receipt。`models status` 只在本地复核 receipt、snapshot 边界与文件校验值。
- 本地 Agent 依赖位于 `agent-local` 可选依赖；CLI 和 Host Runtime 都惰性隔离模型管理模块，Host 转换不导入 PyTorch、不探测或下载本地模型。
- Task 11 只实现模型管理边界，没有下载模型；Local Provider 推理执行链已由后续 Task 12 接通。
- Task 11 指定回归：`94 passed, 1 skipped`；最终全量回归：`1297 passed, 19 skipped`。Windows 仅跳过当前环境无权限创建文件软链接的兼容测试，Hugging Face copy fallback 与其余 receipt 校验均通过。

## P2.3 Task 12 已完成

- Local Provider 已接入统一组件状态机：Runtime 在内部完成“生成请求 → 单轮本地视觉推理 → 严格记录计划 → 确定性执行/门禁”的循环，不进入 Host 的公开 `awaiting_agent`，也不读取 Host 握手、会话或计划。
- 初轮和后续轮次不再用源图冒充诊断证据：`numbered-masks.png` 显示彩色组件掩码与准确 ID，`ocr-overlay.png` 显示稳定 `text_XXXX`、逐项 OCR box/text，`ownership.png` 显示独占像素归属，`reconstructed.png` 与 `difference.png` 来自当前确定性重建；证据图改为分批保存释放，避免六张整页 RGB 图同时常驻内存。
- 每个重修轮次启动一个独立 `local_agent_worker` 子进程；视觉模型只在 worker 内惰性加载，结束或超时后以进程退出作为 RAM/VRAM 释放边界。模型与处理器均使用已确认 snapshot 和 `local_files_only=True`，同时启用 Hugging Face/Transformers 离线环境，不会在转换中下载模型。
- worker 只允许十二种既定动作；请求、八项证据和组件图重新校验路径与 SHA-256，生成计划先在 worker 校验，再由父进程使用与 Host 相同的严格 Schema/组件图校验器复核并原子记录。
- 本地计划按页面、轮次和请求 hash 持久化且不可覆盖；相同计划可恢复，不同计划拒绝。最多五个页面级批量重修轮次、冻结规则、父子互斥、父组件/原页回退和质量门禁继续由共同状态机控制，Local Agent 无权放宽。
- OCR 文字以冻结、只读的 text 节点进入真实组件图，不参与视觉组件质量计数；`attach_text` 只允许待修视觉节点引用冻结文字节点，其他动作不能修改文字节点。原始 `text_items` 继续进入最终可编辑 PPTX，而不是只画在诊断图上。
- 硬件与依赖推荐在一次性 `models recommend --json` 子进程中执行，避免 PyTorch/CUDA 常驻转换主进程；除总 RAM/VRAM 外也检查当前可用 RAM/VRAM。标称 16GB RAM/8GB VRAM 目标按至少 15 GiB 总 RAM、8 GiB 总 VRAM、6 GiB 可用 RAM、6.5 GiB 可用 VRAM 判定，避免 Windows 可见容量换算造成永久误拒绝。
- 每轮证据发布前、动作执行前和 Local worker 启动前都检查页面磁盘预算；按解码后整页像素、当前/计划新增节点、剩余轮次和至少 256 MiB 安全余量预留，空间不足时不创建下一轮 evidence 或 execution 目录。
- 首次 Local 执行将模型 ID、请求 revision、解析 commit、snapshot、文件清单及 receipt hash 冻结到 Run；恢复时若全局模型已变更则拒绝混用，Local PPTX 完成态可重复校验该摘要。Host 完成路径不读取、也不接受 Local provenance。
- worker 超时、非零退出或无效计划会通过受保护目录链和独占文件写入有界诊断；不安全诊断目录不能越界，也不会掩盖原始 worker 错误。不自动切换 Provider。
- 父组件在五轮修复后回退时从首轮 hash-bound 资产恢复完整原始掩码，而不是沿用被 expand/shrink 等动作修改后的父掩码。
- `README.md` 与 `skills/image-to-ppt/SKILL.md` 已同步 Host/Local 两种用法：Host 复用具备视觉能力的 Codex/Claude 等宿主，Local 必须先按实时硬件推荐并取得明确下载授权；两者在单个 Run 内互斥且不共享私有状态。
- 当前只完成 mock/契约验收，没有下载或真实加载模型，Local 状态继续为 `experimental`；`models status` 显示 `installed=false`。`attach_text` 契约定向回归为 `61 passed`，最终全量回归为 `1319 passed, 20 skipped`。

## P2.3 Task 13 已完成

- README、英文 README 和 PPTX Skill 已补充双 Provider、每页 5 轮、`preserved_with_warning`、敏感内容、逐图重新判断及透明图片组件边界；新增通用内容/输入类型验收清单契约，定向回归 `4 passed`。
- 真实 Host PNG 已推进到 R6：Agent 将表格、页脚和标题各自的重叠碎片通过 `absorb_into_parent` 合并为三个完整父组件；任意活动视觉掩码发生像素重叠时触发 `component_overlap`，阻止不安全冻结和最终组装失败。
- 新增 Agent 显式 `rebuild_background`：只在页面外缘颜色足够一致时，按页面短边比例扩张活动组件与文字联合掩码并重建画布；每页仍最多五个重修批次，后续轮次沿用已认证背景，不缓存跨图片语义判断。
- 质量重建和最终组件使用 Prepared Page v2 的 `text-clean` 像素；OCR 连通区域按至少 45% 覆盖率唯一归属给最匹配父组件，避免透明文字框、白块、浅灰残影和原文字重影。文字残影门使用稠密文字核心的局部中值底色与非文字环带双路径，不再把彩色标题条或组件底色误判为 ghost；整页距离变换和三通道中值滤波每页只执行一次，候选组件复用单一 `text_ink` 布尔图。
- PPTX donor 现在应用 OCR 字号、字体、中日韩字体映射、粗细、颜色和对齐；文字框零边距、垂直居中且不自动换行。组件 alpha 不再扣除矩形文字掩码，彩色标题条和卡片底色不会留下透明白洞。
- 真实 PNG、三页 PDF、两页图片版 `test1.pptx` 和原生/图片混合 `混合.pptx` 已完成 Host 验收；两种 PPTX 均由 PowerPoint COM 实际重开，混合文件未命中页面逐字节保持不变。最终全量回归为 `1372 passed, 20 skipped`，主脚本与 Skill 镜像 SHA-256 一致。

## 当前项目状态

- 当前分支：`codex/agent-runtime-foundation`；只保留本地提交，不推送、不合并 `main`。
- Unified Runtime 已支持图片/图片目录、PDF 和 PPTX 输入。
- 图片与 PDF 进入同一套 OCR、视觉分层、背景修复和 PPTX 组装流程。
- PPTX 先只读扫描原生对象；只有 Agent 高置信确认的整页截图候选进入重建，其余文字、形状、表格、图表、备注和未命中页面保持原生。
- P2.2 已接通：Agent 决策 → 串行 CV 重建 → OOXML 原位替换 → 结构校验 → 单页安全回退。
- P2.3 Task 10 已接通 PPTX 获批候选 → 组件状态机 → 已验收 donor → OOXML 原位替换；未通过质量门禁的页面保留原截图并给出 warning。
- P2.3 Task 13 的真实 PNG/PDF/PPTX Host 验收和全量回归已完成；Host 模式完全不依赖本地模型。真实验收暴露的父组件组装、残影门误报、文字样式丢失和透明白洞均已修复；Local 尚未下载实际模型或通过真实文件验收，因此仍为实验性。

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
- `image2editable agent record RUN_DIR --plan PLAN.json` 在首次写入和半提交恢复前均先校验当前请求 SHA；同时读取 Task 4 已认证组件图，严格校验动作对象数量与真实 kind/parent 角色。`attach_text` 只接受 visual→text，`collapse_to_parent/absorb_into_parent` 可操作候选子组件关联的完整 parent，child merge 只能同父级；过期哈希、错误 Provider/轮次/页面、未知或冻结对象、跨角色/跨父级及冲突动作均拒绝。
- Agent next/record 与执行、恢复共用同一把 Run OS lease；`next` 最多有界等待 30 秒并在单一临界区内读取或发布 challenge，跨平台并发调用只会加载同一个完整结果，超时明确失败；`record` 仍非阻塞拒绝并发。计划以临时文件加排他链接原子发布，重复或并发记录不能覆盖。若计划已发布但状态切换中断，仅同一份且重新严格验证通过的计划可补完 `awaiting_agent → prepared`，不同计划和已恢复后的重复提交仍拒绝。

## P2.3 Task 6 本轮变更

- 新增十二类严格组件动作执行：接受、丢弃冗余候选、合并、真实连通域拆分、按页面短边比例扩张/收缩、SAM 框/点提示重试、文字归属、折叠到父组件、吸收碎片到父组件以及显式重建背景；执行器只做确定性变换，不自行通过质量门禁。动作执行后从真实输出图刷新下一轮候选 ID，merge/split/discard/absorb 不再把已停用的旧 ID 带入质量门。
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
- 文字重影检测会排除贯穿 OCR 框两侧的细长结构线，避免把表格/矩阵边界误判为残字；稀疏低方差文字改用 2%–98% 局部对比检测，并按文字色与背景色距离自适应约束清理范围。
- 背景重建会清理认证组件图内包括 `inactive/discarded` 在内的全部视觉遮罩；视觉 margin 只扩张组件遮罩，不扩张 OCR 文字框，避免丢弃对象残留灰尾或标题清理吞掉邻近副标题。多轮 `absorb_into_parent` 只允许恢复同一认证图中的 inactive 父组件作为首个目标。

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
- 用户与 Agent 使用说明：`README.md`、`skills/image-to-ppt/SKILL.md`
- 本地模型目录、管理与隔离推理：`image2editable/model_catalog.json`、`image2editable/models.py`、`image2editable/local_agent.py`、`image2editable/local_agent_worker.py`
- CLI 与可选依赖：`image2editable/cli.py`、`pyproject.toml`

## 运行入口

```bash
image2editable doctor
image2editable models recommend --json
image2editable models status
image2editable models install agent
image2editable convert input.pdf -o output.pptx --slide-size original --agent-provider host
image2editable convert input.pdf -o output.pptx --slide-size original --agent-provider local
image2editable convert images/ -o output.pptx --slide-size 16:9
image2editable prepare input.pptx --run-dir runs/pptx-job
image2editable agent next runs/pptx-job
image2editable agent record runs/pptx-job --plan plan.json
image2editable run next runs/pptx-job
image2editable decision record runs/pptx-job --page page_001 --object background --decision replace --confidence 0.99 --category full_slide_screenshot --evidence "complete slide layout"
image2editable run execute runs/pptx-job
```

## 真实文件验收

- `test1.pptx`：两页都通过真实共享 OCR/CV 初始化、Host Agent 语义重组、质量门禁和 OOXML 原位替换。第 1 页由 32 个候选重组为 3 个完整父组件并在第 1 批通过；第 2 页由 21 个候选重组为 3 个父组件，在修正彩色标题条的文字残影误报后于第 3 批通过。
  - 验收输出：`tmp/task13-host-test1-style-qa-r1/output.pptx`；第 1 页为 3 个图片组件与 35 个可编辑文字框，第 2 页为 3 个图片组件与 26 个可编辑文字框。
  - PowerPoint COM 重开和 1600×900 PNG 导出通过；白色 OCR 方块、浅灰文字栅格残影、组件缺失、Graph RAG 横向拥挤和底部总结裁切均未再出现。OCR 原始识别仍可能造成空格与个别粗细差异，后续按文字样式精修处理，不回退为栅格文字。
  - 资源策略保持单页串行；第 1 页冷启动约 331 秒，页间短时进程交接峰值约 2.9 GiB，随后降至约 1.9 GiB，第 2 页复用热缓存后仅需数秒。资源峰值尚未完全解决，不能并行启动多个重型页面。
- `混合.pptx`：当前代码复验为 3 页、78 个原生对象、0 个合格整页截图候选；输出与源文件逐字节相同，PowerPoint COM 重开为 3 页、960×540。
  - 输出：`tmp/task13-host-mixed-r4/final/output.pptx`
  - 源与输出 SHA-256 均为 `bb7b11d24f9db74f0a31a52809bfbaa46ca275f4a49085a3e0a1fbe8668ecc0d`。
- `research_layout_demo_3pages.pdf` 已完成 3 页真实 Host Agent 分层、语义父组件重组、背景重建、质量门禁、最终组装及 PowerPoint 原生重开/PNG 渲染验收；副标题、表格/科研图、流程卡完整，无浅灰栅格残影、白色 OCR 方块、组件重影或右侧灰尾。
  - 输出：`tmp/task13-host-pdf-r3/final/output_original.pptx`、`tmp/task13-host-pdf-r3/final/output_16x9.pptx`。
  - 原生对象统计：第 1 页 25 个对象（11 图片组件、14 文字框），第 2 页 11 个对象（7 图片组件、4 文字框），第 3 页 29 个对象（6 图片组件、23 文字框）。
  - 资源策略保持 `heavy_page_concurrency=1`；重型进程实测峰值约 2.26 GiB，未并行处理其他文件，结束后无 Python 转换进程残留。
- 验收记录：`tmp/task10-real-acceptance-20260802-v3/acceptance_summary.json`；目录约 85.7 MB，结束后无视觉、OCR 或 SAM 子进程残留。

## 当前注意事项

- P2.3 通用组件 Agent 重建设计已确认并写入 `docs/superpowers/specs/2026-07-31-component-agent-reconstruction-design.md`；Task 1–13 已完成统一运行时、最多五轮重修、最终组装、截图型 PPTX 原生对象保护、本地模型管理、Local Provider 隔离推理、双 Provider 文档及真实 PNG/PDF/PPTX 验收。
- 本地模型目录目前仍为 `revision=main`、`stability=experimental`；Task 12 没有下载模型。只有取得用户明确下载授权并完成 Task 13 的真实图片、PDF、图片版 PPTX 验收后，才固定验收 commit 并调整稳定性。
- Agent 只自动执行 `replace + full_slide_screenshot + confidence >= 0.92`；不确定候选继续保留。
- 每页最多记录一个自动替换决策；旧运行若存在同页双批准，会按单页 `preserved_with_warning` 回退。
- 普通与 Agent 转换均不再把整页清空组件作为成功结果；不稳定组件按组件失败，由 Task 8 重修或折叠到完整父组件。
- OCR 未识别的符号会完整保留在底图中，而不是冒险生成错误文字对象。
- 实测 PDF 重型页约 2.2 GiB，截图型 PPTX 冷启动页间交接峰值约 2.9 GiB；当前必须保持 `heavy_page_concurrency=1`，资源峰值仍是下一阶段的优化项。
- 主脚本与 `skills/image-to-ppt/scripts/` 镜像必须保持 SHA-256 一致。
