# 项目接手说明

## 当前状态

- 项目提供图片、PDF、图片版 PPTX 与混合 PPTX 到分层可编辑 PowerPoint 的统一 Runtime。
- 图片、PDF、图片版 PPTX 和获批的混合 PPTX 整页截图共用组件重建与硬质量门禁；混合 PPTX 中未命中的原生文字、形状、表格、图表、备注、z-order 和图片保持不变。
- Agent Provider 在 Run 创建时冻结为 `host` 或 `local`。两者共享相同的严格请求/证据契约、十三类动作、确定性执行、最多五批修复和质量门禁；Host 不读取本地模型状态，Local 不读取 Host 握手状态。
- 本地模型只在用户明确安装后离线使用；转换过程不下载模型。模型权重可缓存，但图片语义、组件计划和证据不得跨图片、跨页或跨批复用。
- 每页以可独立移动的最小完整视觉单元为组件。已通过组件冻结；五批后完整父组件仍失败时仅该页进入 `preserved_with_warning`，不发布 donor，不允许通过清空组件或保留栅格文字伪装成功。
- 全页 OCR 漏检时，prepare 会对未被文字遮罩覆盖的小型组件候选做双视图有界复查；一致高置信文字回灌后从 source 重建全部真实资产。

## 本轮变更

- 严格 evidence 集新增逐轮生成、逐文件 SHA-256 绑定的 `component-isolation.png`。联系表每格只展示一个带编号候选，使用最终完整 alpha 与 text-clean RGB，不叠加 OCR 文字。
- 组件质量新增 `component_text_residual_ratio` 与 `component_text_residual`：在最终会发布的完整组件 alpha 内，匹配源字形的连通残留超过页面自适应像素下限即硬失败。
- 背景质量新增 `background_text_residual_ratio` 与页级 `background_text_clean`：无组件、无文字的背景在 OCR 区仍保留相对局部底色可见的源字形时硬失败；页级检查避免通过清空组件绕过。
- 合成质量新增 `editable_text_once`：存在可靠文字掩码时必须存在可编辑文字对象，且栅格层不得保留第二份字形。已删除 `raster_text_preserved` 成功分支，结果不再以栅格文字兜底。
- OCR 通用噪声过滤现按置信度、字符组成、分隔符位置与重复、主体长度保留结构规范的技术标签，同时继续拒绝乱码、纯符号与无意义短串。
- targeted OCR 每页检查 16–24 个候选，两个确定性视图的最长边分别不超过 512 与 448 像素，总 crop 不超过 6 MiPixel；大候选和低 alpha 摘要候选先跳过，已知文字逐项去重。多项结果按页坐标一一匹配，文本比较只做 NFKC、casefold 与去空白并保留语义标点。
- targeted OCR 恢复文字的颜色与粗体仍从局部 bbox 邻域估计，字号换算则显式使用原页面宽度；局部 crop 不再被误当作整页而放大字号。显式参考宽度只接受非布尔正整数，省略时保持整图调用的旧行为。
- prepared-page schema v3 严格绑定 `initial_diagnostics` 的 source SHA-256、稳定 `candidate_id`、文字 bbox 和两视图文本/置信度；v1/v2 读取为空诊断。后续 native-check 必须与请求中哈希绑定的初始 diagnostics 结构和内容完全一致。`unowned_raster_text` 不解冻已通过叶组件；没有真实失败组件时立即 `preserved_with_warning`，否则最多执行五批真实修复，不制造空批次。
- 每个 OCR 视图最多保留 32 项，整页 diagnostics 最多 96 项并确定性截断；每轮 request、quality 与 native-check 必须原样携带相同的初始 diagnostics，删除或替换都会被拒绝。恢复文字只裁剪 bbox 邻域估计样式，不创建整页 RGB。
- 二次 prepare 只清理 `components`、`element-masks`、`semantic-masks` 三个约定目录中的普通单链接文件；工作根内的 source、OCR mask、其他文件及工作根外文件均拒绝，文件/目录 symlink、reparse、hardlink 或身份变化也拒绝。
- 组件文字、背景文字和合成检查复用 `_PageQualityContext` 的页级 RGB、差异、亮度与 text-ink 缓存；逐组件仅处理当前 mask/邻域和连通残留，不重复创建整页 float/blur/distance 数组。
- 保留既有 `text_ghost` 兼容报告、`over_merged_component` 跨轮 sticky、merge/collapse 非活动来源 provenance、紧 bbox 低内存与 TOCTOU 门禁；Agent confidence 不能放宽任何硬失败。
- 页面质量门禁现显式使用当前 validated graph 的活动视觉组件数，不再用历史 `initial_component_count` 猜测本轮候选。仅剩 frozen 组件且本轮报告为空时仍执行页级文字残留等检查，不解冻或重评 frozen；初始非空但当前活动视觉组件为零仍硬拒绝，不能通过丢弃整页组件绕过门禁。
- 组件执行现会在像素 ownership 生效前识别近乎完全套叠的活动父级 pair，以 `contained_parent_review` 暂停双方冻结；精确 pair 会写入下一批 Agent 可见的质量证据。Agent 查看两个隔离单元后可选择唯一像素所有者；若两者确实独立，则必须对双方显式高置信 `accept`，且两条 evidence 都引用 pair 的两个 ID，才允许共同保留，不再由固定面积规则替 Agent 决定。原始候选遮罩不被独占结果覆盖写回；最终发布仍使用唯一 ownership，并把去字区域的连续底色交给唯一组件，避免 PowerPoint 分别缩放重复父层时产生字形白边、浅灰残影或矩形补丁。
- Host/Local 重修适配层现严格按回调契约把源图、框和正负点各传一次；修复 `retry_with_box` / `retry_with_points` 首次执行时重复传入 `image` 导致整页安全失败的问题，不改变 SAM 分割算法或五批上限。
- Agent 新增 `suppress_text`：仅当视觉证据明确证明冻结 OCR 候选并非文字时，才允许把该文字节点从 `frozen` 转为 `inactive`，同步撤销既有 `attach_text` 关联；普通错字或不确定文字仍禁止删除。被抑制区域从 source 恢复给非冻结视觉修复层，并从有效文字列表、文字遮罩、质量统计和最终 PPTX 中移除。
- 文字重分类保持原始 `text_XXXX` ID，不因过滤首项或中间项而重新编号；已冻结视觉组件继续复用上一轮四类 presentation asset 引用与 SHA-256，不能因文字重分类暗中换图。Host/Local 使用同一动作契约和提示词，Skill 镜像已同步。
- 所有 OCR、LaMa、SAM、视觉与组件重修重型子进程启动前统一执行 Python GC 和 Windows working-set trim；真实 `test1.pptx` 首次分层观测峰值约 1.68 GiB，低于既定 2.4 GiB 安全线。当前主要性能债务是 targeted OCR 逐候选/逐视图反复冷启动模型，尚未改成同页批处理。

## 关键文件

- 统一入口与调度：`image2editable/cli.py`、`image2editable/runtime.py`、`image2editable/agent.py`
- 输入与 PPTX 原位替换：`image2editable/inputs.py`、`image2editable/pptx_reconstruct.py`
- 组件证据与旧管线适配：`image2editable/legacy.py`
- 组件契约、状态机与质量：`image2editable/component_contracts.py`、`image2editable/component_repair.py`、`image2editable/component_quality.py`
- OCR 准备与镜像：`image_to_ppt.py`、`scripts/text_detect.py`、`skills/image-to-ppt/scripts/image_to_ppt.py`、`skills/image-to-ppt/scripts/text_detect.py`
- Agent：`image2editable/host_agent.py`、`image2editable/local_agent.py`、`image2editable/local_agent_worker.py`
- Skill 镜像：`skills/image-to-ppt/SKILL.md`、`skills/image-to-ppt/scripts/component_contracts.py`、`skills/image-to-ppt/scripts/component_quality.py`
- 主要测试：`tests/test_targeted_ocr.py`、`tests/test_ocr_isolation.py`、`tests/test_component_contracts.py`、`tests/test_component_quality.py`、`tests/test_component_repair.py`、`tests/test_runtime_execution.py`、`tests/test_task10_runtime_e2e.py`、`tests/test_local_agent.py`
- 已批准待实现设计：`docs/superpowers/specs/2026-08-03-dual-mask-component-underlay-design.md`
- 已确认实施计划：`docs/superpowers/plans/2026-08-03-dual-mask-component-underlay.md`

## 运行入口

```bash
image2editable convert input.png -o output.pptx --agent-provider host
image2editable convert input.pdf -o output.pptx --agent-provider local
image2editable prepare input.pptx --run-dir runs/pptx-job --agent-provider host
image2editable run execute runs/pptx-job
image2editable agent next runs/pptx-job
image2editable agent record runs/pptx-job --plan response.json
```

Local 使用前检查：

```bash
image2editable models recommend --json
image2editable models status
```

## 验证事实

- 本轮 TDD 红灯覆盖多项 OCR 只取 max、部分已知文字导致整候选跳过、OCR 返回顺序变化、逐项 conflict bbox/ID、跨轮 diagnostics 删除/置空/替换、伪造空批次、整页 hash/RGB 副本和首轮视觉残留；32/96 上限另以 40/97 mutation 验证测试确实能捕获旧缺陷。
- 误 OCR 重分类相关聚焦回归为 `456 passed, 8 skipped`；覆盖动作契约、显式冻结转换授权、既有文字关联撤销、有效文字遮罩与 source 像素恢复、稳定文字 ID、冻结视觉资产哈希不变，以及最终 PPTX 不复活被抑制文字。此前全量回归为 `1586 passed, 20 skipped`，本轮最终全量仍待真实验收后重跑。
- 重修适配回归已先精确复现 `got multiple values for argument 'image'`，再验证源图只传一次、归一化框正确映射；组件重修与运行时相关回归为 `295 passed, 7 skipped`。
- `wsl和虚拟机对比.png` 已用全新 r11 Run 完成 Host Agent 两批闭环：首批冻结 13 个独立图标，第二批依据精确包含关系保留完整表格 `parent_0002` 与独立页脚 `parent_0005`，丢弃冗余子区域 `parent_0004`、`parent_0006`。最终含 15 个视觉组件、1 个背景、32 个可编辑文字，共 48 个 PowerPoint 对象；原画幅与 16:9 均经 PowerPoint COM 重开和原生渲染，删除全部文字对象后重新保存、重开所得图片层仍保留完整图标与结构，未见文字轮廓、浅灰残影或白色字形补丁。
- 真实 r3 已确认 targeted OCR 能把全页漏检的小型候选文字恢复为可编辑对象，同时暴露局部 crop 导致的字号放大；该字号回归已有通用自动化测试和修复，仍需重新生成真实输出后再宣称真实验收通过。

## 当前注意事项

- 真实文件必须串行、一次一个文件/重型页面；监控内存和磁盘，结束后确认无残留 Python/OCR/SAM 进程。禁止下载模型。
- 待自动闭环复验文件：`research_layout_demo_3pages.pdf`、`test1.pptx`、`混合.pptx`。验收需检查隔离联系表、background-only、最终渲染、可编辑文字、叶组件数、warning、PowerPoint COM reopen，以及混合 PPTX 未命中原生对象不变。
- `test1.pptx` 不得再把“每页 3 个整块父组件”视为成功条件；必须通过最小完整视觉单元和三层文字隔离门禁。
- `test1.pptx` r10/r11 已完成低内存首次分层和第 1 页真实 `suppress_text` 重修：误判为 `Q` 的蓝色放大镜已从文字层移除，并在 r10 中以高 z-order 独立为 `component_0016`，搜索卡片 `component_0011` 不再包含该图标；另外两张顶层文档也已独立补建。当前剩余硬失败是表格/大面板/卡片底层的 `component_text_residual`、`underlay_seam` 与 `underlay_gradient_break`，不得通过继续冻结整块组件或放宽门禁掩盖。
- `tests/test_component_acceptance.py` 是 ignored 的本地历史文件，不得 force-add。
