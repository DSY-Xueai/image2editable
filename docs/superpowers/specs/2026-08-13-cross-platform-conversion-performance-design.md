# 跨平台转换性能优化设计

## 背景

当前 standalone `image-to-ppt` skill 为控制峰值内存，默认以隔离子进程执行 OCR、Grounding DINO、SAM 2.1 Large 和 LaMa。隔离本身是必要的兼容性边界，但粒度过细：同一页的 prompted SAM、automatic SAM、残差轮和组件修复会分别创建进程并重新加载相同模型。复杂页还可能因为候选 OCR 补回文字而重新执行整套视觉准备。其结果是单页转换可能耗时数小时，同时 Host/Local Agent 在多轮修复中重复接收大量不变证据。

本轮在原有项目和现有重建闭环内优化，不建立第二套转换架构。

## 目标

- 保持 SAM 2.1 Large、Grounding DINO、OCR、LaMa、现有阈值、候选覆盖、最多五轮修复和全部质量门禁不变。
- 降低同一页内重模型重复加载次数，并使多个同类操作共享一次模型加载。
- 避免 OCR 新增文字区域后无条件重跑整页视觉准备。
- 后续 Agent 修复轮只传递本轮决策需要的完整证据，减少重复图片和结构化上下文。
- 在 Windows、Linux 和 macOS 上保持同一功能语义；优化不能依赖某一台机器、CUDA 或 WSL。
- 为真实耗时、设备选择、模型加载次数和 Agent 输入规模提供可核对记录。
- 修正 standalone skill 中未固定的 SAM 依赖和不恰当的 WSL 优先建议。

## 明确不做

- 不用确定性规则替代 Agent 对组件语义和独立可编辑性的判断。
- 不引入轻量分割模型、低质量模式或按页面复杂度降级模型。
- 不降低置信度、残差、文字重复、ownership、视觉差异或 PPTX reopen 门禁。
- 不通过减少候选、跳过失败区域、保留整页原图或压低修复轮上限换取速度。
- 不承诺未实测硬件上的固定分钟数。

## 方案选择

采用“隔离边界不变、同模型同阶段批处理”。模型仍在可回收的 worker 中运行，但一个 worker 在一次请求内完成该阶段的全部同类操作，之后释放资源。

不采用整页常驻 SAM/DINO。该方案虽有更低延迟，但会把 SAM、DINO、OCR 或 LaMa 的内存生命周期重叠，低显存、统一内存较小或 CPU-only 的机器更容易 OOM。

不采用按硬件自动切换常驻/隔离。当前没有覆盖足够多 Windows、Linux、Apple Silicon、独显和 CPU-only 环境的性能语料，新增双执行路径会扩大未经验证的行为面。

## 架构与数据流

### 1. 性能记录

新增页级 JSONL 性能记录器，使用单调时钟记录以下事件：

- `worker_start` / `worker_finish`
- `model_load_start` / `model_load_finish`
- `inference_start` / `inference_finish`
- `agent_request_published`
- `agent_plan_recorded`

记录字段限定为 schema version、run/page、阶段、模型类别、操作数量、耗时、进程号、平台、Python、设备类别、CUDA/MPS 可用状态、输入文件数量和总字节数。不得记录图像内容、OCR 文本、提示文本、文件原始路径或模型响应正文。

每页汇总写入现有 reconstruction 目录，run summary 只引用汇总数值。性能记录失败不得改变转换结果，但必须写入普通运行日志。

### 2. SAM 阶段批处理

扩展现有 `sam_worker.py` 请求协议，使一次进程可以接收有序的 operation 列表。operation 仍只包含现有四类语义：

- prompted candidate generation
- automatic candidate generation
- hole recheck
- component box/point retry

worker 只创建一次 SAM generator，按请求顺序执行全部 operation，并为每项返回独立结果。调用方验证 operation 数量、类型、ID、输出尺寸和顺序；任一项无效时整批失败，不发布部分结果。

初始分割的 prompted 和 automatic 合为一个批次。每个残差阶段的 prompted 和 automatic 合为一个批次。同一组件修复轮中的全部 box/point retry 合为一个批次。hole recheck 只有在能与同阶段 SAM 生命周期合并且依赖已满足时才并入；否则保留独立批次，不改变执行顺序来追求加载次数。

目标不是让 SAM 永久驻留，而是从“每个操作加载一次”收敛到“每个依赖阶段加载一次”。

### 3. 增量 OCR 补回

候选 OCR 发现新文字后，先计算新增文字清理 mask 相对旧 mask 的差集。只复用满足以下条件的既有视觉资产：

- 组件 ownership、semantic mask 和 bbox 均不与差集及其安全扩张区域相交；
- 组件输入 source hash、旧文字 mask hash 和模型版本仍匹配；
- 组件不是与受影响组件具有父子、重叠或共享边界依赖的节点。

本轮只对可以证明与全部视觉节点及其 3px 依赖域完全不相交的非空文字差集执行增量复用。此时复用首 pass 已逐项绑定 hash、shape 和关系的 component/element/semantic 资产，不重新运行 DINO/SAM，但重新生成文字清理、背景、removal、foreground/ownership 证据，并由原质量路径复核。受影响闭包非空时不在现有全局视觉流水线上伪装局部 SAM 等价，而是执行原完整第二视觉 pass。

组件 bbox 必须由已验证 element/semantic mask 的实际非零范围重算并写入 cache identity，不能信任组件元数据中的旧坐标。依赖闭包对累计 mask crop、候选节点对和局部像素工作量设置固定预算；超过预算不裁剪节点或降低检查精度，直接执行完整第二视觉 pass。

首视觉 pass 的 source hash 必须绑定模型实际读取的字节：caller 通过受控读取器计算 SHA-256 和字节数，并通过进程参数而非 workdir 内可替换的 request 文件传给隔离 worker；request 本身同样由进程参数绑定 SHA-256/字节数，其中记录 OCR mask/text-clean 的受控读取 hash。worker 在加载 DINO/SAM 前再次验证 source、request 和引用资产均为普通文件、非 link/reparse、单硬链接、句柄身份稳定，且字节数和 SHA-256 匹配，再从已验证内存快照解码 source、mask 和 text-clean。后续视觉处理不再从这些路径读取图像；首 pass 返回的绑定 hash 与随后 prepared manifest 的 source hash 不一致时禁止复用。

`_text_delta_recompute_scope` 仍返回完整的相交、父子、mask 重叠和 3px 邻接闭包，为未来独立拆分视觉流水线保留正确契约。无法证明资产不受影响、差集为空但 OCR 内容变化、source/旧 cleanup mask/SAM/DINO protocol/cache identity 任一 hash 不一致、mask 不可读或依赖关系不完整时，同样回退到现有完整第二视觉 pass。回退是完整质量路径，不是整页图片输出。

### 4. 局部残差调度

从确定性的 unexplained/差异 mask 提取 connected components，按原分辨率生成带上下文的无损 crop。相邻或重叠 crop 合并，坐标在进入和离开 worker 时做显式映射。

局部结果必须满足：

- crop 完整覆盖绑定残差及安全边距；
- SAM 结果不触碰 crop 的非页面边界；
- 回映射后 mask 尺寸、bbox 和像素计数有效；
- 该轮最终 ownership 与页面质量门禁重新计算。

任一条件不满足，或残差 mask 过于分散使 crop 总面积没有实际收益时，自动执行原有整页 DINO/SAM 路径。局部调度只减少输入面积，不删除残差，不改变模型和阈值。

### 5. Agent 增量证据

第一轮继续发布完整证据集。后续轮根据 pending、reopened、quality violations 和依赖邻居生成无损 `round-review.png`，其中每个相关组件包含 source、isolation、ownership、reconstruction、difference 和 residual 的同坐标视图，并清晰标注稳定 ID。

后续轮仍提供完整 source、当前 reconstruction、当前 quality report、完整请求 hash 以及本轮可操作节点和必要关系邻居。对本轮决策无影响且 hash 未变化的全页 numbered masks、OCR overlay 等不再重复作为独立视觉输入；其 hash 和前一轮引用保留在请求中，供完整性审计。

Local Agent 的验证器仍使用磁盘上的完整 graph 和 request 校验计划，不因为 prompt 中采用图摘要而放宽 action、object ID、frozen state、request hash 或置信度约束。若无法生成完整 round review，发布原有完整证据集。

### 6. 跨平台设备行为

删除“优先使用 WSL”的统一建议。文档改为优先使用当前平台已经正确安装并通过 `doctor`/设备预检的加速后端：

- Windows/Linux：PyTorch 报告 CUDA 可用时使用 CUDA；ROCm 环境沿用 PyTorch 提供的兼容设备接口。
- macOS：当前生产路径保持现有受支持设备选择；没有真实 Apple Silicon 回归证据前，不把 MPS 自动设为新默认值。
- 其他环境：使用 CPU，保留相同模型和质量门禁，并明确性能会显著较慢。
- 不为了 WSL 建议从一个可用 GPU 环境切换到缺少 PyTorch、CUDA 或模型缓存的环境。

实现不得依赖 Windows 路径、PowerShell、NVIDIA 型号或固定显存大小。设备信息探测失败时记录 `unknown`，不阻止原有模型代码自行选择设备。

### 7. 依赖固定

standalone skill 的 `references/requirements.txt` 将 SAM 依赖从 `@main` 改为项目根依赖使用的固定 commit：

`2b90b9f5ceec907a1c18123530e92e794ad901a4`

产品脚本与 `skills/image-to-ppt/scripts/` 的运行时镜像继续保持字节一致；依赖文件按各自发布位置同步固定值。

## 错误处理和恢复

- 批处理 worker 使用临时输出，全部结果验证后原子发布；崩溃、超时或结果缺失不留下可复用的半批次。
- 增量 OCR、局部残差或增量证据只要无法证明输入绑定完整，就回退到当前完整路径。
- 已经验证并绑定 hash 的 OCR、SAM、组件和 donor 资产继续沿用现有 retry 复用规则。
- 性能记录不参与质量判断，也不能成为成功条件。
- 所有新缓存均绑定 source、mask、模型/协议版本和相关参数；不跨不相同页面复用推理结果。

## 测试与验收

### 自动测试

- SAM 批处理：证明一次模型构建处理多项 operation，结果顺序和单项旧路径等价；畸形或部分结果整批拒绝。
- 修复批处理：同轮多个 box/point 只启动一个 worker，并保持 action 顺序和 graph transition。
- OCR 增量：不相交资产复用；相交、父子依赖、hash 变化和证据不完整触发重算或完整回退。
- 残差 crop：坐标回映射、padding、合并、触边回退、分散区域整页回退和最终质量重算。
- Agent 证据：首轮完整，后续轮只省略 hash 未变且与本轮无关的独立视觉文件；相关节点、关系邻居和当前质量证据完整。
- 跨平台：用 platform/device probe 单元测试覆盖 Windows、Linux、macOS、CUDA unavailable 和探测失败；不伪造真实硬件性能结论。
- 依赖和产品/skill 镜像一致性测试。

### 回归与真实文件

- 全量 pytest 必须不低于本轮基线：`1763 passed, 22 skipped`。基线退出码为 0；另有既有 pytest-asyncio 配置 warning 和 PowerPoint COM `0x80010108` 退出期诊断。
- 继续用现有真实图片和 `test1.pptx` 验证可编辑组件、TextBox、无整页原图以及质量 ledger。
- 对相同输入比较优化前后 component/text 数量、有效 mask、quality violations 和最终 PPTX reopen 结果；不得以耗时改善覆盖质量回归。
- 性能验收至少核对模型加载事件和 Agent 输入规模。固定分钟目标只作为特定硬件实测结果报告，不作为跨平台承诺。

## 预期效果与限制

主要收益来自减少 SAM、Local Agent 模型和重复证据的冷启动/重复输入。收益大小随页面复杂度、修复 action 数和硬件变化，不能在实现前承诺统一倍数。

CPU-only 的 SAM 2.1 Large 本身仍然昂贵。本设计能消除重复计算，但不会把大型模型的首次推理优化到秒级。进一步更换模型、量化或自动降级需要独立的大规模语料验证，不属于本轮范围。
