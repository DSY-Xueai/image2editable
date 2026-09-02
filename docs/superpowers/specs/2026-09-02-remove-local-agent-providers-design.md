# Remove Local Agent Providers Design

## 背景与根因

项目的转换 Runtime 默认使用 `host`，但 CLI、运行时契约、Skill 和公开文档仍把 `local` 与 `local-service` 作为正式 Agent Provider。`image-to-ppt` Skill 还要求在敏感内容场景选择本地路径，并规定 `local-service` 缺少配置时停止。因此 Code 选择这条公开路径后，会暂停转换并要求用户配置本地模型或本地服务；这不是 Host 默认值失效，而是已公开的 Provider 行为。

用户决定完整删除 `local` 与 `local-service`，只保留 Host Agent。删除范围仅限组件判断与修复计划所使用的 Agent Provider，不删除 OCR、SAM、LaMa、Grounding DINO、PPTX/PSD 组装等核心本机转换能力。

## 目标

- 项目只支持 `host` Agent Provider，Code 使用 Skill 转换时不得要求用户配置、安装或下载本地 Agent 模型。
- 删除 `local` 与 `local-service` 的执行代码、配置、依赖、模型目录、公开入口和专属测试，不保留不可达的死代码。
- 保留 CLI、Host Agent 编排、OCR、SAM、LaMa、Grounding DINO、质量门、PPTX/PSD 输出和 runtime 模型完整性校验。
- 中英文 README、Security 说明、发布说明和两个 Skill 与实际 Host-only 行为一致。

## 非目标

- 不删除命令行工具本身。
- 不删除 `image2editable models install runtime` 所安装的 SAM、LaMa 和 Grounding DINO。
- 不修改 OCR、组件拆分、背景修复、质量门或组装算法。
- 不迁移已有的 `local` / `local-service` Run；这些 Run 在新版本中明确拒绝继续执行。
- 不顺手重构与 Provider 删除无关的代码。

## 方案决策

采用完整删除方案，不采用仅隐藏文档或保留废弃占位的方案。仅隐藏文档会遗留无关实现；废弃占位仍会向用户暴露 Local 概念。完整删除能同时消除误导入口和维护负担。

## 对外契约

运行清单继续保存 `agent_provider: "host"`，避免改动 Run schema 和 Host Agent 的既有校验。CLI 继续接受 `--agent-provider host`，兼容现有 Host 命令；`local` 与 `local-service` 不再是合法选项。Python Runtime 的 `agent_provider` 参数也只接受 `host`。

`image2editable models` 只管理核心 runtime 模型：

- 保留 `models install runtime`。
- 保留 `models status`，结果只报告 runtime 模型状态。
- 删除 Agent 模型推荐、安装和状态入口。
- 删除 `doctor --agent-local`，普通 `doctor` 继续检查转换依赖、OCR 和 runtime 模型。

旧 Run 的 manifest 如果记录 `local` 或 `local-service`，所有公开恢复、状态和执行入口都应在开始实际处理前拒绝，并明确说明当前只支持 `host`。

## 代码边界

删除仅服务于两种 Local Provider 的模块和资源，包括本地 Agent worker、本地 OpenAI-compatible service client、Agent 模型硬件推荐与安装、Agent 模型 catalog 及其专属测试。移除 `agent-local` 可选依赖和只为本地服务存在的 `.env.example`。

Runtime 删除以下分支：自动加载本地 Agent receipt、本地模型来源绑定、local-service 配置加载、PPTX 候选自动决策、组件计划自动生成，以及完成摘要中的本地 Agent 模型信息。Host 路径仍按既有流程进入 `awaiting_agent`，通过 `agent next` / `agent record` 接收当前宿主生成的结构化决策。

以下内容必须保留：

- `runtime_models.py` 与 `runtime_model_catalog.json`；
- 被 runtime 模型复用的 `model_receipts.py`；
- OCR、SAM、LaMa、Grounding DINO 和相应路径、下载、完整性校验代码；
- Host Agent、组件契约、质量门、PPTX/PSD 构建和 standalone Skill 的核心脚本。

文件或函数名中出现 `local` 不足以判定可删除。例如“本地 TorchScript LaMa adapter”和本地文件路径属于核心转换实现，必须保留。

## Skill 与文档

`image-to-ppt` 和 `image-to-psd` Skill 删除所有 `local` / `local-service` Provider 说明、命令和隐私建议，统一描述为 Host Agent 流程。Skill 明确规定：不得要求用户配置或下载本地 Agent 模型或 OpenAI-compatible 本地服务；转换需要的 runtime 模型仍按已有授权流程安装。

README 与 README_EN 删除本地模型服务配置、Provider 对比和相关环境变量说明；CLI 安装章节保留，但避免把“本机运行转换”与“本地 Agent 模型”混为一谈。Security 和当前发布说明不再宣称支持 `local-service`。`Course.md` 在实现完成时同步当前状态、关键文件、运行入口和注意事项。

## 测试策略

先添加或修改失败测试，证明当前代码仍错误地接受 Local Provider 或公开 Local 指引，再实施删除：

- CLI 仅接受 `host`，并拒绝 `local` 与 `local-service`。
- Runtime 和所有 manifest 入口只接受 `host`。
- `models` 与 `doctor` 不再暴露 Agent 模型命令和参数，但 runtime 模型安装、状态与检查保持工作。
- wheel/sdist 不再包含 Local Agent 模块和 Agent 模型 catalog，仍包含 runtime catalog。
- 两个 Skill、中英文 README、Security、发布说明和 `.env` 模板不再出现 Local Provider 配置或引导。
- Host Agent 的 prepare、execute、next、record、恢复和完成摘要回归测试通过。
- OCR、runtime 模型、PPTX、PSD、依赖契约和完整测试通过。

删除专门验证 Local 推理成功路径的测试；把共享契约测试改为 Host-only 断言。测试删除必须与对应产品能力删除一一对应，不能借机减少核心转换覆盖。

## 验收标准

- `rg` 在产品代码、公开文档和 Skill 中找不到 `local-service`、`agent-local`、本地 Agent 模型配置变量或调用链；历史 benchmark 工件中的固定结果数据可保留，但不得作为当前使用说明。
- `image2editable convert ... --agent-provider host` 仍可准备 Host Run；传入两种已删除 Provider 时在参数或契约层失败。
- `image2editable doctor` 和 `image2editable models install runtime` 保持可用。
- 完整测试通过，`Course.md` 已同步，Git 提交不包含 `local-resources/` 或其他无关文件。
