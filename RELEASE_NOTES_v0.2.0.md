# image2editable v0.2.0

## 发布范围

- v0.2 核心门禁是严格的 14 页 benchmark：8 个图片页、`pdf-rotated-page` 的完整 2 页、`pptx-mixed-screenshot-candidates` 的完整 4 页。
- 核心集合包含 10 个 case；每个 case 执行 3 次独立重复，共 30 次尝试和 42 个累计页面。
- 受保护的 GitHub-hosted Windows 门禁把 case 分成 5 组并行运行，再由独立步骤核对完整覆盖、运行环境和性能。只有最终聚合报告可以代表 benchmark 通过。
- 仓库中的其他生成语料用于补充覆盖，不计入 v0.2 核心成功率。

## 严格门禁

每页必须同时满足 manifest 约束、预期状态、0 warning、0 fallback、0 unexplained pixels 和 0 quality violations；任一重复失败都会使报告失败。核心 runner 使用已安装发行包和固定 plan 证据。diagnostic 只用于审核 GitHub-hosted 环境的 plan 绑定，不能生成正式通过报告。

## 运行与发布边界

- `image2editable --version` 从已安装 distribution metadata 读取版本；当前版本为 `0.2.0`。
- `image2editable doctor` 用于检查本地依赖；用户自行部署的 OpenAI-compatible 视觉模型通过明确的 `local-service` 模式接入。
- 发行包契约矩阵覆盖 Windows、Linux、macOS 的 Python 3.10–3.12；真实性能比较只接受与 manifest、依赖约束和运行环境完全一致的基线。
- 本地模型权重、模型缓存、临时 workspace 和生成的 PPTX 不进入 wheel、Git 或 benchmark 工件；Release Gate 只保存必要的 JSON 证据。
- PowerPoint 原生对象、截图候选和 OCR 文本的边界保持严格校验；不以 warning、fallback 或未解释像素换取通过。

## 安全与版本

`SECURITY.md` 定义 0.2.x 的私密漏洞报告流程：48 小时内确认、7 天内完成初步评估。发布 workflow 只响应 `v0.2.0` tag，验证同一 commit 的 release-gate 产物后创建 draft release，不自动发布。
