# image2editable v0.2.0

## 发布范围

- v0.2 核心门禁是严格的 14 页 benchmark：8 个图片页、`pdf-rotated-page` 的完整 2 页、`pptx-mixed-screenshot-candidates` 的完整 4 页。
- 核心集合包含 10 个 case；runner 执行 3 次独立重复，共 42 个 page attempt。2026-08-23 在 Windows AMD64、Python 3.12、CUDA 环境通过，`failed_attempts=0`。
- 三次总耗时中位数为 3,623,083 ms；逐 case 中位数记录在 `benchmarks/release/BASELINE.json`，并绑定核心 manifest 与 `constraints/runtime.txt` 的 SHA-256。
- 原 30 页集合保留为扩展语料库。本版本不宣称 30 页扩展集合已全部严格重放，扩展未完成页不计入核心成功率。

## 严格门禁

每页必须同时满足 manifest 约束、`validated` 状态、0 warning、0 unexplained pixels 和 0 quality violations；任一重复失败都会使报告失败。核心 runner 使用已安装发行包和固定 plan 证据，模型缓存与运行证据只保留在本机。

核心验证命令：

```bash
python -m scripts.release_benchmark \
  --manifest benchmarks/release/core-v0.2-manifest.json \
  --workspace <fresh-workspace> \
  --report <fresh-workspace>/benchmark-report.json
```

## 运行与发布边界

- `image2editable --version` 从已安装 distribution metadata 读取版本；当前版本为 `0.2.0`。
- `image2editable doctor` 用于检查本地依赖；`--agent-local` 使用内置 Qwen 路线，旧的 OpenAI-compatible 服务使用明确的 `local-service` 模式。
- 发行包契约矩阵覆盖 Windows、Linux、macOS 的 Python 3.10–3.12；真实性能基线绑定本次 Windows AMD64/Python 3.12/CUDA 环境。CI 发布门禁在 Ubuntu/Tesseract 上执行契约检查，并由受保护环境执行核心 benchmark。
- 本地模型权重、模型缓存、临时 workspace、benchmark report 和私有运行证据不进入 wheel 或 Git。
- PowerPoint 原生对象、截图候选和 OCR 文本的边界保持严格校验；不以 warning、fallback 或未解释像素换取通过。

## 安全与版本

`SECURITY.md` 定义 0.2.x 的私密漏洞报告流程：48 小时内确认、7 天内完成初步评估。发布 workflow 只响应 `v0.2.0` tag，验证同一 commit 的 release-gate 产物后创建 draft release，不自动发布。
