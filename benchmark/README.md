# 转换基准

## 公开语料

`benchmark/corpus/` 固定包含 8 张图片、3 页 PDF、3 页 mixed PPTX，共 10 个输入、14 页、3 条 routes：`images`、`pdf` 和 `mixed_pptx`。

## 环境前置

先按项目根目录 README 准备 Local Agent 依赖、OCR、运行时模型、Agent 模型和本地视觉模型服务，然后运行：

```bash
image2editable doctor --agent-local
```

只有 `ready=true` 才运行真实 benchmark。runner 不会自动下载模型；若报告缺少模型，按根目录 README 的“安装运行时模型”和“可选：安装 Local Agent”小节执行现有安装命令，再重新检查环境。

## 运行

从项目根目录运行，且输出目录必须尚不存在：

```bash
python scripts/benchmark_conversion.py --corpus benchmark/corpus --output-dir benchmark-results
```

## 通过标准

`passed` 必须同时满足：3 routes、14 pages、0 failed_routes、0 warning_pages，并且所有必须重建页都通过可编辑结构门禁。`preserved_with_warning`、缺页、损坏输出、整页单图或不可见组件绕过均判定失败。

## 安全报告

`benchmark-results/benchmark-report.json` 的主要字段为 `schema_version`、`status`、`corpus_sha256`、`environment`、`routes` 和 `totals`；每条 route 记录类型、输入数、页数、耗时、状态、安全错误类型、warning 页数、输出摘要和性能汇总。报告不包含任何绝对路径、URL、密钥、stderr 或异常正文。

## 私有语料

私有语料放在 `benchmark/private/`，且不提交。runner 要求相同的严格 manifest schema 和固定的三 route 语义，不能随便放入图片直接运行。当前没有私有 manifest 生成器；需要复制公开 manifest 的结构，保持 route ids 与页数契约，并重算每个 case 的 bytes 和 SHA-256，以及 corpus_sha256。

## 结果解释

先查看顶层 `status` 和 `totals`，再按 route 的 `status`、`error_type`、`warning_pages` 和 `performance` 定位失败。报告中的耗时只是本机事实，不代表其他机器或输入。
