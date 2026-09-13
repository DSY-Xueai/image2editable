# 转换基准

## 公开语料

`benchmarks/corpus/` 固定包含 8 张图片、3 页 PDF、3 页 mixed PPTX，共 10 个输入、14 页、3 条 routes：`images`、`pdf` 和 `mixed_pptx`。

## 当前用途

`benchmarks/corpus/` 用于输入契约和回归覆盖。发布门禁由 `scripts/release_benchmark.py` 和 `benchmarks/release/` 下的固定 manifest 驱动，使用 Agent 计划证据。Skill 日常转换不读取这些基准文件，按需安装源码时不下载此目录。

## 发布门禁标准

正式报告固定覆盖 10 个 case，每个 case 独立重复 3 次，共 30 次尝试、42 个累计页面。`status=passed` 必须同时满足完整覆盖、`failed_attempts=0`、所有页面的结构与质量门禁通过，以及相同性能基线和运行环境约束。`preserved_with_warning`、缺页、损坏输出、整页单图或不可见组件绕过均判定失败。

## 安全报告

发布门禁报告不包含任何绝对路径、URL、密钥、stderr 或异常正文。

## 结果解释

先查看顶层 `report_kind`、`status` 和 `totals`，再按 `attempts` 中的 `case_id`、`repeat`、`status` 和安全错误类型定位失败；性能问题查看 `performance` 与 `performance_comparison`。报告中的耗时只代表该固定运行环境和输入。
