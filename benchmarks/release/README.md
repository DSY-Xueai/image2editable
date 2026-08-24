# v0.2 核心 14 页 benchmark

这里保存 v0.2 发布门禁使用的固定语料、manifest 和审核过的 Host plans。核心语料由 10 个完整 case、14 页组成：8 张图片、2 页旋转 PDF 和 4 页 mixed screenshot candidates PPTX。多页文件会整份测试，不截取其中几页。

仓库还保留了用于补充覆盖的生成语料，记录在 `manifest.json` 中；它不属于 v0.2 核心通过条件，也不会被写进核心完成率。

## 覆盖内容

核心图片覆盖中英双语仪表盘、柱线组合图、流程图、图标矩阵、细线网络图、小元素表格、深色海报和非 16:9 信息图。PDF 用来验证旋转页面，PPTX 用来验证 native objects 与 screenshot candidates 混合存在时的处理。

输入由项目脚本生成，manifest 记录 `source=project-generated` 和 `license=CC0-1.0`。每个输入的 bytes 与 SHA-256 都已固定，运行时会重新校验。

## Release Gate 如何运行

真实模型 benchmark 不放进普通 push CI，需要在受保护的 Release Gate 中显式开启。GitHub-hosted Windows 会把 10 个 case 分成 5 组并行运行；每个 case 仍执行 3 次独立重复。分片只缩短等待时间，不减少测试次数，也不放宽任何质量门禁。

五份分片报告会由独立的聚合步骤重新校验。只有 manifest、依赖约束、运行环境、10 个 case、30 次尝试和 42 个累计页面全部一致，且性能没有超过同环境基线 15%，才会生成 `report_kind: official`、`status: passed` 的正式报告。单个分片不能代表 benchmark 通过。

当 GitHub-hosted 环境产生的 request/graph hash 与已有 plans 不一致时，可以先运行 diagnostic。diagnostic 只允许更新 plan 的绑定 hash，原有 decision、actions、parameters、confidence 和 evidence 必须保持不变；三次重复得到的绑定也必须完全一致。它会继续执行相同的页面与质量检查，但报告只会是 `report_kind: diagnostic`，不能算作正式通过。

Release Gate 只上传 JSON 报告，以及 diagnostic 产生的候选 plan JSON。模型文件、模型缓存、输入副本、运行 workspace 和生成的 PPTX 都不会上传为 benchmark 工件，也不会提交到 Git。

## 严格通过标准

每页必须达到 manifest 中的最小非文本视觉组件数 `min_visual_components` 和最小原生文本框数 `min_text_boxes`。两类对象独立计数；已经转为原生文本的 OCR 内容不能继续留在 raster 中重复计数。

所有页面还必须满足预期状态，并同时达到 0 warning、0 fallback、0 unexplained pixels 和 0 quality violations。缺少质量文件、plan 不匹配、任一重复失败或性能回归都会使门禁失败。

## 字体与重新生成

生成器使用仓库中的 Google Fonts `Noto Sans SC` variable TTF：`fonts/NotoSansSC[wght].ttf`。字体按 `fonts/OFL.txt` 中的 SIL Open Font License 1.1 分发，固定来源为 `google/fonts` commit `e1118da94a8cb00cf6d06cdac9ef13eb1e5c6ab7`。字体本身不是 CC0；由它渲染出的 benchmark 输入按 CC0-1.0 发布。

可以运行 `python scripts/build_release_corpus.py <output-root>` 在一个不存在的目录中重新生成语料。同一环境的两次 fresh generation 要求 PNG RGB 像素一致；不同平台只承诺格式、尺寸、页数和对象 inventory 等明确语义一致，不承诺 PNG 像素或 PDF/PPTX 字节完全相同。
