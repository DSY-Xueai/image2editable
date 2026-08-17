# 发布质量语料契约

本目录包含固定发布 benchmark 语料：18 个输入、30 页，包括 12 张图片、3 个双页 PDF、3 个四页 PPTX。

## 覆盖范围

图片固定覆盖：中英双语仪表盘、密集参数对比、人物信息卡、四段时间线、柱线组合图、流程图、图标矩阵、浅色文字渐变页、细线网络图、小元素表格、深色海报、非 16:9 信息图。

PDF 分别覆盖双页不同尺寸、旋转、高 DPI。PPTX 分别覆盖 `image_only`、`mixed_native`、`mixed_screenshot_candidates`。

## 来源与许可

输入由项目脚本公开生成，manifest 固定记录 `source=project-generated`、`license=CC0-1.0`。所有 case 默认使用已支持的 `agent_provider=host`。

## 字体来源与许可

生成器仅使用仓库内完整、未修改的 Google Fonts `Noto Sans SC` variable TTF；常规文本固定选择 `Regular`，既有粗体文本选择同一文件中的 `Bold` 实例，文件位于 `fonts/NotoSansSC[wght].ttf`。字体按 `fonts/OFL.txt` 中的 SIL Open Font License 1.1 分发，固定来源为 `google/fonts` commit `e1118da94a8cb00cf6d06cdac9ef13eb1e5c6ab7`；字体本身不是 CC0。字体渲染所得 benchmark 输入继续按 CC0-1.0 发布，manifest 中的许可字段不适用于捆绑字体文件。

## 阶段状态

当前目录已包含全部 18 个输入文件，仓库 canonical bytes 由 manifest 中的真实 `sha256` 绑定。可用 `python scripts/build_release_corpus.py <output-root>` 在不存在的目录重新生成语料；同一环境的两次 fresh generation 要求 PNG RGB 像素一致，fresh 与仓库 canonical 只比较格式、尺寸、页数和对象 inventory 等明确语义。跨平台同样只承诺这些语义等价，不承诺 PNG 像素或 PDF/PPTX 字节完全相同。本阶段不包含 runner。

## 通过标准

每页必须达到 manifest 中的最小组件数和文本框数、状态为 `validated`，并同时满足 0 warning、0 unexplained pixels、0 quality violations。后续 runner 对全部 30 页执行 `repeat=3`；runner 和 CI 接入不属于本阶段。
