# 发布质量语料契约

本目录包含固定的 30 页扩展语料库：18 个输入，包括 12 张图片、3 个双页 PDF、3 个四页 PPTX。v0.2 发布门禁使用 `core-v0.2-manifest.json` 中的 10 个完整 case、14 页，不裁剪多页输入。

## 覆盖范围

图片固定覆盖：中英双语仪表盘、密集参数对比、人物信息卡、四段时间线、柱线组合图、流程图、图标矩阵、浅色文字渐变页、细线网络图、小元素表格、深色海报、非 16:9 信息图。

PDF 分别覆盖双页不同尺寸、旋转、高 DPI。PPTX 分别覆盖 `image_only`、`mixed_native`、`mixed_screenshot_candidates`。

v0.2 核心 14 页由 8 张图片、完整 2 页 `pdf-rotated-page` 和完整 4 页 `pptx-mixed-screenshot-candidates` 组成。`manifest.json` 继续描述 18 case / 30 页扩展语料库，不作为 v0.2 核心完成声明。

## 来源与许可

输入由项目脚本公开生成，manifest 固定记录 `source=project-generated`、`license=CC0-1.0`。所有 case 默认使用已支持的 `agent_provider=host`。

## 字体来源与许可

生成器仅使用仓库内完整、未修改的 Google Fonts `Noto Sans SC` variable TTF；常规文本固定选择 `Regular`，既有粗体文本选择同一文件中的 `Bold` 实例，文件位于 `fonts/NotoSansSC[wght].ttf`。字体按 `fonts/OFL.txt` 中的 SIL Open Font License 1.1 分发，固定来源为 `google/fonts` commit `e1118da94a8cb00cf6d06cdac9ef13eb1e5c6ab7`；字体本身不是 CC0。字体渲染所得 benchmark 输入继续按 CC0-1.0 发布，manifest 中的许可字段不适用于捆绑字体文件。

## 阶段状态

当前目录已包含全部 18 个输入文件，仓库 canonical bytes 由 manifest 中的真实 `sha256` 绑定。可用 `python scripts/build_release_corpus.py <output-root>` 在不存在的目录重新生成语料；同一环境的两次 fresh generation 要求 PNG RGB 像素一致，fresh 与仓库 canonical 只比较格式、尺寸、页数和对象 inventory 等明确语义。跨平台同样只承诺这些语义等价，不承诺 PNG 像素或 PDF/PPTX 字节完全相同。v0.2 核心 14 页 benchmark 正在验证；30 页集合始终称为扩展语料库，未完成的扩展页面不得计入核心成功率。

严格 Host runner 使用已安装发行包和固定 plans。v0.2 核心命令为：

```bash
python -m scripts.release_benchmark --manifest benchmarks/release/core-v0.2-manifest.json --workspace <fresh-workspace> --report <fresh-workspace>/benchmark-report.json
```

完整扩展语料命令为：

```bash
python -m scripts.release_benchmark --manifest benchmarks/release/manifest.json --workspace <fresh-workspace> --report <fresh-workspace>/extended-benchmark-report.json
```

runner 固定执行 3 次独立重复；每次都重新校验输入 hash、按真实 request/graph hash 选择 plans，并拒绝非 `validated` 页面、warning、fallback、缺失/异常质量工件。报告只有三次重复的所有 case 都通过时才是 `status: passed`；任一失败会保留失败类型并返回非零退出码。模型缓存和 run 证据留在本机，不进入 Git。

## 通过标准

每页必须分别达到 manifest 中的最小非文本视觉组件数 `min_visual_components` 和最小原生文本框数 `min_text_boxes`；两类对象独立计数，已转为原生文本的 OCR 内容不得重复保留在 raster 中凑视觉组件数。页面还必须为 `validated`，并同时满足 0 warning、0 unexplained pixels、0 quality violations。v0.2 核心 10 个 case、14 页必须三次重复全部通过；真实模型 runner 不放入普通 push CI，而由受保护的发布门禁显式执行并保存报告。核心完成不代表 30 页扩展语料已全部完成。
