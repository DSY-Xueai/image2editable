# 发布质量语料契约

本目录定义真实发布 benchmark 的第一阶段契约：18 个输入、30 页，包括 12 张图片、3 个双页 PDF、3 个四页 PPTX。

## 覆盖范围

图片固定覆盖：中英双语仪表盘、密集参数对比、人物信息卡、四段时间线、柱线组合图、流程图、图标矩阵、浅色文字渐变页、细线网络图、小元素表格、深色海报、非 16:9 信息图。

PDF 分别覆盖双页不同尺寸、旋转、高 DPI。PPTX 分别覆盖 `image_only`、`mixed_native`、`mixed_screenshot_candidates`。

## 来源与许可

后续输入由项目公开生成，manifest 固定记录 `source=project-generated`、`license=CC0-1.0`。所有 case 默认使用已支持的 `agent_provider=host`。

## 阶段状态

当前阶段不包含输入文件，也不包含 runner。当前 `sha256` 是满足 schema 的 64 位占位值；SHA-256 为 64 位占位值不代表语料已完成。Task 2 必须生成全部输入文件，并把占位值替换为文件的真实 SHA-256；测试中的严格 xfail 保留这项 RED 契约。

## 通过标准

每页必须达到 manifest 中的最小组件数和文本框数、状态为 `validated`，并同时满足 0 warning、0 unexplained pixels、0 quality violations。后续 runner 对全部 30 页执行 `repeat=3`；runner、输入生成和 CI 接入不属于本阶段。
