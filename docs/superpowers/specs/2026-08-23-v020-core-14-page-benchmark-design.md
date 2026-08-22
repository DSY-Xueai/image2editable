# v0.2 核心 14 页 Benchmark 设计

## 目标

v0.2 发布门禁使用一个明确、可重复、严格的 14 页核心 benchmark。现有 30 页公开 CC0 语料继续保留为扩展语料库，但不再要求全部页面在 v0.2 发布前完成真实模型重放。

## 核心组成

核心集合包含 10 个完整 case、共 14 页，不裁剪多页输入，也不生成只为凑页数的派生 PDF/PPTX。

### 图片：8 case / 8 页

1. `image-bilingual-dashboard`：中英文混排与仪表盘结构。
2. `image-combo-chart`：柱线组合图、图例和细网格。
3. `image-flowchart`：流程卡片、连接线与箭头。
4. `image-icon-matrix`：多图标、小元素和独立 ownership。
5. `image-thin-line-network`：细线、节点和低面积视觉组件。
6. `image-tiny-element-table`：密集表格与大量原生文本框。
7. `image-dark-poster`：深色背景、大写标题、终止标点和大字号。
8. `image-non-16-9-infographic`：非 16:9 原始比例与纵向排版。

### PDF：1 case / 2 页

- `pdf-rotated-page`：完整两页输入，覆盖正常方向与旋转页面、方向感知 OCR、坐标回映和原始页面比例。

### PPTX：1 case / 4 页

- `pptx-mixed-screenshot-candidates`：完整四页输入，覆盖 screenshot candidate 决策、原生对象保留、视觉重建和多页组装。

## 文件与职责

- `benchmarks/release/manifest.json`：继续描述完整 18 case / 30 页扩展语料库及每页严格阈值；不作为 v0.2 核心完成声明。
- `benchmarks/release/core-v0.2-manifest.json`：仅包含上述 10 case / 14 页，复用扩展 manifest 中相同的输入路径、SHA、case 元数据和页面阈值。
- `scripts/release_benchmark.py`：继续接受显式 `--manifest`，固定执行 `repeat=3`；不增加跳过页面、降低重复次数或容错通过选项。
- `benchmarks/release/README.md`：明确区分“30 页扩展语料库”和“v0.2 核心 14 页发布门禁”，分别记录覆盖范围与完成状态。
- `tests/test_release_benchmark.py`：同时锁定扩展 manifest 的 18 case / 30 页完整性和核心 manifest 的精确 10 case / 14 页组成，防止核心集合被静默扩大、缩小或替换。
- CI/Release 配置：普通跨平台 CI 继续运行 model-free 测试；真实模型发布门禁显式使用核心 manifest。完整 30 页重放只能作为后续扩展验证，不阻塞 v0.2。

## 数据与执行流程

1. 两份 manifest 都引用 `benchmarks/release/inputs/` 中的同一 canonical 输入，不复制或修改语料。
2. 核心 runner 对 10 个 case 按 manifest 顺序执行，每个 case 在三个独立 workspace 中重新 prepare、完成 Host capability handshake、按 request/graph 双 SHA 选择固定计划并生成 PPTX。
3. 每个页面必须返回 `validated`，并满足 manifest 中独立的 visual/text 下限、`max_unexplained_pixels=0` 和 `max_quality_violations=0`。
4. 任一重复出现 warning、fallback、非 `validated` 页面、陈旧/缺失计划、组件不足、文本不足、残差或质量违规时，整份核心报告失败。
5. 核心批量报告通过后，对 PDF 和 PPTX 输出使用桌面 PowerPoint 原生渲染，并检查页面尺寸、对象边界、文本内容和渲染一致性。

## 完成状态与命名

- 在核心批量 `repeat=3` 全部通过前，只能写“v0.2 核心 14 页 benchmark 正在验证”。
- 全部通过后，允许写“v0.2 核心 14 页 benchmark 已严格通过”。
- 30 页集合始终称为“30 页扩展语料库”；只有实际通过严格重放的页面才能计入“扩展验证进度”。
- 核心完成不等于 30 页扩展语料全部完成，也不允许把未进入核心的页面计入 v0.2 发布门禁成功率。

## 失败处理

- `stale_plan` 或 `missing_plan`：从当前 wheel 的新鲜 author 证据重新绑定并审查计划，禁止只改哈希而不验证 graph、动作和最终渲染。
- 模型、OCR 或平台差异导致 request/graph 变化：保留失败运行目录，在受支持 Windows/Python 3.12/Paddle 环境重新 author；不得放宽 schema 或质量阈值。
- 性能过慢：记录每次和每页耗时并继续现有性能优化路线；不得减少 `repeat=3`、降低模型、跳过页面或复用前次 run 状态来伪造提速。
- GitHub 原生 macOS/Windows/Linux 门禁未通过：不发布；本地成功不能替代远端门禁。

## 验证要求

1. 核心 manifest schema、输入 SHA、case 顺序、页数总计和页面阈值均有自动化测试。
2. 扩展 manifest 仍保持 18 case / 30 页且 canonical 输入 SHA 不变。
3. 所选 PDF 和 PPTX 必须分别完成 fresh author、独立严格重放及 PowerPoint 原生渲染检查。
4. 最终必须从空的新 workspace 对整个核心 manifest 执行一次批量 `repeat=3`，报告中应为 10 case、42 页尝试、0 failed attempts。
5. 完成前运行聚焦回归、全仓 pytest、wheel 内容检查、`pip check`、镜像一致性和 `git diff --check`。

## 非目标

- 不删除 30 页语料或其现有固定计划。
- 不把扩展 manifest 改写成只有 14 页。
- 不为核心集合增加页面切片、单页派生 PDF/PPTX 或 runner page-filter 功能。
- 不在本轮重新实现无关的 OCR、分割、UI 或模型配置功能。
- 未经用户单独授权不执行 `git push` 或发布。
