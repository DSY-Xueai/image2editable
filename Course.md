# Course

## 当前项目状态
- 当前目录已初始化为 Git 仓库。
- 项目当前已进入“创建 PDF/图片转可编辑 PPT skill”的初始实现阶段。
- 已完成该 skill 的正式设计文档、实现计划，以及第一版 skill/脚本/测试骨架。

## 本轮新增或变更
- 新增 `docs/superpowers/specs/2026-04-11-pdf-image-to-editable-ppt-design.md`。
- 新增 `docs/superpowers/plans/2026-04-11-pdf-image-to-editable-ppt.md`。
- 新增 `skills/pdf-image-to-editable-ppt/` skill 包目录。
- 新增 `tests/`，包含 skill 文档、数据模型、过滤逻辑、PPT 构建与脚本接口测试。
- 设计已明确采用“双层方案”：
  - 底图层保证视觉 100% 保真、原底原色。
  - 编辑层仅在高置信情况下提取文字和图片。
- 明确支持多页 PDF，默认“每页一张幻灯片”，可按用户要求切分长页/长图。
- 明确失败回退策略：宁可少提取，也绝不破坏视觉完整性。
- 已落地第一版实现骨架：
  - `SKILL.md` 写入核心保证、回退规则和脚本入口。
  - `models.py`/`filtering.py` 定义页面计划与保守筛选逻辑。
  - `build_ppt.py` 可生成最小可保存的 `.pptx`。
  - 渲染、文字提取、图片提取脚本先提供适配器接口。

## 关键修改文件
- `Course.md`
- `docs/superpowers/specs/2026-04-11-pdf-image-to-editable-ppt-design.md`
- `docs/superpowers/plans/2026-04-11-pdf-image-to-editable-ppt.md`
- `skills/pdf-image-to-editable-ppt/SKILL.md`
- `skills/pdf-image-to-editable-ppt/scripts/*.py`
- `skills/pdf-image-to-editable-ppt/references/README.md`
- `tests/*.py`

## 运行入口
- 当前可用入口：
  - `skills/pdf-image-to-editable-ppt/SKILL.md`
  - `skills/pdf-image-to-editable-ppt/scripts/build_ppt.py`
  - `skills/pdf-image-to-editable-ppt/scripts/render_pdf_pages.py`
  - `skills/pdf-image-to-editable-ppt/scripts/extract_text_layout.py`
  - `skills/pdf-image-to-editable-ppt/scripts/extract_images.py`

## 当前注意事项
- 设计已冻结，`spec` 与实现计划都已写入文档。
- 该 skill 的承诺是“视觉保真优先 + 尽量提取可编辑元素”，不是保证任意输入都能完整转换为全部可编辑对象。
- 目前是第一版骨架实现，真实 PDF/OCR/PPT 元素映射仍需后续接入具体工具链。
- 当前工作树尚未提交；是否提交需等用户明确确认。
