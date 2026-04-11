# Course

## 当前项目状态
- 当前目录已初始化为 Git 仓库。
- 项目当前已进入“创建 PDF/图片转可编辑 PPT skill”的升级设计阶段。
- 已完成第一版骨架实现与阶段一 runtime pipeline 提交。
- 已新增第二阶段设计文档，准备进入“文字严格拟合 + 常见视觉效果映射”的设计落地阶段。
- 已新增 2B 设计文档，准备进入“分层拆解 + 复杂矢量 + 有限混合效果重建”的高难扩展阶段。

## 本轮新增或变更
- 新增 `docs/superpowers/specs/2026-04-11-pdf-image-to-editable-ppt-design.md`。
- 新增 `docs/superpowers/plans/2026-04-11-pdf-image-to-editable-ppt.md`。
- 新增 `skills/pdf-image-to-editable-ppt/` skill 包目录。
- 新增 `tests/`，包含 skill 文档、数据模型、过滤逻辑、PPT 构建与脚本接口测试。
- 新增 `docs/superpowers/specs/2026-04-11-pdf-image-to-editable-ppt-runtime-upgrade-design.md`。
- 新增 `docs/superpowers/specs/2026-04-11-pdf-image-to-editable-ppt-stage2-design.md`。
- 新增 `docs/superpowers/specs/2026-04-11-pdf-image-to-editable-ppt-2b-design.md`。
- 新增 `docs/superpowers/plans/2026-04-11-pdf-image-to-editable-ppt-2b.md`。
- 新增 `docs/superpowers/plans/2026-04-11-pdf-image-to-editable-ppt-stage2.md`。
- 新增 `docs/superpowers/plans/2026-04-11-pdf-image-to-editable-ppt-runtime-upgrade.md`。
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
- 已完成升级设计收敛：
  - 阶段一接入真实 PDF 渲染、原生文本提取、OCR、图片提取、PPT 坐标映射。
  - 阶段二再补复杂矢量、透明效果、字体拟合等高难能力。
- 已完成升级实现计划拆解，下一步可进入真实管线开发。
- 已完成第二阶段设计收敛：
  - 2A 做文字严格拟合与常见效果映射。
  - 2B 做复杂矢量、多层透明和更高难分层重建。
- 已完成第二阶段实现计划拆解，下一步可进入 2A 开发。
- 已完成 2B 设计收敛：
  - 2B-1 先做页面分层与版面拆解。
  - 2B-2 再做复杂矢量路径转 PPT 形状。
  - 2B-3 最后做多层透明/混合效果的有限重建。
- 已完成 2B 实现计划拆解，下一步可进入 2B-1 / 2B-2 开发。

## 关键修改文件
- `Course.md`
- `docs/superpowers/specs/2026-04-11-pdf-image-to-editable-ppt-design.md`
- `docs/superpowers/specs/2026-04-11-pdf-image-to-editable-ppt-runtime-upgrade-design.md`
- `docs/superpowers/specs/2026-04-11-pdf-image-to-editable-ppt-stage2-design.md`
- `docs/superpowers/specs/2026-04-11-pdf-image-to-editable-ppt-2b-design.md`
- `docs/superpowers/plans/2026-04-11-pdf-image-to-editable-ppt-2b.md`
- `docs/superpowers/plans/2026-04-11-pdf-image-to-editable-ppt-stage2.md`
- `docs/superpowers/plans/2026-04-11-pdf-image-to-editable-ppt-runtime-upgrade.md`
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
- 当前主分支仍是第一版骨架实现；真实 PDF/OCR/PPT 元素映射尚未接入。
- 阶段一 runtime pipeline 已在 feature worktree 中提交，尚未合并回主分支。
- 第二阶段 2A 计划已写好，下一步可选择子代理执行或当前会话内联执行。
- 2B 仍未开始实现，下一步应先写 2B 的实现计划。
- 2B 计划已写好，下一步可选择子代理执行或当前会话内联执行。
- 当前工作树尚未提交；是否提交需等用户明确确认。
