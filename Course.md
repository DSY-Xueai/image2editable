# Course

## 当前项目状态
- `skills/pdf-image-to-editable-ppt` 当前在 `feature/pdf-image-runtime-upgrade` worktree 中持续开发。
- 阶段一 runtime pipeline 已完成并单独提交。
- 阶段二已补到两层：
  - `2A`：文字严格拟合校验、常见效果映射、stage2 保守增强入口。
  - `2B`：页面分层、矢量候选映射、blend 分组回退基础设施。

## 本轮新增或变更
- 新增 `EffectBlock`、`LayeredObject`、`VectorInstruction` 等数据结构。
- `PagePlan` 新增：
  - `effect_blocks`
  - `layered_objects`
  - `vector_instructions`
- 新增脚本：
  - `text_fitting.py`
  - `effect_mapping.py`
  - `stage2_enhance.py`
  - `layout_layers.py`
  - `vector_mapping.py`
  - `blend_mapping.py`
- `convert_to_ppt.py` 新增 `enable_stage2=True` 的阶段二增强入口。
- `build_ppt.py` 已支持更严格的字体名写入，并对暂不支持的效果层安全忽略。
- 新增 stage2 / 2B 测试：
  - `tests/test_models_stage2.py`
  - `tests/test_text_fitting.py`
  - `tests/test_effect_mapping.py`
  - `tests/test_stage2_enhance.py`
  - `tests/test_build_ppt_stage2.py`
  - `tests/test_convert_to_ppt_stage2.py`
  - `tests/test_models_2b.py`
  - `tests/test_layout_layers.py`
  - `tests/test_vector_mapping.py`
  - `tests/test_blend_mapping.py`
  - `tests/test_page_planner_2b.py`

## 关键修改文件
- `Course.md`
- `.gitignore`
- `skills/pdf-image-to-editable-ppt/SKILL.md`
- `skills/pdf-image-to-editable-ppt/references/README.md`
- `skills/pdf-image-to-editable-ppt/scripts/models.py`
- `skills/pdf-image-to-editable-ppt/scripts/build_ppt.py`
- `skills/pdf-image-to-editable-ppt/scripts/convert_to_ppt.py`
- `skills/pdf-image-to-editable-ppt/scripts/page_planner.py`
- `skills/pdf-image-to-editable-ppt/scripts/text_fitting.py`
- `skills/pdf-image-to-editable-ppt/scripts/effect_mapping.py`
- `skills/pdf-image-to-editable-ppt/scripts/stage2_enhance.py`
- `skills/pdf-image-to-editable-ppt/scripts/layout_layers.py`
- `skills/pdf-image-to-editable-ppt/scripts/vector_mapping.py`
- `skills/pdf-image-to-editable-ppt/scripts/blend_mapping.py`
- `tests/*.py`

## 运行入口
- `skills/pdf-image-to-editable-ppt/scripts/convert_to_ppt.py`
- `skills/pdf-image-to-editable-ppt/scripts/build_ppt.py`

## 当前注意事项
- OCR 仍然是可选依赖；缺失时必须回退到背景优先输出。
- 2A 已经可运行，但仍遵守“文字不能完全一致就不提升”的规则。
- 2B 目前只完成分层、矢量候选和 blend 分组回退的基础设施；复杂矢量真重建、多层透明/混合效果真重建还没落地。
- `.tmp-pytest/` 已清理；`stage2tmp/` 因本机权限异常暂未删掉，已加入 `.gitignore`，不会进入提交。
