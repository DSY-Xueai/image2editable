# Course

## 当前项目状态
- 核心功能：图片转可编辑 PPTX，流程为读图、OCR 文本检测、背景建模/修复、前景组件提取、PPTX 分层组装。
- 默认输出不包含原图参考页；需要参考页时显式使用 `--reference`。
- 前景组件以透明 PNG 图层写入 PPT，文本以可编辑文本框写入 PPT。
- GitHub 仓库：`https://github.com/DSY-Xueai/any2ppt`。

## 本轮变更
- 修复前景大面积误检：`fg_extract.py` 对单个 detector mask 增加覆盖率保护，避免颜色/亮度差异把整页红色背景识别成一个巨大前景组件。
- 修复合并后 mask 仍可能过大的问题：`fg_extract.py` 对多个 detector 合并后的最终 mask 再做一次覆盖率保护，且 oversized edge fallback 不再无条件回流。
- 修复背景 refinement 污染：`bg_model.py` 对过大的 `fg_hint_mask` 增加保护，第一轮 mask 失控时不再污染第二轮背景修复。
- 恢复前景组件提取能力：撤回“清空前景组件”的错误 fallback，`test-image/` 转换结果重新输出多组件图层。
- 保留文字提取的阶段性修复：Tesseract 语言映射、全大写英文过滤、中文字体写入 PPT 的兼容性处理。
- 修复 PPT 组装细节：正文文本恢复自动换行，大标题和短中文标题关闭换行；东亚字体改为 PowerPoint DrawingML 的 `typeface` 子节点写法。
- 修复 OCR 竖排装饰文本误重建：`text_detect.py` 过滤明显竖排/装饰性 OCR 碎片，使其继续作为前景组件保留，避免右侧装饰字被错误重建成重叠文本框。
- 补充回归测试：新增 `tests/test_regressions.py` 覆盖 reference 参数、过大 mask 拒绝、OCR 过滤、Tesseract 语言映射和中文标题字体策略。

## 关键修改文件
- `image_to_ppt.py`：主入口与 reference 参数解析。
- `scripts/fg_extract.py`：前景 mask 生成与组件拆分。
- `scripts/bg_model.py`：背景建模与 fg hint 保护。
- `scripts/text_detect.py`：OCR fallback、噪声过滤、字体/字号估计。
- `scripts/ppt_assemble.py`：PPT 文本框字体写入。
- `skills/image-to-ppt/scripts/`：与主脚本保持同步的 skill 版本。
- `tests/test_regressions.py`：本轮回归测试。

## 运行入口
```bash
python image_to_ppt.py input.png
python image_to_ppt.py img1.png img2.png -o output.pptx
python image_to_ppt.py ./slides_folder/ -o output.pptx
python image_to_ppt.py img1.png img2.png --reference
```

## 验证结果
- `python -m pytest -q`：通过。
- `python -m compileall image_to_ppt.py scripts skills/image-to-ppt/scripts tests`：通过。
- 使用根目录 `test-image/` 生成 `test_output_files/foreground_components_checked.pptx`，共 5 页，不含原图参考页。
- 组件数量检查：第 1 页 64 个前景组件，第 5 页 117 个前景组件；未再出现整页红色遮挡。
- 已导出并人工查看第 1 页、第 5 页 PNG，确认前景组件恢复、页面可读性明显优于失败版本。
- 审查修复后回归测试增至 10 个，覆盖最终 mask 合并保护、edge fallback 超限拒绝、正文换行策略和字体 XML 写入。
- OCR 竖排装饰文本修复后回归测试增至 11 个；`test-image/2.png` OCR 输出从 17 个文本框降为 12 个，右侧装饰碎片不再进入可编辑文本层。

## 当前注意事项
- 文字提取与字体风格仍是近似重建，不应宣称与原图完全一致。
- 当前前景保护阈值会拒绝覆盖率超过 45% 的单个 detector mask；这能防止整页误检，但极端大前景页面后续仍需更多样本验证。
- 背景 inpainting 在复杂纹理、渐变和文字密集区域仍可能有局部修补痕迹，后续优化应单独处理，避免再次影响前景组件提取。
- 本轮教训：视觉修复不能只看单元测试和 PPTX 可打开，必须导出/打开实际页面逐页检查；不能用“清空组件”绕过前景失败，因为项目目标是可编辑 PPT。
