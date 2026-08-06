# 16:9 Content-Aware Retarget Implementation Plan

> **For agentic workers:** Implement inline in the current session. Subagents and commits are intentionally omitted because project rules require explicit user authorization.

**Goal:** Replace 16:9 contain-and-side-fill output with deterministic, single-slide, content-aware retargeting while preserving the original-ratio output.

**Architecture:** `scripts/bg_model.py` owns the monotonic pixel mapping and background resampling. `image_to_ppt.py` builds the protection mask and passes mapping metadata. `scripts/ppt_assemble.py` maps component/text geometry into the 16:9 slide while preserving protected component aspect ratios.

**Tech Stack:** Python 3.10–3.12, NumPy, OpenCV, Pillow, python-pptx, pytest.

---

### Task 1: 单调内容感知映射

**Files:**
- Modify: `tests/test_regressions.py`
- Modify: `scripts/bg_model.py`

- [ ] 添加失败测试：竖图、方图、超宽图、原生 16:9 的映射必须严格单调并覆盖 1920×1080。
- [ ] 运行目标测试，确认因 API 不存在而失败。
- [ ] 实现 `retarget_background_to_widescreen(background, protection_mask, canvas_width, canvas_height)`，返回 `(image, x_edges, y_edges)`。
- [ ] 使用平滑重要度投影分配剩余尺度；用 `cv2.remap` 生成全画布背景。
- [ ] 运行目标测试和背景模块相关回归测试。

### Task 2: 组件保护规则与保护遮罩

**Files:**
- Modify: `tests/test_regressions.py`
- Modify: `scripts/fg_extract.py`
- Modify: `image_to_ppt.py`

- [ ] 添加失败测试：小型/不规则组件默认保护宽高比，大面积稠密结构组件允许随布局伸缩，OCR 区域始终受保护。
- [ ] 运行测试并确认预期失败。
- [ ] 在导出组件字典中增加 `preserve_aspect`；用统一规则构建 16:9 protection mask。
- [ ] 在 `_process_image` 中生成 widescreen 背景和映射，并写入 `slide_data`。
- [ ] 运行目标测试。

### Task 3: PPT 图层统一映射

**Files:**
- Modify: `tests/test_regressions.py`
- Modify: `scripts/ppt_assemble.py`
- Modify: `image_to_ppt.py`

- [ ] 添加失败测试：16:9 组件/文字使用映射坐标；保护组件保持宽高比；结构组件填满映射边界；原比例变换不变。
- [ ] 运行测试并确认预期失败。
- [ ] 为单图和多图组装传递 `widescreen_x_edges`、`widescreen_y_edges`。
- [ ] 在组件和文字添加路径中统一映射边界；保护组件执行 contain-in-mapped-box。
- [ ] 运行目标测试及 `python -m pytest -q`。

### Task 4: 镜像、文档与端到端验证

**Files:**
- Modify: `skills/image-to-ppt/scripts/bg_model.py`
- Modify: `skills/image-to-ppt/scripts/fg_extract.py`
- Modify: `skills/image-to-ppt/scripts/ppt_assemble.py`
- Modify: `skills/image-to-ppt/scripts/image_to_ppt.py`
- Modify: `Course.md`

- [ ] 将主项目 PPTX 相关改动逐文件同步到 Skill 镜像并检查一致性。
- [ ] 更新 `Course.md` 的当前 16:9 管线、本轮变更、关键文件和验证结果。
- [ ] 用 `test-image/test.png --slide-size 16:9` 重新生成输出。
- [ ] 渲染每页并全尺寸检查；运行越界检查。
- [ ] 重新运行全量 pytest、镜像一致性检查和 Git diff 范围检查。
- [ ] 保留未提交状态，向用户报告并询问是否提交新版本。
