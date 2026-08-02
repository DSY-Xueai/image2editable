# 真实验收数据链修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不增加图片类型特例、不下载模型的前提下，保留语义父组件与可编辑子组件、阻止组件修复重新污染文字区域，并让首轮 Agent 证据等同于真实交付画面。

**Architecture:** 将视觉阶段已有的 `semantic_mask` 与 `mask` 一起写入 hash-bound Prepared Page v2；Runtime 对 v2 构建 inactive parent + pending child，对 v1 保持原单层 parent 行为。背景修复后用可信 `text_clean_image` 做确定性像素恢复，首轮证据则从 clean background 合成当前活跃视觉节点，并扣除 OCR 文字掩码。

**Tech Stack:** Python 3.10–3.12、NumPy、Pillow、OpenCV、pytest、现有 `image2editable` Runtime/Host Agent CLI。

---

## 文件结构

- `scripts/bg_model.py`：背景修复后恢复可信文字清理像素。
- `image_to_ppt.py`：保存两层掩码、写入/加载 Prepared Page v2，并兼容 v1。
- `image2editable/legacy.py`：从 v2 构建父子组件图并生成符合交付语义的首轮证据。
- `skills/image-to-ppt/scripts/bg_model.py`：Skill 内背景实现镜像。
- `skills/image-to-ppt/scripts/image_to_ppt.py`：Skill 内主转换实现镜像。
- `tests/test_regressions.py`：背景文字恢复、Prepared Page v2/v1 与掩码契约回归测试。
- `tests/test_runtime_execution.py`：Runtime 父子图和首轮证据合成测试。
- `tests/test_ocr_isolation.py`：补齐 `bg_model.py` 的产品/Skill 镜像一致性校验。
- `Course.md`：记录功能状态、关键文件、运行入口与真实验收结果。

### Task 1: 背景修复后恢复可信文字区域

**Files:**
- Modify: `scripts/bg_model.py:489-497`
- Modify: `image_to_ppt.py:1280-1285,1348-1353,1381-1386`
- Modify: `skills/image-to-ppt/scripts/bg_model.py:489-497`
- Modify: `skills/image-to-ppt/scripts/image_to_ppt.py:1280-1285,1348-1353,1381-1386`
- Test: `tests/test_regressions.py`
- Test: `tests/test_ocr_isolation.py:1218-1251`

- [ ] **Step 1: 写背景污染的失败测试**

在 `tests/test_regressions.py` 增加一个纯合成测试。让 fake large inpainter 把整个修复区写成灰色，再断言文字掩码内必须逐像素等于 `text_clean_image`：

```python
def test_clean_background_restores_trusted_text_pixels_after_component_inpaint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = image_to_ppt.np.full((12, 16, 3), 240, dtype=image_to_ppt.np.uint8)
    element_mask = image_to_ppt.np.ones((12, 16), dtype=bool)
    text_mask = image_to_ppt.np.zeros((12, 16), dtype=image_to_ppt.np.uint8)
    text_mask[4:8, 5:11] = 255
    text_clean = source.copy()
    text_clean[4:8, 5:11] = (17, 31, 47)

    monkeypatch.setattr(bg_model, "needs_large_mask_inpaint", lambda mask: True)
    repaired = bg_model.build_clean_background(
        source,
        [element_mask],
        text_mask,
        large_inpainter=lambda image, mask: image_to_ppt.np.full_like(image, 128),
        text_clean_image=text_clean,
    )

    assert image_to_ppt.np.array_equal(repaired[text_mask > 0], text_clean[text_mask > 0])
    assert image_to_ppt.np.all(repaired[text_mask == 0] == 128)
```

- [ ] **Step 2: 运行测试并确认因新参数不存在而失败**

Run: `python -m pytest tests/test_regressions.py::test_clean_background_restores_trusted_text_pixels_after_component_inpaint -q`

Expected: FAIL，错误包含 `unexpected keyword argument 'text_clean_image'`。

- [ ] **Step 3: 写最小背景恢复实现**

将 `build_clean_background` 扩展为仅在传入可信图像时恢复文字掩码，不改变无 `text_clean_image` 的旧调用：

```python
def build_clean_background(
    img: np.ndarray,
    element_masks: list[np.ndarray],
    text_mask: np.ndarray,
    large_inpainter=None,
    text_clean_image: np.ndarray | None = None,
) -> np.ndarray:
    """Remove visual elements and text from an image."""
    removal = build_removal_mask(element_masks, text_mask)
    background = repair_masked_background(img, removal, large_inpainter)
    if text_clean_image is None:
        return background
    trusted = np.asarray(text_clean_image)
    if trusted.shape != background.shape:
        raise ValueError("text-clean image must match the source image shape")
    background[text_mask > 0] = trusted[text_mask > 0]
    return background
```

在 `_process_image` 的三次 `build_clean_background(...)` 调用中都传入：

```python
text_clean_image=text_clean_image,
```

同步同内容到两个 Skill 镜像文件，并把 `"bg_model.py"` 加入 `test_ocr_product_and_skill_mirrors_match` 的 `script_names`。

- [ ] **Step 4: 运行聚焦测试**

Run: `python -m pytest tests/test_regressions.py::test_clean_background_restores_trusted_text_pixels_after_component_inpaint tests/test_ocr_isolation.py::test_ocr_product_and_skill_mirrors_match -q`

Expected: `2 passed`。

- [ ] **Step 5: 提交确定性文字区域恢复**

```bash
git add scripts/bg_model.py image_to_ppt.py skills/image-to-ppt/scripts/bg_model.py skills/image-to-ppt/scripts/image_to_ppt.py tests/test_regressions.py tests/test_ocr_isolation.py
git commit -m "修复：阻止组件修复重新污染文字区域"
```

### Task 2: Prepared Page v2 持久化父子掩码并兼容 v1

**Files:**
- Modify: `image_to_ppt.py:587-598,1378-1443,1449-1480,1801-2004`
- Modify: `skills/image-to-ppt/scripts/image_to_ppt.py`（与产品入口保持逐字节一致）
- Test: `tests/test_regressions.py:4980-5090,5181-5217,5425-5449`

- [ ] **Step 1: 扩展测试 fixture 并写 v2 往返失败测试**

让 `_prepare_component_layers_fixture` 同时创建 `semantic-masks/`。子掩码必须非空，父掩码比子掩码更完整：

```python
semantic_masks_dir = target / "semantic-masks"
semantic_masks_dir.mkdir()
semantic_mask_paths = []
for index, mask_path in enumerate(mask_paths):
    semantic_path = semantic_masks_dir / f"{index:04d}.png"
    with Image.open(mask_path) as child_image:
        parent = child_image.convert("L").copy()
    if parent.getbbox() is None:
        parent.putpixel((index + 1, index + 1), 255)
        Image.new("L", (16, 10), 0).save(mask_path)
        with Image.open(mask_path) as child_image:
            child = child_image.convert("L").copy()
        child.putpixel((index + 1, index + 1), 255)
        child.save(mask_path)
        child.close()
    parent.putpixel((min(index + 2, 15), min(index + 2, 9)), 255)
    parent.save(semantic_path)
    parent.close()
    semantic_mask_paths.append(semantic_path)
```

fixture 返回值增加：

```python
"_semantic_mask_paths": [str(mask) for mask in semantic_mask_paths],
```

更新 `test_prepare_component_layers_persists_initial_components_without_quality`：

```python
assert manifest["schema_version"] == 2
assert len(manifest["assets"]["semantic_masks"]) == 2
```

更新 `test_load_component_layers_recovers_absolute_owned_paths` 的 `path_values`：

```python
*restored["_semantic_mask_paths"],
```

- [ ] **Step 2: 写 v2 拒绝损坏层级、v1 可读的失败测试**

增加三个测试：

```python
def test_load_component_layers_rejects_v2_mask_count_mismatch(tmp_path, monkeypatch):
    prepared, _ = _prepare_component_layers_fixture(tmp_path, monkeypatch)
    state_path = Path(prepared["state_path"])
    manifest = json.loads(state_path.read_text(encoding="utf-8"))
    manifest["assets"]["semantic_masks"].pop()
    _write_prepared_manifest(state_path, manifest)
    with pytest.raises(ValueError, match="mask counts"):
        image_to_ppt.load_component_layers(state_path)


def test_load_component_layers_rejects_v2_child_outside_parent(tmp_path, monkeypatch):
    prepared, work_dir = _prepare_component_layers_fixture(tmp_path, monkeypatch)
    state_path = Path(prepared["state_path"])
    manifest = json.loads(state_path.read_text(encoding="utf-8"))
    record = manifest["assets"]["semantic_masks"][0]
    semantic_path = work_dir / record["path"]
    Image.new("L", (16, 10), 0).save(semantic_path)
    record["sha256"] = hashlib.sha256(semantic_path.read_bytes()).hexdigest()
    _write_prepared_manifest(state_path, manifest)
    with pytest.raises(ValueError, match="inside its parent"):
        image_to_ppt.load_component_layers(state_path)


def test_load_component_layers_reads_v1_without_inventing_semantic_masks(tmp_path, monkeypatch):
    prepared, _ = _prepare_component_layers_fixture(tmp_path, monkeypatch)
    state_path = Path(prepared["state_path"])
    manifest = json.loads(state_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest["assets"].pop("semantic_masks")
    _write_prepared_manifest(state_path, manifest)
    restored = image_to_ppt.load_component_layers(state_path)
    assert "_semantic_mask_paths" not in restored
```

- [ ] **Step 3: 运行 Prepared Page 测试并确认失败**

Run: `python -m pytest tests/test_regressions.py -k "prepare_component_layers_persists or recovers_absolute_owned_paths or v2_mask_count or v2_child_outside or reads_v1" -q`

Expected: 新测试 FAIL，原因是当前 schema 仍为 v1、没有 `semantic_masks`。

- [ ] **Step 4: 保存两组掩码并写 Prepared Page v2**

用一个复用两次的最小 helper 取代只写 element mask 的实现：

```python
def _persist_visual_masks(
    work_dir: Path,
    directory_name: str,
    masks: list[np.ndarray],
) -> list[str]:
    masks_dir = (work_dir / directory_name).resolve()
    masks_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, mask in enumerate(masks):
        mask_path = (masks_dir / f"{index:04d}.png").resolve()
        Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L").save(mask_path)
        paths.append(str(mask_path))
    return paths
```

在 `_process_image` 末尾保存并返回两组路径：

```python
element_mask_paths = _persist_visual_masks(work_dir, "element-masks", element_masks)
semantic_mask_paths = _persist_visual_masks(work_dir, "semantic-masks", semantic_masks)
# slide_data
"_element_mask_paths": element_mask_paths,
"_semantic_mask_paths": semantic_mask_paths,
```

Prepared 常量改为：

```python
_PREPARED_PAGE_SCHEMA_VERSION = 2
_PREPARED_ASSET_FIELDS_V1 = {
    "source_image", "ocr_mask", "text_clean", "element_masks",
    "background_original", "background_widescreen",
    "background_removal_mask", "background_difference",
}
_PREPARED_ASSET_FIELDS = _PREPARED_ASSET_FIELDS_V1 | {"semantic_masks"}
```

`_write_prepared_page` 的 assets 增加：

```python
"semantic_masks": [
    _prepared_asset_record(work_dir, path, "semantic mask")
    for path in slide_data["_semantic_mask_paths"]
],
```

- [ ] **Step 5: 实现 schema-aware v1/v2 加载和逐对验证**

Loader 只接受 1 或 2，并按版本校验字段：

```python
schema_version = manifest["schema_version"]
if type(schema_version) is not int or schema_version not in {1, 2}:
    raise ValueError("prepared page schema_version is invalid")
expected_asset_fields = (
    _PREPARED_ASSET_FIELDS if schema_version == 2 else _PREPARED_ASSET_FIELDS_V1
)
if not isinstance(assets, dict) or set(assets) != expected_asset_fields:
    raise ValueError("prepared page assets are invalid")
```

v2 加载 element/semantic 记录后逐对读取，避免同时常驻所有大掩码：

```python
def _validate_prepared_mask_pair(
    child_path: str, parent_path: str, image_size: tuple[int, int]
) -> None:
    with Image.open(child_path) as image:
        child = np.asarray(image.convert("L")) > 0
    with Image.open(parent_path) as image:
        parent = np.asarray(image.convert("L")) > 0
    expected_shape = (image_size[1], image_size[0])
    if child.shape != expected_shape or parent.shape != expected_shape:
        raise ValueError("prepared component masks have invalid dimensions")
    if not np.any(child) or not np.any(parent):
        raise ValueError("prepared component masks cannot be empty")
    if np.any(child & ~parent):
        raise ValueError("prepared child mask must stay inside its parent")
```

```python
if schema_version == 2:
    semantic_records = assets["semantic_masks"]
    if (
        not isinstance(semantic_records, list)
        or len(element_mask_paths) != initial_count
        or len(semantic_records) != initial_count
    ):
        raise ValueError("prepared page mask counts are invalid")
    semantic_mask_paths = [
        _load_prepared_asset(work_dir, record, "semantic mask")
        for record in semantic_records
    ]
    for child_path, parent_path in zip(
        element_mask_paths, semantic_mask_paths, strict=True
    ):
        _validate_prepared_mask_pair(
            child_path,
            parent_path,
            (dimensions["img_width"], dimensions["img_height"]),
        )
    loaded["_semantic_mask_paths"] = semantic_mask_paths
```

v1 不设置 `_semantic_mask_paths`。同步产品 `image_to_ppt.py` 到 Skill 镜像。

- [ ] **Step 6: 运行 Prepared Page 聚焦测试与镜像测试**

Run: `python -m pytest tests/test_regressions.py -k "prepare_component_layers or load_component_layers" tests/test_ocr_isolation.py::test_ocr_product_and_skill_mirrors_match -q`

Expected: 全部 PASS。

- [ ] **Step 7: 提交 Prepared Page v2**

```bash
git add image_to_ppt.py skills/image-to-ppt/scripts/image_to_ppt.py tests/test_regressions.py tests/test_ocr_isolation.py
git commit -m "功能：保留组件父子掩码数据链"
```

### Task 3: Runtime 构建真实父子图并修正首轮证据

**Files:**
- Modify: `image2editable/legacy.py:550-640,713-805`
- Test: `tests/test_runtime_execution.py:762-833`

- [ ] **Step 1: 写 v2 父子图的失败测试**

把现有单层用例保留为 `test_initial_page_session_keeps_v1_parent_fallback`，另加 v2 用例。测试数据中 parent 覆盖 `(2, 1)..(5, 4)`，child 覆盖 `(3, 2)..(5, 4)`，OCR mask 与 child 在 `(4, 3)` 重叠：

```python
prepared["_semantic_mask_paths"] = [str(parent_mask)]
session = legacy._build_initial_page_session(
    store, "page_001", prepared, reconstruction
)
graph = json.loads(Path(session["evidence"]["component-graph.json"]).read_text())
parent, child, text = graph["nodes"]
assert (parent["id"], parent["kind"], parent["state"], parent["parent_id"]) == (
    "parent_0001", "parent", "inactive", None,
)
assert (child["id"], child["kind"], child["state"], child["parent_id"]) == (
    "component_0001", "child", "pending", "parent_0001",
)
request = json.loads(
    legacy.build_component_agent_request(session, repair_round=1).read_text(encoding="utf-8")
)
assert request["candidate_ids"] == ["component_0001"]
assert request["frozen_ids"] == ["text_0001"]
```

- [ ] **Step 2: 写首轮合成排除文字像素的失败断言**

在同一 v2 测试中断言非文字 child 像素来自 source，而 child 与文字重叠处仍保留 clean background：

```python
with Image.open(session["evidence"]["reconstructed.png"]) as reconstructed:
    assert reconstructed.getpixel((3, 2)) == (1, 2, 3)
    assert reconstructed.getpixel((4, 3)) == (0, 0, 0)
```

- [ ] **Step 3: 运行 Runtime 测试并确认失败**

Run: `python -m pytest tests/test_runtime_execution.py -k "initial_page_session" -q`

Expected: v2 测试 FAIL，当前 graph 没有 inactive parent/child 层级，文字重叠像素也被重新贴回。

- [ ] **Step 4: 按 Prepared 版本构建两类图**

在 `_build_initial_page_session` 中按 `_semantic_mask_paths` 是否存在分支：

```python
semantic_masks = prepared.get("_semantic_mask_paths")
has_semantic_parents = semantic_masks is not None
if has_semantic_parents and len(semantic_masks) != len(components):
    raise ValueError("prepared component and semantic mask counts differ")
visual_node_count = len(components) * (2 if has_semantic_parents else 1)
```

v2 每对掩码写两个 hash-bound 节点：

```python
for component_id, kind, state, parent_id, mask_source in (
    (f"parent_{index:04d}", "parent", "inactive", None, semantic_masks[index - 1]),
    (f"component_{index:04d}", "child", "pending", f"parent_{index:04d}", masks[index - 1]),
):
    # 复制掩码、验证非空、计算 bbox 和 sha256
    nodes.append({
        "id": component_id,
        "kind": kind,
        "parent_id": parent_id,
        "state": state,
        "mask": f"masks/{mask_target.name}",
        "mask_sha256": sha256_file(mask_target),
        "bbox": [left, top, right, bottom],
        "z_index": component.get("z_index", index - 1),
        "text_ids": [],
    })
```

v1 继续生成 `kind=parent,state=pending,parent_id=None` 的旧节点。磁盘预留的 `node_count` 改用 `visual_node_count + len(text_items)`。

- [ ] **Step 5: 首轮证据按真实交付语义合成**

在 `_render_component_evidence` 开头先加载并校验 `text_mask`，循环中对活跃视觉节点构建有效渲染掩码：

```python
with Image.open(text_mask_path) as image:
    text_mask = image.convert("L").copy()
if text_mask.size != source.size:
    raise ValueError("component evidence text mask dimensions differ")
```

```python
render_mask = ImageChops.subtract(mask, text_mask)
ownership.paste(color_layer, (0, 0), render_mask)
if reconstructed_path is None:
    reconstructed.paste(source, (0, 0), render_mask)
render_mask.close()
```

`numbered-masks.png` 仍显示完整活跃 mask 供 Agent 判断边界；inactive parent 和 text 节点继续不渲染。删除函数后半段重复的 text mask 加载，并补充 `from PIL import ImageChops`。

- [ ] **Step 6: 运行 Runtime 聚焦测试**

Run: `python -m pytest tests/test_runtime_execution.py -k "initial_page_session or legacy_assembly_uses_accepted" -q`

Expected: 全部 PASS；v1 fallback 行为不变，v2 candidate 只包含 child。

- [ ] **Step 7: 提交 Runtime 父子图和证据修复**

```bash
git add image2editable/legacy.py tests/test_runtime_execution.py
git commit -m "修复：按真实父子层级生成Agent证据"
```

### Task 4: 全量验证、文档同步与真实 PNG 重新验收

**Files:**
- Modify: `Course.md`
- Inspect only: `tmp/task13-host-image-a31c484/`
- Create runtime artifacts: `tmp/task13-host-image-v2/`

- [ ] **Step 1: 运行格式与聚焦回归测试**

Run: `python -m pytest tests/test_regressions.py tests/test_runtime_execution.py tests/test_ocr_isolation.py -q`

Expected: 全部 PASS。

- [ ] **Step 2: 运行完整自动化测试**

Run: `python -m pytest -q`

Expected: 全部 PASS；允许仅有项目原先声明的 skip，不允许新失败。

- [ ] **Step 3: 保留旧诊断 Run，创建全新的 Host PNG Run**

不删除 `tmp/task13-host-image-a31c484`，它作为根因证据保留；用新目录执行：

```bash
python -m image2editable prepare "wsl和虚拟机对比.png" --run-dir "tmp/task13-host-image-v2" --agent-provider host
python -m image2editable run execute "tmp/task13-host-image-v2"
python -m image2editable agent next "tmp/task13-host-image-v2"
```

Expected: Run 进入 `awaiting_agent`；首轮 graph 中可见节点为 pending child，且各有 inactive parent；`reconstructed.png` 已合成 child，OCR 区域没有被 child 覆盖。

- [ ] **Step 4: 有界完成 Host Agent 验收循环**

逐轮查看 `agent next` 返回的八项证据，写入与当前 request SHA、page、round、provider 完全匹配的计划，然后执行：

```bash
python -m image2editable agent record "tmp/task13-host-image-v2" --plan "tmp/task13-host-image-v2/host-plan.json"
python -m image2editable run execute "tmp/task13-host-image-v2"
```

若仍返回 `awaiting_agent`，再次执行 `agent next → 视觉检查 → agent record → run execute`；每页最多 5 轮，不能绕过质量门禁。验收记录：父子节点数、最终组件数、文字残影、组件残缺/阴影/重影、峰值 RAM/VRAM、临时磁盘和重型子进程退出情况。

- [ ] **Step 5: 更新 Course.md**

把“真实 Run 暂停待设计”的旧描述替换为已实现状态，至少记录：

```markdown
- Prepared Page 升级到 v2：同时绑定 semantic parent mask 与 editable child mask；v1 仍按旧单层 parent 安全恢复。
- Runtime 首轮图使用 inactive parent + pending child，Agent 只处理可见 child；最多 5 个重修批次和父级回退不变。
- clean background 每次组件修复后恢复可信 text-clean 像素；首轮 reconstructed 按活跃视觉节点合成并排除 OCR mask。
```

随后追加一条真实 Host PNG 验收记录，直接写明本轮实际目录、最终状态、父/子/最终组件数量、重修轮次、峰值 RAM/VRAM、临时磁盘峰值和重型子进程退出结果；只记录命令与系统监控实际观察到的值。

- [ ] **Step 6: 检查 diff 与仓库状态**

Run: `git diff --check`

Expected: 无输出。

Run: `git status --short`

Expected: 只包含本任务预期的文档改动以及用户已知的 Task 13 README/Skill 文档改动；真实输入与 `tmp/` 验收产物不进入提交。

- [ ] **Step 7: 提交文档与验收记录**

```bash
git add Course.md README.md README_EN.md skills/image-to-ppt/SKILL.md
git commit -m "文档：记录Agent父子组件真实验收"
```

## 自检结果

- Spec coverage：Prepared Page v2、v1 fallback、父子 Runtime 图、文字像素恢复、首轮真实合成、5 轮限制、Provider 隔离、真实 PNG 重验均有对应任务。
- 非目标检查：计划没有增加图片类别规则、语义缓存、`retry_text_cleanup`、矢量/SmartArt 承诺或 Local 模型下载。
- 类型一致性：持久化键固定为 `_element_mask_paths` / `_semantic_mask_paths`；manifest 资产固定为 `element_masks` / `semantic_masks`；Runtime ID 固定为 `parent_XXXX` / `component_XXXX`。
- Placeholder scan：无 `TBD`、`TODO` 或待替换示例；动态验收数据明确要求只写命令与系统监控实测值。
