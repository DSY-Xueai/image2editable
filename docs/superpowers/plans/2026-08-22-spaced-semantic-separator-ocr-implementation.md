# Spaced Semantic Separator OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让高置信度 OCR 标题保留带空格的语义 `/` 与 `-`，并用全新 wheel、全新 author run、双页视觉 QA 和固定三次严格 replay 关闭 `pdf-mixed-page-sizes` 的假绿问题。

**Architecture:** 在 OCR 噪声过滤入口增加一个保守纯函数，只豁免“分隔符两侧均有明确语义”的标题，不改 OCR 引擎、遮罩、targeted recovery 或 benchmark 阈值。根脚本和 standalone skill 镜像保持逐字节一致；真实发布证据必须从重新安装的 wheel 和不存在的新工作目录生成。

**Tech Stack:** Python 3.12、pytest、PaddleOCR 3.7.0、PaddlePaddle 3.3.1、python-pptx、PowerPoint 渲染工具、`scripts.release_benchmark`。

---

## 文件结构

- 修改 `scripts/text_detect.py`：定义并使用窄范围空格语义分隔符判定。
- 修改 `skills/image-to-ppt/scripts/text_detect.py`：与产品入口逐字节同步。
- 修改 `tests/test_ocr_isolation.py`：覆盖合法标题、非法边界和最终文本结果。
- 修改 `tests/test_regressions.py`：锁定两个 text detection 入口完全一致。
- 修改本地忽略文件 `Course.md`：记录行为变化、关键文件、验证状态和尚未完成的真实门禁。
- 替换 `benchmarks/release/plans/pdf-mixed-page-sizes--component-page00{1,2}-round-0{1,2,3}-gpu.json`：只接受 fresh author 产生且与新 request/graph hash 匹配的六份计划。

### Task 1: 用回归测试复现标点丢失

**Files:**
- Modify: `tests/test_ocr_isolation.py`

- [ ] **Step 1: 写合法空格分隔符的失败测试**

在现有 `test_filter_noise_keeps_high_confidence_technical_labels` 后加入：

```python
def test_filter_noise_keeps_spaced_semantic_separators() -> None:
    labels = [
        "LETTER PORTRAIT / MIXED SIZE",
        "ALPHA - BETA",
        "ALPHA / BETA / GAMMA",
    ]
    boxes = [
        {"box": (0, index * 20, 240, 16), "text": label, "confidence": 0.95}
        for index, label in enumerate(labels)
    ]

    assert [box["text"] for box in text_detect._filter_noise(boxes)] == labels
```

- [ ] **Step 2: 写非法边界的保持拒绝测试**

```python
def test_filter_noise_rejects_ambiguous_spaced_separators() -> None:
    labels = [
        "MCOULE ST:SETMP",
        "ALPHA /",
        "/ BETA",
        "ALPHA // BETA",
        "ALPHA / BETA:V2",
        "A / B",
    ]
    boxes = [
        {"box": (0, index * 20, 180, 16), "text": label, "confidence": 0.95}
        for index, label in enumerate(labels)
    ]
    boxes.append(
        {
            "box": (0, len(boxes) * 20, 240, 16),
            "text": "LETTER PORTRAIT / MIXED SIZE",
            "confidence": 0.4,
        }
    )

    assert text_detect._filter_noise(boxes) == []
```

- [ ] **Step 3: 写最终文本结果保留内部斜杠的失败测试**

```python
def test_build_text_result_preserves_internal_semantic_separator(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        text_detect,
        "_estimate_style",
        lambda *_: {"font_size": 24.0, "color": (0, 0, 0), "bold": True},
    )
    image = np.full((80, 280, 3), 255, dtype=np.uint8)
    raw_boxes = [
        {
            "box": (10, 10, 250, 32),
            "text": "LETTER PORTRAIT / MIXED SIZE",
            "confidence": 0.99,
        }
    ]

    text_items, _ = text_detect._build_text_result(image, raw_boxes, 0.7, 2)

    assert [item["text"] for item in text_items] == [
        "LETTER PORTRAIT / MIXED SIZE"
    ]
```

- [ ] **Step 4: 运行定向测试并确认 RED 原因正确**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_ocr_isolation.py -k "spaced_semantic_separators or ambiguous_spaced_separators or preserves_internal_semantic_separator" -vv
```

Expected: 合法 `/` 标题和 `_build_text_result` 测试失败；反例测试通过。失败原因必须是 `_filter_noise` 删除合法标题，而不是 import、fixture 或环境错误。

### Task 2: 实现最小过滤豁免并同步镜像

**Files:**
- Modify: `scripts/text_detect.py:591-672`
- Modify: `skills/image-to-ppt/scripts/text_detect.py:591-672`

- [ ] **Step 1: 在 `_filter_noise` 前增加纯判定函数**

```python
def _is_spaced_semantic_separator_text(text: str) -> bool:
    if any(separator in text for separator in ":;.\\"):
        return False

    separator_indexes = [
        index for index, char in enumerate(text) if char in "/-"
    ]
    if not separator_indexes:
        return False
    if any(
        index == 0
        or index == len(text) - 1
        or not text[index - 1].isspace()
        or not text[index + 1].isspace()
        for index in separator_indexes
    ):
        return False
    if any(
        left in "/-" and right in "/-"
        for left, right in zip(text, text[1:])
    ):
        return False

    parts = re.split(r"\s+[/\-]\s+", text)
    return len(parts) == len(separator_indexes) + 1 and all(
        sum(
            1
            for char in part
            if char.isalnum()
            or "\u4e00" <= char <= "\u9fff"
            or "\u3400" <= char <= "\u4dbf"
        )
        >= 2
        for part in parts
    )
```

- [ ] **Step 2: 只在全大写乱码分支加入窄豁免**

把分支改成：

```python
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) >= 4:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            has_cjk = any("\u4e00" <= c <= "\u9fff" for c in text)
            has_garbled_separator = any(c in text for c in ":;./\\")
            if (
                upper_ratio > 0.8
                and has_garbled_separator
                and not has_cjk
                and not valid_technical_label
                and not _is_spaced_semantic_separator_text(text)
            ):
                continue
```

- [ ] **Step 3: 机械同步 standalone skill 镜像**

Run:

```powershell
Copy-Item -LiteralPath scripts\text_detect.py -Destination skills\image-to-ppt\scripts\text_detect.py
```

Expected: 两个文件的 SHA-256 完全一致。

- [ ] **Step 4: 运行定向测试并确认 GREEN**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_ocr_isolation.py -k "filter_noise or preserves_internal_semantic_separator" -vv
```

Expected: 全部通过；既有 compact technical label 和 malformed label 断言不变。

### Task 3: 锁定镜像契约并完成代码级回归

**Files:**
- Modify: `tests/test_regressions.py`
- Modify: `Course.md`

- [ ] **Step 1: 写根脚本与 skill 镜像一致性测试**

```python
def test_text_detect_product_and_skill_mirrors_match() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (root / "scripts" / "text_detect.py").read_bytes() == (
        root / "skills" / "image-to-ppt" / "scripts" / "text_detect.py"
    ).read_bytes()
```

- [ ] **Step 2: 运行相关测试文件**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_ocr_isolation.py tests\test_targeted_ocr.py tests\test_regressions.py -q
```

Expected: 全部通过，无新增 warning；原有乱码、低置信度、竖排碎片和 targeted OCR 行为不变。

- [ ] **Step 3: 运行静态检查和镜像哈希检查**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m ruff check scripts\text_detect.py skills\image-to-ppt\scripts\text_detect.py tests\test_ocr_isolation.py tests\test_regressions.py
E:\i2e-release-py312\Scripts\python.exe -m py_compile scripts\text_detect.py skills\image-to-ppt\scripts\text_detect.py
$rootHash = (Get-FileHash scripts\text_detect.py -Algorithm SHA256).Hash
$skillHash = (Get-FileHash skills\image-to-ppt\scripts\text_detect.py -Algorithm SHA256).Hash
if ($rootHash -ne $skillHash) { throw "text_detect mirrors differ" }
git diff --check
```

Expected: Ruff、编译和 diff-check 均通过；两个哈希相同。

- [ ] **Step 4: 同步 `Course.md`**

在“当前项目状态”记录：高置信度、空格包围且两侧至少两个有效字符的 `/` 或 `-` 标题可通过噪声过滤；其他标点、缺边、重复分隔和低置信度仍拒绝。明确说明 case 13 仍需 fresh wheel、fresh author、视觉 QA 和三次严格 replay，不能因单元测试通过而计为完成。

- [ ] **Step 5: 提交代码级修复**

```powershell
git add scripts\text_detect.py skills\image-to-ppt\scripts\text_detect.py tests\test_ocr_isolation.py tests\test_regressions.py
git commit -m "修复：保留标题语义分隔符"
```

Expected: 提交不包含六份旧 benchmark plans，也不包含模型、wheel、运行目录或忽略的 `Course.md`。

### Task 4: 重建 wheel 并验证安装态 OCR

**Files:**
- Build artifact only: `E:\rpdf13n-wheel\image2editable-0.2.0-py3-none-any.whl`
- Runtime evidence only: `E:\rpdf13n-0822-author1\`

- [ ] **Step 1: 创建不存在的 E 盘构建目录并构建 wheel**

Run:

```powershell
$wheelRoot = 'E:\rpdf13n-wheel'
if (Test-Path -LiteralPath $wheelRoot) { throw "wheel root already exists" }
New-Item -ItemType Directory -Path $wheelRoot | Out-Null
$env:TEMP = 'E:\i2e-temp'
$env:TMP = 'E:\i2e-temp'
E:\i2e-release-py312\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation --wheel-dir $wheelRoot .
$wheel = Get-ChildItem -LiteralPath $wheelRoot -Filter '*.whl' | Select-Object -Single -ExpandProperty FullName
```

Expected: 只生成一个 `image2editable-0.2.0-py3-none-any.whl`，模型文件不进入 wheel。

- [ ] **Step 2: 强制重装 fresh wheel 并做 checkout 外 smoke**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pip install --no-deps --force-reinstall $wheel
Push-Location E:\i2e-temp
E:\i2e-release-py312\Scripts\python.exe -I -c "from scripts.text_detect import _filter_noise; b={'box':(0,0,240,20),'text':'LETTER PORTRAIT / MIXED SIZE','confidence':0.99}; assert _filter_noise([b])[0]['text']==b['text']; print('installed separator smoke: pass')"
Pop-Location
```

Expected: 输出 `installed separator smoke: pass`；导入来自 `E:\i2e-release-py312`，不是 checkout。

- [ ] **Step 3: 验证依赖与模型位置**

Run:

```powershell
$env:IMAGE2EDITABLE_MODEL_CACHE = 'E:\image2editable-model-cache'
E:\i2e-release-py312\Scripts\python.exe -m pip check
E:\i2e-release-py312\Scripts\image2editable.exe doctor --agent-local
```

Expected: `pip check` 和 doctor 通过；运行时模型继续位于 E 盘，不写入 Git。

### Task 5: 从全新状态重新作者化 case 13

**Files:**
- Runtime evidence only: `E:\rpdf13n-0822-author1\run\`
- Replace after validation: `benchmarks/release/plans/pdf-mixed-page-sizes--component-page001-round-01-gpu.json`
- Replace after validation: `benchmarks/release/plans/pdf-mixed-page-sizes--component-page001-round-02-gpu.json`
- Replace after validation: `benchmarks/release/plans/pdf-mixed-page-sizes--component-page001-round-03-gpu.json`
- Replace after validation: `benchmarks/release/plans/pdf-mixed-page-sizes--component-page002-round-01-gpu.json`
- Replace after validation: `benchmarks/release/plans/pdf-mixed-page-sizes--component-page002-round-02-gpu.json`
- Replace after validation: `benchmarks/release/plans/pdf-mixed-page-sizes--component-page002-round-03-gpu.json`

- [ ] **Step 1: 准备全新 Host run**

Run:

```powershell
$authorRoot = 'E:\rpdf13n-0822-author1'
if (Test-Path -LiteralPath $authorRoot) { throw "author root already exists" }
$env:TEMP = 'E:\i2e-temp'
$env:TMP = 'E:\i2e-temp'
$env:IMAGE2EDITABLE_MODEL_CACHE = 'E:\image2editable-model-cache'
E:\i2e-release-py312\Scripts\image2editable.exe prepare benchmarks\release\inputs\13-mixed-page-sizes.pdf -o "$authorRoot\output.pptx" --run-dir "$authorRoot\run" --lang en --format pptx --agent-provider host --slide-size original
```

Expected: 两个页面产生全新的 request/graph；page 2 的最终 OCR 文本包含完整 `LETTER PORTRAIT / MIXED SIZE`。

- [ ] **Step 2: 逐轮记录只绑定当前证据的 Host plans**

对每个 `image2editable agent next "$authorRoot\run"` 返回的 request，只使用该 request 的 component id、允许动作和当前 graph hash 生成计划，再执行：

```powershell
$planPath = Join-Path $authorRoot 'host-plan-current.json'
E:\i2e-release-py312\Scripts\image2editable.exe agent record "$authorRoot\run" --plan $planPath
E:\i2e-release-py312\Scripts\image2editable.exe run execute "$authorRoot\run"
```

每轮必须检查：没有 `warning`、没有 fallback、没有 accept-all、没有降低阈值；任何组件动作都可从当前 numbered masks、OCR overlay、ownership 和质量诊断直接证明。旧 author6 的计划只能用于对照意图，不能复用其 request/graph hash。

Expected: 两页最终均为 `validated`，每页至少 6 个视觉组件、4 个文本框、0 unexplained pixels、0 quality violations。

- [ ] **Step 3: 只提升新 run 产生的六份固定计划**

把两页三轮的最终计划复制到上述六个 repo 路径，并逐份核对 `request_sha256`、`graph_sha256`、page id、round 和 GPU capability 均与新 run 工件一致。

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_release_benchmark.py -q
git diff --check
```

Expected: release benchmark 测试全部通过；六份计划无 duplicate、stale 或 hash mismatch。

### Task 6: 完成语义、渲染和三次严格 replay 门禁

**Files:**
- Runtime evidence only: `E:\rpdf13n-0822-author1\output.pptx`
- Runtime evidence only: `E:\release-benchmark-pdf-mixed-v020-fixed\`
- Runtime evidence only: `E:\release-benchmark-pdf-mixed-v020-fixed-report.json`
- Modify: `Course.md`

- [ ] **Step 1: 校验 PPTX 原生文本语义**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -c "from pptx import Presentation; import sys; texts=[s.text for slide in Presentation(sys.argv[1]).slides for s in slide.shapes if hasattr(s,'text_frame')]; assert 'LETTER PORTRAIT / MIXED SIZE' in texts, texts; print('native slash text: pass')" E:\rpdf13n-0822-author1\output.pptx
```

Expected: 输出 `native slash text: pass`，标题是原生文本而非栅格补丁。

- [ ] **Step 2: 完整渲染并检查两页**

Run:

```powershell
$qaRoot = 'E:\rpdf13n-0822-author1\qa'
E:\i2e-release-py312\Scripts\python.exe "C:\Users\d's'y\.codex\plugins\cache\openai-primary-runtime\presentations\26.819.11345\skills\presentations\container_tools\render_slides.py" E:\rpdf13n-0822-author1\output.pptx --output_dir $qaRoot
E:\i2e-release-py312\Scripts\python.exe "C:\Users\d's'y\.codex\plugins\cache\openai-primary-runtime\presentations\26.819.11345\skills\presentations\container_tools\slides_test.py" E:\rpdf13n-0822-author1\output.pptx
```

随后逐张以原始分辨率检查 `$qaRoot` 下两页 PNG。Expected: page 2 可见完整 `LETTER PORTRAIT / MIXED SIZE`；page 1 蓝色横幅无白色重影；两页无裁切、溢出、遮罩残影或缺失结构。

- [ ] **Step 3: 在不存在的新 workspace 执行固定三次严格 replay**

Run:

```powershell
$replayRoot = 'E:\release-benchmark-pdf-mixed-v020-fixed'
$report = 'E:\release-benchmark-pdf-mixed-v020-fixed-report.json'
if (Test-Path -LiteralPath $replayRoot) { throw "replay root already exists" }
if (Test-Path -LiteralPath $report) { throw "report already exists" }
$env:TEMP = 'E:\i2e-temp'
$env:TMP = 'E:\i2e-temp'
$env:IMAGE2EDITABLE_MODEL_CACHE = 'E:\image2editable-model-cache'
E:\i2e-release-py312\Scripts\python.exe -m scripts.release_benchmark --manifest .superpowers\pdf-mixed-page-sizes-manifest.json --workspace $replayRoot --report $report
```

Expected: 进程返回 0；报告中三次独立运行、两页均 `validated`，并且每页满足 `min_components=6`、`min_text_boxes=4`、`max_unexplained_pixels=0`、`max_quality_violations=0`，没有 warning 或 fallback。

- [ ] **Step 4: 更新项目状态并提交 case 13 证据计划**

更新 `Course.md`：记录修复提交、wheel 路径、fresh author/replay 路径、三次运行结果、原生斜杠断言、双页渲染和 overflow 结果；把 case 13 从“尚未完成”改成“严格完成”，同时保持剩余 8 个 case 的未完成描述准确。

提交前运行：

```powershell
git status --short
git diff --check
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_ocr_isolation.py tests\test_targeted_ocr.py tests\test_regressions.py tests\test_release_benchmark.py -q
```

Expected: 仅代码、测试和六份新计划属于可提交 repo 改动；模型、wheel、author/replay/QA 证据均留在 E 盘且不进入 Git。

```powershell
git add benchmarks\release\plans\pdf-mixed-page-sizes--component-page001-round-01-gpu.json benchmarks\release\plans\pdf-mixed-page-sizes--component-page001-round-02-gpu.json benchmarks\release\plans\pdf-mixed-page-sizes--component-page001-round-03-gpu.json benchmarks\release\plans\pdf-mixed-page-sizes--component-page002-round-01-gpu.json benchmarks\release\plans\pdf-mixed-page-sizes--component-page002-round-02-gpu.json benchmarks\release\plans\pdf-mixed-page-sizes--component-page002-round-03-gpu.json
git commit -m "基准：完成混合页面尺寸严格重放"
```

Expected: 本地提交成功；不执行 `git push`。
