# v0.2 Core 14-Page Benchmark Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不放宽 warning、fallback、质量阈值或 `repeat=3` 的前提下，完成 v0.2 核心 14 页 benchmark、本地发布候选门禁和六份发布可靠性计划的剩余工作；本地全部完成并复审后才请求 push 授权。

**Architecture:** 保留 `benchmarks/release/manifest.json` 作为 18 case / 30 页扩展语料库，新增精确子集 `core-v0.2-manifest.json` 作为 10 case / 14 页发布门禁。Host runner 仍从已安装 wheel 执行真实模型流程并固定三次独立重放。性能只聚合已通过的三次结果，以相同环境的中位数建立基线；环境不一致时只报告，不进行伪比较。GitHub 普通 CI 继续 model-free，核心真实模型门禁使用单独受保护、显式 opt-in 的 job。发布 workflow 只创建 draft，tag、push、外部安全设置和公开 Release 均需要单独授权。

**Tech Stack:** Python 3.12、pytest、JSON、GitHub Actions、PowerPoint COM、python-pptx、Pillow、现有 `scripts.release_benchmark`、本地 E 盘 runtime/model cache。

---

## Phase A：全部本地工作（不得 push）

### Task 1: 用 TDD 固定核心 manifest 的精确组成

**Files:**
- Modify: `tests/test_release_benchmark.py`
- Create: `benchmarks/release/core-v0.2-manifest.json`

- [ ] **Step 1: 写核心 manifest RED 测试**

在现有常量旁增加：

```python
CORE_MANIFEST_PATH = RELEASE_ROOT / "core-v0.2-manifest.json"
CORE_CASE_IDS = (
    "image-bilingual-dashboard",
    "image-combo-chart",
    "image-flowchart",
    "image-icon-matrix",
    "image-thin-line-network",
    "image-tiny-element-table",
    "image-dark-poster",
    "image-non-16-9-infographic",
    "pdf-rotated-page",
    "pptx-mixed-screenshot-candidates",
)
```

增加独立读取函数和测试：

```python
def _core_manifest() -> dict[str, object]:
    return json.loads(
        CORE_MANIFEST_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def test_core_v020_manifest_is_exact_subset_of_extended_corpus() -> None:
    extended = _manifest()
    core = _core_manifest()
    selected = {case["id"]: case for case in extended["cases"]}

    _assert_exact_fields(core)
    _assert_numeric_contract(core)
    assert [case["id"] for case in core["cases"]] == list(CORE_CASE_IDS)
    assert core["cases"] == [selected[case_id] for case_id in CORE_CASE_IDS]
    assert core["categories"] == [case["categories"][0] for case in core["cases"]]
    assert len(core["cases"]) == 10
    assert sum(case["page_count"] for case in core["cases"]) == 14
    assert {case["kind"]: sum(item["kind"] == case["kind"] for item in core["cases"])
            for case in core["cases"]} == {"image": 8, "pdf": 1, "pptx": 1}
```

另加一个 core duplicate-key mutant，确认根、case、page 任何重复 key 均 fail closed。原扩展 manifest 的 18 case / 30 页测试保持不变。

- [ ] **Step 2: 运行 RED**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_release_benchmark.py -k "core_v020" -q
```

Expected: FAIL，仅因 `core-v0.2-manifest.json` 尚不存在。

- [ ] **Step 3: 创建核心 manifest**

从扩展 manifest 按 `CORE_CASE_IDS` 顺序逐对象原样复制；根字段严格为 `schema_version`、`cases`、`categories`。不得修改路径、SHA、阈值或页面 ID，不复制输入文件。

- [ ] **Step 4: 运行 GREEN 与扩展语料回归**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_release_benchmark.py -k "manifest" -q
```

Expected: PASS；扩展仍为 18/30，核心为 10/14。

### Task 2: 更新 README 命名但保留完整语料事实

**Files:**
- Modify: `tests/test_release_benchmark.py`
- Modify: `benchmarks/release/README.md`

- [ ] **Step 1: 先更新文档契约测试**

保留现有字体、来源、canonical bytes 和严格质量说明；把变化限制为以下事实，并同步到 `EXPECTED_README`：

```text
本目录包含 18 个输入、30 页的扩展语料库。
v0.2 发布门禁使用 core-v0.2-manifest.json：10 个完整 case、14 页。
核心组成是 8 张图片、完整 2 页 pdf-rotated-page、完整 4 页 pptx-mixed-screenshot-candidates。
```

核心命令必须精确为：

```bash
python -m scripts.release_benchmark --manifest benchmarks/release/core-v0.2-manifest.json --workspace <fresh-workspace> --report <fresh-workspace>/benchmark-report.json
```

扩展命令保留为：

```bash
python -m scripts.release_benchmark --manifest benchmarks/release/manifest.json --workspace <fresh-workspace> --report <fresh-workspace>/benchmark-report.json
```

完成声明在最终批量报告通过前写“正在验证”，通过后才改为“已严格通过”。30 页只能称为“扩展语料库”，不得写成 30/30 已通过。

- [ ] **Step 2: 运行 RED，最小更新 README，再运行 GREEN**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_release_benchmark.py -k "readme" -q
```

Expected: 首次 FAIL、README 同步后 PASS。

### Task 3: 给严格报告增加三次中位数，不让性能掩盖质量失败

**Files:**
- Modify: `scripts/release_benchmark.py`
- Modify: `tests/test_release_benchmark.py`
- Create after real run: `benchmarks/release/BASELINE.json`

- [ ] **Step 1: 写性能聚合 RED 测试**

增加纯函数契约：

```python
def test_performance_summary_uses_three_passed_repeats() -> None:
    attempts = [
        {"case_id": "a", "repeat": 1, "status": "passed", "duration_ms": 300},
        {"case_id": "a", "repeat": 2, "status": "passed", "duration_ms": 100},
        {"case_id": "a", "repeat": 3, "status": "passed", "duration_ms": 200},
        {"case_id": "b", "repeat": 1, "status": "passed", "duration_ms": 30},
        {"case_id": "b", "repeat": 2, "status": "passed", "duration_ms": 10},
        {"case_id": "b", "repeat": 3, "status": "passed", "duration_ms": 20},
    ]
    assert runner.aggregate_performance(attempts) == {
        "repeat_total_duration_ms": [330, 110, 220],
        "median_total_duration_ms": 220,
        "case_median_duration_ms": {"a": 200, "b": 20},
    }
```

失败 attempt、缺重复、bool/负耗时或重复 case/repeat 必须抛 `BenchmarkFailure("invalid_performance_result")`。`run_manifest()` 只有在所有质量 attempt 通过后才写入 `performance`；任一质量失败时报告仍为 `failed`，不能产生可接受 baseline。

- [ ] **Step 2: 运行 RED**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_release_benchmark.py -k "performance_summary" -q
```

- [ ] **Step 3: 实现最小聚合**

使用标准库 `statistics.median`；由于固定三次，结果必须为整数。报告只新增耗时聚合，不加入输入路径、OCR 文本、prompt 或模型响应。

- [ ] **Step 4: 固定 baseline schema**

测试要求 `BASELINE.json` 只含：

```json
{
  "schema_version": 1,
  "benchmark": "v0.2-core-14-page",
  "manifest_sha256": "<64hex>",
  "constraints_sha256": "<64hex>",
  "environment": {
    "os": "Windows",
    "architecture": "AMD64",
    "python": "3.12",
    "device": "cuda"
  },
  "median_total_duration_ms": 0,
  "case_median_duration_ms": {}
}
```

实际数字只能从 Task 7 的通过报告写入，不能预填或估算。manifest、constraints 或 environment 不同只报告 `not_comparable`，不得把不同设备的耗时当回归。

### Task 4: 把核心 benchmark 接到受保护的 Release Gate

**Files:**
- Modify: `tests/test_ci_contract.py`
- Modify: `.github/workflows/release-gate.yml`

- [ ] **Step 1: 写 workflow RED 测试**

扩展手工输入：

```python
"run_core_benchmark": {
    "description": "Run protected v0.2 core 14-page benchmark",
    "required": True,
    "type": "boolean",
    "default": False,
}
```

新 job `core-benchmark` 必须满足：Ubuntu/Python 3.12、`needs: build-distribution`、`if: inputs.run_core_benchmark`、environment `core-benchmark`、`timeout-minutes: 360`；安装同一 wheel、Tesseract 和 runtime models，运行 doctor 后执行：

```bash
python "${{ github.workspace }}/scripts/release_benchmark.py" --manifest "${{ github.workspace }}/benchmarks/release/core-v0.2-manifest.json" --workspace "${{ runner.temp }}/core-v0.2-benchmark/workspace" --report "${{ runner.temp }}/core-v0.2-benchmark/report.json"
```

报告用固定 SHA 的 `actions/upload-artifact` 上传，名称为 `core-v0.2-benchmark-report`，`if-no-files-found: error`。仅报告上传 step 允许 `if: always()`；benchmark step 不允许 `continue-on-error`、shell 吞错或宽泛 skip。

- [ ] **Step 2: 运行 RED，最小修改 workflow，再运行 GREEN**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_ci_contract.py -k "release_gate or core_benchmark" -q
```

GitHub environment/secret 的实际创建是外部写操作，本阶段只提交 fail-closed workflow；外部配置留到 Phase B 并再次确认授权。

### Task 5: 为四页 PPTX fresh author 并提交严格 Host plans

**Files:**
- Create: `benchmarks/release/plans/pptx-mixed-screenshot-candidates--*.json`
- Modify only if a reproducible product defect is exposed: proven source/test pair
- Update ignored local file: `Course.md`

- [ ] **Step 1: 从当前候选 wheel 和全新 E 盘 root author**

固定环境：

```powershell
$env:HF_HOME='E:\image2editable-model-cache\huggingface'
$env:IMAGE2EDITABLE_MODEL_CACHE='E:\image2editable-model-cache'
$env:TEMP='E:\image2editable-temp'
$env:TMP='E:\image2editable-temp'
E:\i2e-release-py312\Scripts\image2editable.exe prepare benchmarks\release\inputs\18-mixed-screenshot-candidates.pptx -o E:\i2e-core14-author-pptx\output.pptx --run-dir E:\i2e-core14-author-pptx\run --agent-provider host --slide-size original
```

随后严格循环 `run next`、必要的四页 screenshot candidate `decision record`、`run execute`、`agent next`、审查证据、`agent record --plan`、`run execute`，直到 completed。每份组件计划必须精确绑定 `page_id`、`repair_round`、`request_sha256`、`graph_sha256`；不得使用 accept-all、只换 hash、warning 或 fallback。

- [ ] **Step 2: 每轮计划做离线 schema/graph 审核**

用现有 `_select_component_plan()` 对真实 request 唯一命中；验证 action 对象存在、graph transition 有效、最终 `failed_ids=[]`、warning null、fallback none、visual/text 达到 manifest 下限。

- [ ] **Step 3: 用 fresh root 单独严格 repeat=3**

使用只包含该完整 PPTX case、但对象与 core manifest 完全相同的本地审计 manifest；运行现有 runner，三次 workspace 均必须新建。报告必须是 3 attempts、12 page attempts、0 failed attempts。

- [ ] **Step 4: PowerPoint 原生 QA**

用 `image2editable.powerpoint_renderer.PowerPointRenderer` 将三份四页输出按原始画布尺寸导出 PNG；逐页检查文本、candidate 决策、原生对象保留、视觉组件、对象边界和三次渲染一致性。再用 `python-pptx` reopen，确认 4 slides、无越界。渲染证据留 E 盘，不进 Git。

### Task 6: 对旋转 PDF 做独立严格重放与 PowerPoint QA

**Files:**
- Existing plans: `benchmarks/release/plans/pdf-rotated-page--*.json`
- Modify only if fresh evidence proves a defect
- Update ignored local file: `Course.md`

- [ ] **Step 1: 从新 root 执行该完整双页 case 的 repeat=3**

使用与 core manifest 对象完全相同的本地单-case审计 manifest。三次必须分别 prepare、真实推理、Host handshake 和计划选择；报告必须是 3 attempts、6 page attempts、0 failed attempts。

- [ ] **Step 2: 校验方向与质量**

每次两页均 `validated`；文本框不少于 11，旋转页文本方向为 90 度，visual 不低于 manifest，warning null、fallback none、0 unexplained、0 quality violation。

- [ ] **Step 3: PowerPoint 原生 QA**

原生导出三份两页结果，确认页面比例、方向、文本内容、对象边界、三次渲染一致性和 python-pptx reopen。任何差异先按 systematic-debugging 找根因，不能调整门槛规避。

### Task 7: 从空 workspace 跑完整核心 14 页并冻结真实性能基线

**Files:**
- Modify with measured data: `benchmarks/release/BASELINE.json`
- Modify after pass: `benchmarks/release/README.md`
- Modify expected doc contract: `tests/test_release_benchmark.py`
- Update ignored local file: `Course.md`

- [ ] **Step 1: 构建并安装唯一候选 wheel**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m build --outdir E:\i2e-core14-build
E:\i2e-release-py312\Scripts\python.exe -m twine check E:\i2e-core14-build\*
E:\i2e-release-py312\Scripts\python.exe -m pip install --force-reinstall --no-deps E:\i2e-core14-build\image2editable-0.2.0-py3-none-any.whl
E:\i2e-release-py312\Scripts\python.exe -m pip check
```

检查 wheel 不包含模型、cache、benchmark 输出或私人路径。

- [ ] **Step 2: 跑完整核心 strict repeat=3**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m scripts.release_benchmark --manifest benchmarks\release\core-v0.2-manifest.json --workspace E:\i2e-core14-final\workspace --report E:\i2e-core14-final\report.json
```

Expected: `status=passed`、10 case、42 page attempts、0 failed attempts；所有 attempt 都无 warning/fallback，严格阈值不变。

- [ ] **Step 3: 从报告写入真实 baseline**

只复制报告的真实 `manifest_sha256`、三次中位数和各 case 中位数；constraints SHA 现场计算；environment 写实际 Python/OS/architecture/device。随后运行 baseline schema 和比较测试。

- [ ] **Step 4: 只有通过后更新完成措辞**

把 README 从“正在验证”改为“v0.2 核心 14 页 benchmark 已严格通过”，同时明确“30 页扩展语料库尚未全部严格重放”。同步 `EXPECTED_README`。

### Task 8: 补齐 v0.2 本地发布契约，不提前伪造远端结果

**Files:**
- Create: `tests/test_release_contract.py`
- Modify: `image2editable/cli.py`
- Create: `CITATION.cff`
- Create: `.github/workflows/release.yml`
- Modify minimally: `README.md`, `README_EN.md`
- Keep and test: `SECURITY.md`
- Create only after evidence is final: `RELEASE_NOTES_v0.2.0.md`
- Update ignored local file: `Course.md`

- [ ] **Step 1: 用 RED 测试锁定单一版本源**

`pyproject.toml` 已为 0.2.0。测试要求 `image2editable --version` 从 `importlib.metadata.version("image2editable")` 读取；`CITATION.cff` version 为 0.2.0；不得新增第二份 `__version__`。

- [ ] **Step 2: 测试 SECURITY 和 release workflow**

SECURITY 必须使用单人维护语气、禁止公开提交漏洞、说明 GitHub Private Vulnerability Reporting、确认时限和受支持版本。`.github/workflows/release.yml` 只接受 `v0.2.0` tag，校验 tag/wheel/CITATION 一致，消费同一 commit 的 release-gate artifacts，生成 `SHA256SUMS`，只创建 draft，不自动 publish；Actions 全部锁定完整 SHA。

- [ ] **Step 3: 写只包含已验证事实的 release notes**

记录核心 14/14 严格 repeat=3、30 页是扩展语料而非 30/30 完成、实际性能中位数、三平台/Python 支持边界、本地模型不进入 Git/wheel、安装/doctor 命令和 PowerPoint/native-shape 已知限制。GitHub URL 和远端状态在 Phase B 实际产生后再补；不写 placeholder、不写尚未通过。

- [ ] **Step 4: 聚焦回归**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest tests\test_release_contract.py tests\test_runtime_cli.py tests\test_ci_contract.py tests\test_release_benchmark.py -q
```

### Task 9: 全仓验证、镜像一致性、代码审查和本地提交

**Files:**
- Modify only files directly proven by failures
- Update ignored local file: `Course.md`

- [ ] **Step 1: 全仓验证**

Run:

```powershell
E:\i2e-release-py312\Scripts\python.exe -m pytest -q
E:\i2e-release-py312\Scripts\python.exe -m pip check
E:\i2e-release-py312\Scripts\python.exe -m compileall -q image2editable scripts skills\image-to-ppt\scripts
git diff --check main...HEAD
```

运行根脚本与 standalone skill 的现有字节镜像测试；候选 wheel 在 checkout 外安装并执行 installed-package smoke、`image2editable --version`、`models status`、`doctor`。

- [ ] **Step 2: 更新 Course.md**

删除 12/18、30 页发布门禁等过时表述，记录核心 10 case /14 页、最终报告路径/摘要、实际性能、候选 wheel SHA、关键文件、运行入口和未 push/未发布事实。

- [ ] **Step 3: 使用 requesting-code-review**

审查六份计划的剩余验收点、核心 manifest、workflow、安全/发布边界、真实性能和所有本轮 diff。Critical/Important 必须修复并重跑受影响测试及全仓验证。

- [ ] **Step 4: 评估工作树后本地提交**

先展示 `git status --short` 与 `git diff --stat`，确认无模型、私有 evidence、临时报告和无关改动。允许拆成少量可审计中文 commit；不得 push。

---

## Phase B：本地全部完成后，仍需用户单独授权

### Task 10: Push、GitHub 门禁与 v0.2.0 draft Release

**Files:**
- No local source changes unless a remote failure has a reproducible root cause

- [ ] **Step 1: 请求并取得明确 push 授权**

展示 `git status -sb`、`git log --oneline origin/main..HEAD`、核心报告摘要、wheel SHA 和本地测试结果。没有明确授权不得 push。

- [ ] **Step 2: Push 后等待普通 CI 与 Release Gate**

普通 CI fast/build/五组合与手工 release-gate 九组合必须全绿。受保护 core-benchmark 必须使用 core manifest 并产出 10 case /42 page attempts /0 failed attempts 的 artifact。失败不得 `continue-on-error`、删平台、放宽 warning 或降低质量门槛。

- [ ] **Step 3: 核对外部安全设置**

先只读确认 GitHub Private Vulnerability Reporting 是否已启用；若未启用，展示准确仓库和设置并再次请求外部写授权。未启用前不得把 SECURITY 的私密入口宣称为可用。

- [ ] **Step 4: 远端失败只按根因修复**

保存 workflow URL、runner/OS/Python、首个产品 traceback；本地先新增 RED，再做外科修复，重跑聚焦与全仓验证，提交后重新请求 push 授权。

- [ ] **Step 5: tag 与 draft Release 分别授权**

所有门禁通过后，先请求创建 annotated tag `v0.2.0` 和 push tag 的授权。release workflow 只创建 draft；下载并校验 wheel、sdist、SHA256SUMS、benchmark report，从空 venv 干净安装。公开 draft 还要再取得一次明确授权。

---

## 完成定义

本地完成必须同时具备：核心 manifest 10/14 契约通过；PPTX 与 PDF fresh author/严格 replay/PowerPoint QA 通过；整个核心 10 case 三次独立重放共 42 page attempts 全部通过；真实性能中位数进入绑定环境的 baseline；全仓测试、wheel、pip、镜像和 diff-check 全绿；Course.md 与发布文档没有把 30 页扩展语料写成已完成；代码审查无 Critical/Important；工作树仅含预期提交。

项目最终发布完成还必须具备：用户授权 push；GitHub 普通 CI、九组合安装、真实模型 smoke 和核心 benchmark 全绿；安全私密报告入口可用；tag/draft artifacts 校验通过；用户再次授权公开 Release。
