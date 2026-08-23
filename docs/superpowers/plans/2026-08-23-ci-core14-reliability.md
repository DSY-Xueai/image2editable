# Core 14-Page CI and Release Evidence Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:verification-before-completion.

**Goal:** 修复核心 14 页发布链路的 CI 安装语义、installed wheel 验收边界和 manifest 跨平台 hash 绑定，不放宽质量门禁。

**Architecture:** Fast job 安装当前项目 metadata 后运行完整 model-free 源码回归；installed-wheel job 只验证真正的 wheel contract/smoke；release runner 对 manifest 使用 LF canonical bytes 计算身份，Release Gate 继续从 checkout 脚本消费已安装 wheel。

**Tech Stack:** GitHub Actions、setuptools、pytest、Python、SHA-256、JSON。

---

### Task 1: Lock the CI boundaries with failing tests

**Files:**
- Modify: `tests/test_ci_contract.py`
- Create: `tests/test_installed_package_contract.py`

- [ ] Add an assertion that fast CI installs the project metadata without changing the checkout import path.
- [ ] Add an assertion that installed-wheel CI runs only the dedicated installed-package contract after `installed_package_smoke`.
- [ ] Run the focused tests and confirm they fail against the current workflow.

### Task 2: Fix the CI installation semantics

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release-gate.yml`

- [ ] Add the minimal no-dependency project install to fast CI.
- [ ] Replace the installed-wheel full checkout suite with the dedicated contract invocation.
- [ ] Keep the full checkout suite in fast CI and the core benchmark in the protected release gate.
- [ ] Run workflow contract tests.

### Task 3: Make manifest identity line-ending independent

**Files:**
- Modify: `scripts/release_benchmark.py`
- Modify: `tests/test_release_benchmark.py`
- Modify: `tests/test_release_contract.py`

- [ ] Add a failing regression proving LF and CRLF copies of the same manifest have one identity.
- [ ] Implement one canonical manifest-byte helper and use it in the runner.
- [ ] Bind contract tests to the same helper and verify the existing core report remains comparable.

### Task 4: Verify the six-plan acceptance surface

**Files:**
- Modify: `Course.md` (ignored local handoff)

- [ ] Run focused CI, release, dependency, LaMa, runtime, and smoke tests.
- [ ] Run the complete local model-free suite and record environment-only failures separately.
- [ ] Build and inspect the wheel; verify no model caches or private paths are packaged.
- [ ] Do not create a tag or Release until GitHub required checks and the protected core gate are green.
