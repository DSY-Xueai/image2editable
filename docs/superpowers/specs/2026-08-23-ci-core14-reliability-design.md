# Core 14-Page CI and Release Evidence Reliability

## Goal

让核心 14 页发布链路在源码测试、installed wheel 验证和 Windows/Linux 换行环境下使用同一套严格门禁，并明确 30 页语料为非门禁扩展。

## Root causes

- Fast model-free CI imports checkout code without installing project metadata, while the CLI reads its version from package metadata.
- The installed-wheel matrix runs checkout-only benchmark/resource tests in a site-packages environment.
- Manifest evidence hashes raw JSON bytes, so Windows CRLF and Linux LF produce different release identities.

## Design

1. Fast model-free installs the project without dependencies after installing the pinned test dependencies, then runs the full model-free checkout suite.
2. Installed-wheel validates the wheel with the existing smoke script and a dedicated installed-package contract; source-only benchmark tests remain in the checkout suite and the release gate.
3. The release runner computes manifest identity from canonical LF bytes. The same helper is used by release contract tests, so reports and baselines are portable across checkout line endings.
4. The core gate remains the only v0.2 release benchmark: 10 cases, 14 pages, repeat=3, 42 page attempts. No warning, fallback, quality violation, missing page, or reopen failure is accepted.
5. The 18-case/30-page manifest remains optional extension material and is not a required CI or release condition.

## Verification

- New regression tests must fail before the implementation changes.
- Fast, installed-wheel, release contract, dependency, and LaMa adapter tests must pass locally.
- The GitHub matrix must be rerun before claiming CI is fixed.
- No tag or Release is created until the required checks and protected core gate pass.
