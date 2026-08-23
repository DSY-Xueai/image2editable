from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from image2editable import cli
from scripts import release_benchmark


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "benchmarks" / "release"


def test_cli_version_comes_from_installed_distribution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cli.metadata,
        "version",
        lambda name: calls.append(name) or "9.8.7",
    )

    with pytest.raises(SystemExit, match="0"):
        cli.build_parser().parse_args(["--version"])

    assert calls == ["image2editable"]
    assert capsys.readouterr().out.strip() == "image2editable 9.8.7"


def test_citation_version_matches_project_version() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert citation["version"] == "0.2.0"
    assert 'version = "0.2.0"' in project


def test_security_policy_uses_private_single_maintainer_process() -> None:
    policy = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    folded = policy.casefold()

    assert "0.2.x" in policy
    assert "private vulnerability reporting" in folded
    assert "public issue" in folded
    assert "48 hours" in folded
    assert "7 days" in folded
    assert "maintainers" not in folded


def test_release_workflow_only_creates_v020_draft_from_same_commit_gate() -> None:
    path = ROOT / ".github/workflows/release.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = path.read_text(encoding="utf-8")

    trigger = workflow.get("on", workflow.get(True))
    assert trigger == {"push": {"tags": ["v0.2.0"]}}
    assert workflow["permissions"] == {"actions": "read", "contents": "write"}
    assert set(workflow["jobs"]) == {"draft-release"}
    job = workflow["jobs"]["draft-release"]
    assert job["runs-on"] == "ubuntu-latest"
    commands = "\n".join(
        str(step.get("run", "")) for step in job["steps"] if isinstance(step, dict)
    )
    assert 'gh run list --workflow release-gate.yml --commit "$GITHUB_SHA"' in commands
    assert 'gh run download "$run_id" --name distribution' in commands
    assert 'gh run download "$run_id" --name core-v0.2-benchmark-report' in commands
    assert "CITATION.cff" in commands
    assert "SHA256SUMS" in commands
    assert "gh release create" in commands
    assert "--draft" in commands
    assert "--verify-tag" in commands
    assert "--latest=false" in commands
    assert "gh release edit" not in commands
    assert "publish" not in commands.casefold()
    assert "continue-on-error" not in raw
    for step in job["steps"]:
        if isinstance(step, dict) and "uses" in step:
            revision = str(step["uses"]).rsplit("@", 1)[-1]
            assert len(revision) == 40
            assert all(character in "0123456789abcdef" for character in revision)


def test_release_workflow_validates_passed_core_report() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert 'report["status"] != "passed"' in workflow
    assert 'report["totals"] != {"cases": 10, "failed_attempts": 0, "pages": 42}' in workflow
    assert 'report["repeat"] != 3' in workflow


def test_core_baseline_is_bound_to_manifest_and_runtime_constraints() -> None:
    baseline = json.loads((RELEASE_ROOT / "BASELINE.json").read_text(encoding="utf-8"))
    manifest_sha = release_benchmark.manifest_sha256(
        RELEASE_ROOT / "core-v0.2-manifest.json"
    )
    constraints_sha = release_benchmark.canonical_text_sha256(
        ROOT / "constraints" / "runtime.txt"
    )

    assert baseline["benchmark"] == "v0.2-core-14-page"
    assert baseline["manifest_sha256"] == manifest_sha
    assert baseline["constraints_sha256"] == constraints_sha
    assert baseline["environment"] == {
        "os": "Windows",
        "architecture": "AMD64",
        "python": "3.12",
        "device": "cuda",
    }
    assert baseline["median_total_duration_ms"] > 0
    assert set(baseline["case_median_duration_ms"]) == {
        "image-bilingual-dashboard",
        "image-combo-chart",
        "image-dark-poster",
        "image-flowchart",
        "image-icon-matrix",
        "image-non-16-9-infographic",
        "image-thin-line-network",
        "image-tiny-element-table",
        "pdf-rotated-page",
        "pptx-mixed-screenshot-candidates",
    }
