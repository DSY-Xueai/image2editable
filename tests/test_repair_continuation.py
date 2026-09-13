import hashlib
from pathlib import Path

import pytest

from image2editable import component_repair, runtime
from image2editable.execution import ExecutionLease
from image2editable.store import RunStore


@pytest.mark.parametrize("resumable", [True, False])
def test_legacy_warning_is_repaired_before_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resumable: bool,
) -> None:
    store = RunStore(tmp_path)
    manifest = {"schema_version": 1, "options": {"pipeline_mode": "fast"}}
    store.write_json("page_jobs.json", {"schema_version": 1, "pages": {
        "page_001": {"schema_version": 1, "status": "preserved_with_warning"},
    }})
    store.write_json("pages/page_001/reconstruction/component_state.json", {})
    monkeypatch.setattr(runtime, "_native_pdf_analysis", lambda *args: {})
    monkeypatch.setattr(runtime, "_page_performance_trace", lambda *args: None)
    monkeypatch.setattr(runtime, "_batch_legacy_ocr", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runtime, "resume_round_limited_component_repair", lambda *args: resumable,
    )

    def advance(*args, **kwargs):
        assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == (
            "processing"
        )
        return {"status": "ready_for_assembly", "page_id": "page_001"}

    monkeypatch.setattr(runtime, "advance_legacy_page", advance)
    with ExecutionLease(tmp_path / "execution.lock", run_root=tmp_path) as lease:
        if not resumable:
            with pytest.raises(RuntimeError, match="could not resume page page_001"):
                runtime._advance_legacy_pages(store, manifest, ["page_001"], lease)
        else:
            assert runtime._advance_legacy_pages(
                store, manifest, ["page_001"], lease,
            ) is None
            assert store.read_json("page_jobs.json")["pages"]["page_001"]["status"] == (
                "validated"
            )


@pytest.mark.parametrize("changed_output", [None, "background", "rgba"])
@pytest.mark.parametrize("resumed", [False, True])
def test_output_cycle_is_not_progress_despite_newly_refrozen_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_output: str | None, resumed: bool,
) -> None:
    store = RunStore(tmp_path)

    def artifact(path, document):
        store.write_json(path, document)
        return {"path": path, "sha256": hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()}

    def quality(round_number, changed):
        presentation = artifact(f"round-{round_number}/presentation.json", {"components": [{
            "component_id": "component_a",
            **{name: {"path": f"round-{round_number}/{name}.png", "sha256": (
                "b" * 64 if changed == name else "a" * 64
            )} for name in (
                "rgba", "ownership_mask", "presentation_alpha_mask", "generated_underlay_mask",
            )},
        }]})
        return artifact(f"round-{round_number}/quality.json", {
            "input_refs": {
                "presentation_manifest": presentation,
                **{name: {"path": f"round-{round_number}/{name}.png", "sha256": (
                    "b" * 64 if changed == name else "a" * 64
                )} for name in ("background", "reconstructed", "text_mask", "native_check")},
            },
            "report": {"violations": ["unexplained_visual_residual"]},
        })

    previous = quality(1, None)
    current = quality(3, changed_output)
    request_path = "pages/page_001/reconstruction/agent/round-03/component_agent_request.json"
    prior_request = {
        "evidence": {"quality-report.json": {
            "path": "quality-report.json", "sha256": previous["sha256"],
        }},
    }
    store.write_json(
        "pages/page_001/reconstruction/agent/round-02/quality-report.json",
        store.read_json(previous["path"]),
    )
    state = {
        "repair_round": 3, "stop_reason": "no_quality_improvement",
        "failed_ids": ["component_a"],
        "current_round": {"request_ref": {"path": request_path}, "quality_ref": current},
        "round_history": [{
            "round": 1, "quality_sha256": previous["sha256"],
            "failed_ids": ["component_a"], "frozen_ids": [],
        }, {
            "round": 3, "quality_sha256": current["sha256"],
            "failed_ids": ["component_a"], "frozen_ids": ["component_b"],
        }],
    }
    if resumed:
        state.update(phase="freeze_committed", status="active")
    monkeypatch.setattr(component_repair, "load_component_agent_request", lambda path: prior_request)

    assert component_repair._repeated_component_output(store, state) is (changed_output is None)
    assert component_repair._next_round_progress_allowed(store, state) is (
        resumed or changed_output is not None
    )


def test_resumed_round_still_rejects_same_plan_on_identical_inputs(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    state = {
        "repair_round": 3, "phase": "awaiting_plan", "status": "active",
        "current_round": {"request_ref": {"path": "agent/round-03/request.json"}},
        "round_history": [{"round": 2, "normalized_plan_sha256": "a" * 64}],
    }
    monkeypatch.setattr(component_repair, "load_component_agent_request", lambda path: {})
    monkeypatch.setattr(component_repair, "_component_request_inputs", lambda *args: {"source": "same"})
    assert component_repair._repeated_component_plan(store, state, {}, "a" * 64)
    assert not component_repair._repeated_component_plan(store, state, {}, "b" * 64)
