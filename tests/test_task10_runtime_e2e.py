from __future__ import annotations

import json
from pathlib import Path

from image2editable.agent import record_decision
from image2editable.pptx_input import prepare_pptx_job

from test_agent_decision import _candidate_pptx


def test_task10_pptx_approval_is_the_only_page_request_gate(tmp_path: Path) -> None:
    source, _ = _candidate_pptx(tmp_path)
    run_dir = prepare_pptx_job(source, run_dir=tmp_path / "run")
    page = run_dir / "pages/page_001"
    assert not (page / "page_request.json").exists()

    rejected = record_decision(
        run_dir,
        page_id="page_001",
        object_id="2",
        decision="preserve",
        confidence=0.99,
        category="full_slide_screenshot",
        evidence=["native object retained"],
    )
    assert rejected["eligible_for_shadow_run"] is False
    assert not (page / "page_request.json").exists()

    # A fresh run models the approved path without hand-written hashes.
    approved_run = prepare_pptx_job(source, run_dir=tmp_path / "approved")
    approved = record_decision(
        approved_run,
        page_id="page_001",
        object_id="2",
        decision="replace",
        confidence=0.99,
        category="full_slide_screenshot",
        evidence=["full-slide screenshot"],
    )
    request = json.loads(
        (approved_run / "pages/page_001/page_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert approved["eligible_for_shadow_run"] is True
    assert request["sha256"] == approved["image_sha256"]
    assert Path(approved_run / request["source"]).is_file()
