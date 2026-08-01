from __future__ import annotations

from pathlib import Path

import pytest

from image2editable import pptx_shadow_run


def test_run_shadow_replacements_replaces_valid_page_and_falls_back_per_page(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.pptx"
    source.write_bytes(b"source")
    output = tmp_path / "output.pptx"
    events = []

    def fake_build(image, donor, work_root, *, decision, lang):
        events.append(("build", Path(image).name, lang))
        if Path(image).name == "second.png":
            raise RuntimeError("quality gate")
        Path(donor).parent.mkdir(parents=True, exist_ok=True)
        Path(donor).write_bytes(b"donor")
        return {"quality": {"p99": 1.0}}

    def fake_patch(
        current,
        donor,
        staged,
        *,
        slide_part,
        source_shape_id,
    ):
        events.append(
            (
                "patch",
                Path(current).read_bytes(),
                slide_part,
                source_shape_id,
            )
        )
        Path(staged).write_bytes(Path(current).read_bytes() + b"+patched")
        return {"slide_part": slide_part}

    monkeypatch.setattr(
        pptx_shadow_run,
        "build_reconstruction_donor",
        fake_build,
    )
    monkeypatch.setattr(
        pptx_shadow_run,
        "patch_slide_background",
        fake_patch,
    )
    monkeypatch.setattr(
        pptx_shadow_run,
        "_validate_shadow_patch",
        lambda before, after, plan: events.append(
            ("validate", Path(after).read_bytes(), plan["page_id"])
        ),
    )

    result = pptx_shadow_run.run_shadow_replacements(
        source,
        output,
        [
            _plan(tmp_path, "page_001", "first.png", "slide1.xml"),
            _plan(tmp_path, "page_002", "second.png", "slide2.xml"),
        ],
        run_root=tmp_path,
        lang="ch",
    )

    assert output.read_bytes() == b"source+patched"
    assert [item["status"] for item in result["page_results"]] == [
        "replaced",
        "preserved_with_warning",
    ]
    assert result["page_results"][1]["warning"] == "quality gate"
    assert events == [
        ("build", "first.png", "ch"),
        ("patch", b"source", "ppt/slides/slide1.xml", "background"),
        ("validate", b"source+patched", "page_001"),
        ("build", "second.png", "ch"),
    ]


def test_run_shadow_replacements_preserves_legacy_conflict_page(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.pptx"
    source.write_bytes(b"source")
    output = tmp_path / "output.pptx"
    plan = _plan(tmp_path, "page_001", "first.png", "slide1.xml")
    plan["conflict_warning"] = "multiple Agent-approved screenshots on one page"
    monkeypatch.setattr(
        pptx_shadow_run,
        "build_reconstruction_donor",
        lambda *args, **kwargs: pytest.fail("conflict must not reconstruct"),
    )

    result = pptx_shadow_run.run_shadow_replacements(
        source,
        output,
        [plan],
        run_root=tmp_path,
    )

    assert output.read_bytes() == b"source"
    assert result["page_results"][0]["status"] == "preserved_with_warning"
    assert result["page_results"][0]["warning"] == plan["conflict_warning"]


def test_run_shadow_replacements_rejects_work_root_outside_run(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.pptx"
    source.write_bytes(b"source")
    output = tmp_path / "output.pptx"
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    plan = _plan(tmp_path, "page_001", "first.png", "slide1.xml")
    plan["work_root"] = str(outside)
    monkeypatch.setattr(
        pptx_shadow_run,
        "build_reconstruction_donor",
        lambda *args, **kwargs: pytest.fail("unsafe work root must not run"),
    )

    result = pptx_shadow_run.run_shadow_replacements(
        source,
        output,
        [plan],
        run_root=tmp_path,
    )

    assert output.read_bytes() == b"source"
    assert result["page_results"][0]["status"] == "preserved_with_warning"
    assert "work_root" in result["page_results"][0]["warning"]
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (outside / "donor.pptx").exists()


def _plan(
    tmp_path: Path,
    page_id: str,
    image_name: str,
    slide_name: str,
) -> dict:
    image = tmp_path / image_name
    image.write_bytes(b"image")
    page_root = tmp_path / "pages" / page_id
    page_root.mkdir(parents=True, exist_ok=True)
    return {
        "page_id": page_id,
        "slide_part": f"ppt/slides/{slide_name}",
        "image_path": str(image),
        "work_root": str(page_root / "reconstruction"),
        "decision": {
            "runtime_action": "shadow_run",
            "eligible_for_shadow_run": True,
            "source_shape_id": "background",
        },
    }


def test_shadow_page_consumes_component_result_without_cv(tmp_path, monkeypatch):
    from image2editable import pptx_shadow_run

    page = tmp_path / "run" / "pages" / "p1"
    page.mkdir(parents=True)
    work = page / "reconstruction"
    work.mkdir()
    result_file = work / "component_result.json"
    result_file.write_text("{}")
    called = []
    monkeypatch.setattr(
        pptx_shadow_run,
        "build_reconstruction_donor_from_result",
        lambda *a, **k: called.append(1) or {"components": 1},
    )
    monkeypatch.setattr(
        pptx_shadow_run,
        "patch_slide_background",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop after donor")),
    )
    source = tmp_path / "source.pptx"
    from pptx import Presentation

    Presentation().save(source)
    result, _ = pptx_shadow_run._run_page(
        source,
        tmp_path,
        tmp_path / "run",
        {
            "page_id": "p1",
            "work_root": str(work),
            "component_result_path": str(result_file),
            "source_screenshot_sha256": "x",
            "provider": "host",
            "image_path": "x",
            "decision": {},
        },
        "ch",
    )
    assert called == [1] and result["status"] == "preserved_with_warning"
