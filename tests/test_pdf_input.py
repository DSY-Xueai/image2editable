from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import NameObject, NumberObject, RectangleObject
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

from image2editable.inputs import validate_pptx_output_path
from image2editable.inputs import prepare_image_job
from image2editable.resources import safe_default_policy
from image2editable.store import RunStore


def _pdf_input():
    import image2editable.pdf_input

    return image2editable.pdf_input


def _write_two_page_pdf(path: Path) -> None:
    document = canvas.Canvas(str(path), pagesize=A4)
    document.drawString(72, 72, "portrait")
    document.showPage()
    document.setPageSize(landscape(A4))
    document.drawString(72, 72, "landscape")
    document.save()


def _write_small_pdf(path: Path) -> None:
    document = canvas.Canvas(str(path), pagesize=(288, 144))
    document.drawString(24, 24, "small")
    document.save()


def _write_same_ratio_pdf(path: Path) -> None:
    document = canvas.Canvas(str(path), pagesize=A4)
    document.showPage()
    document.save()


def _write_same_physical_ratio_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=36)
    writer.add_blank_page(width=73, height=36.5)
    with path.open("wb") as stream:
        writer.write(stream)


def _write_inherited_boxes_pdf(path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=400)
    pages = writer.get_object(writer._pages)
    pages[NameObject("/MediaBox")] = RectangleObject([0, 0, 200, 400])
    pages[NameObject("/CropBox")] = RectangleObject([10, 20, 190, 380])
    del page[NameObject("/MediaBox")]
    page[NameObject("/Rotate")] = NumberObject(90)
    with path.open("wb") as stream:
        writer.write(stream)


def _write_direct_media_inherited_crop_pdf(path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=400)
    pages = writer.get_object(writer._pages)
    pages[NameObject("/CropBox")] = RectangleObject([10, 20, 190, 380])
    pages[NameObject("/Rotate")] = NumberObject(90)
    with path.open("wb") as stream:
        writer.write(stream)


def _assert_detail_not_consumed(run: Path, page_id: str = "page_001") -> None:
    store = RunStore.open(run)
    request = store.read_json(f"pages/{page_id}/page_request.json")
    history = store.read_json(f"pages/{page_id}/render_history.json")
    assert request["source"] == f"pages/{page_id}/source.png"
    assert history["detail_used"] is False
    assert len(history["renders"]) == 1
    assert not (run / f"pages/{page_id}/source_detail.png").exists()
    assert not (run / f"pages/{page_id}/source_detail.png.tmp").exists()


def test_pdf_input_module_is_available() -> None:
    assert importlib.util.find_spec("image2editable.pdf_input") is not None


def test_pdf_page_count_returns_page_total(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)

    assert _pdf_input().pdf_page_count(source) == 2


def test_pdf_page_count_translates_corrupt_encrypted_and_zero_page_files(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a PDF")
    encrypted = tmp_path / "encrypted.pdf"
    encrypted_writer = PdfWriter()
    encrypted_writer.add_blank_page(width=100, height=100)
    encrypted_writer.encrypt("secret")
    with encrypted.open("wb") as stream:
        encrypted_writer.write(stream)
    empty = tmp_path / "empty.pdf"
    with empty.open("wb") as stream:
        PdfWriter().write(stream)

    with pytest.raises(ValueError, match="Cannot open PDF"):
        _pdf_input().pdf_page_count(corrupt)
    with pytest.raises(ValueError, match="encrypted"):
        _pdf_input().pdf_page_count(encrypted)
    with pytest.raises(ValueError, match="Cannot open PDF"):
        _pdf_input().pdf_page_count(empty)


def test_pdf_page_count_translates_non_pdfium_open_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)

    def fail_open(*args, **kwargs):
        raise OSError("open failed")

    monkeypatch.setattr(_pdf_input().pdfium, "PdfDocument", fail_open)

    with pytest.raises(ValueError, match="Cannot open PDF") as error:
        _pdf_input().pdf_page_count(source)

    assert isinstance(error.value.__cause__, OSError)


def test_pdf_page_count_translates_len_error_and_closes_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)

    class FailingDocument:
        closed = False

        def __len__(self) -> int:
            raise OSError("length failed")

        def close(self) -> None:
            self.closed = True

    document = FailingDocument()
    monkeypatch.setattr(_pdf_input().pdfium, "PdfDocument", lambda *args: document)

    with pytest.raises(ValueError, match="Cannot open PDF") as error:
        _pdf_input().pdf_page_count(source)

    assert isinstance(error.value.__cause__, OSError)
    assert document.closed is True


def test_validate_pptx_output_path_rejects_source_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.pptx"
    source.write_bytes(b"source")
    alias = tmp_path / "alias.pptx"
    os.link(source, alias)

    with pytest.raises(ValueError, match="overwrites source"):
        validate_pptx_output_path(
            alias, source_paths=[source], run_root=tmp_path / "run"
        )


def test_prepare_pdf_job_copies_and_renders_pages_in_order(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)

    run = _pdf_input().prepare_pdf_job(
        source, run_dir=tmp_path / "run", slide_size="16:9", lang="en"
    )
    copied = run / "input" / "original.pdf"
    manifest = json.loads((run / "job_manifest.json").read_text(encoding="utf-8"))
    first_request = json.loads(
        (run / "pages" / "page_001" / "page_request.json").read_text(
            encoding="utf-8"
        )
    )
    first_history = json.loads(
        (run / "pages" / "page_001" / "render_history.json").read_text(
            encoding="utf-8"
        )
    )

    assert copied.read_bytes() == source.read_bytes()
    assert manifest["input"] == {
        "type": "pdf",
        "original_path": str(source.resolve()),
        "source": "input/original.pdf",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "page_count": 2,
        "page_ratios_equal": False,
        "page_aspect_ratio": None,
    }
    assert manifest["options"] == {
        "agent_provider": "host",
        "lang": "en",
        "slide_size": "16:9",
        "output_path": None,
        "resource_policy": safe_default_policy(),
    }
    assert manifest["pages"] == ["page_001", "page_002"]
    assert first_request["source_type"] == "pdf"
    assert first_request["source"] == "pages/page_001/source.png"
    assert first_request["sha256"] == first_request["render"]["sha256"]
    assert first_history == {
        "schema_version": 1,
        "renders": [first_request["render"]],
        "detail_used": False,
    }
    assert RunStore.open(run).read_json("run_state.json")["status"] == "prepared"


@pytest.mark.parametrize("agent_provider", ["host", "local"])
def test_prepare_pdf_job_freezes_agent_provider(
    tmp_path: Path, agent_provider: str
) -> None:
    source = tmp_path / "source.pdf"
    _write_small_pdf(source)

    run = _pdf_input().prepare_pdf_job(
        source,
        run_dir=tmp_path / agent_provider,
        agent_provider=agent_provider,
    )

    assert RunStore.open(run).read_json("job_manifest.json")["options"][
        "agent_provider"
    ] == agent_provider
    assert RunStore.open(run).read_json("page_jobs.json")["pages"]["page_001"]["status"] == "pending"


def test_rerender_pdf_page_activates_one_higher_detail_render(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = _pdf_input().prepare_pdf_job(source, run_dir=tmp_path / "run")

    result = _pdf_input().rerender_pdf_page(run, "page_001")
    request = RunStore.open(run).read_json("pages/page_001/page_request.json")
    history = RunStore.open(run).read_json("pages/page_001/render_history.json")

    assert result == {"detail_used": True, "activated": True}
    assert request["source"] == "pages/page_001/source_detail.png"
    assert (run / request["source"]).is_file()
    assert len(history["renders"]) == 2
    assert history["renders"][-1]["profile"] == "detail"
    assert history["detail_used"] is True
    with pytest.raises(RuntimeError, match="already used"):
        _pdf_input().rerender_pdf_page(run, "page_001")


def test_rerender_pdf_page_consumes_detail_when_not_higher(tmp_path: Path) -> None:
    source = tmp_path / "small.pdf"
    _write_small_pdf(source)
    run = _pdf_input().prepare_pdf_job(source, run_dir=tmp_path / "run")

    result = _pdf_input().rerender_pdf_page(run, "page_001")
    request = RunStore.open(run).read_json("pages/page_001/page_request.json")
    history = RunStore.open(run).read_json("pages/page_001/render_history.json")

    assert result == {"detail_used": True, "activated": False}
    assert request["source"] == "pages/page_001/source.png"
    assert not (run / "pages/page_001/source_detail.png").exists()
    assert history["detail_used"] is True
    assert history["renders"][-1]["result"] == "detail_not_higher"
    assert history["renders"][-1]["source"] is None


def test_rerender_pdf_page_rejects_replaced_prepared_source_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pdf_input()
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = module.prepare_pdf_job(source, run_dir=tmp_path / "run")
    (run / "input/original.pdf").write_bytes(b"replaced")

    def unexpected_render(*args, **kwargs):
        raise AssertionError("invalid PDF detail source reached renderer")

    monkeypatch.setattr(
        module, "_render_pdf_page_from_stream", unexpected_render
    )

    with pytest.raises(RuntimeError, match="PDF detail source.*hash"):
        module.rerender_pdf_page(run, "page_001")

    _assert_detail_not_consumed(run)


@pytest.mark.parametrize(
    "source_kind",
    ["absolute", "parent", "duplicate_separator", "non_string", "not_pdf"],
)
def test_rerender_pdf_page_rejects_invalid_manifest_source_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    module = _pdf_input()
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = module.prepare_pdf_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run)
    manifest = store.read_json("job_manifest.json")
    if source_kind == "absolute":
        manifest["input"]["source"] = str(source.resolve())
    elif source_kind == "parent":
        manifest["input"]["source"] = "input/../input/original.pdf"
    elif source_kind == "duplicate_separator":
        manifest["input"]["source"] = "input//original.pdf"
    elif source_kind == "non_string":
        manifest["input"]["source"] = True
    else:
        alias = run / "input/original.txt"
        alias.write_bytes((run / "input/original.pdf").read_bytes())
        manifest["input"]["source"] = "input/original.txt"
    store.write_json("job_manifest.json", manifest)

    def unexpected_render(*args, **kwargs):
        raise AssertionError("invalid PDF detail source reached renderer")

    monkeypatch.setattr(
        module, "_render_pdf_page_from_stream", unexpected_render
    )

    with pytest.raises(RuntimeError, match="PDF detail source"):
        module.rerender_pdf_page(run, "page_001")

    _assert_detail_not_consumed(run)


@pytest.mark.parametrize("invalid_sha256", [True, "A" * 64, "0" * 63])
def test_rerender_pdf_page_rejects_malformed_manifest_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_sha256: object,
) -> None:
    module = _pdf_input()
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = module.prepare_pdf_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run)
    manifest = store.read_json("job_manifest.json")
    manifest["input"]["sha256"] = invalid_sha256
    store.write_json("job_manifest.json", manifest)

    def unexpected_render(*args, **kwargs):
        raise AssertionError("invalid PDF detail source reached renderer")

    monkeypatch.setattr(
        module, "_render_pdf_page_from_stream", unexpected_render
    )

    with pytest.raises(RuntimeError, match="PDF detail source.*sha256"):
        module.rerender_pdf_page(run, "page_001")

    _assert_detail_not_consumed(run)


@pytest.mark.parametrize("symlink_kind", ["entry", "ancestor"])
def test_rerender_pdf_page_rejects_manifest_source_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_kind: str,
) -> None:
    module = _pdf_input()
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = module.prepare_pdf_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run)
    manifest = store.read_json("job_manifest.json")
    try:
        if symlink_kind == "entry":
            copied = run / "input/original.pdf"
            target = run / "input/target.pdf"
            copied.replace(target)
            copied.symlink_to(target)
        else:
            linked = run / "linked-input"
            linked.symlink_to(run / "input", target_is_directory=True)
            manifest["input"]["source"] = "linked-input/original.pdf"
            store.write_json("job_manifest.json", manifest)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    def unexpected_render(*args, **kwargs):
        raise AssertionError("symlinked PDF detail source reached renderer")

    monkeypatch.setattr(
        module, "_render_pdf_page_from_stream", unexpected_render
    )

    with pytest.raises(RuntimeError, match="PDF detail source.*symlink"):
        module.rerender_pdf_page(run, "page_001")

    _assert_detail_not_consumed(run)


def test_rerender_pdf_page_rejects_source_replaced_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pdf_input()
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = module.prepare_pdf_job(source, run_dir=tmp_path / "run")
    copied = run / "input/original.pdf"
    replacement = run / "input/replacement.pdf"
    replacement.write_bytes(copied.read_bytes())
    original_sha256_file = module.sha256_file

    def replace_after_hash(path: Path) -> str:
        digest = original_sha256_file(path)
        if Path(path) == copied:
            module.os.replace(replacement, copied)
        return digest

    def unexpected_render(*args, **kwargs):
        raise AssertionError("unstable PDF detail source reached renderer")

    monkeypatch.setattr(module, "sha256_file", replace_after_hash)
    monkeypatch.setattr(
        module, "_render_pdf_page_from_stream", unexpected_render
    )

    with pytest.raises(RuntimeError, match="changed during verification"):
        module.rerender_pdf_page(run, "page_001")

    _assert_detail_not_consumed(run)


def test_rerender_pdf_page_rolls_back_when_source_changes_during_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pdf_input()
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = module.prepare_pdf_job(source, run_dir=tmp_path / "run")
    copied = run / "input/original.pdf"
    replacement = run / "input/replacement.pdf"
    replacement.write_bytes(copied.read_bytes())
    original_render = module._render_pdf_page_from_stream

    def replace_after_render(*args, **kwargs):
        detail = original_render(*args, **kwargs)
        module.os.replace(replacement, copied)
        return detail

    monkeypatch.setattr(
        module, "_render_pdf_page_from_stream", replace_after_render
    )

    with pytest.raises((RuntimeError, PermissionError)):
        module.rerender_pdf_page(run, "page_001")

    _assert_detail_not_consumed(run)


@pytest.mark.parametrize("mutation", ["path_swap", "in_place"])
def test_rerender_pdf_page_renders_from_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = _pdf_input()
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = module.prepare_pdf_job(source, run_dir=tmp_path / "run")
    copied = run / "input/original.pdf"
    trusted_bytes = copied.read_bytes()
    malicious = run / "input/malicious.pdf"
    malicious.write_bytes(b"malicious")
    held = run / "input/held.pdf"
    history = RunStore.open(run).read_json(
        "pages/page_001/render_history.json"
    )
    standard = history["renders"][0]
    observed: dict[str, bytes] = {}

    def swap_path_while_rendering(source_value, index, output, *, profile):
        swapped = False
        original_status = copied.stat()
        if mutation == "path_swap":
            try:
                module.os.replace(copied, held)
                module.os.replace(malicious, copied)
                swapped = True
            except PermissionError:
                pass
        else:
            copied.write_bytes(b"malicious")
        try:
            source_value.seek(0)
            observed["bytes"] = source_value.read()
            source_value.seek(0)
            detail_bytes = b"detail"
            Path(output).write_bytes(detail_bytes)
            return {
                **standard,
                "profile": profile,
                "pixel_width": standard["pixel_width"] + 1,
                "pixel_height": standard["pixel_height"] + 1,
                "sha256": hashlib.sha256(detail_bytes).hexdigest(),
            }
        finally:
            if swapped:
                module.os.replace(copied, malicious)
                module.os.replace(held, copied)
            elif mutation == "in_place":
                copied.write_bytes(trusted_bytes)
                module.os.utime(
                    copied,
                    ns=(
                        original_status.st_atime_ns,
                        original_status.st_mtime_ns,
                    ),
                )

    monkeypatch.setattr(
        module,
        "_render_pdf_page_from_stream",
        swap_path_while_rendering,
        raising=False,
    )

    assert module.rerender_pdf_page(run, "page_001") == {
        "detail_used": True,
        "activated": True,
    }
    assert observed["bytes"] == trusted_bytes


def test_rerender_pdf_page_rejects_source_changed_during_snapshot_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pdf_input()
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = module.prepare_pdf_job(source, run_dir=tmp_path / "run")
    copied = run / "input/original.pdf"
    trusted_bytes = copied.read_bytes()
    original_status = copied.stat()
    original_copy = module._copy_pdf_detail_snapshot

    def corrupt_snapshot(source_file, snapshot_file):
        copied.write_bytes(b"malicious")
        try:
            original_copy(source_file, snapshot_file)
        finally:
            copied.write_bytes(trusted_bytes)
            module.os.utime(
                copied,
                ns=(
                    original_status.st_atime_ns,
                    original_status.st_mtime_ns,
                ),
            )

    def unexpected_render(*args, **kwargs):
        raise AssertionError("corrupt PDF snapshot reached renderer")

    monkeypatch.setattr(
        module,
        "_copy_pdf_detail_snapshot",
        corrupt_snapshot,
    )
    monkeypatch.setattr(
        module, "_render_pdf_page_from_stream", unexpected_render
    )

    with pytest.raises(RuntimeError, match="snapshot hash"):
        module.rerender_pdf_page(run, "page_001")

    _assert_detail_not_consumed(run)


def test_rerender_pdf_page_keeps_attempt_available_after_render_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = _pdf_input().prepare_pdf_job(source, run_dir=tmp_path / "run")

    def fail_render(source, index, output, *, profile):
        Path(output).write_bytes(b"partial")
        raise OSError("detail failed")

    monkeypatch.setattr(
        _pdf_input(), "_render_pdf_page_from_stream", fail_render
    )
    with pytest.raises(OSError, match="detail failed"):
        _pdf_input().rerender_pdf_page(run, "page_001")

    history = RunStore.open(run).read_json("pages/page_001/render_history.json")
    request = RunStore.open(run).read_json("pages/page_001/page_request.json")
    assert history["detail_used"] is False
    assert len(history["renders"]) == 1
    assert request["source"] == "pages/page_001/source.png"
    assert not (run / "pages/page_001/source_detail.png").exists()
    assert not (run / "pages/page_001/source_detail.png.tmp").exists()


def test_rerender_rolls_back_request_and_files_when_history_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = _pdf_input().prepare_pdf_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run)
    request_path = Path("pages/page_001/page_request.json")
    history_path = Path("pages/page_001/render_history.json")
    standard_request = store.read_json(request_path)
    standard_history = store.read_json(history_path)
    original_write_json = RunStore.write_json

    def fail_history(
        self: RunStore, relative: str | Path, document: dict[str, object]
    ) -> None:
        if Path(relative) == history_path and document.get("detail_used") is True:
            raise OSError("history write failed")
        original_write_json(self, relative, document)

    monkeypatch.setattr(RunStore, "write_json", fail_history)

    with pytest.raises(OSError, match="history write failed"):
        _pdf_input().rerender_pdf_page(run, "page_001")

    assert store.read_json(request_path) == standard_request
    assert store.read_json(history_path) == standard_history
    assert not (run / "pages/page_001/source_detail.png").exists()
    assert not (run / "pages/page_001/source_detail.png.tmp").exists()


def test_rerender_removes_detail_file_when_request_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = _pdf_input().prepare_pdf_job(source, run_dir=tmp_path / "run")
    request_path = Path("pages/page_001/page_request.json")
    original_write_json = RunStore.write_json

    def fail_detail_request(
        self: RunStore, relative: str | Path, document: dict[str, object]
    ) -> None:
        if (
            Path(relative) == request_path
            and document.get("source") == "pages/page_001/source_detail.png"
        ):
            raise OSError("request write failed")
        original_write_json(self, relative, document)

    monkeypatch.setattr(RunStore, "write_json", fail_detail_request)

    with pytest.raises(OSError, match="request write failed"):
        _pdf_input().rerender_pdf_page(run, "page_001")

    assert not (run / "pages/page_001/source_detail.png").exists()
    assert not (run / "pages/page_001/source_detail.png.tmp").exists()


def test_rerender_preserves_primary_error_when_request_rollback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = _pdf_input().prepare_pdf_job(source, run_dir=tmp_path / "run")
    request_path = Path("pages/page_001/page_request.json")
    history_path = Path("pages/page_001/render_history.json")
    original_write_json = RunStore.write_json
    request_writes = 0

    def fail_commit_and_rollback(
        self: RunStore, relative: str | Path, document: dict[str, object]
    ) -> None:
        nonlocal request_writes
        path = Path(relative)
        if path == request_path:
            request_writes += 1
            if request_writes == 3:
                raise PermissionError("rollback failed")
        if path == history_path and document.get("detail_used") is True:
            raise OSError("history write failed")
        original_write_json(self, relative, document)

    monkeypatch.setattr(RunStore, "write_json", fail_commit_and_rollback)

    with pytest.raises(OSError, match="history write failed") as error:
        _pdf_input().rerender_pdf_page(run, "page_001")

    assert isinstance(error.value.__cause__, PermissionError)
    assert str(error.value.__cause__) == "rollback failed"
    assert not (run / "pages/page_001/source_detail.png").exists()
    assert not (run / "pages/page_001/source_detail.png.tmp").exists()


def test_rerender_recovers_interrupted_unused_detail_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    module = _pdf_input()
    run = module.prepare_pdf_job(source, run_dir=tmp_path / "run")
    store = RunStore.open(run)
    request_path = Path("pages/page_001/page_request.json")
    standard_request = store.read_json(request_path)
    polluted_request = dict(standard_request)
    polluted_render = dict(standard_request["render"])
    polluted_render["profile"] = "detail"
    polluted_request.update(
        {
            "source": "pages/page_001/source_detail.png",
            "sha256": "interrupted",
            "render": polluted_render,
        }
    )
    store.write_json(request_path, polluted_request)
    final_detail = run / "pages/page_001/source_detail.png"
    temp_detail = run / "pages/page_001/source_detail.png.tmp"
    shutil.copy2(run / "pages/page_001/source.png", final_detail)
    temp_detail.write_bytes(b"partial")
    original_render = module._render_pdf_page_from_stream
    observed: dict[str, object] = {}

    def observe_recovered_state(*args, **kwargs):
        observed["request"] = store.read_json(request_path)
        observed["final_exists"] = final_detail.exists()
        observed["temp_exists"] = temp_detail.exists()
        return original_render(*args, **kwargs)

    monkeypatch.setattr(
        module, "_render_pdf_page_from_stream", observe_recovered_state
    )

    result = module.rerender_pdf_page(run, "page_001")

    assert result == {"detail_used": True, "activated": True}
    assert observed == {
        "request": standard_request,
        "final_exists": False,
        "temp_exists": False,
    }


def test_prepare_pdf_job_records_equal_page_ratios(tmp_path: Path) -> None:
    source = tmp_path / "same-ratio.pdf"
    _write_same_ratio_pdf(source)

    run = _pdf_input().prepare_pdf_job(source, run_dir=tmp_path / "run")

    assert RunStore.open(run).read_json("job_manifest.json")["input"][
        "page_ratios_equal"
    ] is True


def test_prepare_pdf_job_records_equal_physical_aspect_ratio(tmp_path: Path) -> None:
    source = tmp_path / "physical-ratio.pdf"
    _write_same_physical_ratio_pdf(source)

    run = _pdf_input().prepare_pdf_job(source, run_dir=tmp_path / "run")
    input_document = RunStore.open(run).read_json("job_manifest.json")["input"]

    assert input_document["page_ratios_equal"] is True
    assert input_document["page_aspect_ratio"] == 2.0


def test_prepare_pdf_job_cleans_run_directory_after_render_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = tmp_path / "run"

    def fail_render(*args, **kwargs):
        raise OSError("render failed")

    monkeypatch.setattr(_pdf_input(), "render_pdf_document", fail_render)

    with pytest.raises(OSError, match="render failed"):
        _pdf_input().prepare_pdf_job(source, run_dir=run)

    assert run.is_dir()
    assert not any(run.iterdir())


def test_prepare_pdf_job_preserves_render_error_when_rmtree_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)

    def fail_render(*args, **kwargs):
        raise OSError("render failed")

    def fail_rmtree(*args, **kwargs):
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(_pdf_input(), "render_pdf_document", fail_render)
    monkeypatch.setattr(_pdf_input().shutil, "rmtree", fail_rmtree)

    with pytest.raises(OSError, match="render failed") as error:
        _pdf_input().prepare_pdf_job(source, run_dir=tmp_path / "run")

    assert isinstance(error.value.__cause__, PermissionError)
    assert str(error.value.__cause__) == "cleanup failed"


def test_prepare_pdf_job_preserves_render_error_when_recreate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = (tmp_path / "run").resolve()
    cleanup_started = False
    original_mkdir = Path.mkdir

    def fail_render(*args, **kwargs):
        nonlocal cleanup_started
        cleanup_started = True
        raise OSError("render failed")

    def fail_recreate(self: Path, *args, **kwargs):
        if cleanup_started and self.resolve() == run:
            raise PermissionError("recreate failed")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(_pdf_input(), "render_pdf_document", fail_render)
    monkeypatch.setattr(Path, "mkdir", fail_recreate)

    with pytest.raises(OSError, match="render failed") as error:
        _pdf_input().prepare_pdf_job(source, run_dir=run)

    assert isinstance(error.value.__cause__, PermissionError)
    assert str(error.value.__cause__) == "recreate failed"


def test_prepare_pdf_job_rejects_output_inside_run_internals(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    output = tmp_path / "run" / "pages" / "output.pptx"

    with pytest.raises(ValueError, match="under final"):
        _pdf_input().prepare_pdf_job(
            source, run_dir=tmp_path / "run", output_path=output
        )


def test_rerender_pdf_page_rejects_non_pdf_run_unknown_page_and_nonprepared(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (12, 8)).save(image)
    image_run = prepare_image_job(image, run_dir=tmp_path / "image-run")

    with pytest.raises(RuntimeError, match="PDF"):
        _pdf_input().rerender_pdf_page(image_run, "page_001")

    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    run = _pdf_input().prepare_pdf_job(source, run_dir=tmp_path / "run")
    with pytest.raises(KeyError, match="Unknown page_id"):
        _pdf_input().rerender_pdf_page(run, "page_999")
    RunStore.open(run).transition_run(_pdf_input().RunStatus.RUNNING)
    with pytest.raises(RuntimeError, match="prepared"):
        _pdf_input().rerender_pdf_page(run, "page_001")


def test_standard_a4_plan_uses_standard_dpi() -> None:
    plan = _pdf_input().plan_pdf_render(595, 842, "standard")

    assert plan.target_dpi == 200.0
    assert plan.effective_dpi == 200.0
    assert (plan.pixel_width, plan.pixel_height) == (1653, 2339)
    assert plan.reasons == ()


def test_standard_small_page_raises_short_edge_to_floor() -> None:
    plan = _pdf_input().plan_pdf_render(288, 144, "standard")

    assert plan.target_dpi == 200.0
    assert plan.effective_dpi == 300.0
    assert (plan.pixel_width, plan.pixel_height) == (1200, 600)
    assert plan.reasons == ("short_edge_floor",)


def test_detail_large_page_obeys_both_hard_caps() -> None:
    module = _pdf_input()
    plan = module.plan_pdf_render(4000, 3000, "detail")

    assert plan.target_dpi == 300.0
    assert plan.pixel_width <= module.LONG_EDGE_CEILING
    assert plan.pixel_width * plan.pixel_height <= module.PIXEL_COUNT_CEILING
    assert plan.reasons == ("long_edge_ceiling", "pixel_count_ceiling")


def test_standard_plan_obeys_pixel_cap_with_pdfium_ceil_dimensions() -> None:
    module = _pdf_input()
    width_pt = 4705.759934429373
    height_pt = 4271.179396873589

    plan = module.plan_pdf_render(width_pt, height_pt, "standard")
    pixel_width = math.ceil(width_pt * plan.scale)
    pixel_height = math.ceil(height_pt * plan.scale)

    assert pixel_width * pixel_height <= module.PIXEL_COUNT_CEILING


def test_detail_plan_obeys_long_edge_cap_with_pdfium_ceil_dimensions() -> None:
    module = _pdf_input()
    width_pt = 2991.8990613508954
    height_pt = 452.76835100516956

    plan = module.plan_pdf_render(width_pt, height_pt, "detail")

    assert max(math.ceil(width_pt * plan.scale), math.ceil(height_pt * plan.scale)) <= (
        module.LONG_EDGE_CEILING
    )


def test_extreme_finite_page_keeps_positive_scale_within_hard_caps() -> None:
    module = _pdf_input()

    plan = module.plan_pdf_render(1e200, 1e200, "detail")

    pixel_width = math.ceil(1e200 * plan.scale)
    pixel_height = math.ceil(1e200 * plan.scale)
    assert plan.scale > 0
    assert plan.effective_dpi > 0
    assert max(pixel_width, pixel_height) <= module.LONG_EDGE_CEILING
    assert pixel_width * pixel_height <= module.PIXEL_COUNT_CEILING


def test_detail_a4_is_larger_than_standard() -> None:
    module = _pdf_input()

    assert module.plan_pdf_render(595, 842, "detail").pixel_height > module.plan_pdf_render(
        595, 842, "standard"
    ).pixel_height


@pytest.mark.parametrize("width,height,profile", [(0, 1, "standard"), (1, -1, "detail"), (1, 1, "draft")])
def test_plan_rejects_invalid_dimensions_and_profile(
    width: float, height: float, profile: str
) -> None:
    with pytest.raises(ValueError):
        _pdf_input().plan_pdf_render(width, height, profile)  # type: ignore[arg-type]


def test_render_document_writes_real_pages_in_order_and_records_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    outputs = [tmp_path / "nested" / "portrait.png", tmp_path / "nested" / "landscape.png"]
    _write_two_page_pdf(source)
    module = _pdf_input()
    original_document = module.pdfium.PdfDocument
    open_count = 0

    def counted_document(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return original_document(*args, **kwargs)

    monkeypatch.setattr(module.pdfium, "PdfDocument", counted_document)
    records = module.render_pdf_document(source, outputs, profile="standard")

    assert open_count == 1
    assert [record["page_index"] for record in records] == [0, 1]
    assert [record["page_number"] for record in records] == [1, 2]
    assert [record["rotation"] for record in records] == [0, 0]
    assert records[0]["width_pt"] < records[0]["height_pt"]
    assert records[1]["width_pt"] > records[1]["height_pt"]
    assert records[0]["media_box"] == [0.0, 0.0, records[0]["width_pt"], records[0]["height_pt"]]
    assert records[0]["crop_box"] is None
    assert records[0]["renderer"] == "pypdfium2"
    assert records[0]["renderer_version"]
    for output, record in zip(outputs, records):
        with Image.open(output) as image:
            assert image.size == (record["pixel_width"], record["pixel_height"])
        assert record["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_render_page_does_not_record_pdfium_defaults_for_inherited_boxes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inherited-boxes.pdf"
    output = tmp_path / "rotated.png"
    _write_inherited_boxes_pdf(source)

    record = _pdf_input().render_pdf_page(source, 0, output, profile="standard")

    assert record["media_box"] is None
    assert record["crop_box"] is None
    assert record["rotation"] == 90
    with Image.open(output) as image:
        assert image.width > image.height
        assert image.size == (record["pixel_width"], record["pixel_height"])


def test_render_page_does_not_forge_inherited_crop_from_direct_media(
    tmp_path: Path,
) -> None:
    source = tmp_path / "direct-media-inherited-crop.pdf"
    output = tmp_path / "rotated.png"
    _write_direct_media_inherited_crop_pdf(source)

    record = _pdf_input().render_pdf_page(source, 0, output, profile="standard")

    assert record["media_box"] == [0.0, 0.0, 200.0, 400.0]
    assert record["crop_box"] is None
    assert record["rotation"] == 90
    with Image.open(output) as image:
        assert image.width > image.height


def test_render_page_is_single_page_wrapper(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "page.png"
    _write_two_page_pdf(source)

    record = _pdf_input().render_pdf_page(source, 1, output, profile="detail")

    assert record["page_index"] == 1
    assert record["profile"] == "detail"
    assert output.is_file()


@pytest.mark.parametrize("alias", ["same_path", "hardlink"])
def test_render_page_rejects_output_aliasing_source(
    tmp_path: Path, alias: str
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    original = source.read_bytes()
    output = source
    if alias == "hardlink":
        output = tmp_path / "source-link.pdf"
        os.link(source, output)
    error = None

    try:
        _pdf_input().render_pdf_page(source, 0, output, profile="standard")
    except ValueError as caught:
        error = caught

    assert source.read_bytes() == original
    assert error is not None
    assert "output" in str(error)


def test_render_document_requires_one_output_per_page(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)

    with pytest.raises(ValueError, match="outputs"):
        _pdf_input().render_pdf_document(source, [tmp_path / "only-one.png"], profile="standard")


@pytest.mark.parametrize("alias", ["same_path", "hardlink"])
def test_render_document_rejects_duplicate_outputs_before_opening_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    source = tmp_path / "source.pdf"
    _write_two_page_pdf(source)
    first = tmp_path / "first.png"
    second = first
    if alias == "hardlink":
        first.write_bytes(b"keep")
        second = tmp_path / "second.png"
        os.link(first, second)
    module = _pdf_input()
    original_document = module.pdfium.PdfDocument
    open_count = 0

    def counted_document(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return original_document(*args, **kwargs)

    monkeypatch.setattr(module.pdfium, "PdfDocument", counted_document)
    error = None
    try:
        module.render_pdf_document(source, [first, second], profile="standard")
    except ValueError as caught:
        error = caught

    assert open_count == 0
    assert error is not None
    assert "outputs" in str(error)
    if alias == "same_path":
        assert not first.exists()
    else:
        assert first.read_bytes() == b"keep"
        assert second.read_bytes() == b"keep"
