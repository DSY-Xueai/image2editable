from __future__ import annotations

import importlib
import hashlib
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "image2editable" / "runtime_model_catalog.json"

SAM_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
LAMA_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
DINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


def _expected_catalog() -> dict[str, object]:
    return {
        "schema_version": 1,
        "models": {
            "sam2_large": {
                "kind": "file",
                "url": (
                    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
                    "sam2.1_hiera_large.pt"
                ),
                "size": 898083611,
                "sha256": SAM_SHA256,
                "relative_path": "sam2.1_hiera_large.pt",
            },
            "big_lama": {
                "kind": "file",
                "url": (
                    "https://github.com/enesmsahin/simple-lama-inpainting/"
                    "releases/download/v0.1.0/big-lama.pt"
                ),
                "size": 205803670,
                "sha256": LAMA_SHA256,
                "relative_path": "big-lama.pt",
            },
            "grounding_dino": {
                "kind": "huggingface_snapshot",
                "model_id": "IDEA-Research/grounding-dino-tiny",
                "revision": DINO_REVISION,
            },
        },
    }


def test_runtime_model_catalog_is_exact_and_packaged() -> None:
    assert json.loads(CATALOG_PATH.read_text(encoding="utf-8")) == _expected_catalog()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'image2editable = ["model_catalog.json", "runtime_model_catalog.json"]' in pyproject


def _runtime_models():
    return importlib.import_module("image2editable.runtime_models")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_download(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        remaining = remaining[written:]


def _small_catalog() -> tuple[dict[str, object], dict[str, bytes]]:
    payloads = {"sam2_large": b"sam", "big_lama": b"lama"}
    return (
        {
            "schema_version": 1,
            "models": {
                "sam2_large": {
                    "kind": "file",
                    "url": "https://example.test/sam.pt",
                    "size": len(payloads["sam2_large"]),
                    "sha256": _sha256(payloads["sam2_large"]),
                    "relative_path": "sam.pt",
                },
                "big_lama": {
                    "kind": "file",
                    "url": "https://example.test/lama.pt",
                    "size": len(payloads["big_lama"]),
                    "sha256": _sha256(payloads["big_lama"]),
                    "relative_path": "lama.pt",
                },
                "grounding_dino": {
                    "kind": "huggingface_snapshot",
                    "model_id": "example/dino",
                    "revision": "a" * 40,
                },
            },
        },
        payloads,
    )


def _catalog_sha256(catalog: dict[str, object]) -> str:
    payload = json.dumps(
        catalog,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(payload)


def _install_small_runtime(
    runtime_models,
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    catalog, payloads = _small_catalog()

    def fake_file_download(url: str, descriptor: int) -> None:
        name = "sam2_large" if "sam" in url else "big_lama"
        _write_download(descriptor, payloads[name])

    def fake_snapshot_download(**kwargs: object) -> str:
        snapshot = Path(kwargs["local_dir"])
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_bytes(b"config")
        (snapshot / "weights.bin").write_bytes(b"weights")
        return str(snapshot)

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(runtime_models, "snapshot_download", fake_snapshot_download)
    receipt = runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)
    return catalog, receipt


def test_load_runtime_catalog_validates_the_exact_schema(tmp_path: Path) -> None:
    runtime_models = _runtime_models()
    assert runtime_models.load_runtime_catalog() == _expected_catalog()

    empty_path = json.loads(json.dumps(_expected_catalog()))
    empty_path["models"]["sam2_large"]["relative_path"] = ""
    boolean_size = json.loads(json.dumps(_expected_catalog()))
    boolean_size["models"]["big_lama"]["size"] = True
    snapshot_extra = json.loads(json.dumps(_expected_catalog()))
    snapshot_extra["models"]["grounding_dino"]["unexpected"] = True
    invalid_catalogs = [
        {**_expected_catalog(), "unexpected": True},
        {**_expected_catalog(), "schema_version": 2},
        {
            **_expected_catalog(),
            "models": {
                **_expected_catalog()["models"],
                "unexpected": {},
            },
        },
        empty_path,
        boolean_size,
        snapshot_extra,
    ]
    for index, catalog in enumerate(invalid_catalogs):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        with pytest.raises(ValueError, match="runtime model catalog"):
            runtime_models.load_runtime_catalog(path)


def test_default_runtime_cache_honors_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    configured = tmp_path / "configured"
    monkeypatch.setenv("IMAGE2EDITABLE_MODEL_CACHE", str(configured))
    assert runtime_models.default_runtime_cache() == configured.resolve()
    monkeypatch.delenv("IMAGE2EDITABLE_MODEL_CACHE")
    monkeypatch.setattr(runtime_models.Path, "home", lambda: tmp_path)
    assert runtime_models.default_runtime_cache() == (
        tmp_path / ".cache" / "image2editable" / "models" / "runtime"
    ).resolve()


def test_install_requires_confirmation_without_network_or_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    cache = tmp_path / "runtime-cache"

    def unexpected_call(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"unconfirmed install attempted I/O: {args} {kwargs}")

    monkeypatch.setattr(runtime_models, "download_file", unexpected_call)
    monkeypatch.setattr(runtime_models, "snapshot_download", unexpected_call)

    with pytest.raises(PermissionError, match="explicit confirmation"):
        runtime_models.install_runtime_models(cache_dir=cache)

    assert not cache.exists()


def test_install_downloads_exact_assets_and_writes_bound_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    catalog, payloads = _small_catalog()
    cache = tmp_path / "runtime-cache"
    file_calls: list[tuple[str, int]] = []
    snapshot_calls: list[dict[str, object]] = []

    def fake_file_download(url: str, descriptor: int) -> None:
        file_calls.append((url, descriptor))
        temporary = next(cache.glob(".*.tmp"))
        assert temporary.name.startswith(".") and temporary.name.endswith(".tmp")
        assert not (cache / ("sam.pt" if "sam" in url else "lama.pt")).exists()
        _write_download(
            descriptor,
            payloads["sam2_large" if "sam" in url else "big_lama"],
        )

    def fake_snapshot_download(**kwargs: object) -> str:
        snapshot_calls.append(kwargs)
        snapshot = Path(kwargs["local_dir"])
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_bytes(b"config")
        (snapshot / "weights.bin").write_bytes(b"weights")
        return str(snapshot)

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(runtime_models, "snapshot_download", fake_snapshot_download)

    result = runtime_models.install_runtime_models(
        cache_dir=cache,
        confirmed=True,
    )

    assert [url for url, _ in file_calls] == [
        "https://example.test/sam.pt",
        "https://example.test/lama.pt",
    ]
    snapshot = cache / "grounding_dino" / ("a" * 40)
    assert len(snapshot_calls) == 1
    snapshot_call = snapshot_calls[0]
    staging = Path(snapshot_call["local_dir"])
    assert snapshot_call == {
        "repo_id": "example/dino",
        "revision": "a" * 40,
        "local_dir": str(staging),
        "cache_dir": str(cache / ".huggingface"),
    }
    assert staging != snapshot
    if os.name == "nt":
        assert staging.name == f".grounding-dino-{'a' * 40}.installing"
        assert staging.parent == snapshot.parent
    else:
        assert staging.parent in {Path("/proc/self/fd"), Path("/dev/fd")}
    receipt = {
        "schema_version": 1,
        "catalog_sha256": _catalog_sha256(catalog),
        "models": {
            "sam2_large": {
                "kind": "file",
                "relative_path": "sam.pt",
                "size": 3,
                "sha256": _sha256(b"sam"),
            },
            "big_lama": {
                "kind": "file",
                "relative_path": "lama.pt",
                "size": 4,
                "sha256": _sha256(b"lama"),
            },
            "grounding_dino": {
                "kind": "huggingface_snapshot",
                "model_id": "example/dino",
                "requested_revision": "a" * 40,
                "resolved_revision": "a" * 40,
                "relative_path": f"grounding_dino/{'a' * 40}",
                "files": [
                    {
                        "path": "config.json",
                        "size": 6,
                        "sha256": _sha256(b"config"),
                    },
                    {
                        "path": "weights.bin",
                        "size": 7,
                        "sha256": _sha256(b"weights"),
                    },
                ],
            },
        },
    }
    assert result == receipt
    assert json.loads((cache / "runtime-receipt.json").read_text(encoding="utf-8")) == receipt
    assert list(cache.glob(".*.tmp")) == []


def test_runtime_status_is_read_only_when_receipt_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    cache = tmp_path / "missing"

    def unexpected_call(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"status attempted network I/O: {args} {kwargs}")

    monkeypatch.setattr(runtime_models, "download_file", unexpected_call)
    monkeypatch.setattr(runtime_models, "snapshot_download", unexpected_call)

    assert runtime_models.runtime_model_status(cache_dir=cache) == {
        "installed": False,
        "valid": False,
        "install_command": "image2editable models install runtime",
    }
    assert not cache.exists()
    with pytest.raises(
        runtime_models.RuntimeModelError,
        match=r"image2editable models install runtime",
    ):
        runtime_models.runtime_model_path("grounding_dino", cache_dir=cache)


def test_runtime_status_and_resolver_validate_installed_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    cache = tmp_path / "runtime-cache"
    _, receipt = _install_small_runtime(runtime_models, monkeypatch, cache)

    assert runtime_models.runtime_model_status(cache_dir=cache) == {
        "installed": True,
        "valid": True,
        "install_command": "image2editable models install runtime",
        "receipt": receipt,
    }
    assert runtime_models.runtime_model_path(
        "sam2_large", cache_dir=cache
    ) == (cache / "sam.pt").resolve()
    assert runtime_models.runtime_model_path(
        "big_lama", cache_dir=cache
    ) == (cache / "lama.pt").resolve()
    assert runtime_models.runtime_model_path(
        "grounding_dino", cache_dir=cache
    ) == (cache / "grounding_dino" / ("a" * 40)).resolve()

    with pytest.raises(runtime_models.RuntimeModelError, match="unknown runtime model"):
        runtime_models.runtime_model_path("unknown", cache_dir=cache)


@pytest.mark.parametrize(
    "tamper",
    [
        "file_changed",
        "snapshot_extra",
        "snapshot_missing",
        "revision_changed",
        "absolute_path",
        "duplicate_file",
        "files_reordered",
        "receipt_extra_field",
        "catalog_changed",
    ],
)
def test_runtime_status_rejects_tampering_without_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
) -> None:
    runtime_models = _runtime_models()
    cache = tmp_path / tamper
    catalog, receipt = _install_small_runtime(runtime_models, monkeypatch, cache)
    receipt_path = cache / "runtime-receipt.json"
    snapshot = cache / "grounding_dino" / ("a" * 40)

    if tamper == "file_changed":
        (cache / "sam.pt").write_bytes(b"bad")
    elif tamper == "snapshot_extra":
        (snapshot / "extra.bin").write_bytes(b"extra")
    elif tamper == "snapshot_missing":
        (snapshot / "config.json").unlink()
    elif tamper == "revision_changed":
        receipt["models"]["grounding_dino"]["resolved_revision"] = "b" * 40
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif tamper == "absolute_path":
        receipt["models"]["grounding_dino"]["relative_path"] = str(snapshot.resolve())
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif tamper == "duplicate_file":
        receipt["models"]["grounding_dino"]["files"].append(
            receipt["models"]["grounding_dino"]["files"][0]
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif tamper == "files_reordered":
        receipt["models"]["grounding_dino"]["files"].reverse()
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif tamper == "receipt_extra_field":
        receipt["unexpected"] = True
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    else:
        changed_catalog = json.loads(json.dumps(catalog))
        changed_catalog["models"]["big_lama"]["url"] += "?changed=1"
        monkeypatch.setattr(
            runtime_models,
            "load_runtime_catalog",
            lambda: changed_catalog,
        )

    status = runtime_models.runtime_model_status(cache_dir=cache)
    assert status["installed"] is True
    assert status["valid"] is False
    assert "reason" in status
    with pytest.raises(runtime_models.RuntimeModelError):
        runtime_models.runtime_model_path("grounding_dino", cache_dir=cache)


def test_install_reuses_valid_assets_and_receipt_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    cache = tmp_path / "runtime-cache"
    _, receipt = _install_small_runtime(runtime_models, monkeypatch, cache)

    def unexpected_call(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"idempotent install attempted network: {args} {kwargs}")

    monkeypatch.setattr(runtime_models, "download_file", unexpected_call)
    monkeypatch.setattr(runtime_models, "snapshot_download", unexpected_call)

    assert runtime_models.install_runtime_models(
        cache_dir=cache,
        confirmed=True,
    ) == receipt


@pytest.mark.parametrize("failure", ["file", "snapshot"])
def test_failed_install_never_publishes_a_valid_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    runtime_models = _runtime_models()
    catalog, payloads = _small_catalog()
    cache = tmp_path / failure

    def fake_file_download(url: str, descriptor: int) -> None:
        if failure == "file":
            _write_download(descriptor, b"invalid")
            return
        name = "sam2_large" if "sam" in url else "big_lama"
        _write_download(descriptor, payloads[name])

    def fake_snapshot_download(**kwargs: object) -> str:
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(runtime_models, "snapshot_download", fake_snapshot_download)

    with pytest.raises((OSError, RuntimeError)):
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)

    assert not os.path.lexists(cache / "runtime-receipt.json")
    assert list(cache.glob(".*.tmp")) == []
    assert list((cache / "grounding_dino").glob("*.installing")) == []


def test_failed_snapshot_keeps_one_bounded_staging_when_cleanup_is_unsafe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    catalog, payloads = _small_catalog()
    cache = tmp_path / "runtime-cache"
    snapshot_calls = 0

    def fake_file_download(url: str, descriptor: int) -> None:
        name = "sam2_large" if "sam" in url else "big_lama"
        _write_download(descriptor, payloads[name])

    def failing_snapshot_download(**kwargs: object) -> str:
        nonlocal snapshot_calls
        snapshot_calls += 1
        staging = Path(kwargs["local_dir"])
        (staging / "partial.bin").write_bytes(b"partial")
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(runtime_models, "snapshot_download", failing_snapshot_download)
    monkeypatch.setattr(
        runtime_models.shutil,
        "rmtree",
        lambda path: (_ for _ in ()).throw(OSError("cleanup unavailable")),
    )

    with pytest.raises(RuntimeError, match=r"\.grounding-dino-.*\.installing"):
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)
    with pytest.raises(RuntimeError, match=r"\.grounding-dino-.*\.installing"):
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)

    staging = list((cache / "grounding_dino").glob("*.installing"))
    assert len(staging) == 1
    assert (staging[0] / "partial.bin").read_bytes() == b"partial"
    assert snapshot_calls == 1
    assert not (cache / "runtime-receipt.json").exists()


def test_install_rejects_an_unreceipted_partial_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    catalog, payloads = _small_catalog()
    cache = tmp_path / "runtime-cache"
    partial = cache / "grounding_dino" / ("a" * 40)
    partial.mkdir(parents=True)
    (partial / "config.json").write_bytes(b"partial")

    def fake_file_download(url: str, descriptor: int) -> None:
        name = "sam2_large" if "sam" in url else "big_lama"
        _write_download(descriptor, payloads[name])

    def unexpected_snapshot_download(**kwargs: object) -> str:
        raise AssertionError(f"partial snapshot was overwritten: {kwargs}")

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(
        runtime_models,
        "snapshot_download",
        unexpected_snapshot_download,
    )

    with pytest.raises(RuntimeError, match="snapshot.*already exists"):
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)
    assert not (cache / "runtime-receipt.json").exists()
    assert (partial / "config.json").read_bytes() == b"partial"


def test_snapshot_publish_does_not_replace_a_racing_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    catalog, payloads = _small_catalog()
    cache = tmp_path / "runtime-cache"
    target = cache / "grounding_dino" / ("a" * 40)

    def fake_file_download(url: str, descriptor: int) -> None:
        name = "sam2_large" if "sam" in url else "big_lama"
        _write_download(descriptor, payloads[name])

    def racing_snapshot_download(**kwargs: object) -> str:
        staging = Path(kwargs["local_dir"])
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "model.bin").write_bytes(b"model")
        if staging != target:
            target.mkdir(parents=True)
            (target / "attacker.bin").write_bytes(b"attacker")
        return str(staging)

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(
        runtime_models,
        "snapshot_download",
        racing_snapshot_download,
    )

    with pytest.raises((FileExistsError, RuntimeError), match="already exists"):
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)
    assert (target / "attacker.bin").read_bytes() == b"attacker"
    assert not (cache / "runtime-receipt.json").exists()


def test_snapshot_publish_rejects_parent_symlink_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("Windows replacement blocking is covered by native handle test")
    runtime_models = _runtime_models()
    catalog, payloads = _small_catalog()
    cache = tmp_path / "runtime-cache"

    def fake_file_download(url: str, descriptor: int) -> None:
        name = "sam2_large" if "sam" in url else "big_lama"
        _write_download(descriptor, payloads[name])

    def replacing_snapshot_download(**kwargs: object) -> str:
        staging = Path(kwargs["local_dir"])
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "model.bin").write_bytes(b"model")
        parent = cache / "grounding_dino"
        relocated = cache / "relocated-grounding-dino"
        parent.rename(relocated)
        try:
            parent.symlink_to(relocated, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"directory symlinks are unavailable: {error}")
        return str(staging)

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(
        runtime_models,
        "snapshot_download",
        replacing_snapshot_download,
    )

    with pytest.raises(RuntimeError, match="parent identity changed"):
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)
    assert not (cache / "runtime-receipt.json").exists()


def test_windows_directory_handles_block_parent_and_staging_replacement(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows directory handle contract")
    runtime_models = _runtime_models()
    parent = tmp_path / "grounding_dino"
    parent.mkdir()
    parent_status = parent.lstat()
    parent_binding = runtime_models._open_directory(parent)
    staging_binding = None
    try:
        staging, _, staging_binding = runtime_models._private_directory(
            parent,
            "a" * 40,
            parent_binding,
        )
        with pytest.raises(OSError):
            parent.rename(tmp_path / "moved-parent")
        with pytest.raises(OSError):
            staging.rename(parent / "moved-staging")
        runtime_models._validate_parent(parent, parent_binding, parent_status)
    finally:
        if staging_binding is not None:
            runtime_models._close_directory(staging_binding)
        runtime_models._close_directory(parent_binding)


def test_snapshot_install_rejects_parent_binding_change_before_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    catalog, payloads = _small_catalog()
    cache = tmp_path / "runtime-cache"
    staging_or_download_calls: list[str] = []
    closed_bindings: list[tuple[int, tuple[int, int] | None]] = []
    real_close = runtime_models._close_directory

    def fake_file_download(url: str, descriptor: int) -> None:
        name = "sam2_large" if "sam" in url else "big_lama"
        _write_download(descriptor, payloads[name])

    def changed_parent(*args: object) -> None:
        raise RuntimeError("runtime snapshot parent identity changed")

    def unexpected_private(*args: object) -> object:
        staging_or_download_calls.append("staging")
        raise AssertionError("staging was allocated after parent changed")

    def record_close(binding: tuple[int, tuple[int, int] | None]) -> None:
        closed_bindings.append(binding)
        real_close(binding)

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(runtime_models, "_validate_parent", changed_parent)
    monkeypatch.setattr(runtime_models, "_private_directory", unexpected_private)
    monkeypatch.setattr(runtime_models, "_close_directory", record_close)

    with pytest.raises(RuntimeError, match="parent identity changed"):
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)
    assert staging_or_download_calls == []
    assert len(closed_bindings) == 1
    parent = cache / "grounding_dino"
    parent.rename(cache / "renamed-grounding-dino")


def test_windows_private_staging_is_validated_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows directory handle contract")
    runtime_models = _runtime_models()
    parent = tmp_path / "grounding_dino"
    parent.mkdir()
    parent_binding = runtime_models._open_directory(parent)
    validations: list[Path] = []
    real_validate = runtime_models._validate_directory_binding

    def validate(*args: object) -> None:
        validations.append(Path(args[0]))
        real_validate(*args)

    monkeypatch.setattr(runtime_models, "_validate_directory_binding", validate)
    staging_binding = None
    try:
        staging, _, staging_binding = runtime_models._private_directory(
            parent,
            "a" * 40,
            parent_binding,
        )
        assert validations == [staging]
    finally:
        if staging_binding is not None:
            runtime_models._close_directory(staging_binding)
        runtime_models._close_directory(parent_binding)


@pytest.mark.parametrize(
    ("platform", "symbol", "flag"),
    [("linux", "renameat2", 1), ("darwin", "renameatx_np", 4)],
)
def test_snapshot_publish_uses_platform_no_replace_rename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    symbol: str,
    flag: int,
) -> None:
    runtime_models = _runtime_models()
    staging = tmp_path / ".snapshot.tmp"
    staging.mkdir()
    target = tmp_path / "snapshot"
    parent_status = tmp_path.lstat()
    staging_status = staging.lstat()
    staging_identity = (staging.stat().st_dev, staging.stat().st_ino)
    real_rename = os.rename
    calls: list[tuple[object, ...]] = []

    class FakeRename:
        argtypes: object = None
        restype: object = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            real_rename(staging, target)
            return 0

    class FakeLibrary:
        pass

    library = FakeLibrary()
    setattr(library, symbol, FakeRename())
    monkeypatch.setattr(runtime_models.os, "name", "posix")
    monkeypatch.setattr(runtime_models.sys, "platform", platform)
    monkeypatch.setattr(runtime_models.ctypes, "CDLL", lambda *args, **kwargs: library)
    monkeypatch.setattr(
        runtime_models.os,
        "fstat",
        lambda descriptor: parent_status if descriptor == 37 else staging_status,
    )

    runtime_models._publish_directory(
        staging,
        target,
        (37, None),
        parent_status,
        (38, None),
        staging_identity,
    )

    assert calls == [(37, b".snapshot.tmp", 37, b"snapshot", flag)]
    assert target.is_dir()


def test_snapshot_publish_fails_closed_on_an_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    staging = tmp_path / ".snapshot.tmp"
    staging.mkdir()
    parent_status = tmp_path.lstat()
    staging_status = staging.lstat()
    staging_identity = (staging.stat().st_dev, staging.stat().st_ino)
    monkeypatch.setattr(runtime_models.os, "name", "posix")
    monkeypatch.setattr(runtime_models.sys, "platform", "freebsd")
    monkeypatch.setattr(
        runtime_models.os,
        "fstat",
        lambda descriptor: parent_status if descriptor == 37 else staging_status,
    )

    with pytest.raises(RuntimeError, match="no-replace publication is unavailable"):
        runtime_models._publish_directory(
            staging,
            tmp_path / "snapshot",
            (37, None),
            parent_status,
            (38, None),
            staging_identity,
        )
    assert staging.is_dir()
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize(
    ("platform", "root"),
    [("linux", Path("/proc/self/fd")), ("darwin", Path("/dev/fd"))],
)
def test_snapshot_download_path_uses_a_verified_descriptor_anchor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    root: Path,
) -> None:
    runtime_models = _runtime_models()
    staging = tmp_path / ".grounding-dino-test.installing"
    staging.mkdir()
    identity = (staging.stat().st_dev, staging.stat().st_ino)
    descriptor = 38
    anchor = root / str(descriptor)
    real_resolve = runtime_models.Path.resolve

    def resolve(path: Path, strict: bool = False) -> Path:
        if path == anchor:
            return real_resolve(staging, strict=True)
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(runtime_models.sys, "platform", platform)
    monkeypatch.setattr(runtime_models.Path, "resolve", resolve)
    def fstat(descriptor: int):
        assert descriptor == 38
        return staging.stat()

    monkeypatch.setattr(runtime_models.os, "fstat", fstat)

    assert runtime_models._snapshot_download_path(
        staging,
        identity,
        (descriptor, None),
    ) == anchor


def test_directory_close_error_does_not_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_models = _runtime_models()
    monkeypatch.setattr(
        runtime_models,
        "_close_windows_handle",
        lambda handle: (_ for _ in ()).throw(OSError("close failed")),
    )

    runtime_models._close_directory((37, (1, 2)))


def test_file_downloader_receives_only_the_private_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    catalog, payloads = _small_catalog()
    cache = tmp_path / "runtime-cache"
    destinations: list[object] = []

    def fake_file_download(url: str, destination: object) -> None:
        destinations.append(destination)
        name = "sam2_large" if "sam" in url else "big_lama"
        if isinstance(destination, int):
            _write_download(destination, payloads[name])
        else:
            Path(destination).write_bytes(payloads[name])

    def fake_snapshot_download(**kwargs: object) -> str:
        snapshot = Path(kwargs["local_dir"])
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model.bin").write_bytes(b"model")
        return str(snapshot)

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(runtime_models, "snapshot_download", fake_snapshot_download)

    runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)
    assert destinations and all(isinstance(item, int) for item in destinations)


def test_install_does_not_overwrite_an_invalid_existing_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    catalog, _ = _small_catalog()
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    target = cache / "sam.pt"
    target.write_bytes(b"attacker")

    def unexpected_download(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"invalid target was overwritten: {args} {kwargs}")

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", unexpected_download)

    with pytest.raises(RuntimeError, match="integrity"):
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)

    assert target.read_bytes() == b"attacker"
    assert not (cache / "runtime-receipt.json").exists()


@pytest.mark.parametrize("racing_asset_is_valid", [True, False])
def test_file_publish_is_no_replace_under_a_target_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    racing_asset_is_valid: bool,
) -> None:
    runtime_models = _runtime_models()
    catalog, payloads = _small_catalog()
    cache = tmp_path / "runtime-cache"
    raced = False

    def fake_file_download(url: str, descriptor: int) -> None:
        nonlocal raced
        name = "sam2_large" if "sam" in url else "big_lama"
        _write_download(descriptor, payloads[name])
        if name == "sam2_large" and not raced:
            raced = True
            target = cache / "sam.pt"
            target.write_bytes(b"sam" if racing_asset_is_valid else b"attacker")

    def fake_snapshot_download(**kwargs: object) -> str:
        snapshot = Path(kwargs["local_dir"])
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model.bin").write_bytes(b"model")
        return str(snapshot)

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(runtime_models, "snapshot_download", fake_snapshot_download)

    if racing_asset_is_valid:
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)
        assert (cache / "sam.pt").read_bytes() == b"sam"
        assert (cache / "runtime-receipt.json").is_file()
    else:
        with pytest.raises(RuntimeError, match="integrity"):
            runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)
        assert (cache / "sam.pt").read_bytes() == b"attacker"
        assert not (cache / "runtime-receipt.json").exists()


def test_install_rejects_existing_hard_linked_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    catalog, _ = _small_catalog()
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"sam")
    try:
        os.link(outside, cache / "sam.pt")
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")
    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)

    with pytest.raises(RuntimeError, match="private regular"):
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)

    assert outside.read_bytes() == b"sam"
    assert not (cache / "runtime-receipt.json").exists()


def test_private_cleanup_does_not_delete_a_replacement(tmp_path: Path) -> None:
    runtime_models = _runtime_models()
    path, descriptor, identity = runtime_models._private_file(tmp_path, "model")
    os.close(descriptor)
    path.unlink()
    path.write_bytes(b"replacement")

    runtime_models._cleanup_private(path, identity)

    assert path.read_bytes() == b"replacement"


def test_receipt_publication_handles_short_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    cache = tmp_path / "runtime-cache"
    real_write = runtime_models.os.write
    short_writes = 0

    def short_write(descriptor: int, payload: bytes) -> int:
        nonlocal short_writes
        short_writes += 1
        return real_write(descriptor, payload[: max(1, len(payload) // 2)])

    monkeypatch.setattr(runtime_models.os, "write", short_write)

    _install_small_runtime(runtime_models, monkeypatch, cache)

    assert short_writes > 1
    assert runtime_models.runtime_model_status(cache_dir=cache)["valid"] is True


def test_runtime_status_rejects_a_linked_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    cache = tmp_path / "runtime-cache"
    cache.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    receipt = cache / "runtime-receipt.json"
    try:
        receipt.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    status = runtime_models.runtime_model_status(cache_dir=cache)

    assert status["installed"] is True
    assert status["valid"] is False
    assert outside.read_text(encoding="utf-8") == "{}"


def test_runtime_status_rejects_receipt_replaced_after_initial_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_models = _runtime_models()
    cache = tmp_path / "runtime-cache"
    _install_small_runtime(runtime_models, monkeypatch, cache)
    receipt = cache / "runtime-receipt.json"
    outside = tmp_path / "outside-receipt.json"
    outside.write_bytes(receipt.read_bytes())
    real_record = runtime_models.strict_file_record
    replaced = False

    def replace_after_validation(path: Path, boundary: Path) -> dict[str, object]:
        nonlocal replaced
        record = real_record(path, boundary)
        if path == receipt and not replaced:
            replaced = True
            path.unlink()
            os.link(outside, path)
        return record

    monkeypatch.setattr(runtime_models, "strict_file_record", replace_after_validation)

    status = runtime_models.runtime_model_status(cache_dir=cache)

    assert status["installed"] is True
    assert status["valid"] is False


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_snapshot_manifest_rejects_linked_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    link_kind: str,
) -> None:
    runtime_models = _runtime_models()
    catalog, payloads = _small_catalog()
    cache = tmp_path / link_kind

    def fake_file_download(url: str, descriptor: int) -> None:
        name = "sam2_large" if "sam" in url else "big_lama"
        _write_download(descriptor, payloads[name])

    def fake_snapshot_download(**kwargs: object) -> str:
        snapshot = Path(kwargs["local_dir"])
        snapshot.mkdir(parents=True, exist_ok=True)
        outside = cache / "outside.bin"
        outside.write_bytes(b"model")
        linked = snapshot / "model.bin"
        try:
            if link_kind == "symlink":
                linked.symlink_to(outside)
            else:
                os.link(outside, linked)
        except OSError as error:
            pytest.skip(f"{link_kind} is unavailable: {error}")
        return str(snapshot)

    monkeypatch.setattr(runtime_models, "load_runtime_catalog", lambda: catalog)
    monkeypatch.setattr(runtime_models, "download_file", fake_file_download)
    monkeypatch.setattr(runtime_models, "snapshot_download", fake_snapshot_download)

    with pytest.raises(RuntimeError, match="link|unsafe|private regular"):
        runtime_models.install_runtime_models(cache_dir=cache, confirmed=True)

    assert not (cache / "runtime-receipt.json").exists()
