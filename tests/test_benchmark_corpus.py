import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import PIL
import pytest
from PIL import Image, ImageOps, ImageStat, features

from scripts.generate_benchmark_corpus import (
    LANDSCAPE,
    PORTRAIT,
    SCHEMA_VERSION,
    generate_corpus,
)


EXPECTED_IMAGE_CASES = [
    ("01_zh_courseware", "image", "01-zh-courseware.png", 1),
    ("02_typography", "image", "02-typography.png", 1),
    ("03_flowchart", "image", "03-flowchart.png", 1),
    ("04_table_chart", "image", "04-table-chart.png", 1),
    ("05_photo_overlay", "image", "05-photo-overlay.png", 1),
    ("06_transparency_shadow", "image", "06-transparency-shadow.png", 1),
    ("07_compressed", "image", "07-compressed.jpg", 1),
    ("08_portrait", "image", "08-portrait.png", 1),
]

GOLDEN_SHA256 = {
    "01-zh-courseware.png": "b8a4df5baf6978e0e353494a9d3ff3c991d1105daaf084b52dff36fabbe2c2f8",
    "02-typography.png": "e649cadfd47ee8dc270eb35aa36d6af018e710461e5458686decaaf83006aaca",
    "03-flowchart.png": "5d6c1ff4700a0fb80c566dc7b790f8517a5bd71d5637f4e1ab23e1ea544eb998",
    "04-table-chart.png": "5a5ed76087edbfe029f0f68c94a5080880e0d33b6bac3b6d0edb503109d409ec",
    "05-photo-overlay.png": "d9c1c57ff0e200139cd575f860ed9a5c893c2ae59c57b11d642f6d0bc067c181",
    "06-transparency-shadow.png": "590a5cf4796d575b6e1b71543e82d5624c64be12eac88b1e3c304e587671c1ba",
    "07-compressed.jpg": "9f055966531e0fa4bbb403e368224db1e5b73d685fc48bc6d30faede65fabad1",
    "08-portrait.png": "4891659393a0c3bfdf8cfa5be58b46c9de7cc12830f7f0fec0e7dd4dc9d4a41d",
    "manifest.json": "67325338976a2a2c3f8236ec987e10053e72e421fea48d6cde0711f0dd2ae6d4",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generate_corpus_rejects_unapproved_encoder_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    message = (
        "deterministic benchmark encoder unavailable: requires Pillow 10.4.0, "
        "libjpeg_turbo 3.0.3, zlib 1.3.1, freetype2 2.13.2"
    )

    wrong_freetype = tmp_path / "wrong-freetype"
    with monkeypatch.context() as patch:
        patch.setattr(
            features,
            "version_module",
            lambda name: "2.13.3" if name == "freetype2" else None,
        )
        with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
            generate_corpus(wrong_freetype)
    assert not wrong_freetype.exists()

    wrong_pillow = tmp_path / "wrong-pillow"
    with monkeypatch.context() as patch:
        patch.setattr(PIL, "__version__", "11.1.0")
        with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
            generate_corpus(wrong_pillow)
    assert not wrong_pillow.exists()

    wrong_jpeg = tmp_path / "wrong-jpeg"
    wrong_jpeg.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(
            features,
            "version_feature",
            lambda name: "3.0.2" if name == "libjpeg_turbo" else None,
        )
        with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
            generate_corpus(wrong_jpeg)
    assert not any(wrong_jpeg.iterdir())

    wrong_zlib = tmp_path / "wrong-zlib"
    with monkeypatch.context() as patch:
        patch.setattr(
            features,
            "version_codec",
            lambda name: "1.3.0" if name == "zlib" else None,
        )
        with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
            generate_corpus(wrong_zlib)
    assert not wrong_zlib.exists()


def test_generate_corpus_writes_fixed_manifest_and_nonblank_images(
    tmp_path: Path,
) -> None:
    output = tmp_path / "corpus"

    generate_corpus(output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert SCHEMA_VERSION == 1
    assert LANDSCAPE == (1600, 900)
    assert PORTRAIT == (900, 1600)
    assert set(manifest) == {"schema_version", "cases"}
    assert manifest["schema_version"] == 1
    assert [
        (case["id"], case["kind"], case["path"], case["pages"])
        for case in manifest["cases"]
    ] == EXPECTED_IMAGE_CASES

    for index, case in enumerate(manifest["cases"], start=1):
        assert set(case) == {"id", "kind", "path", "pages", "bytes", "sha256"}
        path = output / case["path"]
        assert case["bytes"] == path.stat().st_size
        assert case["sha256"] == _sha256(path)
        with Image.open(path) as image:
            assert image.format == ("JPEG" if index == 7 else "PNG")
            assert image.size == (PORTRAIT if index == 8 else LANDSCAPE)
            extrema = ImageStat.Stat(image.convert("RGB")).extrema
            assert any(low != high for low, high in extrema)
            if index == 6:
                assert image.getpixel((0, 899)) != image.getpixel((1599, 899))

    source_path = Path(__file__).resolve().parents[1] / "docs/images/demo-source-1.png"
    with (
        Image.open(source_path) as source,
        Image.open(output / "01-zh-courseware.png") as generated,
    ):
        expected = ImageOps.fit(
            source.convert("RGB"), LANDSCAPE, Image.Resampling.LANCZOS
        )
        assert generated.convert("RGB").tobytes() == expected.tobytes()


def test_generate_corpus_refuses_nonempty_output_before_writing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "corpus"
    output.mkdir()
    conflict = output / "01-zh-courseware.png"
    conflict.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        generate_corpus(output)

    assert {path.name for path in output.iterdir()} == {conflict.name}
    assert conflict.read_bytes() == b"keep me"


def test_generate_corpus_is_byte_reproducible_in_two_empty_directories(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    generate_corpus(first)
    generate_corpus(second)

    expected_names = {case[2] for case in EXPECTED_IMAGE_CASES} | {"manifest.json"}
    assert {path.name for path in first.iterdir()} == expected_names
    assert {path.name for path in second.iterdir()} == expected_names
    assert {
        name: (first / name).read_bytes() for name in expected_names
    } == {name: (second / name).read_bytes() for name in expected_names}
    assert {name: _sha256(first / name) for name in expected_names} == GOLDEN_SHA256


def test_cli_generates_manifest_from_checkout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "cli-corpus"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_benchmark_corpus.py",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert len(manifest["cases"]) == 8
