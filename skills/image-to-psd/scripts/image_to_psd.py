#!/usr/bin/env python3
"""Standalone image-to-PSD entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence


skill_root = Path(__file__).resolve().parents[1]
if str(skill_root) not in sys.path:
    sys.path.insert(0, str(skill_root))

from scripts.image_to_ppt import (  # noqa: E402
    _prepare_multiple_images,
    _prepare_single_image,
    _resolve_inputs,
)
from scripts.psd_assemble import (  # noqa: E402
    assemble_psd,
    preflight_psd_runtime,
)


_STANDALONE_MODEL_PATHS = {
    "SAM2_MODEL": "file",
    "LAMA_MODEL": "file",
    "GROUNDING_DINO_MODEL": "directory",
}


def convert(
    image_path: str | Path,
    output_path: str | Path | None = None,
    *,
    lang: str = "ch",
    _work_root: str | Path | None = None,
    _resource_isolation: bool = False,
) -> str:
    """Convert one image to a layered PSD without the product runtime."""
    source = Path(image_path).resolve()
    target = _single_output_path(source, output_path)
    _require_available_outputs([target])
    _preflight_standalone_runtime()

    prepare_kwargs = {"_work_root": _work_root}
    if _resource_isolation:
        prepare_kwargs["_resource_isolation"] = True
    slide, _work_dir = _prepare_single_image(source, lang, **prepare_kwargs)
    _assemble_and_publish([slide], [target])
    return str(target)


def convert_batch(
    image_paths: list[str | Path],
    output_path: str | Path | None = None,
    *,
    lang: str = "ch",
    _work_root: str | Path | None = None,
    _resource_isolation: bool = False,
) -> list[str]:
    """Convert multiple images to one layered PSD per image."""
    sources = [Path(path).resolve() for path in image_paths]
    if not sources:
        raise ValueError("No valid images provided")
    targets = _batch_output_paths(sources, output_path)
    _require_available_outputs(targets)
    _preflight_standalone_runtime()

    prepare_kwargs = {"_work_root": _work_root}
    if _resource_isolation:
        prepare_kwargs["_resource_isolation"] = True
    slides = _prepare_multiple_images(sources, lang, **prepare_kwargs)
    if len(slides) != len(targets):
        raise RuntimeError("Prepared page count does not match PSD output count")
    _assemble_and_publish(slides, targets)
    return [str(path) for path in targets]


def _single_output_path(
    source: Path,
    output_path: str | Path | None,
) -> Path:
    if output_path is None:
        output = source.with_suffix(".psd")
        _require_output_directory(output.parent)
        return output
    output = Path(output_path).resolve()
    if output.suffix.casefold() == ".psd":
        _require_output_directory(output.parent)
        return output
    _require_output_directory(output)
    return output / f"{source.stem}.psd"


def _batch_output_paths(
    sources: list[Path],
    output_path: str | Path | None,
) -> list[Path]:
    output_dir = (
        sources[0].parent if output_path is None else Path(output_path).resolve()
    )
    if output_dir.suffix.casefold() == ".psd":
        raise ValueError("Multiple images require an output directory")
    _require_output_directory(output_dir)

    used_names: set[str] = set()
    outputs = []
    for source in sources:
        suffix = 1
        candidate = f"{source.stem}.psd"
        while candidate.casefold() in used_names:
            suffix += 1
            candidate = f"{source.stem}_{suffix}.psd"
        used_names.add(candidate.casefold())
        outputs.append((output_dir / candidate).resolve())
    return outputs


def _require_output_directory(path: Path) -> None:
    current = path
    while not os.path.lexists(current):
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not current.is_dir():
        raise NotADirectoryError(
            f"PSD output directory is not a directory: {current}"
        )


def _require_available_outputs(outputs: list[Path]) -> None:
    for output in outputs:
        if os.path.lexists(output):
            raise FileExistsError(f"PSD output already exists: {output}")


def _preflight_standalone_runtime() -> None:
    preflight_psd_runtime()
    for env_name, expected_type in _STANDALONE_MODEL_PATHS.items():
        raw_path = os.environ.get(env_name, "")
        path = Path(raw_path).expanduser()
        if not raw_path or not path.is_absolute():
            raise RuntimeError(f"{env_name} must be an absolute local path")
        valid = path.is_file() if expected_type == "file" else path.is_dir()
        if not valid:
            raise RuntimeError(f"{env_name} must point to a local {expected_type}")


def _assemble_and_publish(slides: list[dict], outputs: list[Path]) -> None:
    output_dir = outputs[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".image2editable-psd-",
            dir=output_dir,
        ) as staging_dir:
            staging_root = Path(staging_dir)
            staged_outputs = []
            for index, (slide, output) in enumerate(zip(slides, outputs), start=1):
                staged_output = staging_root / f"{index:04d}-{output.name}"
                assemble_psd(
                    background_path=slide["background_original_path"],
                    components=slide["components"],
                    text_items=slide["text_items"],
                    img_width=slide["img_width"],
                    img_height=slide["img_height"],
                    output_path=staged_output,
                )
                staged_outputs.append(staged_output)

            for staged_output, output in zip(staged_outputs, outputs):
                os.link(staged_output, output)
                published.append(output)
    except BaseException:
        for output in published:
            output.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert image(s) to strictly validated layered PSD files"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Input image file(s) or directory containing images",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PSD path for one image, or output directory for multiple images",
    )
    parser.add_argument("--lang", default="ch", help="OCR language (default: ch)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    image_files = _resolve_inputs(args.images)
    if not image_files:
        raise SystemExit("No valid image files found")
    if len(image_files) == 1:
        convert(
            image_files[0],
            args.output,
            lang=args.lang,
            _resource_isolation=True,
        )
    else:
        convert_batch(
            image_files,
            args.output,
            lang=args.lang,
            _resource_isolation=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
