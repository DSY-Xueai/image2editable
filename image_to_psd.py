#!/usr/bin/env python3
"""Compatibility entry point for the shared image-to-editable PSD runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from image2editable.cli import main as runtime_main
from image2editable.runtime import convert as runtime_convert


def convert(
    image_path: str | Path,
    output_path: str | Path | None = None,
    *,
    lang: str = "ch",
    agent_provider: str = "host",
    run_dir: str | Path | None = None,
) -> dict[str, object]:
    """Convert one image through the shared Agent pipeline and request PSD output."""

    return runtime_convert(
        [image_path],
        output_path=output_path,
        lang=lang,
        agent_provider=agent_provider,
        run_dir=run_dir,
        output_format="psd",
    )


def _resolve_output_paths(
    image_paths: Sequence[Path],
    output_path: str | Path | None,
) -> list[Path]:
    """Retained for callers that only need the legacy output-name calculation."""

    if len(image_paths) == 1:
        source = image_paths[0]
        if output_path is None:
            return [source.with_suffix(".psd")]
        output = Path(output_path)
        if output.suffix.casefold() == ".psd":
            return [output]
        return [output / f"{source.stem}.psd"]

    output_dir = Path(output_path) if output_path is not None else image_paths[0].parent
    if output_dir.suffix.casefold() == ".psd":
        raise ValueError("Multiple images require an output directory")
    return [output_dir / f"{source.stem}.psd" for source in image_paths]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return runtime_main(["convert", *arguments, "--format", "psd"])


if __name__ == "__main__":
    raise SystemExit(main())
