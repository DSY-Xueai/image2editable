#!/usr/bin/env python3
"""Compatibility launcher for the shared image2editable PSD runtime."""

from __future__ import annotations

from pathlib import Path
import sys


repository_root = Path(__file__).resolve().parents[3]
if (repository_root / "image2editable").is_dir():
    sys.path.insert(0, str(repository_root))

try:
    from image2editable.cli import main as runtime_main
except ModuleNotFoundError as error:
    raise SystemExit(
        "image2editable is not installed; install the project with "
        "`pip install -e \".[psd]\"` before using this skill"
    ) from error


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    return runtime_main(["convert", *arguments, "--format", "psd"])


if __name__ == "__main__":
    raise SystemExit(main())
