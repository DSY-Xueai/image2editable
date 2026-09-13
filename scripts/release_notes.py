"""Extract one version from CHANGELOG.md for a GitHub Release."""
from __future__ import annotations

import argparse
from pathlib import Path
import re


def extract_notes(changelog: str, tag: str) -> str:
    version = tag.removeprefix("v")
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## |\Z)",
        changelog, re.MULTILINE | re.DOTALL,
    )
    if match is None or not match.group(1).strip():
        raise ValueError(f"No release notes for {tag}")
    return match.group(1).strip() + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        extract_notes(Path("CHANGELOG.md").read_text(encoding="utf-8"), args.tag),
        encoding="utf-8",
    )
