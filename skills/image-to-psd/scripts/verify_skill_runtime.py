"""Check installed runtime contents against the exact source selected by a Skill."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata, util
import json
from pathlib import Path


def verify(source: Path, distribution) -> list[str]:
    expected = {
        path.relative_to(source).as_posix(): path
        for package in ("image2editable", "scripts")
        for path in (source / package).rglob("*.py")
    }
    for name in ("image_to_ppt.py", "image_to_psd.py", "image2editable/runtime_model_catalog.json"):
        expected[name] = source / name
    if not (source / "image2editable/cli.py").is_file():
        raise ValueError("Source is not an image2editable repository")
    installed = {str(path).replace("\\", "/") for path in distribution.files or []}
    problems = []
    for name, path in expected.items():
        target = Path(distribution.locate_file(name))
        if name not in installed or not target.is_file():
            problems.append(f"missing: {name}")
        elif hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
            problems.append(f"different: {name}")
    for name in installed:
        if (name.startswith(("image2editable/", "scripts/")) and name.endswith(".py")
                and name not in expected):
            problems.append(f"obsolete: {name}")
    return sorted(problems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    try:
        distribution = metadata.distribution("image2editable")
    except metadata.PackageNotFoundError:
        print(json.dumps({"ready": False, "problems": ["image2editable is not installed"]}))
        return 1
    source = args.source.resolve()
    required = ("image2editable/cli.py", "image2editable/runtime_models.py",
                "scripts/__init__.py", "scripts/psd_assemble.py",
                "image_to_ppt.py", "image_to_psd.py",
                "image2editable/runtime_model_catalog.json")
    missing = [name for name in required if not (source / name).is_file()]
    try:
        if missing:
            raise ValueError("Source is incomplete: " + ", ".join(missing))
        problems = verify(source, distribution)
    except (OSError, ValueError) as error:
        print(json.dumps({"ready": False, "problems": [str(error)]}))
        return 1
    for name in ("image2editable", "scripts"):
        spec = util.find_spec(name)
        expected = Path(distribution.locate_file(f"{name}/__init__.py")).resolve()
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
            problems.append(f"import shadowed: {name}")
    print(json.dumps({"ready": not problems, "version": distribution.version,
                      "problems": problems}, ensure_ascii=False, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
