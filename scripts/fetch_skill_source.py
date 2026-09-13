"""Fetch only the source files needed to install and run image2editable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


REPOSITORY = "https://github.com/DSY-Xueai/image2editable.git"
RUNTIME_SCRIPTS = (
    "__init__.py", "art_text.py", "bg_model.py", "component_underlay.py",
    "fg_extract.py", "font_match.py", "initial_diagnostics.py",
    "lama_inpaint.py", "lama_worker.py", "object_detect.py", "object_worker.py",
    "ocr_worker.py", "page_routing.py", "performance_trace.py", "ppt_assemble.py",
    "psd_assemble.py", "runtime_model_paths.py", "sam_worker.py",
    "text_context.py", "text_detect.py", "text_runs.py", "visual_compare_qa.py",
    "visual_segment.py", "visual_worker.py", "worker_pool.py", "worker_resources.py",
    "install_release_renderer.ps1",
)
SOURCE_PATTERNS = (
    "/image2editable/", "/pyproject.toml", "/README_EN.md", "/LICENSE",
    "/.gitignore", "/.gitattributes",
    "/THIRD_PARTY_NOTICES.md", "/third_party/licenses/", "/constraints/runtime.txt",
    "/image_to_ppt.py", "/image_to_psd.py",
    *(f"/scripts/{name}" for name in RUNTIME_SCRIPTS),
)


def git(directory: Path, *args: str, input: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *args], input=input, capture_output=True,
        text=True, encoding="utf-8", check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def fetch_source(destination: Path, repository: str = REPOSITORY, *, skill: str | None = None) -> str:
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        if not (destination / ".git").is_dir():
            raise ValueError("Source directory is not a managed Git checkout")
        if git(destination, "config", "--get", "image2editable.skillSource") != "true":
            raise ValueError("Preserve existing checkout; choose a new source directory")
        if git(destination, "remote", "get-url", "origin") != repository:
            raise ValueError("Source remote differs from the expected repository")
        if git(destination, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("Preserve local changes; choose a new source directory")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        git(destination, "init")
        git(destination, "remote", "add", "origin", repository)
        git(destination, "config", "image2editable.skillSource", "true")
    git(destination, "config", "remote.origin.promisor", "true")
    git(destination, "config", "remote.origin.partialclonefilter", "blob:none")
    patterns = (f"/skills/{skill}/",) if skill else SOURCE_PATTERNS
    git(destination, "sparse-checkout", "set", "--no-cone", "--stdin",
        input="\n".join(patterns) + "\n")
    git(destination, "fetch", "--depth=1", "--filter=blob:none", "origin", "main")
    commit = git(destination, "rev-parse", "FETCH_HEAD")
    git(destination, "-c", "advice.detachedHead=false", "checkout", "--detach", commit)
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--skill", choices=("image-to-ppt", "image-to-psd"),
                        help="Fetch only this Skill for initial installation")
    args = parser.parse_args()
    commit = fetch_source(args.destination, skill=args.skill)
    print(json.dumps({"source": str(args.destination.resolve()), "commit": commit}, indent=2))


if __name__ == "__main__":
    main()
