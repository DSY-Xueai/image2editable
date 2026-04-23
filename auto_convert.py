#!/usr/bin/env python3
"""PPT Master - Automated Pipeline Orchestrator.

Automates the mechanical steps of the ppt-master workflow:
  prepare: input file → project init → source import → content extraction
  export:  quality check → finalize SVG → export PPTX

Usage:
    # Phase 1: Prepare project from input file
    python auto_convert.py prepare <input_file> [--format ppt169] [--name my_project]

    # Phase 2: (AI generates SVGs into <project>/svg_output/)

    # Phase 3: Export to PPTX
    python auto_convert.py export <project_path>

    # One-shot (prepare only, prints instructions for next steps)
    python auto_convert.py prepare 178_5.png --format ppt169
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "skills" / "ppt-master" / "scripts"
SOURCE_TO_MD = SCRIPTS_DIR / "source_to_md"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}
DOC_EXTENSIONS = {".docx", ".doc", ".odt", ".epub", ".html", ".ipynb", ".rtf", ".tex"}
PPT_EXTENSIONS = {".pptx", ".pptm", ".ppsx"}
MD_EXTENSIONS = {".md", ".markdown", ".txt"}


def _run(cmd: list[str], label: str) -> bool:
    """Run a subprocess and print status."""
    print(f"\n[STEP] {label}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  [FAIL] {label} (exit code {result.returncode})")
        return False
    return True


def _detect_source_type(path: Path) -> str:
    """Detect source file type."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in DOC_EXTENSIONS:
        return "doc"
    if ext in PPT_EXTENSIONS:
        return "ppt"
    if ext in MD_EXTENSIONS:
        return "markdown"
    return "unknown"


def cmd_prepare(args: argparse.Namespace) -> None:
    """Phase 1: Prepare project from input file."""
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    source_type = _detect_source_type(input_path)
    if source_type == "unknown":
        print(f"[ERROR] Unsupported file type: {input_path.suffix}")
        sys.exit(1)

    project_name = args.name or input_path.stem
    canvas_format = args.format

    print(f"[INFO] Input: {input_path.name} (type: {source_type})")
    print(f"[INFO] Project: {project_name}, Format: {canvas_format}")

    # Step 1: Convert image to PDF if needed
    converted_pdf = None
    if source_type == "image":
        converted_pdf = input_path.parent / f"{input_path.stem}_converted.pdf"
        if not _run(
            [sys.executable, str(SOURCE_TO_MD / "image_to_pdf.py"),
             str(input_path), "-o", str(converted_pdf)],
            "Converting image to PDF"
        ):
            sys.exit(1)

    # Step 2: Convert to Markdown
    md_source = None
    if source_type == "image" and converted_pdf:
        md_source = converted_pdf
        script = SOURCE_TO_MD / "pdf_to_md.py"
    elif source_type == "pdf":
        md_source = input_path
        script = SOURCE_TO_MD / "pdf_to_md.py"
    elif source_type == "doc":
        md_source = input_path
        script = SOURCE_TO_MD / "doc_to_md.py"
    elif source_type == "ppt":
        md_source = input_path
        script = SOURCE_TO_MD / "ppt_to_md.py"
    elif source_type == "markdown":
        md_source = None
        script = None

    if md_source and script:
        if not _run(
            [sys.executable, str(script), str(md_source)],
            f"Converting {source_type} to Markdown"
        ):
            sys.exit(1)

    # Step 3: Init project
    if not _run(
        [sys.executable, str(SCRIPTS_DIR / "project_manager.py"),
         "init", project_name, "--format", canvas_format],
        "Initializing project"
    ):
        sys.exit(1)

    # Find the created project directory
    projects_dir = REPO_ROOT / "projects"
    candidates = sorted(projects_dir.glob(f"{project_name}_{canvas_format}_*"), reverse=True)
    if not candidates:
        print("[ERROR] Project directory not found after init")
        sys.exit(1)
    project_path = candidates[0]

    # Step 4: Import sources
    sources_to_import = [str(input_path)]

    # Also import the generated markdown if it exists
    if md_source and md_source != input_path:
        md_output = md_source.with_suffix(".md")
        if md_output.exists():
            sources_to_import.append(str(md_output))
        # Import the files directory too
        files_dir = md_source.parent / f"{md_source.stem}_files"
        if files_dir.is_dir():
            sources_to_import.append(str(files_dir))
    elif source_type != "markdown":
        # Check for markdown output next to input
        md_output = input_path.with_suffix(".md")
        if md_output.exists():
            sources_to_import.append(str(md_output))
        files_dir = input_path.parent / f"{input_path.stem}_files"
        if files_dir.is_dir():
            sources_to_import.append(str(files_dir))

    if not _run(
        [sys.executable, str(SCRIPTS_DIR / "project_manager.py"),
         "import-sources", str(project_path)] + sources_to_import + ["--copy"],
        "Importing sources"
    ):
        sys.exit(1)

    # Step 5: Analyze images if any
    images_dir = project_path / "images"
    source_images = list((project_path / "sources").glob("*.png")) + \
                    list((project_path / "sources").glob("*.jpg")) + \
                    list((project_path / "sources").glob("*.jpeg"))
    if source_images:
        # Copy images to images/ for analysis
        import shutil
        images_dir.mkdir(exist_ok=True)
        for img in source_images:
            shutil.copy2(img, images_dir / img.name)

        _run(
            [sys.executable, str(SCRIPTS_DIR / "analyze_images.py"),
             str(images_dir)],
            "Analyzing images"
        )

    # Clean up intermediate files
    if converted_pdf and converted_pdf.exists():
        converted_pdf.unlink()
        converted_md = converted_pdf.with_suffix(".md")
        if converted_md.exists():
            converted_md.unlink()
        converted_files = converted_pdf.parent / f"{converted_pdf.stem}_files"
        if converted_files.is_dir():
            import shutil
            shutil.rmtree(converted_files)

    print(f"\n{'='*60}")
    print(f"[OK] Project prepared: {project_path}")
    print(f"{'='*60}")
    print(f"\nNext: Generate SVG pages into {project_path / 'svg_output'}/")
    print(f"Then: python auto_convert.py export {project_path}")


def cmd_export(args: argparse.Namespace) -> None:
    """Phase 3: Quality check → finalize → export PPTX."""
    project_path = Path(args.project).resolve()
    if not project_path.exists():
        print(f"[ERROR] Project not found: {project_path}")
        sys.exit(1)

    svg_output = project_path / "svg_output"
    svg_files = list(svg_output.glob("*.svg"))
    if not svg_files:
        print(f"[ERROR] No SVG files found in {svg_output}")
        sys.exit(1)

    print(f"[INFO] Project: {project_path.name}")
    print(f"[INFO] SVG files: {len(svg_files)}")

    # Step 1: Quality check
    if not _run(
        [sys.executable, str(SCRIPTS_DIR / "svg_quality_checker.py"),
         str(project_path)],
        "SVG quality check"
    ):
        print("[WARN] Quality check had issues, continuing anyway...")

    # Step 2: Finalize SVG
    if not _run(
        [sys.executable, str(SCRIPTS_DIR / "finalize_svg.py"),
         str(project_path)],
        "Finalizing SVGs"
    ):
        sys.exit(1)

    # Step 3: Export PPTX
    if not _run(
        [sys.executable, str(SCRIPTS_DIR / "svg_to_pptx.py"),
         str(project_path), "-s", "final"],
        "Exporting to PPTX"
    ):
        sys.exit(1)

    # Find the exported file
    exports_dir = project_path / "exports"
    pptx_files = sorted(exports_dir.glob("*.pptx"), reverse=True)
    native = [f for f in pptx_files if "_svg" not in f.name]

    print(f"\n{'='*60}")
    print(f"[OK] Export complete!")
    print(f"{'='*60}")
    if native:
        print(f"  Native PPTX (editable): {native[0]}")
    if len(pptx_files) > len(native):
        svg_ver = [f for f in pptx_files if "_svg" in f.name]
        if svg_ver:
            print(f"  SVG reference PPTX:     {svg_ver[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPT Master - Automated Pipeline Orchestrator"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare subcommand
    prep = subparsers.add_parser("prepare", help="Prepare project from input file")
    prep.add_argument("input", help="Input file (image, PDF, DOCX, etc.)")
    prep.add_argument("--format", default="ppt169",
                      help="Canvas format (default: ppt169)")
    prep.add_argument("--name", help="Project name (default: input filename)")

    # export subcommand
    exp = subparsers.add_parser("export", help="Export project to PPTX")
    exp.add_argument("project", help="Project directory path")

    args = parser.parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
