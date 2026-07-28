from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from image2editable import runtime
from image2editable.doctor import check_environment


def _add_image_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("images", nargs="+", help="Image files or directories")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--lang", default="ch")
    parser.add_argument(
        "--slide-size",
        choices=("original", "16:9", "both"),
        default="both",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image2editable")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    _add_image_options(convert_parser)

    prepare_parser = subparsers.add_parser("prepare")
    _add_image_options(prepare_parser)

    run_parser = subparsers.add_parser("run")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)

    status_parser = run_subparsers.add_parser("status")
    status_parser.add_argument("run_dir")

    execute_parser = run_subparsers.add_parser("execute")
    execute_parser.add_argument("run_dir")

    retry_parser = run_subparsers.add_parser("retry")
    retry_parser.add_argument("run_dir")
    retry_parser.add_argument("--page", required=True)

    subparsers.add_parser("doctor")
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        report = check_environment()
        _print_json(report)
        return 0 if report["ready"] else 1

    if args.command == "convert":
        summary = runtime.convert(
            args.images,
            run_dir=args.run_dir,
            output_path=args.output,
            slide_size=args.slide_size,
            lang=args.lang,
        )
        _print_json(summary)
        return 0

    if args.command == "prepare":
        run_dir = runtime.prepare_job(
            args.images,
            run_dir=args.run_dir,
            output_path=args.output,
            slide_size=args.slide_size,
            lang=args.lang,
        )
        _print_json({"run_dir": str(Path(run_dir).resolve()), "status": "prepared"})
        return 0

    if args.run_command == "status":
        _print_json(runtime.get_status(args.run_dir))
        return 0
    if args.run_command == "execute":
        _print_json(runtime.run_job(args.run_dir))
        return 0
    if args.run_command == "retry":
        _print_json(runtime.retry_page(args.run_dir, args.page))
        return 0
    raise AssertionError("argparse returned an unsupported command")
