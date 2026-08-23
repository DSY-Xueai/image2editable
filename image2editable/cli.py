from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Sequence

from image2editable import runtime
from image2editable.doctor import check_environment


def _add_image_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "sources",
        nargs="+",
        help="Image files, directories, one PDF or one PPTX",
    )
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--lang", default="ch")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("pptx", "psd"),
        default="pptx",
    )
    parser.add_argument(
        "--agent-provider",
        choices=("host", "local", "local-service"),
        default="host",
    )
    parser.add_argument(
        "--slide-size",
        choices=("original", "16:9", "both"),
        default="both",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image2editable")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {metadata.version('image2editable')}",
    )
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

    next_parser = run_subparsers.add_parser("next")
    next_parser.add_argument("run_dir")

    recover_parser = run_subparsers.add_parser("recover")
    recover_parser.add_argument("run_dir")

    retry_parser = run_subparsers.add_parser("retry")
    retry_parser.add_argument("run_dir")
    retry_parser.add_argument("--page", required=True)

    render_parser = run_subparsers.add_parser("render-detail")
    render_parser.add_argument("run_dir")
    render_parser.add_argument("--page", required=True)

    decision_parser = subparsers.add_parser("decision")
    decision_subparsers = decision_parser.add_subparsers(
        dest="decision_command",
        required=True,
    )
    record_parser = decision_subparsers.add_parser("record")
    record_parser.add_argument("run_dir")
    record_parser.add_argument("--page", required=True)
    record_parser.add_argument("--object", required=True)
    record_parser.add_argument(
        "--decision",
        choices=("replace", "preserve", "ambiguous"),
        required=True,
    )
    record_parser.add_argument("--confidence", type=float, required=True)
    record_parser.add_argument(
        "--category",
        choices=(
            "full_slide_screenshot",
            "partial_slide_screenshot",
            "rasterized_diagram",
            "rasterized_chart",
            "photo",
            "logo",
            "decorative_asset",
            "unknown",
        ),
        required=True,
    )
    record_parser.add_argument("--evidence", action="append", required=True)

    agent_parser = subparsers.add_parser("agent")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command", required=True)
    agent_next_parser = agent_subparsers.add_parser("next")
    agent_next_parser.add_argument("run_dir")
    agent_record_parser = agent_subparsers.add_parser("record")
    agent_record_parser.add_argument("run_dir")
    agent_record_parser.add_argument("--plan", required=True)

    models_parser = subparsers.add_parser("models")
    models_subparsers = models_parser.add_subparsers(
        dest="models_command",
        required=True,
    )
    models_recommend_parser = models_subparsers.add_parser("recommend")
    models_recommend_parser.add_argument("--json", action="store_true")
    models_install_parser = models_subparsers.add_parser("install")
    models_install_parser.add_argument("target", choices=("agent", "runtime"))
    models_install_parser.add_argument("--yes", action="store_true")
    models_subparsers.add_parser("status")

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--agent-local", action="store_true")
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _models_module():
    from image2editable import models

    return models


def _runtime_models_module():
    from image2editable import runtime_models

    return runtime_models


def _print_model_plan(plan: dict[str, object]) -> None:
    print(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        with redirect_stdout(sys.stderr):
            report = check_environment(agent_local=args.agent_local)
        _print_json(report)
        return 0 if report["ready"] else 1

    if args.command == "models":
        models = _models_module()
        if args.models_command == "recommend":
            hardware = models.detect_hardware()
            recommendation = models.recommend_agent_model(hardware)
            if args.json:
                _print_json(recommendation)
            else:
                _print_model_plan(recommendation)
            return 0
        if args.models_command == "status":
            runtime_models = _runtime_models_module()
            _print_json(
                {
                    "agent": models.model_status(),
                    "runtime": runtime_models.runtime_model_status(),
                }
            )
            return 0
        if args.models_command == "install":
            if args.target == "agent":
                hardware = models.detect_hardware()
                plan = models.recommend_agent_model(hardware)
            else:
                runtime_models = _runtime_models_module()
                catalog = runtime_models.load_runtime_catalog()
                entries = catalog["models"]
                plan = {
                    "target": "runtime",
                    "models": {
                        "sam2_large": {
                            "size": entries["sam2_large"]["size"],
                            "sha256": entries["sam2_large"]["sha256"],
                        },
                        "big_lama": {
                            "size": entries["big_lama"]["size"],
                            "sha256": entries["big_lama"]["sha256"],
                        },
                        "grounding_dino": {
                            "model_id": entries["grounding_dino"]["model_id"],
                            "revision": entries["grounding_dino"]["revision"],
                        },
                    },
                    "estimated_download": {
                        "minimum_bytes": entries["sam2_large"]["size"]
                        + entries["big_lama"]["size"],
                        "additional": (
                            "Grounding DINO snapshot (size not declared in catalog)"
                        ),
                    },
                    "cache": (
                        "IMAGE2EDITABLE_MODEL_CACHE or the default user runtime cache"
                    ),
                }
            _print_model_plan(plan)
            confirmed = args.yes
            if not confirmed:
                model_kind = "实验性模型" if args.target == "agent" else "运行时模型"
                print(f"确认下载上述{model_kind}？[y/N] ", file=sys.stderr, end="")
                try:
                    confirmed = input("").strip().casefold() in {"y", "yes"}
                except EOFError:
                    confirmed = False
            if not confirmed:
                _print_json({"status": "cancelled"})
                return 1
            if args.target == "agent":
                receipt = models.install_agent_model(
                    cache_dir=None,
                    confirmed=True,
                    model_id=plan["model_id"],
                    revision=plan["revision"],
                )
            else:
                receipt = runtime_models.install_runtime_models(
                    cache_dir=None,
                    confirmed=True,
                )
            _print_json(receipt)
            return 0

    if args.command == "convert":
        format_kwargs = (
            {"output_format": args.output_format}
            if args.output_format != "pptx"
            else {}
        )
        with redirect_stdout(sys.stderr):
            summary = runtime.convert(
                args.sources,
                run_dir=args.run_dir,
                output_path=args.output,
                slide_size=args.slide_size,
                lang=args.lang,
                agent_provider=args.agent_provider,
                **format_kwargs,
            )
        _print_json(summary)
        return 0

    if args.command == "prepare":
        format_kwargs = (
            {"output_format": args.output_format}
            if args.output_format != "pptx"
            else {}
        )
        with redirect_stdout(sys.stderr):
            run_dir = runtime.prepare_job(
                args.sources,
                run_dir=args.run_dir,
                output_path=args.output,
                slide_size=args.slide_size,
                lang=args.lang,
                agent_provider=args.agent_provider,
                **format_kwargs,
            )
        _print_json({"run_dir": str(Path(run_dir).resolve()), "status": "prepared"})
        return 0

    if args.command == "decision" and args.decision_command == "record":
        with redirect_stdout(sys.stderr):
            result = runtime.record_decision(
                args.run_dir,
                page_id=args.page,
                object_id=args.object,
                decision=args.decision,
                confidence=args.confidence,
                category=args.category,
                evidence=args.evidence,
            )
        _print_json(result)
        return 0

    if args.command == "agent" and args.agent_command == "next":
        with redirect_stdout(sys.stderr):
            item = runtime.next_host_agent_item(args.run_dir)
        _print_json(item)
        return 0
    if args.command == "agent" and args.agent_command == "record":
        with redirect_stdout(sys.stderr):
            result = runtime.record_host_agent_plan(args.run_dir, args.plan)
        _print_json(result)
        return 0

    if args.run_command == "status":
        with redirect_stdout(sys.stderr):
            status = runtime.get_status(args.run_dir)
        _print_json(status)
        return 0
    if args.run_command == "execute":
        with redirect_stdout(sys.stderr):
            summary = runtime.run_job(args.run_dir)
        _print_json(summary)
        return 0
    if args.run_command == "next":
        with redirect_stdout(sys.stderr):
            candidate = runtime.next_candidate(args.run_dir)
        _print_json(candidate)
        return 0
    if args.run_command == "recover":
        with redirect_stdout(sys.stderr):
            status = runtime.recover_job(args.run_dir)
        _print_json(status)
        return 0
    if args.run_command == "retry":
        with redirect_stdout(sys.stderr):
            status = runtime.retry_page(args.run_dir, args.page)
        _print_json(status)
        return 0
    if args.run_command == "render-detail":
        with redirect_stdout(sys.stderr):
            result = runtime.rerender_pdf_page(args.run_dir, args.page)
        _print_json(result)
        return 0
    raise AssertionError("argparse returned an unsupported command")
