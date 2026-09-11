"""Sequential real-input measurements; never treat an unreviewed deck as passed."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def summarize_attempt(*, returncode, summary, pptx_exists, events):
    completed = returncode == 0 and summary.get("status") == "completed" and pptx_exists
    return {
        "status": "completed_unreviewed" if completed else "failed",
        "pptx_created": pptx_exists,
        "local_fast_token_count": 0,
        "host_token_count": None,
        "token_scope": "Standalone fast runtime only; no host agent invoked. Host workflow not measured.",
        "sam_calls": None,
        "lama_calls": None,
        "observed_worker_starts": sum(
            event.get("event") == "worker_start" and event.get("stage") != "worker_reuse"
            for event in events
        ) if events else None,
        "observed_worker_requests": dict(Counter(
            event.get("model", "unknown") for event in events
            if event.get("event") == "worker_finish"
        )) if events else None,
        "trace_scope": "Observed events only; internal model calls and trace completeness are not measured.",
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _save_report(output, report):
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    if not report["files"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
    rows = report["files"]
    fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.7 + 1.5)))
    labels = [f"{r['file']} ({r['pages'] if r['pages'] is not None else '?'}p, {r['status']})" for r in rows]
    bars = ax.barh(labels, [r["duration_s"] for r in rows], color=[
        "#267c77" if r["status"] == "completed_unreviewed" else "#bd4343" for r in rows
    ])
    ax.bar_label(bars, fmt="%.1f s", padding=4)
    ax.margins(x=0.18)
    ax.invert_yaxis()
    ax.set_xlabel("End-to-end conversion seconds (including failed attempts)")
    ax.set_title("Local Fast Runtime: 0 LLM tokens; host workflow not measured")
    fig.tight_layout()
    fig.savefig(output / "performance.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=None)
    args = parser.parse_args(argv)
    sources = [p.resolve(strict=True) for p in args.sources]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    now = lambda: datetime.now(timezone.utc).isoformat()
    report = {
        "started_at": now(), "finished_at": None, "status": "running",
        "planned_files": [str(p) for p in sources], "files": [],
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip(),
        "working_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "--no-ext-diff"], cwd=root,
        )).hexdigest(),
    }
    _save_report(output, report)
    batch_start = time.perf_counter()
    for index, source in enumerate(sources, 1):
        case = output / f"case-{index:02d}"
        case.mkdir()
        run = case / "run"
        pptx = case / "output.pptx"
        command = [sys.executable, "-c", "from image2editable.cli import main; raise SystemExit(main())",
                   "convert", str(source), "-o", str(pptx), "--run-dir", str(run),
                   "--slide-size", "original", "--pipeline-mode", "fast"]
        started_at = now()
        start = time.perf_counter()
        print(f"START {source.name} {started_at}", flush=True)
        timed_out = False
        with (case / "conversion.log").open("w", encoding="utf-8") as log:
            try:
                run_kwargs = {"cwd": root, "stdout": log, "stderr": subprocess.STDOUT}
                if args.timeout_s is not None:
                    run_kwargs["timeout"] = args.timeout_s
                result = subprocess.run(command, **run_kwargs)
            except subprocess.TimeoutExpired:
                timed_out = True
                result = subprocess.CompletedProcess(command, 124)
        duration = time.perf_counter() - start
        finished_at = now()
        summary = _read_json(run / "run_summary.json")
        pages = _read_json(run / "page_jobs.json").get("pages")
        events = []
        for trace in run.glob("performance-*.jsonl"):
            events.extend(json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line)
        attempt = summarize_attempt(
            returncode=result.returncode, summary=summary,
            pptx_exists=pptx.is_file(), events=events,
        )
        attempt.update({
            "file": source.name, "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "pages": len(pages) if pages is not None else None,
            "started_at": started_at, "finished_at": finished_at,
            "duration_s": round(duration, 3), "returncode": result.returncode,
            "error": "conversion timeout" if timed_out else summary.get("error"),
            "run_dir": str(run),
        })
        report["files"].append(attempt)
        report["total_conversion_s"] = round(sum(r["duration_s"] for r in report["files"]), 3)
        _save_report(output, report)
        print(f"END {source.name} {duration:.3f}s {attempt['status']}", flush=True)
    report.update(status="measured", finished_at=now(), batch_wall_s=round(time.perf_counter() - batch_start, 3))
    _save_report(output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
