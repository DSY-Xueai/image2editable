from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
from typing import Callable

from scripts.benchmark_conversion import _read_regular_file, _strict_json


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "benchmarks" / "release"
PLAN_ROOT = RELEASE_ROOT / "plans"
_IDENTIFIER = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PLAN_LIMIT = 1024 * 1024
_JSON_LIMIT = 4 * 1024 * 1024
_CANDIDATE_FIELDS = {
    "schema_version",
    "kind",
    "page_id",
    "candidate_id",
    "source_shape_id",
    "source_object_sha256",
    "image_sha256",
    "decision",
    "confidence",
    "category",
    "evidence",
}
_COMPONENT_FIELDS = {
    "schema_version",
    "kind",
    "page_id",
    "provider",
    "repair_round",
    "request_sha256",
    "graph_sha256",
    "actions",
}
_ACTION_FIELDS = {"action", "object_ids", "parameters", "confidence", "evidence"}
_ACTIONS = {
    "accept",
    "discard",
    "merge",
    "split",
    "expand",
    "shrink",
    "retry_with_box",
    "retry_with_points",
    "attach_text",
    "suppress_text",
    "collapse_to_parent",
    "rebuild_background",
    "absorb_residual",
    "absorb_into_parent",
}
_CAPABILITIES = ["vision", "local_file_read", "tool_use", "structured_json"]
_CHALLENGE_COLORS = ("#2f6fed", "#d9485f", "#2b8a3e", "#9c36b5")


class BenchmarkFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchmarkCaseResult:
    case_id: str
    run_dir: str
    pages: list[dict[str, object]]
    duration_ms: int


Command = Callable[..., subprocess.CompletedProcess[str]]


def run_command(
    arguments: list[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        ["image2editable", *arguments],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _call(command: Command, arguments: list[str], *, cwd: Path) -> dict[str, object]:
    try:
        completed = command(arguments, cwd=cwd)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BenchmarkFailure("command_failed") from None
    if completed.returncode != 0:
        raise BenchmarkFailure("command_failed")
    try:
        return _strict_json(completed.stdout, _JSON_LIMIT)
    except Exception:
        raise BenchmarkFailure("invalid_json") from None


def _is_reparse(metadata: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & flag
    )


def _case_plans(case_id: str) -> list[dict[str, object]]:
    try:
        root_status = PLAN_ROOT.lstat()
        if (
            not stat.S_ISDIR(root_status.st_mode)
            or _is_reparse(root_status)
        ):
            raise ValueError
        paths = sorted(PLAN_ROOT.glob(f"{case_id}--*.json"))
        plans = [
            _strict_json(
                _read_regular_file(path, _PLAN_LIMIT, require_single_link=True),
                _PLAN_LIMIT,
            )
            for path in paths
        ]
    except FileNotFoundError:
        return []
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BenchmarkFailure("invalid_plan") from None
    return plans


def _sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _strings(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _confidence(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0 <= value <= 1
    )


def _valid_candidate_plan(plan: dict[str, object]) -> bool:
    return (
        set(plan) == _CANDIDATE_FIELDS
        and plan.get("schema_version") == 1
        and type(plan.get("schema_version")) is int
        and plan.get("kind") == "candidate_decision"
        and all(
            isinstance(plan.get(field), str) and bool(plan[field])
            for field in ("page_id", "candidate_id", "source_shape_id")
        )
        and _sha256(plan.get("source_object_sha256"))
        and _sha256(plan.get("image_sha256"))
        and plan.get("decision") in {"replace", "preserve", "ambiguous"}
        and _confidence(plan.get("confidence"))
        and isinstance(plan.get("category"), str)
        and bool(plan["category"])
        and _strings(plan.get("evidence"))
    )


def _valid_component_plan(plan: dict[str, object]) -> bool:
    actions = plan.get("actions")
    if not (
        set(plan) == _COMPONENT_FIELDS
        and plan.get("schema_version") == 1
        and type(plan.get("schema_version")) is int
        and plan.get("kind") == "component_plan"
        and isinstance(plan.get("page_id"), str)
        and bool(plan["page_id"])
        and plan.get("provider") == "host"
        and type(plan.get("repair_round")) is int
        and 1 <= plan["repair_round"] <= 5
        and _sha256(plan.get("request_sha256"))
        and _sha256(plan.get("graph_sha256"))
        and isinstance(actions, list)
        and bool(actions)
    ):
        return False
    return all(
        isinstance(action, dict)
        and set(action) == _ACTION_FIELDS
        and action.get("action") in _ACTIONS
        and _strings(action.get("object_ids"))
        and isinstance(action.get("parameters"), dict)
        and _confidence(action.get("confidence"))
        and _strings(action.get("evidence"))
        for action in actions
    )


def _select_candidate_plan(
    case_id: str, candidate: dict[str, object]
) -> dict[str, object]:
    plans = [plan for plan in _case_plans(case_id) if plan.get("kind") == "candidate_decision"]
    if any(not _valid_candidate_plan(plan) for plan in plans):
        raise BenchmarkFailure("invalid_plan")
    if not plans:
        raise BenchmarkFailure("missing_plan")
    identity = [
        plan
        for plan in plans
        if all(
            plan.get(field) == candidate.get(field)
            for field in ("page_id", "candidate_id", "source_shape_id")
        )
    ]
    if not identity:
        raise BenchmarkFailure("mismatched_plan")
    matches = [
        plan
        for plan in identity
        if all(
            plan.get(field) == candidate.get(field)
            for field in ("source_object_sha256", "image_sha256")
        )
    ]
    if not matches:
        raise BenchmarkFailure("stale_plan")
    if len(matches) != 1:
        raise BenchmarkFailure("duplicate_plan")
    return matches[0]


def _select_component_plan(
    case_id: str, request: dict[str, object]
) -> dict[str, object]:
    plans = [plan for plan in _case_plans(case_id) if plan.get("kind") == "component_plan"]
    if any(not _valid_component_plan(plan) for plan in plans):
        raise BenchmarkFailure("invalid_plan")
    if not plans:
        raise BenchmarkFailure("missing_plan")
    identity = [
        plan
        for plan in plans
        if plan.get("page_id") == request.get("page_id")
        and plan.get("repair_round") == request.get("repair_round")
    ]
    if not identity:
        raise BenchmarkFailure("mismatched_plan")
    matches = [
        plan
        for plan in identity
        if plan.get("request_sha256") == request.get("request_sha256")
        and plan.get("graph_sha256") == request.get("graph_sha256")
    ]
    if not matches:
        raise BenchmarkFailure("stale_plan")
    if len(matches) != 1:
        raise BenchmarkFailure("duplicate_plan")
    return matches[0]


def _component_binding(response: dict[str, object], run_dir: Path) -> dict[str, object]:
    binding = dict(response)
    if response.get("kind") != "component_request" or response.get("provider") != "host":
        raise BenchmarkFailure("invalid_response")
    request_path = response.get("request_path")
    if not isinstance(request_path, str):
        raise BenchmarkFailure("invalid_response")
    try:
        path = Path(request_path).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError
        payload = _read_regular_file(path, _JSON_LIMIT, require_single_link=True)
        request = _strict_json(payload, _JSON_LIMIT)
        canonical = (
            json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True).encode(
                "utf-8"
            )
            + b"\n"
        )
        if hashlib.sha256(canonical).hexdigest() != response.get("request_sha256"):
            raise ValueError
        for field in ("page_id", "provider", "repair_round"):
            if request.get(field) != response.get(field):
                raise ValueError
        if "graph_sha256" in binding and binding["graph_sha256"] != request.get(
            "graph_sha256"
        ):
            raise ValueError
        graph_ref = request["evidence"]["component-graph.json"]
        graph_path = (path.parent / graph_ref["path"]).resolve()
        if not graph_path.is_relative_to(path.parent):
            raise ValueError
        graph = _read_regular_file(graph_path, _JSON_LIMIT, require_single_link=True)
        if (
            graph_ref["sha256"] != request.get("graph_sha256")
            or hashlib.sha256(graph).hexdigest() != request.get("graph_sha256")
        ):
            raise ValueError
        binding["graph_sha256"] = request["graph_sha256"]
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BenchmarkFailure("invalid_response") from None
    if not (
        isinstance(binding.get("page_id"), str)
        and binding.get("provider") == "host"
        and type(binding.get("repair_round")) is int
        and _sha256(binding.get("request_sha256"))
        and _sha256(binding.get("graph_sha256"))
    ):
        raise BenchmarkFailure("invalid_response")
    return binding


def _observe_capability_challenge(image_path: Path, run_dir: Path) -> dict[str, object]:
    try:
        path = image_path.resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError
        payload = _read_regular_file(path, 1024 * 1024, require_single_link=True)
        from PIL import Image

        opened = Image.open(io.BytesIO(payload))
        if opened.format != "PNG" or opened.size != (240, 120):
            raise ValueError
        image = opened.convert("RGB")
        pixels = image.load()
        found: list[list[tuple[int, int]]] = []
        found_color = None
        for color_name in _CHALLENGE_COLORS:
            color = tuple(bytes.fromhex(color_name[1:]))
            remaining = {
                (x, y)
                for y in range(120)
                for x in range(240)
                if pixels[x, y] == color
            }
            if not remaining:
                continue
            if found_color is not None:
                raise ValueError
            found_color = color_name
            while remaining:
                pending = [remaining.pop()]
                points = []
                while pending:
                    x, y = pending.pop()
                    points.append((x, y))
                    for neighbor in (
                        (x - 1, y),
                        (x + 1, y),
                        (x, y - 1),
                        (x, y + 1),
                    ):
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            pending.append(neighbor)
                found.append(points)
        if found_color is None or len(found) not in {2, 3, 4}:
            raise ValueError
        shapes = set()
        for points in found:
            ys = [point[1] for point in points]
            top, bottom = min(ys), max(ys)
            middle = (top + bottom) // 2
            widths = [sum(y == row for _, y in points) for row in (top, middle, bottom)]
            if widths[0] == widths[1] == widths[2]:
                shapes.add("square")
            elif widths[0] < widths[1] < widths[2]:
                shapes.add("triangle")
            elif widths[0] < widths[1] and widths[2] < widths[1]:
                shapes.add("circle")
            else:
                raise ValueError
        if len(shapes) != 1:
            raise ValueError
        return {"shape": shapes.pop(), "color": found_color, "count": len(found)}
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BenchmarkFailure("invalid_response") from None


def _capability_response(
    response: dict[str, object], run_dir: Path
) -> dict[str, object]:
    if (
        set(response)
        != {"kind", "challenge_id", "image_path", "required_capabilities"}
        or response.get("kind") != "capability_handshake"
        or not _sha256(response.get("challenge_id"))
        or response.get("required_capabilities") != _CAPABILITIES
        or not isinstance(response.get("image_path"), str)
    ):
        raise BenchmarkFailure("invalid_response")
    return {
        "schema_version": 1,
        "kind": "host_capability_response",
        "challenge_id": response["challenge_id"],
        "observed": _observe_capability_challenge(
            Path(response["image_path"]), run_dir
        ),
    }


def _record_candidate(
    command: Command, plan: dict[str, object], run_dir: Path, workspace: Path
) -> None:
    arguments = [
        "decision",
        "record",
        str(run_dir),
        "--page",
        str(plan["page_id"]),
        "--object",
        str(plan["source_shape_id"]),
        "--decision",
        str(plan["decision"]),
        "--confidence",
        str(plan["confidence"]),
        "--category",
        str(plan["category"]),
    ]
    for evidence in plan["evidence"]:
        arguments.extend(("--evidence", evidence))
    _call(command, arguments, cwd=workspace)


def _record_host_document(
    command: Command,
    document: dict[str, object],
    run_dir: Path,
    workspace: Path,
    sequence: int,
) -> None:
    submission: Path | None = None
    submission_identity: tuple[int, int] | None = None
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".host-plan-{sequence:03d}-",
            suffix=".json",
            dir=run_dir.parent,
        )
        submission = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            metadata = os.fstat(handle.fileno())
            submission_identity = (metadata.st_dev, metadata.st_ino)
        response = _call(
            command,
            ["agent", "record", str(run_dir), "--plan", str(submission.resolve())],
            cwd=workspace,
        )
        current = submission.lstat()
        if (
            (current.st_dev, current.st_ino) != submission_identity
            or _read_regular_file(
                submission, _PLAN_LIMIT, require_single_link=True
            )
            != payload
        ):
            raise ValueError
        if document.get("kind") == "component_plan":
            if (
                set(response) != {"plan_path", "recovered", "status"}
                or response.get("status") != "recorded"
                or response.get("recovered") is not False
                or not isinstance(response.get("plan_path"), str)
            ):
                raise ValueError
            artifact = Path(response["plan_path"]).resolve()
            if (
                not artifact.is_relative_to(run_dir.resolve())
                or _read_regular_file(
                    artifact, _PLAN_LIMIT, require_single_link=True
                )
                != payload
            ):
                raise ValueError
        elif response != {
            "capabilities": _CAPABILITIES,
            "status": "capabilities_recorded",
        }:
            raise ValueError
    except BenchmarkFailure:
        raise
    except Exception:
        raise BenchmarkFailure("invalid_plan") from None
    finally:
        if submission is not None and submission_identity is not None:
            try:
                metadata = submission.lstat()
                if (metadata.st_dev, metadata.st_ino) == submission_identity:
                    submission.unlink()
            except FileNotFoundError:
                pass


def _record_component(
    command: Command,
    plan: dict[str, object],
    run_dir: Path,
    workspace: Path,
    sequence: int,
) -> None:
    _record_host_document(
        command,
        {key: value for key, value in plan.items() if key != "graph_sha256"},
        run_dir,
        workspace,
        sequence,
    )


def _completed_pages(summary: dict[str, object], run_dir: Path) -> list[dict[str, object]]:
    pages = summary.get("page_results")
    if isinstance(pages, list) and all(isinstance(page, dict) for page in pages):
        return [dict(page) for page in pages]
    count = summary.get("pages")
    if type(count) is not int or count < 1:
        raise BenchmarkFailure("invalid_response")
    try:
        return [
            _strict_json(
                _read_regular_file(
                    run_dir / "pages" / f"page_{index:03d}" / "page_result.json",
                    _JSON_LIMIT,
                    require_single_link=True,
                ),
                _JSON_LIMIT,
            )
            for index in range(1, count + 1)
        ]
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BenchmarkFailure("invalid_response") from None


def run_case(
    case: dict[str, object],
    *,
    workspace: Path,
    command: Command = run_command,
) -> BenchmarkCaseResult:
    case_id = case.get("id")
    kind = case.get("kind")
    source_value = case.get("path")
    if (
        not isinstance(case_id, str)
        or _IDENTIFIER.fullmatch(case_id) is None
        or kind not in {"image", "pdf", "pptx"}
        or not isinstance(source_value, str)
        or case.get("agent_provider") != "host"
    ):
        raise BenchmarkFailure("invalid_case")
    try:
        source = (RELEASE_ROOT / source_value).resolve()
        if not source.is_relative_to(RELEASE_ROOT.resolve()) or not source.is_file():
            raise ValueError
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        case_root = workspace / case_id
        case_root.mkdir()
    except Exception:
        raise BenchmarkFailure("invalid_workspace") from None
    run_dir = (case_root / "run").resolve()
    output = (case_root / "output.pptx").resolve()
    started = time.perf_counter()
    prepared = _call(
        command,
        [
            "prepare",
            str(source),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
            "--slide-size",
            "original",
            "--agent-provider",
            "host",
        ],
        cwd=workspace,
    )
    if (
        prepared.get("status") != "prepared"
        or Path(str(prepared.get("run_dir", ""))).resolve() != run_dir
    ):
        raise BenchmarkFailure("invalid_response")

    used_candidates: set[tuple[object, ...]] = set()
    if kind == "pptx":
        while True:
            response = _call(command, ["run", "next", str(run_dir)], cwd=workspace)
            candidate = response.get("candidate")
            if candidate is None:
                break
            if not isinstance(candidate, dict):
                raise BenchmarkFailure("invalid_response")
            plan = _select_candidate_plan(case_id, candidate)
            identity = tuple(
                plan[field]
                for field in (
                    "page_id",
                    "candidate_id",
                    "source_shape_id",
                    "source_object_sha256",
                    "image_sha256",
                )
            )
            if identity in used_candidates:
                raise BenchmarkFailure("stale_plan")
            used_candidates.add(identity)
            _record_candidate(command, plan, run_dir, workspace)

    summary = _call(command, ["run", "execute", str(run_dir)], cwd=workspace)
    sequence = 0
    used_requests: set[tuple[object, ...]] = set()
    while summary.get("status") == "awaiting_agent":
        response = _call(command, ["agent", "next", str(run_dir)], cwd=workspace)
        if response.get("kind") == "capability_handshake":
            if sequence != 0:
                raise BenchmarkFailure("stale_plan")
            sequence += 1
            _record_host_document(
                command,
                _capability_response(response, run_dir),
                run_dir,
                workspace,
                sequence,
            )
            response = _call(
                command, ["agent", "next", str(run_dir)], cwd=workspace
            )
        request = _component_binding(response, run_dir)
        identity = tuple(
            request[field]
            for field in (
                "page_id",
                "repair_round",
                "request_sha256",
                "graph_sha256",
            )
        )
        if identity in used_requests:
            raise BenchmarkFailure("stale_plan")
        used_requests.add(identity)
        plan = _select_component_plan(case_id, request)
        sequence += 1
        _record_component(command, plan, run_dir, workspace, sequence)
        summary = _call(command, ["run", "execute", str(run_dir)], cwd=workspace)
    if summary.get("status") != "completed":
        raise BenchmarkFailure("invalid_response")
    return BenchmarkCaseResult(
        case_id=case_id,
        run_dir=str(run_dir),
        pages=_completed_pages(summary, run_dir),
        duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
    )
