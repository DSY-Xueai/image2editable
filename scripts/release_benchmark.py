from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import posixpath
import re
from statistics import median
import stat
import subprocess
import sys
import tempfile
import time
from typing import Callable
from xml.etree import ElementTree
from zipfile import ZipFile

from scripts.benchmark_conversion import _read_regular_file, _strict_json


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "benchmarks" / "release"
PLAN_ROOT = RELEASE_ROOT / "plans"
RUNTIME_CONSTRAINTS = ROOT / "constraints" / "runtime.txt"
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
    def __init__(
        self, error_type: str, details: dict[str, object] | None = None
    ) -> None:
        super().__init__(error_type)
        self.details = details


@dataclass(frozen=True)
class BenchmarkCaseResult:
    case_id: str
    run_dir: str
    pages: list[dict[str, object]]
    duration_ms: int


@dataclass(frozen=True)
class PlanSelection:
    filename: str
    plan: dict[str, object]
    rebound_plan: dict[str, object] | None


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


def _case_plan_entries(case_id: str) -> list[tuple[str, dict[str, object]]]:
    try:
        root_status = PLAN_ROOT.lstat()
        if (
            not stat.S_ISDIR(root_status.st_mode)
            or _is_reparse(root_status)
        ):
            raise ValueError
        paths = sorted(PLAN_ROOT.glob(f"{case_id}--*.json"))
        plans = [
            (
                path.name,
                _strict_json(
                    _read_regular_file(path, _PLAN_LIMIT, require_single_link=True),
                    _PLAN_LIMIT,
                ),
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


def _case_plans(case_id: str) -> list[dict[str, object]]:
    return [plan for _, plan in _case_plan_entries(case_id)]


def _sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _canonical_manifest_payload(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n")


def canonical_text_sha256(path: Path) -> str:
    payload = _read_regular_file(path, _JSON_LIMIT, require_single_link=True)
    return hashlib.sha256(_canonical_manifest_payload(payload)).hexdigest()


def benchmark_environment() -> dict[str, str]:
    try:
        import torch
    except Exception:
        raise BenchmarkFailure("invalid_environment") from None
    return {
        "os": platform.system(),
        "architecture": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def manifest_sha256(path: Path) -> str:
    return canonical_text_sha256(path)


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


def _resolve_candidate_plan(
    case_id: str,
    candidate: dict[str, object],
    *,
    allow_stale_binding: bool = False,
) -> PlanSelection:
    if allow_stale_binding and (
        not _sha256(candidate.get("source_object_sha256"))
        or not _sha256(candidate.get("image_sha256"))
    ):
        raise BenchmarkFailure("invalid_response")
    entries = [
        (filename, plan)
        for filename, plan in _case_plan_entries(case_id)
        if plan.get("kind") == "candidate_decision"
    ]
    if any(not _valid_candidate_plan(plan) for _, plan in entries):
        raise BenchmarkFailure("invalid_plan")
    if not entries:
        raise BenchmarkFailure("missing_plan")
    identity = [
        (filename, plan)
        for filename, plan in entries
        if all(
            plan.get(field) == candidate.get(field)
            for field in ("page_id", "candidate_id", "source_shape_id")
        )
    ]
    if not identity:
        raise BenchmarkFailure("mismatched_plan")
    if allow_stale_binding and len(identity) != 1:
        raise BenchmarkFailure("duplicate_plan")
    matches = [
        (filename, plan)
        for filename, plan in identity
        if all(
            plan.get(field) == candidate.get(field)
            for field in ("source_object_sha256", "image_sha256")
        )
    ]
    if not matches:
        if allow_stale_binding:
            filename, plan = identity[0]
            rebound = {
                **plan,
                "source_object_sha256": candidate["source_object_sha256"],
                "image_sha256": candidate["image_sha256"],
            }
            return PlanSelection(filename, plan, rebound)
        raise BenchmarkFailure(
            "stale_plan",
            {
                "stage": "candidate_plan",
                "page_id": candidate.get("page_id"),
                "candidate_id": candidate.get("candidate_id"),
                "source_shape_id": candidate.get("source_shape_id"),
                "actual_source_object_sha256": candidate.get(
                    "source_object_sha256"
                ),
                "actual_image_sha256": candidate.get("image_sha256"),
                "expected_bindings": [
                    {
                        "source_object_sha256": plan["source_object_sha256"],
                        "image_sha256": plan["image_sha256"],
                    }
                    for _, plan in identity
                ],
            },
        )
    if len(matches) != 1:
        raise BenchmarkFailure("duplicate_plan")
    filename, plan = matches[0]
    return PlanSelection(filename, plan, None)


def _select_candidate_plan(
    case_id: str, candidate: dict[str, object]
) -> dict[str, object]:
    return _resolve_candidate_plan(case_id, candidate).plan


def _resolve_component_plan(
    case_id: str,
    request: dict[str, object],
    *,
    allow_stale_binding: bool = False,
) -> PlanSelection:
    if allow_stale_binding and (
        not _sha256(request.get("request_sha256"))
        or not _sha256(request.get("graph_sha256"))
    ):
        raise BenchmarkFailure("invalid_response")
    entries = [
        (filename, plan)
        for filename, plan in _case_plan_entries(case_id)
        if plan.get("kind") == "component_plan"
    ]
    if any(not _valid_component_plan(plan) for _, plan in entries):
        raise BenchmarkFailure("invalid_plan")
    if not entries:
        raise BenchmarkFailure("missing_plan")
    identity = [
        (filename, plan)
        for filename, plan in entries
        if plan.get("page_id") == request.get("page_id")
        and plan.get("repair_round") == request.get("repair_round")
    ]
    if not identity:
        raise BenchmarkFailure("mismatched_plan")
    if allow_stale_binding and len(identity) != 1:
        raise BenchmarkFailure("duplicate_plan")
    matches = [
        (filename, plan)
        for filename, plan in identity
        if plan.get("request_sha256") == request.get("request_sha256")
        and plan.get("graph_sha256") == request.get("graph_sha256")
    ]
    if not matches:
        if allow_stale_binding:
            filename, plan = identity[0]
            rebound = {
                **plan,
                "request_sha256": request["request_sha256"],
                "graph_sha256": request["graph_sha256"],
            }
            return PlanSelection(filename, plan, rebound)
        raise BenchmarkFailure(
            "stale_plan",
            {
                "stage": "component_plan",
                "page_id": request.get("page_id"),
                "repair_round": request.get("repair_round"),
                "actual_request_sha256": request.get("request_sha256"),
                "actual_graph_sha256": request.get("graph_sha256"),
                "expected_bindings": [
                    {
                        "request_sha256": plan["request_sha256"],
                        "graph_sha256": plan["graph_sha256"],
                    }
                    for _, plan in identity
                ],
            },
        )
    if len(matches) != 1:
        raise BenchmarkFailure("duplicate_plan")
    filename, plan = matches[0]
    return PlanSelection(filename, plan, None)


def _select_component_plan(
    case_id: str, request: dict[str, object]
) -> dict[str, object]:
    return _resolve_component_plan(case_id, request).plan


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
                if (
                    (metadata.st_dev, metadata.st_ino) == submission_identity
                    and _read_regular_file(
                        submission, _PLAN_LIMIT, require_single_link=True
                    )
                    == payload
                ):
                    submission.unlink()
            except (OSError, ValueError):
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
    allow_stale_bindings: bool = False,
    plan_observer: Callable[[str, dict[str, object], bool], None] | None = None,
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
            selection = _resolve_candidate_plan(
                case_id,
                candidate,
                allow_stale_binding=allow_stale_bindings,
            )
            plan = selection.rebound_plan or selection.plan
            if allow_stale_bindings and plan_observer is not None:
                plan_observer(
                    selection.filename,
                    plan,
                    selection.rebound_plan is not None,
                )
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
        selection = _resolve_component_plan(
            case_id,
            request,
            allow_stale_binding=allow_stale_bindings,
        )
        plan = selection.rebound_plan or selection.plan
        if allow_stale_bindings and plan_observer is not None:
            plan_observer(
                selection.filename,
                plan,
                selection.rebound_plan is not None,
            )
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


def _load_batch_cases(manifest_path: Path) -> tuple[list[dict[str, object]], str]:
    try:
        manifest_payload = _read_regular_file(
            manifest_path, _JSON_LIMIT, require_single_link=True
        )
        manifest = _strict_json(manifest_payload, _JSON_LIMIT)
    except Exception:
        raise BenchmarkFailure("invalid_manifest") from None
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 2
    ):
        raise BenchmarkFailure("invalid_manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkFailure("invalid_manifest")
    identifiers: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise BenchmarkFailure("invalid_manifest")
        identifier = case.get("id")
        pages = case.get("expected_pages")
        source = case.get("path")
        if (
            not isinstance(identifier, str)
            or identifier in identifiers
            or _IDENTIFIER.fullmatch(identifier) is None
            or not isinstance(source, str)
            or not isinstance(pages, list)
            or type(case.get("page_count")) is not int
            or case["page_count"] != len(pages)
        ):
            raise BenchmarkFailure("invalid_manifest")
        identifiers.add(identifier)
        try:
            source_path = (RELEASE_ROOT / source).resolve()
            if not source_path.is_relative_to(RELEASE_ROOT.resolve()):
                raise ValueError
            source_payload = _read_regular_file(
                source_path, _JSON_LIMIT * 16, require_single_link=True
            )
        except Exception:
            raise BenchmarkFailure("invalid_manifest") from None
        digest = case.get("sha256")
        if (
            not isinstance(digest, str)
            or hashlib.sha256(source_payload).hexdigest() != digest
        ):
            raise BenchmarkFailure("invalid_manifest")
        for page in pages:
            if (
                not isinstance(page, dict)
                or not isinstance(page.get("page_id"), str)
                or page.get("expected_status")
                not in {"validated", "replaced", "preserved"}
                or type(page.get("min_visual_components")) is not int
                or page["min_visual_components"] < 0
                or type(page.get("min_text_boxes")) is not int
                or page["min_text_boxes"] < 0
                or type(page.get("max_unexplained_pixels")) is not int
                or page["max_unexplained_pixels"] < 0
                or type(page.get("max_quality_violations")) is not int
                or page["max_quality_violations"] < 0
            ):
                raise BenchmarkFailure("invalid_manifest")
    return [dict(case) for case in cases], hashlib.sha256(
        _canonical_manifest_payload(manifest_payload)
    ).hexdigest()


def _validate_preserved_pptx_page(
    case: dict[str, object], result: BenchmarkCaseResult, page: dict[str, object], page_number: int
) -> None:
    source_value = case.get("path")
    if case.get("kind") != "pptx" or not isinstance(source_value, str):
        raise BenchmarkFailure("invalid_page_result")
    source = (RELEASE_ROOT / source_value).resolve()
    output = Path(result.run_dir).resolve().parent / "output.pptx"
    slide_part = f"ppt/slides/slide{page_number}.xml"
    relationships_part = f"ppt/slides/_rels/slide{page_number}.xml.rels"
    try:
        with ZipFile(source) as source_archive, ZipFile(output) as output_archive:
            source_slide = source_archive.read(slide_part)
            if source_slide != output_archive.read(slide_part):
                raise BenchmarkFailure("quality_gate")
            source_relationships = source_archive.read(relationships_part)
            if source_relationships != output_archive.read(relationships_part):
                raise BenchmarkFailure("quality_gate")
            relationships = ElementTree.fromstring(source_relationships)
            for relationship in relationships:
                if relationship.get("TargetMode") == "External":
                    continue
                target = relationship.get("Target")
                if not isinstance(target, str) or not target:
                    raise ValueError
                target_part = posixpath.normpath(
                    posixpath.join(posixpath.dirname(slide_part), target)
                )
                if target_part.startswith("../") or target_part.startswith("/"):
                    raise ValueError
                if source_archive.read(target_part) != output_archive.read(target_part):
                    raise BenchmarkFailure("quality_gate")
    except BenchmarkFailure:
        raise
    except Exception:
        raise BenchmarkFailure("invalid_quality_result") from None

    try:
        slide = ElementTree.fromstring(source_slide)
        namespace = {
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        }
        shape_tree = slide.find(".//p:spTree", namespace)
        if shape_tree is None:
            raise ValueError
        text_boxes = sum(
            1
            for shape in shape_tree.findall("p:sp", namespace)
            if any((node.text or "").strip() for node in shape.findall(".//a:t", namespace))
        )
        visual_components = sum(
            len(shape_tree.findall(f"p:{name}", namespace))
            for name in ("pic", "graphicFrame", "cxnSp", "grpSp", "contentPart")
        )
    except Exception:
        raise BenchmarkFailure("invalid_quality_result") from None
    if (
        visual_components < page.get("min_visual_components", 0)
        or text_boxes < page.get("min_text_boxes", 0)
    ):
        raise BenchmarkFailure("quality_gate")


def _validate_batch_case(
    case: dict[str, object], result: BenchmarkCaseResult
) -> None:
    expected = case["expected_pages"]
    actual = {page.get("page_id"): page for page in result.pages}
    if len(actual) != len(expected):
        raise BenchmarkFailure("invalid_page_result")
    for page_number, page in enumerate(expected, start=1):
        page_id = f"page_{page_number:03d}"
        observed = actual.get(page_id)
        if not isinstance(observed, dict):
            raise BenchmarkFailure("invalid_page_result")
        expected_status = page.get("expected_status")
        observed_status = observed.get("status")
        if observed_status != expected_status:
            raise BenchmarkFailure("invalid_page_result")
        if observed_status == "preserved":
            _validate_preserved_pptx_page(case, result, page, page_number)
            continue
        if observed_status not in {"validated", "replaced"}:
            raise BenchmarkFailure("invalid_page_result")
        component_result_path = (
            Path(result.run_dir)
            / "pages"
            / page_id
            / "reconstruction"
            / "component_result.json"
        )
        component_result = None
        if component_result_path.is_file():
            try:
                component_result = _strict_json(
                    _read_regular_file(
                        component_result_path,
                        _JSON_LIMIT,
                        require_single_link=True,
                    ),
                    _JSON_LIMIT,
                )
            except Exception:
                raise BenchmarkFailure("invalid_quality_result") from None
        if not isinstance(component_result, dict):
            raise BenchmarkFailure("invalid_quality_result")
        if "warning" not in component_result:
            raise BenchmarkFailure("invalid_quality_result")
        if component_result["warning"] is not None:
            raise BenchmarkFailure("warning_fallback")
        fallback = component_result.get("fallback")
        if not isinstance(fallback, dict) or set(fallback) != {"status", "parent_ids"}:
            raise BenchmarkFailure("invalid_quality_result")
        if fallback["status"] != "none":
            raise BenchmarkFailure("warning_fallback")
        if fallback["parent_ids"] != []:
            raise BenchmarkFailure("invalid_quality_result")

        minimum_visual = page.get("min_visual_components")
        final_components = component_result.get("final_component_ids")
        if (
            not isinstance(final_components, list)
            or any(
                not isinstance(identifier, str) or not identifier
                for identifier in final_components
            )
            or len(set(final_components)) != len(final_components)
        ):
            raise BenchmarkFailure("invalid_quality_result")
        if type(minimum_visual) is int and len(final_components) < minimum_visual:
            raise BenchmarkFailure("quality_gate")

        minimum_text = page.get("min_text_boxes")
        text_items = component_result.get("text_items")
        if (
            not isinstance(text_items, list)
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("_component_id"), str)
                or not item["_component_id"]
                for item in text_items
            )
            or len({item["_component_id"] for item in text_items}) != len(text_items)
        ):
            raise BenchmarkFailure("invalid_quality_result")
        if type(minimum_text) is int and len(text_items) < minimum_text:
            raise BenchmarkFailure("quality_gate")

        repair_rounds = component_result.get("repair_rounds")
        accepted_graph = component_result.get("accepted_graph_sha256")
        if (
            type(repair_rounds) is not int
            or repair_rounds < 1
            or not _sha256(accepted_graph)
        ):
            raise BenchmarkFailure("invalid_quality_result")
        quality_path = (
            Path(result.run_dir)
            / "pages"
            / page_id
            / "reconstruction"
            / f"execution-{repair_rounds:02d}"
            / "component-quality.json"
        )
        try:
            quality = _strict_json(
                _read_regular_file(
                    quality_path,
                    _JSON_LIMIT,
                    require_single_link=True,
                ),
                _JSON_LIMIT,
            )
            if (
                not isinstance(quality, dict)
                or quality.get("page_id") != page_id
                or quality.get("repair_round") != repair_rounds
                or quality.get("input_graph_sha256") != accepted_graph
            ):
                raise ValueError
            report = quality.get("report")
            if not isinstance(report, dict):
                raise ValueError
            metrics = report.get("visual_metrics")
            violations = report.get("violations")
            component_reports = report.get("component_reports")
            if (
                not isinstance(metrics, dict)
                or not isinstance(violations, list)
                or any(not isinstance(value, str) for value in violations)
                or not isinstance(component_reports, list)
                or any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("violations"), list)
                    or any(
                        not isinstance(value, str) for value in item["violations"]
                    )
                    for item in component_reports
                )
            ):
                raise ValueError
            unexplained = metrics.get("unexplained_visual_pixels")
            if type(unexplained) is not int or unexplained < 0:
                raise ValueError
            active_violations = [
                value for value in violations if value != "pptx_reopen_unknown"
            ]
            component_violations = sum(
                len(item["violations"]) for item in component_reports
            )
        except Exception:
            raise BenchmarkFailure("invalid_quality_result") from None
        if (
            unexplained > page.get("max_unexplained_pixels", 0)
            or active_violations
            or component_violations > page.get("max_quality_violations", 0)
        ):
            raise BenchmarkFailure("quality_gate")


def aggregate_performance(
    attempts: list[dict[str, object]],
) -> dict[str, object]:
    if not isinstance(attempts, list) or not attempts:
        raise BenchmarkFailure("invalid_performance_result")
    case_durations: dict[str, dict[int, int]] = {}
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise BenchmarkFailure("invalid_performance_result")
        repeat = attempt.get("repeat")
        duration = attempt.get("duration_ms")
        case_id = attempt.get("case_id")
        if (
            attempt.get("status") != "passed"
            or not isinstance(case_id, str)
            or _IDENTIFIER.fullmatch(case_id) is None
            or type(repeat) is not int
            or repeat not in (1, 2, 3)
            or type(duration) is not int
            or duration < 0
        ):
            raise BenchmarkFailure("invalid_performance_result")
        durations = case_durations.setdefault(case_id, {})
        if repeat in durations:
            raise BenchmarkFailure("invalid_performance_result")
        durations[repeat] = duration
    if any(set(durations) != {1, 2, 3} for durations in case_durations.values()):
        raise BenchmarkFailure("invalid_performance_result")
    repeat_totals = [
        sum(durations[repeat] for durations in case_durations.values())
        for repeat in (1, 2, 3)
    ]
    return {
        "repeat_total_duration_ms": repeat_totals,
        "median_total_duration_ms": int(median(repeat_totals)),
        "case_median_duration_ms": {
            case_id: int(median(durations.values()))
            for case_id, durations in case_durations.items()
        },
    }


def compare_baseline(
    report: dict[str, object],
    baseline: dict[str, object],
    *,
    constraints_sha256: str,
    environment: object,
) -> dict[str, object]:
    baseline_fields = {
        "schema_version",
        "benchmark",
        "manifest_sha256",
        "constraints_sha256",
        "environment",
        "median_total_duration_ms",
        "case_median_duration_ms",
    }
    environment_fields = {"os", "architecture", "python", "device"}
    baseline_environment = baseline.get("environment")
    baseline_cases = baseline.get("case_median_duration_ms")
    baseline_total = baseline.get("median_total_duration_ms")
    if (
        set(baseline) != baseline_fields
        or baseline.get("schema_version") != 1
        or baseline.get("benchmark") != "v0.2-core-14-page"
        or not _sha256(baseline.get("manifest_sha256"))
        or not _sha256(baseline.get("constraints_sha256"))
        or not isinstance(baseline_environment, dict)
        or set(baseline_environment) != environment_fields
        or not all(
            isinstance(baseline_environment[field], str)
            and bool(baseline_environment[field])
            for field in environment_fields
        )
        or type(baseline_total) is not int
        or baseline_total < 0
        or not isinstance(baseline_cases, dict)
        or not baseline_cases
        or not all(
            isinstance(case_id, str)
            and _IDENTIFIER.fullmatch(case_id) is not None
            and type(value) is int
            and value >= 0
            for case_id, value in baseline_cases.items()
        )
    ):
        raise BenchmarkFailure("invalid_baseline")

    performance = report.get("performance")
    totals = report.get("totals")
    if (
        report.get("status") != "passed"
        or report.get("repeat") != 3
        or not _sha256(report.get("manifest_sha256"))
        or not isinstance(totals, dict)
        or totals.get("failed_attempts") != 0
        or not isinstance(performance, dict)
        or set(performance)
        != {
            "repeat_total_duration_ms",
            "median_total_duration_ms",
            "case_median_duration_ms",
        }
        or type(performance.get("median_total_duration_ms")) is not int
        or performance["median_total_duration_ms"] < 0
        or not isinstance(performance.get("case_median_duration_ms"), dict)
        or set(performance["case_median_duration_ms"]) != set(baseline_cases)
    ):
        raise BenchmarkFailure("invalid_performance_result")

    reasons = []
    if report["manifest_sha256"] != baseline["manifest_sha256"]:
        reasons.append("manifest_sha256")
    if constraints_sha256 != baseline["constraints_sha256"]:
        reasons.append("constraints_sha256")
    if environment != baseline_environment:
        reasons.append("environment")
    if reasons:
        return {"status": "not_comparable", "reasons": reasons}

    current = performance["median_total_duration_ms"]
    limit = baseline_total * 115 // 100
    return {
        "status": "regressed" if current > limit else "passed",
        "median_total_duration_ms": current,
        "limit_ms": limit,
    }


def _select_manifest_cases(
    cases: list[dict[str, object]], case_ids: list[str]
) -> list[dict[str, object]]:
    if (
        not case_ids
        or any(
            not isinstance(case_id, str)
            or _IDENTIFIER.fullmatch(case_id) is None
            for case_id in case_ids
        )
        or len(set(case_ids)) != len(case_ids)
    ):
        raise BenchmarkFailure("invalid_case_selection")
    requested = set(case_ids)
    selected = [case for case in cases if case["id"] in requested]
    if len(selected) != len(requested):
        raise BenchmarkFailure("invalid_case_selection")
    return selected


def _write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode(
        "utf-8"
    ) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def run_shard_manifest(
    manifest_path: Path,
    *,
    case_ids: list[str],
    workspace: Path,
    report_path: Path,
    constraints_path: Path = RUNTIME_CONSTRAINTS,
    repeat: int = 3,
    case_runner: Callable[..., BenchmarkCaseResult] = run_case,
    command: Command = run_command,
) -> dict[str, object]:
    if type(repeat) is not int or repeat != 3:
        raise BenchmarkFailure("invalid_repeat")
    cases, manifest_sha256 = _load_batch_cases(manifest_path)
    selected = _select_manifest_cases(cases, case_ids)
    constraints_hash = canonical_text_sha256(constraints_path)
    environment = benchmark_environment()
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, object]] = []
    for index in range(1, repeat + 1):
        repeat_workspace = workspace / f"repeat-{index:02d}"
        repeat_workspace.mkdir()
        for case in selected:
            started = time.perf_counter()
            try:
                result = case_runner(
                    case, workspace=repeat_workspace, command=command
                )
                _validate_batch_case(case, result)
                attempt = {
                    "case_id": case["id"],
                    "repeat": index,
                    "status": "passed",
                    "duration_ms": result.duration_ms,
                    "pages": result.pages,
                }
            except BenchmarkFailure as error:
                attempt = {
                    "case_id": case["id"],
                    "repeat": index,
                    "status": "failed",
                    "error_type": str(error),
                    "duration_ms": max(
                        0, int((time.perf_counter() - started) * 1000)
                    ),
                }
                if error.details is not None:
                    attempt["diagnostics"] = error.details
            except Exception:
                attempt = {
                    "case_id": case["id"],
                    "repeat": index,
                    "status": "failed",
                    "error_type": "conversion_failed",
                    "duration_ms": max(
                        0, int((time.perf_counter() - started) * 1000)
                    ),
                }
            attempts.append(attempt)

    failed = sum(attempt["status"] != "passed" for attempt in attempts)
    report = {
        "schema_version": 1,
        "report_kind": "shard",
        "status": "passed" if failed == 0 else "failed",
        "manifest_sha256": manifest_sha256,
        "constraints_sha256": constraints_hash,
        "environment": environment,
        "repeat": repeat,
        "selected_case_ids": sorted(case["id"] for case in selected),
        "attempts": attempts,
        "totals": {
            "cases": len(selected),
            "attempts": len(attempts),
            "pages": sum(case["page_count"] for case in selected) * repeat,
            "failed_attempts": failed,
        },
    }
    if failed == 0:
        report["performance"] = aggregate_performance(attempts)
    _write_json_exclusive(report_path, report)
    return report


def run_diagnostic_manifest(
    manifest_path: Path,
    *,
    case_ids: list[str],
    workspace: Path,
    report_path: Path,
    plans_output: Path,
    constraints_path: Path = RUNTIME_CONSTRAINTS,
    repeat: int = 3,
    case_runner: Callable[..., BenchmarkCaseResult] = run_case,
    command: Command = run_command,
) -> dict[str, object]:
    if type(repeat) is not int or repeat != 3:
        raise BenchmarkFailure("invalid_repeat")
    plans_output = plans_output.resolve()
    if plans_output.exists():
        raise BenchmarkFailure("invalid_workspace")
    cases, manifest_sha256 = _load_batch_cases(manifest_path)
    selected = _select_manifest_cases(cases, case_ids)
    constraints_hash = canonical_text_sha256(constraints_path)
    environment = benchmark_environment()
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    observed_plans: dict[str, dict[str, object]] = {}
    rebound_plans: dict[str, dict[str, object]] = {}

    def observe_plan(
        filename: str, plan: dict[str, object], rebound: bool = True
    ) -> None:
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise BenchmarkFailure("invalid_plan")
        existing = observed_plans.get(filename)
        if existing is None:
            observed_plans[filename] = plan
        elif existing != plan:
            raise BenchmarkFailure("unstable_plan_binding")
        if rebound:
            rebound_plans[filename] = plan

    attempts: list[dict[str, object]] = []
    for index in range(1, repeat + 1):
        repeat_workspace = workspace / f"repeat-{index:02d}"
        repeat_workspace.mkdir()
        for case in selected:
            started = time.perf_counter()
            try:
                result = case_runner(
                    case,
                    workspace=repeat_workspace,
                    command=command,
                    allow_stale_bindings=True,
                    plan_observer=observe_plan,
                )
                _validate_batch_case(case, result)
                attempt = {
                    "case_id": case["id"],
                    "repeat": index,
                    "status": "passed",
                    "duration_ms": result.duration_ms,
                    "pages": result.pages,
                }
            except BenchmarkFailure as error:
                attempt = {
                    "case_id": case["id"],
                    "repeat": index,
                    "status": "failed",
                    "error_type": str(error),
                    "duration_ms": max(
                        0, int((time.perf_counter() - started) * 1000)
                    ),
                }
                if error.details is not None:
                    attempt["diagnostics"] = error.details
            except Exception:
                attempt = {
                    "case_id": case["id"],
                    "repeat": index,
                    "status": "failed",
                    "error_type": "conversion_failed",
                    "duration_ms": max(
                        0, int((time.perf_counter() - started) * 1000)
                    ),
                }
            attempts.append(attempt)

    failed = sum(attempt["status"] != "passed" for attempt in attempts)
    report = {
        "schema_version": 1,
        "report_kind": "diagnostic",
        "status": "diagnostic_complete" if failed == 0 else "failed",
        "manifest_sha256": manifest_sha256,
        "constraints_sha256": constraints_hash,
        "environment": environment,
        "repeat": repeat,
        "selected_case_ids": [case["id"] for case in selected],
        "attempts": attempts,
        "totals": {
            "cases": len(selected),
            "attempts": len(attempts),
            "pages": sum(case["page_count"] for case in selected) * repeat,
            "failed_attempts": failed,
        },
    }
    if failed == 0:
        report["performance"] = aggregate_performance(attempts)

    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode(
        "utf-8"
    ) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(report_path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)

    if failed == 0:
        try:
            plans_output.mkdir()
            for filename, plan in sorted(rebound_plans.items()):
                plan_payload = json.dumps(
                    plan, ensure_ascii=False, sort_keys=True, indent=2
                ).encode("utf-8") + b"\n"
                descriptor = os.open(plans_output / filename, flags, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(plan_payload)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise BenchmarkFailure("invalid_workspace") from None
    return report


def _load_report(path: Path, error_type: str) -> dict[str, object]:
    try:
        report = _strict_json(
            _read_regular_file(path, _JSON_LIMIT, require_single_link=True),
            _JSON_LIMIT,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BenchmarkFailure(error_type) from None
    if not isinstance(report, dict):
        raise BenchmarkFailure(error_type)
    return report


def _validated_shard_attempts(
    report: dict[str, object],
    *,
    cases: dict[str, dict[str, object]],
    manifest_hash: str,
    constraints_hash: str,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    fields = {
        "schema_version",
        "report_kind",
        "status",
        "manifest_sha256",
        "constraints_sha256",
        "environment",
        "repeat",
        "selected_case_ids",
        "attempts",
        "totals",
        "performance",
    }
    environment_fields = {"os", "architecture", "python", "device"}
    environment = report.get("environment")
    selected = report.get("selected_case_ids")
    attempts = report.get("attempts")
    totals = report.get("totals")
    if (
        set(report) != fields
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("report_kind") != "shard"
        or report.get("status") != "passed"
        or report.get("manifest_sha256") != manifest_hash
        or report.get("constraints_sha256") != constraints_hash
        or report.get("repeat") != 3
        or not isinstance(environment, dict)
        or set(environment) != environment_fields
        or not all(
            isinstance(environment[field], str) and bool(environment[field])
            for field in environment_fields
        )
        or not isinstance(selected, list)
        or not selected
        or selected != sorted(selected)
        or len(set(selected)) != len(selected)
        or any(case_id not in cases for case_id in selected)
        or not isinstance(attempts, list)
        or not isinstance(totals, dict)
    ):
        raise BenchmarkFailure("invalid_shard_report")

    expected_attempts = {
        (case_id, repeat) for case_id in selected for repeat in (1, 2, 3)
    }
    observed_attempts: set[tuple[str, int]] = set()
    for attempt in attempts:
        if not isinstance(attempt, dict) or set(attempt) != {
            "case_id",
            "repeat",
            "status",
            "duration_ms",
            "pages",
        }:
            raise BenchmarkFailure("invalid_shard_report")
        case_id = attempt.get("case_id")
        repeat = attempt.get("repeat")
        duration = attempt.get("duration_ms")
        pages = attempt.get("pages")
        if (
            case_id not in selected
            or type(repeat) is not int
            or repeat not in (1, 2, 3)
            or attempt.get("status") != "passed"
            or type(duration) is not int
            or duration < 0
            or not isinstance(pages, list)
        ):
            raise BenchmarkFailure("invalid_shard_report")
        identity = (case_id, repeat)
        if identity in observed_attempts:
            raise BenchmarkFailure("invalid_shard_report")
        observed_attempts.add(identity)
        expected_pages = cases[case_id]["expected_pages"]
        if len(pages) != len(expected_pages):
            raise BenchmarkFailure("invalid_shard_report")
        for page_number, (page, expected) in enumerate(
            zip(pages, expected_pages), start=1
        ):
            if (
                not isinstance(page, dict)
                or page.get("page_id") != f"page_{page_number:03d}"
                or page.get("status") != expected["expected_status"]
            ):
                raise BenchmarkFailure("invalid_shard_report")
    expected_totals = {
        "cases": len(selected),
        "attempts": len(expected_attempts),
        "pages": sum(cases[case_id]["page_count"] for case_id in selected) * 3,
        "failed_attempts": 0,
    }
    if (
        observed_attempts != expected_attempts
        or totals != expected_totals
        or report.get("performance") != aggregate_performance(attempts)
    ):
        raise BenchmarkFailure("invalid_shard_report")
    return attempts, environment


def aggregate_shard_reports(
    manifest_path: Path,
    *,
    shard_report_paths: list[Path],
    constraints_path: Path,
    baseline_path: Path,
    report_path: Path,
) -> dict[str, object]:
    cases_list, manifest_hash = _load_batch_cases(manifest_path)
    if (
        len(cases_list) != 10
        or sum(case["page_count"] for case in cases_list) != 14
        or len(shard_report_paths) != 5
    ):
        raise BenchmarkFailure("invalid_shard_report")
    cases = {case["id"]: case for case in cases_list}
    constraints_hash = canonical_text_sha256(constraints_path)
    all_attempts: list[dict[str, object]] = []
    covered: set[str] = set()
    shared_environment: dict[str, str] | None = None
    for path in shard_report_paths:
        shard = _load_report(path, "invalid_shard_report")
        attempts, environment = _validated_shard_attempts(
            shard,
            cases=cases,
            manifest_hash=manifest_hash,
            constraints_hash=constraints_hash,
        )
        selected = set(shard["selected_case_ids"])
        if covered & selected:
            raise BenchmarkFailure("invalid_shard_report")
        covered.update(selected)
        if shared_environment is None:
            shared_environment = environment
        elif environment != shared_environment:
            raise BenchmarkFailure("invalid_shard_report")
        all_attempts.extend(attempts)
    if covered != set(cases) or shared_environment is None:
        raise BenchmarkFailure("invalid_shard_report")

    case_order = {case["id"]: index for index, case in enumerate(cases_list)}
    all_attempts.sort(key=lambda item: (item["repeat"], case_order[item["case_id"]]))
    performance = aggregate_performance(all_attempts)
    report = {
        "schema_version": 1,
        "report_kind": "official",
        "status": "passed",
        "manifest_sha256": manifest_hash,
        "constraints_sha256": constraints_hash,
        "environment": shared_environment,
        "repeat": 3,
        "attempts": all_attempts,
        "totals": {
            "cases": 10,
            "attempts": 30,
            "pages": 42,
            "failed_attempts": 0,
        },
        "performance": performance,
    }
    baseline = _load_report(baseline_path, "invalid_baseline")
    comparison = compare_baseline(
        report,
        baseline,
        constraints_sha256=constraints_hash,
        environment=shared_environment,
    )
    if comparison.get("status") != "passed":
        raise BenchmarkFailure("performance_gate")
    report["performance_comparison"] = comparison
    _write_json_exclusive(report_path, report)
    return report


def run_manifest(
    manifest_path: Path,
    *,
    workspace: Path,
    report_path: Path,
    repeat: int = 3,
    case_runner: Callable[..., BenchmarkCaseResult] = run_case,
    command: Command = run_command,
) -> dict[str, object]:
    if type(repeat) is not int or repeat != 3:
        raise BenchmarkFailure("invalid_repeat")
    cases, manifest_sha256 = _load_batch_cases(manifest_path)
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, object]] = []
    for index in range(1, repeat + 1):
        repeat_workspace = workspace / f"repeat-{index:02d}"
        repeat_workspace.mkdir()
        for case in cases:
            started = time.perf_counter()
            try:
                result = case_runner(
                    case, workspace=repeat_workspace, command=command
                )
                _validate_batch_case(case, result)
                attempt = {
                    "case_id": case["id"],
                    "repeat": index,
                    "status": "passed",
                    "duration_ms": result.duration_ms,
                    "pages": result.pages,
                }
            except BenchmarkFailure as error:
                attempt = {
                    "case_id": case["id"],
                    "repeat": index,
                    "status": "failed",
                    "error_type": str(error),
                    "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
                }
                if error.details is not None:
                    attempt["diagnostics"] = error.details
            except Exception:
                attempt = {
                    "case_id": case["id"],
                    "repeat": index,
                    "status": "failed",
                    "error_type": "conversion_failed",
                    "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
                }
            attempts.append(attempt)
    failed = sum(attempt["status"] != "passed" for attempt in attempts)
    report = {
        "schema_version": 1,
        "status": "passed" if failed == 0 else "failed",
        "manifest_sha256": manifest_sha256,
        "repeat": repeat,
        "attempts": attempts,
        "totals": {
            "cases": len(cases),
            "pages": sum(case["page_count"] for case in cases) * repeat,
            "failed_attempts": failed,
        },
    }
    if failed == 0:
        report["performance"] = aggregate_performance(attempts)
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode(
        "utf-8"
    ) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(report_path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("official-shard", "diagnostic-shard", "aggregate"),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--plans-output", type=Path)
    parser.add_argument("--shard-report", action="append", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--constraints", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.mode == "official-shard":
        if (
            arguments.workspace is None
            or not arguments.case_ids
            or arguments.plans_output is not None
            or arguments.shard_report is not None
            or arguments.baseline is not None
            or arguments.constraints is not None
        ):
            parser.error("invalid official-shard arguments")
    elif arguments.mode == "diagnostic-shard":
        if (
            arguments.workspace is None
            or not arguments.case_ids
            or arguments.plans_output is None
            or arguments.shard_report is not None
            or arguments.baseline is not None
            or arguments.constraints is not None
        ):
            parser.error("invalid diagnostic-shard arguments")
    elif (
        arguments.workspace is not None
        or arguments.case_ids is not None
        or arguments.plans_output is not None
        or not arguments.shard_report
        or arguments.baseline is None
        or arguments.constraints is None
    ):
        parser.error("invalid aggregate arguments")
    try:
        if arguments.mode == "official-shard":
            report = run_shard_manifest(
                arguments.manifest,
                case_ids=arguments.case_ids,
                workspace=arguments.workspace,
                report_path=arguments.report,
                repeat=3,
            )
            success_status = "passed"
        elif arguments.mode == "diagnostic-shard":
            report = run_diagnostic_manifest(
                arguments.manifest,
                case_ids=arguments.case_ids,
                workspace=arguments.workspace,
                report_path=arguments.report,
                plans_output=arguments.plans_output,
                repeat=3,
            )
            success_status = "diagnostic_complete"
        else:
            report = aggregate_shard_reports(
                arguments.manifest,
                shard_report_paths=arguments.shard_report,
                constraints_path=arguments.constraints,
                baseline_path=arguments.baseline,
                report_path=arguments.report,
            )
            success_status = "passed"
    except BenchmarkFailure as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0 if report["status"] == success_status else 1


if __name__ == "__main__":
    raise SystemExit(main())
