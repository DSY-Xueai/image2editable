import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


_MANIFEST_LIMIT = 64 * 1024
_ASSET_LIMIT = 12 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TOP_FIELDS = {"schema_version", "cases", "routes", "corpus_sha256"}
_CASE_FIELDS = {"id", "kind", "path", "pages", "bytes", "sha256"}
_ROUTE_FIELDS = {"id", "cases", "pages"}
_ROUTE_IDS = ("images", "pdf", "mixed_pptx")
_WINDOWS_RESERVED_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_DEVICE_NAMES = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in ("¹", "²", "³")
}


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Route:
    identifier: str
    sources: tuple[Path, ...]
    expected_pages: int


@dataclass(frozen=True)
class CorpusManifest:
    root: Path
    corpus_sha256: str
    cases: tuple[dict[str, object], ...]
    routes: tuple[Route, ...]


@dataclass(frozen=True)
class RouteExecution:
    route: Route
    run_root: Path
    output_path: Path
    returncode: int
    stdout: str
    duration_seconds: float


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_nlink,
    )


def _read_regular_file(path: Path, limit: int, *, require_single_link: bool) -> bytes:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size > limit
        or (require_single_link and before.st_nlink != 1)
    ):
        raise ValueError
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if _identity(opened) != _identity(before):
            raise ValueError
        payload = handle.read(limit + 1)
    after = path.lstat()
    if len(payload) > limit or _identity(after) != _identity(before):
        raise ValueError
    return payload


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(_: str) -> object:
    raise ValueError


def _strict_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _safe_filename(value: object) -> bool:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        return False
    windows_path = PureWindowsPath(value)
    device_name = value.split(".", 1)[0].casefold()
    return (
        not any(
            character in _WINDOWS_RESERVED_CHARS
            or ord(character) < 32
            or ord(character) == 127
            for character in value
        )
        and not value.endswith((".", " "))
        and device_name not in _WINDOWS_DEVICE_NAMES
        and not PurePosixPath(value).is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and not windows_path.is_reserved()
    )


def _windows_filename_key(value: str) -> str:
    return value.casefold()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_assets(
    root: Path,
    cases: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> None:
    for case in cases:
        source = root / case["path"]
        asset = _read_regular_file(source, _ASSET_LIMIT, require_single_link=True)
        if (
            len(asset) != case["bytes"]
            or hashlib.sha256(asset).hexdigest() != case["sha256"]
        ):
            raise ValueError


def _load_manifest(path: Path) -> CorpusManifest:
    payload = _read_regular_file(path, _MANIFEST_LIMIT, require_single_link=False)
    manifest = json.loads(payload, object_pairs_hook=_object_without_duplicate_keys)
    if not isinstance(manifest, dict) or set(manifest) != _TOP_FIELDS:
        raise ValueError
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError
    cases = manifest["cases"]
    routes = manifest["routes"]
    corpus_sha256 = manifest["corpus_sha256"]
    if not isinstance(cases, list) or not isinstance(routes, list):
        raise ValueError
    if not isinstance(corpus_sha256, str) or not _SHA256_PATTERN.fullmatch(
        corpus_sha256
    ):
        raise ValueError

    canonical_value = {
        "schema_version": manifest["schema_version"],
        "cases": cases,
        "routes": routes,
    }
    if _canonical_sha256(canonical_value) != corpus_sha256:
        raise ValueError

    root = path.resolve(strict=True).parent
    case_by_id: dict[str, dict[str, object]] = {}
    path_keys: set[str] = set()
    declared_bytes = 0
    for case in cases:
        if not isinstance(case, dict) or set(case) != _CASE_FIELDS:
            raise ValueError
        identifier = case["id"]
        kind = case["kind"]
        filename = case["path"]
        pages = case["pages"]
        size = case["bytes"]
        digest = case["sha256"]
        if not isinstance(identifier, str) or not identifier or identifier in case_by_id:
            raise ValueError
        if kind not in {"image", "pdf", "pptx"}:
            raise ValueError
        if not _safe_filename(filename):
            raise ValueError
        filename_key = _windows_filename_key(filename)
        if filename_key in path_keys:
            raise ValueError
        if not _strict_positive_int(pages) or not _strict_positive_int(size):
            raise ValueError
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError
        case_by_id[identifier] = case
        path_keys.add(filename_key)
        declared_bytes += size
    if declared_bytes > _ASSET_LIMIT:
        raise ValueError

    _validate_assets(root, cases)

    if len(routes) != 3:
        raise ValueError
    built_routes = []
    used_case_ids: list[str] = []
    expected_route_shapes = (("image", 8, 8), ("pdf", 1, 3), ("pptx", 1, 3))
    for route, expected_id, shape in zip(routes, _ROUTE_IDS, expected_route_shapes):
        if not isinstance(route, dict) or set(route) != _ROUTE_FIELDS:
            raise ValueError
        route_cases = route["cases"]
        pages = route["pages"]
        if route["id"] != expected_id or not isinstance(route_cases, list):
            raise ValueError
        if not _strict_positive_int(pages):
            raise ValueError
        if any(not isinstance(identifier, str) for identifier in route_cases):
            raise ValueError
        try:
            selected = [case_by_id[identifier] for identifier in route_cases]
        except KeyError as error:
            raise ValueError from error
        expected_kind, expected_count, expected_pages = shape
        if (
            len(selected) != expected_count
            or pages != expected_pages
            or any(case["kind"] != expected_kind for case in selected)
            or sum(case["pages"] for case in selected) != expected_pages
        ):
            raise ValueError
        used_case_ids.extend(route_cases)
        built_routes.append(
            Route(
                identifier=expected_id,
                sources=tuple(root / case["path"] for case in selected),
                expected_pages=pages,
            )
        )
    if len(used_case_ids) != len(set(used_case_ids)) or set(used_case_ids) != set(
        case_by_id
    ):
        raise ValueError

    return CorpusManifest(
        root=root,
        corpus_sha256=corpus_sha256,
        cases=tuple(cases),
        routes=tuple(built_routes),
    )


def load_manifest(path: Path) -> CorpusManifest:
    try:
        return _load_manifest(path)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BenchmarkError("invalid_corpus") from None


def _base_command() -> list[str]:
    return [sys.executable, "-I", "-B", "-m", "image2editable"]


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return environment


def _run_cli(
    arguments: list[str], *, cwd: Path, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_base_command(), *arguments],
        cwd=cwd,
        env=_clean_environment(),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _require_ready_doctor(cwd: Path) -> None:
    try:
        completed = _run_cli(["doctor", "--agent-local"], cwd=cwd, timeout=180.0)
        status = json.loads(
            completed.stdout,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        ready = (
            completed.returncode == 0
            and isinstance(status, dict)
            and set(status) == {"ready", "checks"}
            and status.get("ready") is True
            and isinstance(status.get("checks"), dict)
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        ready = False
    if not ready:
        raise BenchmarkError("doctor_not_ready")


def execute_routes(
    manifest: CorpusManifest, result_root: Path
) -> tuple[RouteExecution, ...]:
    try:
        _validate_assets(manifest.root, manifest.cases)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise BenchmarkError("invalid_corpus") from None

    try:
        result_root.lstat()
    except FileNotFoundError:
        pass
    except OSError:
        raise BenchmarkError("invalid_output") from None
    else:
        raise BenchmarkError("invalid_output")
    absolute_result_root = result_root.resolve()
    try:
        absolute_result_root.mkdir()
    except OSError:
        raise BenchmarkError("invalid_output") from None

    _require_ready_doctor(absolute_result_root)
    executions = []
    for route in manifest.routes:
        run_root = absolute_result_root / route.identifier / "run"
        output_path = absolute_result_root / route.identifier / "output.pptx"
        arguments = [
            "convert",
            *(str(source) for source in route.sources),
            "--run-dir",
            str(run_root),
            "--output",
            str(output_path),
            "--slide-size",
            "16:9",
            "--agent-provider",
            "local",
        ]
        started = time.perf_counter()
        completed = _run_cli(arguments, cwd=absolute_result_root, timeout=None)
        duration = time.perf_counter() - started
        if not math.isfinite(duration) or duration < 0:
            duration = 0.0
        executions.append(
            RouteExecution(
                route=route,
                run_root=run_root,
                output_path=output_path,
                returncode=completed.returncode,
                stdout=completed.stdout,
                duration_seconds=duration,
            )
        )
    return tuple(executions)
