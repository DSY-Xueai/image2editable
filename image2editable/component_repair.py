from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import uuid
from typing import BinaryIO

from image2editable.component_contracts import (
    COMPONENT_EVIDENCE_NAMES,
    validate_agent_provider,
    validate_component_agent_request,
    validate_component_graph,
    validate_repair_round,
)


EVIDENCE_NAMES = tuple(sorted(COMPONENT_EVIDENCE_NAMES))
REQUEST_NAME = "component_agent_request.json"
MARKER_SUFFIX = ".request.json"


def build_component_agent_request(
    page_session: dict,
    *,
    repair_round: int,
) -> Path:
    repair_round = validate_repair_round(repair_round)
    page_id, provider, reconstruction, sources = _validate_page_session(page_session)
    agent_dir = reconstruction / "agent"
    _ensure_owned_directory(agent_dir, reconstruction)
    round_dir = agent_dir / f"round-{repair_round:02d}"
    if round_dir.exists() or round_dir.is_symlink():
        raise RuntimeError(f"Agent evidence round is already published: {round_dir}")
    staging = agent_dir / f".{round_dir.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        records: dict[str, dict[str, str]] = {}
        payloads: dict[str, bytes] = {}
        for name in EVIDENCE_NAMES:
            source = _contained_path(Path(sources[name]), reconstruction)
            payload = _read_bound_file(source, reconstruction)
            payloads[name] = payload
            records[name] = {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        graph = json.loads(payloads["component-graph.json"].decode("utf-8"))
        validate_component_graph(graph)
        request = {
            "schema_version": 1,
            "page_id": page_id,
            "provider": provider,
            "repair_round": repair_round,
            "source_sha256": records["source.png"]["sha256"],
            "graph_sha256": records["component-graph.json"]["sha256"],
            "candidate_ids": sorted(
                node["id"] for node in graph["nodes"] if node["state"] == "pending"
            ),
            "frozen_ids": sorted(
                node["id"] for node in graph["nodes"] if node["state"] == "frozen"
            ),
            "evidence": records,
        }
        validate_component_agent_request(request)
        for name, payload in payloads.items():
            _write_exclusive(staging / name, payload, reconstruction)
        request_bytes = json.dumps(
            request,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        _write_exclusive(staging / REQUEST_NAME, request_bytes, reconstruction)
        agent_identity = _snapshot_directory_chain(agent_dir, reconstruction)
        try:
            staging.rename(round_dir)
        except OSError as error:
            raise RuntimeError(
                f"Agent evidence round is already published: {round_dir}"
            ) from error
        _require_directory_chain_identity(agent_identity)
        marker = {
            "schema_version": 1,
            "page_id": page_id,
            "provider": provider,
            "repair_round": repair_round,
            "request_path": f"{round_dir.name}/{REQUEST_NAME}",
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        }
        marker_bytes = json.dumps(
            marker,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        _write_exclusive(
            agent_dir / f"{round_dir.name}{MARKER_SUFFIX}",
            marker_bytes,
            reconstruction,
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return round_dir / REQUEST_NAME


def load_component_agent_request(request_path: str | Path) -> dict:
    request_path = Path(request_path)
    if request_path.name != REQUEST_NAME:
        raise RuntimeError("Component agent request path is invalid")
    round_dir = request_path.parent
    agent_dir = round_dir.parent
    reconstruction = agent_dir.parent
    if (
        reconstruction.name != "reconstruction"
        or reconstruction.parent.parent.name != "pages"
        or agent_dir.name != "agent"
    ):
        raise RuntimeError(
            "Component agent request path must be pages/<page_id>/"
            "reconstruction/agent/round-XX"
        )
    _validate_directory_chain(round_dir, reconstruction)
    marker_path = agent_dir / f"{round_dir.name}{MARKER_SUFFIX}"
    try:
        marker = json.loads(_read_bound_file(marker_path, reconstruction).decode("utf-8"))
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Component agent request marker is missing or invalid") from error
    _validate_request_marker(marker)
    request_bytes = _read_bound_file(request_path, reconstruction)
    if hashlib.sha256(request_bytes).hexdigest() != marker["request_sha256"]:
        raise RuntimeError("Component agent request hash mismatch")
    request = json.loads(request_bytes.decode("utf-8"))
    validate_component_agent_request(request)
    expected_round = f"round-{request['repair_round']:02d}"
    if (
        round_dir.name != expected_round
        or reconstruction.parent.name != request["page_id"]
        or marker["page_id"] != request["page_id"]
        or marker["provider"] != request["provider"]
        or marker["repair_round"] != request["repair_round"]
        or marker["request_path"] != f"{round_dir.name}/{REQUEST_NAME}"
    ):
        raise RuntimeError("Component agent request belongs to another page or round")
    graph_payload = b""
    for name, record in request["evidence"].items():
        evidence_path = _resolve_evidence_path(record["path"], round_dir, reconstruction)
        payload = _read_bound_file(evidence_path, reconstruction)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record["sha256"]:
            raise RuntimeError(f"Component evidence hash mismatch: {name}")
        if name == "component-graph.json":
            graph_payload = payload
    if request["source_sha256"] != request["evidence"]["source.png"]["sha256"]:
        raise RuntimeError("Component source evidence hash mismatch")
    if request["graph_sha256"] != request["evidence"]["component-graph.json"]["sha256"]:
        raise RuntimeError("Component graph evidence hash mismatch")
    graph = json.loads(graph_payload.decode("utf-8"))
    validate_component_graph(graph)
    candidate_ids = sorted(
        node["id"] for node in graph["nodes"] if node["state"] == "pending"
    )
    frozen_ids = sorted(
        node["id"] for node in graph["nodes"] if node["state"] == "frozen"
    )
    if request["candidate_ids"] != candidate_ids or request["frozen_ids"] != frozen_ids:
        raise RuntimeError("Component agent request component ids do not match graph")
    return request


def _validate_page_session(session: object) -> tuple[str, str, Path, dict]:
    fields = {"page_id", "provider", "reconstruction_dir", "evidence"}
    if not isinstance(session, dict) or set(session) != fields:
        raise ValueError("page_session fields are invalid")
    page_id = session["page_id"]
    if type(page_id) is not str or not page_id or session["reconstruction_dir"] is None:
        raise ValueError("page_session page_id is invalid")
    provider = validate_agent_provider(session["provider"])
    reconstruction = Path(session["reconstruction_dir"])
    if (
        reconstruction.name != "reconstruction"
        or reconstruction.parent.name != page_id
        or reconstruction.parent.parent.name != "pages"
    ):
        raise RuntimeError(
            "page_session path must be pages/<page_id>/reconstruction"
        )
    _validate_directory_chain(reconstruction, reconstruction)
    evidence = session["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != COMPONENT_EVIDENCE_NAMES:
        raise ValueError("page_session evidence fields are invalid")
    return page_id, provider, reconstruction, evidence


def _resolve_evidence_path(path: str, round_dir: Path, reconstruction: Path) -> Path:
    relative = Path(*PurePosixPath(path).parts)
    candidate = round_dir / relative
    if not candidate.exists():
        candidate = reconstruction / relative
    return _contained_path(candidate, reconstruction)


def _contained_path(path: Path, root: Path) -> Path:
    lexical = path if path.is_absolute() else Path.cwd() / path
    root_absolute = root if root.is_absolute() else Path.cwd() / root
    try:
        lexical.relative_to(root_absolute)
    except ValueError as error:
        raise RuntimeError(f"Evidence path is outside page reconstruction: {path}") from error
    _validate_directory_chain(lexical.parent, root_absolute)
    return lexical


def _validate_directory_chain(directory: Path, root: Path) -> None:
    _snapshot_directory_chain(directory, root)


def _snapshot_directory_chain(
    directory: Path,
    root: Path,
) -> list[tuple[Path, tuple[int, int, int, int]]]:
    root = root if root.is_absolute() else Path.cwd() / root
    directory = directory if directory.is_absolute() else Path.cwd() / directory
    try:
        relative = directory.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Path is outside page reconstruction: {directory}") from error
    trust_root = root.parent.parent
    current = trust_root
    relative = directory.relative_to(trust_root)
    identities = []
    for part in (Path(), *relative.parts):
        if part != Path():
            current /= part
        status = current.lstat()
        if _is_link_or_reparse(status):
            raise RuntimeError(f"Evidence directory is a link or reparse point: {current}")
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(f"Evidence parent is not a directory: {current}")
        identities.append((current, _directory_identity(status)))
    return identities


def _require_directory_chain_identity(
    identities: list[tuple[Path, tuple[int, int, int, int]]],
) -> None:
    for path, expected in identities:
        try:
            status = path.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(f"Evidence directory identity changed: {path}") from error
        if _is_link_or_reparse(status) or _directory_identity(status) != expected:
            raise RuntimeError(f"Evidence directory identity changed: {path}")


def _directory_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        getattr(status, "st_file_attributes", 0),
    )


def _ensure_owned_directory(directory: Path, root: Path) -> None:
    try:
        directory.mkdir()
    except FileExistsError:
        pass
    _validate_directory_chain(directory, root)


def _read_bound_file(path: Path, reconstruction: Path) -> bytes:
    _contained_path(path, reconstruction)
    directory_identity = _snapshot_directory_chain(path.parent, reconstruction)
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"Evidence file cannot be opened safely: {path}") from error
    with os.fdopen(descriptor, "rb") as source:
        _require_directory_chain_identity(directory_identity)
        opened = os.fstat(source.fileno())
        path_status = path.lstat()
        if _is_link_or_reparse(path_status):
            raise RuntimeError(f"Evidence file is a link or reparse point: {path}")
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(path_status.st_mode):
            raise RuntimeError(f"Evidence is not a regular file: {path}")
        if opened.st_nlink != 1 or path_status.st_nlink != 1:
            raise RuntimeError(f"Evidence file is an unsafe hard link: {path}")
        if (opened.st_dev, opened.st_ino) != (path_status.st_dev, path_status.st_ino):
            raise RuntimeError(f"Evidence file identity changed: {path}")
        payload = source.read()
        stable = os.fstat(source.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            stable.st_dev,
            stable.st_ino,
            stable.st_size,
        ):
            raise RuntimeError(f"Evidence file changed while reading: {path}")
        _require_directory_chain_identity(directory_identity)
        return payload


def _write_exclusive(path: Path, payload: bytes, reconstruction: Path) -> None:
    _contained_path(path, reconstruction)
    directory_identity = _snapshot_directory_chain(path.parent, reconstruction)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        _require_directory_chain_identity(directory_identity)
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
        _require_directory_chain_identity(directory_identity)


def _validate_request_marker(marker: object) -> dict:
    fields = {
        "schema_version",
        "page_id",
        "provider",
        "repair_round",
        "request_path",
        "request_sha256",
    }
    if not isinstance(marker, dict) or set(marker) != fields:
        raise ValueError("Component agent request marker fields are invalid")
    if type(marker["schema_version"]) is not int or marker["schema_version"] != 1:
        raise ValueError("Component agent request marker schema_version is invalid")
    validate_agent_provider(marker["provider"])
    validate_repair_round(marker["repair_round"])
    if type(marker["page_id"]) is not str or not marker["page_id"]:
        raise ValueError("Component agent request marker page_id is invalid")
    expected_path = f"round-{marker['repair_round']:02d}/{REQUEST_NAME}"
    if marker["request_path"] != expected_path:
        raise ValueError("Component agent request marker path is invalid")
    digest = marker["request_sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Component agent request marker sha256 is invalid")
    return marker


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )
