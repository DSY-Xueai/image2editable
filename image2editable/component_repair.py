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
            payload = _read_bound_file(source)
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
            _write_exclusive(staging / name, payload)
        request_bytes = json.dumps(
            request,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        _write_exclusive(staging / REQUEST_NAME, request_bytes)
        try:
            staging.rename(round_dir)
        except OSError as error:
            raise RuntimeError(
                f"Agent evidence round is already published: {round_dir}"
            ) from error
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return round_dir / REQUEST_NAME


def load_component_agent_request(request_path: str | Path) -> dict:
    request_path = Path(request_path)
    if request_path.name != REQUEST_NAME or not request_path.parent.name.startswith("round-"):
        raise RuntimeError("Component agent request path is invalid")
    round_dir = request_path.parent
    reconstruction = round_dir.parent.parent
    _validate_directory_chain(round_dir, reconstruction)
    request = json.loads(_read_bound_file(request_path).decode("utf-8"))
    validate_component_agent_request(request)
    expected_round = f"round-{request['repair_round']:02d}"
    if round_dir.name != expected_round or reconstruction.parent.name != request["page_id"]:
        raise RuntimeError("Component agent request belongs to another page or round")
    graph_payload = b""
    for name, record in request["evidence"].items():
        evidence_path = _resolve_evidence_path(record["path"], round_dir, reconstruction)
        payload = _read_bound_file(evidence_path)
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
    if reconstruction.name != "reconstruction" or reconstruction.parent.name != page_id:
        raise RuntimeError("page_session reconstruction belongs to another page")
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
    root = root if root.is_absolute() else Path.cwd() / root
    directory = directory if directory.is_absolute() else Path.cwd() / directory
    try:
        relative = directory.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Path is outside page reconstruction: {directory}") from error
    current = root
    for part in (Path(), *relative.parts):
        if part != Path():
            current /= part
        status = current.lstat()
        if _is_link_or_reparse(status):
            raise RuntimeError(f"Evidence directory is a link or reparse point: {current}")
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(f"Evidence parent is not a directory: {current}")


def _ensure_owned_directory(directory: Path, root: Path) -> None:
    try:
        directory.mkdir()
    except FileExistsError:
        pass
    _validate_directory_chain(directory, root)


def _read_bound_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"Evidence file cannot be opened safely: {path}") from error
    with os.fdopen(descriptor, "rb") as source:
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
        return payload


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )
