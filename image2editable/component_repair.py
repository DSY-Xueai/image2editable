from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
import uuid

from image2editable.component_contracts import (
    COMPONENT_EVIDENCE_NAMES,
    validate_agent_provider,
    validate_component_agent_request,
    validate_component_graph,
    validate_repair_round,
)


EVIDENCE_NAMES = tuple(sorted(COMPONENT_EVIDENCE_NAMES))
REQUEST_NAME = "component_agent_request.json"
MARKER_NAME = "publication-marker.json"
INTEGRITY_DIRECTORY = ".component-agent-integrity"
INTEGRITY_KEY_NAME = "key.bin"
IO_CHUNK_SIZE = 1024 * 1024
GRAPH_JSON_LIMIT = 16 * 1024 * 1024
REQUEST_JSON_LIMIT = 4 * 1024 * 1024
MARKER_JSON_LIMIT = 64 * 1024


def build_component_agent_request(
    page_session: dict,
    *,
    repair_round: int,
) -> Path:
    repair_round = validate_repair_round(repair_round)
    page_id, provider, reconstruction, sources = _validate_page_session(page_session)
    integrity_key = _load_or_create_integrity_key(reconstruction)
    agent_dir = reconstruction / "agent"
    _ensure_owned_directory(agent_dir, reconstruction)
    round_dir = agent_dir / f"round-{repair_round:02d}"
    if round_dir.exists() or round_dir.is_symlink():
        raise RuntimeError(f"Agent evidence round is already published: {round_dir}")
    staging = agent_dir / f".{round_dir.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    staging_identity = _directory_identity(staging.lstat())
    try:
        records: dict[str, dict[str, str]] = {}
        graph_payload = b""
        for name in EVIDENCE_NAMES:
            _require_single_directory_identity(staging, staging_identity)
            source = _contained_path(Path(sources[name]), reconstruction)
            digest, captured = _copy_bound_file(
                source,
                staging / name,
                reconstruction,
                capture_limit=(
                    GRAPH_JSON_LIMIT if name == "component-graph.json" else None
                ),
            )
            records[name] = {
                "path": name,
                "sha256": digest,
            }
            if captured is not None:
                graph_payload = captured
            _require_single_directory_identity(staging, staging_identity)
        graph = json.loads(graph_payload.decode("utf-8"))
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
        request_bytes = json.dumps(
            request,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        if len(request_bytes) > REQUEST_JSON_LIMIT:
            raise RuntimeError("Component agent request JSON size limit exceeded")
        _require_single_directory_identity(staging, staging_identity)
        _write_exclusive(staging / REQUEST_NAME, request_bytes, reconstruction)
        _require_single_directory_identity(staging, staging_identity)
        marker_fields = {
            "schema_version": 1,
            "page_id": page_id,
            "provider": provider,
            "repair_round": repair_round,
            "request_path": f"{round_dir.name}/{REQUEST_NAME}",
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        }
        marker = {
            **marker_fields,
            "hmac_sha256": hmac.new(
                integrity_key,
                _canonical_marker_fields(marker_fields),
                hashlib.sha256,
            ).hexdigest(),
        }
        marker_bytes = json.dumps(
            marker,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        if len(marker_bytes) > MARKER_JSON_LIMIT:
            raise RuntimeError("Component agent marker JSON size limit exceeded")
        _write_exclusive(staging / MARKER_NAME, marker_bytes, reconstruction)
        _require_single_directory_identity(staging, staging_identity)
        _verify_staged_bundle(
            staging,
            reconstruction,
            staging_identity,
            records,
            request_bytes,
            marker_bytes,
        )
        agent_identity = _snapshot_directory_chain(agent_dir, reconstruction)
        _require_single_directory_identity(staging, staging_identity)
        try:
            staging.rename(round_dir)
        except OSError as error:
            raise RuntimeError(
                f"Agent evidence round is already published: {round_dir}"
            ) from error
        _require_directory_chain_identity(agent_identity)
        try:
            round_status = round_dir.lstat()
        except FileNotFoundError as error:
            raise RuntimeError("Agent evidence staging identity changed") from error
        if (
            _is_link_or_reparse(round_status)
            or not stat.S_ISDIR(round_status.st_mode)
            or _directory_identity(round_status) != staging_identity
        ):
            _remove_rejected_round(round_dir, agent_dir)
            raise RuntimeError("Agent evidence staging identity changed")
    except BaseException:
        _cleanup_owned_staging(staging, staging_identity)
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
    integrity_key = _load_integrity_key(reconstruction)
    marker_path = round_dir / MARKER_NAME
    try:
        marker_bytes = _read_bound_file(
            marker_path,
            reconstruction,
            max_bytes=MARKER_JSON_LIMIT,
            label="marker JSON",
        )
    except RuntimeError as error:
        if "size limit" in str(error):
            raise
        raise RuntimeError("Component agent request marker is missing or invalid") from error
    try:
        marker = json.loads(marker_bytes.decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Component agent request marker is missing or invalid") from error
    _validate_request_marker(marker)
    marker_fields = {
        key: value for key, value in marker.items() if key != "hmac_sha256"
    }
    expected_signature = hmac.new(
        integrity_key,
        _canonical_marker_fields(marker_fields),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(marker["hmac_sha256"], expected_signature):
        raise RuntimeError("Component agent publication signature mismatch")
    request_bytes = _read_bound_file(
        request_path,
        reconstruction,
        max_bytes=REQUEST_JSON_LIMIT,
        label="request JSON",
    )
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
        if name == "component-graph.json":
            graph_payload = _read_bound_file(
                evidence_path,
                reconstruction,
                max_bytes=GRAPH_JSON_LIMIT,
                label="component graph JSON",
            )
            digest = hashlib.sha256(graph_payload).hexdigest()
        else:
            digest = _hash_bound_file(evidence_path, reconstruction)
        if digest != record["sha256"]:
            raise RuntimeError(f"Component evidence hash mismatch: {name}")
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


def _require_single_directory_identity(
    directory: Path,
    expected: tuple[int, int, int, int],
) -> None:
    try:
        status = directory.lstat()
    except FileNotFoundError as error:
        raise RuntimeError("Agent evidence staging identity changed") from error
    if (
        _is_link_or_reparse(status)
        or not stat.S_ISDIR(status.st_mode)
        or _directory_identity(status) != expected
    ):
        raise RuntimeError("Agent evidence staging identity changed")


def _cleanup_owned_staging(
    staging: Path,
    expected: tuple[int, int, int, int],
) -> None:
    try:
        status = staging.lstat()
    except FileNotFoundError:
        return
    if (
        _is_link_or_reparse(status)
        or not stat.S_ISDIR(status.st_mode)
        or _directory_identity(status) != expected
    ):
        return
    shutil.rmtree(staging)


def _remove_rejected_round(round_dir: Path, agent_dir: Path) -> None:
    if round_dir.parent != agent_dir or not _is_round_name(round_dir.name):
        raise RuntimeError("Refusing to clean an invalid Agent round path")
    status = round_dir.lstat()
    if _is_link_or_reparse(status):
        if stat.S_ISLNK(status.st_mode):
            round_dir.unlink()
        else:
            round_dir.rmdir()
        return
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError("Refusing to clean an unsafe Agent round")
    expected = _directory_identity(status)
    if _directory_identity(round_dir.lstat()) != expected:
        raise RuntimeError("Refusing to clean a replaced Agent round")
    shutil.rmtree(round_dir)


def _verify_staged_bundle(
    staging: Path,
    reconstruction: Path,
    staging_identity: tuple[int, int, int, int],
    records: dict[str, dict[str, str]],
    request_bytes: bytes,
    marker_bytes: bytes,
) -> None:
    _require_single_directory_identity(staging, staging_identity)
    for name, record in records.items():
        if _hash_bound_file(staging / name, reconstruction) != record["sha256"]:
            raise RuntimeError(f"Staged component evidence hash mismatch: {name}")
        _require_single_directory_identity(staging, staging_identity)
    if _read_bound_file(
        staging / REQUEST_NAME,
        reconstruction,
        max_bytes=REQUEST_JSON_LIMIT,
        label="request JSON",
    ) != request_bytes:
        raise RuntimeError("Staged component request changed before publication")
    if _read_bound_file(
        staging / MARKER_NAME,
        reconstruction,
        max_bytes=MARKER_JSON_LIMIT,
        label="marker JSON",
    ) != marker_bytes:
        raise RuntimeError("Staged component marker changed before publication")
    _require_single_directory_identity(staging, staging_identity)


def _ensure_owned_directory(directory: Path, root: Path) -> None:
    try:
        directory.mkdir()
    except FileExistsError:
        pass
    _validate_directory_chain(directory, root)


def _read_bound_file(
    path: Path,
    reconstruction: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
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
        chunks = []
        total = 0
        while True:
            chunk = source.read(IO_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(f"Component agent {label} size limit exceeded")
            chunks.append(chunk)
        stable = os.fstat(source.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            stable.st_dev,
            stable.st_ino,
            stable.st_size,
        ):
            raise RuntimeError(f"Evidence file changed while reading: {path}")
        _require_directory_chain_identity(directory_identity)
        return b"".join(chunks)


def _hash_bound_file(path: Path, reconstruction: Path) -> str:
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
        opened = _validate_open_regular_file(path, source.fileno(), directory_identity)
        digest = hashlib.sha256()
        while True:
            chunk = source.read(IO_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        _validate_stable_open_file(path, source.fileno(), opened, directory_identity)
        return digest.hexdigest()


def _copy_bound_file(
    source_path: Path,
    target_path: Path,
    reconstruction: Path,
    *,
    capture_limit: int | None,
) -> tuple[str, bytes | None]:
    _contained_path(source_path, reconstruction)
    _contained_path(target_path, reconstruction)
    source_directories = _snapshot_directory_chain(
        source_path.parent, reconstruction
    )
    target_directories = _snapshot_directory_chain(
        target_path.parent, reconstruction
    )
    read_flags = os.O_RDONLY
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        read_flags |= getattr(os, name, 0)
        write_flags |= getattr(os, name, 0)
    source_descriptor = os.open(source_path, read_flags)
    try:
        target_descriptor = os.open(target_path, write_flags, 0o600)
    except BaseException:
        os.close(source_descriptor)
        raise
    with os.fdopen(source_descriptor, "rb") as source, os.fdopen(
        target_descriptor, "wb"
    ) as target:
        source_status = _validate_open_regular_file(
            source_path, source.fileno(), source_directories
        )
        target_status = _validate_open_regular_file(
            target_path, target.fileno(), target_directories
        )
        digest = hashlib.sha256()
        captured = bytearray() if capture_limit is not None else None
        while True:
            chunk = source.read(IO_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            target.write(chunk)
            if captured is not None:
                if len(captured) + len(chunk) > capture_limit:
                    raise RuntimeError(
                        "Component agent component graph JSON size limit exceeded"
                    )
                captured.extend(chunk)
        target.flush()
        os.fsync(target.fileno())
        _validate_stable_open_file(
            source_path, source.fileno(), source_status, source_directories
        )
        _validate_stable_open_file(
            target_path,
            target.fileno(),
            target_status,
            target_directories,
            allow_size_change=True,
        )
        return digest.hexdigest(), None if captured is None else bytes(captured)


def _validate_open_regular_file(
    path: Path,
    descriptor: int,
    directory_identity: list[tuple[Path, tuple[int, int, int, int]]],
) -> os.stat_result:
    _require_directory_chain_identity(directory_identity)
    opened = os.fstat(descriptor)
    path_status = path.lstat()
    if _is_link_or_reparse(path_status):
        raise RuntimeError(f"Evidence file is a link or reparse point: {path}")
    if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(path_status.st_mode):
        raise RuntimeError(f"Evidence is not a regular file: {path}")
    if opened.st_nlink != 1 or path_status.st_nlink != 1:
        raise RuntimeError(f"Evidence file is an unsafe hard link: {path}")
    if (opened.st_dev, opened.st_ino) != (path_status.st_dev, path_status.st_ino):
        raise RuntimeError(f"Evidence file identity changed: {path}")
    return opened


def _validate_stable_open_file(
    path: Path,
    descriptor: int,
    opened: os.stat_result,
    directory_identity: list[tuple[Path, tuple[int, int, int, int]]],
    *,
    allow_size_change: bool = False,
) -> None:
    stable = os.fstat(descriptor)
    if (
        (opened.st_dev, opened.st_ino) != (stable.st_dev, stable.st_ino)
        or (not allow_size_change and opened.st_size != stable.st_size)
    ):
        raise RuntimeError(f"Evidence file changed while reading: {path}")
    _require_directory_chain_identity(directory_identity)


def _write_exclusive(path: Path, payload: bytes, reconstruction: Path) -> None:
    _contained_path(path, reconstruction)
    directory_identity = _snapshot_directory_chain(path.parent, reconstruction)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        _require_directory_chain_identity(directory_identity)
        raise RuntimeError(f"Evidence file cannot be created safely: {path}") from error
    with os.fdopen(descriptor, "wb") as target:
        _require_directory_chain_identity(directory_identity)
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
        _require_directory_chain_identity(directory_identity)


def _load_or_create_integrity_key(reconstruction: Path) -> bytes:
    run_root = reconstruction.parent.parent.parent
    anchor = run_root / INTEGRITY_DIRECTORY
    if anchor.exists() or anchor.is_symlink():
        return _read_integrity_key(anchor, run_root)
    if _run_has_published_rounds(run_root):
        raise RuntimeError(
            "Run has published Agent rounds but its integrity key is missing"
        )
    staging = run_root / f".{INTEGRITY_DIRECTORY}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir(mode=0o700)
        key_path = staging / INTEGRITY_KEY_NAME
        key = secrets.token_bytes(32)
        _write_new_integrity_key(key_path, key, staging, run_root)
        try:
            staging.rename(anchor)
        except OSError:
            if staging.exists():
                shutil.rmtree(staging)
            if not anchor.exists() and not anchor.is_symlink():
                raise RuntimeError("Component agent integrity key creation failed")
        return _read_integrity_key(anchor, run_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _load_integrity_key(reconstruction: Path) -> bytes:
    run_root = reconstruction.parent.parent.parent
    anchor = run_root / INTEGRITY_DIRECTORY
    if not anchor.exists() and not anchor.is_symlink():
        raise RuntimeError("Component agent integrity key is missing")
    return _read_integrity_key(anchor, run_root)


def _run_has_published_rounds(run_root: Path) -> bool:
    pages = run_root / "pages"
    if not pages.exists() and not pages.is_symlink():
        return False
    _require_safe_directory(pages, "pages")
    found = False
    for page in pages.iterdir():
        page_status = page.lstat()
        if _is_link_or_reparse(page_status):
            raise RuntimeError(f"Run pages entry is a link or reparse point: {page}")
        if not stat.S_ISDIR(page_status.st_mode):
            continue
        reconstruction = page / "reconstruction"
        if not reconstruction.exists() and not reconstruction.is_symlink():
            continue
        _require_safe_directory(reconstruction, "reconstruction")
        agent = reconstruction / "agent"
        if not agent.exists() and not agent.is_symlink():
            continue
        _require_safe_directory(agent, "agent")
        for child in agent.iterdir():
            if not _is_round_name(child.name):
                continue
            status = child.lstat()
            if _is_link_or_reparse(status):
                raise RuntimeError(
                    f"Published Agent round is a link or reparse point: {child}"
                )
            if not stat.S_ISDIR(status.st_mode):
                raise RuntimeError(f"Published Agent round is not a directory: {child}")
            found = True
    return found


def _require_safe_directory(path: Path, label: str) -> None:
    status = path.lstat()
    if _is_link_or_reparse(status):
        raise RuntimeError(f"Run {label} is a link or reparse point: {path}")
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"Run {label} is not a directory: {path}")


def _is_round_name(name: str) -> bool:
    return len(name) == 8 and name.startswith("round-") and name[6:] in {
        "01",
        "02",
        "03",
        "04",
        "05",
    }


def _write_new_integrity_key(
    path: Path,
    key: bytes,
    staging: Path,
    run_root: Path,
) -> None:
    directory_identity = _snapshot_key_directory_chain(staging, run_root)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        _require_directory_chain_identity(directory_identity)
        target.write(key)
        target.flush()
        os.fsync(target.fileno())
        os.chmod(path, 0o600)
        _require_directory_chain_identity(directory_identity)


def _read_integrity_key(anchor: Path, run_root: Path) -> bytes:
    directory_identity = _snapshot_key_directory_chain(anchor, run_root)
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    key_path = anchor / INTEGRITY_KEY_NAME
    try:
        descriptor = os.open(key_path, flags)
    except OSError as error:
        raise RuntimeError("Component agent integrity key cannot be opened safely") from error
    with os.fdopen(descriptor, "rb") as source:
        _require_directory_chain_identity(directory_identity)
        opened = os.fstat(source.fileno())
        path_status = key_path.lstat()
        if _is_link_or_reparse(path_status):
            raise RuntimeError("Component agent integrity key is a link or reparse point")
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(path_status.st_mode)
            or opened.st_nlink != 1
            or path_status.st_nlink != 1
        ):
            raise RuntimeError("Component agent integrity key is an unsafe hard link")
        if (opened.st_dev, opened.st_ino) != (path_status.st_dev, path_status.st_ino):
            raise RuntimeError("Component agent integrity key identity changed")
        key = source.read(33)
        stable = os.fstat(source.fileno())
        if (
            len(key) != 32
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (stable.st_dev, stable.st_ino, stable.st_size)
        ):
            raise RuntimeError("Component agent integrity key is damaged")
        if os.name != "nt" and stat.S_IMODE(path_status.st_mode) & 0o077:
            raise RuntimeError("Component agent integrity key permissions are unsafe")
        _require_directory_chain_identity(directory_identity)
        return key


def _snapshot_key_directory_chain(
    directory: Path,
    run_root: Path,
) -> list[tuple[Path, tuple[int, int, int, int]]]:
    try:
        relative = directory.relative_to(run_root)
    except ValueError as error:
        raise RuntimeError("Component agent integrity key is outside run root") from error
    current = run_root
    identities = []
    for part in (Path(), *relative.parts):
        if part != Path():
            current /= part
        status = current.lstat()
        if _is_link_or_reparse(status):
            raise RuntimeError(
                "Component agent integrity key directory is a link or reparse point"
            )
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError("Component agent integrity key parent is not a directory")
        identities.append((current, _directory_identity(status)))
    return identities


def _validate_request_marker(marker: object) -> dict:
    fields = {
        "schema_version",
        "page_id",
        "provider",
        "repair_round",
        "request_path",
        "request_sha256",
        "hmac_sha256",
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
    for field in ("request_sha256", "hmac_sha256"):
        digest = marker[field]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Component agent request marker {field} is invalid")
    return marker


def _canonical_marker_fields(fields: dict) -> bytes:
    return json.dumps(
        fields,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )
