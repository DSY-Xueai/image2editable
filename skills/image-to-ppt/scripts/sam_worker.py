from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import unicodedata

import numpy as np
from PIL import Image

_MODULE_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _MODULE_ROOT:
    while _MODULE_ROOT in sys.path:
        sys.path.remove(_MODULE_ROOT)
    sys.path.insert(0, _MODULE_ROOT)

from scripts.worker_resources import run_isolated_worker


def _load_tools():
    from scripts.object_detect import ObjectProposal
    from scripts.visual_segment import (
        VisualElement,
        create_sam_generator,
        generate_mask_candidates,
        generate_prompted_mask_candidates,
        recheck_visual_element_holes,
        resolve_sam_checkpoint,
    )

    return (
        ObjectProposal,
        create_sam_generator,
        generate_mask_candidates,
        generate_prompted_mask_candidates,
        resolve_sam_checkpoint,
        VisualElement,
        recheck_visual_element_holes,
    )


def _mask_record(mask, name: str = "mask") -> dict:
    binary = np.asarray(mask, dtype=bool)
    return {
        name: base64.b64encode(np.packbits(binary, axis=None).tobytes()).decode(
            "ascii"
        ),
        f"{name}_shape": list(binary.shape),
    }


def _decode_mask(record: dict, name: str = "mask") -> np.ndarray:
    shape = tuple(record[f"{name}_shape"])
    packed = np.frombuffer(
        base64.b64decode(record[name]),
        dtype=np.uint8,
    )
    return (
        np.unpackbits(
            packed,
            count=int(np.prod(shape)),
        )
        .reshape(shape)
        .astype(bool, copy=False)
    )


def _candidate_record(candidate) -> dict:
    return {
        **_mask_record(candidate.mask),
        "score": candidate.score,
        "source": candidate.source,
        "crop_box": (
            list(candidate.crop_box) if candidate.crop_box is not None else None
        ),
        "touches_crop_edge": candidate.touches_crop_edge,
        "label": candidate.label,
        "role": candidate.role,
        "object_box": (
            list(candidate.object_box) if candidate.object_box is not None else None
        ),
    }


_BATCH_SCHEMA_VERSION = 1
_BATCH_MAX_OPERATIONS = 2
_BATCH_MAX_REQUEST_BYTES = 64 * 1024
_BATCH_MAX_INPUT_BYTES = 256 * 1024 * 1024
_BATCH_MAX_PROPOSALS_BYTES = 16 * 1024 * 1024
_BATCH_MAX_RESULT_BYTES = 512 * 1024 * 1024
_BATCH_MAX_PROPOSALS = 16_384
_BATCH_MAX_PROMPTED_CANDIDATES = 2 * _BATCH_MAX_PROPOSALS
_BATCH_MAX_AUTOMATIC_CANDIDATES = 16 * 16 * 3
_BATCH_MAX_STRING_LENGTH = 256
_BATCH_CANDIDATE_FIELDS = {
    "mask",
    "mask_shape",
    "score",
    "source",
    "crop_box",
    "touches_crop_edge",
    "label",
    "role",
    "object_box",
}
_BATCH_PROPOSAL_FIELDS = {
    "box_xyxy",
    "score",
    "label",
    "role",
    "source",
    "crop_box",
    "touches_crop_edge",
}


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse
    )


def _validate_batch_directory_chain(directory: Path) -> os.stat_result:
    current = directory
    chain = []
    while True:
        chain.append(current)
        if current == current.parent:
            break
        current = current.parent
    for path in reversed(chain):
        try:
            status = path.lstat()
        except OSError as exc:
            raise ValueError(f"SAM batch directory does not exist: {path}") from exc
        if _is_link_or_reparse(status) or not stat.S_ISDIR(status.st_mode):
            raise ValueError(f"SAM batch directory is unsafe: {path}")
    return status


def _bind_batch_regular_file(path: Path, limit: int, label: str) -> dict:
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} does not exist") from exc
    if (
        _is_link_or_reparse(status)
        or not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or status.st_size > limit
    ):
        raise ValueError(f"{label} is unsafe or exceeds its size limit")
    return {
        "path": path,
        "identity": (status.st_dev, status.st_ino),
        "size": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "limit": limit,
        "label": label,
    }


def _read_batch_bound_bytes(binding: dict) -> bytes:
    path = binding["path"]
    label = binding["label"]
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} changed before it was read") from exc
    if (
        _is_link_or_reparse(status)
        or not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or (status.st_dev, status.st_ino) != binding["identity"]
        or status.st_size != binding["size"]
        or status.st_mtime_ns != binding["mtime_ns"]
        or status.st_size > binding["limit"]
    ):
        raise ValueError(f"{label} changed before it was read")
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != binding["identity"]
            or opened.st_size != binding["size"]
        ):
            raise ValueError(f"{label} identity changed")
        chunks = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, binding["limit"] + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > binding["limit"]:
                raise ValueError(f"{label} exceeds its size limit")
        stable = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            stable.st_dev,
            stable.st_ino,
            stable.st_size,
            stable.st_mtime_ns,
        ):
            raise ValueError(f"{label} changed while it was read")
        try:
            after = path.lstat()
        except OSError as exc:
            raise ValueError(f"{label} changed while it was read") from exc
        if (
            _is_link_or_reparse(after)
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or (after.st_dev, after.st_ino) != binding["identity"]
            or after.st_size != binding["size"]
            or after.st_mtime_ns != binding["mtime_ns"]
        ):
            raise ValueError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _read_batch_bound_json(binding: dict):
    try:
        return json.loads(
            _read_batch_bound_bytes(binding).decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{binding['label']} is invalid JSON") from exc


def _batch_file(root: Path, value, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"batch {field} must be a relative file name")
    lexical = root / value
    try:
        lexical_status = lexical.lstat()
        path = lexical.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"batch {field} does not exist") from exc
    if _is_link_or_reparse(lexical_status):
        raise ValueError(f"batch {field} must not be a link or reparse point")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"batch {field} must stay inside the request directory") from exc
    if not path.is_file():
        raise ValueError(f"batch {field} does not exist")
    return path


def _validate_batch_request(request_path: Path, result_path: Path) -> tuple[list[dict], dict]:
    request_path = Path(os.path.abspath(request_path))
    result_path = Path(os.path.abspath(result_path))
    lexical_root = request_path.parent
    if result_path.parent != lexical_root:
        raise ValueError("batch result must stay beside its request")
    root_status_before = _validate_batch_directory_chain(lexical_root)
    root = lexical_root.resolve(strict=True)
    if root != lexical_root:
        raise ValueError("SAM batch request directory must not resolve through a link")
    try:
        result_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ValueError("SAM batch result path is unsafe") from exc
    else:
        raise ValueError("SAM batch result already exists")

    request_binding = _bind_batch_regular_file(
        request_path,
        _BATCH_MAX_REQUEST_BYTES,
        "SAM batch request",
    )
    root_status = lexical_root.lstat()
    if (
        _is_link_or_reparse(root_status)
        or not stat.S_ISDIR(root_status.st_mode)
        or (root_status.st_dev, root_status.st_ino)
        != (root_status_before.st_dev, root_status_before.st_ino)
    ):
        raise ValueError("SAM batch request directory changed during validation")
    result_binding = {
        "path": root / result_path.name,
        "parent": root,
        "parent_identity": (root_status.st_dev, root_status.st_ino),
    }

    request = _read_batch_bound_json(request_binding)
    if not isinstance(request, dict) or set(request) != {
        "schema_version",
        "operations",
    }:
        raise ValueError("invalid SAM batch request")
    if (
        type(request["schema_version"]) is not int
        or request["schema_version"] != _BATCH_SCHEMA_VERSION
    ):
        raise ValueError("unsupported SAM batch schema version")
    operations = request["operations"]
    if not isinstance(operations, list) or len(operations) != _BATCH_MAX_OPERATIONS:
        raise ValueError("SAM batch must contain prompted and automatic operations")

    validated = []
    bound_inputs = {request_binding["path"]: request_binding}
    expected_operations = (("prompted", "prompted"), ("automatic", "automatic"))
    for operation, (expected_id, expected_kind) in zip(
        operations,
        expected_operations,
    ):
        if not isinstance(operation, dict):
            raise ValueError("invalid SAM batch operation")
        operation_id = operation.get("id")
        kind = operation.get("kind")
        if (operation_id, kind) != (expected_id, expected_kind):
            raise ValueError("SAM batch operations must be prompted then automatic")
        if kind == "prompted":
            expected_fields = {"id", "kind", "image", "text_mask", "proposals"}
        elif kind == "automatic":
            expected_fields = {"id", "kind", "image"}
        else:
            raise ValueError(f"unsupported SAM batch operation kind: {kind}")
        if set(operation) != expected_fields:
            raise ValueError(f"invalid {kind} SAM batch operation")

        image_path = _batch_file(root, operation["image"], "image")
        image_binding = bound_inputs.get(image_path)
        if image_binding is None:
            image_binding = _bind_batch_regular_file(
                image_path,
                _BATCH_MAX_INPUT_BYTES,
                "SAM batch image",
            )
            bound_inputs[image_path] = image_binding
        validated_operation = {
            "id": operation_id,
            "kind": kind,
            "image_binding": image_binding,
        }
        if kind == "prompted":
            text_mask_path = _batch_file(
                root,
                operation["text_mask"],
                "text_mask",
            )
            proposals_path = _batch_file(
                root,
                operation["proposals"],
                "proposals",
            )
            text_mask_binding = _bind_batch_regular_file(
                text_mask_path,
                _BATCH_MAX_INPUT_BYTES,
                "SAM batch text mask",
            )
            proposals_binding = _bind_batch_regular_file(
                proposals_path,
                _BATCH_MAX_PROPOSALS_BYTES,
                "SAM batch proposals",
            )
            validated_operation["text_mask_binding"] = text_mask_binding
            validated_operation["proposals_binding"] = proposals_binding
            bound_inputs[text_mask_path] = text_mask_binding
            bound_inputs[proposals_path] = proposals_binding
        validated.append(validated_operation)
    if result_binding["path"] in bound_inputs:
        raise ValueError("SAM batch result must not alias a request input")
    return validated, result_binding


def _validate_batch_string(value, label: str, *, allow_empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value)
        or len(value) > _BATCH_MAX_STRING_LENGTH
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise ValueError(f"invalid {label}")
    return value


def _validate_batch_score(value, label: str) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"invalid {label}")
    return value


def _validate_batch_box(
    value,
    label: str,
    image_shape: tuple[int, int],
    *,
    allow_none: bool,
) -> tuple[float, float, float, float] | None:
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            type(coordinate) not in {int, float} or not math.isfinite(coordinate)
            for coordinate in value
        )
    ):
        raise ValueError(f"invalid {label}")
    x1, y1, x2, y2 = value
    height, width = image_shape
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError(f"invalid {label}")
    return tuple(value)


def _validate_batch_proposals(records, image_shape: tuple[int, int]) -> list[dict]:
    if not isinstance(records, list) or len(records) > _BATCH_MAX_PROPOSALS:
        raise ValueError("prompted proposal count exceeds its limit")
    validated = []
    for record in records:
        if not isinstance(record, dict) or set(record) != _BATCH_PROPOSAL_FIELDS:
            raise ValueError("invalid prompted proposal record")
        if type(record["touches_crop_edge"]) is not bool:
            raise ValueError("invalid prompted proposal crop-edge flag")
        validated.append(
            {
                "box_xyxy": _validate_batch_box(
                    record["box_xyxy"],
                    "prompted proposal box",
                    image_shape,
                    allow_none=False,
                ),
                "score": _validate_batch_score(
                    record["score"],
                    "prompted proposal score",
                ),
                "label": _validate_batch_string(
                    record["label"],
                    "prompted proposal label",
                ),
                "role": _validate_batch_string(
                    record["role"],
                    "prompted proposal role",
                ),
                "source": _validate_batch_string(
                    record["source"],
                    "prompted proposal source",
                ),
                "crop_box": _validate_batch_box(
                    record["crop_box"],
                    "prompted proposal crop box",
                    image_shape,
                    allow_none=False,
                ),
                "touches_crop_edge": record["touches_crop_edge"],
            }
        )
    return validated


def _load_batch_inputs(operations: list[dict]) -> None:
    images = {}
    for operation in operations:
        image_binding = operation["image_binding"]
        image_path = image_binding["path"]
        if image_path not in images:
            with Image.open(
                io.BytesIO(_read_batch_bound_bytes(image_binding))
            ) as stored_image:
                images[image_path] = np.asarray(stored_image.convert("RGB")).copy()
        operation["image"] = images[image_path]
        if operation["kind"] == "prompted":
            with Image.open(
                io.BytesIO(
                    _read_batch_bound_bytes(operation["text_mask_binding"])
                )
            ) as stored_mask:
                text_mask = np.asarray(stored_mask.convert("L")).copy()
            if text_mask.shape != operation["image"].shape[:2]:
                raise ValueError("prompted text mask shape does not match the image")
            proposal_records = _validate_batch_proposals(
                _read_batch_bound_json(operation["proposals_binding"]),
                tuple(operation["image"].shape[:2]),
            )
            operation["text_mask"] = text_mask
            operation["proposal_records"] = proposal_records


def _validate_batch_output(payload: dict, operations: list[dict]) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "operations",
    }:
        raise RuntimeError("invalid SAM batch output")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != _BATCH_SCHEMA_VERSION
    ):
        raise RuntimeError("invalid SAM batch output schema version")
    output_operations = payload["operations"]
    if not isinstance(output_operations, list) or len(output_operations) != len(
        operations
    ):
        raise RuntimeError("invalid SAM batch output operation count")

    operation_records = []
    for expected, actual in zip(operations, output_operations):
        if not isinstance(actual, dict) or set(actual) != {
            "id",
            "kind",
            "candidates",
        }:
            raise RuntimeError("invalid SAM batch output operation")
        if (actual["id"], actual["kind"]) != (
            expected["id"],
            expected["kind"],
        ):
            raise RuntimeError("invalid SAM batch output operation order")
        records = actual["candidates"]
        if not isinstance(records, list):
            raise RuntimeError("invalid SAM batch candidate records")
        candidate_limit = (
            _BATCH_MAX_PROMPTED_CANDIDATES
            if expected["kind"] == "prompted"
            else _BATCH_MAX_AUTOMATIC_CANDIDATES
        )
        if len(records) > candidate_limit:
            raise RuntimeError("SAM batch candidate count exceeds its limit")
        operation_records.append((expected, records))

    for expected, records in operation_records:
        expected_shape = tuple(expected["image"].shape[:2])
        expected_bytes = (int(np.prod(expected_shape)) + 7) // 8
        expected_base64_length = ((expected_bytes + 2) // 3) * 4
        for record in records:
            if not isinstance(record, dict) or set(record) != _BATCH_CANDIDATE_FIELDS:
                raise RuntimeError("invalid SAM batch candidate record")
            shape = record["mask_shape"]
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(value) is not int for value in shape)
                or tuple(shape) != expected_shape
            ):
                raise RuntimeError("SAM batch candidate mask shape does not match image")
            if (
                not isinstance(record["mask"], str)
                or len(record["mask"]) != expected_base64_length
            ):
                raise RuntimeError("invalid SAM batch candidate mask length")
            try:
                packed = base64.b64decode(record["mask"], validate=True)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("invalid SAM batch candidate mask") from exc
            if len(packed) != expected_bytes:
                raise RuntimeError("invalid SAM batch candidate mask length")
            try:
                _validate_batch_score(record["score"], "SAM batch candidate score")
                _validate_batch_string(
                    record["source"],
                    "SAM batch candidate source",
                )
                _validate_batch_box(
                    record["crop_box"],
                    "SAM batch candidate crop box",
                    expected_shape,
                    allow_none=True,
                )
                _validate_batch_string(
                    record["label"],
                    "SAM batch candidate label",
                    allow_empty=True,
                )
                _validate_batch_string(
                    record["role"],
                    "SAM batch candidate role",
                    allow_empty=True,
                )
                _validate_batch_box(
                    record["object_box"],
                    "SAM batch candidate object box",
                    expected_shape,
                    allow_none=True,
                )
            except ValueError as exc:
                raise RuntimeError("invalid SAM batch candidate metadata") from exc
            if type(record["touches_crop_edge"]) is not bool:
                raise RuntimeError("invalid SAM batch candidate crop-edge flag")


def _write_batch_result(
    result_binding: dict,
    payload: dict,
    operations: list[dict],
) -> None:
    _validate_batch_output(payload, operations)
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(payload_bytes) > _BATCH_MAX_RESULT_BYTES:
        raise RuntimeError("SAM batch result exceeds its size limit")
    result_path = result_binding["path"]
    _verify_batch_result_binding(result_binding)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{result_path.name}.",
        suffix=".tmp",
        dir=result_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        descriptor_identity = None
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(payload_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            descriptor_stat = os.fstat(temporary_file.fileno())
            path_stat = os.stat(temporary_path, follow_symlinks=False)
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
                or descriptor_stat.st_size != len(payload_bytes)
                or path_stat.st_size != len(payload_bytes)
            ):
                raise RuntimeError("SAM batch result temp identity changed")
            descriptor_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        closed_stat = os.stat(temporary_path, follow_symlinks=False)
        if (
            _is_link_or_reparse(closed_stat)
            or not stat.S_ISREG(closed_stat.st_mode)
            or (closed_stat.st_dev, closed_stat.st_ino) != descriptor_identity
            or closed_stat.st_size != len(payload_bytes)
        ):
            raise RuntimeError("SAM batch result temp identity changed")
        _verify_batch_result_binding(result_binding)
        try:
            os.link(temporary_path, result_path)
        except FileExistsError as exc:
            raise RuntimeError("SAM batch result path already exists") from exc
        except OSError as exc:
            raise RuntimeError("SAM batch result could not be published") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _verify_batch_result_binding(result_binding: dict) -> None:
    parent = result_binding["parent"]
    try:
        parent_status = parent.lstat()
    except OSError as exc:
        raise RuntimeError("SAM batch result directory changed") from exc
    if (
        _is_link_or_reparse(parent_status)
        or not stat.S_ISDIR(parent_status.st_mode)
        or (parent_status.st_dev, parent_status.st_ino)
        != result_binding["parent_identity"]
    ):
        raise RuntimeError("SAM batch result directory changed")
    try:
        result_binding["path"].lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError("SAM batch result path changed") from exc
    raise RuntimeError("SAM batch result path already exists")


def _run_candidate_batch(request_path: Path, result_path: Path) -> int:
    operations, result_binding = _validate_batch_request(request_path, result_path)
    _load_batch_inputs(operations)
    (
        proposal_type,
        create_generator,
        generate_automatic,
        generate_prompted,
        resolve_checkpoint,
        _,
        _,
    ) = _load_tools()
    generator = create_generator(resolve_checkpoint(), resource_safe=True)
    output_operations = []
    for operation in operations:
        image = operation["image"]
        if operation["kind"] == "prompted":
            proposals = [
                proposal_type(
                    **{
                        **record,
                        "box_xyxy": tuple(record["box_xyxy"]),
                        "crop_box": tuple(record["crop_box"]),
                    }
                )
                for record in operation["proposal_records"]
            ]
            candidates = generate_prompted(
                image,
                proposals,
                generator,
                operation["text_mask"],
            )
        else:
            candidates = generate_automatic(
                image,
                generator,
                crop_size=max(image.shape[:2]),
                include_geometry=False,
                min_score=0.90,
            )
        output_operations.append(
            {
                "id": operation["id"],
                "kind": operation["kind"],
                "candidates": [
                    _candidate_record(candidate) for candidate in candidates
                ],
            }
        )

    payload = {
        "schema_version": _BATCH_SCHEMA_VERSION,
        "operations": output_operations,
    }
    _write_batch_result(result_binding, payload, operations)
    return 0


def component_prompt_mask(generator, image: np.ndarray, prompt: dict) -> np.ndarray:
    """Run one box/point prompt inside the isolated SAM worker process."""

    predictor = generator.predictor
    predictor.set_image(image)
    positive = prompt.get("positive", [])
    negative = prompt.get("negative", [])
    points = np.asarray(positive + negative, dtype=np.float32)
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32)
    box = prompt.get("box")
    masks, scores, _ = predictor.predict(
        point_coords=points if len(points) else None,
        point_labels=labels if len(points) else None,
        box=np.asarray(box, dtype=np.float32) if box is not None else None,
        multimask_output=True,
    )
    return np.asarray(masks[int(np.argmax(scores))], dtype=bool)


def run_component_prompt_worker(
    image: np.ndarray,
    *,
    box,
    positive,
    negative,
    work_dir: str | Path,
) -> np.ndarray:
    """Run one component prompt in a disposable SAM subprocess."""

    with tempfile.TemporaryDirectory(prefix="component-sam-", dir=work_dir) as temporary:
        root = Path(temporary)
        image_path = root / "image.png"
        prompt_path = root / "prompt.json"
        result_path = root / "result.json"
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(image_path)
        prompt_path.write_text(
            json.dumps({"box": box, "positive": positive, "negative": negative}),
            encoding="utf-8",
        )
        run_isolated_worker(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode", "component",
                "--image", str(image_path),
                "--prompt", str(prompt_path),
                "--result", str(result_path),
            ],
            check=True,
            timeout=600,
        )
        records = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or len(records) != 1:
            raise RuntimeError("SAM component worker returned an invalid result")
        return _decode_mask(records[0]).copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prompted", "automatic", "recheck", "component", "batch"),
        required=True,
    )
    parser.add_argument("--image")
    parser.add_argument("--request")
    parser.add_argument("--text-mask")
    parser.add_argument("--proposals")
    parser.add_argument("--elements")
    parser.add_argument("--prompt")
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    if args.mode == "batch":
        if not args.request:
            raise ValueError("batch mode requires request")
        return _run_candidate_batch(Path(args.request), Path(args.result))
    if not args.image:
        raise ValueError(f"{args.mode} mode requires image")

    (
        proposal_type,
        create_generator,
        generate_automatic,
        generate_prompted,
        resolve_checkpoint,
        visual_element_type,
        recheck_holes,
    ) = _load_tools()
    with Image.open(args.image) as stored_image:
        image = np.asarray(stored_image.convert("RGB")).copy()
    generator = create_generator(
        resolve_checkpoint(),
        resource_safe=True,
    )
    if args.mode == "recheck":
        if not args.elements:
            raise ValueError("recheck mode requires elements")
        element_records = json.loads(Path(args.elements).read_text(encoding="utf-8"))
        elements = [
            visual_element_type(
                mask=_decode_mask(record),
                z_index=record["z_index"],
                score=record["score"],
                source=record["source"],
                semantic_mask=_decode_mask(record, "semantic_mask"),
                object_box=(
                    tuple(record["object_box"])
                    if record["object_box"] is not None
                    else None
                ),
            )
            for record in element_records
        ]
        recheck_holes(image, elements, generator)
        output_records = [
            {
                **_mask_record(element.mask),
                **_mask_record(element.semantic_mask, "semantic_mask"),
            }
            for element in elements
        ]
    elif args.mode == "prompted":
        if not args.text_mask or not args.proposals:
            raise ValueError("prompted mode requires text mask and proposals")
        with Image.open(args.text_mask) as stored_mask:
            text_mask = np.asarray(stored_mask.convert("L")).copy()
        records = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
        proposals = [
            proposal_type(
                **{
                    **record,
                    "box_xyxy": tuple(record["box_xyxy"]),
                    "crop_box": tuple(record["crop_box"]),
                }
            )
            for record in records
        ]
        candidates = generate_prompted(
            image,
            proposals,
            generator,
            text_mask,
        )
        output_records = [_candidate_record(candidate) for candidate in candidates]
    elif args.mode == "automatic":
        candidates = generate_automatic(
            image,
            generator,
            crop_size=max(image.shape[:2]),
            include_geometry=False,
            min_score=0.90,
        )
        output_records = [_candidate_record(candidate) for candidate in candidates]
    else:
        if not args.prompt:
            raise ValueError("component mode requires prompt")
        prompt = json.loads(Path(args.prompt).read_text(encoding="utf-8"))
        output_records = [_mask_record(component_prompt_mask(generator, image, prompt))]

    result_path = Path(args.result)
    temporary_path = result_path.with_name(f".{result_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            output_records,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_path, result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
