from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

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
        "crop_box": candidate.crop_box,
        "touches_crop_edge": candidate.touches_crop_edge,
        "label": candidate.label,
        "role": candidate.role,
        "object_box": candidate.object_box,
    }


_BATCH_SCHEMA_VERSION = 1
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


def _batch_file(root: Path, value, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"batch {field} must be a relative file name")
    try:
        path = (root / value).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"batch {field} does not exist") from exc
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"batch {field} must stay inside the request directory") from exc
    if not path.is_file():
        raise ValueError(f"batch {field} does not exist")
    return path


def _validate_batch_request(request_path: Path, result_path: Path) -> list[dict]:
    root = request_path.resolve().parent
    result_path = result_path.resolve()
    if result_path.parent != root:
        raise ValueError("batch result must stay beside its request")
    result_path.unlink(missing_ok=True)

    request = json.loads(request_path.read_text(encoding="utf-8"))
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
    if not isinstance(operations, list) or not operations:
        raise ValueError("SAM batch operations must be a non-empty list")

    ids = set()
    validated = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("invalid SAM batch operation")
        operation_id = operation.get("id")
        kind = operation.get("kind")
        if (
            not isinstance(operation_id, str)
            or not operation_id
            or operation_id in ids
        ):
            raise ValueError("SAM batch operation IDs must be unique strings")
        ids.add(operation_id)
        if kind == "prompted":
            expected_fields = {"id", "kind", "image", "text_mask", "proposals"}
        elif kind == "automatic":
            expected_fields = {"id", "kind", "image"}
        else:
            raise ValueError(f"unsupported SAM batch operation kind: {kind}")
        if set(operation) != expected_fields:
            raise ValueError(f"invalid {kind} SAM batch operation")

        validated_operation = {
            "id": operation_id,
            "kind": kind,
            "image_path": _batch_file(root, operation["image"], "image"),
        }
        if kind == "prompted":
            validated_operation["text_mask_path"] = _batch_file(
                root,
                operation["text_mask"],
                "text_mask",
            )
            validated_operation["proposals_path"] = _batch_file(
                root,
                operation["proposals"],
                "proposals",
            )
        validated.append(validated_operation)
    return validated


def _load_batch_inputs(operations: list[dict]) -> None:
    images = {}
    for operation in operations:
        image_path = operation["image_path"]
        if image_path not in images:
            with Image.open(image_path) as stored_image:
                images[image_path] = np.asarray(stored_image.convert("RGB")).copy()
        operation["image"] = images[image_path]
        if operation["kind"] == "prompted":
            with Image.open(operation["text_mask_path"]) as stored_mask:
                text_mask = np.asarray(stored_mask.convert("L")).copy()
            if text_mask.shape != operation["image"].shape[:2]:
                raise ValueError("prompted text mask shape does not match the image")
            proposal_records = json.loads(
                operation["proposals_path"].read_text(encoding="utf-8")
            )
            if not isinstance(proposal_records, list):
                raise ValueError("prompted proposals must be a list")
            operation["text_mask"] = text_mask
            operation["proposal_records"] = proposal_records


def _validate_batch_output(payload: dict, operations: list[dict]) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "operations",
    }:
        raise RuntimeError("invalid SAM batch output")
    if payload["schema_version"] != _BATCH_SCHEMA_VERSION:
        raise RuntimeError("invalid SAM batch output schema version")
    output_operations = payload["operations"]
    if not isinstance(output_operations, list) or len(output_operations) != len(
        operations
    ):
        raise RuntimeError("invalid SAM batch output operation count")

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
        expected_shape = tuple(expected["image"].shape[:2])
        for record in records:
            if not isinstance(record, dict) or set(record) != _BATCH_CANDIDATE_FIELDS:
                raise RuntimeError("invalid SAM batch candidate record")
            shape = record["mask_shape"]
            if not isinstance(shape, list) or tuple(shape) != expected_shape:
                raise RuntimeError("SAM batch candidate mask shape does not match image")
            try:
                packed = base64.b64decode(record["mask"], validate=True)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("invalid SAM batch candidate mask") from exc
            expected_bytes = (int(np.prod(expected_shape)) + 7) // 8
            if len(packed) != expected_bytes:
                raise RuntimeError("invalid SAM batch candidate mask length")


def _write_batch_result(
    result_path: Path,
    payload: dict,
    operations: list[dict],
) -> None:
    _validate_batch_output(payload, operations)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{result_path.name}.",
        suffix=".tmp",
        dir=result_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w+", encoding="utf-8") as temporary_file:
            json.dump(payload, temporary_file, ensure_ascii=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_file.seek(0)
            _validate_batch_output(json.load(temporary_file), operations)
            descriptor_stat = os.fstat(temporary_file.fileno())
            path_stat = os.stat(temporary_path, follow_symlinks=False)
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise RuntimeError("SAM batch result temp identity changed")
        os.replace(temporary_path, result_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_candidate_batch(request_path: Path, result_path: Path) -> int:
    operations = _validate_batch_request(request_path, result_path)
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
    _validate_batch_output(payload, operations)
    _write_batch_result(result_path, payload, operations)
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
