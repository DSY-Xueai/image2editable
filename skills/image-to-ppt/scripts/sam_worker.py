from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

_MODULE_ROOT = str(Path(__file__).resolve().parent.parent)
if _MODULE_ROOT not in sys.path:
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
        choices=("prompted", "automatic", "recheck", "component"),
        required=True,
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--text-mask")
    parser.add_argument("--proposals")
    parser.add_argument("--elements")
    parser.add_argument("--prompt")
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

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
