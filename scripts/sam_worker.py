from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def _load_tools():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prompted", "automatic", "recheck"),
        required=True,
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--text-mask")
    parser.add_argument("--proposals")
    parser.add_argument("--elements")
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
    else:
        candidates = generate_automatic(
            image,
            generator,
            crop_size=max(image.shape[:2]),
            include_geometry=False,
            min_score=0.90,
        )
        output_records = [_candidate_record(candidate) for candidate in candidates]

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
