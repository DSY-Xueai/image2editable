from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def _load_object_tools():
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir.parent))
    from scripts.object_detect import (
        create_object_detector,
        filter_text_overlapping_proposals,
        generate_object_proposals,
    )

    return (
        create_object_detector,
        filter_text_overlapping_proposals,
        generate_object_proposals,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text-mask", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    with Image.open(args.image) as stored_image:
        image = np.asarray(stored_image.convert("RGB")).copy()
    with Image.open(args.text_mask) as stored_mask:
        text_mask = np.asarray(stored_mask.convert("L")).copy()
    create_detector, filter_proposals, generate_proposals = _load_object_tools()
    detector = create_detector()
    proposals = filter_proposals(
        generate_proposals(image, detector),
        text_mask,
    )
    result_path = Path(args.result)
    temporary_path = result_path.with_name(f".{result_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            [asdict(proposal) for proposal in proposals],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_path, result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
