from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from lama_inpaint import inpaint_large_mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Image.open(args.image) as stored_image:
        image = np.asarray(stored_image.convert("RGB")).copy()
    with Image.open(args.mask) as stored_mask:
        mask = np.asarray(stored_mask.convert("L")).copy()
    repaired = inpaint_large_mask(image, mask)
    Image.fromarray(repaired, mode="RGB").save(Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
