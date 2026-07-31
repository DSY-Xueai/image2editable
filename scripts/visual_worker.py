from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _load_process_image():
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir.parent))
    if (script_dir / "image_to_ppt.py").is_file():
        from scripts.image_to_ppt import _process_image
    else:
        from image_to_ppt import _process_image

    return _process_image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    process_image = _load_process_image()
    slide_data = process_image(
        Path(args.image),
        Path(args.work_dir),
        None,
        None,
        args.lang,
        text_analysis=request["text_analysis"],
        defer_quality=True,
        _resource_isolation=True,
    )
    result_path = Path(args.result)
    temporary_path = result_path.with_name(f".{result_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(slide_data, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
