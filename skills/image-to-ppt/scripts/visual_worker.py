from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
from PIL import Image


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )


def _read_source_snapshot(
    image_path: Path,
    work_dir: Path,
    expected_size: int,
    label: str,
) -> bytes:
    root = Path(os.path.abspath(work_dir))
    source = Path(os.path.abspath(image_path))
    if source.parent != root:
        raise ValueError(f"{label} must be directly inside the work directory")
    root_before = os.lstat(root)
    path_before = os.lstat(source)
    try:
        with source.open("rb") as stream:
            handle_before = os.fstat(stream.fileno())
            if handle_before.st_size != expected_size:
                raise ValueError(f"{label} size mismatch")
            content = stream.read(expected_size + 1)
            handle_after = os.fstat(stream.fileno())
        path_after = os.lstat(source)
        root_after = os.lstat(root)
    except OSError as exc:
        raise ValueError(f"{label} changed while being read") from exc

    for status in (root_before, root_after):
        if _is_link_or_reparse(status) or not stat.S_ISDIR(status.st_mode):
            raise ValueError("visual work directory changed while being read")
    if (root_before.st_dev, root_before.st_ino) != (
        root_after.st_dev, root_after.st_ino
    ):
        raise ValueError("visual work directory changed while being read")
    statuses = (path_before, handle_before, handle_after, path_after)
    for status in statuses:
        if (
            _is_link_or_reparse(status)
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
        ):
            raise ValueError(f"{label} changed while being read")
    identities = {
        (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
        for status in statuses
    }
    if len(identities) != 1 or len(content) != expected_size:
        raise ValueError(f"{label} changed while being read")
    return content


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
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--request-size", required=True, type=int)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-size", required=True, type=int)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    if args.request_size <= 0:
        raise ValueError("visual request binding is invalid")
    request_content = _read_source_snapshot(
        Path(args.request), Path(args.work_dir), args.request_size, "visual request"
    )
    if hashlib.sha256(request_content).hexdigest() != args.request_sha256:
        raise ValueError("visual request sha256 mismatch")
    request = json.loads(request_content.decode("utf-8"))
    expected_sha256 = args.source_sha256
    expected_size = args.source_size
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
    ):
        raise ValueError("visual source binding is invalid")
    source_content = _read_source_snapshot(
        Path(args.image),
        Path(args.work_dir),
        expected_size,
        "visual source",
    )
    if hashlib.sha256(source_content).hexdigest() != expected_sha256:
        raise ValueError("visual source sha256 mismatch")
    with Image.open(io.BytesIO(source_content)) as stored_source:
        source_image = np.asarray(stored_source.convert("RGB")).copy()
    text_analysis = request["text_analysis"]
    text_mask_content = _read_source_snapshot(
        Path(text_analysis["mask_path"]),
        Path(args.work_dir),
        request["text_mask_size"],
        "visual text mask",
    )
    if hashlib.sha256(text_mask_content).hexdigest() != request["text_mask_sha256"]:
        raise ValueError("visual text mask sha256 mismatch")
    with Image.open(io.BytesIO(text_mask_content)) as stored_text_mask:
        text_mask = np.asarray(stored_text_mask.convert("L")).copy()
    text_clean_image = None
    if text_analysis.get("text_clean_path") is not None:
        text_clean_content = _read_source_snapshot(
            Path(text_analysis["text_clean_path"]),
            Path(args.work_dir),
            request["text_clean_size"],
            "visual text clean image",
        )
        if (
            hashlib.sha256(text_clean_content).hexdigest()
            != request["text_clean_sha256"]
        ):
            raise ValueError("visual text clean image sha256 mismatch")
        with Image.open(io.BytesIO(text_clean_content)) as stored_text_clean:
            text_clean_image = np.asarray(stored_text_clean.convert("RGB")).copy()
    process_image = _load_process_image()
    slide_data = process_image(
        Path(args.image),
        Path(args.work_dir),
        None,
        None,
        args.lang,
        text_analysis=text_analysis,
        defer_quality=True,
        _resource_isolation=True,
        _source_image=source_image,
        _text_mask=text_mask,
        _text_clean_image=text_clean_image,
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
