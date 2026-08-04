from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def _load_detection_tools():
    from paddleocr import TextDetection
    from paddlex.inference.pipelines.components import (
        CropByPolys,
        SortQuadBoxes,
    )

    return TextDetection, SortQuadBoxes, CropByPolys


def _load_recognition_model():
    from paddleocr import TextRecognition

    return TextRecognition


def _resolve_recognition_model_name(lang: str) -> str:
    from paddleocr import PaddleOCR

    _, model_name = PaddleOCR._get_ocr_model_names(None, lang, None)
    if model_name is None:
        raise ValueError(f"No PaddleOCR recognition model for language: {lang}")
    return model_name


def _value(result: object, name: str, default: object) -> object:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise RuntimeError(f"Cannot read OCR image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Cannot encode OCR crop: {path}")
    encoded.tofile(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_detection(
    image_path: str | Path,
    work_dir: str | Path,
    result_path: str | Path,
) -> None:
    image_path = Path(image_path)
    work_dir = Path(work_dir)
    result_path = Path(result_path)
    detector_type, sorter_type, cropper_type = _load_detection_tools()
    detector = detector_type(
        model_name="PP-OCRv5_mobile_det",
        cpu_threads=1,
        enable_mkldnn=False,
        limit_side_len=64,
        limit_type="min",
        thresh=0.3,
        box_thresh=0.6,
        unclip_ratio=1.5,
    )
    try:
        results = detector.predict(
            str(image_path),
            max_side_limit=4000,
        )
    finally:
        detector.close()
    result = results[0] if results else {}
    polys = list(sorter_type()(_value(result, "dt_polys", [])))
    image = _read_bgr(image_path)
    crops = cropper_type(det_box_type="quad")(
        image,
        polys,
    )
    saved_polys = []
    crop_paths = []
    for index, (crop, poly) in enumerate(zip(crops, polys)):
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        crop_path = (work_dir / f"crop-{index:04d}.png").resolve()
        _write_image(crop_path, crop)
        saved_polys.append(np.asarray(poly).tolist())
        crop_paths.append(str(crop_path))
    _write_json(
        result_path,
        {"polys": saved_polys, "crops": crop_paths},
    )


def run_recognition(
    detection_result: str | Path,
    result_path: str | Path,
    lang: str = "ch",
) -> None:
    result_path = Path(result_path)
    detection = json.loads(
        Path(detection_result).read_text(encoding="utf-8")
    )
    polys = detection["polys"]
    crops = [_read_bgr(Path(path)) for path in detection["crops"]]
    if len(polys) != len(crops):
        raise RuntimeError("OCR detection crop count does not match polygons")
    if not crops:
        _write_json(result_path, {"items": []})
        return
    order = sorted(
        range(len(crops)),
        key=lambda index: crops[index].shape[1] / crops[index].shape[0],
    )
    recognizer_type = _load_recognition_model()
    recognizer = recognizer_type(
        model_name=_resolve_recognition_model_name(lang),
        cpu_threads=1,
        enable_mkldnn=False,
    )
    try:
        results = recognizer.predict([crops[index] for index in order])
    finally:
        recognizer.close()
    if len(results) != len(order):
        raise RuntimeError("OCR recognition result count does not match crops")
    mapped = [None] * len(order)
    for index, result in zip(order, results):
        mapped[index] = {
            "poly": polys[index],
            "text": str(_value(result, "rec_text", "")),
            "score": float(_value(result, "rec_score", 0.0)),
        }
    _write_json(result_path, {"items": mapped})


def run_batch(
    image_paths: list[str | Path],
    result_path: str | Path,
    lang: str = "ch",
) -> None:
    detector_type, sorter_type, cropper_type = _load_detection_tools()
    detector = detector_type(
        model_name="PP-OCRv5_mobile_det",
        cpu_threads=1,
        enable_mkldnn=False,
        limit_side_len=64,
        limit_type="min",
        thresh=0.3,
        box_thresh=0.6,
        unclip_ratio=1.5,
    )
    records = []
    all_crops = []
    try:
        for image_path in map(Path, image_paths):
            results = detector.predict(str(image_path), max_side_limit=4000)
            result = results[0] if results else {}
            polys = list(sorter_type()(_value(result, "dt_polys", [])))
            crops = cropper_type(det_box_type="quad")(
                _read_bgr(image_path), polys,
            )
            kept_polys = []
            crop_indices = []
            for crop, poly in zip(crops, polys):
                if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                    continue
                kept_polys.append(np.asarray(poly).tolist())
                crop_indices.append(len(all_crops))
                all_crops.append(crop)
            records.append({
                "path": str(image_path),
                "polys": kept_polys,
                "crop_indices": crop_indices,
            })
    finally:
        detector.close()

    recognized = [None] * len(all_crops)
    if all_crops:
        order = sorted(
            range(len(all_crops)),
            key=lambda index: all_crops[index].shape[1] / all_crops[index].shape[0],
        )
        recognizer_type = _load_recognition_model()
        recognizer = recognizer_type(
            model_name=_resolve_recognition_model_name(lang),
            cpu_threads=1,
            enable_mkldnn=False,
        )
        try:
            for start in range(0, len(order), 64):
                batch = order[start:start + 64]
                results = list(recognizer.predict([all_crops[index] for index in batch]))
                if len(results) != len(batch):
                    raise RuntimeError("OCR recognition result count does not match crops")
                for index, result in zip(batch, results):
                    recognized[index] = {
                        "text": str(_value(result, "rec_text", "")),
                        "score": float(_value(result, "rec_score", 0.0)),
                    }
        finally:
            recognizer.close()

    images = []
    for record in records:
        items = []
        for poly, crop_index in zip(record["polys"], record["crop_indices"]):
            item = recognized[crop_index]
            items.append({"poly": poly, **item})
        images.append({"path": record["path"], "items": items})
    _write_json(Path(result_path), {"images": images})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    detect = subparsers.add_parser("detect")
    detect.add_argument("--image", required=True)
    detect.add_argument("--work-dir", required=True)
    detect.add_argument("--result", required=True)
    recognize = subparsers.add_parser("recognize")
    recognize.add_argument("--detection-result", required=True)
    recognize.add_argument("--result", required=True)
    recognize.add_argument("--lang", default="ch")
    batch = subparsers.add_parser("batch")
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--result", required=True)
    batch.add_argument("--lang", default="ch")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if args.mode == "detect":
            run_detection(args.image, args.work_dir, args.result)
        elif args.mode == "recognize":
            run_recognition(args.detection_result, args.result, args.lang)
        else:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            run_batch(manifest["images"], args.result, args.lang)
    except Exception as error:
        print(f"OCR {args.mode} worker failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
