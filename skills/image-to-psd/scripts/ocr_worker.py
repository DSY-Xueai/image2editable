from __future__ import annotations

import os


def _cpu_threads() -> int:
    for name in ("FLAGS_paddle_num_threads", "OMP_NUM_THREADS"):
        try:
            value = int(os.environ.get(name, ""))
        except ValueError:
            continue
        if 1 <= value <= 8:
            return value
    return min(8, max(1, (os.cpu_count() or 1) // 2))


os.environ.setdefault("OMP_NUM_THREADS", str(_cpu_threads()))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import argparse
from contextlib import redirect_stdout
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


class _ResidentOcrProcessor:
    """Keep PaddleOCR detector and recognizers alive for one task."""

    def __init__(self) -> None:
        self._detector = None
        self._sorter_type = None
        self._cropper_type = None
        self._recognizers: dict[str, object] = {}

    def detection_tools(self):
        if self._detector is None:
            detector_type, self._sorter_type, self._cropper_type = (
                _load_detection_tools()
            )
            self._detector = detector_type(
                model_name="PP-OCRv5_mobile_det",
                cpu_threads=_cpu_threads(),
                enable_mkldnn=False,
                limit_side_len=64,
                limit_type="min",
                thresh=0.3,
                box_thresh=0.6,
                unclip_ratio=1.5,
            )
        return self._detector, self._sorter_type, self._cropper_type

    def recognizer(self, lang: str):
        recognizer = self._recognizers.get(lang)
        if recognizer is None:
            recognizer_type = _load_recognition_model()
            recognizer = recognizer_type(
                model_name=_resolve_recognition_model_name(lang),
                cpu_threads=_cpu_threads(),
                enable_mkldnn=False,
            )
            self._recognizers[lang] = recognizer
        return recognizer

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        for recognizer in self._recognizers.values():
            recognizer.close()
        self._recognizers.clear()


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
        cpu_threads=_cpu_threads(),
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
        cpu_threads=_cpu_threads(),
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
    *,
    processor: _ResidentOcrProcessor | None = None,
) -> None:
    if processor is None:
        detector_type, sorter_type, cropper_type = _load_detection_tools()
        detector = detector_type(
            model_name="PP-OCRv5_mobile_det",
            cpu_threads=_cpu_threads(),
            enable_mkldnn=False,
            limit_side_len=64,
            limit_type="min",
            thresh=0.3,
            box_thresh=0.6,
            unclip_ratio=1.5,
        )
    else:
        detector, sorter_type, cropper_type = processor.detection_tools()
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
        if processor is None:
            detector.close()

    recognized = [None] * len(all_crops)
    if all_crops:
        order = sorted(
            range(len(all_crops)),
            key=lambda index: all_crops[index].shape[1] / all_crops[index].shape[0],
        )
        if processor is None:
            recognizer_type = _load_recognition_model()
            recognizer = recognizer_type(
                model_name=_resolve_recognition_model_name(lang),
                cpu_threads=_cpu_threads(),
                enable_mkldnn=False,
            )
        else:
            recognizer = processor.recognizer(lang)
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
            if processor is None:
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
    parser.add_argument("--serve", action="store_true")
    subparsers = parser.add_subparsers(dest="mode")
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


def _serve() -> int:
    processor = _ResidentOcrProcessor()
    try:
        for line in sys.stdin:
            request_id = None
            try:
                envelope = json.loads(line)
                if envelope == {"control": "close"}:
                    break
                if not isinstance(envelope, dict):
                    raise ValueError("OCR worker envelope is invalid")
                request_id = envelope.get("request_id")
                payload = envelope.get("payload")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("OCR worker request id is invalid")
                if not isinstance(payload, dict) or set(payload) != {
                    "images", "result", "lang",
                }:
                    raise ValueError("OCR worker payload is invalid")
                if (
                    not isinstance(payload["images"], list)
                    or not payload["images"]
                    or any(not isinstance(path, str) or not path for path in payload["images"])
                    or not isinstance(payload["result"], str)
                    or not isinstance(payload["lang"], str)
                ):
                    raise ValueError("OCR worker payload is invalid")
                with redirect_stdout(sys.stderr):
                    run_batch(
                        payload["images"],
                        payload["result"],
                        payload["lang"],
                        processor=processor,
                    )
                response = {"request_id": request_id, "result": {}}
            except Exception as error:
                response = {
                    "request_id": request_id,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        processor.close()
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.serve:
        return _serve()
    if args.mode is None:
        parser.error("OCR worker requires a mode or --serve")
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
