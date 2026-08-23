from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import importlib
import json


IMAGE_SIZE = 64
MODEL_NAMES = ["sam2_large", "grounding_dino", "big_lama"]


class RuntimeModelSmokeError(RuntimeError):
    pass


def _require_records(value: object, model_name: str) -> None:
    if not isinstance(value, list) or not all(
        isinstance(record, dict) for record in value
    ):
        raise RuntimeModelSmokeError(f"{model_name} returned invalid records")


def _run_records(model_name: str, operation) -> None:
    try:
        value = operation()
    except Exception:
        raise RuntimeModelSmokeError(f"{model_name} inference failed") from None
    _require_records(value, model_name)


def _run_smoke(
    np,
    *,
    resolve_sam_checkpoint,
    create_sam_generator,
    create_object_detector,
    inpaint_large_mask,
) -> dict[str, object]:
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[16:48, 16:48] = (255, 255, 255)
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[24:40, 24:40] = 255

    sam = create_sam_generator(
        resolve_sam_checkpoint(),
        device="cpu",
        resource_safe=True,
    )
    _run_records("sam2_large", lambda generator=sam: generator.generate(image))
    del sam

    detector = create_object_detector(device="cpu")
    _run_records(
        "grounding_dino",
        lambda object_detector=detector: object_detector.detect(
            image, "object.", 0.25, 0.25
        ),
    )
    del detector

    repaired = inpaint_large_mask(image, mask)
    if (
        not isinstance(repaired, np.ndarray)
        or repaired.shape != image.shape
        or repaired.dtype != np.uint8
    ):
        raise RuntimeModelSmokeError("big_lama returned an invalid image")
    return {"models": MODEL_NAMES, "ok": True}


def run_smoke() -> dict[str, object]:
    np = importlib.import_module("numpy")
    lama = importlib.import_module("scripts.lama_inpaint")
    detector = importlib.import_module("scripts.object_detect")
    segmentation = importlib.import_module("scripts.visual_segment")
    return _run_smoke(
        np,
        resolve_sam_checkpoint=segmentation.resolve_sam_checkpoint,
        create_sam_generator=segmentation.create_sam_generator,
        create_object_detector=detector.create_object_detector,
        inpaint_large_mask=lama.inpaint_large_mask,
    )


def main() -> int:
    captured = io.StringIO()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            result = run_smoke()
    except Exception:
        result = {"models": [], "ok": False}
        return_code = 1
    else:
        return_code = 0
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
