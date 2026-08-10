from __future__ import annotations

import base64
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

import image_to_ppt
from scripts import text_detect
from scripts.visual_segment import MaskCandidate, VisualElement


def test_isolated_ocr_runs_detection_then_recognition_workers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    calls = []
    poly = [[4, 2], [14, 5], [11, 11], [1, 8]]

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        result_path = Path(command[command.index("--result") + 1])
        if command[2] == "detect":
            result_path.write_text(
                json.dumps({"polys": [poly], "crops": ["crop-000.png"]}),
                encoding="utf-8",
            )
        else:
            result_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {"poly": poly, "text": "Title", "score": 0.99}
                        ]
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        text_detect,
        "run_isolated_worker",
        fake_run,
        raising=False,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("OCR must use the shared runner"),
    )

    boxes = text_detect._try_isolated_paddleocr(
        image_path,
        "en",
        0.7,
        worker_root=tmp_path,
    )

    worker_path = Path(text_detect.__file__).with_name("ocr_worker.py").resolve()
    assert [command[:3] for command, _ in calls] == [
        [sys.executable, str(worker_path), "detect"],
        [sys.executable, str(worker_path), "recognize"],
    ]
    assert "--lang" not in calls[0][0]
    assert calls[1][0][calls[1][0].index("--lang") + 1] == "en"
    assert all(
        kwargs == {"capture_output": True, "text": True, "check": False}
        for _, kwargs in calls
    )
    assert boxes == [
        {
            "box": (1, 2, 13, 9),
            "text": "Title",
            "confidence": 0.99,
        }
    ]
    assert not any(path.name.startswith("ocr-") for path in tmp_path.iterdir())


def test_isolated_ocr_batch_uses_one_worker_for_multiple_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (20, 10), "white").save(first)
    Image.new("RGB", (30, 12), "white").save(second)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        manifest_path = Path(command[command.index("--manifest") + 1])
        result_path = Path(command[command.index("--result") + 1])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result_path.write_text(
            json.dumps({
                "images": [
                    {
                        "path": path,
                        "items": [{
                            "poly": [[1, 2], [9, 2], [9, 7], [1, 7]],
                            "text": f"T{index}",
                            "score": 0.99,
                        }],
                    }
                    for index, path in enumerate(manifest["images"], start=1)
                ],
            }),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(text_detect, "run_isolated_worker", fake_run)

    results = text_detect.detect_text_batch(
        [first, second], lang="en", isolated=True, worker_root=tmp_path,
    )

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[2] == "batch"
    assert command[command.index("--lang") + 1] == "en"
    assert kwargs == {"capture_output": True, "text": True, "check": False}
    assert [[item["text"] for item in items] for items, _ in results] == [
        ["T1"], ["T2"],
    ]
    assert [mask.shape for _, mask in results] == [(10, 20), (12, 30)]
    assert not any(path.name.startswith("ocr-batch-") for path in tmp_path.iterdir())


def test_worker_sorts_crops_and_maps_recognition_to_sorted_polys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import ocr_worker

    image_path = tmp_path / "source.png"
    Image.new("RGB", (30, 20), "red").save(image_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    detection_result = work_dir / "detection.json"
    recognition_result = work_dir / "recognition.json"
    top = np.array([[1, 1], [11, 1], [11, 6], [1, 6]])
    bottom = np.array([[2, 10], [22, 10], [22, 15], [2, 15]])
    captured: dict[str, object] = {}

    class FakeDetector:
        def __init__(self, **kwargs):
            captured["detector_kwargs"] = kwargs

        def predict(self, image, **kwargs):
            captured["detection_predict_kwargs"] = kwargs
            return [{"dt_polys": [bottom, top]}]

        def close(self):
            pass

    class FakeSorter:
        def __call__(self, polys):
            return [top, bottom]

    class FakeCropper:
        def __init__(self, det_box_type):
            assert det_box_type == "quad"

        def __call__(self, image, polys):
            assert image[0, 0].tolist() == [0, 0, 255]
            assert np.array_equal(polys[0], top)
            return [
                np.zeros((10, 10, 3), dtype=np.uint8),
                np.zeros((5, 20, 3), dtype=np.uint8),
            ]

    class FakeRecognizer:
        def __init__(self, **kwargs):
            captured["recognizer_kwargs"] = kwargs

        def predict(self, crops):
            captured["recognition_ratios"] = [
                crop.shape[1] / crop.shape[0] for crop in crops
            ]
            return [
                {"rec_text": "square", "rec_score": 0.91},
                {"rec_text": "wide", "rec_score": 0.92},
            ]

        def close(self):
            pass

    monkeypatch.setattr(
        ocr_worker,
        "_load_detection_tools",
        lambda: (FakeDetector, FakeSorter, FakeCropper),
    )
    monkeypatch.setattr(
        ocr_worker,
        "_load_recognition_model",
        lambda: FakeRecognizer,
    )
    monkeypatch.setattr(
        ocr_worker,
        "_resolve_recognition_model_name",
        lambda lang: f"{lang}_model",
    )

    ocr_worker.run_detection(image_path, work_dir, detection_result)
    ocr_worker.run_recognition(
        detection_result,
        str(recognition_result),
        lang="en",
    )

    assert captured["detector_kwargs"] == {
        "model_name": "PP-OCRv5_mobile_det",
        "cpu_threads": 1,
        "enable_mkldnn": False,
        "limit_side_len": 64,
        "limit_type": "min",
        "thresh": 0.3,
        "box_thresh": 0.6,
        "unclip_ratio": 1.5,
    }
    assert captured["detection_predict_kwargs"] == {"max_side_limit": 4000}
    assert captured["recognizer_kwargs"] == {
        "model_name": "en_model",
        "cpu_threads": 1,
        "enable_mkldnn": False,
    }
    assert captured["recognition_ratios"] == [1.0, 4.0]
    assert json.loads(recognition_result.read_text(encoding="utf-8")) == {
        "items": [
            {"poly": top.tolist(), "text": "square", "score": 0.91},
            {"poly": bottom.tolist(), "text": "wide", "score": 0.92},
        ]
    }


def test_batch_worker_loads_each_model_once_for_multiple_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import ocr_worker

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (20, 10), "white").save(first)
    Image.new("RGB", (20, 10), "white").save(second)
    result_path = tmp_path / "result.json"
    poly = np.array([[1, 1], [9, 1], [9, 6], [1, 6]])
    loads = {"detector": 0, "recognizer": 0}

    class FakeDetector:
        def __init__(self, **kwargs):
            loads["detector"] += 1

        def predict(self, image, **kwargs):
            return [{"dt_polys": [poly]}]

        def close(self):
            pass

    class FakeSorter:
        def __call__(self, polys):
            return list(polys)

    class FakeCropper:
        def __init__(self, det_box_type):
            pass

        def __call__(self, image, polys):
            return [np.zeros((5, 8, 3), dtype=np.uint8)]

    class FakeRecognizer:
        def __init__(self, **kwargs):
            loads["recognizer"] += 1

        def predict(self, crops):
            return [
                {"rec_text": f"T{index}", "rec_score": 0.99}
                for index, _ in enumerate(crops, start=1)
            ]

        def close(self):
            pass

    monkeypatch.setattr(
        ocr_worker, "_load_detection_tools",
        lambda: (FakeDetector, FakeSorter, FakeCropper),
    )
    monkeypatch.setattr(
        ocr_worker, "_load_recognition_model", lambda: FakeRecognizer,
    )
    monkeypatch.setattr(
        ocr_worker, "_resolve_recognition_model_name", lambda lang: "model",
    )

    ocr_worker.run_batch([first, second], result_path, lang="en")

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert loads == {"detector": 1, "recognizer": 1}
    assert [item["items"][0]["text"] for item in payload["images"]] == [
        "T1", "T2",
    ]


def test_recognition_worker_skips_model_for_empty_detection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import ocr_worker

    detection_result = tmp_path / "detection.json"
    recognition_result = tmp_path / "recognition.json"
    detection_result.write_text(
        json.dumps({"polys": [], "crops": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ocr_worker,
        "_load_recognition_model",
        lambda: (_ for _ in ()).throw(
            AssertionError("recognition model must not load")
        ),
    )

    ocr_worker.run_recognition(detection_result, recognition_result)

    assert json.loads(recognition_result.read_text(encoding="utf-8")) == {
        "items": []
    }


def test_recognition_model_name_uses_paddleocr_language_mapping(
    monkeypatch,
) -> None:
    from scripts import ocr_worker

    class FakePaddleOCR:
        def _get_ocr_model_names(self, lang, version):
            assert self is None
            assert lang == "en"
            assert version is None
            return "PP-OCRv5_server_det", "en_PP-OCRv5_mobile_rec"

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        type("PaddleModule", (), {"PaddleOCR": FakePaddleOCR}),
    )

    assert (
        ocr_worker._resolve_recognition_model_name("en")
        == "en_PP-OCRv5_mobile_rec"
    )


def test_text_box_covers_every_quad_point() -> None:
    poly = [[4, 2], [14, 5], [11, 11], [1, 8]]

    assert text_detect._poly_to_box(poly) == (1, 2, 13, 9)


def test_filter_noise_keeps_high_confidence_technical_labels() -> None:
    labels = ["WSL/WSL2", "API-V2", "GPU_0", "VS"]
    boxes = [
        {"box": (0, index * 20, 80, 16), "text": label, "confidence": 0.95}
        for index, label in enumerate(labels)
    ]

    assert [box["text"] for box in text_detect._filter_noise(boxes)] == labels


def test_filter_noise_rejects_malformed_or_low_confidence_labels() -> None:
    boxes = [
        {"box": (0, 0, 120, 16), "text": "MCOULE ST:SETMP", "confidence": 0.95},
        {"box": (0, 20, 80, 16), "text": "GPU__0", "confidence": 0.95},
        {"box": (0, 40, 30, 16), "text": "A-B", "confidence": 0.95},
        {"box": (0, 60, 50, 16), "text": "API-", "confidence": 0.95},
        {"box": (0, 80, 80, 16), "text": "API-V2", "confidence": 0.4},
    ]

    assert text_detect._filter_noise(boxes) == []


def test_isolated_worker_failure_falls_back_to_tesseract(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"not-used")
    expected = [{"box": (1, 2, 3, 4), "text": "fallback", "confidence": 0.9}]
    monkeypatch.setattr(
        text_detect,
        "run_isolated_worker",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            "",
            "worker exploded",
        ),
    )
    monkeypatch.setattr(
        text_detect,
        "_try_tesseract",
        lambda *args, **kwargs: expected,
    )

    actual = text_detect._ocr_detect(
        image_path,
        "en",
        0.7,
        isolated=True,
        worker_root=tmp_path,
    )

    assert actual is expected
    assert "worker exploded" in caplog.text


def test_in_process_paddleocr_uses_single_thread_and_batch(
    monkeypatch,
) -> None:
    captured = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        type("PaddleModule", (), {"PaddleOCR": FakePaddleOCR}),
    )
    monkeypatch.setattr(text_detect, "_patch_paddle_mkldnn", lambda: None)

    text_detect._create_paddleocr("en")

    assert captured["cpu_threads"] == 1
    assert captured["enable_mkldnn"] is False
    assert captured["text_recognition_batch_size"] == 1


def test_prepare_isolation_reaches_source_and_final_page_ocr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    work_root = tmp_path / "work"
    source_calls = []
    final_calls = []
    events = []

    def fake_detect(path, lang, **kwargs):
        source_calls.append((Path(path), kwargs))
        return (
            [{"box": (1, 1, 5, 3), "text": "Title"}],
            np.zeros((10, 20), dtype=np.uint8),
        )

    def fake_process_isolated(path, work_dir, lang, text_analysis):
        events.append("visual")
        assert Path(text_analysis["text_clean_path"]).is_file()
        return {
            "original_image_path": str(path),
            "_work_dir": str(work_dir),
            "components": [],
        }

    def fake_finalize(slide_data, lang, _resource_isolation=False):
        final_calls.append(
            (
                Path(slide_data["_work_dir"]),
                _resource_isolation,
            )
        )
        return slide_data

    def fake_detector(*, device=None):
        events.append(f"dino:{device}")
        return object()

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)
    monkeypatch.setattr(image_to_ppt, "create_object_detector", fake_detector)
    monkeypatch.setattr(
        image_to_ppt,
        "create_sam_generator",
        lambda checkpoint, resource_safe=False: (
            events.append(f"sam:{resource_safe}") or object()
        ),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "resolve_sam_checkpoint",
        lambda: Path("sam.pt"),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "_process_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("parent process must not run visual processing")
        ),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "_process_image_isolated",
        fake_process_isolated,
        raising=False,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "_release_visual_resources",
        lambda: events.append("release"),
    )
    monkeypatch.setattr(image_to_ppt, "_finalize_slide_quality", fake_finalize)

    image_to_ppt._prepare_multiple_images(
        [image_path],
        "en",
        _work_root=work_root,
        _resource_isolation=True,
    )

    page_work = work_root.resolve() / "page_001"
    assert source_calls == [
        (
            image_path,
            {"isolated": True, "worker_root": page_work},
        )
    ]
    assert final_calls == [(page_work, True)]
    assert events == [
        "visual",
        "release",
    ]


def test_isolated_visual_processing_runs_page_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    text_analysis = {
        "items": [{"box": [1, 2, 3, 4], "text": "Title"}],
        "mask_path": str(tmp_path / "mask.png"),
    }
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        request_path = Path(command[command.index("--request") + 1])
        result_path = Path(command[command.index("--result") + 1])
        assert json.loads(request_path.read_text(encoding="utf-8")) == {
            "text_analysis": text_analysis
        }
        result_path.write_text(
            json.dumps(
                {
                    "original_image_path": str(image_path),
                    "_work_dir": str(work_dir),
                    "components": [],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        image_to_ppt,
        "run_isolated_worker",
        fake_run,
        raising=False,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("visual must use the shared runner"),
    )

    actual = image_to_ppt._process_image_isolated(
        image_path,
        work_dir,
        "ch",
        text_analysis,
    )

    worker_path = (
        Path(image_to_ppt.__file__).resolve().parent
        / "scripts"
        / "visual_worker.py"
    )
    command, kwargs = calls[0]
    assert command[:2] == [sys.executable, str(worker_path)]
    assert command[command.index("--image") + 1] == str(image_path)
    assert command[command.index("--work-dir") + 1] == str(work_dir)
    assert command[command.index("--lang") + 1] == "ch"
    assert kwargs == {"capture_output": True, "text": True}
    assert actual["components"] == []


def test_prepare_component_layers_keeps_isolated_worker_assets_in_work_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    work_dir = tmp_path / "prepared"
    events = []

    def fake_detect(path, *, lang, **kwargs):
        events.append(("ocr", Path(path), kwargs))
        return [], np.zeros((10, 20), dtype=np.uint8)

    def fake_process(path, target, lang, text_analysis):
        events.append(("visual", Path(path), dict(text_analysis)))
        component_dir = target / "components"
        mask_dir = target / "element-masks"
        semantic_mask_dir = target / "semantic-masks"
        component_dir.mkdir()
        mask_dir.mkdir()
        semantic_mask_dir.mkdir()
        component_path = component_dir / "component.png"
        element_mask_path = mask_dir / "0000.png"
        semantic_mask_path = semantic_mask_dir / "0000.png"
        background_path = target / "background-original.png"
        Image.new("RGBA", (3, 3), "red").save(component_path)
        element_mask = np.zeros((10, 20), dtype=np.uint8)
        element_mask[2:5, 2:5] = 255
        semantic_mask = np.zeros((10, 20), dtype=np.uint8)
        semantic_mask[1:6, 1:6] = 255
        Image.fromarray(element_mask, mode="L").save(element_mask_path)
        Image.fromarray(semantic_mask, mode="L").save(semantic_mask_path)
        Image.new("RGB", (20, 10), "white").save(background_path)
        Image.new("L", (20, 10), 0).save(target / "background-removal-mask.png")
        Image.new("RGB", (20, 10), "black").save(
            target / "background-difference.png"
        )
        return {
            "background_path": str(background_path),
            "background_original_path": str(background_path),
            "background_widescreen_path": str(background_path),
            "components": [{
                "path": str(component_path),
                "x": 0,
                "y": 0,
                "w": 3,
                "h": 3,
                "area": 9,
                "z_index": 0,
            }],
            "text_items": [],
            "img_width": 20,
            "img_height": 10,
            "canvas_width": 20,
            "canvas_height": 10,
            "content_offset_x": 0,
            "content_offset_y": 0,
            "widescreen_background_method": "identity",
            "original_image_path": str(path),
            "_work_dir": str(target),
            "_text_mask_path": text_analysis["mask_path"],
            "_element_mask_paths": [str(element_mask_path)],
            "_semantic_mask_paths": [str(semantic_mask_path)],
        }

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(image_to_ppt, "close_ocr_engines", lambda: None)
    monkeypatch.setattr(image_to_ppt, "_process_image_isolated", fake_process)
    monkeypatch.setattr(
        image_to_ppt,
        "_process_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("parent process must not run visual processing")
        ),
    )
    monkeypatch.setattr(image_to_ppt, "_release_visual_resources", lambda: None)

    prepared = image_to_ppt.prepare_component_layers(
        image_path,
        work_dir,
        lang="en",
        resource_isolation=True,
    )

    owned_root = work_dir.resolve()
    assert prepared["_resource_isolation"] is True
    assert events[0] == (
        "ocr",
        owned_root / "source-image.png",
        {"isolated": True, "worker_root": owned_root},
    )
    assert events[1][0] == "visual"
    assert events[1][1].is_relative_to(owned_root)
    assert Path(events[1][2]["mask_path"]).is_relative_to(owned_root)
    manifest = json.loads(Path(prepared["state_path"]).read_text(encoding="utf-8"))
    for record in [
        manifest["assets"]["source_image"],
        manifest["assets"]["ocr_mask"],
        manifest["assets"]["element_masks"][0],
        manifest["assets"]["semantic_masks"][0],
        manifest["assets"]["background_original"],
        manifest["components"][0]["asset"],
    ]:
        path = Path(record["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert (owned_root / path).resolve().is_relative_to(owned_root)


def test_resource_safe_proposals_release_owned_detector(monkeypatch) -> None:
    events = []
    detector = object()

    monkeypatch.setattr(
        image_to_ppt,
        "create_object_detector",
        lambda: events.append("load-dino") or detector,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "generate_object_proposals",
        lambda image, actual_detector: (
            events.append(f"detect:{actual_detector is detector}") or ["box"]
        ),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "filter_text_overlapping_proposals",
        lambda proposals, text_mask: proposals,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "_release_visual_resources",
        lambda: events.append("release"),
    )

    proposals = image_to_ppt._generate_filtered_object_proposals(
        np.zeros((10, 20, 3), dtype=np.uint8),
        np.zeros((10, 20), dtype=np.uint8),
        detector=None,
    )

    assert proposals == ["box"]
    assert events == ["load-dino", "detect:True", "release"]


def test_resource_safe_proposals_run_object_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    text_mask = np.zeros((10, 20), dtype=np.uint8)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        image_path = Path(command[command.index("--image") + 1])
        mask_path = Path(command[command.index("--text-mask") + 1])
        result_path = Path(command[command.index("--result") + 1])
        assert np.asarray(Image.open(image_path)).shape == image.shape
        assert np.asarray(Image.open(mask_path)).shape == text_mask.shape
        result_path.write_text(
            json.dumps(
                [
                    {
                        "box_xyxy": [1.0, 2.0, 8.0, 9.0],
                        "score": 0.95,
                        "label": "icon",
                        "role": "object",
                        "source": "full",
                        "crop_box": [0, 0, 20, 10],
                        "touches_crop_edge": False,
                    }
                ]
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        image_to_ppt,
        "run_isolated_worker",
        fake_run,
        raising=False,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("object must use the shared runner"),
    )

    proposals = image_to_ppt._generate_filtered_object_proposals_isolated(
        image,
        text_mask,
        tmp_path,
    )

    worker_path = (
        Path(image_to_ppt.__file__).resolve().parent
        / "scripts"
        / "object_worker.py"
    )
    command, kwargs = calls[0]
    assert command[:2] == [sys.executable, str(worker_path)]
    assert kwargs == {"capture_output": True, "text": True}
    assert len(proposals) == 1
    assert proposals[0].box_xyxy == (1.0, 2.0, 8.0, 9.0)
    assert proposals[0].crop_box == (0, 0, 20, 10)
    assert not any(path.name.startswith("object-") for path in tmp_path.iterdir())


def test_resource_safe_sam_phases_run_separate_workers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    text_mask = np.zeros((10, 20), dtype=np.uint8)
    expected_mask = np.zeros((10, 20), dtype=bool)
    expected_mask[2:8, 4:16] = True
    proposal = image_to_ppt.ObjectProposal(
        box_xyxy=(4.0, 2.0, 16.0, 8.0),
        score=0.95,
        label="icon",
        role="object",
        source="full",
        crop_box=(0, 0, 20, 10),
    )
    calls = []
    events = []

    def fake_run(command, **kwargs):
        events.append("spawn")
        calls.append((command, kwargs))
        result_path = Path(command[command.index("--result") + 1])
        result_path.write_text(
            json.dumps(
                [
                    {
                        "mask": base64.b64encode(
                            np.packbits(expected_mask, axis=None)
                        ).decode("ascii"),
                        "mask_shape": [10, 20],
                        "score": 0.93,
                        "source": "sam",
                        "crop_box": [0, 0, 20, 10],
                        "touches_crop_edge": False,
                        "label": "icon",
                        "role": "object",
                        "object_box": [4.0, 2.0, 16.0, 8.0],
                    }
                ]
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        image_to_ppt,
        "run_isolated_worker",
        fake_run,
        raising=False,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("SAM must use the shared runner"),
    )

    prompted = image_to_ppt._generate_sam_candidates_isolated(
        image,
        text_mask,
        [proposal],
        tmp_path,
        mode="prompted",
    )
    automatic = image_to_ppt._generate_sam_candidates_isolated(
        image,
        None,
        None,
        tmp_path,
        mode="automatic",
    )

    assert len(calls) == 2
    assert events == ["spawn", "spawn"]
    assert [
        command[command.index("--mode") + 1] for command, _ in calls
    ] == ["prompted", "automatic"]
    assert "--proposals" in calls[0][0]
    assert "--text-mask" in calls[0][0]
    assert "--proposals" not in calls[1][0]
    assert "--text-mask" not in calls[1][0]
    assert all(
        kwargs == {"capture_output": True, "text": True}
        for _, kwargs in calls
    )
    assert np.array_equal(prompted[0].mask, expected_mask)
    assert np.array_equal(automatic[0].mask, expected_mask)
    assert prompted[0].crop_box == (0, 0, 20, 10)
    assert prompted[0].object_box == (4.0, 2.0, 16.0, 8.0)
    assert not any(path.name.startswith("sam-") for path in tmp_path.iterdir())


def test_resource_safe_hole_recheck_runs_separate_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:8, 4:16] = True
    updated_mask = mask.copy()
    updated_mask[4:6, 8:12] = False
    updated_semantic_mask = mask.copy()
    updated_semantic_mask[0:2, 0:3] = True
    element = VisualElement(
        mask=mask.copy(),
        z_index=0,
        score=0.95,
        source="sam",
        semantic_mask=mask.copy(),
        object_box=(4.0, 2.0, 16.0, 8.0),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        request_path = Path(command[command.index("--elements") + 1])
        result_path = Path(command[command.index("--result") + 1])
        records = json.loads(request_path.read_text(encoding="utf-8"))
        packed = np.frombuffer(
            base64.b64decode(records[0]["mask"]),
            dtype=np.uint8,
        )
        actual = np.unpackbits(
            packed,
            count=mask.size,
        ).reshape(mask.shape).astype(bool)
        assert np.array_equal(actual, mask)
        result_path.write_text(
            json.dumps([
                {
                    "mask": base64.b64encode(
                        np.packbits(updated_mask, axis=None)
                    ).decode("ascii"),
                    "mask_shape": list(updated_mask.shape),
                    "semantic_mask": base64.b64encode(
                        np.packbits(updated_semantic_mask, axis=None)
                    ).decode("ascii"),
                    "semantic_mask_shape": list(updated_semantic_mask.shape),
                }
            ]),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        image_to_ppt,
        "run_isolated_worker",
        fake_run,
        raising=False,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("recheck must use the shared runner"),
    )

    image_to_ppt._recheck_visual_element_holes_isolated(
        image,
        [element],
        tmp_path,
    )

    command, kwargs = calls[0]
    assert command[command.index("--mode") + 1] == "recheck"
    assert kwargs == {"capture_output": True, "text": True}
    assert np.array_equal(element.mask, updated_mask)
    assert np.array_equal(element.semantic_mask, updated_semantic_mask)
    assert not any(path.name.startswith("sam-recheck-") for path in tmp_path.iterdir())


def test_sam_worker_rechecks_element_holes(tmp_path: Path, monkeypatch) -> None:
    from scripts import sam_worker

    image_path = tmp_path / "source.png"
    result_path = tmp_path / "result.json"
    elements_path = tmp_path / "elements.json"
    Image.new("RGB", (20, 10), "white").save(image_path)
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:8, 4:16] = True
    records = [{
        "mask": base64.b64encode(
            np.packbits(mask, axis=None)
        ).decode("ascii"),
        "mask_shape": list(mask.shape),
        "semantic_mask": base64.b64encode(
            np.packbits(mask, axis=None)
        ).decode("ascii"),
        "semantic_mask_shape": list(mask.shape),
        "z_index": 0,
        "score": 0.95,
        "source": "sam",
        "object_box": [4.0, 2.0, 16.0, 8.0],
    }]
    elements_path.write_text(json.dumps(records), encoding="utf-8")
    generator = object()
    events = []

    def fake_recheck(image, elements, actual_generator):
        assert actual_generator is generator
        events.append("recheck")
        elements[0].mask[4:6, 8:12] = False
        elements[0].semantic_mask[0:2, 0:3] = True

    monkeypatch.setattr(
        sam_worker,
        "_load_tools",
        lambda: (
            image_to_ppt.ObjectProposal,
            lambda *args, **kwargs: generator,
            lambda *args, **kwargs: [],
            lambda *args, **kwargs: [],
            lambda: Path("sam.pt"),
            VisualElement,
            fake_recheck,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sam_worker.py",
            "--mode",
            "recheck",
            "--image",
            str(image_path),
            "--elements",
            str(elements_path),
            "--result",
            str(result_path),
        ],
    )

    assert sam_worker.main() == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    actual = np.unpackbits(
        np.frombuffer(base64.b64decode(result[0]["mask"]), dtype=np.uint8),
        count=mask.size,
    ).reshape(mask.shape).astype(bool)
    actual_semantic = np.unpackbits(
        np.frombuffer(
            base64.b64decode(result[0]["semantic_mask"]),
            dtype=np.uint8,
        ),
        count=mask.size,
    ).reshape(mask.shape).astype(bool)
    assert events == ["recheck"]
    assert not np.any(actual[4:6, 8:12])
    assert np.all(actual_semantic[0:2, 0:3])


def test_visual_masks_pack_and_restore_exactly() -> None:
    first_mask = np.zeros((10, 20), dtype=bool)
    first_mask[1:4, 2:8] = True
    second_mask = np.zeros((10, 20), dtype=bool)
    second_mask[5:9, 10:18] = True
    candidates = [
        MaskCandidate(first_mask.copy(), 0.9, "sam"),
        MaskCandidate(second_mask.copy(), 0.8, "sam"),
    ]
    element = VisualElement(
        mask=second_mask.copy(),
        z_index=0,
        score=0.8,
        source="sam",
        semantic_mask=candidates[0].mask,
    )
    references = [
        *((candidate, "mask") for candidate in candidates),
        (element, "mask"),
        (element, "semantic_mask"),
    ]

    packed = image_to_ppt._pack_mask_references(references)

    assert all(getattr(owner, name) is None for owner, name in references)
    assert sum(mask.nbytes for mask, _ in packed[0].values()) < (
        first_mask.nbytes * 3
    )

    image_to_ppt._restore_mask_references(packed)

    assert np.array_equal(candidates[0].mask, first_mask)
    assert np.array_equal(candidates[1].mask, second_mask)
    assert np.array_equal(element.mask, second_mask)
    assert element.semantic_mask is candidates[0].mask


def test_resource_safe_pipeline_isolates_all_lama_background_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = np.full((20, 30, 3), 40, dtype=np.uint8)
    image_path = tmp_path / "source.png"
    Image.fromarray(source).save(image_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    text_mask_path = work_dir / "source-text-mask.png"
    Image.fromarray(np.zeros(source.shape[:2], dtype=np.uint8)).save(
        text_mask_path
    )
    mask = np.zeros(source.shape[:2], dtype=bool)
    mask[3:17, 5:25] = True
    candidate = MaskCandidate(mask.copy(), 0.95, "sam")
    element = VisualElement(
        mask=mask.copy(),
        z_index=0,
        score=0.95,
        source="sam",
    )
    sam_calls = []
    background_calls = []
    isolated_calls = []

    monkeypatch.setattr(
        image_to_ppt,
        "_generate_filtered_object_proposals_isolated",
        lambda *args: [],
    )

    def fake_sam(*args, mode, **kwargs):
        sam_calls.append(mode)
        return [candidate] if len(sam_calls) == 1 else []

    monkeypatch.setattr(
        image_to_ppt,
        "_generate_sam_candidates_isolated",
        fake_sam,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "filter_prompt_free_candidates",
        lambda *args: [],
    )
    monkeypatch.setattr(
        image_to_ppt,
        "combine_residual_candidates",
        lambda **kwargs: ([], 1),
    )
    monkeypatch.setattr(
        image_to_ppt,
        "resolve_visual_elements",
        lambda *args: [element],
    )
    monkeypatch.setattr(
        image_to_ppt,
        "_recheck_visual_element_holes_isolated",
        lambda *args: None,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "_release_visual_resources",
        lambda: None,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "inpaint_large_mask",
        lambda *args: pytest.fail("in-process LaMa must not be used"),
    )

    def fake_isolated(input_path, mask_path, output_path):
        isolated_calls.append(
            (Path(input_path), Path(mask_path), Path(output_path))
        )
        Image.open(input_path).save(output_path)

    monkeypatch.setattr(
        image_to_ppt,
        "inpaint_large_mask_isolated",
        fake_isolated,
    )

    def fake_clean_background(
        image,
        element_masks,
        text_mask,
        large_inpainter=None,
        text_clean_image=None,
        text_restore_mask=None,
    ):
        assert callable(large_inpainter)
        assert text_clean_image is None
        assert text_restore_mask.shape == image.shape[:2]
        assert not np.any(text_restore_mask)
        background_calls.append("clean")
        return large_inpainter(image, np.ones(image.shape[:2], dtype=np.uint8))

    def fake_widescreen(background, large_inpainter=None):
        assert callable(large_inpainter)
        background_calls.append("widescreen")
        repaired = large_inpainter(
            background,
            np.ones(background.shape[:2], dtype=np.uint8),
        )
        return repaired, 0, 0, "identity"

    monkeypatch.setattr(
        image_to_ppt,
        "build_clean_background",
        fake_clean_background,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "build_widescreen_background",
        fake_widescreen,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "export_visual_components",
        lambda *args, **kwargs: [],
    )

    image_to_ppt._process_image(
        image_path,
        work_dir,
        object_detector=None,
        mask_generator=None,
        lang="en",
        text_analysis={"items": [], "mask_path": str(text_mask_path)},
        defer_quality=True,
        _resource_isolation=True,
    )

    assert sam_calls == ["prompted", "automatic", "prompted", "automatic"]
    assert background_calls == ["clean", "clean", "clean", "widescreen"]
    assert len(isolated_calls) == 4
    assert all(
        input_path.parent == mask_path.parent == output_path.parent
        for input_path, mask_path, output_path in isolated_calls
    )
    assert not any(path.name.startswith("lama-") for path in work_dir.iterdir())


def test_final_quality_uses_isolation_for_repair_ocr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = np.full((10, 20, 3), 40, dtype=np.uint8)
    image_path = tmp_path / "source.png"
    background_path = tmp_path / "background.png"
    text_mask_path = tmp_path / "text-mask.png"
    Image.fromarray(source).save(image_path)
    Image.fromarray(source).save(background_path)
    Image.fromarray(np.zeros((10, 20), dtype=np.uint8)).save(text_mask_path)
    calls = []

    def fake_detect(path, lang, **kwargs):
        calls.append((Path(path), kwargs))
        if len(calls) == 1:
            return (
                [{"box": [1, 1, 8, 4], "text": "x"}],
                np.zeros((10, 20), dtype=np.uint8),
            )
        return [], np.zeros((10, 20), dtype=np.uint8)

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(
        image_to_ppt,
        "repair_exported_component_text",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "visual_difference",
        lambda *args: {"mae": 0.0, "p95": 0.0},
    )
    monkeypatch.setattr(
        image_to_ppt,
        "write_segmentation_diagnostics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "require_visual_quality",
        lambda quality: None,
    )
    slide_data = {
        "background_original_path": str(background_path),
        "components": [],
        "original_image_path": str(image_path),
        "_work_dir": str(tmp_path),
        "_text_mask_path": str(text_mask_path),
        "_element_mask_paths": [],
    }

    image_to_ppt._finalize_slide_quality(
        slide_data,
        "en",
        _resource_isolation=True,
    )

    assert len(calls) == 2
    assert all(
        kwargs == {"isolated": True, "worker_root": tmp_path}
        for _, kwargs in calls
    )


def test_final_quality_repairs_residual_text_across_overlapping_components(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = np.full((10, 20, 3), 40, dtype=np.uint8)
    image_path = tmp_path / "source.png"
    background_path = tmp_path / "background.png"
    text_mask_path = tmp_path / "text-mask.png"
    Image.fromarray(source).save(image_path)
    Image.fromarray(source).save(background_path)
    Image.fromarray(np.zeros((10, 20), dtype=np.uint8)).save(text_mask_path)
    detected = [
        [{"box": [1, 1, 8, 4], "text": "first"}],
        [{"box": [2, 1, 6, 4], "text": "residual"}],
        [],
    ]
    repair_modes = []

    def fake_detect(path, lang, **kwargs):
        items = detected.pop(0)
        return items, np.zeros((10, 20), dtype=np.uint8)

    def fake_repair(*args, **kwargs):
        repair_modes.append(
            "clear"
            if kwargs.get("clear_alpha")
            else "box" if "cleaned_rgb" in kwargs else "ink"
        )

    monkeypatch.setattr(image_to_ppt, "detect_text", fake_detect)
    monkeypatch.setattr(
        image_to_ppt,
        "repair_exported_component_text",
        fake_repair,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "visual_difference",
        lambda *args: {"mae": 0.0, "p95": 0.0},
    )
    monkeypatch.setattr(
        image_to_ppt,
        "write_segmentation_diagnostics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        image_to_ppt,
        "require_visual_quality",
        lambda quality: None,
    )
    slide_data = {
        "background_original_path": str(background_path),
        "components": [],
        "original_image_path": str(image_path),
        "_work_dir": str(tmp_path),
        "_text_mask_path": str(text_mask_path),
        "_element_mask_paths": [],
    }

    image_to_ppt._finalize_slide_quality(
        slide_data,
        "en",
        _resource_isolation=True,
    )

    assert repair_modes == ["box", "clear"]
    assert detected == []


def test_clear_alpha_repairs_low_contrast_transparent_text(
    tmp_path: Path,
) -> None:
    component_path = tmp_path / "component.png"
    rgba = np.zeros((10, 20, 4), dtype=np.uint8)
    rgba[2:5, 4:12, :3] = 254
    rgba[2:5, 4:12, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(component_path)
    text_mask = np.zeros((10, 20), dtype=np.uint8)
    text_mask[1:7, 2:14] = 255
    source = np.full((10, 20, 3), 250, dtype=np.uint8)

    image_to_ppt.repair_exported_component_text(
        [{"path": str(component_path), "x": 0, "y": 0}],
        text_mask,
        source,
        text_items=[{"box": [2, 1, 12, 6], "text": "residual"}],
        clear_alpha=True,
    )

    repaired = np.asarray(Image.open(component_path).convert("RGBA"))
    assert np.count_nonzero(repaired[:, :, 3]) == 0


def test_ocr_worker_sets_omp_before_lazy_paddle_import_without_torch() -> None:
    from scripts import ocr_worker

    source = Path(ocr_worker.__file__).read_text(encoding="utf-8")
    assert source.index('os.environ["OMP_NUM_THREADS"] = "1"') < source.index(
        "from paddleocr import TextDetection"
    )
    assert source.index(
        'os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"'
    ) < source.index("from paddleocr import TextDetection")
    assert "import torch" not in source


def test_ocr_product_and_skill_mirrors_match() -> None:
    root = Path(__file__).resolve().parents[1]
    script_names = [
        "bg_model.py",
        "fg_extract.py",
        "lama_inpaint.py",
        "lama_worker.py",
        "object_worker.py",
        "ocr_worker.py",
        "sam_worker.py",
        "text_detect.py",
        "visual_segment.py",
        "visual_worker.py",
        "worker_resources.py",
    ]
    pairs = [
        *[
            (
                root / "scripts" / script_name,
                root
                / "skills"
                / "image-to-ppt"
                / "scripts"
                / script_name,
            )
            for script_name in script_names
        ],
        (
            root / "image_to_ppt.py",
            root / "skills" / "image-to-ppt" / "scripts" / "image_to_ppt.py",
        ),
    ]
    for product, mirror in pairs:
        assert hashlib.sha256(product.read_bytes()).digest() == hashlib.sha256(
            mirror.read_bytes()
        ).digest()


def test_root_and_skill_workers_load_outside_repository(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    worker_roots = [
        root / "scripts",
        root / "skills" / "image-to-ppt" / "scripts",
    ]
    for worker_root in worker_roots:
        for worker_name in (
            "lama_worker.py",
            "object_worker.py",
            "ocr_worker.py",
            "sam_worker.py",
            "visual_worker.py",
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(worker_root / worker_name),
                    "--help",
                ],
                cwd=tmp_path,
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, completed.stderr


def test_root_and_skill_sam_worker_loads_local_tools_outside_repository(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    external_scripts = tmp_path / "scripts"
    external_scripts.mkdir()
    (external_scripts / "__init__.py").write_text(
        "SOURCE = 'external'\n",
        encoding="utf-8",
    )
    worker_roots = [
        root / "scripts",
        root / "skills" / "image-to-ppt" / "scripts",
    ]
    probe = (
        "import pathlib, sys; "
        "worker = pathlib.Path(sys.argv[1]).resolve(); "
        "local_root = str(worker.parent); "
        "sys.path.insert(0, local_root); "
        "sys.path.insert(0, str(pathlib.Path.cwd())); "
        "sys.path.insert(0, str(worker)); "
        "import sam_worker; "
        "sam_worker._load_tools(); "
        "import scripts.worker_resources as resources; "
        "expected = pathlib.Path(worker, 'worker_resources.py').resolve(); "
        "assert pathlib.Path(resources.__file__).resolve() == expected"
    )
    for worker_root in worker_roots:
        completed = subprocess.run(
            [sys.executable, "-c", probe, str(worker_root)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_root_and_skill_worker_modules_prefer_local_resources_outside_repository(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    external_scripts = tmp_path / "scripts"
    external_scripts.mkdir()
    (external_scripts / "__init__.py").write_text(
        "SOURCE = 'external'\n",
        encoding="utf-8",
    )
    worker_roots = [
        root / "scripts",
        root / "skills" / "image-to-ppt" / "scripts",
    ]
    probe = (
        "import importlib, pathlib, sys; "
        "worker = pathlib.Path(sys.argv[1]).resolve(); "
        "local_root = str(worker.parent); "
        "sys.path.insert(0, local_root); "
        "sys.path.insert(0, str(pathlib.Path.cwd())); "
        "sys.path.insert(0, str(worker)); "
        "importlib.import_module(sys.argv[2]); "
        "import scripts.worker_resources as resources; "
        "expected = pathlib.Path(worker, 'worker_resources.py').resolve(); "
        "assert pathlib.Path(resources.__file__).resolve() == expected"
    )
    for worker_root in worker_roots:
        for module_name in ("text_detect", "lama_inpaint"):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    probe,
                    str(worker_root),
                    module_name,
                ],
                cwd=tmp_path,
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, completed.stderr
