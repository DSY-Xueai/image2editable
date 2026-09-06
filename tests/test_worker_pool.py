from __future__ import annotations

import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import image_to_ppt
from image2editable.worker_pool import (
    JsonLineWorker,
    TaskWorkerPool,
    WorkerCancelled,
    WorkerDeadlineExceeded,
    WorkerPoolClosed,
    WorkerQueueFull,
    WorkerRemoteError,
)
from scripts.sam_worker import component_prompt_masks


class _FakeWorker:
    def __init__(self) -> None:
        self.requests = []
        self.closed = 0
        self.terminated = 0

    def request(self, envelope: dict, *, deadline: float | None) -> dict:
        self.requests.append((envelope, deadline))
        if envelope["payload"].get("fail"):
            return {
                "request_id": envelope["request_id"],
                "error": {"type": "RuntimeError", "message": "expected failure"},
            }
        return {
            "request_id": envelope["request_id"],
            "result": envelope["payload"],
        }

    def close(self) -> None:
        self.closed += 1

    def terminate(self) -> None:
        self.terminated += 1


def test_ordered_page_prompts_compute_full_image_embedding_once() -> None:
    class CountingPredictor:
        def __init__(self) -> None:
            self.set_image_calls = 0
            self.predict_calls = 0

        def set_image(self, image: np.ndarray) -> None:
            self.set_image_calls += 1

        def predict(self, **kwargs):
            self.predict_calls += 1
            masks = np.zeros((3, 128, 192), dtype=bool)
            masks[0, 10:30, 10:30] = True
            return masks, np.asarray([0.9, 0.2, 0.1]), None

    predictor = CountingPredictor()
    generator = type("Generator", (), {"predictor": predictor})()

    results = component_prompt_masks(
        generator,
        np.zeros((128, 192, 3), dtype=np.uint8),
        [
            {"box": [10, 10, 30, 30]},
            {"box": [80, 20, 120, 70]},
        ],
    )

    assert len(results) == 2
    assert predictor.set_image_calls == 1
    assert predictor.predict_calls == 2


def test_json_line_worker_forces_utf8_for_non_ascii_request_payload(
    tmp_path: Path,
) -> None:
    worker_script = tmp_path / "echo_worker.py"
    worker_script.write_text(
        "import json\n"
        "import sys\n"
        "for line in sys.stdin:\n"
        "    envelope = json.loads(line)\n"
        "    print(json.dumps({'request_id': envelope['request_id'], "
        "'result': {'path': envelope['payload']['path'], "
        "'stdin_encoding': sys.stdin.encoding}}, ensure_ascii=False), flush=True)\n",
        encoding="utf-8",
    )
    worker = JsonLineWorker([sys.executable, str(worker_script)])
    try:
        response = worker.request(
            {"request_id": "one", "payload": {"path": "C:/转换/页面.png"}},
            deadline=time.monotonic() + 5,
        )
    finally:
        worker.close()

    assert response == {
        "request_id": "one",
        "result": {"path": "C:/转换/页面.png", "stdin_encoding": "utf-8"},
    }


def test_json_line_worker_discards_unbounded_child_stderr(
    tmp_path: Path,
) -> None:
    worker_script = tmp_path / "noisy_worker.py"
    worker_script.write_text(
        "import json\n"
        "import sys\n"
        "for line in sys.stdin:\n"
        "    envelope = json.loads(line)\n"
        "    sys.stderr.write('x' * (1024 * 1024))\n"
        "    sys.stderr.flush()\n"
        "    print(json.dumps({'request_id': envelope['request_id'], 'result': {}}), flush=True)\n",
        encoding="utf-8",
    )
    worker = JsonLineWorker([sys.executable, str(worker_script)])
    try:
        response = worker.request(
            {"request_id": "one", "payload": {}},
            deadline=time.monotonic() + 5,
        )
    finally:
        worker.close()

    assert response == {"request_id": "one", "result": {}}


def test_task_worker_pool_reuses_one_worker_and_preserves_request_order() -> None:
    created = []

    def factory() -> _FakeWorker:
        worker = _FakeWorker()
        created.append(worker)
        return worker

    pool = TaskWorkerPool(factory, queue_limit=1, worker_name="visual")
    try:
        assert pool.request({"sequence": 1}) == {"sequence": 1}
        assert pool.request({"sequence": 2}) == {"sequence": 2}
    finally:
        pool.close()

    assert len(created) == 1
    assert [item[0]["request_id"] for item in created[0].requests] == [
        "visual-00000001",
        "visual-00000002",
    ]
    assert [item[0]["payload"] for item in created[0].requests] == [
        {"sequence": 1},
        {"sequence": 2},
    ]
    assert created[0].closed == 1


def test_task_worker_pool_applies_a_bounded_default_deadline() -> None:
    worker = _FakeWorker()
    pool = TaskWorkerPool(lambda: worker, worker_name="ocr")
    before = time.monotonic()
    try:
        assert pool.request({"sequence": 1}) == {"sequence": 1}
    finally:
        pool.close()
    after = time.monotonic()

    deadline = worker.requests[0][1]
    assert deadline is not None
    assert before + 299 <= deadline <= after + 301


def test_task_worker_pool_rebuilds_after_real_worker_deadline(
    tmp_path: Path,
) -> None:
    worker_script = tmp_path / "blocking_worker.py"
    worker_script.write_text(
        "import json\n"
        "import sys\n"
        "import time\n"
        "for line in sys.stdin:\n"
        "    envelope = json.loads(line)\n"
        "    if envelope == {'control': 'close'}:\n"
        "        break\n"
        "    if envelope['payload'].get('block'):\n"
        "        time.sleep(2)\n"
        "    print(json.dumps({'request_id': envelope['request_id'], 'result': envelope['payload']}), flush=True)\n",
        encoding="utf-8",
    )
    pool = TaskWorkerPool(
        lambda: JsonLineWorker([sys.executable, str(worker_script)]),
        worker_name="visual",
    )
    try:
        started = time.monotonic()
        with pytest.raises(WorkerDeadlineExceeded):
            pool.request({"block": True}, deadline=time.monotonic() + 0.5)
        assert time.monotonic() - started < 2
        assert pool.request(
            {"block": False}, deadline=time.monotonic() + 5,
        ) == {"block": False}
    finally:
        pool.close()


def test_task_worker_pool_keeps_following_requests_after_remote_failure() -> None:
    worker = _FakeWorker()
    pool = TaskWorkerPool(lambda: worker, worker_name="ocr")
    try:
        with pytest.raises(WorkerRemoteError, match="expected failure"):
            pool.request({"fail": True})
        assert pool.request({"sequence": 2}) == {"sequence": 2}
    finally:
        pool.close()

    assert [item[0]["request_id"] for item in worker.requests] == [
        "ocr-00000001",
        "ocr-00000002",
    ]


def test_task_worker_pool_cancels_without_starting_worker_and_rejects_after_close() -> None:
    cancelled = threading.Event()
    cancelled.set()
    created = []
    pool = TaskWorkerPool(
        lambda: created.append(_FakeWorker()) or created[-1],
        worker_name="visual",
    )

    with pytest.raises(WorkerCancelled):
        pool.request({"sequence": 1}, cancel=cancelled)
    assert created == []

    pool.close()
    with pytest.raises(WorkerPoolClosed):
        pool.request({"sequence": 2})


def test_task_worker_pool_rejects_requests_beyond_its_queue_limit() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingWorker(_FakeWorker):
        def request(self, envelope: dict, *, deadline: float | None) -> dict:
            self.requests.append((envelope, deadline))
            started.set()
            assert release.wait(5)
            return {
                "request_id": envelope["request_id"],
                "result": envelope["payload"],
            }

    worker = BlockingWorker()
    pool = TaskWorkerPool(lambda: worker, queue_limit=1, worker_name="ocr")
    result = []

    def first_request() -> None:
        result.append(pool.request({"sequence": 1}))

    thread = threading.Thread(target=first_request)
    thread.start()
    assert started.wait(5)
    try:
        with pytest.raises(WorkerQueueFull):
            pool.request({"sequence": 2})
    finally:
        release.set()
        thread.join(5)
        pool.close()

    assert result == [{"sequence": 1}]


def test_task_worker_pool_cancels_inflight_request_and_rebuilds_worker() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingWorker(_FakeWorker):
        def request(self, envelope: dict, *, deadline: float | None) -> dict:
            self.requests.append((envelope, deadline))
            started.set()
            assert release.wait(5)
            return {
                "request_id": envelope["request_id"],
                "result": envelope["payload"],
            }

        def terminate(self) -> None:
            super().terminate()
            release.set()

    first = BlockingWorker()
    second = _FakeWorker()
    created = [first, second]
    pool = TaskWorkerPool(lambda: created.pop(0), worker_name="visual")
    cancelled = threading.Event()
    failure = []

    def request_then_cancel() -> None:
        try:
            pool.request({"sequence": 1}, cancel=cancelled)
        except BaseException as error:
            failure.append(error)

    thread = threading.Thread(target=request_then_cancel)
    thread.start()
    assert started.wait(5)
    cancelled.set()
    thread.join(5)
    try:
        assert len(failure) == 1
        assert isinstance(failure[0], WorkerCancelled)
        assert first.terminated == 1
        assert pool.request({"sequence": 2}) == {"sequence": 2}
    finally:
        pool.close()

    assert second.closed == 1


def test_task_worker_pool_traces_model_load_queue_wait_and_cancellation() -> None:
    class Trace:
        def __init__(self) -> None:
            self.events = []

        def event(self, event: str, **fields) -> None:
            self.events.append((event, fields))

    trace = Trace()
    cancelled = threading.Event()
    pool = TaskWorkerPool(lambda: _FakeWorker(), worker_name="ocr")
    try:
        assert pool.request(
            {"sequence": 1}, performance_trace=trace, page_id="page_001",
        ) == {"sequence": 1}
        cancelled.set()
        with pytest.raises(WorkerCancelled):
            pool.request(
                {"sequence": 2},
                cancel=cancelled,
                performance_trace=trace,
                page_id="page_001",
            )
    finally:
        pool.close()

    assert ("model_load_start", {"model": "ocr", "page_id": "page_001"}) in trace.events
    assert any(
        event == "model_load_finish"
        and fields["model"] == "ocr"
        and fields["status"] == "success"
        for event, fields in trace.events
    )
    assert any(
        event == "span" and fields["stage"] == "worker_queue"
        for event, fields in trace.events
    )
    assert any(
        event == "worker" and fields["stage"] == "worker_cancel"
        for event, fields in trace.events
    )


def test_resident_ocr_processor_reuses_models_and_closes_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import ocr_worker

    loaded = []
    closed = []

    class Detector:
        def __init__(self, **kwargs) -> None:
            loaded.append("detector")

        def predict(self, *args, **kwargs):
            return [{"dt_polys": [[[0, 0], [3, 0], [3, 2], [0, 2]]]}]

        def close(self) -> None:
            closed.append("detector")

    class Recognizer:
        def __init__(self, **kwargs) -> None:
            loaded.append("recognizer")

        def predict(self, crops):
            return [{"rec_text": "text", "rec_score": 0.99} for _ in crops]

        def close(self) -> None:
            closed.append("recognizer")

    monkeypatch.setattr(
        ocr_worker,
        "_load_detection_tools",
        lambda: (Detector, lambda: lambda polys: polys, lambda **kwargs: lambda image, polys: [np.zeros((2, 3, 3), dtype=np.uint8) for _ in polys]),
    )
    monkeypatch.setattr(ocr_worker, "_load_recognition_model", lambda: Recognizer)
    monkeypatch.setattr(
        ocr_worker,
        "_read_bgr",
        lambda path: np.zeros((3, 4, 3), dtype=np.uint8),
    )

    processor = ocr_worker._ResidentOcrProcessor()
    try:
        for index in range(2):
            result_path = tmp_path / f"result-{index}.json"
            ocr_worker.run_batch(
                [tmp_path / f"source-{index}.png"],
                result_path,
                "en",
                processor=processor,
            )
            assert json.loads(result_path.read_text(encoding="utf-8"))["images"][0]["items"][0]["text"] == "text"
    finally:
        processor.close()

    assert loaded == ["detector", "recognizer"]
    assert closed == ["detector", "recognizer"]


def test_ocr_worker_serve_continues_after_request_error(monkeypatch, capsys) -> None:
    from scripts import ocr_worker

    closed = []

    class Processor:
        def close(self) -> None:
            closed.append(True)

    calls = []

    def run_batch(images, result, lang, *, processor) -> None:
        calls.append((images, result, lang, processor))
        if len(calls) == 1:
            raise RuntimeError("first failure")

    monkeypatch.setattr(ocr_worker, "_ResidentOcrProcessor", Processor)
    monkeypatch.setattr(ocr_worker, "run_batch", run_batch)
    monkeypatch.setattr(
        ocr_worker.sys,
        "stdin",
        io.StringIO(
            '{"request_id":"one","payload":{"images":["first.png"],"result":"first.json","lang":"en"}}\n'
            '{"request_id":"two","payload":{"images":["second.png"],"result":"second.json","lang":"en"}}\n'
            '{"control":"close"}\n'
        ),
    )

    assert ocr_worker._serve() == 0
    responses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert responses == [
        {"request_id": "one", "error": {"type": "RuntimeError", "message": "first failure"}},
        {"request_id": "two", "result": {}},
    ]
    assert len(calls) == 2
    assert closed == [True]


def test_ocr_worker_serve_keeps_model_stdout_out_of_json_protocol(
    monkeypatch,
    capsys,
) -> None:
    from scripts import ocr_worker

    class Processor:
        def close(self) -> None:
            pass

    def run_batch(*args, **kwargs) -> None:
        print("PaddleOCR startup message")

    monkeypatch.setattr(ocr_worker, "_ResidentOcrProcessor", Processor)
    monkeypatch.setattr(ocr_worker, "run_batch", run_batch)
    monkeypatch.setattr(
        ocr_worker.sys,
        "stdin",
        io.StringIO(
            '{"request_id":"one","payload":{"images":["source.png"],"result":"result.json","lang":"en"}}\n'
            '{"control":"close"}\n'
        ),
    )

    assert ocr_worker._serve() == 0
    captured = capsys.readouterr()

    assert [json.loads(line) for line in captured.out.splitlines()] == [
        {"request_id": "one", "result": {}},
    ]
    assert captured.err == "PaddleOCR startup message\n"


def test_visual_worker_serve_continues_after_request_error(monkeypatch, capsys) -> None:
    from scripts import visual_worker

    closed = []

    class Processor:
        def close(self) -> None:
            closed.append(True)

    calls = []

    def process_request(payload, processor) -> None:
        calls.append((payload, processor))
        if len(calls) == 1:
            raise RuntimeError("first failure")

    monkeypatch.setattr(visual_worker, "_ResidentVisualProcessor", Processor)
    monkeypatch.setattr(visual_worker, "_process_request", process_request)
    monkeypatch.setattr(
        visual_worker.sys,
        "stdin",
        io.StringIO(
            '{"request_id":"one","payload":{}}\n'
            '{"request_id":"two","payload":{}}\n'
            '{"control":"close"}\n'
        ),
    )

    assert visual_worker._serve() == 0
    responses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert responses == [
        {"request_id": "one", "error": {"type": "RuntimeError", "message": "first failure"}},
        {"request_id": "two", "result": {}},
    ]
    assert len(calls) == 2
    assert closed == [True]


def test_resident_visual_processor_reuses_dino_sam_and_lama_models(monkeypatch) -> None:
    from scripts import lama_inpaint, visual_worker

    loaded = []
    detector = object()
    generator = object()

    def fake_process(*args, **kwargs) -> dict:
        loaded.append((kwargs["_resource_isolation"], args[2], args[3]))
        lama_inpaint._get_model()
        return {"components": []}

    monkeypatch.setattr(
        lama_inpaint,
        "_create_model",
        lambda: loaded.append("lama") or object(),
    )
    lama_inpaint.release_model()
    processor = visual_worker._ResidentVisualProcessor(
        process_image=fake_process,
        create_detector=lambda: loaded.append("dino") or detector,
        create_generator=lambda checkpoint: loaded.append("sam") or generator,
        resolve_checkpoint=lambda: "checkpoint",
    )
    try:
        for _ in range(2):
            assert processor.process(
                Path("source.png"),
                Path("work"),
                "en",
                {"items": [], "mask_path": "mask.png"},
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.uint8),
                None,
                visual_worker.strict_page_policy(),
            ) == {"components": []}
    finally:
        processor.close()
        lama_inpaint.release_model()

    assert loaded == [
        "dino", "sam", (False, detector, generator), "lama",
        (False, detector, generator),
    ]


def test_visual_worker_pool_uses_one_resident_visual_server(monkeypatch) -> None:
    commands = []

    class FakeJsonLineWorker(_FakeWorker):
        def __init__(self, command) -> None:
            super().__init__()
            commands.append(command)

    monkeypatch.setattr(image_to_ppt, "JsonLineWorker", FakeJsonLineWorker)

    pool = image_to_ppt.create_visual_worker_pool()
    try:
        assert pool.request({"sequence": 1}) == {"sequence": 1}
        assert pool.request({"sequence": 2}) == {"sequence": 2}
    finally:
        pool.close()

    worker_path = (
        Path(image_to_ppt.__file__).resolve().parent
        / "scripts"
        / "visual_worker.py"
    )
    assert commands == [[str(image_to_ppt.sys.executable), str(worker_path), "--serve"]]


def test_process_image_isolated_sends_verified_request_to_visual_pool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "source.png"
    work_dir = tmp_path / "work"
    mask_path = work_dir / "source-text-mask.png"
    work_dir.mkdir()
    image_path = work_dir / "source.png"
    from PIL import Image

    Image.new("RGB", (4, 3), "white").save(image_path)
    Image.new("L", (4, 3), 0).save(mask_path)
    received = []

    class FakePool:
        def request(self, payload, **kwargs):
            received.append((payload, kwargs))
            Path(payload["result"]).write_text(
                '{"components": []}', encoding="utf-8",
            )
            return {}

    monkeypatch.setattr(
        image_to_ppt,
        "run_isolated_worker",
        lambda *args, **kwargs: pytest.fail("pool must not start a one-shot worker"),
    )

    result = image_to_ppt._process_image_isolated(
        image_path,
        work_dir,
        "en",
        {"items": [], "mask_path": str(mask_path)},
        worker_pool=FakePool(),
    )

    assert result["components"] == []
    payload, kwargs = received[0]
    assert payload["image"] == str(image_path)
    assert payload["source_size"] == image_path.stat().st_size
    request = json.loads(Path(payload["request"]).read_text(encoding="utf-8"))
    assert request["text_mask_size"] == mask_path.stat().st_size
    assert kwargs == {"performance_trace": None, "page_id": None}
