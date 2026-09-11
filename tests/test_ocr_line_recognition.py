import json
from types import SimpleNamespace

from PIL import Image
import pytest

from scripts import ocr_worker
from scripts import text_detect


def test_line_recognition_does_not_initialize_detector(tmp_path):
    source = tmp_path / "line.png"
    Image.new("RGB", (120, 30), "white").save(source)
    calls = []
    class Recognizer:
        def predict(self, crops, **kwargs):
            calls.append((len(crops), kwargs))
            return [{"rec_text": "Complete line", "rec_score": .99} for crop in crops]
    def forbidden():
        pytest.fail("Known text lines must not run detection again")
    processor = SimpleNamespace(detection_tools=forbidden, recognizer=lambda lang: Recognizer())
    result_path = tmp_path / "result.json"
    ocr_worker.run_batch([source], result_path, processor=processor, recognition_only=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))["images"][0]["items"]
    assert calls == [(1, {"return_word_box": True})]
    assert result == [{"poly": [[0, 0], [120, 0], [120, 30], [0, 30]],
                       "text": "Complete line", "score": .99}]


def test_isolated_line_reader_forwards_recognition_only(tmp_path):
    requests = []
    def request(payload, **kwargs):
        requests.append(payload)
        from pathlib import Path
        Path(payload["result"]).write_text(json.dumps({"images": [{"items": [
            {"poly": [[0, 0], [120, 0], [120, 30], [0, 30]],
             "text": "Complete line", "score": .99}]}]}), encoding="utf-8")
    result = text_detect._try_isolated_paddleocr_batch(
        [tmp_path / "line.png"], "en", .9, worker_root=tmp_path,
        worker_pool=SimpleNamespace(request=request), recognition_only=True,
    )
    assert requests[0]["recognition_only"] is True
    assert result[0][0]["text"] == "Complete line"
