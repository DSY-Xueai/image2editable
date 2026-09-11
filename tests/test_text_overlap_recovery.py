from PIL import Image

from scripts import text_detect


def test_cached_context_restores_missing_edge_character_without_recognition(tmp_path, monkeypatch):
    from scripts.text_context import refine_overlapping_text
    source = tmp_path / "source.png"
    Image.new("RGB", (300, 100), "white").save(source)
    items = [{"text": "类的问题", "box": [60, 20, 160, 40], "confidence": .99}]
    context = [{"text": "这类的问题", "score": .999, "box": [15, 18, 220, 44],
                "words": [{"text": c, "box": [i/5, 0, .2, 1]}
                          for i, c in enumerate("这类的问题")]}]
    monkeypatch.setattr(text_detect, "_try_isolated_paddleocr_batch", lambda *a, **k: 1/0)
    result = refine_overlapping_text(source, items, tmp_path, lang="ch", context_readings=context)
    assert [item["text"] for item in result] == ["这类的问题"]
    assert result[0]["box"][0] < 60


def test_cached_context_quote_absorbs_only_small_overlapping_fragment(tmp_path):
    from scripts.text_context import refine_overlapping_text
    source = tmp_path / "source.png"
    Image.new("RGB", (400, 100), "white").save(source)
    items = [{"text": "小小的花朵。", "box": [40, 20, 220, 40], "confidence": .99},
             {"text": "99", "box": [275, 20, 15, 12], "confidence": .98},
             {"text": "99", "box": [330, 20, 30, 40], "confidence": .99}]
    context = [{"text": "小小的花朵。”", "score": .999, "box": [20, 18, 280, 44],
                "words": [{"text": "小小的花朵。", "box": [.07, 0, .8, 1]},
                          {"text": "”", "box": [.9, 0, .08, 1]}]}]
    result = refine_overlapping_text(source, items, tmp_path, lang="ch", context_readings=context)
    assert [item["text"] for item in result] == ["小小的花朵。”", "99"]


def test_cached_context_cannot_rewrite_content_or_expand_to_neighboring_line(tmp_path):
    from scripts.text_context import refine_overlapping_text
    items = [{"text": "大大的花朵", "box": [40, 20, 220, 40], "confidence": .99}]
    for text in ("小小的花朵", "大大的花朵旁边有树"):
        context = [{"text": text, "score": .9999, "box": [20, 18, 280, 44]}]
        assert refine_overlapping_text(tmp_path / "unused.png", items, tmp_path,
                                       lang="ch", context_readings=context) == items


def test_bracketed_label_recovers_visible_horizontal_leader(tmp_path):
    from PIL import ImageDraw
    from scripts.text_context import refine_overlapping_text
    source = tmp_path / "source.png"
    image = Image.new("RGB", (300, 100), "white")
    ImageDraw.Draw(image).rectangle((30, 43, 95, 47), fill=(45, 30, 15))
    image.save(source)
    item = {"text": "[答案]", "box": [20, 20, 220, 50], "confidence": .99,
            "words": [{"text": "[答案]", "box": [.4, 0, .6, 1]}]}
    result = refine_overlapping_text(source, [item], tmp_path, lang="ch")
    assert result[0]["text"] == "—[答案]"
    assert result[0]["words"][0]["text"] == "—"


def test_bracketed_label_does_not_invent_missing_leader(tmp_path):
    from scripts.text_context import refine_overlapping_text
    source = tmp_path / "source.png"
    Image.new("RGB", (300, 100), "white").save(source)
    item = {"text": "[答案]", "box": [20, 20, 220, 50], "confidence": .99}
    assert refine_overlapping_text(source, [item], tmp_path, lang="ch") == [item]


def test_conflicting_fragments_use_agreed_context_reading(tmp_path, monkeypatch):
    from scripts.text_context import refine_overlapping_text
    source = tmp_path / "source.png"
    Image.new("RGB", (400, 100), "white").save(source)
    left = {"text": "alpha beta", "box": [30, 20, 130, 30], "confidence": .99}
    right = {"text": "beta gamma", "box": [90, 20, 130, 30], "confidence": .98}
    calls = []
    def recognize(paths, *args, **kwargs):
        calls.append(kwargs)
        return [[{"text": "alpha beta gamma", "box": [0, 0, Image.open(path).width, 30],
                  "confidence": .999, "words": [{"text": "alpha beta gamma", "box": [0, 0, 1, 1]}]}]
                for path in paths]
    monkeypatch.setattr(text_detect, "_try_isolated_paddleocr_batch", recognize)
    result = refine_overlapping_text(source, [left, right], tmp_path, lang="en")
    assert [item["text"] for item in result] == ["alpha beta gamma"]
    assert calls[0]["recognition_only"] is True
    assert result[0]["words"]


def test_disagreeing_context_does_not_remove_original_text(tmp_path, monkeypatch):
    from scripts.text_context import refine_overlapping_text
    source = tmp_path / "source.png"
    Image.new("RGB", (400, 100), "white").save(source)
    items = [{"text": "alpha beta", "box": [30, 20, 130, 30], "confidence": .99},
             {"text": "beta gamma", "box": [90, 20, 130, 30], "confidence": .98}]
    monkeypatch.setattr(text_detect, "_try_isolated_paddleocr_batch", lambda *args, **kwargs: [
        [{"text": text, "box": [0,0,200,30], "confidence": .999}]
        for text in ("alpha beta gamma", "alpha theta gamma")])
    assert refine_overlapping_text(source, items, tmp_path, lang="en") == items


def test_disjoint_text_never_triggers_recognition(tmp_path, monkeypatch):
    from scripts.text_context import refine_overlapping_text
    items = [{"text": "alpha", "box": [0,0,100,30]},
             {"text": "beta", "box": [0,50,100,30]}]
    monkeypatch.setattr(text_detect, "_try_isolated_paddleocr_batch", lambda *a, **k: 1/0)
    assert refine_overlapping_text(tmp_path / "unused.png", items, tmp_path, lang="en") == items


def test_targeted_sweep_resolves_overlap_before_building_final_mask(tmp_path, monkeypatch):
    import numpy as np
    import image_to_ppt
    from scripts import text_context

    source = tmp_path / "source.png"
    Image.new("RGB", (400, 100), "white").save(source)
    items = [{"text": "alpha beta", "box": [30, 20, 130, 30]},
             {"text": "beta gamma", "box": [90, 20, 130, 30]}]
    corrected = {"text": "alpha beta gamma", "box": [20, 20, 230, 30]}
    calls = []

    def refine(path, candidates, work_dir, **kwargs):
        calls.append(candidates)
        return [corrected]

    monkeypatch.setattr(text_context, "refine_overlapping_text", refine)
    result = image_to_ppt._targeted_candidate_ocr_sweep(
        source, [], items, np.zeros((100, 400), dtype=np.uint8), tmp_path,
        lang="en", isolated=True,
    )
    assert calls == [items]
    assert result["items"] == [{**corrected, "align": 0}]
    assert result["text_mask"][30, 240] == 255


def test_agreed_context_corrects_one_conflicting_character(tmp_path, monkeypatch):
    from scripts.text_context import refine_overlapping_text
    source = tmp_path / "source.png"
    Image.new("RGB", (400, 100), "white").save(source)
    items = [{"text": "alpha beta", "box": [30, 20, 130, 30]},
             {"text": "beto gamma", "box": [90, 20, 130, 30]}]
    monkeypatch.setattr(text_detect, "_try_isolated_paddleocr_batch", lambda *a, **k: [
        [{"text": "alpha beta gamma", "box": [0, 0, 200, 30], "confidence": .999}]
        for _ in range(2)])
    result = refine_overlapping_text(source, items, tmp_path, lang="en")
    assert [item["text"] for item in result] == ["alpha beta gamma"]


def test_context_cannot_replace_two_characters_or_use_low_confidence(tmp_path, monkeypatch):
    from scripts.text_context import refine_overlapping_text
    source = tmp_path / "source.png"
    Image.new("RGB", (400, 100), "white").save(source)
    for text, confidence in [("beto gamma", .99), ("beto ganna", .999)]:
        items = [{"text": "alpha beta", "box": [30, 20, 130, 30]},
                 {"text": text, "box": [90, 20, 130, 30]}]
        monkeypatch.setattr(text_detect, "_try_isolated_paddleocr_batch", lambda *a, **k: [
            [{"text": "alpha beta gamma", "box": [0, 0, 200, 30], "confidence": confidence}]
            for _ in range(2)])
        assert refine_overlapping_text(source, items, tmp_path, lang="en") == items


def test_unchanged_context_reuses_readings_even_when_they_disagree(tmp_path, monkeypatch):
    from scripts.text_context import refine_overlapping_text
    source = tmp_path / "source.png"
    Image.new("RGB", (400, 100), "white").save(source)
    items = [{"text": "alpha beta", "box": [30, 20, 130, 30]},
             {"text": "beta gamma", "box": [90, 20, 130, 30]}]
    calls = []
    def recognize(*args, **kwargs):
        calls.append(True)
        return [[{"text": text, "box": [0, 0, 200, 30], "confidence": .999}]
                for text in ("alpha beta gamma", "alpha theta gamma")]
    monkeypatch.setattr(text_detect, "_try_isolated_paddleocr_batch", recognize)
    for _ in range(2):
        assert refine_overlapping_text(source, items, tmp_path, lang="en") == items
    assert len(calls) == 1
