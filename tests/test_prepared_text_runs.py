from __future__ import annotations

import pytest

import image_to_ppt


def _payload(version=8):
    return {
        "schema_version": version,
        "dimensions": {"img_width": 200, "img_height": 100,
            "canvas_width": 200, "canvas_height": 100, "content_offset_x": 0,
            "content_offset_y": 0, "widescreen_background_method": "identity"},
        "text_items": [{"text": "AB", "box": [10, 10, 100, 40], "font_size": 20,
            "font": "Arial", "bold": False, "color": "#000000", "rotation": 0,
            "confidence": 0.99, "align": 1,
            "words": [{"text": "AB", "box": [0.1, 0.1, 0.8, 0.8]}],
            "runs": [{"text": "A", "box": [0.1, 0.1, 0.3, 0.8], "color": "#abcdef"},
                     {"text": "B", "box": [0.6, 0.2, 0.3, 0.7], "rotation": -10.5}]}],
        "components": [], "initial_diagnostics": [],
        "page_policy": {"schema_version": 1, "route": "strict", "confidence": 1.0, "reasons": []},
    }


def test_prepared_schema_accepts_validated_words_and_positioned_runs():
    payload = _payload()
    image_to_ppt._validate_prepared_payload(payload)
    assert payload == _payload()


def test_prepared_schema_preserves_explicit_ink_geometry():
    payload = _payload()
    payload["text_items"][0]["runs"][0]["box_kind"] = "ink"
    image_to_ppt._validate_prepared_payload(payload)
    assert payload["text_items"][0]["runs"][0]["box_kind"] == "ink"


def test_word_geometry_accepts_recognizer_whitespace_tokens():
    from scripts.text_runs import validate_text_words
    item = {"text": "AB", "words": [
        {"text": "A", "box": [0, 0, .4, 1]},
        {"text": " ", "box": [.4, 0, .1, 1]},
        {"text": "B", "box": [.5, 0, .5, 1]},
    ]}
    validate_text_words(item)
    item["words"][1]["box"][0] = -1
    with pytest.raises(ValueError, match="text words"):
        validate_text_words(item)


@pytest.mark.parametrize("field,value", [
    ("words", [{"text": "Wrong", "box": [0, 0, 1, 1]}]),
    ("words", [{"text": "AB", "box": [0, 0, float("nan"), 1]}]),
    ("words", []),
    ("runs", [{"text": "A", "box": [0, 0, 1, 1]}]),
    ("runs", [{"text": "AB", "box": [0, 0, 1, 1], "outline_width": -1}]),
])
def test_prepared_rejects_corrupt_text_geometry(field, value):
    payload = _payload()
    payload["text_items"][0][field] = value
    with pytest.raises(ValueError, match="text (runs|words)"):
        image_to_ppt._validate_prepared_payload(payload)


def test_legacy_schema_does_not_accept_unsigned_new_text_fields():
    payload = _payload(7)
    with pytest.raises(ValueError, match="text item fields"):
        image_to_ppt._validate_prepared_payload(payload)
    item = payload["text_items"][0]
    item.pop("words")
    item.pop("runs")
    image_to_ppt._validate_prepared_payload(payload)
