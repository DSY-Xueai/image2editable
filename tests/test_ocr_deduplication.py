from itertools import permutations

import image_to_ppt


def test_complete_line_replaces_all_contained_fragments_in_any_order():
    left = {"text": "alpha beta", "box": [0, 0, 100, 30], "confidence": .99}
    right = {"text": "gamma delta", "box": [110, 0, 110, 30], "confidence": .99}
    complete = {"text": "alpha beta gamma delta", "box": [0, 0, 220, 30],
                "confidence": .98, "words": [{"text": "alpha beta gamma delta",
                                               "box": [0, 0, 1, 1]}]}
    for items in permutations([left, right, complete]):
        result = image_to_ppt._deduplicate_overlapping_text_items(list(items))
        assert result == [complete]
        assert result[0] is complete


def test_deduplication_keeps_distinct_locations_and_conflicting_readings():
    items = [
        {"text": "alpha beta", "box": [0, 0, 100, 30]},
        {"text": "alpha beta", "box": [0, 100, 100, 30]},
        {"text": "alpha theta", "box": [0, 0, 100, 30]},
    ]
    assert image_to_ppt._deduplicate_overlapping_text_items(items) == items


def test_equal_text_keeps_stronger_reading_and_preserves_unrelated_order():
    weak = {"text": "alpha", "box": [0, 0, 100, 30], "confidence": .8}
    strong = {**weak, "confidence": .99}
    unrelated = {"text": "beta", "box": [0, 100, 100, 30]}
    assert image_to_ppt._deduplicate_overlapping_text_items([weak, unrelated, strong]) == [strong, unrelated]


def test_spanning_detection_preserves_columns_and_added_punctuation():
    left = {"text": "A large caption.", "box": [10, 10, 220, 60], "font_size": 30}
    right = {"text": "Small note.", "box": [240, 15, 100, 30], "font_size": 15}
    spanning = {"text": 'A large caption.” Small note.', "box": [10, 10, 330, 60], "font_size": 30}
    for items in permutations([left, right, spanning]):
        result = image_to_ppt._deduplicate_overlapping_text_items(list(items))
        result = sorted(result, key=lambda item: item["box"][0])
        assert [item["text"] for item in result] == ['A large caption.”', 'Small note.']
        assert [item["font_size"] for item in result] == [30, 15]
        assert [item["box"] for item in result] == [left["box"], right["box"]]


def test_spanning_detection_with_new_words_is_not_discarded():
    left = {"text": "Caption", "box": [10, 10, 220, 60], "font_size": 30}
    right = {"text": "Note", "box": [240, 15, 100, 30], "font_size": 15}
    spanning = {"text": "Caption extra Note", "box": [10, 10, 330, 60], "font_size": 30}
    result = image_to_ppt._deduplicate_overlapping_text_items([left, right, spanning])
    assert any("extra" in item["text"] for item in result)


def test_repeated_text_in_separate_columns_keeps_each_position():
    left = {"text": "Total", "box": [0, 0, 100, 40], "font_size": 30}
    right = {"text": "Total", "box": [140, 0, 70, 20], "font_size": 15}
    spanning = {"text": "Total Total", "box": [0, 0, 210, 40], "font_size": 30}
    for items in permutations([left, right, spanning]):
        result = image_to_ppt._deduplicate_overlapping_text_items(list(items))
        assert sorted(result, key=lambda item: item["box"][0]) == [left, right]


def test_equal_height_columns_with_large_gap_keep_their_styles():
    left = {"text": "Revenue", "box": [0, 0, 100, 30], "color": "#ff0000"}
    right = {"text": "Profit", "box": [400, 0, 100, 30], "color": "#0000ff"}
    spanning = {"text": "Revenue Profit", "box": [0, 0, 500, 30]}
    for items in permutations([left, right, spanning]):
        result = image_to_ppt._deduplicate_overlapping_text_items(list(items))
        assert sorted(result, key=lambda item: item["box"][0]) == [left, right]
