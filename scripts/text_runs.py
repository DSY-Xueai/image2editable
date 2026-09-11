"""Validated native text runs positioned relative to an OCR line."""

from __future__ import annotations

import math


def validate_text_words(item: dict) -> None:
    words = item.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError("text words must be a nonempty list")
    for word in words:
        if not isinstance(word, dict) or set(word) != {"text", "box"}:
            raise ValueError("text words fields are invalid")
    try:
        recognized = "".join(word["text"] for word in words)
        validate_text_runs({"text": recognized, "runs": words})
        if "".join(recognized.split()) != "".join(item["text"].split()):
            raise ValueError("text words must preserve source text")
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("text words content or geometry is invalid") from error


def validate_text_runs(item: dict) -> None:
    runs = item.get("runs")
    required = {"text", "box"}
    allowed = required | {
        "font", "font_size", "color", "bold", "rotation",
        "outline_color", "outline_width", "box_kind", "gradient",
    }
    if not isinstance(runs, list) or not runs:
        raise ValueError("text runs must be a nonempty list")
    for run in runs:
        if not isinstance(run, dict) or not required <= set(run) <= allowed:
            raise ValueError("text runs fields are invalid")
        if run.get("box_kind", "layout") not in ("layout", "ink"):
            raise ValueError("text runs box kind is invalid")
        box = run["box"]
        if (
            not isinstance(run["text"], str) or not run["text"]
            or not isinstance(box, list) or len(box) != 4
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in box)
            or min(box[:2]) < 0 or min(box[2:]) <= 0
            or box[0] + box[2] > 1.000001 or box[1] + box[3] > 1.000001
        ):
            raise ValueError("text runs content or box is invalid")
        for key in ("font_size", "rotation", "outline_width"):
            if key in run and (
                type(run[key]) not in {int, float} or not math.isfinite(run[key])
                or (key == "font_size" and not 0 < run[key] <= 4000)
                or (key == "outline_width" and not 0 <= run[key] <= 1584)
                or (key == "rotation" and not -360 <= run[key] <= 360)
            ):
                raise ValueError("text runs numeric style is invalid")
        for key in ("color", "outline_color"):
            if key in run and (
                not isinstance(run[key], str) or len(run[key]) != 7
                or run[key][0] != "#"
                or any(c not in "0123456789abcdefABCDEF" for c in run[key][1:])
            ):
                raise ValueError("text runs color is invalid")
        if ("outline_color" in run) != ("outline_width" in run):
            raise ValueError("text runs outline requires color and width")
        if "bold" in run and type(run["bold"]) is not bool:
            raise ValueError("text runs weight is invalid")
        if "font" in run and (not isinstance(run["font"], str) or not run["font"].strip()):
            raise ValueError("text runs font is invalid")
        if "gradient" in run:
            gradient = run["gradient"]
            if (not isinstance(gradient, dict) or set(gradient) != {"angle", "colors"}
                or type(gradient["angle"]) not in {int, float} or not math.isfinite(gradient["angle"])
                or not 0 <= gradient["angle"] < 360 or not isinstance(gradient["colors"], list)
                or len(gradient["colors"]) != 2
                or any(not isinstance(color, str) or len(color) != 7 or color[0] != "#"
                       or any(c not in "0123456789abcdefABCDEF" for c in color[1:]) for color in gradient["colors"])):
                raise ValueError("text runs gradient is invalid")
    if "".join(run["text"] for run in runs) != item.get("text"):
        raise ValueError("text runs must preserve all source text in order")
