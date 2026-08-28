from __future__ import annotations

import math
import unicodedata

MAX_INITIAL_DIAGNOSTICS = 96


def _normalized_text(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_initial_diagnostics(
    diagnostics: object,
    *,
    source_sha256: str,
    image_size: tuple[int, int] | None = None,
) -> list[dict]:
    if not isinstance(diagnostics, list) or len(diagnostics) > MAX_INITIAL_DIAGNOSTICS:
        raise ValueError("initial diagnostics are invalid")
    seen_ids = set()
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict) or set(diagnostic) != {
            "kind", "source_sha256", "candidate_id", "bbox", "views"
        }:
            raise ValueError("initial diagnostic fields are invalid")
        candidate_id = diagnostic["candidate_id"]
        parts = candidate_id.split("_") if isinstance(candidate_id, str) else []
        valid_id = (
            len(parts) in {2, 3}
            and parts[0] == "candidate"
            and len(parts[1]) == 4
            and parts[1].isdigit()
            and int(parts[1]) > 0
            and (
                len(parts) == 2
                or len(parts[2]) == 2 and parts[2].isdigit() and int(parts[2]) > 0
            )
        )
        if (
            diagnostic["kind"] != "unowned_raster_text"
            or diagnostic["source_sha256"] != source_sha256
            or not valid_id
            or candidate_id in seen_ids
        ):
            raise ValueError("initial diagnostic identity is invalid")
        seen_ids.add(candidate_id)
        bbox = diagnostic["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or bbox[0] < 0
            or bbox[1] < 0
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
            or image_size is not None
            and (bbox[2] > image_size[0] or bbox[3] > image_size[1])
        ):
            raise ValueError("initial diagnostic bbox is invalid")
        views = diagnostic["views"]
        if not isinstance(views, list) or len(views) != 2:
            raise ValueError("initial diagnostic views are invalid")
        for view in views:
            if not isinstance(view, dict) or set(view) != {
                "normalized_text", "confidence"
            }:
                raise ValueError("initial diagnostic view fields are invalid")
            text = view["normalized_text"]
            confidence = view["confidence"]
            if (
                not isinstance(text, str)
                or not 1 <= len(text) <= 256
                or _normalized_text(text) != text
                or not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not math.isfinite(confidence)
                or not 0.88 <= confidence <= 1
            ):
                raise ValueError("initial diagnostic view values are invalid")
    return diagnostics
