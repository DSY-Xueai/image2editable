from __future__ import annotations


def group_blend_candidates(candidates: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for candidate in candidates:
        group_id = candidate.get("group_id", "default")
        state = grouped.setdefault(
            group_id,
            {"group_id": group_id, "items": [], "must_fallback": False},
        )
        state["items"].append(candidate)
        if candidate.get("must_fallback", False):
            state["must_fallback"] = True
    return list(grouped.values())
