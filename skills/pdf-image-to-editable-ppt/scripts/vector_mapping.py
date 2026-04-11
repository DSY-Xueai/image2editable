from __future__ import annotations

from .models import VectorInstruction


def map_vector_candidate(candidate: dict):
    if candidate.get("object_type") != "vector":
        return None
    if candidate.get("complexity") != "simple":
        return None
    if not candidate.get("rebuildable", False):
        return None
    if candidate.get("must_fallback", False):
        return None
    return VectorInstruction(
        shape_type="freeform",
        left=float(candidate.get("left", 0.0)),
        top=float(candidate.get("top", 0.0)),
        width=float(candidate.get("width", 0.0)),
        height=float(candidate.get("height", 0.0)),
        payload={"source": "vector_candidate"},
    )


def map_vector_candidates(candidates: list[dict]) -> list[VectorInstruction]:
    return [
        instruction
        for candidate in candidates
        if (instruction := map_vector_candidate(candidate)) is not None
    ]
