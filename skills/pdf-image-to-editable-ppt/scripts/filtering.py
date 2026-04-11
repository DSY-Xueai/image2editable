from __future__ import annotations

from .models import PagePlan


def select_editable_blocks(
    plan: PagePlan,
    *,
    min_text_confidence: float,
    min_image_confidence: float,
) -> PagePlan:
    filtered = PagePlan(
        page_number=plan.page_number,
        width_px=plan.width_px,
        height_px=plan.height_px,
        background_path=plan.background_path,
    )
    filtered.text_blocks = [
        block for block in plan.text_blocks if block.confidence >= min_text_confidence
    ]
    filtered.image_blocks = [
        block
        for block in plan.image_blocks
        if block.extractable and block.confidence >= min_image_confidence
    ]
    return filtered
