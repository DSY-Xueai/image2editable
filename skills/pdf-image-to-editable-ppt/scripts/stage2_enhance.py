from __future__ import annotations

from .models import PagePlan


def apply_stage2_enhancements(plan: PagePlan) -> PagePlan:
    enhanced = PagePlan(
        page_number=plan.page_number,
        width_px=plan.width_px,
        height_px=plan.height_px,
        background_path=plan.background_path,
        source_type=plan.source_type,
        page_width_points=plan.page_width_points,
        page_height_points=plan.page_height_points,
    )
    enhanced.text_blocks = [block for block in plan.text_blocks if block.fit_verified]
    enhanced.image_blocks = list(plan.image_blocks)
    enhanced.effect_blocks = list(plan.effect_blocks)
    return enhanced
