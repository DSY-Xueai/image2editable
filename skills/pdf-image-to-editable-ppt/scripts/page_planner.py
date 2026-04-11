from __future__ import annotations

from .models import (
    ImageBlock,
    LayeredObject,
    PagePlan,
    TextBlock,
    VectorInstruction,
)


def build_page_plan(
    *,
    page_number,
    width_px,
    height_px,
    background_path,
    text_items,
    image_items,
    layered_objects=None,
    vector_instructions=None,
    source_type="image",
    page_width_points=None,
    page_height_points=None,
):
    plan = PagePlan(
        page_number=page_number,
        width_px=width_px,
        height_px=height_px,
        background_path=background_path,
        source_type=source_type,
        page_width_points=page_width_points,
        page_height_points=page_height_points,
    )
    plan.text_blocks = [TextBlock(**item) for item in text_items]
    plan.image_blocks = [ImageBlock(**item) for item in image_items]
    plan.layered_objects = [
        item if isinstance(item, LayeredObject) else LayeredObject(**item)
        for item in (layered_objects or [])
    ]
    plan.vector_instructions = [
        item if isinstance(item, VectorInstruction) else VectorInstruction(**item)
        for item in (vector_instructions or [])
    ]
    return plan
