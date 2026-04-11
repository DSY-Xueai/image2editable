from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TextBlock:
    text: str
    left: float
    top: float
    width: float
    height: float
    font_size: float
    color: str
    alignment: str
    confidence: float
    font_name: str | None = None
    line_height: float | None = None
    fit_verified: bool = False


@dataclass(slots=True)
class ImageBlock:
    path: str
    left: float
    top: float
    width: float
    height: float
    confidence: float
    extractable: bool = True


@dataclass(slots=True)
class EffectBlock:
    effect_type: str
    left: float
    top: float
    width: float
    height: float
    confidence: float
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class LayeredObject:
    object_type: str
    left: float
    top: float
    width: float
    height: float
    z_index: int
    rebuildable: bool
    must_fallback: bool
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class VectorInstruction:
    shape_type: str
    left: float
    top: float
    width: float
    height: float
    payload: dict = field(default_factory=dict)


@dataclass(slots=True)
class PagePlan:
    page_number: int
    width_px: int
    height_px: int
    background_path: str
    source_type: str = "image"
    page_width_points: float | None = None
    page_height_points: float | None = None
    text_blocks: list[TextBlock] = field(default_factory=list)
    image_blocks: list[ImageBlock] = field(default_factory=list)
    effect_blocks: list[EffectBlock] = field(default_factory=list)
    layered_objects: list[LayeredObject] = field(default_factory=list)
    vector_instructions: list[VectorInstruction] = field(default_factory=list)
