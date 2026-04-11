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
class PagePlan:
    page_number: int
    width_px: int
    height_px: int
    background_path: str
    text_blocks: list[TextBlock] = field(default_factory=list)
    image_blocks: list[ImageBlock] = field(default_factory=list)
