from __future__ import annotations

from .models import LayeredObject


def decompose_page_layers(
    *, page_width: int, page_height: int, objects: list[dict]
) -> list[LayeredObject]:
    layered: list[LayeredObject] = []
    for index, obj in enumerate(objects):
        object_type = obj["object_type"]
        layered.append(
            LayeredObject(
                object_type=object_type,
                left=float(obj["left"]),
                top=float(obj["top"]),
                width=float(obj["width"]),
                height=float(obj["height"]),
                z_index=index,
                rebuildable=object_type != "background",
                must_fallback=object_type == "complex_blend",
            )
        )
    return layered
