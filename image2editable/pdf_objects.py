"""Low-cost inspection of PDF content streams for editable-page routing.

The parser is intentionally conservative.  Unsupported PDF operators are
recorded instead of being guessed, so a page can safely fall back to the
existing raster/quality-gate pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image as PILImage
from pypdf import PdfReader
from pypdf.generic import ContentStream


SCHEMA_VERSION = 1
_EPSILON = 1e-6
_BOX_EPSILON = 1e-3
_MATRIX_EPSILON = 1e-4
_MAX_LOCAL_PATCH_AREA_RATIO = 0.35
_FONT_FAMILY_ALIASES = {
    "arialmt": "Arial",
    "calibri-light": "Calibri Light",
    "microsoftyahei": "Microsoft YaHei",
}
_SUPPORTED_OPERATORS = {
    b"cm", b"q", b"Q", b"BT", b"ET", b"Tf", b"Tm", b"Td", b"TD",
    b"T*", b"Tj", b"TJ", b"'", b'"', b"TL", b"Tc", b"Tw", b"Tz",
    b"Tr", b"Ts", b"rg", b"RG", b"g", b"G", b"k", b"K", b"sc", b"SC",
    b"cs", b"CS", b"n", b"m", b"l", b"c", b"v", b"y", b"h", b"re",
    b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"S", b"s", b"Do",
    b"W", b"W*", b"sh", b"BI", b"ID", b"EI", b"w", b"BMC", b"BDC",
    b"EMC", b"gs", b"j", b"M",
}


def analyze_pdf_page(
    source: str | Path,
    page_index: int,
    *,
    asset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect one PDF page and optionally persist extracted image assets."""

    reader = PdfReader(str(Path(source).resolve()))
    if type(page_index) is not int or page_index < 0 or page_index >= len(reader.pages):
        raise ValueError("PDF page index is invalid")
    target_dir = Path(asset_dir).resolve() if asset_dir is not None else None
    document = pdfium.PdfDocument(str(Path(source).resolve()))
    try:
        page = document[page_index]
        try:
            return _analyze_open_page(reader, page_index, target_dir, page)
        finally:
            page.close()
    finally:
        document.close()


def _analyze_open_page(
    reader: PdfReader,
    page_index: int,
    asset_dir: Path | None,
    pdfium_page: Any,
) -> dict[str, Any]:
    page = reader.pages[page_index]
    media = page.mediabox
    left, bottom, right, top = (float(value) for value in media)
    width = right - left
    height = top - bottom
    if width <= 0 or height <= 0:
        raise ValueError("PDF page dimensions are invalid")
    context = _ParserContext(page, reader, width, height, asset_dir)
    if _has_visible_annotations(page):
        context.unsupported_features.add("annotations")
    if abs(left) > _EPSILON or abs(bottom) > _EPSILON:
        context.unsupported_features.add("media_box_origin")
    if int(page.get("/Rotate", 0) or 0) % 360:
        context.unsupported_features.add("rotation")
    crop = page.cropbox
    crop_values = [float(value) for value in crop]
    if any(
        abs(value - expected) > _EPSILON
        for value, expected in zip(crop_values, [left, bottom, right, top])
    ):
        context.unsupported_features.add("crop_box")
    contents = page.get_contents()
    if contents is not None:
        for index, (operands, operator) in enumerate(ContentStream(contents, reader).operations):
            before = len(context.objects)
            context.handle(operands, operator)
            for item in context.objects[before:]:
                item["paint_operation_index"] = index
    _replace_text_objects(context, pdfium_page)
    _apply_object_clips(context)
    objects = context.objects
    _mark_unsupported_native_transforms(context, objects)
    page_box = [0.0, 0.0, width, height]
    native_objects = [
        item for item in objects
        if item["type"] in {"text", "image", "shape", "patch"}
    ]
    full_page_image = _has_full_page_image(native_objects, width, height)
    if not context.unsupported_features and native_objects and not full_page_image:
        context.persist_image_assets()
    unsupported = sorted(context.unsupported_features)
    if unsupported:
        classification = "mixed" if native_objects else "raster"
        requires_visual = True
    elif not native_objects:
        classification = "raster"
        requires_visual = True
    elif full_page_image:
        classification = "raster"
        requires_visual = True
    else:
        classification = (
            "hybrid" if any(item["type"] == "patch" for item in objects)
            else "native"
        )
        requires_visual = False
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": page_index,
        "width_pt": width,
        "height_pt": height,
        "objects": objects,
        "unsupported_features": unsupported,
        "classification": classification,
        "requires_visual": requires_visual,
        "localized_features": sorted(context.localized_features),
        "visual_regions": (
            [page_box] if requires_visual else [
                item["bbox_pt"] for item in objects if item["type"] == "patch"
            ]
        ),
    }


def _mark_unsupported_native_transforms(
    context: _ParserContext, objects: list[dict[str, Any]]
) -> None:
    for item in objects:
        if item.get("type") == "image":
            transform = item.get("transform")
            if (
                not isinstance(transform, list)
                or len(transform) != 6
                or abs(float(transform[0])) <= _EPSILON
                or abs(float(transform[3])) <= _EPSILON
                or abs(float(transform[1])) > _MATRIX_EPSILON
                or abs(float(transform[2])) > _MATRIX_EPSILON
            ):
                context.unsupported_features.add("image_transform")
            else:
                if float(transform[0]) < 0:
                    item["flip_horizontal"] = True
                if float(transform[3]) < 0:
                    item["flip_vertical"] = True
        elif item.get("type") == "text":
            matrix = item.get("matrix")
            if (
                not isinstance(matrix, list)
                or len(matrix) != 6
                or abs(float(matrix[1])) > _EPSILON
                or abs(float(matrix[2])) > _EPSILON
            ):
                context.unsupported_features.add("text_transform")


def analyze_pdf_document(
    source: str | Path,
    *,
    asset_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Inspect all pages while writing page-local image assets."""

    resolved = Path(source).resolve()
    reader = PdfReader(str(resolved))
    document = pdfium.PdfDocument(str(resolved))
    root = Path(asset_root).resolve() if asset_root is not None else None
    try:
        results = []
        for index in range(len(reader.pages)):
            page_assets = (
                root / f"page_{index + 1:03d}" / "pdf-assets"
                if root is not None else None
            )
            page = document[index]
            try:
                try:
                    result = _analyze_open_page(reader, index, page_assets, page)
                except Exception:
                    _discard_page_assets(page_assets)
                    result = _analysis_error_result(reader.pages[index], index)
                results.append(result)
            finally:
                page.close()
        return results
    finally:
        document.close()


class _ParserContext:
    def __init__(
        self,
        page: Any,
        reader: PdfReader,
        width: float,
        height: float,
        asset_dir: Path | None,
    ):
        self.page = page
        self.reader = reader
        self.width = width
        self.height = height
        self.asset_dir = asset_dir
        self.objects: list[dict[str, Any]] = []
        self.unsupported_features: set[str] = set()
        self.localized_features: set[str] = set()
        self._ctm = _identity()
        self._stack: list[tuple[Any, ...]] = []
        self._path: list[tuple[float, float]] = []
        self._path_start: tuple[float, float] | None = None
        self._path_closed = False
        self._clip_box = [0.0, 0.0, width, height]
        self._pending_clip: list[float] | None = None
        self._pending_clip_active = False
        self._fill = [0, 0, 0]
        self._stroke = [0, 0, 0]
        self._line_width = 1.0
        self._line_join = 0
        self._miter_limit = 10.0
        self._fill_alpha = 1.0
        self._stroke_alpha = 1.0
        self._text_matrix = _identity()
        self._line_matrix = _identity()
        self._font_name = ""
        self._font_bold = False
        self._font_size = 12.0
        self._leading = 0.0
        self._character_spacing = 0.0
        self._image_count = 0
        self._shape_count = 0
        self._patch_count = 0

    def handle(self, operands: list[Any], operator: bytes) -> None:
        if operator not in _SUPPORTED_OPERATORS:
            self.unsupported_features.add(f"unknown_operator:{operator.decode('latin1', 'replace')}")
            return
        if operator == b"q":
            self._stack.append((
                self._ctm,
                list(self._clip_box),
                list(self._pending_clip) if self._pending_clip is not None else None,
                self._pending_clip_active,
                list(self._fill),
                list(self._stroke),
                self._line_width,
                self._line_join,
                self._miter_limit,
                self._fill_alpha,
                self._stroke_alpha,
                self._font_name,
                self._font_bold,
                self._font_size,
                self._leading,
                self._character_spacing,
            ))
        elif operator == b"Q":
            if self._stack:
                (
                    self._ctm,
                    self._clip_box,
                    self._pending_clip,
                    self._pending_clip_active,
                    self._fill,
                    self._stroke,
                    self._line_width,
                    self._line_join,
                    self._miter_limit,
                    self._fill_alpha,
                    self._stroke_alpha,
                    self._font_name,
                    self._font_bold,
                    self._font_size,
                    self._leading,
                    self._character_spacing,
                ) = self._stack.pop()
            else:
                self.unsupported_features.add("unbalanced_graphics_state")
        elif operator == b"cm":
            if len(operands) == 6:
                self._ctm = _multiply(self._ctm, _numbers(operands))
        elif operator in {b"rg", b"k", b"g"}:
            self._fill = _color(operands, operator)
        elif operator in {b"RG", b"K", b"G"}:
            self._stroke = _color(operands, operator)
        elif operator == b"w" and operands:
            self._line_width = max(0.0, _float(operands[0], 1.0))
        elif operator == b"j" and operands:
            value = _float(operands[0], 0.0)
            if value not in {0.0, 1.0, 2.0}:
                self.unsupported_features.add("line_join")
            else:
                self._line_join = int(value)
        elif operator == b"M" and operands:
            self._miter_limit = max(1.0, _float(operands[0], 10.0))
        elif operator in {b"BMC", b"BDC", b"EMC"}:
            pass
        elif operator == b"gs" and operands:
            self._apply_graphics_state(str(operands[0]))
        elif operator in {b"sc", b"SC", b"cs", b"CS"}:
            self.unsupported_features.add("color_space")
        elif operator == b"BT":
            self._text_matrix = _identity()
            self._line_matrix = _identity()
        elif operator == b"Tf" and len(operands) == 2:
            self._font_name, self._font_bold = _font_details(
                self.page, str(operands[0])
            )
            self._font_size = _float(operands[1], 12.0)
        elif operator == b"TL" and operands:
            self._leading = abs(_float(operands[0], 0.0))
        elif operator == b"Tc" and operands:
            self._character_spacing = _float(operands[0], 0.0)
        elif operator == b"Tw" and operands:
            if abs(_float(operands[0], 0.0)) > _EPSILON:
                self.unsupported_features.add("word_spacing")
        elif operator == b"Tz" and operands:
            if abs(_float(operands[0], 100.0) - 100.0) > _EPSILON:
                self.unsupported_features.add("text_scale")
        elif operator == b"Tr" and operands:
            if abs(_float(operands[0], 0.0)) > _EPSILON:
                self.unsupported_features.add("text_rendering")
        elif operator == b"Ts" and operands:
            if abs(_float(operands[0], 0.0)) > _EPSILON:
                self.unsupported_features.add("text_rise")
        elif operator == b"Tm" and len(operands) == 6:
            self._text_matrix = _multiply(self._ctm, _numbers(operands))
            self._line_matrix = self._text_matrix
        elif operator in {b"Td", b"TD"} and len(operands) == 2:
            tx, ty = _float(operands[0], 0.0), _float(operands[1], 0.0)
            if operator == b"TD":
                self._leading = abs(ty)
            translation = (1.0, 0.0, 0.0, 1.0, tx, ty)
            self._line_matrix = _multiply(self._line_matrix, translation)
            self._text_matrix = self._line_matrix
        elif operator == b"T*":
            self._line_matrix = _multiply(
                self._line_matrix, (1.0, 0.0, 0.0, 1.0, 0.0, -self._leading)
            )
            self._text_matrix = self._line_matrix
        elif operator in {b"Tj", b"TJ", b"'", b'"'}:
            if operator in {b"'", b'"'}:
                self.handle([], b"T*")
            if operator == b"TJ" and any(
                type(item) in {int, float} and abs(float(item)) > _EPSILON
                for item in (operands[0] if operands else [])
            ):
                self.unsupported_features.add("text_positioning")
            if operator == b'"' and any(
                abs(_float(item, 0.0)) > _EPSILON for item in operands[:2]
            ):
                self.unsupported_features.add("text_spacing")
            text = _text_value(operands[-1:] if operator == b'"' else operands)
            if text:
                self._add_text(text)
        elif operator == b"Do" and operands:
            self._add_xobject(str(operands[0]))
        elif operator in {b"W", b"W*"}:
            self._pending_clip = _axis_aligned_path_box(self._path)
            self._pending_clip_active = True
        elif operator == b"sh":
            self.unsupported_features.add("shading")
        elif operator in {b"BI", b"ID", b"EI"}:
            self.unsupported_features.add("inline_image")
        elif operator == b"re" and len(operands) == 4:
            if self._path:
                self.unsupported_features.add("complex_path")
            x, y, w, h = (_float(item, 0.0) for item in operands)
            self._path = [_point(self._ctm, x, y), _point(self._ctm, x + w, y),
                          _point(self._ctm, x + w, y + h), _point(self._ctm, x, y + h)]
            self._path_start = self._path[0]
            self._path_closed = True
        elif operator == b"m" and len(operands) == 2:
            if self._path:
                self.unsupported_features.add("complex_path")
            point = _point(self._ctm, _float(operands[0], 0.0), _float(operands[1], 0.0))
            self._path = [point]
            self._path_start = point
            self._path_closed = False
        elif operator == b"l" and len(operands) == 2:
            self._path.append(_point(self._ctm, _float(operands[0], 0.0), _float(operands[1], 0.0)))
        elif operator == b"c" and len(operands) == 6:
            self._path.extend([
                _point(self._ctm, _float(operands[0], 0.0), _float(operands[1], 0.0)),
                _point(self._ctm, _float(operands[2], 0.0), _float(operands[3], 0.0)),
                _point(self._ctm, _float(operands[4], 0.0), _float(operands[5], 0.0)),
            ])
        elif operator in {b"v", b"y"} and len(operands) == 4:
            self._path.extend([
                _point(self._ctm, _float(operands[0], 0.0), _float(operands[1], 0.0)),
                _point(self._ctm, _float(operands[2], 0.0), _float(operands[3], 0.0)),
            ])
        elif operator == b"h":
            self._path_closed = True
        elif operator in {b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"S", b"s"}:
            self._finish_path(operator)
        elif operator == b"n":
            if self._pending_clip_active:
                self._commit_pending_clip()
            self._path = []
            self._path_start = None
            self._path_closed = False

    def _add_text(self, text: str) -> None:
        matrix = self._text_matrix
        origin = _point(matrix, 0.0, 0.0)
        width = max(self._font_size * 0.25, len(text) * self._font_size * 0.5)
        end = _point(matrix, width, 0.0)
        bottom = _point(matrix, 0.0, -self._font_size * 0.2)
        right = _point(matrix, width, self._font_size * 0.8)
        top = _point(matrix, 0.0, self._font_size * 0.8)
        bbox = _bbox([origin, end, bottom, top, right])
        item = {
            "id": f"text-{len(self.objects) + 1:04d}",
            "type": "text",
            "text": text,
            "font": self._font_name or "Helvetica",
            "font_size": float(self._font_size),
            "bold": self._font_bold,
            "color_rgb": list(self._fill),
            "bbox_pt": bbox,
            "matrix": [float(value) for value in matrix],
            "character_spacing_pt": float(self._character_spacing),
            "fill_opacity": float(self._fill_alpha),
            "z_index": len(self.objects) + 1,
        }
        self._attach_clip(item)
        self.objects.append(item)
        advance = (1.0, 0.0, 0.0, 1.0, width, 0.0)
        self._text_matrix = _multiply(self._text_matrix, advance)

    def _add_xobject(self, resource_name: str) -> None:
        resources = self.page.get("/Resources") or {}
        xobjects = resources.get("/XObject") or {}
        key = resource_name if resource_name.startswith("/") else f"/{resource_name}"
        xobject = xobjects.get(key)
        if xobject is None:
            self.unsupported_features.add("missing_xobject")
            return
        subtype = str(xobject.get("/Subtype"))
        if subtype == "/Form":
            if self._add_rectangular_frame(xobject):
                return
            matrix = _numbers(list(xobject.get("/Matrix", _identity())))
            combined = _multiply(self._ctm, matrix)
            if self._add_text_form(xobject, combined):
                return
            values = [_float(value, 0.0) for value in xobject.get("/BBox", [])]
            if len(values) != 4:
                self.unsupported_features.add("form_xobject")
                return
            left, bottom, right, top = values
            self._add_patch(
                "form_xobject",
                _bbox([
                    _point(combined, left, bottom),
                    _point(combined, right, bottom),
                    _point(combined, right, top),
                    _point(combined, left, top),
                ]),
            )
            return
        if subtype != "/Image":
            self.unsupported_features.add("xobject")
            return
        self._image_count += 1
        bbox = _bbox([_point(self._ctm, 0, 0), _point(self._ctm, 1, 0),
                      _point(self._ctm, 0, 1), _point(self._ctm, 1, 1)])
        item = {
            "id": f"image-{self._image_count:04d}",
            "type": "image",
            "resource": resource_name.lstrip("/"),
            "bbox_pt": bbox,
            "transform": [float(value) for value in self._ctm],
            "z_index": len(self.objects) + 1,
        }
        self._attach_clip(item)
        self.objects.append(item)

    def _add_text_form(self, xobject: Any, combined: tuple) -> bool:
        # Text-only Forms can be read directly without rasterising formulas or
        # running OCR. Other Forms still need their existing visual treatment.
        operations = ContentStream(xobject, self.reader).operations
        if any(operator == b"Do" for _, operator in operations):
            return False
        child = _ParserContext(xobject, self.reader, self.width, self.height, None)
        child._ctm = combined
        for name in (
            "_fill", "_stroke", "_fill_alpha", "_stroke_alpha", "_font_name",
            "_font_bold", "_font_size", "_character_spacing", "_leading",
        ):
            setattr(child, name, getattr(self, name))
        bbox = xobject.get("/BBox")
        if bbox is None or len(bbox) != 4:
            return False
        left, bottom, right, top = (float(value) for value in bbox)
        clip = _box_intersection(self._clip_box, _bbox([
            _point(combined, left, bottom), _point(combined, right, bottom),
            _point(combined, right, top), _point(combined, left, top),
        ]))
        if clip is None:
            return False
        child._clip_box = clip
        for operands, operator in operations:
            child.handle(operands, operator)
        if child.unsupported_features or not child.objects or any(
            item["type"] != "text" for item in child.objects
        ):
            return False
        for item in child.objects:
            item["id"] = f"text-{len(self.objects) + 1:04d}"
            item["z_index"] = len(self.objects) + 1
            self.objects.append(item)
        return True

    def _add_patch(self, feature: str, bbox: list[float], *, z_index=None) -> bool:
        visible = _box_intersection(
            [0.0, 0.0, self.width, self.height], bbox
        )
        if visible is None:
            return True
        left, bottom, right, top = visible
        area = (right - left) * (top - bottom)
        if area > self.width * self.height * _MAX_LOCAL_PATCH_AREA_RATIO:
            self.unsupported_features.add(feature)
            return False
        self._patch_count += 1
        self.localized_features.add(feature)
        self.objects.append({
            "id": f"patch-{self._patch_count:04d}",
            "type": "patch",
            "feature": feature,
            "bbox_pt": visible,
            "z_index": z_index or len(self.objects) + 1,
        })
        return True

    def _add_rectangular_frame(self, xobject: Any) -> bool:
        frame = _rectangular_frame(
            ContentStream(xobject, self.reader).operations,
            _multiply(
                self._ctm,
                _numbers(list(xobject.get("/Matrix", _identity()))),
            ),
        )
        if frame is None:
            return False
        color, boxes = frame
        for box in boxes:
            self._shape_count += 1
            item = {
                "id": f"shape-{self._shape_count:04d}",
                "type": "shape",
                "shape_type": "rectangle",
                "bbox_pt": box,
                "fill_rgb": color,
                "stroke_rgb": None,
                "line_width": 0.0,
                "fill_opacity": float(self._fill_alpha),
                "z_index": len(self.objects) + 1,
            }
            self._attach_clip(item)
            self.objects.append(item)
        return True

    def _apply_graphics_state(self, resource_name: str) -> None:
        resources = self.page.get("/Resources") or {}
        states = resources.get("/ExtGState") or {}
        key = resource_name if resource_name.startswith("/") else f"/{resource_name}"
        state = states.get(key)
        if state is None:
            self.unsupported_features.add("graphics_state")
            return
        state = state.get_object()
        allowed = {"/Type", "/BM", "/ca", "/CA"}
        state_type = state.get("/Type")
        if (
            any(str(name) not in allowed for name in state)
            or (state_type is not None and str(state_type) != "/ExtGState")
            or str(state.get("/BM", "/Normal")) != "/Normal"
        ):
            self.unsupported_features.add("graphics_state")
            return
        if "/ca" in state:
            self._fill_alpha = max(
                0.0, min(1.0, _float(state.get("/ca"), 1.0))
            )
        if "/CA" in state:
            self._stroke_alpha = max(
                0.0, min(1.0, _float(state.get("/CA"), 1.0))
            )

    def _commit_pending_clip(self) -> None:
        self._pending_clip_active = False
        if self._pending_clip is None:
            self.unsupported_features.add("clip")
            return
        candidate = self._pending_clip
        self._pending_clip = None
        if _box_contains(candidate, self._clip_box):
            return
        intersection = _box_intersection(candidate, self._clip_box)
        if intersection is None:
            self.unsupported_features.add("clip")
            return
        self._clip_box = intersection

    def _attach_clip(self, item: dict[str, Any]) -> None:
        item["_clip_box_pt"] = list(self._clip_box)

    def persist_image_assets(self) -> None:
        if self.asset_dir is None:
            return
        for item in self.objects:
            if item["type"] != "image":
                continue
            resource = item["resource"]
            try:
                image = next(
                    candidate.image
                    for candidate in self.page.images
                    if candidate.name.rsplit(".", 1)[0] == resource
                    or candidate.name.startswith(resource + ".")
                )
                if image is None:
                    raise ValueError("PDF image has no decodable payload")
                if item.get("flip_horizontal"):
                    transformed = image.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
                    image.close()
                    image = transformed
                if item.get("flip_vertical"):
                    transformed = image.transpose(PILImage.Transpose.FLIP_TOP_BOTTOM)
                    image.close()
                    image = transformed
                self.asset_dir.mkdir(parents=True, exist_ok=True)
                target = self.asset_dir / f"{item['id']}.png"
                try:
                    image.save(target, format="PNG")
                finally:
                    image.close()
            except Exception:
                self.unsupported_features.add("image_decode")
                target = self.asset_dir / f"{item['id']}.png"
                target.unlink(missing_ok=True)
                continue
            item["asset_path"] = str(target)

    def _finish_path(self, operator: bytes) -> None:
        points = self._path[:]
        if self._path_closed and self._path_start is not None and points:
            points.append(self._path_start)
        if len(points) < 2:
            self._path = []
            return
        fill = operator in {b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}
        stroke = operator in {b"B", b"B*", b"b", b"b*", b"S", b"s"}
        stroke_scale = _uniform_stroke_scale(self._ctm) if stroke else 1.0
        if stroke_scale is None:
            self.unsupported_features.add("stroke_transform")
        if len(points) == 5 and points[-1] == points[0] and _is_axis_aligned_rectangle(points[:-1]):
            shape_type = "rectangle"
        elif len(points) == 2 and stroke:
            shape_type = "line"
        else:
            self._add_patch("complex_path", _bbox(points))
            self._path = []
            self._path_start = None
            self._path_closed = False
            return
        if stroke and shape_type != "line" and self._line_join != 0:
            self.unsupported_features.add("line_join")
        item = {
            "id": f"shape-{self._shape_count + 1:04d}",
            "type": "shape",
            "shape_type": shape_type,
            "bbox_pt": _bbox(points),
            "fill_rgb": list(self._fill) if fill else None,
            "stroke_rgb": list(self._stroke) if stroke else None,
            "line_width": self._line_width * (stroke_scale or 1.0),
            "fill_opacity": float(self._fill_alpha),
            "stroke_opacity": float(self._stroke_alpha),
            "z_index": len(self.objects) + 1,
        }
        if shape_type == "line":
            item["line_start"] = [float(value) for value in points[0]]
            item["line_end"] = [float(value) for value in points[1]]
        self._shape_count += 1
        self._attach_clip(item)
        self.objects.append(item)
        self._path = []
        self._path_start = None
        self._path_closed = False


def _pdfium_text_objects(page: Any, form=None, parent=None, depth=0):
    if depth > 15:
        raise ValueError("PDF Form nesting exceeds text inspection limit")
    parent = _identity() if parent is None else parent
    for item in page.get_objects(max_depth=1, form=form):
        if isinstance(item, pdfium.PdfTextObj):
            yield item, parent
        elif item.type == pdfium.raw.FPDF_PAGEOBJ_FORM:
            combined = _multiply(parent, tuple(item.get_matrix().get()))
            yield from _pdfium_text_objects(page, item.raw, combined, depth + 1)


def _replace_text_objects(context: _ParserContext, page: Any) -> None:
    parsed = [item for item in context.objects if item["type"] == "text"]
    text_page = None
    try:
        text_page = page.get_textpage()
        exact = []
        for item, parent in _pdfium_text_objects(page):
            bound = pdfium.PdfTextObj(item.raw, textpage=text_page)
            text = bound.extract()
            font = item.get_font()
            font_name = _normalize_font_name(font.get_base_name())
            matrix = list(_multiply(parent, tuple(item.get_matrix().get())))
            text_scale = _axis_aligned_text_scale(matrix)
            if text_scale is None:
                context.unsupported_features.add("text_transform")
                text_scale = 1.0
            left, bottom, right, top = item.get_bounds()
            bounds = _bbox([
                _point(parent, left, bottom), _point(parent, right, bottom),
                _point(parent, right, top), _point(parent, left, top),
            ])
            bounds[0] = min(bounds[0], matrix[4])
            exact.append({
                "text": text,
                "font": font_name,
                "font_size": float(item.get_font_size()) * text_scale,
                "bold": font.get_weight() >= 600 or "bold" in font_name.casefold(),
                "italic": any(
                    marker in font_name.casefold()
                    for marker in ("italic", "oblique")
                ),
                "bbox_pt": [round(value, 6) for value in bounds],
                "matrix": matrix,
            })
    except Exception:
        context.unsupported_features.add("text_metrics")
        return
    finally:
        if text_page is not None:
            text_page.close()
    if len(parsed) != len(exact):
        context.unsupported_features.add("text_object_mapping")
        return
    for current, replacement in zip(parsed, exact, strict=True):
        text = replacement["text"]
        if not text:
            context.unsupported_features.add("text_decode")
            continue
        if "\ufffd" in text or any(
            ord(character) < 32 and character not in "\t\r\n"
            for character in text
        ):
            context.unsupported_features.add("text_decode")
        current.update(replacement)


def _apply_object_clips(context: _ParserContext) -> None:
    remove = []
    for item in list(context.objects):
        clip = item.pop("_clip_box_pt", None)
        if item.get("type") == "patch":
            continue
        bbox = item.get("bbox_pt")
        if not isinstance(clip, list) or not isinstance(bbox, list):
            context.unsupported_features.add("clip")
            continue
        if _box_contains(clip, bbox):
            continue
        intersection = _box_intersection(clip, bbox)
        if intersection is None:
            remove.append(item)
            continue
        if item.get("type") == "text":
            context.unsupported_features.add("text_clip")
            continue
        if item.get("type") != "image":
            if context._add_patch(
                "clip", intersection, z_index=item.get("z_index")
            ):
                context.objects[-1]["paint_operation_index"] = item["paint_operation_index"]
                remove.append(item)
                continue
            context.unsupported_features.add("clip")
            continue
        left, bottom, right, top = (float(value) for value in bbox)
        width = right - left
        height = top - bottom
        if width <= _EPSILON or height <= _EPSILON:
            context.unsupported_features.add("clip")
            continue
        clipped_left, clipped_bottom, clipped_right, clipped_top = intersection
        item["bbox_pt"] = intersection
        item["crop"] = {
            "left": max(0.0, min(1.0, (clipped_left - left) / width)),
            "top": max(0.0, min(1.0, (top - clipped_top) / height)),
            "right": max(0.0, min(1.0, (right - clipped_right) / width)),
            "bottom": max(0.0, min(1.0, (clipped_bottom - bottom) / height)),
        }
    for item in remove:
        context.objects.remove(item)


def _rectangular_frame(
    operations: list[tuple[list[Any], bytes]],
    transform: tuple[float, ...],
) -> tuple[list[int], list[list[float]]] | None:
    fill = [0, 0, 0]
    paths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] | None = None
    for operands, operator in operations:
        if operator == b"rg":
            fill = _color(operands, operator)
        elif operator == b"m" and len(operands) == 2:
            current = [_point(
                transform,
                _float(operands[0], 0.0),
                _float(operands[1], 0.0),
            )]
        elif operator == b"l" and current is not None and len(operands) == 2:
            current.append(_point(
                transform,
                _float(operands[0], 0.0),
                _float(operands[1], 0.0),
            ))
        elif operator == b"h" and current is not None:
            paths.append(current)
            current = None
        elif operator == b"n":
            paths = []
            current = None
        elif operator in {b"f", b"f*"}:
            if len(paths) != 2 or any(
                not _is_axis_aligned_rectangle(
                    path, tolerance=_frame_rectangle_tolerance(path)
                )
                for path in paths
            ):
                return None
            outer, inner = sorted(
                (_bbox(path) for path in paths),
                key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
                reverse=True,
            )
            if not _box_contains(outer, inner):
                return None
            left, bottom, right, top = outer
            inner_left, inner_bottom, inner_right, inner_top = inner
            boxes = [
                [left, bottom, inner_left, top],
                [inner_right, bottom, right, top],
                [inner_left, bottom, inner_right, inner_bottom],
                [inner_left, inner_top, inner_right, top],
            ]
            if any(
                box[2] - box[0] <= _EPSILON
                or box[3] - box[1] <= _EPSILON
                for box in boxes
            ):
                return None
            return fill, boxes
        elif operator in {
            b"Tj", b"TJ", b"'", b'"', b"Do", b"sh", b"BI", b"ID", b"EI"
        }:
            return None
    return None


def _frame_rectangle_tolerance(points: list[tuple[float, float]]) -> float:
    left, bottom, right, top = _bbox(points)
    longest_edge = max(right - left, top - bottom)
    return min(0.75, max(0.1, longest_edge * 0.001))


def _identity() -> tuple[float, float, float, float, float, float]:
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    a, b, c, d, e, f = left
    g, h, i, j, k, offset_y = right
    return (
        a * g + c * h,
        b * g + d * h,
        a * i + c * j,
        b * i + d * j,
        a * k + c * offset_y + e,
        b * k + d * offset_y + f,
    )


def _point(matrix: tuple[float, ...], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _numbers(values: list[Any]) -> tuple[float, ...]:
    return tuple(_float(value, 0.0) for value in values)


def _float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _text_value(operands: list[Any]) -> str:
    if not operands:
        return ""
    value = operands[0]
    if isinstance(value, list):
        return "".join(_text_value([item]) for item in value if isinstance(item, (str, bytes)))
    if isinstance(value, bytes):
        return value.decode("latin1", "replace")
    return str(value)


def _color(operands: list[Any], operator: bytes) -> list[int]:
    values = [_float(item, 0.0) for item in operands]
    if operator in {b"g", b"G"}:
        values = [values[0] if values else 0.0] * 3
    elif operator in {b"k", b"K"}:
        c, m, y, k = (values + [0.0] * 4)[:4]
        values = [1 - min(1, c + k), 1 - min(1, m + k), 1 - min(1, y + k)]
    values = (values + [0.0, 0.0, 0.0])[:3]
    return [max(0, min(255, round(value * 255))) for value in values]


def _font_details(page: Any, resource_name: str) -> tuple[str, bool]:
    resources = page.get("/Resources") or {}
    fonts = resources.get("/Font") or {}
    key = resource_name if resource_name.startswith("/") else f"/{resource_name}"
    font = fonts.get(key)
    base = str(font.get("/BaseFont")) if font is not None else resource_name
    name = _normalize_font_name(base)
    bold = "bold" in name.casefold()
    for suffix in ("-BoldItalic", "-BoldOblique", "-Bold"):
        if name.casefold().endswith(suffix.casefold()):
            name = name[:-len(suffix)]
            break
    return name or "Arial", bold


def _normalize_font_name(value: object) -> str:
    name = str(value).lstrip("/").split("+", 1)[-1] or "Arial"
    return _FONT_FAMILY_ALIASES.get(name.casefold(), name)


def _bbox(points: list[tuple[float, float]]) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [round(min(xs), 6), round(min(ys), 6), round(max(xs), 6), round(max(ys), 6)]


def _is_axis_aligned_rectangle(
    points: list[tuple[float, float]],
    *,
    tolerance: float = _BOX_EPSILON,
) -> bool:
    if len(points) == 5 and _points_close(points[0], points[-1]):
        points = points[:-1]
    if len(points) != 4:
        return False
    xs = _cluster_coordinates(
        (point[0] for point in points), tolerance=tolerance
    )
    ys = _cluster_coordinates(
        (point[1] for point in points), tolerance=tolerance
    )
    if len(xs) != 2 or len(ys) != 2:
        return False
    corners = {(x, y) for x in xs for y in ys}
    actual = {
        (
            min(xs, key=lambda value: abs(value - point[0])),
            min(ys, key=lambda value: abs(value - point[1])),
        )
        for point in points
    }
    return actual == corners and all(
        math.isclose(first[0], second[0], abs_tol=tolerance)
        or math.isclose(first[1], second[1], abs_tol=tolerance)
        for first, second in zip(points, points[1:] + points[:1])
    )


def _cluster_coordinates(values: Any, *, tolerance: float) -> list[float]:
    clustered: list[float] = []
    for value in sorted(float(item) for item in values):
        if not clustered or not math.isclose(
            value, clustered[-1], abs_tol=tolerance
        ):
            clustered.append(value)
    return clustered


def _points_close(
    first: tuple[float, float], second: tuple[float, float]
) -> bool:
    return (
        math.isclose(first[0], second[0], abs_tol=_BOX_EPSILON)
        and math.isclose(first[1], second[1], abs_tol=_BOX_EPSILON)
    )


def _axis_aligned_path_box(points: list[tuple[float, float]]) -> list[float] | None:
    if not _is_axis_aligned_rectangle(points):
        return None
    return _bbox(points)


def _box_contains(outer: list[float], inner: list[float]) -> bool:
    return (
        outer[0] <= inner[0] + _BOX_EPSILON
        and outer[1] <= inner[1] + _BOX_EPSILON
        and outer[2] >= inner[2] - _BOX_EPSILON
        and outer[3] >= inner[3] - _BOX_EPSILON
    )


def _box_intersection(first: list[float], second: list[float]) -> list[float] | None:
    intersection = [
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    ]
    if (
        intersection[2] - intersection[0] <= _BOX_EPSILON
        or intersection[3] - intersection[1] <= _BOX_EPSILON
    ):
        return None
    return [float(value) for value in intersection]


def _uniform_stroke_scale(matrix: tuple[float, ...]) -> float | None:
    a, b, c, d, _, _ = matrix
    first = math.hypot(a, b)
    second = math.hypot(c, d)
    tolerance = _EPSILON * max(1.0, first, second)
    if (
        first <= _EPSILON
        or second <= _EPSILON
        or abs(first - second) > tolerance
        or abs(a * c + b * d) > _EPSILON * max(1.0, first * second)
    ):
        return None
    return (first + second) / 2.0


def _axis_aligned_text_scale(matrix: list[float]) -> float | None:
    a, b, c, d, _, _ = matrix
    tolerance = _EPSILON * max(1.0, abs(a), abs(d))
    if (
        a <= _EPSILON
        or d <= _EPSILON
        or abs(b) > tolerance
        or abs(c) > tolerance
        or abs(a - d) > tolerance
    ):
        return None
    return (a + d) / 2.0


def _has_visible_annotations(page: Any) -> bool:
    annotations = page.get("/Annots") or []
    for reference in annotations:
        annotation = reference.get_object()
        flags = int(annotation.get("/F", 0) or 0)
        if flags & (1 | 2 | 32):
            continue
        if str(annotation.get("/Subtype")) == "/Link" and "/AP" not in annotation:
            border = annotation.get("/Border")
            if border is not None and len(border) >= 3 and _float(border[2], 1.0) == 0:
                continue
        return True
    return False


def _analysis_error_result(page: Any, page_index: int) -> dict[str, Any]:
    media = page.mediabox
    left, bottom, right, top = (float(value) for value in media)
    width = right - left
    height = top - bottom
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": page_index,
        "width_pt": width,
        "height_pt": height,
        "objects": [],
        "unsupported_features": ["analysis_error"],
        "classification": "raster",
        "requires_visual": True,
        "visual_regions": [[0.0, 0.0, width, height]],
    }


def _discard_page_assets(asset_dir: Path | None) -> None:
    if asset_dir is None or not asset_dir.is_dir():
        return
    for path in asset_dir.iterdir():
        if path.is_file():
            path.unlink()
    try:
        asset_dir.rmdir()
    except OSError:
        pass


def _has_full_page_image(objects: list[dict], width: float, height: float) -> bool:
    images = [item for item in objects if item["type"] == "image"]
    return any(
        max(0.0, right - left) * max(0.0, top - bottom)
        >= width * height * 0.95
        and left <= width * 0.02
        and bottom <= height * 0.02
        and right >= width * 0.98
        and top >= height * 0.98
        for left, bottom, right, top in (
            item["bbox_pt"] for item in images
        )
    )
