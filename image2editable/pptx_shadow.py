from __future__ import annotations

from copy import copy, deepcopy
import os
from pathlib import Path
import posixpath
import tempfile
import zipfile

from lxml import etree


P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"p": P, "a": A, "r": R}

SHAPE_TAGS = {
    f"{{{P}}}sp",
    f"{{{P}}}grpSp",
    f"{{{P}}}graphicFrame",
    f"{{{P}}}cxnSp",
    f"{{{P}}}pic",
    f"{{{P}}}contentPart",
}


def patch_slide_background(
    source_pptx: str | Path,
    donor_pptx: str | Path,
    output_pptx: str | Path,
    *,
    slide_part: str,
    source_shape_id: str = "background",
) -> dict:
    """Replace one screenshot background with donor reconstruction shapes."""
    source = Path(source_pptx).resolve()
    donor = Path(donor_pptx).resolve()
    output = Path(output_pptx).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not donor.is_file():
        raise FileNotFoundError(donor)

    normalized_slide_part = _normalize_package_part(slide_part)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with (
            zipfile.ZipFile(source) as source_archive,
            zipfile.ZipFile(donor) as donor_archive,
        ):
            donor_slide_part = _first_slide_part(donor_archive)
            replacements, media, result = _build_replacements(
                source_archive,
                donor_archive,
                normalized_slide_part,
                donor_slide_part,
                source_shape_id,
            )
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output.stem}-",
                suffix=".pptx",
                dir=output.parent,
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            with zipfile.ZipFile(temporary_path, "w") as destination:
                for info in source_archive.infolist():
                    contents = replacements.get(
                        info.filename,
                        source_archive.read(info.filename),
                    )
                    destination.writestr(info, contents)
                for part_name, donor_info, contents in media:
                    media_info = copy(donor_info)
                    media_info.filename = part_name
                    destination.writestr(media_info, contents)
        os.link(temporary_path, output)
        return result
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _build_replacements(
    source: zipfile.ZipFile,
    donor: zipfile.ZipFile,
    source_slide_part: str,
    donor_slide_part: str,
    source_shape_id: str,
) -> tuple[dict[str, bytes], list[tuple[str, zipfile.ZipInfo, bytes]], dict]:
    source_names = set(source.namelist())
    if source_slide_part not in source_names:
        raise ValueError(f"slide part is missing: {source_slide_part}")

    source_rels_part = _relationships_part(source_slide_part)
    donor_rels_part = _relationships_part(donor_slide_part)
    source_slide = etree.fromstring(source.read(source_slide_part))
    donor_slide = etree.fromstring(donor.read(donor_slide_part))
    source_rels = etree.fromstring(source.read(source_rels_part))
    donor_rels = etree.fromstring(donor.read(donor_rels_part))

    common = source_slide.find(f"{{{P}}}cSld")
    source_tree = (
        common.find(f"{{{P}}}spTree") if common is not None else None
    )
    if common is None or source_tree is None:
        raise ValueError("target slide has no shape tree")
    source_object, insertion_index, target_bounds = _source_screenshot_object(
        source_slide,
        common,
        source_tree,
        source_shape_id,
    )
    source_relationships = _referenced_relationship_ids(source_object)
    remaining_relationships = _referenced_relationship_ids(source_slide)
    for relationship_id in source_relationships - remaining_relationships:
        relationship = _relationship_by_id(source_rels, relationship_id)
        if (
            relationship is not None
            and relationship.get("Type", "").endswith("/image")
        ):
            source_rels.remove(relationship)

    donor_tree = donor_slide.find(f"{{{P}}}cSld/{{{P}}}spTree")
    if donor_tree is None:
        raise ValueError("source or donor slide has no shape tree")
    donor_shapes = [
        deepcopy(child) for child in donor_tree if child.tag in SHAPE_TAGS
    ]
    if not donor_shapes:
        raise ValueError("donor slide contains no reconstruction shapes")

    source_size = _slide_size(source)
    donor_size = _slide_size(donor)
    if target_bounds is None:
        target_bounds = (0, 0, source_size[0], source_size[1])
    for shape in donor_shapes:
        _map_shape(shape, donor_size, target_bounds)
    _assign_shape_ids(
        source_slide,
        donor_shapes,
        reserved_ids={source_shape_id},
    )

    occupied_parts = set(source_names)
    occupied_relationship_ids = {
        item.get("Id")
        for item in source_rels.findall(f"{{{PR}}}Relationship")
    }
    next_relationship_number = 1
    relationship_map: dict[str, str] = {}
    imported_parts: dict[str, str] = {}
    media: list[tuple[str, zipfile.ZipInfo, bytes]] = []

    for donor_relationship_id in _referenced_relationship_ids(donor_shapes):
        donor_relationship = _relationship_by_id(
            donor_rels,
            donor_relationship_id,
        )
        if donor_relationship is None:
            raise ValueError(
                f"donor relationship is missing: {donor_relationship_id}"
            )
        while (
            f"rId{next_relationship_number}" in occupied_relationship_ids
        ):
            next_relationship_number += 1
        new_relationship_id = f"rId{next_relationship_number}"
        next_relationship_number += 1
        occupied_relationship_ids.add(new_relationship_id)
        relationship_map[donor_relationship_id] = new_relationship_id

        imported_relationship = deepcopy(donor_relationship)
        imported_relationship.set("Id", new_relationship_id)
        relationship_type = donor_relationship.get("Type", "")
        target_mode = donor_relationship.get("TargetMode")
        if relationship_type.endswith("/image") and target_mode != "External":
            donor_media_part = _resolve_target(
                donor_slide_part,
                donor_relationship.get("Target"),
            )
            if donor_media_part not in donor.namelist():
                raise ValueError(
                    f"donor image part is missing: {donor_media_part}"
                )
            imported_media_part = imported_parts.get(donor_media_part)
            if imported_media_part is None:
                imported_media_part = _unique_media_part(
                    donor_media_part,
                    occupied_parts,
                )
                imported_parts[donor_media_part] = imported_media_part
                occupied_parts.add(imported_media_part)
                media.append(
                    (
                        imported_media_part,
                        donor.getinfo(donor_media_part),
                        donor.read(donor_media_part),
                    )
                )
            imported_relationship.set(
                "Target",
                posixpath.relpath(
                    imported_media_part,
                    posixpath.dirname(source_slide_part),
                ),
            )
        elif target_mode != "External":
            raise ValueError(
                f"unsupported donor relationship: {relationship_type}"
            )
        source_rels.append(imported_relationship)

    _rewrite_relationship_ids(donor_shapes, relationship_map)
    for offset, shape in enumerate(donor_shapes):
        source_tree.insert(insertion_index + offset, shape)

    content_types = _updated_content_types(
        source,
        donor,
        imported_parts,
    )
    replacements = {
        source_slide_part: _serialize(source_slide),
        source_rels_part: _serialize(source_rels),
    }
    if content_types is not None:
        replacements["[Content_Types].xml"] = content_types
    return (
        replacements,
        media,
        {
            "source_shape_id": source_shape_id,
            "slide_part": source_slide_part,
            "imported_shapes": len(donor_shapes),
            "imported_media": len(media),
        },
    )


def _first_slide_part(archive: zipfile.ZipFile) -> str:
    presentation_part = "ppt/presentation.xml"
    presentation = etree.fromstring(archive.read(presentation_part))
    first_slide = presentation.find("p:sldIdLst/p:sldId", NS)
    if first_slide is None:
        raise ValueError("donor presentation contains no slides")
    relationship_id = first_slide.get(f"{{{R}}}id")
    relationships = etree.fromstring(
        archive.read(_relationships_part(presentation_part))
    )
    relationship = _relationship_by_id(relationships, relationship_id)
    if relationship is None:
        raise ValueError("donor first-slide relationship is missing")
    return _resolve_target(
        presentation_part,
        relationship.get("Target"),
    )


def _slide_size(archive: zipfile.ZipFile) -> tuple[int, int]:
    presentation = etree.fromstring(archive.read("ppt/presentation.xml"))
    size = presentation.find(f"{{{P}}}sldSz")
    if size is None:
        raise ValueError("presentation slide size is missing")
    return int(size.get("cx")), int(size.get("cy"))


def _map_shape(
    shape: etree._Element,
    donor_size: tuple[int, int],
    target_bounds: tuple[int, int, int, int],
) -> None:
    transform = next(
        (
            item
            for item in shape.iter()
            if etree.QName(item).localname == "xfrm"
        ),
        None,
    )
    if transform is None:
        return
    target_x, target_y, target_width, target_height = target_bounds
    scale_x = target_width / donor_size[0]
    scale_y = target_height / donor_size[1]
    for child_name, fields in (
        ("off", (("x", scale_x), ("y", scale_y))),
        ("ext", (("cx", scale_x), ("cy", scale_y))),
    ):
        child = transform.find(f"{{{A}}}{child_name}")
        if child is None:
            continue
        for field, scale in fields:
            if child.get(field) is not None:
                value = round(int(child.get(field)) * scale)
                if child_name == "off":
                    value += target_x if field == "x" else target_y
                child.set(field, str(value))


def _assign_shape_ids(
    source_slide: etree._Element,
    donor_shapes: list[etree._Element],
    *,
    reserved_ids: set[str] | None = None,
) -> None:
    existing = [
        int(item.get("id"))
        for item in source_slide.findall(".//p:cNvPr", NS)
        if (item.get("id") or "").isdigit()
    ]
    existing.extend(
        int(item)
        for item in reserved_ids or set()
        if item.isdigit()
    )
    next_id = max(existing, default=0) + 1
    for shape in donor_shapes:
        for properties in shape.findall(".//p:cNvPr", NS):
            properties.set("id", str(next_id))
            next_id += 1


def _source_screenshot_object(
    source_slide: etree._Element,
    common: etree._Element,
    source_tree: etree._Element,
    source_shape_id: str,
) -> tuple[
    etree._Element,
    int,
    tuple[int, int, int, int] | None,
]:
    if source_shape_id == "background":
        background = common.find(f"{{{P}}}bg")
        if (
            background is None
            or background.find("p:bgPr/a:blipFill", NS) is None
        ):
            raise ValueError(
                "target slide does not contain a screenshot background image"
            )
        common.remove(background)
        return background, _shape_insertion_index(source_tree), None

    for index, shape in enumerate(source_tree):
        properties = shape.find(".//p:cNvPr", NS)
        if (
            shape.tag == f"{{{P}}}pic"
            and properties is not None
            and properties.get("id") == source_shape_id
        ):
            _reject_incoming_shape_references(
                source_slide,
                source_shape_id,
            )
            bounds = _picture_bounds(shape, source_shape_id)
            source_tree.remove(shape)
            return shape, index, bounds
    raise ValueError(
        f"target slide does not contain screenshot picture: {source_shape_id}"
    )


def _picture_bounds(
    picture: etree._Element,
    source_shape_id: str,
) -> tuple[int, int, int, int]:
    transform = picture.find("p:spPr/a:xfrm", NS)
    offset = transform.find("a:off", NS) if transform is not None else None
    extent = transform.find("a:ext", NS) if transform is not None else None
    values = (
        offset.get("x") if offset is not None else None,
        offset.get("y") if offset is not None else None,
        extent.get("cx") if extent is not None else None,
        extent.get("cy") if extent is not None else None,
    )
    if any(value is None for value in values):
        raise ValueError(
            f"screenshot picture has no usable bounds: {source_shape_id}"
        )
    x, y, width, height = (int(value) for value in values)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"screenshot picture has invalid bounds: {source_shape_id}"
        )
    return x, y, width, height


def _reject_incoming_shape_references(
    slide: etree._Element,
    source_shape_id: str,
) -> None:
    connector_reference = any(
        item.get("id") == source_shape_id
        for name in ("stCxn", "endCxn")
        for item in slide.findall(f".//a:{name}", NS)
    )
    timing_reference = any(
        item.get("spid") == source_shape_id
        for item in slide.findall(".//p:spTgt", NS)
    )
    if connector_reference or timing_reference:
        raise ValueError(
            f"screenshot picture is referenced by another object: "
            f"{source_shape_id}"
        )


def _referenced_relationship_ids(elements) -> set[str]:
    if isinstance(elements, etree._Element):
        elements = [elements]
    return {
        value
        for element in elements
        for item in element.iter()
        for name, value in item.attrib.items()
        if etree.QName(name).namespace == R
    }


def _rewrite_relationship_ids(
    shapes: list[etree._Element],
    relationship_map: dict[str, str],
) -> None:
    for shape in shapes:
        for item in shape.iter():
            for name, value in list(item.attrib.items()):
                if (
                    etree.QName(name).namespace == R
                    and value in relationship_map
                ):
                    item.set(name, relationship_map[value])


def _relationship_by_id(
    relationships: etree._Element,
    relationship_id: str | None,
) -> etree._Element | None:
    return next(
        (
            item
            for item in relationships.findall(f"{{{PR}}}Relationship")
            if item.get("Id") == relationship_id
        ),
        None,
    )


def _shape_insertion_index(tree: etree._Element) -> int:
    metadata = {f"{{{P}}}nvGrpSpPr", f"{{{P}}}grpSpPr"}
    index = 0
    while index < len(tree) and tree[index].tag in metadata:
        index += 1
    return index


def _updated_content_types(
    source: zipfile.ZipFile,
    donor: zipfile.ZipFile,
    imported_parts: dict[str, str],
) -> bytes | None:
    source_contents = source.read("[Content_Types].xml")
    source_types = etree.fromstring(source_contents)
    donor_types = etree.fromstring(donor.read("[Content_Types].xml"))
    defaults = {
        item.get("Extension", "").lower()
        for item in source_types.findall(f"{{{CT}}}Default")
    }
    changed = False
    for donor_part, imported_part in imported_parts.items():
        extension = posixpath.splitext(imported_part)[1].lstrip(".").lower()
        if extension in defaults:
            continue
        donor_default = next(
            (
                item
                for item in donor_types.findall(f"{{{CT}}}Default")
                if item.get("Extension", "").lower() == extension
            ),
            None,
        )
        if donor_default is None:
            raise ValueError(
                f"donor content type is missing for: {donor_part}"
            )
        source_types.append(deepcopy(donor_default))
        defaults.add(extension)
        changed = True
    return _serialize(source_types) if changed else None


def _unique_media_part(part: str, occupied: set[str]) -> str:
    directory, filename = posixpath.split(part)
    stem, suffix = posixpath.splitext(filename)
    candidate = part
    number = 2
    while candidate in occupied:
        candidate = posixpath.join(directory, f"{stem}{number}{suffix}")
        number += 1
    return candidate


def _relationships_part(part: str) -> str:
    directory, filename = posixpath.split(part)
    return posixpath.join(directory, "_rels", f"{filename}.rels")


def _resolve_target(owner_part: str, target: str | None) -> str:
    if not target:
        raise ValueError("relationship target is missing")
    return _normalize_package_part(
        posixpath.join(posixpath.dirname(owner_part), target)
    )


def _normalize_package_part(part: str) -> str:
    normalized = posixpath.normpath(part.replace("\\", "/")).lstrip("/")
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"package part escapes archive root: {part}")
    return normalized


def _serialize(element: etree._Element) -> bytes:
    return etree.tostring(
        element,
        encoding="UTF-8",
        xml_declaration=True,
        standalone=True,
    )
