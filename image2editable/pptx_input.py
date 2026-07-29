"""Read-only OOXML structure scanning for PPTX input files."""

from __future__ import annotations

import hashlib
import posixpath
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from image2editable.inputs import sha256_file


P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
NS = {"p": P, "a": A, "r": R, "pr": PR, "ct": CT, "mc": MC}

MAX_MEMBERS = 10_000
MAX_PART_SIZE = 512 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 2 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
MAX_XML_SIZE = 16 * 1024 * 1024
MAX_IMAGE_READ_SIZE = 128 * 1024 * 1024
_HASH_CHUNK_SIZE = 1024 * 1024
PRESENTATION_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
SLIDE_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
RELS_CONTENT_TYPE = "application/vnd.openxmlformats-package.relationships+xml"
NOTES_SLIDE_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"
SLIDE_RELATIONSHIP_TYPES = frozenset(
    {
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
        "http://purl.oclc.org/ooxml/officeDocument/relationships/slide",
    }
)
NOTES_SLIDE_RELATIONSHIP_TYPES = frozenset(
    {
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide",
        "http://purl.oclc.org/ooxml/officeDocument/relationships/notesSlide",
    }
)
SUPPORTED_MC_NAMESPACE_URIS = frozenset({P, A, R})
SHAPE_TAGS = frozenset(
    {
        f"{{{P}}}sp",
        f"{{{P}}}pic",
        f"{{{P}}}grpSp",
        f"{{{P}}}cxnSp",
        f"{{{P}}}graphicFrame",
        f"{{{P}}}contentPart",
    }
)
SHAPE_TREE_METADATA_TAGS = frozenset(
    {
        f"{{{P}}}nvGrpSpPr",
        f"{{{P}}}grpSpPr",
        f"{{{P}}}extLst",
    }
)


def scan_pptx(path: str | Path) -> dict[str, object]:
    """Return a stable, JSON-compatible inventory without modifying *path*."""
    source = Path(path).resolve()
    if not source.is_file() or source.suffix.casefold() != ".pptx":
        raise ValueError(f"PPTX input must be an existing .pptx file: {source}")
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Cannot open PPTX ZIP: {source}") from error
    try:
        try:
            names = _validate_archive(archive)
            parts = {
                name: _sha256_member(archive, name)
                for name in sorted(names)
            }
            content_types = _content_types(archive, names)
            _require_content_type(
                content_types,
                "ppt/presentation.xml",
                PRESENTATION_CONTENT_TYPE,
            )
            presentation = _xml(archive, "ppt/presentation.xml", names)
            presentation_rels = _relationships(
                archive, "ppt/_rels/presentation.xml.rels", "ppt/presentation.xml", names, content_types
            )
            size = presentation.find("p:sldSz", NS)
            if size is None or size.get("cx") is None or size.get("cy") is None:
                raise ValueError("Missing slide size in ppt/presentation.xml")
            slides: list[dict[str, object]] = []
            ids = presentation.findall("p:sldIdLst/p:sldId", NS)
            for index, slide_id in enumerate(ids, start=1):
                rel_id = slide_id.get(f"{{{R}}}id")
                if not rel_id or rel_id not in presentation_rels:
                    raise ValueError(f"Missing presentation relationship for slide {index}")
                relation = presentation_rels[rel_id]
                if relation["target_mode"] == "External" or not relation["target"]:
                    raise ValueError(f"Slide relationship must target an internal part: {rel_id}")
                if relation["type"] not in SLIDE_RELATIONSHIP_TYPES:
                    raise ValueError(f"Invalid slide relationship type: {rel_id}")
                slide_part = relation["target"]
                _require_content_type(
                    content_types,
                    slide_part,
                    SLIDE_CONTENT_TYPE,
                )
                slide, choice_namespace_scopes = _xml_with_namespace_scopes(
                    archive,
                    slide_part,
                    names,
                )
                slide_rels_name = _rels_name(slide_part)
                slide_rels = (
                    _relationships(archive, slide_rels_name, slide_part, names, content_types)
                    if slide_rels_name in names
                    else {}
                )
                timing_ids = {
                    node.get("spid")
                    for node in slide.findall(".//p:spTgt", NS)
                    if node.get("spid") is not None
                }
                notes_part, notes_sha256 = _notes(
                    slide_rels, parts, archive, names
                )
                tree = slide.find("p:cSld/p:spTree", NS)
                if tree is None:
                    raise ValueError(f"Missing shape tree in {slide_part}")
                objects = _objects(
                    tree,
                    slide_part,
                    slide_rels,
                    timing_ids,
                    archive,
                    names,
                    parts,
                    choice_namespace_scopes,
                )
                slides.append(
                    {
                        "slide_index": index,
                        "slide_part": slide_part,
                        "sha256": parts[slide_part],
                        "notes_part": notes_part,
                        "notes_sha256": notes_sha256,
                        "objects": objects,
                    }
                )
            return {
                "schema_version": 1,
                "source": str(source),
                "source_sha256": sha256_file(source),
                "slide_count": len(slides),
                "slide_width": int(size.get("cx")),
                "slide_height": int(size.get("cy")),
                "parts": parts,
                "slides": slides,
            }
        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
            raise ValueError(f"Cannot read PPTX ZIP: {source}") from error
    finally:
        archive.close()


def _validate_archive(archive: zipfile.ZipFile) -> set[str]:
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ValueError(f"PPTX ZIP has too many members: {len(infos)}")
    names: set[str] = set()
    normalized_names: set[str] = set()
    total_size = 0
    for info in infos:
        name = info.filename
        if info.is_dir():
            raise ValueError(f"Directory ZIP member is not allowed: {name}")
        if (
            not name
            or posixpath.isabs(name)
            or "\\" in name
            or "\x00" in name
        ):
            raise ValueError(f"Unsafe ZIP member name: {name}")
        normalized = posixpath.normpath(name)
        if (
            normalized in {".", ".."}
            or normalized.startswith("../")
            or normalized != name
        ):
            raise ValueError(f"Unsafe ZIP member name: {name}")
        if name in names:
            raise ValueError(f"Duplicate ZIP member: {name}")
        if normalized in normalized_names:
            raise ValueError(f"Duplicate normalized ZIP member: {name}")
        if info.flag_bits & 1:
            raise ValueError(f"Encrypted ZIP member: {name}")
        if info.file_size > MAX_PART_SIZE:
            raise ValueError(f"PPTX ZIP member is too large: {name}")
        total_size += info.file_size
        if total_size > MAX_TOTAL_UNCOMPRESSED:
            raise ValueError("PPTX ZIP total uncompressed size is too large")
        if info.file_size:
            if (
                info.compress_size == 0
                or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
            ):
                raise ValueError(f"PPTX ZIP member compression ratio is too high: {name}")
        names.add(name)
        normalized_names.add(normalized)
    return names


def _sha256_member(archive: zipfile.ZipFile, part: str) -> str:
    digest = hashlib.sha256()
    with archive.open(part) as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _read_xml_bytes(
    archive: zipfile.ZipFile,
    part: str,
    names: set[str],
) -> bytes:
    if part not in names:
        raise ValueError(f"Missing required PPTX part: {part}")
    if archive.getinfo(part).file_size > MAX_XML_SIZE:
        raise ValueError(f"XML part is too large: {part}")
    data = archive.read(part)
    lowered = data.lower()
    null_stripped = lowered.replace(b"\x00", b"")
    if (
        b"<!doctype" in lowered
        or b"<!entity" in lowered
        or b"<!doctype" in null_stripped
        or b"<!entity" in null_stripped
    ):
        raise ValueError(f"Unsafe XML in {part}: DOCTYPE and ENTITY are not allowed")
    return data


def _xml(archive: zipfile.ZipFile, part: str, names: set[str]) -> ET.Element:
    data = _read_xml_bytes(archive, part, names)
    try:
        return ET.fromstring(data)
    except ET.ParseError as error:
        raise ValueError(f"Invalid XML in {part}: {error}") from error


def _xml_with_namespace_scopes(
    archive: zipfile.ZipFile,
    part: str,
    names: set[str],
) -> tuple[ET.Element, dict[int, dict[str, str]]]:
    data = _read_xml_bytes(archive, part, names)
    namespace_scopes: dict[int, dict[str, str]] = {}
    scope_stack: list[dict[str, str]] = []
    pending_namespaces: list[tuple[str, str]] = []
    try:
        parser = ET.iterparse(
            BytesIO(data),
            events=("start-ns", "start", "end"),
        )
        for event, item in parser:
            if event == "start-ns":
                pending_namespaces.append(item)
                continue
            if event == "start":
                scope = dict(scope_stack[-1]) if scope_stack else {}
                scope.update(pending_namespaces)
                pending_namespaces.clear()
                scope_stack.append(scope)
                if item.tag == f"{{{MC}}}Choice":
                    namespace_scopes[id(item)] = scope
                continue
            scope_stack.pop()
        return parser.root, namespace_scopes
    except ET.ParseError as error:
        raise ValueError(f"Invalid XML in {part}: {error}") from error


def _content_types(archive: zipfile.ZipFile, names: set[str]) -> dict[str, str]:
    root = _xml(archive, "[Content_Types].xml", names)
    defaults: dict[str, str] = {}
    for node in root.findall("ct:Default", NS):
        extension = node.get("Extension", "").casefold()
        if extension in defaults:
            raise ValueError(f"Duplicate content type Default: {extension}")
        defaults[extension] = node.get("ContentType", "")
    overrides: dict[str, str] = {}
    for node in root.findall("ct:Override", NS):
        part_name = node.get("PartName", "").lstrip("/")
        if part_name in overrides:
            raise ValueError(f"Duplicate content type Override: {part_name}")
        overrides[part_name] = node.get("ContentType", "")
    return {name: overrides.get(name, defaults.get(posixpath.splitext(name)[1][1:].casefold())) for name in names}


def _require_content_type(
    content_types: dict[str, str],
    part: str,
    expected: str,
) -> None:
    if content_types.get(part) != expected:
        raise ValueError(f"Invalid content type for {part}: {content_types.get(part)}")


def _relationships(
    archive: zipfile.ZipFile,
    rels_part: str,
    source_part: str,
    names: set[str],
    content_types: dict[str, str],
) -> dict[str, dict[str, object]]:
    _require_content_type(content_types, rels_part, RELS_CONTENT_TYPE)
    root = _xml(archive, rels_part, names)
    records: dict[str, dict[str, object]] = {}
    for node in root.findall("pr:Relationship", NS):
        rel_id = node.get("Id")
        if not rel_id:
            continue
        if rel_id in records:
            raise ValueError(f"Duplicate relationship Id in {rels_part}: {rel_id}")
        target_mode = node.get("TargetMode", "Internal")
        if target_mode not in {"Internal", "External"}:
            raise ValueError(
                f"Invalid relationship TargetMode in {rels_part}: {target_mode}"
            )
        target = node.get("Target", "")
        internal_target = (
            None
            if target_mode == "External"
            else _part_target(source_part, target)
        )
        records[rel_id] = {
            "id": rel_id,
            "type": node.get("Type"),
            "target": internal_target if internal_target is not None else target,
            "target_mode": target_mode,
            "content_type": content_types.get(internal_target) if internal_target else None,
        }
    return records


def _part_target(source_part: str, target: str) -> str:
    if not target or "\\" in target or target.startswith("//"):
        raise ValueError(f"Unsafe relationship target from {source_part}: {target}")
    candidate = (
        target.lstrip("/")
        if target.startswith("/")
        else posixpath.join(posixpath.dirname(source_part), target)
    )
    normalized = posixpath.normpath(candidate)
    if normalized in {".", ".."} or normalized.startswith("../"):
        raise ValueError(f"Relationship target escapes package from {source_part}: {target}")
    return normalized


def _rels_name(part: str) -> str:
    directory, filename = posixpath.split(part)
    return f"{directory}/_rels/{filename}.rels"


def _notes(
    rels: dict[str, dict[str, object]],
    parts: dict[str, str],
    archive: zipfile.ZipFile,
    names: set[str],
) -> tuple[str | None, str | None]:
    for relation in rels.values():
        relation_type = relation["type"]
        is_notes = (
            relation_type in NOTES_SLIDE_RELATIONSHIP_TYPES
            or relation["content_type"] == NOTES_SLIDE_CONTENT_TYPE
        )
        if not is_notes:
            continue
        if relation_type not in NOTES_SLIDE_RELATIONSHIP_TYPES:
            raise ValueError(
                f"Invalid notes relationship type: {relation_type}"
            )
        if relation["target_mode"] != "Internal":
            raise ValueError(
                f"Invalid notes relationship TargetMode: {relation['target_mode']}"
            )
        if relation["content_type"] != NOTES_SLIDE_CONTENT_TYPE:
            raise ValueError(
                f"Invalid notes content type: {relation['content_type']}"
            )
        target = relation["target"]
        if not isinstance(target, str) or target not in parts:
            raise ValueError(f"Missing notes part: {target}")
        _xml(archive, target, names)
        return target, parts[target]
    return None, None


def _objects(
    parent: ET.Element,
    slide_part: str,
    rels: dict[str, dict[str, object]],
    timing_ids: set[str],
    archive: zipfile.ZipFile,
    names: set[str],
    parts: dict[str, str],
    choice_namespace_scopes: dict[int, dict[str, str]],
    group_path: list[str] | None = None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    group_path = [] if group_path is None else group_path
    shape_children = _shape_children(parent, choice_namespace_scopes)
    for z_order, node in enumerate(shape_children):
        item = _object(
            node,
            z_order,
            group_path,
            slide_part,
            rels,
            timing_ids,
            archive,
            names,
            parts,
        )
        result.append(item)
        if node.tag == f"{{{P}}}grpSp":
            result.extend(
                _objects(
                    node,
                    slide_part,
                    rels,
                    timing_ids,
                    archive,
                    names,
                    parts,
                    choice_namespace_scopes,
                    group_path + [item["shape_id"]],
                )
            )
    return result


def _is_shape(node: ET.Element) -> bool:
    if node.tag in SHAPE_TREE_METADATA_TAGS:
        return False
    if node.tag in SHAPE_TAGS:
        return True
    c_nv_pr = node.find(".//p:cNvPr", NS)
    return (
        c_nv_pr is not None
        and bool(c_nv_pr.get("id"))
        and c_nv_pr.get("name") is not None
    )


def _shape_children(
    parent: ET.Element,
    choice_namespace_scopes: dict[int, dict[str, str]],
) -> list[ET.Element]:
    result: list[ET.Element] = []
    for child in parent:
        if child.tag != f"{{{MC}}}AlternateContent":
            if _is_shape(child):
                result.append(child)
            continue
        branch = None
        for choice in child.findall("mc:Choice", NS):
            requires = choice.get("Requires", "").split()
            namespace_scope = choice_namespace_scopes.get(id(choice), {})
            if requires and all(
                namespace_scope.get(prefix) in SUPPORTED_MC_NAMESPACE_URIS
                for prefix in requires
            ):
                branch = choice
                break
        if branch is None:
            branch = child.find("mc:Fallback", NS)
        if branch is None:
            raise ValueError("Unsupported AlternateContent without a usable branch")
        expanded = _shape_children(branch, choice_namespace_scopes)
        if not expanded:
            raise ValueError("Unsupported AlternateContent branch without a shape")
        result.extend(expanded)
    return result


def _object(
    node: ET.Element,
    z_order: int,
    group_path: list[str],
    slide_part: str,
    rels: dict[str, dict[str, object]],
    timing_ids: set[str],
    archive: zipfile.ZipFile,
    names: set[str],
    parts: dict[str, str],
) -> dict[str, object]:
    shape_id, name = _non_visual(node)
    object_type = _type(node)
    transform = _transform(node)
    related = _referenced_relationships(node, rels)
    result: dict[str, object] = {
        "shape_id": shape_id,
        "name": name,
        "type": object_type,
        "z_order": z_order,
        "group_path": list(group_path),
        "inside_group": bool(group_path),
        "slide_part": slide_part,
        **transform,
        "crop_left": 0,
        "crop_top": 0,
        "crop_right": 0,
        "crop_bottom": 0,
        "has_extension": any(element.tag.endswith("}extLst") for element in node.iter()),
        "has_timing_reference": shape_id in timing_ids,
        "relationships": related,
        "xml_c14n_sha256": _canonical_hash(node),
    }
    if object_type == "picture":
        result.update(_picture_details(node, related, archive, names, parts))
    return result


def _non_visual(node: ET.Element) -> tuple[str, str]:
    c_nv_pr = node.find(".//p:cNvPr", NS)
    if c_nv_pr is None:
        return "", ""
    return c_nv_pr.get("id", ""), c_nv_pr.get("name", "")


def _type(node: ET.Element) -> str:
    if node.tag == f"{{{P}}}sp":
        return "text" if node.find("p:txBody", NS) is not None else "autoshape"
    mapping = {f"{{{P}}}pic": "picture", f"{{{P}}}grpSp": "group", f"{{{P}}}cxnSp": "connector", f"{{{P}}}contentPart": "media"}
    if node.tag in mapping:
        return mapping[node.tag]
    if node.tag != f"{{{P}}}graphicFrame":
        return "unknown"
    graphic_data = node.find("a:graphic/a:graphicData", NS)
    uri = graphic_data.get("uri", "") if graphic_data is not None else ""
    if uri.endswith("/table"):
        return "table"
    if uri.endswith("/chart"):
        return "chart"
    if "diagram" in uri.casefold() or "smartart" in uri.casefold():
        return "smartart"
    return "unknown"


def _transform(node: ET.Element) -> dict[str, object]:
    if node.tag == f"{{{P}}}graphicFrame":
        xfrm = node.find("p:xfrm", NS)
    elif node.tag == f"{{{P}}}grpSp":
        xfrm = node.find("p:grpSpPr/a:xfrm", NS)
    elif node.tag in {f"{{{P}}}sp", f"{{{P}}}pic", f"{{{P}}}cxnSp"}:
        xfrm = node.find("p:spPr/a:xfrm", NS)
    elif node.tag == f"{{{P}}}contentPart":
        xfrm = node.find("p:xfrm", NS)
    else:
        xfrm = None
    off = xfrm.find("a:off", NS) if xfrm is not None else None
    ext = xfrm.find("a:ext", NS) if xfrm is not None else None
    rotation = int(xfrm.get("rot", "0")) if xfrm is not None else 0
    return {
        "x": int(off.get("x", "0")) if off is not None else 0,
        "y": int(off.get("y", "0")) if off is not None else 0,
        "cx": int(ext.get("cx", "0")) if ext is not None else 0,
        "cy": int(ext.get("cy", "0")) if ext is not None else 0,
        "rotation": rotation,
        "rotation_degrees": rotation / 60000.0,
        "flip_h": _bool(xfrm.get("flipH")) if xfrm is not None else False,
        "flip_v": _bool(xfrm.get("flipV")) if xfrm is not None else False,
    }


def _bool(value: str | None) -> bool:
    return value is not None and value.casefold() in {"1", "true", "on"}


def _referenced_relationships(node: ET.Element, rels: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    result = []
    seen: set[str] = set()
    for element in node.iter():
        for attribute, rel_id in element.attrib.items():
            namespace, _, local = attribute[1:].partition("}") if attribute.startswith("{") else ("", "", attribute)
            if namespace == R and local in {"id", "embed", "link"} and rel_id not in seen:
                seen.add(rel_id)
                relation = rels.get(rel_id)
                if relation is None:
                    result.append({"id": rel_id, "type": None, "target": None, "target_mode": None, "content_type": None})
                else:
                    result.append(dict(relation))
    return result


def _picture_details(
    node: ET.Element,
    related: list[dict[str, object]],
    archive: zipfile.ZipFile,
    names: set[str],
    parts: dict[str, str],
) -> dict[str, object]:
    src_rect = node.find(".//a:srcRect", NS)
    crop = {
        "crop_left": int(src_rect.get("l", "0")) if src_rect is not None else 0,
        "crop_top": int(src_rect.get("t", "0")) if src_rect is not None else 0,
        "crop_right": int(src_rect.get("r", "0")) if src_rect is not None else 0,
        "crop_bottom": int(src_rect.get("b", "0")) if src_rect is not None else 0,
    }
    blip = node.find("p:blipFill/a:blip", NS)
    primary_id = None
    if blip is not None:
        primary_id = blip.get(f"{{{R}}}embed") or blip.get(f"{{{R}}}link")
    primary = next(
        (relation for relation in related if relation["id"] == primary_id),
        None,
    )
    media_sha256 = None
    pixel_width = pixel_height = None
    if primary and primary["target_mode"] == "Internal" and isinstance(primary["target"], str) and primary["target"] in names:
        target = primary["target"]
        media_sha256 = parts[target]
        if archive.getinfo(target).file_size <= MAX_IMAGE_READ_SIZE:
            data = archive.read(target)
            try:
                with Image.open(BytesIO(data)) as image:
                    pixel_width, pixel_height = image.size
            except Exception:
                pass
    return {**crop, "primary_relationship": primary, "media_sha256": media_sha256, "pixel_width": pixel_width, "pixel_height": pixel_height}


def _canonical_hash(node: ET.Element) -> str:
    canonical = ET.canonicalize(
        xml_data=ET.tostring(node, encoding="unicode"),
        rewrite_prefixes=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
