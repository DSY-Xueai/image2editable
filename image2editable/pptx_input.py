"""Read-only OOXML structure scanning for PPTX input files."""

from __future__ import annotations

import hashlib
import math
import os
import posixpath
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from image2editable.contracts import (
    PageStatus,
    RunStatus,
    SCHEMA_VERSION,
    validate_schema_version,
)
from image2editable.inputs import (
    new_job_id,
    sha256_file,
    validate_pptx_output_path,
)
from image2editable.store import RunStore


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
MIN_OOXML_INTEGER = -(2**63)
MAX_OOXML_INTEGER = 2**63 - 1
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
IMAGE_RELATIONSHIP_TYPES = frozenset(
    {
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
        "http://purl.oclc.org/ooxml/officeDocument/relationships/image",
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


def picture_slide_coverage(
    picture: dict[str, object],
    slide_width: int | float,
    slide_height: int | float,
) -> float:
    """Return the fraction of the slide covered by the clipped picture bounds."""
    if (
        isinstance(slide_width, bool)
        or isinstance(slide_height, bool)
        or not isinstance(slide_width, (int, float))
        or not isinstance(slide_height, (int, float))
        or not _reliable_ooxml_number(slide_width)
        or not _reliable_ooxml_number(slide_height)
        or slide_width <= 0
        or slide_height <= 0
    ):
        raise ValueError("Slide dimensions must be positive numbers")
    if picture.get(
        "transform_reliable", picture.get("_transform_reliable")
    ) is False:
        raise ValueError("Picture transform must be reliable")
    values = [picture.get(key) for key in ("x", "y", "cx", "cy")]
    if any(
        isinstance(value, bool)
        or not _reliable_ooxml_number(value)
        for value in values
    ) or values[2] <= 0 or values[3] <= 0:
        raise ValueError("Picture transform must be reliable with positive extents")
    x, y, width, height = values
    intersection_width = max(0, min(x + width, slide_width) - max(x, 0))
    intersection_height = max(0, min(y + height, slide_height) - max(y, 0))
    return intersection_width * intersection_height / (slide_width * slide_height)


def prepare_pptx_job(
    source: str | Path,
    *,
    run_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    slide_size: str = "both",
    lang: str = "ch",
) -> Path:
    source_path = Path(source).resolve()
    if not source_path.is_file() or source_path.suffix.casefold() != ".pptx":
        raise ValueError(
            f"PPTX input must be an existing .pptx file: {source_path}"
        )
    if slide_size not in {"original", "16:9", "both"}:
        raise ValueError(f"Unsupported slide_size: {slide_size}")

    job_id = new_job_id()
    root = (
        Path(run_dir).resolve()
        if run_dir is not None
        else Path.cwd() / "runs" / job_id
    )
    checked_output = (
        _path_without_symlinks(output_path)
        if output_path is not None
        else None
    )
    resolved_output = validate_pptx_output_path(
        checked_output, source_paths=[source_path], run_root=root
    )
    store = RunStore.create(root)
    try:
        copied_relative = Path("input") / "original.pptx"
        copied_path = store.root / copied_relative
        copied_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, copied_path)
        inventory = scan_pptx(copied_path)
        digest = inventory["source_sha256"]
        pages = [
            f"page_{index:03d}"
            for index in range(1, inventory["slide_count"] + 1)
        ]
        object_count = 0
        candidate_count = 0
        for page_id, slide in zip(pages, inventory["slides"]):
            objects = slide["objects"]
            candidates = [
                item for item in objects if item["action"] == "candidate"
            ]
            object_count += len(objects)
            candidate_count += len(candidates)
            common = {
                "schema_version": SCHEMA_VERSION,
                "page_id": page_id,
                "slide_index": slide["slide_index"],
                "slide_part": slide["slide_part"],
                "slide_width": inventory["slide_width"],
                "slide_height": inventory["slide_height"],
            }
            store.write_json(
                Path("pages") / page_id / "native_objects.json",
                {**common, "objects": objects},
            )
            store.write_json(
                Path("pages") / page_id / "screenshot_candidates.json",
                {**common, "candidates": candidates},
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "input": {
                "type": "pptx",
                "original_path": str(source_path),
                "source": copied_relative.as_posix(),
                "sha256": digest,
                "slide_count": inventory["slide_count"],
                "object_count": object_count,
                "candidate_count": candidate_count,
                "slide_width": inventory["slide_width"],
                "slide_height": inventory["slide_height"],
            },
            "output_format": "pptx",
            "options": {
                "lang": lang,
                "slide_size": slide_size,
                "output_path": (
                    str(resolved_output)
                    if resolved_output is not None
                    else None
                ),
            },
            "pages": pages,
        }
        store.initialize(manifest, pages)
        for page_id in pages:
            store.transition_page(page_id, PageStatus.ANALYZED)
        store.transition_run(RunStatus.PREPARED)
        return store.root
    except Exception as error:
        cleanup_error = None
        try:
            shutil.rmtree(store.root)
        except Exception as caught:
            cleanup_error = caught
        try:
            store.root.mkdir(parents=True, exist_ok=True)
        except Exception as caught:
            if cleanup_error is None:
                cleanup_error = caught
        if cleanup_error is not None:
            raise error from cleanup_error
        raise


def execute_pptx_preserve(store: RunStore) -> dict[str, object]:
    manifest = store.read_json("job_manifest.json")
    validate_schema_version(manifest)
    input_record = manifest.get("input")
    if not isinstance(input_record, dict) or input_record.get("type") != "pptx":
        raise RuntimeError("PPTX preserve requires a PPTX manifest")
    if manifest.get("output_format") != "pptx":
        raise RuntimeError("PPTX manifest output_format must be pptx")
    pages = _manifest_count(input_record, "slide_count")
    preserved_objects = _manifest_count(input_record, "object_count")
    pending_candidates = _manifest_count(input_record, "candidate_count")
    page_ids = manifest.get("pages")
    if (
        not isinstance(page_ids, list)
        or len(page_ids) != pages
        or any(
            page_id != f"page_{index:03d}"
            for index, page_id in enumerate(page_ids, start=1)
        )
    ):
        raise RuntimeError("PPTX manifest pages do not match slide_count")
    if pending_candidates > preserved_objects:
        raise RuntimeError(
            "PPTX manifest candidate_count exceeds object_count"
        )
    source_value = input_record.get("source")
    if not isinstance(source_value, str):
        raise RuntimeError("PPTX manifest source must be a relative path")
    source_path = (store.root / source_value).resolve()
    if (
        Path(source_value).is_absolute()
        or not source_path.is_relative_to(store.root)
        or not source_path.is_file()
        or source_path.suffix.casefold() != ".pptx"
    ):
        raise RuntimeError(f"Invalid PPTX manifest source: {source_value}")

    options = manifest.get("options")
    if not isinstance(options, dict):
        raise RuntimeError("PPTX manifest options must be an object")
    output_value = options.get("output_path")
    if output_value is None:
        output_path = _path_without_symlinks(
            store.root / "final" / "output.pptx"
        )
    elif isinstance(output_value, str):
        checked_output = _path_without_symlinks(output_value)
        output_path = validate_pptx_output_path(
            checked_output,
            source_paths=[source_path],
            run_root=store.root,
        )
    else:
        raise RuntimeError("PPTX manifest output_path must be a string or null")
    if output_path is None:
        raise RuntimeError("PPTX output path is missing")
    if _same_file(source_path, output_path):
        raise ValueError(f"PPTX output must not overwrite source: {output_path}")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"PPTX output already exists: {output_path}")

    input_sha256 = sha256_file(source_path)
    expected_sha256 = input_record.get("sha256")
    if expected_sha256 != input_sha256:
        raise RuntimeError("PPTX input hash does not match manifest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _path_without_symlinks(output_path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
        shutil.copyfile(source_path, temporary)
        output_sha256 = sha256_file(temporary)
        if output_sha256 != input_sha256:
            raise RuntimeError("PPTX preserve copy hash mismatch")
        _publish_pptx_no_clobber(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    warnings = (
        ["P1 preserved screenshot candidates without replacement"]
        if pending_candidates
        else []
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": RunStatus.COMPLETED.value,
        "pages": pages,
        "preserved_objects": preserved_objects,
        "pending_candidates": pending_candidates,
        "warnings": warnings,
        "outputs": {"pptx": str(output_path)},
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
    }


def _manifest_count(record: dict[str, object], name: str) -> int:
    value = record.get(name)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"Invalid PPTX manifest {name}: {value}")
    return value


def _reliable_ooxml_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return MIN_OOXML_INTEGER <= value <= MAX_OOXML_INTEGER


def _path_without_symlinks(value: str | Path) -> Path:
    lexical = Path(os.path.abspath(value))
    try:
        resolved = lexical.resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"PPTX output path contains a symlink: {lexical}") from error
    if resolved != lexical:
        raise ValueError(f"PPTX output path contains a symlink: {lexical}")
    return lexical


def _publish_pptx_no_clobber(temporary: Path, output: Path) -> None:
    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise FileExistsError(f"PPTX output already exists: {output}") from error
    temporary.unlink()


def _same_file(first: Path, second: Path) -> bool:
    return first == second or (
        first.exists() and second.exists() and os.path.samefile(first, second)
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
            try:
                slide_width = int(size.get("cx"))
                slide_height = int(size.get("cy"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Slide dimensions must be positive numbers"
                ) from error
            if slide_width <= 0 or slide_height <= 0:
                raise ValueError("Slide dimensions must be positive numbers")
            if (
                slide_width > MAX_OOXML_INTEGER
                or slide_height > MAX_OOXML_INTEGER
            ):
                raise ValueError("Slide dimensions must be reliable integers")
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
                    slide_width,
                    slide_height,
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
                "slide_width": slide_width,
                "slide_height": slide_height,
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
    slide_width: int,
    slide_height: int,
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
            slide_width,
            slide_height,
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
                    slide_width,
                    slide_height,
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
    slide_width: int,
    slide_height: int,
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
        _mark_picture_safety(result, slide_width, slide_height, names, parts)
    else:
        result.pop("_transform_reliable")
        result.update({"action": "preserve", "safety_reasons": []})
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
    x, x_reliable = _integer_attribute(off, "x")
    y, y_reliable = _integer_attribute(off, "y")
    cx, cx_reliable = _integer_attribute(ext, "cx")
    cy, cy_reliable = _integer_attribute(ext, "cy")
    rotation, rotation_reliable = _integer_attribute(
        xfrm, "rot", default=0
    )
    flip_h, flip_h_reliable = _boolean_attribute(xfrm, "flipH")
    flip_v, flip_v_reliable = _boolean_attribute(xfrm, "flipV")
    return {
        "x": x,
        "y": y,
        "cx": cx,
        "cy": cy,
        "rotation": rotation,
        "rotation_degrees": (
            rotation / 60000.0 if rotation_reliable else 0.0
        ),
        "flip_h": flip_h,
        "flip_v": flip_v,
        "_transform_reliable": all(
            (
                x_reliable,
                y_reliable,
                cx_reliable,
                cy_reliable,
                rotation_reliable,
                flip_h_reliable,
                flip_v_reliable,
            )
        ),
    }


def _integer_attribute(
    element: ET.Element | None,
    name: str,
    *,
    default: int | None = None,
) -> tuple[int, bool]:
    if element is None:
        return (0 if default is None else default), False
    value = element.get(name)
    if value is None:
        return (0 if default is None else default), default is not None
    try:
        parsed = int(value)
    except ValueError:
        return (0 if default is None else default), False
    return parsed, MIN_OOXML_INTEGER <= parsed <= MAX_OOXML_INTEGER


def _boolean_attribute(
    element: ET.Element | None, name: str
) -> tuple[bool, bool]:
    if element is None:
        return False, False
    value = element.get(name)
    if value is None:
        return False, True
    normalized = value.casefold()
    if normalized in {"1", "true", "on"}:
        return True, True
    if normalized in {"0", "false", "off"}:
        return False, True
    return False, False


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
        "crop_left": _crop_value(src_rect, "l"),
        "crop_top": _crop_value(src_rect, "t"),
        "crop_right": _crop_value(src_rect, "r"),
        "crop_bottom": _crop_value(src_rect, "b"),
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


def _crop_value(src_rect: ET.Element | None, name: str) -> int:
    if src_rect is None:
        return 0
    try:
        return int(src_rect.get(name, "0"))
    except ValueError:
        return 1


def _mark_picture_safety(
    picture: dict[str, object],
    slide_width: int,
    slide_height: int,
    names: set[str],
    parts: dict[str, str],
) -> None:
    reasons: list[str] = []
    reliable = bool(picture.pop("_transform_reliable"))
    coverage = None
    if not reliable or picture["cx"] <= 0 or picture["cy"] <= 0:
        reasons.append("unreliable_transform")
    elif picture["inside_group"]:
        reasons.append("inside_group")
    else:
        coverage = picture_slide_coverage(picture, slide_width, slide_height)
        if coverage < 0.80:
            reasons.append("coverage_below_threshold")
    picture["slide_coverage"] = coverage

    if picture["rotation"] != 0 or picture["flip_h"] or picture["flip_v"]:
        reasons.append("rotation_or_flip")
    if any(
        picture[key] != 0
        for key in (
            "crop_left",
            "crop_top",
            "crop_right",
            "crop_bottom",
        )
    ):
        reasons.append("nonzero_crop")

    relation = picture.get("primary_relationship")
    if isinstance(relation, dict) and relation.get("target_mode") == "External":
        reasons.append("external_relationship")
    elif not _is_valid_picture_media(picture, relation, names, parts):
        reasons.append("missing_media")
    if picture["has_extension"] or picture["has_timing_reference"]:
        reasons.append("unsupported_extension")

    picture["action"] = "candidate" if not reasons else "preserve"
    picture["safety_reasons"] = reasons


def _is_valid_picture_media(
    picture: dict[str, object],
    relation: object,
    names: set[str],
    parts: dict[str, str],
) -> bool:
    if not isinstance(relation, dict):
        return False
    target = relation.get("target")
    content_type = relation.get("content_type")
    return (
        relation.get("target_mode") == "Internal"
        and relation.get("type") in IMAGE_RELATIONSHIP_TYPES
        and isinstance(target, str)
        and target.startswith("ppt/media/")
        and target in names
        and target in parts
        and isinstance(content_type, str)
        and content_type.casefold().startswith("image/")
        and picture.get("media_sha256") == parts[target]
    )


def _canonical_hash(node: ET.Element) -> str:
    canonical = ET.canonicalize(
        xml_data=ET.tostring(node, encoding="unicode"),
        rewrite_prefixes=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
