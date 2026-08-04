from __future__ import annotations

import importlib
import io
import json
import os
import copy
import ctypes
import errno
import hashlib
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image

SAM21_LARGE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_large.pt"
)
SAM21_LARGE_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class VisualSegmentationError(RuntimeError):
    pass


def _complete_opaque_mask_regions(
    mask: np.ndarray, image: np.ndarray | None = None
) -> np.ndarray:
    """Fill small topology gaps only inside already-solid visual regions."""
    source = np.asarray(mask, dtype=bool)
    completed = source.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        source.astype(np.uint8), 8
    )
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        box_area = width * height
        if min(width, height) < 8 or area < 64 or area / max(1, box_area) < 0.78:
            continue
        component = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
        closed = cv2.morphologyEx(
            component, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
        )
        if np.count_nonzero(closed) <= area * 1.25:
            completed[y:y + height, x:x + width] |= closed.astype(bool)
    if image is None or not np.any(completed):
        return completed
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.shape[:2] != completed.shape or pixels.ndim != 3:
        raise ValueError("mask completion image dimensions differ")
    ys, xs = np.nonzero(completed)
    pad = max(4, min(12, round(min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * 0.05)))
    y0, y1 = max(0, int(ys.min()) - pad), min(completed.shape[0], int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(completed.shape[1], int(xs.max()) + pad + 1)
    local = completed[y0:y1, x0:x1]
    dilated = cv2.dilate(local.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    near = cv2.dilate(local.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ring = dilated & ~near
    if np.count_nonzero(ring) < 32:
        return completed
    crop = pixels[y0:y1, x0:x1]
    background = np.median(crop[ring], axis=0)
    distance = np.linalg.norm(crop.astype(np.float32) - background, axis=2)
    quiet = distance[ring] <= 14.0
    if np.count_nonzero(quiet) < np.count_nonzero(ring) * 0.6:
        return completed
    foreground = distance > 18.0
    foreground &= ~local
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), 8
    )
    touching = cv2.dilate(local.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    recovered = local.copy()
    for label in range(1, count):
        candidate = labels == label
        contact = candidate & touching
        contact_count = int(np.count_nonzero(contact))
        if stats[label, cv2.CC_STAT_AREA] < 9 or contact_count < 3:
            continue
        compatible = 0
        for contact_y, contact_x in zip(*np.nonzero(contact)):
            neighbor_y0 = max(0, int(contact_y) - 2)
            neighbor_y1 = min(local.shape[0], int(contact_y) + 3)
            neighbor_x0 = max(0, int(contact_x) - 2)
            neighbor_x1 = min(local.shape[1], int(contact_x) + 3)
            neighbor_mask = local[
                neighbor_y0:neighbor_y1,
                neighbor_x0:neighbor_x1,
            ]
            if not np.any(neighbor_mask):
                continue
            neighbor_colors = crop[
                neighbor_y0:neighbor_y1,
                neighbor_x0:neighbor_x1,
            ][neighbor_mask].astype(np.float32)
            contact_color = crop[contact_y, contact_x].astype(np.float32)
            color_distance = np.linalg.norm(neighbor_colors - contact_color, axis=1)
            if np.count_nonzero(color_distance <= 30.0) >= 3:
                compatible += 1
        if compatible >= max(3, round(contact_count * 0.6)):
            recovered |= candidate
    if np.count_nonzero(recovered) <= np.count_nonzero(local) * 1.5:
        completed[y0:y1, x0:x1] = recovered
    return completed


def execute_component_actions(
    image: np.ndarray,
    graph: dict,
    actions: list[dict],
    *,
    sam_runner,
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Execute requested mask edits; never decide quality-gate outcomes."""

    try:
        from image2editable.component_contracts import (
            validate_component_action,
            validate_component_graph,
            validate_graph_transition,
        )
    except ModuleNotFoundError:
        from component_contracts import (  # type: ignore[no-redef]
            validate_component_action,
            validate_component_graph,
            validate_graph_transition,
        )

    source = Path(input_dir)
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Component action output already exists: {target}")
    validated = validate_component_graph(graph)
    result = copy.deepcopy(validated)
    nodes = {node["id"]: node for node in result["nodes"]}
    loaded_masks = {
        component_id: _read_action_mask(source / node["mask"], image.shape[:2], node["mask_sha256"])
        for component_id, node in nodes.items()
    }
    masks = {component_id: loaded[0] for component_id, loaded in loaded_masks.items()}
    mask_payloads = {component_id: loaded[1] for component_id, loaded in loaded_masks.items()}
    touched = set()
    suppressed_text_ids = set()
    for action in actions:
        validate_component_action(action, graph=validated)
        object_ids = action["object_ids"]
        name = action["action"]
        if name == "attach_text":
            visual, text = object_ids
            valid_states = (
                nodes[visual]["state"] == "pending"
                and nodes[text]["state"] == "frozen"
            )
            if not valid_states:
                raise ValueError("attach_text requires pending visual and frozen text")
        elif name == "suppress_text":
            valid_states = (
                nodes[object_ids[0]]["kind"] == "text"
                and nodes[object_ids[0]]["state"] == "frozen"
            )
            if not valid_states:
                raise ValueError("suppress_text requires a frozen text object")
            suppressed_text_ids.add(object_ids[0])
        elif name == "collapse_to_parent":
            allowed_states = {"inactive", "pending"}
            valid_states = all(
                nodes[value]["state"] in allowed_states for value in object_ids
            )
        elif name == "absorb_into_parent":
            parent, *absorbed = object_ids
            valid_states = (
                nodes[parent]["state"] in {"inactive", "pending"}
                and all(nodes[value]["state"] == "pending" for value in absorbed)
            )
        elif name == "rebuild_background":
            valid_states = all(
                nodes[value]["state"] in {"pending", "frozen"}
                for value in object_ids
            )
        else:
            allowed_states = {"pending"}
            valid_states = all(
                nodes[value]["state"] in allowed_states for value in object_ids
            )
        if not valid_states:
            raise ValueError(f"{name} requires a pending component")
        if name != "rebuild_background":
            if touched & set(object_ids):
                raise ValueError("component plan has conflicting object actions")
            touched.update(object_ids)
    for action in actions:
        object_ids = action["object_ids"]
        name = action["action"]
        if name == "accept":
            if nodes[object_ids[0]]["state"] != "pending":
                raise ValueError("accept requires a pending component")
            accepted = nodes[object_ids[0]]
            if action["parameters"].get("independent") is True:
                accepted["kind"] = "parent"
                accepted["parent_id"] = None
            masks[object_ids[0]] = _complete_opaque_mask_regions(
                masks[object_ids[0]], image
            )
            accepted["state"] = "pending_gate"
        elif name == "discard":
            nodes[object_ids[0]]["state"] = "inactive"
        elif name == "rebuild_background":
            pass
        elif name == "attach_text":
            visual, text = object_ids
            nodes[visual]["text_ids"] = sorted(set(nodes[visual]["text_ids"] + [text]))
        elif name == "suppress_text":
            text_id = object_ids[0]
            nodes[text_id]["state"] = "inactive"
            for node in nodes.values():
                node["text_ids"] = [
                    value for value in node["text_ids"] if value != text_id
                ]
        elif name == "collapse_to_parent":
            parent = object_ids[0]
            nodes[parent]["state"] = "pending"
            _deactivate_descendants(nodes, parent)
        elif name == "absorb_into_parent":
            parent, *absorbed = object_ids
            masks[parent] = np.logical_or.reduce(
                [masks[value] for value in object_ids]
            )
            nodes[parent]["state"] = "pending"
            for component_id in absorbed:
                nodes[component_id]["state"] = "inactive"
            _deactivate_descendants(nodes, parent)
        elif name == "merge":
            selected = [nodes[value] for value in object_ids]
            merged = np.logical_or.reduce([masks[value] for value in object_ids])
            for node in selected:
                node["state"] = "inactive"
            new_id = _new_action_id(nodes, "merge")
            merged_kind = selected[0]["kind"]
            merged_parent = selected[0]["parent_id"]
            nodes[new_id] = {
                "id": new_id, "kind": merged_kind, "parent_id": merged_parent,
                "state": "pending", "mask": f"masks/{new_id}.png",
                "mask_sha256": "", "bbox": [0, 0, 1, 1],
                "z_index": min(node["z_index"] for node in selected),
                "text_ids": sorted({value for node in selected for value in node["text_ids"]}),
            }
            masks[new_id] = merged
        elif name == "split":
            component_id = object_ids[0]
            parts = _connected_action_parts(masks[component_id], action["parameters"]["parts"])
            nodes[component_id]["state"] = "inactive"
            next_z = max(node["z_index"] for node in nodes.values()) + 1
            for index, part in enumerate(parts, start=1):
                new_id = _new_action_id(nodes, "split")
                original = nodes[component_id]
                kind = "child" if original["kind"] == "parent" else original["kind"]
                parent_id = component_id if original["kind"] == "parent" else original["parent_id"]
                nodes[new_id] = {
                    "id": new_id, "kind": kind, "parent_id": parent_id,
                    "state": "pending", "mask": f"masks/{new_id}.png",
                    "mask_sha256": "", "bbox": [0, 0, 1, 1],
                    "z_index": next_z + index - 1, "text_ids": [],
                }
                masks[new_id] = part
        elif name in {"expand", "shrink"}:
            component_id = object_ids[0]
            radius = max(1, round(min(image.shape[:2]) * action["parameters"]["margin_ratio"]))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            current = masks[component_id].astype(np.uint8)
            changed = cv2.dilate(current, kernel) if name == "expand" else cv2.erode(current, kernel)
            parent_id = nodes[component_id]["parent_id"]
            if name == "expand":
                support = masks[parent_id] if parent_id is not None else cv2.dilate(current, kernel)
                changed = np.asarray(changed, dtype=bool) & np.asarray(support, dtype=bool)
            masks[component_id] = np.asarray(changed, dtype=bool)
        elif name in {"retry_with_box", "retry_with_points"}:
            component_id = object_ids[0]
            parameters = action["parameters"]
            height, width = image.shape[:2]
            box = parameters.get("box")
            mapped_box = None if box is None else [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
            map_points = lambda values: [[point[0] * (width - 1), point[1] * (height - 1)] for point in values]
            runner = sam_runner
            if runner is None:
                from scripts.sam_worker import run_component_prompt_worker

                runner = lambda **values: run_component_prompt_worker(
                    values["image"],
                    box=values["box"],
                    positive=values["positive"],
                    negative=values["negative"],
                    work_dir=target.parent,
                )
            proposed = runner(
                image=image,
                box=mapped_box,
                positive=map_points(parameters.get("positive", [])),
                negative=map_points(parameters.get("negative", [])),
            )
            proposed = np.asarray(proposed, dtype=bool)
            if proposed.shape != image.shape[:2] or not proposed.any():
                raise VisualSegmentationError("SAM component retry returned an invalid mask")
            masks[component_id] = proposed
            parent_id = nodes[component_id]["parent_id"]
            if parent_id is not None and (
                parameters.get("independent") is True
                or np.any(proposed & ~masks[parent_id])
            ):
                nodes[component_id]["kind"] = "parent"
                nodes[component_id]["parent_id"] = None
        else:
            raise AssertionError(f"Unsupported component action: {name}")
    result["nodes"] = list(nodes.values())
    staging = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        staging.mkdir(parents=False)
        mask_dir = staging / "masks"
        mask_dir.mkdir()
        for node in result["nodes"]:
            mask = masks[node["id"]]
            if not mask.any():
                raise VisualSegmentationError(f"Component action produced an empty mask: {node['id']}")
            if node["state"] == "frozen":
                path = staging / node["mask"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(mask_payloads[node["id"]])
                continue
            path = mask_dir / f"{node['id']}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            node["mask"] = f"masks/{node['id']}.png"
            node["mask_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            ys, xs = np.where(mask)
            node["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        validate_graph_transition(
            before=validated,
            after=result,
            allowed_suppressed_text_ids=suppressed_text_ids,
        )
        (staging / "component-graph.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_action_directory(staging, target)
    except BaseException:
        raise
    return result


def _publish_action_directory(staging: Path, target: Path) -> None:
    """Atomically publish a directory without replacing an existing target."""

    if os.name == "nt":
        try:
            staging.rename(target)
        except FileExistsError:
            raise
        except OSError as error:
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Component action output already exists: {target}") from error
            raise
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic no-replace directory publication is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(staging), -100, os.fsencode(target), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"Component action output already exists: {target}")
    raise OSError(error_number, os.strerror(error_number), str(target))


def _read_action_mask(path: Path, shape: tuple[int, int], digest: str) -> tuple[np.ndarray, bytes]:
    status = path.lstat()
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(status.st_mode)
        or bool(getattr(status, "st_file_attributes", 0) & reparse)
        or not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
    ):
        raise VisualSegmentationError(f"Component action mask path is unsafe: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (status.st_dev, status.st_ino)
        ):
            raise VisualSegmentationError(f"Component action mask identity changed: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != (
            opened.st_dev, opened.st_ino, opened.st_size
        ):
            raise VisualSegmentationError(f"Component action mask changed while reading: {path}")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise VisualSegmentationError(f"Component action mask hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as stored:
        mask = np.asarray(stored.convert("L")) > 0
    if mask.shape != shape:
        raise VisualSegmentationError(f"Component action mask shape mismatch: {path}")
    return mask, payload


def _new_action_id(nodes: dict[str, dict], prefix: str) -> str:
    index = 1
    while f"{prefix}_{index:04d}" in nodes:
        index += 1
    return f"{prefix}_{index:04d}"


def _connected_action_parts(mask: np.ndarray, expected: int) -> list[np.ndarray]:
    from scripts.fg_extract import connected_mask_proposals

    parts = connected_mask_proposals(mask, expected)
    if len(parts) != expected:
        raise VisualSegmentationError("split did not find exact connected proposals")
    return parts


def _deactivate_descendants(nodes: dict[str, dict], parent_id: str) -> None:
    children = [node for node in nodes.values() if node["parent_id"] == parent_id]
    for child in children:
        child["state"] = "inactive"
        _deactivate_descendants(nodes, child["id"])


@contextmanager
def _sam_inference_context(generator):
    if not str(
        getattr(generator, "_image2editable_device", "")
    ).startswith("cuda"):
        yield
        return
    torch = importlib.import_module("torch")
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            yield


def _binary_visual_mask(mask: object) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype.kind not in "biuf":
        raise ValueError("visual mask must be two-dimensional and numeric")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError("visual mask contains non-finite values")
    if array.dtype.kind in "if" and np.any(array < 0):
        raise ValueError("visual mask contains negative values")
    return array if array.dtype == np.bool_ else array > 0


def validate_visual_masks(element_masks: list[np.ndarray]) -> None:
    if not element_masks:
        return
    first = _binary_visual_mask(element_masks[0])
    claimed = np.zeros(first.shape, dtype=bool)
    duplicate = np.zeros(first.shape, dtype=bool)
    for mask in element_masks:
        active = _binary_visual_mask(mask)
        if active.shape != claimed.shape:
            raise ValueError("visual mask shapes must match")
        np.logical_and(claimed, active, out=duplicate)
        if np.any(duplicate):
            raise VisualSegmentationError(
                "overlapping visual ownership detected"
            )
        np.logical_or(claimed, active, out=claimed)


def visual_difference(
    source: np.ndarray,
    reconstructed: np.ndarray,
    text_mask: np.ndarray,
) -> dict:
    valid = text_mask == 0
    if not np.any(valid):
        return {
            "mae": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "changed_ratio": 0.0,
            "largest_artifact_ratio": 0.0,
        }
    difference = np.mean(
        np.abs(source.astype(np.float32) - reconstructed.astype(np.float32)),
        axis=2,
    )
    pixel_difference = difference[valid]
    artifact_mask = ((difference > 8.0) & valid).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        artifact_mask,
        connectivity=8,
    )
    largest_artifact = (
        int(np.max(stats[1:, cv2.CC_STAT_AREA]))
        if count > 1
        else 0
    )
    return {
        "mae": float(np.mean(pixel_difference)),
        "p95": float(np.percentile(pixel_difference, 95)),
        "p99": float(np.percentile(pixel_difference, 99)),
        "changed_ratio": float(np.mean(pixel_difference > 3.0)),
        "largest_artifact_ratio": (
            largest_artifact / int(np.count_nonzero(valid))
        ),
    }


def background_residual_metrics(
    source: np.ndarray,
    background: np.ndarray,
    removal_mask: np.ndarray,
) -> dict:
    """Measure source edges that remain visible in the repaired background."""
    source = np.asarray(source, dtype=np.uint8)
    background = np.asarray(background, dtype=np.uint8)
    removal = np.asarray(removal_mask) > 0
    if source.shape != background.shape or removal.shape != source.shape[:2]:
        raise ValueError("background residual inputs must have matching shapes")
    if not np.any(removal):
        return {
            "source_edge_pixels": 0,
            "retained_edge_pixels": 0,
            "retained_edge_ratio": 0.0,
        }

    support = cv2.dilate(
        removal.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ) > 0
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    source_edges = cv2.Canny(source_gray, 8, 24) > 0
    background_edges = cv2.Canny(background_gray, 8, 24) > 0
    background_edges = cv2.dilate(
        background_edges.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ) > 0
    relevant = source_edges & support
    source_edge_pixels = int(np.count_nonzero(relevant))
    retained_edge_pixels = int(
        np.count_nonzero(relevant & background_edges)
    )
    return {
        "source_edge_pixels": source_edge_pixels,
        "retained_edge_pixels": retained_edge_pixels,
        "retained_edge_ratio": (
            retained_edge_pixels / source_edge_pixels
            if source_edge_pixels
            else 0.0
        ),
    }


def has_background_residual(metrics: dict) -> bool:
    """Reject a background that retains a material source-object outline."""
    return (
        metrics.get("source_edge_pixels", 0) >= 16
        and metrics.get("retained_edge_ratio", 0.0) >= 0.45
    )


def needs_text_only_fallback(metrics: dict) -> bool:
    """Prefer the text-clean background when sparse artifacts stay visible."""
    return (
        (
            metrics.get("p99", 0.0) > 5.0
            and metrics.get("changed_ratio", 0.0) > 0.01
        )
        or metrics.get("largest_artifact_ratio", 0.0) > 0.001
    )


def require_visual_quality(metrics: dict) -> None:
    if metrics["mae"] > 12.0 or metrics["p95"] > 48.0:
        raise VisualSegmentationError(
            "visual reconstruction did not meet the quality threshold"
        )


def write_segmentation_diagnostics(
    output_dir: Path,
    source: np.ndarray,
    masks: list[np.ndarray],
    reconstructed: np.ndarray,
    metrics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(source).save(output_dir / "source.png")
    Image.fromarray(reconstructed).save(output_dir / "reconstructed.png")

    ownership = np.zeros(source.shape[:2], dtype=np.uint16)
    for index, mask in enumerate(masks, start=1):
        ownership[np.asarray(mask, dtype=bool)] = index
    normalized = ((ownership * 37) % 255).astype(np.uint8)
    Image.fromarray(normalized).save(output_dir / "ownership.png")

    report = dict(metrics)
    report["component_count"] = len(masks)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@dataclass
class MaskCandidate:
    mask: np.ndarray
    score: float
    source: str
    crop_box: tuple[int, int, int, int] | None = None
    touches_crop_edge: bool = False
    label: str = ""
    role: str = ""
    object_box: tuple[float, float, float, float] | None = None


@dataclass
class VisualElement:
    mask: np.ndarray
    z_index: int
    score: float
    source: str
    semantic_mask: np.ndarray | None = None
    object_box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.semantic_mask is None:
            self.semantic_mask = np.asarray(self.mask, dtype=bool).copy()


def build_component_mask_layers(elements: list[VisualElement]) -> list[dict]:
    """Keep an intact semantic parent beside each detachable visible mask."""

    child_masks = [_binary_visual_mask(element.mask) for element in elements]
    validate_visual_masks(child_masks)
    layers = []
    for element, child_mask in zip(elements, child_masks):
        parent_mask = _binary_visual_mask(element.semantic_mask)
        if parent_mask.shape != child_mask.shape:
            raise VisualSegmentationError(
                "parent and child component masks must have the same shape"
            )
        if np.any(child_mask & ~parent_mask):
            raise VisualSegmentationError(
                "child component mask must stay inside its parent"
            )
        if not np.any(parent_mask) or not np.any(child_mask):
            raise VisualSegmentationError("component masks cannot be empty")
        layers.append(
            {
                "parent_mask": parent_mask,
                "child_mask": child_mask,
                "z_index": element.z_index,
            }
        )
    return layers


def resolve_visual_elements(
    candidates: list[MaskCandidate],
    min_area: int = 20,
    duplicate_iou: float = 0.92,
) -> list[VisualElement]:
    candidates = _merge_semantic_candidates(candidates)
    valid = []
    for candidate in candidates:
        if candidate.mask.dtype != bool:
            continue
        area = int(np.count_nonzero(candidate.mask))
        if area < min_area or area / candidate.mask.size >= 0.95:
            continue
        ys, xs = np.nonzero(candidate.mask)
        bbox = (
            int(ys.min()),
            int(ys.max()) + 1,
            int(xs.min()),
            int(xs.max()) + 1,
        )
        valid.append((candidate, area, bbox, candidate.mask))

    unique = []
    for candidate_stats in sorted(
        valid,
        key=lambda item: (
            item[0].touches_crop_edge,
            -item[1] if item[0].touches_crop_edge else 0,
            -item[0].score,
        ),
    ):
        candidate, candidate_area, candidate_bbox, candidate_support = (
            candidate_stats
        )
        duplicate = False
        for index, (
            retained,
            retained_area,
            retained_bbox,
            retained_support,
        ) in enumerate(unique):
            smaller_area = min(candidate_area, retained_area)
            larger_area = max(candidate_area, retained_area)
            smaller = candidate if candidate_area <= retained_area else retained

            y1 = max(candidate_bbox[0], retained_bbox[0])
            y2 = min(candidate_bbox[1], retained_bbox[1])
            x1 = max(candidate_bbox[2], retained_bbox[2])
            x2 = min(candidate_bbox[3], retained_bbox[3])
            if y1 >= y2 or x1 >= x2:
                continue

            area_ratio = smaller_area / larger_area
            if area_ratio < duplicate_iou and not smaller.touches_crop_edge:
                continue

            intersection = int(
                np.count_nonzero(
                    candidate.mask[y1:y2, x1:x2]
                    & retained.mask[y1:y2, x1:x2]
                )
            )
            if (
                smaller.touches_crop_edge
                and smaller_area - intersection < min_area
            ):
                duplicate = True
                unique[index] = (
                    retained,
                    retained_area,
                    retained_bbox,
                    retained_support | candidate_support,
                )
                break
            if area_ratio < duplicate_iou:
                continue

            union = candidate_area + retained_area - intersection
            if intersection / max(union, 1) < duplicate_iou:
                continue

            parent_child = (
                smaller_area - intersection < min_area
                and larger_area - intersection >= min_area
            )
            if not parent_child or smaller.touches_crop_edge:
                duplicate = True
                unique[index] = (
                    retained,
                    retained_area,
                    retained_bbox,
                    retained_support | candidate_support,
                )
                break

        if duplicate:
            continue
        unique.append(candidate_stats)

    front_to_back = sorted(
        unique,
        key=lambda item: (item[1], -item[0].score),
    )
    if not front_to_back:
        return []

    claimed = np.zeros(front_to_back[0][0].mask.shape, dtype=bool)
    elements = []
    for candidate, _, _, semantic_support in front_to_back:
        visible = candidate.mask & ~claimed
        if np.count_nonzero(visible) < min_area:
            continue
        elements.append(
            VisualElement(
                mask=visible,
                z_index=0,
                score=candidate.score,
                source=candidate.source,
                semantic_mask=semantic_support,
                object_box=candidate.object_box,
            )
        )
        claimed |= visible

    elements.reverse()
    for z_index, element in enumerate(elements):
        element.z_index = z_index
    return elements


def _enclosed_holes(mask: np.ndarray) -> np.ndarray:
    background = ~np.asarray(mask, dtype=bool)
    if not np.any(background):
        return np.zeros(background.shape, dtype=bool)
    count, labels = cv2.connectedComponents(background.astype(np.uint8), connectivity=8)
    border_labels = set(labels[0, :])
    border_labels.update(labels[-1, :])
    border_labels.update(labels[:, 0])
    border_labels.update(labels[:, -1])
    keep = np.ones(count, dtype=bool)
    keep[list(border_labels)] = False
    keep[0] = False
    return keep[labels]


def recheck_visual_element_holes(
    image: np.ndarray,
    elements: list[VisualElement],
    generator,
    min_hole_area: int = 20,
) -> None:
    if not elements or not any(
        element.object_box is not None for element in elements
    ):
        return

    predictor = generator.predictor
    with _sam_inference_context(generator):
        predictor.set_image(image)
    owned = np.logical_or.reduce([element.mask for element in elements])
    height, width = image.shape[:2]

    for element in reversed(elements):
        if element.object_box is None:
            continue
        holes = _enclosed_holes(element.mask)
        other_owned = owned & ~element.mask
        holes &= ~other_owned
        count, labels = cv2.connectedComponents(
            holes.astype(np.uint8),
            connectivity=8,
        )
        for label in range(1, count):
            hole = labels == label
            hole_area = int(np.count_nonzero(hole))

            semantic_coverage = float(np.count_nonzero(hole & element.semantic_mask))
            if semantic_coverage / max(hole_area, 1) >= 0.90:
                recovered = hole & element.semantic_mask & ~owned
                element.mask |= recovered
                owned |= recovered
                continue
            if hole_area < min_hole_area:
                continue

            distance = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
            point_y, point_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            x1, y1, x2, y2 = element.object_box
            point_coords = np.asarray(
                [
                    [point_x, point_y],
                    [max(0.0, x1 - 2.0), max(0.0, y1 - 2.0)],
                    [min(width - 1.0, x2 + 2.0), max(0.0, y1 - 2.0)],
                    [max(0.0, x1 - 2.0), min(height - 1.0, y2 + 2.0)],
                    [min(width - 1.0, x2 + 2.0), min(height - 1.0, y2 + 2.0)],
                ],
                dtype=np.float32,
            )
            with _sam_inference_context(generator):
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=np.asarray([1, 0, 0, 0, 0], dtype=np.int32),
                    box=np.asarray(element.object_box, dtype=np.float32),
                    multimask_output=True,
                )
            candidate = np.asarray(masks[int(np.argmax(scores))], dtype=bool)
            element_area = int(np.count_nonzero(element.mask))
            if (
                np.count_nonzero(candidate & hole) / max(hole_area, 1) < 0.90
                or np.count_nonzero(candidate & element.mask) / max(element_area, 1)
                < 0.85
            ):
                continue

            box_mask = np.zeros(candidate.shape, dtype=bool)
            box_x1 = max(0, int(np.floor(x1)))
            box_y1 = max(0, int(np.floor(y1)))
            box_x2 = min(width, int(np.ceil(x2)))
            box_y2 = min(height, int(np.ceil(y2)))
            box_mask[box_y1:box_y2, box_x1:box_x2] = True
            if np.count_nonzero(candidate & ~box_mask) > element_area * 0.01:
                continue

            element.semantic_mask |= candidate
            recovered = hole & candidate & ~owned
            element.mask |= recovered
            owned |= recovered


def _merge_semantic_candidates(
    candidates: list[MaskCandidate],
) -> list[MaskCandidate]:
    """Merge partial duplicate detections without joining different object roles."""
    rules = {
        "container": (0.95, 0.50),
        "person": (0.80, 0.0),
    }
    passthrough = [candidate for candidate in candidates if candidate.role not in rules]
    for role, (min_containment, min_iou) in rules.items():
        passthrough.extend(
            _merge_role_candidates(
                candidates,
                role=role,
                min_containment=min_containment,
                min_iou=min_iou,
            )
        )
    return passthrough


def _merge_role_candidates(
    candidates: list[MaskCandidate],
    *,
    role: str,
    min_containment: float,
    min_iou: float,
) -> list[MaskCandidate]:
    same_role = sorted(
        (candidate for candidate in candidates if candidate.role == role),
        key=lambda candidate: int(np.count_nonzero(candidate.mask)),
        reverse=True,
    )
    merged: list[MaskCandidate] = []
    for candidate in same_role:
        candidate_area = int(np.count_nonzero(candidate.mask))
        for index, retained in enumerate(merged):
            if not _same_semantic_instance(candidate, retained, role):
                continue
            retained_area = int(np.count_nonzero(retained.mask))
            intersection = int(np.count_nonzero(candidate.mask & retained.mask))
            union = candidate_area + retained_area - intersection
            if (
                intersection / max(min(candidate_area, retained_area), 1)
                < min_containment
                or intersection / max(union, 1) < min_iou
            ):
                continue
            base = retained if retained_area >= candidate_area else candidate
            merged[index] = MaskCandidate(
                mask=retained.mask | candidate.mask,
                score=max(retained.score, candidate.score),
                source=base.source,
                crop_box=base.crop_box,
                touches_crop_edge=(
                    retained.touches_crop_edge and candidate.touches_crop_edge
                ),
                label=base.label,
                role=role,
                object_box=base.object_box,
            )
            break
        else:
            merged.append(candidate)
    return merged


def _same_semantic_instance(
    first: MaskCandidate,
    second: MaskCandidate,
    role: str,
) -> bool:
    if first.object_box is None or second.object_box is None:
        return False
    first_box = first.object_box
    second_box = second.object_box
    x1 = max(first_box[0], second_box[0])
    y1 = max(first_box[1], second_box[1])
    x2 = min(first_box[2], second_box[2])
    y2 = min(first_box[3], second_box[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first_box[2] - first_box[0]) * max(
        0.0, first_box[3] - first_box[1]
    )
    second_area = max(0.0, second_box[2] - second_box[0]) * max(
        0.0, second_box[3] - second_box[1]
    )
    box_iou = intersection / max(first_area + second_area - intersection, 1.0)
    if box_iou >= (0.55 if role == "person" else 0.70):
        return True
    first_tokens = {token.strip(".,").lower() for token in first.label.split()}
    second_tokens = {token.strip(".,").lower() for token in second.label.split()}
    return (
        role == "container"
        and first.source != second.source
        and bool(first_tokens & second_tokens)
        and box_iou >= 0.30
    )


def _crop_origins(length: int, crop_size: int, overlap: int) -> list[int]:
    if length <= crop_size:
        return [0]
    step = max(crop_size - overlap, 1)
    origins = list(range(0, max(length - crop_size, 0) + 1, step))
    last = length - crop_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def generate_mask_candidates(
    image: np.ndarray,
    generator,
    crop_size: int = 768,
    overlap: int = 128,
    include_geometry: bool = True,
    min_score: float = 0.0,
) -> list[MaskCandidate]:
    if crop_size <= 0 or overlap < 0 or overlap >= crop_size:
        raise ValueError(
            "crop_size must be > 0 and overlap must satisfy 0 <= overlap < crop_size"
        )

    height, width = image.shape[:2]
    crop_boxes = [(0, 0, width, height)]
    if height > crop_size or width > crop_size:
        crop_height = min(crop_size, height)
        crop_width = min(crop_size, width)
        for y in _crop_origins(height, crop_height, overlap):
            for x in _crop_origins(width, crop_width, overlap):
                crop_boxes.append(
                    (x, y, min(x + crop_width, width), min(y + crop_height, height))
                )

    candidates = []
    seen_boxes = set()
    for x1, y1, x2, y2 in crop_boxes:
        box = (x1, y1, x2, y2)
        if box in seen_boxes:
            continue
        seen_boxes.add(box)
        crop = image[y1:y2, x1:x2]
        with _sam_inference_context(generator):
            records = generator.generate(crop)
        for record in records:
            score = min(
                float(record.get("predicted_iou", 0.0)),
                float(record.get("stability_score", 0.0)),
            )
            if score < min_score:
                continue
            segmentation = record.pop("segmentation")
            if isinstance(segmentation, dict):
                rle_to_mask = importlib.import_module(
                    "sam2.utils.amg"
                ).rle_to_mask
                mask = np.asarray(rle_to_mask(segmentation), dtype=bool)
            else:
                mask = np.asarray(segmentation, dtype=bool)
            if (x1, y1, x2, y2) == (0, 0, width, height):
                full_mask = mask
            else:
                full_mask = np.zeros((height, width), dtype=bool)
                full_mask[y1:y2, x1:x2] = mask
            touches_crop_edge = bool(
                (x1 > 0 and np.any(mask[:, 0]))
                or (x2 < width and np.any(mask[:, -1]))
                or (y1 > 0 and np.any(mask[0, :]))
                or (y2 < height and np.any(mask[-1, :]))
            )
            candidates.append(
                MaskCandidate(full_mask, score, "sam", box, touches_crop_edge)
            )

    if include_geometry:
        candidates.extend(generate_geometry_candidates(image))
    return candidates


def _mask_box_fill(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    height, width = mask.shape
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(width, int(np.ceil(box[2])))
    y2 = min(height, int(np.ceil(box[3])))
    return float(np.count_nonzero(mask[y1:y2, x1:x2])) / max(
        (x2 - x1) * (y2 - y1),
        1,
    )


def _positive_hits(mask: np.ndarray, points: np.ndarray) -> int:
    height, width = mask.shape
    return sum(
        bool(
            mask[
                min(height - 1, max(0, int(round(y)))),
                min(width - 1, max(0, int(round(x)))),
            ]
        )
        for x, y in points
    )


def _drop_small_mask_islands(
    mask: np.ndarray,
    min_relative_area: float = 0.10,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:
        return np.asarray(mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(20, int(np.max(areas) * min_relative_area))
    keep_labels = np.flatnonzero(areas >= min_area) + 1
    return np.isin(labels, keep_labels)


def _select_person_mask(
    generator,
    predictor,
    box: np.ndarray,
    a_mask: np.ndarray,
    a_score: float,
) -> tuple[np.ndarray, float]:
    if _mask_box_fill(a_mask, tuple(box.tolist())) < 0.70:
        return a_mask, a_score

    x_mid = (box[0] + box[2]) / 2
    positive = np.asarray(
        [
            [x_mid, box[1] + (box[3] - box[1]) * fraction]
            for fraction in (0.25, 0.50, 0.65)
        ],
        dtype=np.float32,
    )
    inset_x = (box[2] - box[0]) * 0.10
    inset_y = (box[3] - box[1]) * 0.10
    negative = np.asarray(
        [
            [box[0] + inset_x, box[1] + inset_y],
            [box[2] - inset_x, box[1] + inset_y],
            [box[0] + inset_x, box[3] - inset_y],
            [box[2] - inset_x, box[3] - inset_y],
        ],
        dtype=np.float32,
    )
    with _sam_inference_context(generator):
        masks, scores, _ = predictor.predict(
            point_coords=np.vstack((positive, negative)),
            point_labels=np.asarray(
                [1, 1, 1, 0, 0, 0, 0], dtype=np.int32
            ),
            box=box,
            multimask_output=True,
        )
    eligible = [
        (np.asarray(mask, dtype=bool), float(score))
        for mask, score in zip(masks, scores, strict=True)
        if float(score) >= a_score - 0.10
        and _positive_hits(np.asarray(mask, dtype=bool), positive) >= 2
    ]
    if not eligible:
        return a_mask, a_score
    return min(
        eligible,
        key=lambda item: (
            _mask_box_fill(item[0], tuple(box.tolist())),
            -item[1],
        ),
    )


def generate_prompted_mask_candidates(
    image: np.ndarray,
    proposals,
    generator,
    text_mask: np.ndarray,
    *,
    set_image: bool = True,
) -> list[MaskCandidate]:
    predictor = generator.predictor
    if set_image:
        with _sam_inference_context(generator):
            predictor.set_image(image)

    candidates = []
    for proposal in proposals:
        box = np.asarray(proposal.box_xyxy, dtype=np.float32)
        with _sam_inference_context(generator):
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=True,
            )
        best_index = int(np.argmax(scores))
        a_mask = _drop_small_mask_islands(masks[best_index])
        a_score = float(scores[best_index])
        requested_roles = (
            ("container", "person")
            if proposal.role == "mixed"
            else (proposal.role,)
        )
        for role in requested_roles:
            mask, sam_score = (
                _select_person_mask(
                    generator,
                    predictor,
                    box,
                    a_mask,
                    a_score,
                )
                if role == "person"
                else (a_mask, a_score)
            )
            mask = _drop_small_mask_islands(mask)
            visible = np.asarray(mask, dtype=bool) & (text_mask == 0)
            if np.count_nonzero(visible) < 20:
                continue
            candidates.append(
                MaskCandidate(
                    mask=visible,
                    score=min(float(proposal.score), sam_score),
                    source=f"grounded:{proposal.source}:{role}",
                    crop_box=proposal.crop_box,
                    touches_crop_edge=proposal.touches_crop_edge,
                    label=proposal.label,
                    role=role,
                    object_box=tuple(float(value) for value in proposal.box_xyxy),
                )
            )
    return candidates


def filter_prompt_free_candidates(
    candidates: list[MaskCandidate],
    grounded_candidates: list[MaskCandidate],
    text_mask: np.ndarray,
    duplicate_containment: float = 0.60,
    duplicate_iou: float = 0.50,
    nested_containment: float = 0.80,
    min_area_fraction: float = 0.0005,
    min_score: float = 0.90,
) -> list[MaskCandidate]:
    """Keep prompt-free masks that add visual ownership beyond grounded objects."""
    min_area = max(20, int(text_mask.size * min_area_fraction))
    retained = []
    for candidate in candidates:
        visible = _drop_small_mask_islands(candidate.mask) & (text_mask == 0)
        area = int(np.count_nonzero(visible))
        if area < min_area or candidate.score < min_score:
            continue
        duplicate = None
        max_containment = 0.0
        for grounded in grounded_candidates:
            grounded_mask = np.asarray(grounded.mask, dtype=bool)
            grounded_area = int(np.count_nonzero(grounded_mask))
            intersection = int(np.count_nonzero(visible & grounded_mask))
            union = area + grounded_area - intersection
            containment = intersection / area
            max_containment = max(max_containment, containment)
            if (
                containment >= duplicate_containment
                and intersection / max(union, 1) >= duplicate_iou
            ):
                duplicate = grounded
                break
        if duplicate is not None:
            duplicate.mask = np.asarray(duplicate.mask, dtype=bool) | visible
            continue
        if max_containment >= nested_containment:
            continue
        candidate.mask = visible
        retained.append(candidate)
    return retained


def filter_unchanged_residual_candidates(
    source: np.ndarray,
    clean_background: np.ndarray,
    candidates: list[MaskCandidate],
    text_mask: np.ndarray,
    unchanged_threshold: int = 8,
    unchanged_fraction: float = 0.75,
):
    difference = np.max(
        np.abs(source.astype(np.int16) - clean_background.astype(np.int16)),
        axis=2,
    )
    retained = []
    for candidate in candidates:
        valid = np.asarray(candidate.mask, dtype=bool) & (text_mask == 0)
        if not np.any(valid):
            continue
        unchanged = difference[valid] < unchanged_threshold
        if float(np.mean(unchanged)) >= unchanged_fraction:
            retained.append(candidate)
    return retained


def reconcile_residual_candidates(
    residual_candidates: list[MaskCandidate],
    existing_candidates: list[MaskCandidate],
    image_shape: tuple[int, int],
) -> tuple[list[MaskCandidate], int]:
    """Attach structural fragments and reject unassigned edge background."""
    height, width = image_shape
    contact_radius = max(2, int(round(min(height, width) * 0.003)))
    kernel = np.ones((contact_radius * 2 + 1,) * 2, dtype=np.uint8)
    completion_radius = max(
        contact_radius + 2,
        int(round(min(height, width) * 0.009)),
    )
    completion_kernel = np.ones(
        (completion_radius * 2 + 1,) * 2, dtype=np.uint8
    )
    containers = [
        candidate for candidate in existing_candidates if candidate.role == "container"
    ]
    structural_tokens = {"line", "border", "frame", "decoration"}
    retained = []
    attached = 0

    for residual in residual_candidates:
        mask = np.asarray(residual.mask, dtype=bool)
        tokens = {token.strip(".,").lower() for token in residual.label.split()}
        target = None
        best_contact = 0.0
        if tokens & structural_tokens:
            area = max(int(np.count_nonzero(mask)), 1)
            for container in containers:
                expanded = cv2.dilate(
                    np.asarray(container.mask, dtype=np.uint8), kernel, iterations=1
                ).astype(bool)
                contact = float(np.count_nonzero(mask & expanded)) / area
                if contact > best_contact:
                    target = container
                    best_contact = contact
        if target is not None and best_contact >= 0.15:
            target.mask = np.asarray(target.mask, dtype=bool) | mask
            attached += 1
            continue

        if residual.score >= 0.24:
            target = None
            best_contact = 0.0
            area = max(int(np.count_nonzero(mask)), 1)
            for container in containers:
                expanded = cv2.dilate(
                    np.asarray(container.mask, dtype=np.uint8),
                    completion_kernel,
                    iterations=1,
                ).astype(bool)
                contact = float(np.count_nonzero(mask & expanded)) / area
                if contact > best_contact:
                    target = container
                    best_contact = contact
            if target is not None and best_contact >= 0.25:
                target.mask = np.asarray(target.mask, dtype=bool) | mask
                attached += 1
                continue

        touches_image_edge = bool(
            np.any(mask[0, :])
            or np.any(mask[-1, :])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        )
        if touches_image_edge:
            continue
        if residual.score < 0.24:
            continue
        retained.append(residual)

    return retained, attached


def generate_geometry_candidates(
    image: np.ndarray,
    min_area: int = 20,
) -> list[MaskCandidate]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []
    max_area = image.shape[0] * image.shape[1] * 0.9
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        candidates.append(MaskCandidate(mask > 0, 0.70, "geometry"))
    return candidates


def resolve_sam_checkpoint(cache_dir=None, downloader=urlretrieve) -> Path:
    cache_root = Path(
        cache_dir
        or os.environ.get("IMAGE2EDITABLE_MODEL_CACHE")
        or Path.home() / ".cache" / "image2editable"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = cache_root / "sam2.1_hiera_large.pt"
    if checkpoint_path.exists() and checkpoint_path.stat().st_size:
        return checkpoint_path

    partial_path = checkpoint_path.with_suffix(".pt.part")
    try:
        downloader(SAM21_LARGE_URL, str(partial_path))
        if not partial_path.exists() or not partial_path.stat().st_size:
            raise VisualSegmentationError("SAM 2.1 checkpoint download was empty")
        partial_path.replace(checkpoint_path)
    except VisualSegmentationError:
        partial_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial_path.unlink(missing_ok=True)
        raise VisualSegmentationError(
            f"Unable to download SAM 2.1 checkpoint: {exc}"
        ) from exc

    return checkpoint_path


def _build_resource_safe_sam_model(
    build_sam,
    torch,
    checkpoint_path,
    selected_device,
):
    init_empty_weights = importlib.import_module(
        "accelerate"
    ).init_empty_weights
    config = build_sam.compose(config_name=SAM21_LARGE_CONFIG)
    build_sam.OmegaConf.resolve(config)
    with init_empty_weights():
        model = build_sam.instantiate(
            config["model"],
            _recursive_=True,
        )
    state = torch.load(
        checkpoint_path,
        map_location=selected_device,
        weights_only=True,
        mmap=True,
    )["model"]
    missing_keys, unexpected_keys = model.load_state_dict(
        state,
        assign=True,
    )
    if missing_keys or unexpected_keys:
        raise VisualSegmentationError(
            "SAM 2.1 checkpoint does not match the Large model"
        )
    return model.eval()


def create_sam_generator(
    checkpoint_path,
    device=None,
    resource_safe=False,
):
    try:
        torch = importlib.import_module("torch")
        build_sam = importlib.import_module("sam2.build_sam")
        mask_generator = importlib.import_module("sam2.automatic_mask_generator")
    except ModuleNotFoundError as exc:
        raise VisualSegmentationError(
            "SAM 2.1 is required. Install project segmentation dependencies."
        ) from exc

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        _build_resource_safe_sam_model(
            build_sam,
            torch,
            checkpoint_path,
            selected_device,
        )
        if resource_safe
        else build_sam.build_sam2(
            SAM21_LARGE_CONFIG,
            str(checkpoint_path),
            device=selected_device,
            apply_postprocessing=False,
        )
    )
    generator = mask_generator.SAM2AutomaticMaskGenerator(
        model,
        points_per_side=16,
        points_per_batch=1 if resource_safe else 4,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=20,
        output_mode=(
            "uncompressed_rle" if resource_safe else "binary_mask"
        ),
    )
    generator._image2editable_device = selected_device
    return generator
