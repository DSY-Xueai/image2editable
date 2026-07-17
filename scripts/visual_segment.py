from __future__ import annotations

import importlib
import json
import os
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


def validate_visual_masks(element_masks: list[np.ndarray]) -> None:
    if not element_masks:
        return
    ownership = np.sum(
        [np.asarray(mask, dtype=np.uint8) for mask in element_masks],
        axis=0,
    )
    if np.max(ownership) > 1:
        raise VisualSegmentationError("overlapping visual ownership detected")


def visual_difference(
    source: np.ndarray,
    reconstructed: np.ndarray,
    text_mask: np.ndarray,
) -> dict:
    valid = text_mask == 0
    if not np.any(valid):
        return {"mae": 0.0, "p95": 0.0}
    pixel_difference = np.mean(
        np.abs(source.astype(np.float32) - reconstructed.astype(np.float32)),
        axis=2,
    )[valid]
    return {
        "mae": float(np.mean(pixel_difference)),
        "p95": float(np.percentile(pixel_difference, 95)),
    }


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
    semantic_mask: np.ndarray
    z_index: int
    score: float
    source: str
    object_box: tuple[float, float, float, float] | None = None


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
        valid.append((candidate, area, bbox, candidate.mask.copy()))

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
                visible,
                semantic_support,
                0,
                candidate.score,
                candidate.source,
                candidate.object_box,
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
        for record in generator.generate(crop):
            mask = np.asarray(record["segmentation"], dtype=bool)
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y1:y2, x1:x2] = mask
            score = min(
                float(record.get("predicted_iou", 0.0)),
                float(record.get("stability_score", 0.0)),
            )
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
    masks, scores, _ = predictor.predict(
        point_coords=np.vstack((positive, negative)),
        point_labels=np.asarray([1, 1, 1, 0, 0, 0, 0], dtype=np.int32),
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
        predictor.set_image(image)

    candidates = []
    for proposal in proposals:
        box = np.asarray(proposal.box_xyxy, dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
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
                _select_person_mask(predictor, box, a_mask, a_score)
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
        retained.append(
            MaskCandidate(
                mask=visible,
                score=candidate.score,
                source=candidate.source,
                crop_box=candidate.crop_box,
                touches_crop_edge=candidate.touches_crop_edge,
                label=candidate.label,
                role=candidate.role,
                object_box=candidate.object_box,
            )
        )
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


def create_sam_generator(checkpoint_path, device=None):
    try:
        torch = importlib.import_module("torch")
        build_sam = importlib.import_module("sam2.build_sam")
        mask_generator = importlib.import_module("sam2.automatic_mask_generator")
    except ModuleNotFoundError as exc:
        raise VisualSegmentationError(
            "SAM 2.1 is required. Install project segmentation dependencies."
        ) from exc

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sam.build_sam2(
        SAM21_LARGE_CONFIG,
        str(checkpoint_path),
        device=selected_device,
        apply_postprocessing=False,
    )
    return mask_generator.SAM2AutomaticMaskGenerator(
        model,
        points_per_side=16,
        points_per_batch=16 if str(selected_device).startswith("cuda") else 4,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=20,
    )
