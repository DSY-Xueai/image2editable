from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass

import numpy as np
from PIL import Image


MODEL_ID = "IDEA-Research/grounding-dino-tiny"
OBJECT_PROMPT = (
    "person. portrait. player. card. panel. flag. icon. information icon. logo. badge. trophy. "
    "medal. wreath. table. chart. frame. border. line. decoration."
)

PERSON_TOKENS = {"person", "player", "portrait"}
CONTAINER_TOKENS = {"card", "panel", "chart", "table", "frame", "border"}
STRONG_CONTAINER_TOKENS = {"card", "panel", "table", "frame"}


@dataclass(frozen=True)
class ObjectProposal:
    box_xyxy: tuple[float, float, float, float]
    score: float
    label: str
    role: str
    source: str
    crop_box: tuple[int, int, int, int]
    touches_crop_edge: bool = False


class _LazyGroundingDino:
    def __init__(self, model_id: str, device: str | None) -> None:
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(
                self.model_id, local_files_only=True
            )
            model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, local_files_only=True
            )
        except OSError:
            self.processor = transformers.AutoProcessor.from_pretrained(self.model_id)
            model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id
            )
        self.model = model.to(self.device).eval()

    def detect(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        self._load()
        torch = importlib.import_module("torch")
        pil_image = Image.fromarray(image)
        inputs = self.processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        post_process = self.processor.post_process_grounded_object_detection
        threshold_name = (
            "box_threshold"
            if "box_threshold" in inspect.signature(post_process).parameters
            else "threshold"
        )
        result = post_process(
            outputs,
            inputs.input_ids,
            **{
                threshold_name: box_threshold,
                "text_threshold": text_threshold,
                "target_sizes": [(pil_image.height, pil_image.width)],
            },
        )[0]
        labels = result.get("text_labels")
        if labels is None:
            labels = result.get("labels", [])
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        boxes = result["boxes"].detach().cpu().tolist()
        return [
            {
                "box_xyxy": tuple(float(value) for value in box),
                "score": float(score),
                "label": str(label),
            }
            for label, score, box in zip(labels, scores, boxes, strict=True)
        ]


def create_object_detector(
    model_id: str = MODEL_ID,
    device: str | None = None,
):
    detector = _LazyGroundingDino(model_id, device)
    detector._load()
    return detector


def classify_object_role(label: str) -> str:
    tokens = _tokens(label)
    person = bool(tokens & PERSON_TOKENS)
    container = bool(tokens & CONTAINER_TOKENS)
    if person and container:
        return "mixed"
    if person:
        return "person"
    if container:
        return "container"
    return "object"


def _tokens(label: str) -> set[str]:
    return {token.strip(".,").lower() for token in label.split()}


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(_box_area(first) + _box_area(second) - intersection, 1.0)


def _crop_origins(length: int, crop_size: int, overlap: int) -> list[int]:
    if length <= crop_size:
        return [0]
    step = crop_size - overlap
    origins = list(range(0, length - crop_size + 1, step))
    last = length - crop_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def _contains_center(
    container: tuple[float, float, float, float],
    child: tuple[float, float, float, float],
) -> bool:
    center_x = (child[0] + child[2]) / 2
    center_y = (child[1] + child[3]) / 2
    return (
        container[0] <= center_x <= container[2]
        and container[1] <= center_y <= container[3]
    )


def filter_object_proposals(
    proposals: list[ObjectProposal],
    image_shape: tuple[int, int],
) -> list[ObjectProposal]:
    height, width = image_shape
    canvas_area = max(height * width, 1)
    eligible = []
    for proposal in proposals:
        area = _box_area(proposal.box_xyxy)
        box_width = max(0.0, proposal.box_xyxy[2] - proposal.box_xyxy[0])
        box_height = max(0.0, proposal.box_xyxy[3] - proposal.box_xyxy[1])
        crop_area = max(
            (proposal.crop_box[2] - proposal.crop_box[0])
            * (proposal.crop_box[3] - proposal.crop_box[1]),
            1,
        )
        if area / canvas_area >= 0.85:
            continue
        if "portrait" in _tokens(proposal.label) and area / crop_area >= 0.80:
            continue
        if (
            _tokens(proposal.label) == {"information", "icon"}
            and min(box_width, box_height) / max(box_width, box_height, 1.0) < 0.85
        ):
            continue
        eligible.append(proposal)

    retained = []
    for proposal in sorted(
        eligible,
        key=lambda item: (item.touches_crop_edge, -item.score),
    ):
        duplicate = False
        for kept in retained:
            if proposal.role != kept.role:
                continue
            if (
                proposal.role == "object"
                and not (_tokens(proposal.label) & _tokens(kept.label))
            ):
                continue
            smaller = min(_box_area(proposal.box_xyxy), _box_area(kept.box_xyxy))
            larger = max(_box_area(proposal.box_xyxy), _box_area(kept.box_xyxy), 1.0)
            if (
                smaller / larger >= 0.85
                and _box_iou(proposal.box_xyxy, kept.box_xyxy) >= 0.75
            ):
                duplicate = True
                break
        if not duplicate:
            retained.append(proposal)

    people = [proposal for proposal in retained if proposal.role == "person"]
    result = []
    for proposal in retained:
        tokens = _tokens(proposal.label)
        if proposal.role not in {"container", "mixed"} or tokens & STRONG_CONTAINER_TOKENS:
            result.append(proposal)
            continue
        contained = [
            person
            for person in people
            if _contains_center(proposal.box_xyxy, person.box_xyxy)
        ]
        spans_two_people = any(
            _box_iou(first.box_xyxy, second.box_xyxy) <= 0.05
            for index, first in enumerate(contained)
            for second in contained[index + 1 :]
        )
        if not spans_two_people:
            result.append(proposal)

    return sorted(result, key=lambda item: (item.box_xyxy[1], item.box_xyxy[0], -item.score))


def filter_text_overlapping_proposals(
    proposals: list[ObjectProposal],
    text_mask: np.ndarray,
    max_text_fraction: float = 0.50,
) -> list[ObjectProposal]:
    """Drop object proposals that are predominantly OCR text boxes."""
    height, width = text_mask.shape
    retained = []
    for proposal in proposals:
        if proposal.role != "object":
            retained.append(proposal)
            continue
        x1 = max(0, int(np.floor(proposal.box_xyxy[0])))
        y1 = max(0, int(np.floor(proposal.box_xyxy[1])))
        x2 = min(width, int(np.ceil(proposal.box_xyxy[2])))
        y2 = min(height, int(np.ceil(proposal.box_xyxy[3])))
        area = max((x2 - x1) * (y2 - y1), 1)
        text_fraction = int(np.count_nonzero(text_mask[y1:y2, x1:x2])) / area
        if text_fraction <= max_text_fraction:
            retained.append(proposal)
    return retained


def generate_object_proposals(
    image: np.ndarray,
    detector,
    crop_size: int = 768,
    overlap: int = 128,
    box_threshold: float = 0.18,
    text_threshold: float = 0.15,
    prompt: str = OBJECT_PROMPT,
) -> list[ObjectProposal]:
    if crop_size <= 0 or overlap < 0 or overlap >= crop_size:
        raise ValueError("crop_size must be > 0 and 0 <= overlap < crop_size")
    height, width = image.shape[:2]
    crop_width = min(crop_size, width)
    crop_height = min(crop_size, height)
    crop_boxes = [(0, 0, width, height)]
    for y in _crop_origins(height, crop_height, overlap):
        for x in _crop_origins(width, crop_width, overlap):
            box = (x, y, x + crop_width, y + crop_height)
            if box not in crop_boxes:
                crop_boxes.append(box)

    proposals = []
    for index, crop_box in enumerate(crop_boxes):
        x1, y1, x2, y2 = crop_box
        crop = image[y1:y2, x1:x2]
        source = "full" if index == 0 else f"tile_{index}"
        for record in detector.detect(
            crop,
            prompt,
            box_threshold,
            text_threshold,
        ):
            local = tuple(float(value) for value in record["box_xyxy"])
            global_box = (
                local[0] + x1,
                local[1] + y1,
                local[2] + x1,
                local[3] + y1,
            )
            touches_crop_edge = bool(
                (x1 > 0 and local[0] <= 1)
                or (x2 < width and local[2] >= crop.shape[1] - 1)
                or (y1 > 0 and local[1] <= 1)
                or (y2 < height and local[3] >= crop.shape[0] - 1)
            )
            label = str(record["label"])
            proposals.append(
                ObjectProposal(
                    box_xyxy=global_box,
                    score=float(record["score"]),
                    label=label,
                    role=classify_object_role(label),
                    source=source,
                    crop_box=crop_box,
                    touches_crop_edge=touches_crop_edge,
                )
            )
    return filter_object_proposals(proposals, (height, width))
