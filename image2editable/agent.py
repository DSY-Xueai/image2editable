from __future__ import annotations

from datetime import datetime, timedelta
import math
import os
from pathlib import Path
import stat
from typing import Sequence

from image2editable.contracts import (
    PageStatus,
    RunStatus,
    SCHEMA_VERSION,
    utc_now,
    validate_schema_version,
)
from image2editable.execution import ExecutionLease
from image2editable.inputs import sha256_file
from image2editable.pptx_input import validate_pptx_inventories
from image2editable.store import RunStore


REPLACE_CONFIDENCE = 0.92
DECISIONS = frozenset({"replace", "preserve", "ambiguous"})
CATEGORIES = frozenset(
    {
        "full_slide_screenshot",
        "partial_slide_screenshot",
        "rasterized_diagram",
        "rasterized_chart",
        "photo",
        "logo",
        "decorative_asset",
        "unknown",
    }
)


def next_candidate(run_dir: str | Path) -> dict[str, object]:
    store, manifest = _open_prepared_pptx(run_dir)
    for page_id in manifest["pages"]:
        request, candidates = _page_candidates(store, page_id)
        document = _decision_document(store, page_id, candidates)
        decided = {item["source_shape_id"] for item in document["decisions"]}
        for candidate in candidates:
            if candidate["source_shape_id"] in decided:
                continue
            image_path = _candidate_image_path(store, page_id, candidate)
            return {
                "candidate": {
                    **candidate,
                    "page_id": page_id,
                    "slide_index": request["slide_index"],
                    "native_object_counts": request["native_object_counts"],
                    "image_path": str(image_path),
                    "allowed_decisions": sorted(DECISIONS),
                    "allowed_categories": sorted(CATEGORIES),
                    "replace_confidence_threshold": REPLACE_CONFIDENCE,
                }
            }
    return {"candidate": None}


def record_decision(
    run_dir: str | Path,
    *,
    page_id: str,
    object_id: str,
    decision: str,
    confidence: float,
    category: str,
    evidence: Sequence[str],
) -> dict[str, object]:
    store = RunStore.open(run_dir)
    with ExecutionLease(
        store.root / "execution.lock",
        run_root=store.root,
    ):
        return _record_decision(
            store.root,
            page_id=page_id,
            object_id=object_id,
            decision=decision,
            confidence=confidence,
            category=category,
            evidence=evidence,
        )


def shadow_replacement_plans(
    store: RunStore,
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    plans = []
    for page_id in manifest["pages"]:
        request, candidates = _page_candidates(store, page_id)
        decisions = _decision_document(store, page_id, candidates)["decisions"]
        by_shape = {
            candidate["source_shape_id"]: candidate
            for candidate in candidates
        }
        shadow_decisions = [
            decision for decision in decisions
            if decision["runtime_action"] == "shadow_run"
        ]
        if not shadow_decisions:
            continue
        decision = shadow_decisions[0]
        candidate = by_shape[decision["source_shape_id"]]
        work_root = _owned_reconstruction_root(store, page_id)
        plan = {
            "page_id": page_id,
            "slide_part": request["slide_part"],
            "image_path": str(
                _candidate_image_path(
                    store,
                    page_id,
                    candidate,
                )
            ),
            "work_root": str(work_root),
            "decision": decision,
        }
        state_path = store.root / "pages" / page_id / "reconstruction" / "component_state.json"
        if state_path.is_file():
            state = store.read_json(
                f"pages/{page_id}/reconstruction/component_state.json"
            )
            if state.get("phase") == "ready_for_assembly" and state.get("result_ref"):
                result_ref = state["result_ref"]
                if (
                    not isinstance(result_ref, dict)
                    or set(result_ref) != {"path", "sha256"}
                    or not isinstance(result_ref["path"], str)
                    or Path(result_ref["path"]).is_absolute()
                ):
                    raise RuntimeError("component result reference is invalid")
                result_path = (store.root / Path(result_ref["path"])).resolve()
                if (
                    not result_path.is_relative_to(store.root.resolve())
                    or not result_path.is_file()
                    or sha256_file(result_path) != result_ref["sha256"]
                ):
                    raise RuntimeError("component result reference is invalid")
                plan.update({
                    "provider": state.get("provider", "host"),
                    "component_result_path": str(
                        result_path
                    ),
                    "component_result_sha256": result_ref["sha256"],
                    # The accepted result is bound to the immutable evidence
                    # ``source.png`` snapshot.  That PNG can be re-encoded
                    # from the original PPTX media, so its hash is not
                    # necessarily the container-media hash in page_request.
                    "source_screenshot_sha256": state["source_sha256"],
                    "initial_component_count": state.get(
                        "initial_component_count", 0
                    ),
                })
            elif state.get("phase") == "preserved_with_warning":
                plan.update({
                    "provider": state.get("provider", "host"),
                    "conflict_warning": (
                        "component reconstruction preserved the original page: "
                        f"{state.get('stop_reason') or 'quality gate failed'}"
                    ),
                })
        if len(shadow_decisions) > 1:
            plan["conflict_warning"] = (
                "multiple Agent-approved screenshots on one page"
            )
        plans.append(plan)
    return plans


def _owned_reconstruction_root(store: RunStore, page_id: str) -> Path:
    current = store.root
    for name in ("pages", page_id):
        current /= name
        try:
            status = current.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(
                f"PPTX page directory is missing: {current}"
            ) from error
        if _is_link_or_reparse(status):
            raise RuntimeError(
                f"PPTX page directory is a link or reparse point: {current}"
            )
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(f"PPTX page path is not a directory: {current}")

    reconstruction = current / "reconstruction"
    try:
        status = reconstruction.lstat()
    except FileNotFoundError:
        return reconstruction
    if _is_link_or_reparse(status):
        raise RuntimeError(
            "PPTX reconstruction directory is a link or reparse point: "
            f"{reconstruction}"
        )
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(
            f"PPTX reconstruction path is not a directory: {reconstruction}"
        )
    resolved = reconstruction.resolve()
    if not resolved.is_relative_to(current):
        raise RuntimeError(
            "PPTX reconstruction directory is outside its page directory: "
            f"{reconstruction}"
        )
    return resolved


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )


def _record_decision(
    run_dir: str | Path,
    *,
    page_id: str,
    object_id: str,
    decision: str,
    confidence: float,
    category: str,
    evidence: Sequence[str],
) -> dict[str, object]:
    store, manifest = _open_prepared_pptx(run_dir)
    if page_id not in manifest["pages"]:
        raise KeyError(f"Unknown page_id: {page_id}")
    _, candidates = _page_candidates(store, page_id)
    matches = [
        candidate
        for candidate in candidates
        if candidate["source_shape_id"] == object_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Object is not a screenshot candidate: {page_id}/{object_id}")
    if decision not in DECISIONS:
        raise ValueError(f"Unsupported decision: {decision}")
    if category not in CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence must be a finite number from 0 to 1")
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise ValueError("evidence must be a non-empty string list")
    normalized_evidence = [
        item.strip() for item in evidence if isinstance(item, str) and item.strip()
    ]
    if len(normalized_evidence) != len(evidence) or not normalized_evidence:
        raise ValueError("evidence must be a non-empty string list")

    path = Path("pages") / page_id / "decision.json"
    document = _decision_document(store, page_id, candidates)
    if any(
        item.get("source_shape_id") == object_id
        for item in document["decisions"]
        if isinstance(item, dict)
    ):
        raise ValueError(f"Decision already recorded: {page_id}/{object_id}")

    candidate = matches[0]
    eligible = (
        decision == "replace"
        and category == "full_slide_screenshot"
        and confidence >= REPLACE_CONFIDENCE
    )
    if eligible and any(
        item["runtime_action"] == "shadow_run"
        for item in document["decisions"]
    ):
        raise ValueError(
            f"Only one shadow replacement is allowed per page: {page_id}"
        )
    record = {
        "source_shape_id": object_id,
        "source_object_sha256": candidate["source_object_sha256"],
        "image_sha256": candidate["image_sha256"],
        "decision": decision,
        "confidence": float(confidence),
        "category": category,
        "evidence": normalized_evidence,
        "eligible_for_shadow_run": eligible,
        "runtime_action": "shadow_run" if eligible else "preserve",
        "recorded_at": utc_now(),
    }
    document["decisions"].append(record)
    store.write_json(path, document)
    if eligible:
        full_page_candidate = (
            type(candidate.get("slide_coverage")) is float
            and candidate.get("slide_coverage") == 1.0
        )
        page_request = {
            "schema_version": SCHEMA_VERSION,
            "page_id": page_id,
            "source": (Path("pages") / page_id / candidate["image"]).as_posix(),
            "sha256": candidate["image_sha256"],
            "full_page_candidate": full_page_candidate,
            "source_shape_id": candidate["source_shape_id"],
            "slide_coverage": candidate["slide_coverage"],
        }
        store.write_json(
            Path("pages") / page_id / "page_request.json", page_request
        )
    return record


def _open_prepared_pptx(
    run_dir: str | Path,
) -> tuple[RunStore, dict[str, object]]:
    store = RunStore.open(run_dir)
    manifest = store.read_json("job_manifest.json")
    validate_schema_version(manifest)
    input_record = manifest.get("input")
    if not isinstance(input_record, dict) or input_record.get("type") != "pptx":
        raise RuntimeError("Agent screenshot decisions require a PPTX run")
    state = store.read_json("run_state.json")
    if state.get("status") != RunStatus.PREPARED.value:
        raise RuntimeError("Agent screenshot decisions require a prepared run")
    page_jobs = store.read_json("page_jobs.json")
    if any(
        page.get("status") != PageStatus.ANALYZED.value
        for page in page_jobs["pages"].values()
    ):
        raise RuntimeError("PPTX pages must be analyzed before Agent decisions")
    validate_pptx_inventories(store, manifest)
    return store, manifest


def _page_candidates(
    store: RunStore,
    page_id: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    request = store.read_json(Path("pages") / page_id / "agent_request.json")
    validate_schema_version(request)
    if request.get("page_id") != page_id:
        raise RuntimeError(f"Agent request page mismatch: {page_id}")
    candidates = request.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError(f"Invalid Agent candidate list: {page_id}")
    inventory = store.read_json(Path("pages") / page_id / "screenshot_candidates.json")
    native = store.read_json(Path("pages") / page_id / "native_objects.json")
    authoritative = inventory["candidates"]
    if len(candidates) != len(authoritative):
        raise RuntimeError(f"Agent candidate count mismatch: {page_id}")
    metadata_keys = (
        "page_id",
        "slide_index",
        "slide_part",
        "slide_width",
        "slide_height",
    )
    counts: dict[str, int] = {}
    for item in native["objects"]:
        object_type = item["type"]
        counts[object_type] = counts.get(object_type, 0) + 1
    if (
        any(request.get(key) != inventory.get(key) for key in metadata_keys)
        or request.get("native_object_counts") != counts
    ):
        raise RuntimeError(f"Agent evidence mismatch: {page_id}")
    slide_width = inventory["slide_width"]
    slide_height = inventory["slide_height"]
    for index, (candidate, source) in enumerate(
        zip(candidates, authoritative), start=1
    ):
        expected = {
            "candidate_id": f"candidate_{index:03d}",
            "source_shape_id": source["shape_id"],
            "source_object_sha256": source["xml_c14n_sha256"],
            "image_sha256": source["media_sha256"],
            "media_format": source["media_format"],
            "pixel_width": source["pixel_width"],
            "pixel_height": source["pixel_height"],
            "slide_coverage": source["slide_coverage"],
            "edge_gaps": {
                "left": max(source["x"], 0) / slide_width,
                "top": max(source["y"], 0) / slide_height,
                "right": max(
                    slide_width - source["x"] - source["cx"],
                    0,
                )
                / slide_width,
                "bottom": max(
                    slide_height - source["y"] - source["cy"],
                    0,
                )
                / slide_height,
            },
            "z_order": source["z_order"],
        }
        if not isinstance(candidate, dict) or any(
            candidate.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError(f"Agent evidence mismatch: {page_id}")
        _candidate_image_path(store, page_id, candidate)
    return request, candidates


def _candidate_image_path(
    store: RunStore,
    page_id: str,
    candidate: dict[str, object],
) -> Path:
    value = candidate.get("image")
    if not isinstance(value, str) or Path(value).is_absolute():
        raise RuntimeError(f"Invalid Agent candidate asset path: {page_id}")
    page_root = (store.root / "pages" / page_id).resolve()
    path = (page_root / value).resolve()
    if not path.is_relative_to(page_root) or not path.is_file():
        raise RuntimeError(f"Invalid Agent candidate asset path: {page_id}")
    if sha256_file(path) != candidate.get("image_sha256"):
        raise RuntimeError(f"Agent candidate asset hash mismatch: {page_id}")
    return path


def _decision_document(
    store: RunStore,
    page_id: str,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    path = Path("pages") / page_id / "decision.json"
    by_shape = {candidate["source_shape_id"]: candidate for candidate in candidates}
    if len(by_shape) != len(candidates):
        raise RuntimeError(f"Invalid decision candidates: {page_id}")
    try:
        document = store.read_json(path)
    except FileNotFoundError:
        return {
            "schema_version": SCHEMA_VERSION,
            "page_id": page_id,
            "decisions": [],
        }
    validate_schema_version(document)
    if (
        set(document) != {"schema_version", "page_id", "decisions"}
        or document.get("page_id") != page_id
        or not isinstance(document.get("decisions"), list)
    ):
        raise RuntimeError(f"Invalid decision document: {page_id}")

    expected_keys = {
        "source_shape_id",
        "source_object_sha256",
        "image_sha256",
        "decision",
        "confidence",
        "category",
        "evidence",
        "eligible_for_shadow_run",
        "runtime_action",
        "recorded_at",
    }
    seen = set()
    for item in document["decisions"]:
        shape_id = item.get("source_shape_id") if isinstance(item, dict) else None
        candidate = by_shape.get(shape_id)
        decision = item.get("decision") if isinstance(item, dict) else None
        category = item.get("category") if isinstance(item, dict) else None
        confidence = item.get("confidence") if isinstance(item, dict) else None
        evidence = item.get("evidence") if isinstance(item, dict) else None
        eligible = (
            decision == "replace"
            and category == "full_slide_screenshot"
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(confidence)
            and confidence >= REPLACE_CONFIDENCE
        )
        if (
            not isinstance(item, dict)
            or set(item) != expected_keys
            or candidate is None
            or shape_id in seen
            or item.get("source_object_sha256") != candidate["source_object_sha256"]
            or item.get("image_sha256") != candidate["image_sha256"]
            or decision not in DECISIONS
            or category not in CATEGORIES
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(evidence, list)
            or not evidence
            or any(
                not isinstance(value, str) or not value or value != value.strip()
                for value in evidence
            )
            or item.get("eligible_for_shadow_run") is not eligible
            or item.get("runtime_action") != ("shadow_run" if eligible else "preserve")
            or not _canonical_utc_time(item.get("recorded_at"))
        ):
            raise RuntimeError(f"Invalid decision record: {page_id}")
        seen.add(shape_id)
    return document


def _canonical_utc_time(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        return False
    return (
        parsed.utcoffset() == timedelta(0)
        and parsed.isoformat().replace("+00:00", "Z") == value
    )
