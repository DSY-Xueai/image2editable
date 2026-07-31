from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from typing import Any


SCHEMA_VERSION = 1


class RunStatus(str, Enum):
    CREATED = "created"
    PREPARED = "prepared"
    RUNNING = "running"
    AWAITING_AGENT = "awaiting_agent"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PageStatus(str, Enum):
    PENDING = "pending"
    ANALYZED = "analyzed"
    PRESERVED = "preserved"
    PROCESSING = "processing"
    AWAITING_AGENT = "awaiting_agent"
    VALIDATED = "validated"
    REPLACED = "replaced"
    PRESERVED_WITH_WARNING = "preserved_with_warning"
    FAILED = "failed"


_RUN_TRANSITIONS = {
    RunStatus.CREATED: {RunStatus.PREPARED, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.PREPARED: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.RUNNING: {
        RunStatus.AWAITING_AGENT,
        RunStatus.FINALIZING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.AWAITING_AGENT: {RunStatus.PREPARED},
    RunStatus.FINALIZING: {RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.FAILED: {RunStatus.PREPARED},
    RunStatus.COMPLETED: set(),
    RunStatus.CANCELLED: set(),
}

_PAGE_TRANSITIONS = {
    PageStatus.PENDING: {PageStatus.ANALYZED, PageStatus.PROCESSING, PageStatus.FAILED},
    PageStatus.ANALYZED: {PageStatus.PRESERVED, PageStatus.PROCESSING, PageStatus.FAILED},
    PageStatus.PROCESSING: {
        PageStatus.AWAITING_AGENT,
        PageStatus.VALIDATED,
        PageStatus.PRESERVED_WITH_WARNING,
        PageStatus.FAILED,
    },
    PageStatus.AWAITING_AGENT: {PageStatus.PROCESSING},
    PageStatus.VALIDATED: {PageStatus.REPLACED, PageStatus.FAILED},
    PageStatus.FAILED: {PageStatus.PENDING},
    PageStatus.PRESERVED: set(),
    PageStatus.REPLACED: set(),
    PageStatus.PRESERVED_WITH_WARNING: set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_schema_version(document: dict[str, Any]) -> None:
    version = document.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {version}")


def transition_run_document(
    document: dict[str, Any], target: RunStatus
) -> dict[str, Any]:
    if not isinstance(target, RunStatus):
        raise TypeError("target must be a RunStatus")
    validate_schema_version(document)
    current = RunStatus(document["status"])
    if target not in _RUN_TRANSITIONS[current]:
        raise ValueError(f"Invalid status transition: {current.value} -> {target.value}")
    updated = deepcopy(document)
    updated["status"] = target.value
    updated["updated_at"] = utc_now()
    return updated


def transition_page_document(
    document: dict[str, Any], target: PageStatus
) -> dict[str, Any]:
    if not isinstance(target, PageStatus):
        raise TypeError("target must be a PageStatus")
    validate_schema_version(document)
    current = PageStatus(document["status"])
    if target not in _PAGE_TRANSITIONS[current]:
        raise ValueError(f"Invalid status transition: {current.value} -> {target.value}")
    updated = deepcopy(document)
    updated["status"] = target.value
    updated["updated_at"] = utc_now()
    return updated
