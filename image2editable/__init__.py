"""Public API for the image2editable runtime."""

from image2editable.contracts import PageStatus, RunStatus, SCHEMA_VERSION
from image2editable.doctor import check_environment

__all__ = [
    "PageStatus",
    "RunStatus",
    "SCHEMA_VERSION",
    "check_environment",
    "convert",
    "get_status",
    "next_candidate",
    "prepare_job",
    "record_decision",
    "recover_job",
    "rerender_pdf_page",
    "retry_page",
    "run_job",
]


def __getattr__(name: str):
    # Model preparation and diagnostics must work before conversion can import.
    if name in __all__:
        from importlib import import_module

        value = getattr(import_module("image2editable.runtime"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
