from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable, MutableMapping


logger = logging.getLogger(__name__)

_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "FLAGS_paddle_num_threads",
)
_POLICY_FIELDS = {
    "name",
    "cpu_threads",
    "heavy_page_concurrency",
    "sam_points_per_batch",
}


def safe_default_policy(cpu_count: int | None = None) -> dict[str, object]:
    logical_cpu_count = os.cpu_count() if cpu_count is None else cpu_count
    logical_cpu_count = max(1, logical_cpu_count or 1)
    return {
        "name": "safe-default",
        "cpu_threads": min(8, max(1, logical_cpu_count // 2)),
        "heavy_page_concurrency": 1,
        "sam_points_per_batch": 1,
    }


def validate_resource_policy(policy: object) -> dict[str, object]:
    if type(policy) is not dict:
        raise ValueError("Run manifest resource policy must be an object")
    if set(policy) != _POLICY_FIELDS:
        raise ValueError("Run manifest resource policy fields are invalid")
    if policy["name"] != "safe-default":
        raise ValueError("Run manifest resource policy name is invalid")
    cpu_threads = policy["cpu_threads"]
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
        raise ValueError("Run manifest resource policy cpu_threads is invalid")
    if (
        type(policy["heavy_page_concurrency"]) is not int
        or policy["heavy_page_concurrency"] != 1
    ):
        raise ValueError(
            "Run manifest resource policy heavy_page_concurrency is invalid"
        )
    if (
        type(policy["sam_points_per_batch"]) is not int
        or policy["sam_points_per_batch"] not in (1, 4)
    ):
        raise ValueError(
            "Run manifest resource policy sam_points_per_batch is invalid"
        )
    return dict(policy)


def _lower_process_priority() -> None:
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        set_priority_class = kernel32.SetPriorityClass
        set_priority_class.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        set_priority_class.restype = ctypes.c_int
        if not set_priority_class(get_current_process(), 0x00004000):
            raise ctypes.WinError()
        return
    if os.getpriority(os.PRIO_PROCESS, 0) < 5:
        os.setpriority(os.PRIO_PROCESS, 0, 5)


def apply_resource_policy(
    policy: object,
    *,
    environ: MutableMapping[str, str] | None = None,
    priority_setter: Callable[[], None] | None = None,
) -> list[str]:
    validated = validate_resource_policy(policy)
    environment = os.environ if environ is None else environ
    thread_budget = str(validated["cpu_threads"])
    for name in _THREAD_ENVIRONMENT:
        if name in environment:
            logger.info(
                "Preserving explicit thread environment %s=%s",
                name,
                environment[name],
            )
        environment.setdefault(name, thread_budget)

    set_priority = (
        _lower_process_priority if priority_setter is None else priority_setter
    )
    try:
        set_priority()
    except OSError as error:
        warning = f"Could not lower process priority: {error}"
        logger.warning(warning)
        return [warning]
    return []
