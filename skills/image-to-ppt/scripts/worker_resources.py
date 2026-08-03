"""Resource handling for blocking heavyweight worker processes."""

from __future__ import annotations

import ctypes
import gc
import os
import subprocess


def _empty_current_process_working_set_windows() -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    empty_working_set = psapi.EmptyWorkingSet
    empty_working_set.argtypes = [ctypes.c_void_p]
    empty_working_set.restype = ctypes.c_int
    if not empty_working_set(get_current_process()):
        raise OSError(ctypes.get_last_error(), "EmptyWorkingSet failed")


def trim_parent_working_set_before_worker() -> None:
    gc.collect()
    if os.name != "nt":
        return
    try:
        _empty_current_process_working_set_windows()
    except OSError:
        return


def run_isolated_worker(command: list[str], **kwargs):
    trim_parent_working_set_before_worker()
    return subprocess.run(command, **kwargs)
