from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import BinaryIO


class ExecutionLease:
    """Hold the non-blocking execution lock for one Run."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_root: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_root = (
            Path(run_root).resolve() if run_root is not None else None
        )
        self._file: BinaryIO | None = None
        self._file_locked = False
        self._parent_descriptor: int | None = None
        self._parent_locked = False

    def __enter__(self) -> ExecutionLease:
        if self._file is not None:
            raise RuntimeError(
                f"Run is already executing: {self.path.parent}"
            )
        path = self._validated_path()
        try:
            self._acquire_parent(path.parent)
            descriptor = _open_file(path)
            try:
                self._file = os.fdopen(descriptor, "r+b")
            except BaseException as error:
                try:
                    os.close(descriptor)
                except BaseException as cleanup_error:
                    _attach_cleanup_failure(error, cleanup_error)
                raise
            _validate_open_file(path, self._file)
            self._file.seek(0)
            try:
                _lock(self._file)
            except OSError as error:
                raise RuntimeError(
                    f"Run is already executing: {path.parent}"
                ) from error
            self._file_locked = True
            _validate_open_file(path, self._file)
            self._file.seek(0)
            self._file.write(b"\0")
            self._file.truncate(1)
            self._file.flush()
        except BaseException as error:
            self._release(error)
            raise
        self.path = path
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._release(exc)

    def assert_authorizes(self, run_root: str | Path) -> None:
        root = Path(run_root).resolve()
        if self._file is None or not self._file_locked:
            raise RuntimeError("Execution lease is not held")
        if self.run_root != root or self.path != root / "execution.lock":
            raise RuntimeError("Execution lease authorizes a different Run")
        if self._parent_descriptor is not None:
            if not self._parent_locked:
                raise RuntimeError("Execution lease parent is not held")
            _validate_open_parent(root, self._parent_descriptor)
        elif os.name != "nt":
            raise RuntimeError("Execution lease parent is not held")
        _validate_open_file(self.path, self._file)

    def _validated_path(self) -> Path:
        parent = self.path.parent.resolve()
        path = parent / self.path.name
        if self.run_root is not None and parent != self.run_root:
            raise RuntimeError(
                f"Execution lease is outside run directory: {self.path}"
            )
        return path

    def _acquire_parent(self, parent: Path) -> None:
        descriptor = _open_parent(parent)
        if descriptor is None:
            return
        self._parent_descriptor = descriptor
        _validate_open_parent(parent, descriptor)
        try:
            _lock_parent(descriptor)
        except OSError as error:
            raise RuntimeError(
                f"Run is already executing: {parent}"
            ) from error
        self._parent_locked = True
        _validate_open_parent(parent, descriptor)

    def _release(self, primary: BaseException | None) -> None:
        failures: list[BaseException] = []
        if self._file is not None:
            if self._file_locked:
                try:
                    _unlock(self._file)
                except BaseException as error:
                    failures.append(error)
            try:
                self._file.close()
            except BaseException as error:
                failures.append(error)
            self._file = None
            self._file_locked = False
        if self._parent_descriptor is not None:
            if self._parent_locked:
                try:
                    _unlock_parent(self._parent_descriptor)
                except BaseException as error:
                    failures.append(error)
            try:
                os.close(self._parent_descriptor)
            except BaseException as error:
                failures.append(error)
            self._parent_descriptor = None
            self._parent_locked = False
        if primary is not None:
            for failure in failures:
                _attach_cleanup_failure(primary, failure)
            return
        if failures:
            failure = failures[0]
            for extra in failures[1:]:
                _attach_cleanup_failure(failure, extra)
            raise failure


def _attach_cleanup_failure(
    primary: BaseException,
    cleanup: BaseException,
) -> None:
    note = f"Execution lease cleanup failed: {cleanup!r}"
    add_note = getattr(primary, "add_note", None)
    if add_note is not None:
        try:
            add_note(note)
        except BaseException:
            pass
        return
    try:
        tail = primary
        while tail.__cause__ is not None:
            tail = tail.__cause__
        tail.__cause__ = cleanup
    except BaseException:
        pass


def _open_file(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    for name in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    try:
        return os.open(path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(
            f"Execution lease cannot be opened safely: {path}"
        ) from error


def _validate_open_file(path: Path, file: BinaryIO) -> None:
    opened_status = os.fstat(file.fileno())
    try:
        path_status = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Execution lease path identity changed: {path}"
        ) from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(path_status.st_mode) or bool(
        getattr(path_status, "st_file_attributes", 0) & reparse_flag
    ):
        raise RuntimeError(
            f"Execution lease is a link or reparse point: {path}"
        )
    if not stat.S_ISREG(path_status.st_mode) or not stat.S_ISREG(
        opened_status.st_mode
    ):
        raise RuntimeError(f"Execution lease is not a regular file: {path}")
    if path_status.st_nlink != 1 or opened_status.st_nlink != 1:
        raise RuntimeError(f"Execution lease is an unsafe hard link: {path}")
    if (
        path_status.st_dev,
        path_status.st_ino,
    ) != (
        opened_status.st_dev,
        opened_status.st_ino,
    ):
        raise RuntimeError(
            f"Execution lease path identity changed: {path}"
        )


def _validate_open_parent(parent: Path, descriptor: int) -> None:
    opened_status = os.fstat(descriptor)
    try:
        path_status = parent.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Execution lease parent identity changed: {parent}"
        ) from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(path_status.st_mode) or bool(
        getattr(path_status, "st_file_attributes", 0) & reparse_flag
    ):
        raise RuntimeError(
            f"Execution lease parent is a link or reparse point: {parent}"
        )
    if not stat.S_ISDIR(path_status.st_mode) or not stat.S_ISDIR(
        opened_status.st_mode
    ):
        raise RuntimeError(
            f"Execution lease parent is not a directory: {parent}"
        )
    if (
        path_status.st_dev,
        path_status.st_ino,
    ) != (
        opened_status.st_dev,
        opened_status.st_ino,
    ):
        raise RuntimeError(
            f"Execution lease parent identity changed: {parent}"
        )


if os.name == "nt":
    import msvcrt

    def _lock(file: BinaryIO) -> None:
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(file: BinaryIO) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)

    def _open_parent(parent: Path) -> None:
        return None

    def _lock_parent(descriptor: int) -> None:
        raise AssertionError("Windows does not use a parent lease")

    def _unlock_parent(descriptor: int) -> None:
        raise AssertionError("Windows does not use a parent lease")

else:
    import fcntl

    def _lock(file: BinaryIO) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(file: BinaryIO) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)

    def _open_parent(parent: Path) -> int:
        directory_flag = getattr(os, "O_DIRECTORY", None)
        nofollow_flag = getattr(os, "O_NOFOLLOW", None)
        if directory_flag is None or nofollow_flag is None:
            raise RuntimeError(
                f"Execution lease parent cannot be opened safely: {parent}"
            )
        flags = os.O_RDONLY | directory_flag | nofollow_flag
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            return os.open(parent, flags)
        except OSError as error:
            raise RuntimeError(
                f"Execution lease parent cannot be opened safely: {parent}"
            ) from error

    def _lock_parent(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_parent(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
