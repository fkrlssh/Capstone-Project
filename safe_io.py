"""
Cross-process safe file I/O for background training + concurrent metric export.

- Exclusive lock files (*.lock) prevent torn reads/writes on Windows/Linux.
- atomic_write_json / atomic_write_text replace targets atomically via os.replace.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, TypeVar

T = TypeVar("T")

DEFAULT_LOCK_TIMEOUT_SEC = 30.0
DEFAULT_LOCK_POLL_SEC = 0.05


class FileLockTimeout(RuntimeError):
    pass


@contextmanager
def file_lock(
    resource_path: str,
    timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC,
    poll_sec: float = DEFAULT_LOCK_POLL_SEC,
) -> Iterator[None]:
    """Exclusive lock via O_EXCL lock file next to the resource."""
    lock_path = f"{os.path.abspath(resource_path)}.lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    deadline = time.time() + max(0.1, float(timeout_sec))
    fd: Optional[int] = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii", errors="replace"))
            break
        except FileExistsError:
            if time.time() >= deadline:
                raise FileLockTimeout(f"Could not acquire lock: {lock_path}")
            time.sleep(poll_sec)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(lock_path)
        except OSError:
            pass


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding=encoding, newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, data: Any, indent: int = 2) -> None:
    text = json.dumps(data, indent=indent, ensure_ascii=False)
    atomic_write_text(path, text + "\n")


def read_json_locked(path: str, timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC) -> Any:
    if not os.path.exists(path):
        return None
    with file_lock(path, timeout_sec=timeout_sec):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def with_locked_resource(
    path: str,
    fn: Callable[[], T],
    timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC,
) -> T:
    with file_lock(path, timeout_sec=timeout_sec):
        return fn()
