"""Exclusive per-run-directory lock (``run_dir/RUNNING.lock``).

``fcntl.flock(fd, LOCK_EX | LOCK_NB)`` on an open file: released automatically by the kernel
when the process exits or dies (no stale-lock cleanup needed; the file itself may remain and is
harmless). A second process on the same run directory fails immediately -> train.py exits with
code 5. On platforms without ``fcntl`` (Windows) locking is a no-op (single-user dev machines
only; the pods run Linux). Note: flock is advisory and may not work on some network file
systems (NFS without lock support).
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # POSIX only
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

LOCK_FILE = "RUNNING.lock"


class RunLockError(RuntimeError):
    """Another process holds the run-directory lock."""


class RunDirLock:
    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / LOCK_FILE
        self._fh = None

    @property
    def supported(self) -> bool:
        return fcntl is not None

    def acquire(self) -> None:
        if fcntl is None:
            return
        fh = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise RunLockError(
                f"{self.path} is locked by another running process; refusing to run the same "
                f"run directory twice."
            ) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()}\n")
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


__all__ = ["RunDirLock", "RunLockError", "LOCK_FILE"]
