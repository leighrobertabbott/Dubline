from __future__ import annotations

"""Cross-process execution lease for a single dubbing job."""

import os
from pathlib import Path
from typing import BinaryIO


def _try_lock(handle: BinaryIO) -> bool:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def acquire_job_lock(folder: Path) -> BinaryIO | None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ".pipeline.lock"
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0"); handle.flush()
    if not _try_lock(handle):
        handle.close()
        return None
    return handle


def release_job_lock(handle: BinaryIO | None) -> None:
    if handle is None:
        return
    try:
        _unlock(handle)
    finally:
        handle.close()


def job_is_running(folder: Path) -> bool:
    """Return True only when another live process holds the job lease."""
    handle = acquire_job_lock(folder)
    if handle is None:
        return True
    release_job_lock(handle)
    return False
