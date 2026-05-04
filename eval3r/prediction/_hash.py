"""sha256 helpers for prediction artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from eval3r.utils.typing import PathLike


def sha256_file(path: PathLike, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def relpath(target: PathLike, root: PathLike) -> str:
    return str(Path(target).relative_to(Path(root)).as_posix())
