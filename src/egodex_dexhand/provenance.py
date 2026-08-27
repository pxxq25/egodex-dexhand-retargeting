from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file, optionally using a process-safe persistent stat cache.

    Immutable model weights were previously re-read for every adaptive chunk.
    ``EGODEX_SHA256_CACHE_DIR`` makes those provenance checks constant-time
    after the first read on a host. The cache key includes the resolved path,
    size, and nanosecond mtime; atomic replacement makes concurrent readers
    safe without serializing the compute path.
    """

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    cache_root_value = os.environ.get("EGODEX_SHA256_CACHE_DIR")
    if not cache_root_value:
        return _hash_file(source)
    stat = source.stat()
    identity = {
        "path": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    cache_root = Path(cache_root_value)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_name = hashlib.sha256(str(source).encode()).hexdigest() + ".json"
    cache = cache_root / cache_name
    try:
        value = json.loads(cache.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        value = None
    if isinstance(value, dict) and all(
        value.get(key) == expected for key, expected in identity.items()
    ):
        digest = value.get("sha256")
        if isinstance(digest, str) and len(digest) == 64:
            return digest
    digest = _hash_file(source)
    temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({**identity, "sha256": digest}) + "\n")
    os.replace(temporary, cache)
    return digest
