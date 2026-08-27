from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from egodex_dexhand.provenance import sha256_file


class ProvenanceCacheTests(unittest.TestCase):
    def test_cache_tracks_path_size_and_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "weight.pt"
            cache = root / "cache"
            source.write_bytes(b"first")
            with mock.patch.dict(
                os.environ, {"EGODEX_SHA256_CACHE_DIR": str(cache)}
            ):
                first = sha256_file(source)
                self.assertEqual(first, hashlib.sha256(b"first").hexdigest())
                cache_files = list(cache.glob("*.json"))
                self.assertEqual(len(cache_files), 1)
                self.assertEqual(json.loads(cache_files[0].read_text())["sha256"], first)
                source.write_bytes(b"other")
                os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 1))
                second = sha256_file(source)
                self.assertEqual(second, hashlib.sha256(b"other").hexdigest())
                self.assertNotEqual(first, second)

    def test_corrupt_cache_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "weight.pt"
            cache = root / "cache"
            source.write_bytes(b"model")
            cache.mkdir()
            (cache / (hashlib.sha256(str(source.resolve()).encode()).hexdigest() + ".json")).write_text("broken")
            with mock.patch.dict(
                os.environ, {"EGODEX_SHA256_CACHE_DIR": str(cache)}
            ):
                self.assertEqual(
                    sha256_file(source), hashlib.sha256(b"model").hexdigest()
                )


if __name__ == "__main__":
    unittest.main()
