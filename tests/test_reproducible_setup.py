from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReproducibleSetupTests(unittest.TestCase):
    def test_lock_manifests_are_complete_and_machine_readable(self) -> None:
        environment = json.loads((ROOT / "environment.lock.json").read_text())
        third_party = json.loads((ROOT / "third_party.lock.json").read_text())
        self.assertEqual(environment["schema_version"], 1)
        self.assertEqual(third_party["schema_version"], 1)
        self.assertEqual(
            {entry["name"] for entry in third_party["repositories"]},
            {"dex-retargeting", "sam2", "sam3", "ProPainter"},
        )
        destinations = set()
        for entry in third_party["repositories"]:
            self.assertRegex(entry["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(entry["url"].startswith("https://github.com/"))
            for submodule in entry.get("submodules", []):
                self.assertFalse(Path(submodule["path"]).is_absolute())
                self.assertNotIn("..", Path(submodule["path"]).parts)
                self.assertRegex(submodule["commit"], r"^[0-9a-f]{40}$")
        for entry in third_party["checkpoints"]:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["bytes"], 0)
            self.assertNotIn(entry["destination"], destinations)
            destinations.add(entry["destination"])
            self.assertTrue("url" in entry or "huggingface_repo" in entry)

    def test_every_python_requirement_is_exactly_pinned(self) -> None:
        for relative in (
            "requirements/pytorch-cu126.lock.txt",
            "requirements/runtime-cu126.lock.txt",
            "requirements/mediapipe.lock.txt",
        ):
            lines = (ROOT / relative).read_text().splitlines()
            requirements = [
                line.strip()
                for line in lines
                if line.strip() and not line.lstrip().startswith("#")
            ]
            self.assertTrue(requirements, relative)
            for requirement in requirements:
                self.assertRegex(
                    requirement,
                    r"^[A-Za-z0-9_.-]+==[^=<>~!]+$",
                    f"unlocked requirement in {relative}: {requirement}",
                )

    def test_mediapipe_is_isolated_from_numpy_two(self) -> None:
        primary = (ROOT / "requirements/runtime-cu126.lock.txt").read_text()
        sidecar = (ROOT / "requirements/mediapipe.lock.txt").read_text()
        self.assertIn("numpy==2.2.6", primary)
        self.assertNotIn("mediapipe==", primary)
        self.assertIn("numpy==1.26.4", sidecar)
        self.assertIn("mediapipe==0.10.21", sidecar)

    def test_setup_runtime_dry_run_requires_no_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/setup_runtime.py"),
                    "--runtime",
                    str(runtime),
                    "--project",
                    str(ROOT),
                    "--python",
                    str(runtime / ".venv/bin/python"),
                    "--dry-run",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
        self.assertIn("git init", result.stdout)
        self.assertIn("download_models.py", result.stdout)
        self.assertIn("runtime ready", result.stdout)

    def test_checkpoint_listing_does_not_require_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/download_models.py"),
                    "--third-party-root",
                    directory,
                    "--list",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
        listed = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(listed), 5)
        self.assertEqual(sum(bool(entry["gated"]) for entry in listed), 1)

    def test_shell_entrypoints_parse(self) -> None:
        for relative in ("bootstrap.sh", "activate.sh", "docker/entrypoint.sh"):
            subprocess.run(["bash", "-n", str(ROOT / relative)], check=True)

    def test_no_huggingface_token_is_committed(self) -> None:
        token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
        for relative in (
            "bootstrap.sh",
            "Dockerfile",
            "README.md",
            "scripts/download_models.py",
        ):
            self.assertIsNone(token_pattern.search((ROOT / relative).read_text()))


if __name__ == "__main__":
    unittest.main()
