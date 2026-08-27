import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts/run_interact_batch.py"
SPEC = importlib.util.spec_from_file_location("run_interact_batch", SCRIPT)
BATCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BATCH)


class EpisodeSelectorTests(unittest.TestCase):
    def test_ranges_and_duplicates(self):
        self.assertEqual(
            BATCH.parse_episode_selector("1,3-5,5"), {"1", "3", "4", "5"}
        )

    def test_rejects_descending_range(self):
        with self.assertRaises(ValueError):
            BATCH.parse_episode_selector("5-3")


class ValidationTests(unittest.TestCase):
    def test_completed_episode_requires_zero_skips_and_all_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            segment = run / "segments/000_right_00000_00005"
            (segment / "final").mkdir(parents=True)
            (segment / "human_mask").mkdir()
            for index in range(5):
                (segment / "human_mask" / f"{index:05d}.png").touch()
            (segment / "final/composite_full.mp4").touch()
            (run / "final").mkdir(exist_ok=True)
            (run / BATCH.FINAL_VIDEO).touch()
            (run / BATCH.SIDE_BY_SIDE_VIDEO).touch()
            (run / BATCH.DIAGNOSTIC_VIDEO).touch()
            (run / "episode.json").write_text(json.dumps({
                "total_frames": 5,
                "chunks": [{"index": 0, "mode": "right", "start": 0, "end": 5}],
            }))
            with mock.patch.object(BATCH, "validate_video", return_value=[]):
                self.assertEqual(
                    BATCH.validate_episode(run, 5, require_diagnostic=True), []
                )
                skipped = run / "skipped_unrenderable/x"
                skipped.mkdir(parents=True)
                (skipped / "skip_reason.json").write_text("{}")
                errors = BATCH.validate_episode(run, 5, require_diagnostic=True)
                self.assertTrue(any("skipped" in error for error in errors))

    def test_sam3_episode_requires_backend_and_segment_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            segment = run / "segments/000_right_00000_00005"
            (segment / "final").mkdir(parents=True)
            (segment / "human_mask").mkdir()
            for index in range(5):
                (segment / "human_mask" / f"{index:05d}.png").touch()
            (segment / "final/composite_full.mp4").touch()
            (segment / "sam3_recompose.json").write_text("{}")
            (run / "final").mkdir(exist_ok=True)
            (run / BATCH.FINAL_VIDEO).touch()
            (run / BATCH.SIDE_BY_SIDE_VIDEO).touch()
            (run / "episode.json").write_text(json.dumps({
                "total_frames": 5,
                "chunks": [{"index": 0, "mode": "right", "start": 0, "end": 5}],
            }))
            (run / "pipeline.json").write_text(json.dumps({
                "backend": BATCH.SAM3_BACKEND,
            }))

            with mock.patch.object(BATCH, "validate_video", return_value=[]):
                self.assertEqual(
                    BATCH.validate_episode(
                        run,
                        5,
                        require_diagnostic=False,
                        mask_backend=BATCH.SAM3_BACKEND,
                    ),
                    [],
                )
                (segment / "sam3_recompose.json").unlink()
                errors = BATCH.validate_episode(
                    run,
                    5,
                    require_diagnostic=False,
                    mask_backend=BATCH.SAM3_BACKEND,
                )
                self.assertTrue(any("SAM3 provenance" in error for error in errors))


class ClaimTests(unittest.TestCase):
    def test_claim_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with BATCH.claim_episode(root, "7", stale_hours=1) as first:
                self.assertTrue(first)
                with BATCH.claim_episode(root, "7", stale_hours=1) as second:
                    self.assertFalse(second)
            self.assertFalse((root / "claims/7.lock").exists())


class GpuSelectionTests(unittest.TestCase):
    def test_filters_busy_gpus_and_honors_limit(self):
        args = mock.Mock(
            gpu_devices=None, minimum_gpu_free_gib=36.0,
            maximum_gpus=2, minimum_gpus=2,
        )
        with mock.patch.object(
            BATCH, "gpu_free_mib", return_value={0: 80 * 1024, 1: 20 * 1024, 2: 70 * 1024}
        ):
            self.assertEqual(BATCH.select_gpus(args), [0, 2])


class CheckpointTests(unittest.TestCase):
    def test_pipeline_fingerprint_includes_persistent_propainter_worker(self):
        source = SCRIPT.read_text()
        self.assertIn('project / "scripts/run_propainter_jobs.py"', source)

    def test_sam3_cache_keeps_only_frozen_direct_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            (source / "checkpoints").mkdir(parents=True)
            (source / "sam3").mkdir()
            (source / "sam3/model_builder.py").write_text("# model\n")
            (source / f"checkpoints/{BATCH.SAM3_CHECKPOINT}").write_bytes(b"direct")
            (destination / "checkpoints").mkdir(parents=True)
            stale = destination / "checkpoints/sam3.1_multiplex.pt"
            stale.write_bytes(b"multiplex")

            BATCH.copy_sam3_runtime(source, destination)

            self.assertEqual(
                (destination / f"checkpoints/{BATCH.SAM3_CHECKPOINT}").read_bytes(),
                b"direct",
            )
            self.assertFalse(stale.exists())

    def test_live_rsync_tolerates_vanished_source_exit(self):
        result = mock.Mock(returncode=24, args=["rsync"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            with mock.patch.object(BATCH.subprocess, "run", return_value=result):
                BATCH.rsync(source, root / "destination", tolerate_vanished=True)
                with self.assertRaises(BATCH.subprocess.CalledProcessError):
                    BATCH.rsync(source, root / "destination")

    def test_long_command_checkpoints_while_running_and_at_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            BATCH.run_logged(
                [sys.executable, "-c", "import time; time.sleep(0.08)"],
                Path(directory) / "run.log",
                {},
                checkpoint=lambda: calls.append("checkpoint"),
                checkpoint_interval_seconds=0.02,
            )

        self.assertGreaterEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
