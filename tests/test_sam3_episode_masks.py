import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts/run_sam3_episode_masks.py"
SPEC = importlib.util.spec_from_file_location("run_sam3_episode_masks", SCRIPT)
MASKS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MASKS)


class CheckpointCompatibilityTests(unittest.TestCase):
    def validate(self, state):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            fake_torch = SimpleNamespace(load=mock.Mock(return_value=state))
            with mock.patch.dict("sys.modules", {"torch": fake_torch}):
                MASKS.validate_direct_video_checkpoint(checkpoint)

    def test_accepts_direct_video_checkpoint_layout(self):
        self.validate({
            "detector.backbone.vision_backbone.trunk.pos_embed": object(),
            "tracker.maskmem_tpos_enc": object(),
        })

    def test_rejects_multiplex_checkpoint_layout(self):
        with self.assertRaisesRegex(RuntimeError, "not sam3.1_multiplex.pt"):
            self.validate({
                "detector.backbone.vision_backbone.trunk.pos_embed": object(),
                "tracker.model.maskmem_tpos_enc": object(),
            })


if __name__ == "__main__":
    unittest.main()
