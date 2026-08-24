from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/run_egoquest_adaptive_recording.py"
SPEC = importlib.util.spec_from_file_location("adaptive_recording", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
adaptive_recording = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adaptive_recording)


class AdaptiveVisibilityTests(unittest.TestCase):
    def test_entry_and_exit_padding_expand_interval_symmetrically_in_time(self) -> None:
        visible = np.zeros(12, dtype=bool)
        visible[5:7] = True

        expanded = adaptive_recording.expand_visibility_intervals(
            visible,
            entry_padding_frames=3,
            exit_padding_frames=2,
        )

        expected = np.zeros_like(visible)
        expected[2:9] = True
        np.testing.assert_array_equal(expanded, expected)

    def test_visibility_padding_clamps_at_video_boundaries(self) -> None:
        visible = np.asarray([True, False, False, False, True], dtype=bool)

        expanded = adaptive_recording.expand_visibility_intervals(
            visible,
            entry_padding_frames=2,
            exit_padding_frames=1,
        )

        np.testing.assert_array_equal(expanded, [True, True, True, True, True])

    def test_visibility_padding_rejects_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            adaptive_recording.expand_visibility_intervals(
                np.ones(3, dtype=bool),
                entry_padding_frames=-1,
                exit_padding_frames=0,
            )


class AdaptiveArmVisibilityTests(unittest.TestCase):
    def test_single_hand_keeps_shadow_forearm_but_hides_proximal_ur5e(self) -> None:
        command = adaptive_recording.single_cli(
            "right", "python", Path("/runtime"), Path("video.mp4"),
            Path("input.hdf5"), Path("output"),
        )
        hidden = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--hide-arm-visual-link"
        ]

        self.assertIn("forearm_link", hidden)
        self.assertNotIn("forearm", hidden)

    def test_bimanual_keeps_both_shadow_forearms(self) -> None:
        command = adaptive_recording.bimanual_cli(
            "python", Path("/runtime"), Path("video.mp4"),
            Path("input.hdf5"), Path("output"),
        )

        for side in ("left", "right"):
            flag = f"--{side}-hide-arm-visual-link"
            hidden = [
                command[index + 1]
                for index, value in enumerate(command[:-1])
                if value == flag
            ]
            self.assertIn("forearm_link", hidden)
            self.assertNotIn("forearm", hidden)


if __name__ == "__main__":
    unittest.main()
