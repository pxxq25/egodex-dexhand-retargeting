from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/run_egoquest_adaptive_recording.py"
SPEC = importlib.util.spec_from_file_location("adaptive_recording", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
adaptive_recording = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adaptive_recording)


class AdaptiveVisibilityTests(unittest.TestCase):
    def test_visual_resume_preserves_prior_skip(self) -> None:
        with mock.patch.object(adaptive_recording.Path, "exists", return_value=True):
            self.assertTrue(
                adaptive_recording.resume_skipped_visual(
                    "visual", True, Path("skip_reason.json")
                )
            )
        self.assertFalse(
            adaptive_recording.resume_skipped_visual(
                "trajectory", True, Path("skip_reason.json")
            )
        )

    def test_short_interior_visibility_gap_is_bridged(self) -> None:
        visible = np.asarray([False, True, True, False, False, True, False])

        bridged = adaptive_recording.bridge_short_visibility_gaps(
            visible, maximum_gap_frames=2
        )

        np.testing.assert_array_equal(
            bridged, [False, True, True, True, True, True, False]
        )

    def test_visibility_gap_at_edge_is_not_bridged(self) -> None:
        visible = np.asarray([False, False, True, True, False])

        bridged = adaptive_recording.bridge_short_visibility_gaps(
            visible, maximum_gap_frames=2
        )

        np.testing.assert_array_equal(bridged, visible)

    def test_balanced_split_avoids_two_frame_tail(self) -> None:
        ranges = adaptive_recording.balanced_chunk_ranges(
            1057, 1299, min_frames=6, max_frames=240
        )

        self.assertEqual(ranges, [(1057, 1178), (1178, 1299)])
        self.assertTrue(all(6 <= end - start <= 240 for start, end in ranges))

    def test_balanced_split_rejects_inverted_chunk_limits(self) -> None:
        with self.assertRaises(ValueError):
            adaptive_recording.balanced_chunk_ranges(
                0, 20, min_frames=10, max_frames=5
            )

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
        with mock.patch.dict(os.environ, {"EGODEX_RENDER_DEVICE": "cuda:0"}):
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
        self.assertIn("--allow-hidden-arm", command)

    def test_bimanual_keeps_both_shadow_forearms(self) -> None:
        with mock.patch.dict(os.environ, {"EGODEX_RENDER_DEVICE": "cuda:0"}):
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
        self.assertIn("--allow-hidden-arm", command)


class VulkanDeviceTests(unittest.TestCase):
    def test_persistent_render_routes_only_supported_module_commands(self) -> None:
        environment = {"EGODEX_IN_PROCESS_RENDER": "1"}
        self.assertTrue(
            adaptive_recording.in_process_render_command(
                ["python", "-m", "egodex_dexhand.bimanual_cli"],
                environment,
            )
        )
        self.assertFalse(
            adaptive_recording.in_process_render_command(
                ["python", "script.py"], environment
            )
        )

    def test_command_executes_supported_render_module_in_process(self) -> None:
        environment = {"EGODEX_IN_PROCESS_RENDER": "1", "PYTHONPATH": ""}
        arguments = ["python", "-m", "egodex_dexhand.ur5e_shadow_cli", "--x"]
        with mock.patch.object(
            adaptive_recording.runpy, "run_module"
        ) as run_module:
            adaptive_recording.command(arguments, env=environment)

        run_module.assert_called_once_with(
            "egodex_dexhand.ur5e_shadow_cli", run_name="__main__"
        )

    def test_renderer_environment_accepts_usr_share_nvidia_icd(self) -> None:
        with mock.patch.object(
            adaptive_recording.Path, "is_file", side_effect=[False, True]
        ):
            environment = adaptive_recording.renderer_environment(
                Path("/project"), Path("/runtime"), {}
            )

        self.assertEqual(
            environment["VK_ICD_FILENAMES"],
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
        )

    def test_adaptive_cli_rejects_implicit_render_device(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "EGODEX_RENDER_DEVICE"):
                adaptive_recording.common_flags(
                    "python", Path("/runtime"), Path("video.mp4"),
                    Path("input.hdf5"), Path("output"),
                )

    def test_render_device_is_forwarded_to_cli(self) -> None:
        with mock.patch.dict(os.environ, {"EGODEX_RENDER_DEVICE": "cuda:6"}):
            command = adaptive_recording.single_cli(
                "right", "python", Path("/runtime"), Path("video.mp4"),
                Path("input.hdf5"), Path("output"),
            )

        index = command.index("--render-device")
        self.assertEqual(command[index + 1], "cuda:6")
        self.assertIn("--camera-relative-base", command)

    def test_preflight_retries_after_timeout(self) -> None:
        with mock.patch.object(
            adaptive_recording,
            "command",
            side_effect=[subprocess.TimeoutExpired("probe", 1), None],
        ) as run:
            healthy = adaptive_recording.preflight_render_device(
                "python",
                Path("/project"),
                "cuda:3",
                {},
                timeout=1,
                retries=2,
            )

        self.assertTrue(healthy)
        self.assertEqual(run.call_count, 2)

    def test_preflight_exhaustion_marks_device_unhealthy(self) -> None:
        with mock.patch.object(
            adaptive_recording,
            "command",
            side_effect=subprocess.TimeoutExpired("probe", 1),
        ) as run:
            healthy = adaptive_recording.preflight_render_device(
                "python",
                Path("/project"),
                "cuda:1",
                {},
                timeout=1,
                retries=2,
            )

        self.assertFalse(healthy)
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
