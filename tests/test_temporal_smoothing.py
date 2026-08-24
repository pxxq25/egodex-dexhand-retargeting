from __future__ import annotations

import argparse
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from egodex_dexhand.temporal_smoothing import (
    SHADOW_ROOT_JOINT_NAMES,
    TemporalSmoothingConfig,
    add_temporal_smoothing_arguments,
    smooth_se3_trajectory,
    smooth_shadow_qpos,
    temporal_smoothing_config_from_args,
)


ACTUATED_NAMES = tuple(f"shadow_joint_{index}" for index in range(24))
JOINT_NAMES = SHADOW_ROOT_JOINT_NAMES + ACTUATED_NAMES


class TemporalSmoothingTests(unittest.TestCase):
    def test_cli_defaults_enable_documented_filter(self) -> None:
        parser = argparse.ArgumentParser()
        add_temporal_smoothing_arguments(parser)
        config = temporal_smoothing_config_from_args(parser.parse_args([]))
        self.assertTrue(config.enabled)
        self.assertEqual(config.window_size, 7)
        self.assertEqual(config.filter_passes, 1)
        self.assertEqual(config.hand_max_velocity, 6.0)
        self.assertEqual(config.hand_max_acceleration, 60.0)
        disabled = temporal_smoothing_config_from_args(
            parser.parse_args(["--no-temporal-smoothing"])
        )
        self.assertFalse(disabled.enabled)

    def test_shadow_filter_preserves_root_limits_and_derivative_bounds(self) -> None:
        fps = 30.0
        count = 61
        time = np.arange(count) / fps
        root = np.column_stack(
            [0.1 * np.sin(time * (index + 1)) for index in range(6)]
        )
        hand = np.column_stack(
            [
                0.25 * np.sin(2.0 * np.pi * (0.4 + index / 100.0) * time)
                for index in range(24)
            ]
        )
        hand[30, 3] = 1.0
        qpos = np.column_stack([root, hand]).astype(np.float32)
        limits = np.tile(np.asarray([-0.8, 0.8], dtype=np.float32), (30, 1))
        limits[:3] = (-5.0, 5.0)
        limits[3:6] = (-2.0 * np.pi, 2.0 * np.pi)
        config = TemporalSmoothingConfig(
            hand_max_velocity=2.0,
            hand_max_acceleration=20.0,
        )

        result = smooth_shadow_qpos(qpos, JOINT_NAMES, limits, fps, config)

        np.testing.assert_array_equal(result[:, :6], qpos[:, :6])
        self.assertLessEqual(float(result[:, 6:].max()), 0.8 + 1e-6)
        self.assertGreaterEqual(float(result[:, 6:].min()), -0.8 - 1e-6)
        maximum_velocity = float(np.max(np.abs(np.diff(result[:, 6:], axis=0))) * fps)
        maximum_acceleration = float(
            np.max(np.abs(np.diff(result[:, 6:], n=2, axis=0))) * fps**2
        )
        self.assertLessEqual(maximum_velocity, 2.0 + 2e-5)
        self.assertLessEqual(maximum_acceleration, 20.0 + 2e-4)
        self.assertLess(abs(float(result[30, 9])), abs(float(qpos[30, 9])))

    def test_centered_hand_filter_does_not_introduce_directional_lag(self) -> None:
        count = 41
        qpos = np.zeros((count, 30), dtype=np.float64)
        # A symmetric feature must remain symmetric around the same center.
        qpos[17:24, 6:] = np.asarray([1, 2, 3, 4, 3, 2, 1])[:, None] * 0.05
        limits = np.tile(np.asarray([-1.0, 1.0]), (30, 1))
        config = TemporalSmoothingConfig(
            hand_max_velocity=100.0,
            hand_max_acceleration=10000.0,
        )

        result = smooth_shadow_qpos(qpos, JOINT_NAMES, limits, 30.0, config)

        np.testing.assert_allclose(
            result[17:24, 6], result[17:24, 6][::-1], atol=1e-12
        )
        self.assertEqual(int(np.argmax(result[:, 6])), 20)

    def test_se3_filter_is_valid_robust_and_derivative_limited(self) -> None:
        fps = 30.0
        count = 51
        time = np.arange(count) / fps
        positions = np.column_stack(
            [0.2 * time, 0.01 * np.sin(3.0 * time), np.zeros(count)]
        )
        positions[0] += np.asarray([0.8, -0.5, 0.3])
        angles = 0.3 * time
        angles[25] += 1.4
        rotations = Rotation.from_euler("z", angles).as_matrix()
        config = TemporalSmoothingConfig(
            forearm_max_linear_velocity=0.5,
            forearm_max_linear_acceleration=4.0,
            forearm_max_angular_velocity=1.0,
        )

        xyz, matrices = smooth_se3_trajectory(
            positions.astype(np.float32),
            rotations.astype(np.float32),
            fps,
            config,
        )

        np.testing.assert_allclose(
            matrices.transpose(0, 2, 1) @ matrices,
            np.broadcast_to(np.eye(3), matrices.shape),
            atol=2e-6,
        )
        np.testing.assert_allclose(np.linalg.det(matrices), 1.0, atol=2e-6)
        linear_velocity = np.max(np.abs(np.diff(xyz, axis=0))) * fps
        linear_acceleration = np.max(np.abs(np.diff(xyz, n=2, axis=0))) * fps**2
        self.assertLessEqual(float(linear_velocity), 0.5 + 2e-5)
        self.assertLessEqual(float(linear_acceleration), 4.0 + 2e-4)
        angular_steps = (
            Rotation.from_matrix(matrices[:-1]).inv()
            * Rotation.from_matrix(matrices[1:])
        ).magnitude()
        self.assertLessEqual(float(angular_steps.max() * fps), 1.0 + 2e-5)
        self.assertLess(np.linalg.norm(xyz[0] - xyz[1]), 0.03)
        self.assertLess(angular_steps[24], 0.04)

    def test_shadow_shape_validation_rejects_non_shadow_trajectory(self) -> None:
        qpos = np.zeros((3, 29))
        limits = np.tile(np.asarray([-1.0, 1.0]), (29, 1))
        with self.assertRaisesRegex(ValueError, "24 actuated"):
            smooth_shadow_qpos(
                qpos,
                SHADOW_ROOT_JOINT_NAMES + ACTUATED_NAMES[:-1],
                limits,
                30.0,
                TemporalSmoothingConfig(),
            )


if __name__ == "__main__":
    unittest.main()
