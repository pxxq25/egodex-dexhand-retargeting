import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from egodex_dexhand.screen_registration import (
    fit_wrist_locked_similarity_sequence,
)


class WristLockedSimilarityTest(unittest.TestCase):
    def test_recovers_scale_rotation_and_exact_wrist(self) -> None:
        robot = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
                [0.1, 0.1, 0.05],
            ]
        )
        robot = np.stack([robot, robot + [0.03, -0.02, 0.01]])
        rotations = Rotation.from_euler(
            "xyz", [[10, -20, 35], [-15, 5, 70]], degrees=True
        ).as_matrix()
        translations = np.asarray([[0.2, -0.3, 0.7], [-0.1, 0.4, 0.6]])
        observed = (
            0.73 * np.einsum("tij,tkj->tki", rotations, robot)
            + translations[:, None]
        )
        result = fit_wrist_locked_similarity_sequence(
            robot, observed, np.asarray([0, 2, 2, 2, 1], dtype=np.float64)
        )
        np.testing.assert_allclose(result.transform(robot), observed, atol=1e-6)
        np.testing.assert_allclose(result.determinant, 1.0, atol=1e-6)
        np.testing.assert_allclose(
            result.transform(robot)[:, 0], observed[:, 0], atol=1e-7
        )

    def test_ignores_zero_confidence_outlier(self) -> None:
        robot = np.asarray(
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]],
            dtype=np.float64,
        )
        observed = 0.5 * robot + np.asarray([[[0.2, -0.1, 0.8]]])
        observed[0, 4] = [99, 99, 99]
        result = fit_wrist_locked_similarity_sequence(
            robot,
            observed,
            np.asarray([0, 1, 1, 1, 1], dtype=np.float64),
            confidence=np.asarray([[1, 1, 1, 1, 0]], dtype=np.float64),
        )
        np.testing.assert_allclose(result.transform(robot)[0, 1:4], observed[0, 1:4])
        self.assertEqual(int(result.effective_landmarks[0]), 3)

    def test_never_returns_reflection(self) -> None:
        robot = np.asarray(
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            dtype=np.float64,
        )
        observed = robot.copy()
        observed[..., 0] *= -1
        result = fit_wrist_locked_similarity_sequence(
            robot, observed, np.asarray([0, 1, 1, 1], dtype=np.float64)
        )
        self.assertGreater(float(result.determinant[0]), 0.999999)

    def test_rejects_underconstrained_frame(self) -> None:
        robot = np.zeros((1, 4, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "fewer than three"):
            fit_wrist_locked_similarity_sequence(
                robot,
                robot,
                np.ones(4),
                confidence=np.asarray([[1, 1, 0, 0]], dtype=np.float64),
            )


if __name__ == "__main__":
    unittest.main()
