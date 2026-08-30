import unittest

import cv2
import numpy as np

from egodex_dexhand.visual_forearm import (
    estimate_forearm_observation_sequence,
    estimate_forearm_silhouette,
    refine_human_silhouette,
)


class ForearmSilhouetteTest(unittest.TestCase):
    def test_refines_missing_sleeve_side_from_appearance(self) -> None:
        frame = np.full((180, 260, 3), (90, 90, 90), dtype=np.uint8)
        cv2.line(frame, (80, 70), (245, 125), (20, 125, 230), 58)
        cv2.circle(frame, (80, 70), 30, (20, 125, 230), -1)
        complete = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.line(complete, (80, 70), (245, 125), 58, 255)
        cv2.circle(complete, (80, 70), 30, 255, -1)
        incomplete = complete.copy()
        incomplete[:80] = 0
        cv2.circle(incomplete, (80, 70), 16, 255, -1)

        refined, growth, rejected = refine_human_silhouette(
            frame, incomplete, np.asarray([80.0, 70.0]), 48.0
        )
        recovered = np.count_nonzero((refined > 0) & (complete > 0))
        self.assertFalse(rejected)
        self.assertGreater(growth, 1.1)
        self.assertGreater(recovered, 1.1 * np.count_nonzero(incomplete))

        bounded, bounded_growth, bounded_rejected = refine_human_silhouette(
            frame,
            incomplete,
            np.asarray([80.0, 70.0]),
            48.0,
            max_area_growth=1.01,
        )
        self.assertTrue(bounded_rejected)
        self.assertEqual(bounded_growth, 1.0)
        self.assertEqual(
            int(np.count_nonzero(bounded)), int(np.count_nonzero(incomplete))
        )

    def test_selects_wrist_component_and_measures_axis(self) -> None:
        mask = np.zeros((240, 320), dtype=np.uint8)
        wrist = np.asarray([110.0, 95.0])
        cv2.line(mask, (110, 95), (292, 170), 34, 255)
        cv2.circle(mask, (110, 95), 30, 255, -1)
        cv2.rectangle(mask, (10, 10), (65, 65), 255, -1)  # distractor
        direction, length, width, confidence = estimate_forearm_silhouette(
            mask, wrist
        )
        expected = np.asarray([182.0, 75.0])
        expected /= np.linalg.norm(expected)
        angle = np.degrees(np.arccos(np.clip(np.dot(direction, expected), -1, 1)))
        self.assertLess(angle, 5.0)
        self.assertGreater(length, 160)
        self.assertGreater(width, 25)
        self.assertGreater(confidence, 0.25)

    def test_selects_visible_component_for_offscreen_wrist(self) -> None:
        mask = np.zeros((120, 160), dtype=np.uint8)
        cv2.line(mask, (4, 112), (100, 60), 20, 255)
        cv2.circle(mask, (4, 112), 14, 255, -1)
        direction, length, width, _ = estimate_forearm_silhouette(
            mask, np.asarray([-18.0, 132.0])
        )
        self.assertGreater(direction[0], 0.0)
        self.assertLess(direction[1], 0.0)
        self.assertGreater(length, 80.0)
        self.assertGreater(width, 10.0)

    def test_sequence_backprojects_without_scene_assumptions(self) -> None:
        masks = []
        for offset in range(5):
            mask = np.zeros((120, 160), dtype=np.uint8)
            cv2.line(mask, (50, 40), (145, 80 + offset), 18, 255)
            cv2.circle(mask, (50, 40), 16, 255, -1)
            masks.append(mask)
        wrist_camera = np.repeat([[0.0, 0.0, 0.5]], 5, axis=0)
        wrist_pixels = np.repeat([[50.0, 40.0]], 5, axis=0)
        intrinsic = np.asarray([[200, 0, 50], [0, 200, 40], [0, 0, 1]])
        result = estimate_forearm_observation_sequence(
            masks,
            wrist_camera,
            wrist_pixels,
            intrinsic,
            palm_width_pixels=np.full(5, 20.0),
            temporal_window=3,
        )
        self.assertEqual(result.guide_camera.shape, (5, 3))
        self.assertTrue(np.all(result.length_camera > 0))
        self.assertTrue(np.all(result.width_camera > 0))
        np.testing.assert_allclose(result.guide_camera[:, 2], 0.5)


if __name__ == "__main__":
    unittest.main()
