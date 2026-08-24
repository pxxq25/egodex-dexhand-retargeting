from __future__ import annotations

import unittest

import numpy as np

from egodex_dexhand.inpaint import blend_inpainted_frame, soft_inpaint_alpha
from egodex_dexhand.segment import (
    _arm_prompt_for_frame,
    _prompt_for_frame,
    stabilize_binary_mask_sequence,
    tracked_hand_support_mask,
)


class TemporalMaskStabilizationTests(unittest.TestCase):
    def test_tracked_hand_support_covers_thin_pointing_finger(self) -> None:
        joints = np.zeros((21, 3), dtype=np.float32)
        joints[:, 2] = 1.0
        joints[:, 0] = 30.0
        joints[:, 1] = 30.0
        joints[5:9, 0] = (35.0, 50.0, 65.0, 80.0)

        support = tracked_hand_support_mask(
            joints,
            np.eye(3, dtype=np.float32),
            width=100,
            height=60,
            radius=4,
        )

        self.assertTrue(support[30, 35:81].all())
        self.assertFalse(support[5, 5])

    def test_hand_prompt_falls_back_when_confidence_is_uniformly_low(self) -> None:
        joints = np.zeros((21, 3), dtype=np.float32)
        joints[:, 0] = np.linspace(20.0, 60.0, 21)
        joints[:, 1] = np.linspace(15.0, 45.0, 21)
        joints[:, 2] = 1.0
        confidence = np.full(21, 0.29, dtype=np.float32)

        points, labels, box = _prompt_for_frame(
            joints,
            np.eye(3, dtype=np.float32),
            width=100,
            height=80,
            padding=5,
            joint_confidence=confidence,
        )

        self.assertGreaterEqual(int(np.count_nonzero(labels == 1)), 4)
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue(np.isfinite(box).all())

    def test_arm_prompt_falls_back_when_wrist_is_past_last_pixel_center(self) -> None:
        width, height = 100, 60
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[:, :] = (40.0, 30.0, 1.0)
        arm = np.zeros((4, 3), dtype=np.float32)
        arm[0:2, :] = (20.0, 10.0, 1.0)
        arm[2, :] = (180.0, 159.5, 1.0)
        # This is inside the continuous [0, height) extent but beyond the
        # final valid pixel center at y=59.  It must not become the ray base.
        arm[3, :] = (80.0, 59.5, 1.0)

        points, labels, box = _arm_prompt_for_frame(
            hand,
            arm,
            np.eye(3, dtype=np.float32),
            width,
            height,
            np.empty((0, 2), dtype=np.float32),
        )

        positive = points[labels == 1]
        self.assertTrue(np.isfinite(positive).all())
        self.assertTrue((positive[:, 0] >= 0).all())
        self.assertTrue((positive[:, 0] <= width - 1).all())
        self.assertTrue((positive[:, 1] >= 0).all())
        self.assertTrue((positive[:, 1] <= height - 1).all())
        self.assertTrue(np.isfinite(box).all())

    def test_removes_one_frame_detached_spike(self) -> None:
        masks = []
        for frame_index in range(5):
            mask = np.zeros((64, 64), dtype=bool)
            mask[18:48, 14:42] = True
            if frame_index == 2:
                mask[5:14, 50:59] = True
            masks.append(mask)

        stabilized = stabilize_binary_mask_sequence(masks)

        self.assertEqual(len(stabilized), len(masks))
        self.assertFalse(stabilized[2][5:14, 50:59].any())
        self.assertTrue(stabilized[2][20:46, 16:40].all())
        self.assertGreater(stabilized[2].sum(), 0.9 * masks[1].sum())

    def test_rejects_misaligned_frames(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        with self.assertRaises(ValueError):
            stabilize_binary_mask_sequence(
                [mask, mask], [np.zeros((32, 32), np.uint8)]
            )

    def test_flow_alignment_preserves_translating_object(self) -> None:
        rng = np.random.default_rng(7)
        texture = rng.integers(60, 240, (24, 24), dtype=np.uint8)
        masks = []
        frames = []
        for frame_index in range(5):
            x = 8 + 3 * frame_index
            y = 22
            mask = np.zeros((72, 80), dtype=bool)
            frame = np.zeros((72, 80), dtype=np.uint8)
            mask[y : y + 24, x : x + 24] = True
            frame[y : y + 24, x : x + 24] = texture
            if frame_index == 2:
                mask[4:12, 60:68] = True
            masks.append(mask)
            frames.append(frame)

        stabilized = stabilize_binary_mask_sequence(
            masks, frames, flow_scale=0.75
        )

        for frame_index, stabilized_mask in enumerate(stabilized):
            x = 8 + 3 * frame_index
            moving_core = stabilized_mask[22:46, x : x + 24]
            self.assertGreater(float(moving_core.mean()), 0.98)
        self.assertFalse(stabilized[2][4:12, 60:68].any())


class SoftInpaintSeamTests(unittest.TestCase):
    def test_default_has_six_solid_and_four_soft_pixels(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:30, 20:30] = 255

        alpha = soft_inpaint_alpha(mask)

        self.assertEqual(alpha.dtype, np.float32)
        self.assertEqual(float(alpha[24, 24]), 1.0)
        # The square ends at x=29. Six solid dilation pixels end at x=35.
        self.assertEqual(float(alpha[24, 35]), 1.0)
        self.assertAlmostEqual(float(alpha[24, 36]), 0.8, places=6)
        self.assertAlmostEqual(float(alpha[24, 39]), 0.2, places=6)
        self.assertEqual(float(alpha[24, 40]), 0.0)

    def test_blend_uses_soft_alpha(self) -> None:
        source = np.zeros((2, 3, 3), dtype=np.uint8)
        inpainted = np.full_like(source, 200)
        alpha = np.asarray([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]], np.float32)

        blended = blend_inpainted_frame(source, inpainted, alpha)

        np.testing.assert_array_equal(blended[0, :, 0], [0, 100, 200])
        np.testing.assert_array_equal(blended[..., 0], blended[..., 1])
        np.testing.assert_array_equal(blended[..., 1], blended[..., 2])

    def test_invalid_feather_is_rejected(self) -> None:
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255
        with self.assertRaises(ValueError):
            soft_inpaint_alpha(mask, mask_dilation=4, seam_feather=5)


if __name__ == "__main__":
    unittest.main()
