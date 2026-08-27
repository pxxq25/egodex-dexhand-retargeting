from __future__ import annotations

import unittest

import cv2
import numpy as np
from unittest.mock import Mock, patch

from egodex_dexhand.inpaint import (
    blend_inpainted_frame,
    propainter_environment,
    soft_inpaint_alpha,
)


def test_propainter_uses_gpu_with_most_free_memory(monkeypatch) -> None:
    monkeypatch.delenv("EGODEX_INPAINT_CUDA_VISIBLE_DEVICES", raising=False)
    probe = Mock(stdout="0, 12000\n1, 70000\n2, 30000\n")
    with patch("egodex_dexhand.inpaint.subprocess.run", return_value=probe):
        environment = propainter_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
from egodex_dexhand.segment import (
    _signed_mask_distance,
    _arm_prompt_for_frame,
    _prompt_for_frame,
    adaptive_arm_hand_envelopes,
    appearance_refine_arm_hand_masks,
    arm_hand_geometry_envelope,
    stabilize_binary_mask_sequence,
    tracked_hand_support_mask,
)


class TemporalMaskStabilizationTests(unittest.TestCase):
    def test_uniform_warped_masks_have_clipped_signed_distance(self) -> None:
        empty = _signed_mask_distance(np.zeros((12, 16), dtype=bool), 7.5)
        full = _signed_mask_distance(np.ones((12, 16), dtype=bool), 7.5)

        np.testing.assert_array_equal(empty, np.full((12, 16), -7.5))
        np.testing.assert_array_equal(full, np.full((12, 16), 7.5))
        self.assertEqual(empty.dtype, np.float32)
        self.assertEqual(full.dtype, np.float32)

    def test_appearance_refinement_rejects_differently_colored_leak(self) -> None:
        width, height, count = 180, 110, 5
        hand_sequence = np.zeros((count, 21, 3), dtype=np.float32)
        arm_sequence = np.zeros((count, 4, 3), dtype=np.float32)
        frames = []
        masks = []
        for frame_index in range(count):
            wrist = np.asarray([95.0, 55.0, 1.0], np.float32)
            hand_sequence[frame_index, :, :] = wrist
            hand_sequence[frame_index, :, 0] += np.linspace(-10.0, 10.0, 21)
            arm_sequence[frame_index, :, 2] = 1.0
            arm_sequence[frame_index, 2, :2] = (20.0, 80.0)
            arm_sequence[frame_index, 3] = wrist
            frame = np.full((height, width, 3), (70, 145, 190), np.uint8)
            cv2.line(frame, (0, 80), (95, 55), (25, 25, 25), 42)
            cv2.circle(frame, (95, 55), 22, (120, 155, 195), -1)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.line(mask, (0, 80), (95, 55), 1, 42)
            cv2.circle(mask, (95, 55), 22, 1, -1)
            if frame_index == count - 1:
                mask[5:38, 5:80] = 1
            frames.append(frame)
            masks.append(mask > 0)

        refined, _, _, _ = appearance_refine_arm_hand_masks(
            masks,
            frames,
            hand_sequence,
            arm_sequence,
            np.eye(3, dtype=np.float32),
            width,
            height,
        )

        self.assertTrue(refined[-1][80, 10])
        self.assertTrue(refined[-1][55, 95])
        self.assertFalse(refined[-1][10:30, 10:70].any())

    def test_adaptive_envelope_learns_reference_sleeve_width(self) -> None:
        width, height, count = 180, 110, 5
        hand_sequence = np.zeros((count, 21, 3), dtype=np.float32)
        arm_sequence = np.zeros((count, 4, 3), dtype=np.float32)
        masks = []
        for frame_index in range(count):
            wrist = np.asarray(
                [95.0 + frame_index, 35.0 + 5.0 * frame_index, 1.0], np.float32
            )
            hand_sequence[frame_index, :, :] = wrist
            hand_sequence[frame_index, :, 0] += np.linspace(-10.0, 10.0, 21)
            arm_sequence[frame_index, :, 2] = 1.0
            arm_sequence[frame_index, 2, :2] = (20.0, 80.0)
            arm_sequence[frame_index, 3] = wrist
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.line(mask, (0, 80), tuple(wrist[:2].astype(int)), 1, 42)
            cv2.circle(mask, tuple(wrist[:2].astype(int)), 22, 1, -1)
            masks.append(mask > 0)

        envelopes, reference, radii = adaptive_arm_hand_envelopes(
            masks,
            hand_sequence,
            arm_sequence,
            np.eye(3, dtype=np.float32),
            width,
            height,
            profile_margin=2.0,
        )

        self.assertIn(reference, range(count))
        self.assertEqual(len(envelopes), count)
        self.assertEqual(len(radii), 12)
        self.assertTrue(
            all(
                envelope[
                    int(hand_sequence[index, 0, 1]),
                    int(hand_sequence[index, 0, 0]),
                ]
                for index, envelope in enumerate(envelopes)
            )
        )
        self.assertFalse(any(envelope[8, 90] for envelope in envelopes))

    def test_arm_geometry_envelope_rejects_lateral_table_region(self) -> None:
        width, height = 160, 100
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[:, :] = (88.0, 54.0, 1.0)
        hand[:, 0] += np.linspace(-12.0, 12.0, 21)
        arm = np.zeros((4, 3), dtype=np.float32)
        arm[:, 2] = 1.0
        arm[2, :2] = (25.0, 95.0)
        arm[3, :2] = (82.0, 58.0)

        envelope = arm_hand_geometry_envelope(
            hand,
            arm,
            np.eye(3, dtype=np.float32),
            width,
            height,
            corridor_radius=18,
        )

        self.assertTrue(envelope[58, 82])
        self.assertTrue(envelope[94, 26])
        self.assertFalse(envelope[12, 28])
        self.assertFalse(envelope[18, 138])

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
    def test_empty_mask_is_a_zero_alpha_noop(self) -> None:
        alpha = soft_inpaint_alpha(np.zeros((16, 20), dtype=np.uint8))

        self.assertEqual(alpha.dtype, np.float32)
        self.assertEqual(alpha.shape, (16, 20))
        self.assertFalse(alpha.any())

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
