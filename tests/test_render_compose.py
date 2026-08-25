from __future__ import annotations

import unittest

import numpy as np

from egodex_dexhand.compose import (
    _composite_premultiplied,
    _fallback_premultiplied_matte,
    _frame_visibility_selector,
    _filter_premultiplied,
)
from egodex_dexhand.data import fuse_hand_visibility, projected_hand_visibility
from egodex_dexhand.render import (
    DEFAULT_TEMPORAL_SAMPLES,
    _decontaminate_foreground,
    _finalize_temporal_accumulation,
    _interpolate_rigid_transform,
    _interpolate_sequence,
    _normalize_hidden_arm_visual_links,
    _partition_temporal_masks,
    _temporal_sample_positions,
)


class MotionAwareRenderTests(unittest.TestCase):
    def test_only_configured_arm_visuals_can_be_hidden(self) -> None:
        self.assertEqual(
            _normalize_hidden_arm_visual_links(
                [
                    "wrist_3_link",
                    "forearm",
                    "forearm_link",
                    "upper_arm_link",
                    "shoulder_link",
                    "upper_arm_link",
                ]
            ),
            (
                "forearm",
                "forearm_link",
                "shoulder_link",
                "upper_arm_link",
                "wrist_3_link",
            ),
        )
        with self.assertRaisesRegex(ValueError, "base"):
            _normalize_hidden_arm_visual_links(["base"])

    def test_temporal_samples_are_centered_without_changing_frame_index(self) -> None:
        positions = _temporal_sample_positions(2, 5)

        self.assertEqual(len(positions), DEFAULT_TEMPORAL_SAMPLES)
        self.assertAlmostEqual(float(np.mean(positions)), 2.0)
        self.assertTrue(np.all(np.diff(positions) > 0))
        self.assertGreater(float(positions[0]), 1.5)
        self.assertLess(float(positions[-1]), 2.5)
        np.testing.assert_array_equal(
            _temporal_sample_positions(0, 1), np.asarray([0.0])
        )

    def test_pose_interpolation_preserves_rigid_transform(self) -> None:
        transforms = np.repeat(np.eye(4)[None], 2, axis=0)
        transforms[1, :3, :3] = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        transforms[1, :3, 3] = [2.0, 4.0, 6.0]

        midpoint = _interpolate_rigid_transform(transforms, 0.5)

        np.testing.assert_allclose(midpoint[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            midpoint[:3, :3].T @ midpoint[:3, :3], np.eye(3), atol=1e-7
        )
        self.assertAlmostEqual(float(np.linalg.det(midpoint[:3, :3])), 1.0)
        np.testing.assert_allclose(
            _interpolate_sequence(np.asarray([[0.0], [10.0], [20.0]]), 1.25),
            [12.5],
        )

    def test_temporal_accumulation_keeps_edge_color_unassociated(self) -> None:
        alpha_sum = np.asarray([[1.0, 2.0, 1.0]], dtype=np.float32)
        premultiplied_sum = np.zeros((1, 3, 3), dtype=np.float32)
        premultiplied_sum[..., 2] = alpha_sum

        straight, premultiplied, alpha = _finalize_temporal_accumulation(
            premultiplied_sum, alpha_sum, 2
        )

        np.testing.assert_allclose(alpha, [[0.5, 1.0, 0.5]])
        np.testing.assert_allclose(premultiplied[..., 2], alpha)
        np.testing.assert_allclose(straight[..., 2], 1.0)
        np.testing.assert_allclose(straight[..., :2], 0.0)

    def test_temporal_component_masks_remain_an_exact_disjoint_partition(self) -> None:
        partition = _partition_temporal_masks(
            {
                "left_arm": np.asarray([[2, 1, 0, 0]], dtype=np.uint16),
                "left_hand": np.asarray([[0, 2, 1, 0]], dtype=np.uint16),
                "right_arm": np.asarray([[0, 0, 2, 1]], dtype=np.uint16),
                "right_hand": np.asarray([[0, 0, 0, 2]], dtype=np.uint16),
            }
        )

        stack = np.stack(list(partition.values()), axis=0)
        np.testing.assert_array_equal(np.sum(stack, axis=0), [[1, 1, 1, 1]])
        self.assertTrue(partition["left_arm"][0, 0])
        self.assertTrue(partition["left_hand"][0, 1])
        self.assertTrue(partition["right_arm"][0, 2])
        self.assertTrue(partition["right_hand"][0, 3])


class HaloFreeCompositeTests(unittest.TestCase):
    def test_decontamination_replaces_gray_boundary_before_premultiplication(self) -> None:
        mask = np.zeros((15, 15), dtype=bool)
        mask[3:12, 3:12] = True
        rgb = np.zeros((15, 15, 3), dtype=np.float32)
        rgb[mask] = [0.4, 0.4, 0.4]
        rgb[5:10, 5:10] = [1.0, 0.0, 0.0]

        clean = _decontaminate_foreground(rgb, mask, radius=2)

        np.testing.assert_allclose(clean[3, 7], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(clean[7, 3], [1.0, 0.0, 0.0], atol=1e-6)
        self.assertFalse(clean[~mask].any())

    def test_decontamination_seeds_a_component_thinner_than_erosion_band(self) -> None:
        mask = np.zeros((9, 15), dtype=bool)
        mask[4, 3:12] = True
        rgb = np.full((9, 15, 3), 0.4, dtype=np.float32)
        rgb[4, 7] = [1.0, 0.0, 0.0]

        clean = _decontaminate_foreground(rgb, mask, radius=2)

        np.testing.assert_allclose(
            clean[mask], np.broadcast_to([1.0, 0.0, 0.0], (mask.sum(), 3))
        )
        self.assertFalse(clean[~mask].any())

    def test_gaussian_feather_filters_premultiplied_color_and_alpha_together(
        self,
    ) -> None:
        alpha = np.zeros((31, 31), dtype=np.float32)
        alpha[10:21, 10:21] = 1.0
        premultiplied = np.zeros((31, 31, 3), dtype=np.float32)
        premultiplied[..., 2] = alpha

        filtered_color, filtered_alpha = _filter_premultiplied(
            premultiplied, alpha, sigma=1.5
        )

        edge = (filtered_alpha > 1e-3) & (filtered_alpha < 0.999)
        unassociated_red = filtered_color[..., 2][edge] / filtered_alpha[edge]
        np.testing.assert_allclose(unassociated_red, 1.0, atol=2e-6)
        np.testing.assert_allclose(filtered_color[..., :2][edge], 0.0, atol=1e-7)

    def test_legacy_gray_clear_color_does_not_create_a_halo(self) -> None:
        mask = np.zeros((21, 21), dtype=bool)
        mask[5:16, 5:16] = True
        legacy_rgb = np.full((21, 21, 3), 128, dtype=np.uint8)
        legacy_rgb[7:14, 7:14] = [0, 255, 0]
        premultiplied, alpha = _fallback_premultiplied_matte(legacy_rgb, mask)
        premultiplied, alpha = _filter_premultiplied(
            premultiplied, alpha, sigma=1.2
        )

        edge = (alpha > 1e-3) & (alpha < 0.999)
        unassociated = premultiplied[edge] / alpha[edge][:, None]
        np.testing.assert_allclose(unassociated[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(unassociated[:, 1], 1.0, atol=1e-6)
        np.testing.assert_allclose(unassociated[:, 2], 0.0, atol=1e-6)

        background = np.zeros_like(legacy_rgb)
        composite = _composite_premultiplied(background, premultiplied, alpha)
        transition = composite[edge]
        self.assertTrue(np.all(transition[:, 0] == 0))
        self.assertTrue(np.all(transition[:, 2] == 0))


class FrameVisibilityGateTests(unittest.TestCase):
    def test_direct_rgb_detection_recovers_projection_edge_case(self) -> None:
        projected = np.asarray([True, False, False])
        confidence = np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [0.55, 0.55]], dtype=np.float32
        )

        fused = fuse_hand_visibility(projected, confidence)

        np.testing.assert_array_equal(fused, [True, True, False])

    def test_projection_marks_only_landmarks_inside_camera_frustum(self) -> None:
        intrinsic = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        joints = np.asarray(
            [
                [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
                [[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
                [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            ],
            dtype=np.float32,
        )

        visible = projected_hand_visibility(
            joints,
            intrinsic,
            100,
            80,
            minimum_visible_landmarks=1,
            image_margin=0,
        )

        np.testing.assert_array_equal(visible, [True, False, False])

    def test_robot_is_suppressed_when_human_is_not_projected(self) -> None:
        union = np.ones((3, 4), dtype=bool)
        rendered = np.full((3, 4), 255, dtype=np.uint8)

        permitted, use_inpainted, decisions = _frame_visibility_selector(
            0,
            union,
            {"right": np.asarray([False])},
            {"right": [rendered]},
            minimum_robot_pixels=4,
        )

        self.assertFalse(permitted.any())
        self.assertFalse(use_inpainted)
        self.assertEqual(decisions, {"right": (False, True)})

    def test_human_without_robot_uses_inpaint_but_has_no_robot_pixels(self) -> None:
        union = np.ones((3, 4), dtype=bool)
        tiny_render = np.zeros((3, 4), dtype=np.uint8)
        tiny_render[0, 0] = 255

        permitted, use_inpainted, decisions = _frame_visibility_selector(
            0,
            union,
            {"left": np.asarray([True])},
            {"left": [tiny_render]},
            minimum_robot_pixels=4,
        )

        self.assertFalse(permitted.any())
        self.assertTrue(use_inpainted)
        self.assertEqual(decisions, {"left": (True, False)})

    def test_bimanual_gate_keeps_only_visible_side(self) -> None:
        union = np.ones((2, 4), dtype=bool)
        left = np.zeros((2, 4), dtype=np.uint8)
        right = np.zeros((2, 4), dtype=np.uint8)
        left[:, :2] = 255
        right[:, 2:] = 255

        permitted, use_inpainted, _ = _frame_visibility_selector(
            0,
            union,
            {"left": np.asarray([True]), "right": np.asarray([False])},
            {"left": [left], "right": [right]},
            minimum_robot_pixels=2,
        )

        np.testing.assert_array_equal(
            permitted,
            [[True, True, False, False], [True, True, False, False]],
        )
        self.assertTrue(use_inpainted)


if __name__ == "__main__":
    unittest.main()
