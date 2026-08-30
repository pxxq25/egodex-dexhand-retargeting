import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from egodex_dexhand.inpaint import (
    build_robot_aware_removal_masks,
    build_robot_context_frames,
)


class RobotAwareRemovalMaskTest(unittest.TestCase):
    def test_robot_context_hides_foreground_under_robot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / "source"
            rgb_dir = root / "rgb"
            alpha_dir = root / "alpha"
            output_dir = root / "context"
            source_dir.mkdir()
            rgb_dir.mkdir()
            alpha_dir.mkdir()
            source = np.full((20, 30, 3), (0, 120, 240), dtype=np.uint8)
            robot = np.full_like(source, (20, 20, 20))
            alpha = np.zeros(source.shape[:2], dtype=np.uint8)
            alpha[:, 10:20] = 255
            cv2.imwrite(str(source_dir / "00000.jpg"), source)
            cv2.imwrite(str(rgb_dir / "00000.png"), robot)
            cv2.imwrite(str(alpha_dir / "00000.png"), alpha)

            build_robot_context_frames(
                source_dir, rgb_dir, alpha_dir, output_dir
            )
            context = cv2.imread(str(output_dir / "00000.jpg"))
            self.assertIsNotNone(context)
            self.assertLess(int(context[10, 15, 2]), 40)
            self.assertGreater(int(context[10, 2, 2]), 200)

    def test_keeps_opaque_robot_interior_out_of_inpaint_hole(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            human_dir = root / "human"
            alpha_dir = root / "alpha"
            output_dir = root / "output"
            human_dir.mkdir()
            alpha_dir.mkdir()
            human = np.zeros((80, 100), dtype=np.uint8)
            human[20:60, 20:90] = 255
            alpha = np.zeros_like(human)
            alpha[24:56, 25:85] = 255
            cv2.imwrite(str(human_dir / "00000.png"), human)
            cv2.imwrite(str(alpha_dir / "00000.png"), alpha)
            summary = build_robot_aware_removal_masks(
                human_dir, alpha_dir, output_dir, seam_dilation_pixels=0
            )
            result = cv2.imread(
                str(output_dir / "00000.png"), cv2.IMREAD_GRAYSCALE
            )
            self.assertIsNotNone(result)
            self.assertEqual(int(result[40, 50]), 0)
            self.assertEqual(int(result[20, 20]), 255)
            self.assertEqual(summary.human_pixels, 40 * 70)
            self.assertEqual(summary.opaque_robot_covered_human_pixels, 32 * 60)
            self.assertEqual(
                summary.residual_human_pixels,
                summary.human_pixels - summary.opaque_robot_covered_human_pixels,
            )
            self.assertEqual(
                [path.suffix for path in output_dir.iterdir()], [".png"]
            )
            self.assertTrue((root / "output_summary.json").is_file())

    def test_resolution_scaled_seam_is_bounded_by_human_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            human_dir = root / "human"
            alpha_dir = root / "alpha"
            output_dir = root / "output"
            human_dir.mkdir()
            alpha_dir.mkdir()
            human = np.zeros((480, 640), dtype=np.uint8)
            human[200:280, 250:390] = 255
            alpha = np.zeros_like(human)
            alpha[210:270, 260:380] = 255
            cv2.imwrite(str(human_dir / "00000.png"), human)
            cv2.imwrite(str(alpha_dir / "00000.png"), alpha)
            build_robot_aware_removal_masks(human_dir, alpha_dir, output_dir)
            result = cv2.imread(
                str(output_dir / "00000.png"), cv2.IMREAD_GRAYSCALE
            )
            self.assertEqual(int(result[100, 100]), 0)
            self.assertGreater(int(np.count_nonzero(result)), 0)


if __name__ == "__main__":
    unittest.main()
