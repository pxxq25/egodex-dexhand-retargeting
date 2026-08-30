import tempfile
import unittest
from pathlib import Path

import numpy as np

from egodex_dexhand.screen_render import (
    _directed_segment_image_length,
    _mesh_extents,
    _point_image_distance,
    detached_forearm_transform_camera,
)


class RobotMeshBoundsTest(unittest.TestCase):
    def test_measures_distance_for_offscreen_padding(self) -> None:
        self.assertEqual(_point_image_distance([20, 30], 100, 80), 0.0)
        self.assertAlmostEqual(_point_image_distance([-3, 30], 100, 80), 3.0)
        self.assertAlmostEqual(
            _point_image_distance([102, 83], 100, 80), 5.0
        )

    def test_reads_obj_bounds_without_an_extra_mesh_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mesh.obj"
            path.write_text(
                "v -1 -2 -3\n"
                "v 4 5 6\n"
                "v 0 1 2\n"
                "f 1 2 3\n"
            )
            np.testing.assert_allclose(_mesh_extents(path), [5, 7, 9])

    def test_directed_visibility_rejects_an_outward_offscreen_axis(self) -> None:
        self.assertEqual(
            _directed_segment_image_length(
                [-5, 30], [-1, 0], 40, width=100, height=80
            ),
            0.0,
        )
        self.assertAlmostEqual(
            _directed_segment_image_length(
                [-5, 30], [1, 0], 40, width=100, height=80
            ),
            35.0,
        )
        self.assertAlmostEqual(
            _directed_segment_image_length(
                [70, 30], [1, 0], 60, width=100, height=80
            ),
            29.0,
        )

    def test_anisotropic_scale_does_not_bend_the_forearm_axis(self) -> None:
        wrist = np.asarray([0.2, -0.1, 1.2])
        guide = wrist + np.asarray([0.3, 0.4, 0.0])
        wrist_local = np.asarray([0.0, -0.01, 0.21301])
        scale = np.asarray([0.8, 0.25, 0.65])
        rotation, origin = detached_forearm_transform_camera(
            wrist,
            guide,
            np.eye(3),
            wrist_local,
            scale,
        )
        mapped_wrist = rotation @ (scale * wrist_local)
        np.testing.assert_allclose(origin + mapped_wrist, wrist, atol=1e-10)
        expected = wrist - guide
        np.testing.assert_allclose(
            mapped_wrist / np.linalg.norm(mapped_wrist),
            expected / np.linalg.norm(expected),
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
