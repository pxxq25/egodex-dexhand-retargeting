from __future__ import annotations

import unittest

from egodex_dexhand.render import _create_sapien_scene


class ExplicitVulkanTests(unittest.TestCase):
    def test_scene_creation_rejects_missing_device_before_importing_sapien(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit SAPIEN Vulkan"):
            _create_sapien_scene("")


if __name__ == "__main__":
    unittest.main()
