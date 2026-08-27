#!/usr/bin/env python3
"""Initialize one explicit SAPIEN Vulkan device and capture one frame."""

from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    import sapien

    started = time.monotonic()
    render_system = sapien.render.RenderSystem(args.device)
    scene = sapien.Scene(
        [sapien.physx.PhysxCpuSystem(), render_system]
    )
    camera = scene.add_camera("vulkan-preflight", 32, 32, 1.0, 0.01, 10.0)
    scene.update_render()
    camera.take_picture()
    color = camera.get_picture("Color")
    if color.shape != (32, 32, 4):
        raise RuntimeError(f"unexpected preflight frame shape: {color.shape}")
    print(json.dumps({
        "device": args.device,
        "frame_shape": list(color.shape),
        "elapsed_seconds": time.monotonic() - started,
    }))


if __name__ == "__main__":
    main()
