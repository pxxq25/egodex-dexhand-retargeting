#!/usr/bin/env python3
"""Blend two lossless robot-render variants over a frame interval."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def numbered(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.png"), key=lambda path: int(path.stem))


def read(path: Path, grayscale: bool = False) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path, help="render root used before the fade")
    parser.add_argument("alternate", type=Path, help="render root used after the fade")
    parser.add_argument("output", type=Path)
    parser.add_argument("--fade-start", type=int, required=True)
    parser.add_argument("--fade-end", type=int, required=True)
    args = parser.parse_args()
    if args.fade_end <= args.fade_start:
        raise ValueError("fade-end must be greater than fade-start")

    names = ("robot_rgb", "robot_premultiplied", "robot_alpha", "robot_mask")
    sources = {
        variant: {name: numbered(root / name) for name in names}
        for variant, root in (("base", args.base), ("alternate", args.alternate))
    }
    frame_count = len(sources["base"]["robot_rgb"])
    if not frame_count or any(
        len(paths) != frame_count
        for variant in sources.values()
        for paths in variant.values()
    ):
        raise ValueError("render variants have inconsistent frame counts")
    for name in names:
        (args.output / name).mkdir(parents=True, exist_ok=True)

    for frame in range(frame_count):
        weight = np.clip(
            (frame - args.fade_start) / (args.fade_end - args.fade_start), 0.0, 1.0
        )
        base_alpha = read(sources["base"]["robot_alpha"][frame], True).astype(np.float32)
        alt_alpha = read(sources["alternate"]["robot_alpha"][frame], True).astype(np.float32)
        base_pre = read(sources["base"]["robot_premultiplied"][frame]).astype(np.float32)
        alt_pre = read(sources["alternate"]["robot_premultiplied"][frame]).astype(np.float32)
        alpha = (1.0 - weight) * base_alpha + weight * alt_alpha
        premultiplied = (1.0 - weight) * base_pre + weight * alt_pre
        straight = np.zeros_like(premultiplied)
        visible = alpha > 0.5
        straight[visible] = (
            premultiplied[visible]
            * 255.0
            / np.maximum(alpha[visible][:, None], 1.0)
        )
        filename = f"{frame:05d}.png"
        cv2.imwrite(str(args.output / "robot_alpha" / filename), np.clip(alpha, 0, 255).astype(np.uint8))
        cv2.imwrite(
            str(args.output / "robot_premultiplied" / filename),
            np.clip(premultiplied, 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            str(args.output / "robot_rgb" / filename),
            np.clip(straight, 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            str(args.output / "robot_mask" / filename),
            (alpha > 1.0).astype(np.uint8) * 255,
        )

    print(
        f"blended {frame_count} frames: base through {args.fade_start}, "
        f"alternate from {args.fade_end}"
    )


if __name__ == "__main__":
    main()
