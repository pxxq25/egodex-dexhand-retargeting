#!/usr/bin/env python3
"""Assemble a frame-aligned four-panel diagnostic for one retarget segment."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np

from egodex_dexhand.data import (
    load_egodex_arm_sequence,
    load_egodex_sequence,
    project_camera_points,
)


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
ARM_EDGES = ((0, 1), (1, 2), (2, 3))
SIDE_COLORS = {"left": (80, 255, 80), "right": (0, 180, 255)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--panel-width", type=int, default=512)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def label(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(
        result, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        result, subtitle, (12, 50), cv2.FONT_HERSHEY_SIMPLEX,
        0.46, (215, 215, 215), 1, cv2.LINE_AA,
    )
    return result


def draw_sequence(
    frame: np.ndarray,
    points: np.ndarray,
    confidence: np.ndarray,
    intrinsic: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    pixels = project_camera_points(points, intrinsic)
    valid = (
        np.isfinite(pixels).all(axis=1)
        & np.isfinite(points).all(axis=1)
        & (points[:, 2] > 1e-4)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    for first, second in edges:
        if valid[first] and valid[second]:
            cv2.line(
                frame,
                tuple(np.rint(pixels[first]).astype(int)),
                tuple(np.rint(pixels[second]).astype(int)),
                color, 4, cv2.LINE_AA,
            )
    for index, point in enumerate(pixels):
        if not valid[index]:
            continue
        center = tuple(np.rint(point).astype(int))
        point_color = color if float(confidence[index]) >= 0.3 else (0, 0, 255)
        cv2.circle(frame, center, 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 4, point_color, -1, cv2.LINE_AA)


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    return image


def main() -> None:
    args = parse_args()
    fields = args.segment.name.split("_")
    if len(fields) < 4:
        raise ValueError(f"invalid segment name: {args.segment.name}")
    mode = fields[1]
    global_start = int(fields[-2])
    sides = ("left", "right") if mode == "both" else (mode,)
    hdf5 = args.candidate / "annotations_rgb_aligned.hdf5"
    hands = {side: load_egodex_sequence(hdf5, side) for side in sides}
    arms = {side: load_egodex_arm_sequence(hdf5, side) for side in sides}

    source_paths = sorted((args.segment / "frames").glob("*.jpg"))
    if not source_paths:
        raise RuntimeError("segment has no source frames")
    first = read_image(source_paths[0])
    height, width = first.shape[:2]
    panel_height = int(round(height * args.panel_width / width))
    output_size = (args.panel_width * 4, panel_height)

    composite_reader = cv2.VideoCapture(str(args.segment / "final/composite_full.mp4"))
    if not composite_reader.isOpened():
        raise RuntimeError("could not open final composite")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{output_size[0]}x{output_size[1]}",
            "-r", str(args.fps), "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None

    for local_index, source_path in enumerate(source_paths):
        original = read_image(source_path)
        keypoint_view = original.copy()
        for side in sides:
            hand = hands[side]
            arm = arms[side]
            color = SIDE_COLORS[side]
            draw_sequence(
                keypoint_view, arm.joints_camera_cv[local_index],
                arm.joint_confidence[local_index], hand.intrinsic,
                ARM_EDGES, color,
            )
            draw_sequence(
                keypoint_view, hand.joints_camera_cv[local_index],
                hand.joint_confidence[local_index], hand.intrinsic,
                HAND_EDGES, color,
            )

        mask = read_image(
            args.segment / "human_mask" / f"{local_index:05d}.png",
            cv2.IMREAD_GRAYSCALE,
        )
        alpha = (mask.astype(np.float32) / 255.0 * 0.58)[..., None]
        tint = np.full_like(original, (255, 220, 0))
        mask_view = np.clip(
            original * (1.0 - alpha) + tint * alpha, 0, 255
        ).astype(np.uint8)

        robot_view = read_image(
            args.segment / "render/robot_rgb" / f"{local_index:05d}.png"
        )
        ok, composite = composite_reader.read()
        if not ok:
            raise RuntimeError(f"composite ended at frame {local_index}")

        subtitle = f"source frame {global_start + local_index:05d} | {mode}"
        panels = (
            label(keypoint_view, "GT + HTS keypoints", subtitle),
            label(mask_view, "SAM3 stabilized mask", subtitle),
            label(robot_view, "SAPIEN robot render", subtitle),
            label(composite, "Retarget output", subtitle),
        )
        resized = [
            cv2.resize(panel, (args.panel_width, panel_height), cv2.INTER_AREA)
            for panel in panels
        ]
        encoder.stdin.write(np.concatenate(resized, axis=1).tobytes())

    encoder.stdin.close()
    returncode = encoder.wait()
    composite_reader.release()
    if returncode:
        raise RuntimeError(f"ffmpeg failed with status {returncode}")


if __name__ == "__main__":
    main()
