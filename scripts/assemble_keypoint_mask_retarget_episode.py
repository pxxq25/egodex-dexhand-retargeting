#!/usr/bin/env python3
"""Assemble a full episode keypoint/mask/retarget diagnostic video."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass
class Chunk:
    start: int
    end: int
    mode: str
    output: Path
    hands: dict[str, object]
    arms: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def label(frame: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(
        result, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.68, (255, 255, 255), 2, cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            result, subtitle, (12, 47), cv2.FONT_HERSHEY_SIMPLEX,
            0.44, (210, 210, 210), 1, cv2.LINE_AA,
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


def load_chunks(run: Path, candidates: Path) -> list[Chunk]:
    chunks = []
    for output in sorted((run / "segments").iterdir()):
        if not output.is_dir():
            continue
        fields = output.name.split("_")
        if len(fields) < 4:
            continue
        mode, start, end = fields[1], int(fields[-2]), int(fields[-1])
        hdf5 = candidates / output.name / "annotations_rgb_aligned.hdf5"
        sides = ("left", "right") if mode == "both" else (mode,)
        hands = {side: load_egodex_sequence(hdf5, side) for side in sides}
        arms = {side: load_egodex_arm_sequence(hdf5, side) for side in sides}
        chunks.append(Chunk(start, end, mode, output, hands, arms))
    return chunks


def main() -> None:
    args = parse_args()
    chunks = load_chunks(args.run, args.candidates)
    frame_to_chunk: dict[int, tuple[Chunk, int]] = {}
    for chunk in chunks:
        for local_index, frame_index in enumerate(range(chunk.start, chunk.end)):
            frame_to_chunk[frame_index] = (chunk, local_index)

    original_reader = cv2.VideoCapture(str(args.recording / "camera.mp4"))
    retarget_reader = cv2.VideoCapture(
        str(args.run / "final/composite_full_recording.mp4")
    )
    if not original_reader.isOpened() or not retarget_reader.isOpened():
        raise RuntimeError("could not open episode videos")
    width = int(original_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(original_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    panel_height = int(round(height * args.panel_width / width))
    output_size = (args.panel_width * 3, panel_height)
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
    for frame_index in range(frame_count):
        ok_original, original = original_reader.read()
        ok_retarget, retargeted = retarget_reader.read()
        if not ok_original or not ok_retarget:
            raise RuntimeError(f"video ended at frame {frame_index}")
        keypoint_view = original.copy()
        mask_view = original.copy()
        active = frame_to_chunk.get(frame_index)
        if active is None:
            subtitle = "no active hand interval"
        else:
            chunk, local_index = active
            for side, hand in chunk.hands.items():
                arm = chunk.arms[side]
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
            mask = cv2.imread(
                str(chunk.output / "human_mask" / f"{local_index:05d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if mask is not None:
                alpha = (mask.astype(np.float32) / 255.0 * 0.58)[..., None]
                tint = np.full_like(mask_view, (255, 220, 0))
                mask_view = np.clip(
                    mask_view * (1.0 - alpha) + tint * alpha, 0, 255
                ).astype(np.uint8)
            subtitle = f"{chunk.mode} | source frame {frame_index:05d}"
        panels = (
            label(keypoint_view, "Projected keypoints", subtitle),
            label(mask_view, "Stabilized removal mask", subtitle),
            label(retargeted, "Retargeted", subtitle),
        )
        resized = [
            cv2.resize(panel, (args.panel_width, panel_height), cv2.INTER_AREA)
            for panel in panels
        ]
        encoder.stdin.write(np.concatenate(resized, axis=1).tobytes())
    encoder.stdin.close()
    returncode = encoder.wait()
    original_reader.release()
    retarget_reader.release()
    if returncode:
        raise RuntimeError(f"ffmpeg failed with status {returncode}")


if __name__ == "__main__":
    main()
