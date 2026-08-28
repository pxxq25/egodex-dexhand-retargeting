#!/usr/bin/env python3
"""Assemble a four-panel diagnostic over a contiguous episode window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess

import cv2
import numpy as np

from assemble_segment_four_panel import (
    ARM_EDGES,
    HAND_EDGES,
    SIDE_COLORS,
    draw_sequence,
    label,
    read_image,
)
from egodex_dexhand.data import load_egodex_arm_sequence, load_egodex_sequence


@dataclass
class Chunk:
    start: int
    end: int
    mode: str
    output: Path
    hands: dict[str, object]
    arms: dict[str, object]
    composite: cv2.VideoCapture
    next_composite_index: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--panel-width", type=int, default=512)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


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
        composite_path = output / "final/composite_full.mp4"
        if not hdf5.exists() or not composite_path.exists():
            continue
        sides = ("left", "right") if mode == "both" else (mode,)
        reader = cv2.VideoCapture(str(composite_path))
        if not reader.isOpened():
            raise RuntimeError(f"could not open {composite_path}")
        chunks.append(
            Chunk(
                start=start,
                end=end,
                mode=mode,
                output=output,
                hands={side: load_egodex_sequence(hdf5, side) for side in sides},
                arms={side: load_egodex_arm_sequence(hdf5, side) for side in sides},
                composite=reader,
            )
        )
    return chunks


def main() -> None:
    args = parse_args()
    if args.start_frame < 0 or args.end_frame <= args.start_frame:
        raise ValueError("invalid frame window")
    chunks = load_chunks(args.run, args.candidates)
    frame_to_chunk: dict[int, tuple[Chunk, int]] = {}
    for chunk in chunks:
        for local_index, frame_index in enumerate(range(chunk.start, chunk.end)):
            frame_to_chunk[frame_index] = (chunk, local_index)

    original_reader = cv2.VideoCapture(str(args.recording / "camera.mp4"))
    if not original_reader.isOpened():
        raise RuntimeError("could not open source recording")
    width = int(original_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_reader.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    panel_height = int(round(height * args.panel_width / width))
    output_size = (args.panel_width * 4, panel_height)

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

    for frame_index in range(args.start_frame, args.end_frame):
        ok, original = original_reader.read()
        if not ok:
            raise RuntimeError(f"source video ended at frame {frame_index}")
        keypoint_view = original.copy()
        mask_view = original.copy()
        robot_view = np.zeros_like(original)
        retargeted = original.copy()
        active = frame_to_chunk.get(frame_index)
        if active is None:
            subtitle = f"source frame {frame_index:05d} | no active hand interval"
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
            mask = read_image(
                chunk.output / "human_mask" / f"{local_index:05d}.png",
                cv2.IMREAD_GRAYSCALE,
            )
            alpha = (mask.astype(np.float32) / 255.0 * 0.58)[..., None]
            tint = np.full_like(mask_view, (255, 220, 0))
            mask_view = np.clip(
                mask_view * (1.0 - alpha) + tint * alpha, 0, 255
            ).astype(np.uint8)
            robot_view = read_image(
                chunk.output / "render/robot_rgb" / f"{local_index:05d}.png"
            )
            if local_index != chunk.next_composite_index:
                chunk.composite.set(cv2.CAP_PROP_POS_FRAMES, local_index)
            ok, retargeted = chunk.composite.read()
            chunk.next_composite_index = local_index + 1
            if not ok:
                raise RuntimeError(
                    f"composite {chunk.output.name} ended at {local_index}"
                )
            subtitle = f"source frame {frame_index:05d} | {chunk.mode}"

        panels = (
            label(keypoint_view, "GT + HTS keypoints", subtitle),
            label(mask_view, "SAM3 stabilized mask", subtitle),
            label(robot_view, "SAPIEN robot render", subtitle),
            label(retargeted, "Retarget output", subtitle),
        )
        resized = [
            cv2.resize(panel, (args.panel_width, panel_height), cv2.INTER_AREA)
            for panel in panels
        ]
        encoder.stdin.write(np.concatenate(resized, axis=1).tobytes())

    encoder.stdin.close()
    returncode = encoder.wait()
    original_reader.release()
    for chunk in chunks:
        chunk.composite.release()
    if returncode:
        raise RuntimeError(f"ffmpeg failed with status {returncode}")


if __name__ == "__main__":
    main()
