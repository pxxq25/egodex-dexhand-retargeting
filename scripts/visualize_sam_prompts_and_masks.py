#!/usr/bin/env python3
"""Visualize geometry prompts and raw SAM2/SAM3 propagation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from egodex_dexhand.data import load_egodex_arm_sequence, load_egodex_sequence
from egodex_dexhand.segment import _arm_prompt_for_frame, _prompt_for_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--hand", required=True, choices=("left", "right"))
    parser.add_argument("--sam2-raw", required=True, type=Path)
    parser.add_argument("--sam3-raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-stride", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def read_mask(directory: Path, index: int) -> np.ndarray:
    mask = cv2.imread(str(directory / f"{index:05d}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"missing mask {index:05d}.png in {directory}")
    return mask > 0


def banner(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(
        result, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2
    )
    cv2.putText(
        result, subtitle, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1
    )
    return result


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = frame.copy()
    tint = np.empty_like(frame)
    tint[:] = color
    alpha = (mask.astype(np.float32) * 0.55)[..., None]
    result = np.clip(result * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, color, 2, cv2.LINE_AA)
    return result


def draw_box(frame: np.ndarray, box: np.ndarray, color: tuple[int, int, int], text: str) -> None:
    x0, y0, x1, y1 = np.rint(box).astype(int)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
    cv2.putText(
        frame,
        text,
        (max(0, x0), max(78, y0 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    positive_color: tuple[int, int, int],
    negative_color: tuple[int, int, int],
) -> None:
    for point, point_label in zip(points, labels):
        center = tuple(np.rint(point).astype(int))
        if int(point_label) > 0:
            cv2.circle(frame, center, 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, center, 5, positive_color, -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(
                frame,
                center,
                negative_color,
                cv2.MARKER_TILTED_CROSS,
                15,
                3,
                cv2.LINE_AA,
            )


def main() -> None:
    args = parse_args()
    frame_paths = sorted(args.frames.glob("*.jpg"))
    frames = [cv2.imread(str(path)) for path in frame_paths]
    if not frames or any(frame is None for frame in frames):
        raise RuntimeError("failed to load input frames")
    height, width = frames[0].shape[:2]
    hand = load_egodex_sequence(args.hdf5, args.hand)
    arm = load_egodex_arm_sequence(args.hdf5, args.hand)
    if hand.frame_count != len(frames) or arm.frame_count != len(frames):
        raise RuntimeError("frame and annotation counts do not match")

    stride = max(1, int(args.prompt_stride))
    sam3_prompt_frames = set(range(0, len(frames), stride))
    sam2_prompt_frames = set(sam3_prompt_frames)
    sam2_prompt_frames.add(len(frames) - 1)
    panel_width = 640
    panel_height = int(round(height * panel_width / width))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (panel_width * 4, panel_height),
    )

    for index, frame in enumerate(frames):
        sam2_prompted = index in sam2_prompt_frames
        sam3_prompted = index in sam3_prompt_frames
        original = banner(frame, "Original", f"frame {index:05d}")
        prompt_view = frame.copy()
        if sam2_prompted or sam3_prompted:
            hand_points, hand_labels, hand_box = _prompt_for_frame(
                hand.joints_camera_cv[index],
                hand.intrinsic,
                width,
                height,
                18,
                hand.joint_confidence[index],
            )
            arm_points, arm_labels, arm_box = _arm_prompt_for_frame(
                hand.joints_camera_cv[index],
                arm.joints_camera_cv[index],
                hand.intrinsic,
                width,
                height,
                hand_points[hand_labels == 0],
            )
            draw_box(prompt_view, hand_box, (0, 255, 0), "HAND BOX")
            draw_box(prompt_view, arm_box, (0, 165, 255), "ARM BOX")
            draw_points(prompt_view, hand_points, hand_labels, (0, 255, 0), (0, 0, 255))
            draw_points(prompt_view, arm_points, arm_labels, (255, 255, 0), (255, 0, 255))
            prompt_status = "boxes + clicks supplied"
        else:
            prompt_status = "no new input; model propagates"
        prompt_view = banner(prompt_view, "Geometry prompt", prompt_status)

        sam2_mask = read_mask(args.sam2_raw, index)
        sam3_mask = read_mask(args.sam3_raw, index)
        sam2_status = "PROMPTED" if sam2_prompted else "PROPAGATED"
        sam3_status = "PROMPTED" if sam3_prompted else "PROPAGATED"
        sam2_view = banner(
            overlay_mask(frame, sam2_mask, (0, 0, 255)), "Raw SAM2 output", sam2_status
        )
        sam3_view = banner(
            overlay_mask(frame, sam3_mask, (255, 255, 0)), "Raw SAM3 output", sam3_status
        )
        panels = [original, prompt_view, sam2_view, sam3_view]
        panels = [cv2.resize(panel, (panel_width, panel_height)) for panel in panels]
        writer.write(np.concatenate(panels, axis=1))
    writer.release()


if __name__ == "__main__":
    main()
