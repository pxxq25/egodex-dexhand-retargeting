#!/usr/bin/env python3
"""Show the full projected skeleton before automatic SAM prompt selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from egodex_dexhand.data import (
    load_egodex_arm_sequence,
    load_egodex_sequence,
    project_camera_points,
)
from egodex_dexhand.segment import _arm_prompt_for_frame, _prompt_for_frame


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
ARM_EDGES = ((0, 1), (1, 2), (2, 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--hand", required=True, choices=("left", "right"))
    parser.add_argument("--sam3-raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-stride", type=int, default=5)
    parser.add_argument("--prompt-mode", choices=("geometry", "keypoints"), default="geometry")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def banner(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(result, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    cv2.putText(result, subtitle, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1)
    return result


def visible(point: np.ndarray, width: int, height: int) -> bool:
    return bool(
        np.isfinite(point).all()
        and 0 <= float(point[0]) < width
        and 0 <= float(point[1]) < height
    )


def draw_skeleton(
    frame: np.ndarray,
    pixels: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    confidence: np.ndarray,
    width: int,
    height: int,
    line_color: tuple[int, int, int],
    prefix: str,
) -> None:
    valid = [visible(point, width, height) for point in pixels]
    for first, second in edges:
        if valid[first] and valid[second]:
            cv2.line(
                frame,
                tuple(np.rint(pixels[first]).astype(int)),
                tuple(np.rint(pixels[second]).astype(int)),
                line_color,
                2,
                cv2.LINE_AA,
            )
    for index, point in enumerate(pixels):
        if not valid[index]:
            continue
        center = tuple(np.rint(point).astype(int))
        trusted = float(confidence[index]) >= 0.3
        color = (0, 255, 0) if trusted else (0, 165, 255)
        cv2.circle(frame, center, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{prefix}{index}",
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_prompt(
    frame: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    box: np.ndarray,
    positive: tuple[int, int, int],
    negative: tuple[int, int, int],
    box_color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = np.rint(box).astype(int)
    cv2.rectangle(frame, (x0, y0), (x1, y1), box_color, 3, cv2.LINE_AA)
    for point, label in zip(points, labels):
        center = tuple(np.rint(point).astype(int))
        if int(label) > 0:
            cv2.circle(frame, center, 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, center, 4, positive, -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(frame, center, negative, cv2.MARKER_TILTED_CROSS, 14, 3)


def real_keypoint_prompt(
    joints: np.ndarray,
    intrinsic: np.ndarray,
    confidence: np.ndarray,
    width: int,
    height: int,
    padding: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = project_camera_points(joints, intrinsic)
    geometric = np.isfinite(pixels).all(axis=1) & (joints[:, 2] > 1e-4)
    geometric &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    geometric &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    valid = geometric & (np.asarray(confidence) >= 0.3)
    if not valid.any():
        valid = geometric
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise RuntimeError("no visible real keypoints")
    if len(indices) > max_points:
        indices = indices[
            np.rint(np.linspace(0, len(indices) - 1, max_points)).astype(int)
        ]
    points = pixels[indices].astype(np.float32)
    extent = pixels[valid]
    box = np.asarray(
        [
            max(0.0, float(extent[:, 0].min() - padding)),
            max(0.0, float(extent[:, 1].min() - padding)),
            min(float(width - 1), float(extent[:, 0].max() + padding)),
            min(float(height - 1), float(extent[:, 1].max() + padding)),
        ],
        dtype=np.float32,
    )
    return points, np.ones(len(points), dtype=np.int32), box


def main() -> None:
    args = parse_args()
    frame_paths = sorted(args.frames.glob("*.jpg"))
    frames = [cv2.imread(str(path)) for path in frame_paths]
    if not frames or any(frame is None for frame in frames):
        raise RuntimeError("failed to load frames")
    height, width = frames[0].shape[:2]
    hand = load_egodex_sequence(args.hdf5, args.hand)
    arm = load_egodex_arm_sequence(args.hdf5, args.hand)
    if hand.frame_count != len(frames) or arm.frame_count != len(frames):
        raise RuntimeError("frame and annotation counts do not match")

    prompt_frames = set(range(0, len(frames), max(1, args.prompt_stride)))
    panel_width = 640
    panel_height = int(round(height * panel_width / width))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (panel_width * 4, panel_height),
    )
    for index, frame in enumerate(frames):
        original = banner(frame, "1. Original", f"frame {index:05d}")

        keypoint_view = frame.copy()
        hand_pixels = project_camera_points(hand.joints_camera_cv[index], hand.intrinsic)
        arm_pixels = project_camera_points(arm.joints_camera_cv[index], hand.intrinsic)
        draw_skeleton(
            keypoint_view, hand_pixels, HAND_EDGES, hand.joint_confidence[index],
            width, height, (255, 200, 0), "H",
        )
        draw_skeleton(
            keypoint_view, arm_pixels, ARM_EDGES, arm.joint_confidence[index],
            width, height, (255, 0, 255), "A",
        )
        keypoint_view = banner(
            keypoint_view, "2. Projected 3D keypoints", "green=confidence >=0.3; orange=fallback",
        )

        prompt_view = frame.copy()
        if index in prompt_frames:
            if args.prompt_mode == "keypoints":
                hand_points, hand_labels, hand_box = real_keypoint_prompt(
                    hand.joints_camera_cv[index], hand.intrinsic,
                    hand.joint_confidence[index], width, height, 18, 14,
                )
                try:
                    arm_points, arm_labels, arm_box = real_keypoint_prompt(
                        arm.joints_camera_cv[index], hand.intrinsic,
                        arm.joint_confidence[index], width, height, 82, 4,
                    )
                except RuntimeError:
                    wrist = project_camera_points(
                        hand.joints_camera_cv[index, :1], hand.intrinsic
                    )[0]
                    if not visible(wrist, width, height):
                        wrist = hand_points[0]
                    arm_points = wrist[None].astype(np.float32)
                    arm_labels = np.ones(1, dtype=np.int32)
                    arm_box = np.asarray(
                        [
                            max(0.0, float(wrist[0] - 82)),
                            max(0.0, float(wrist[1] - 82)),
                            min(float(width - 1), float(wrist[0] + 82)),
                            min(float(height - 1), float(wrist[1] + 82)),
                        ],
                        dtype=np.float32,
                    )
            else:
                hand_points, hand_labels, hand_box = _prompt_for_frame(
                    hand.joints_camera_cv[index], hand.intrinsic, width, height, 18,
                    hand.joint_confidence[index],
                )
                arm_points, arm_labels, arm_box = _arm_prompt_for_frame(
                    hand.joints_camera_cv[index], arm.joints_camera_cv[index],
                    hand.intrinsic, width, height, hand_points[hand_labels == 0],
                )
            draw_prompt(prompt_view, hand_points, hand_labels, hand_box, (0, 255, 0), (0, 0, 255), (0, 255, 0))
            draw_prompt(prompt_view, arm_points, arm_labels, arm_box, (255, 255, 0), (255, 0, 255), (0, 165, 255))
            prompt_text = f"{args.prompt_mode}: selected boxes + clicks"
        else:
            prompt_text = "no prompt on this frame"
        prompt_view = banner(prompt_view, "3. SAM prompt", prompt_text)

        mask = cv2.imread(str(args.sam3_raw / f"{index:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"missing SAM3 mask {index}")
        binary = mask > 0
        mask_view = frame.copy()
        tint = np.empty_like(frame)
        tint[:] = (255, 255, 0)
        alpha = (binary.astype(np.float32) * 0.55)[..., None]
        mask_view = np.clip(mask_view * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
        mask_view = banner(
            mask_view, "4. Raw SAM3 mask", "PROMPTED" if index in prompt_frames else "PROPAGATED",
        )
        panels = [original, keypoint_view, prompt_view, mask_view]
        panels = [cv2.resize(panel, (panel_width, panel_height)) for panel in panels]
        writer.write(np.concatenate(panels, axis=1))
    writer.release()


if __name__ == "__main__":
    main()
