#!/usr/bin/env python3
"""Build image-aligned hand/forearm prompts from RGB with MediaPipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import mediapipe as mp
import numpy as np


HAND_SUFFIXES = (
    "Hand", "ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip",
    "ThumbTip", "IndexFingerKnuckle", "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip", "IndexFingerTip", "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase", "MiddleFingerIntermediateTip",
    "MiddleFingerTip", "RingFingerKnuckle", "RingFingerIntermediateBase",
    "RingFingerIntermediateTip", "RingFingerTip", "LittleFingerKnuckle",
    "LittleFingerIntermediateBase", "LittleFingerIntermediateTip",
    "LittleFingerTip",
)


def transforms(points: np.ndarray) -> np.ndarray:
    result = np.broadcast_to(np.eye(4, dtype=np.float32), (*points.shape[:-1], 4, 4)).copy()
    result[..., :3, 3] = points
    return result


def interpolate_track(track: np.ndarray, direct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.flatnonzero(direct)
    if not len(valid):
        raise RuntimeError("RGB detector never found a required hand")
    frames = np.arange(len(track))
    filled = track.copy()
    for joint in range(21):
        for axis in range(2):
            filled[:, joint, axis] = np.interp(
                frames, valid, track[valid, joint, axis]
            )
    confidence = np.where(direct, 1.0, 0.55).astype(np.float32)
    return filled.astype(np.float32), confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("left", "right", "both"), required=True)
    parser.add_argument("--min-detection-confidence", type=float, default=0.2)
    parser.add_argument("--brightness-gain", type=float, default=1.0)
    parser.add_argument("--brightness-offset", type=float, default=0.0)
    parser.add_argument(
        "--allow-missing-hands",
        action="store_true",
        help="write zero-confidence tracks when a requested side is never detected",
    )
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"could not decode {args.video}")
    height, width = frames[0].shape[:2]
    count = len(frames)
    tracks = {
        side: np.full((count, 21, 2), np.nan, dtype=np.float32)
        for side in ("left", "right")
    }
    direct = {side: np.zeros(count, dtype=bool) for side in ("left", "right")}

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=0.2,
        model_complexity=1,
    ) as detector:
        for frame_index, frame in enumerate(frames):
            # Brightening is detection-only. Saved source frames and final
            # composites retain the original exposure and colors.
            detection_frame = cv2.convertScaleAbs(
                frame,
                alpha=args.brightness_gain,
                beta=args.brightness_offset,
            )
            result = detector.process(
                cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            )
            detections: list[tuple[str, float, np.ndarray]] = []
            for landmarks, handedness in zip(
                result.multi_hand_landmarks or [], result.multi_handedness or []
            ):
                classification = handedness.classification[0]
                # MediaPipe handedness assumes mirrored selfie input. EgoQuest
                # is unmirrored egocentric video, so swap its labels.
                side = "left" if classification.label.lower() == "right" else "right"
                points = np.asarray(
                    [[p.x * width, p.y * height] for p in landmarks.landmark],
                    dtype=np.float32,
                )
                detections.append((side, float(classification.score), points))
            for side in ("left", "right"):
                candidates = [item for item in detections if item[0] == side]
                if candidates:
                    _, _, points = max(candidates, key=lambda item: item[1])
                    tracks[side][frame_index] = points
                    direct[side][frame_index] = True

    active_sides = ("left", "right") if args.mode == "both" else (args.mode,)
    filled: dict[str, np.ndarray] = {}
    confidence: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        if side in active_sides:
            if direct[side].any():
                filled[side], confidence[side] = interpolate_track(
                    tracks[side], direct[side]
                )
            elif args.allow_missing_hands:
                filled[side] = np.zeros((count, 21, 2), dtype=np.float32)
                confidence[side] = np.zeros(count, dtype=np.float32)
            else:
                raise RuntimeError(f"RGB detector never found the {side} hand")
        else:
            filled[side] = np.zeros((count, 21, 2), dtype=np.float32)
            confidence[side] = np.zeros(count, dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("camera/intrinsic", data=np.eye(3, dtype=np.float32))
        handle.create_dataset(
            "transforms/camera",
            data=np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)),
        )
        handle.attrs["source"] = "MediaPipe RGB mask landmarks"
        handle.attrs["mode"] = args.mode
        handle.attrs["detection_brightness_gain"] = args.brightness_gain
        handle.attrs["detection_brightness_offset"] = args.brightness_offset
        for side in ("left", "right"):
            points3 = np.concatenate(
                [filled[side], np.ones((count, 21, 1), dtype=np.float32)], axis=2
            )
            for joint, suffix in enumerate(HAND_SUFFIXES):
                name = f"{side}{suffix}"
                handle.create_dataset(
                    f"transforms/{name}", data=transforms(points3[:, joint])
                )
                handle.create_dataset(
                    f"confidences/{name}", data=confidence[side]
                )

            wrist = filled[side][:, 0]
            palm = filled[side][:, [5, 9, 13, 17]].mean(axis=1)
            direction = wrist - palm
            norm = np.linalg.norm(direction, axis=1, keepdims=True)
            direction /= np.maximum(norm, 1e-4)
            for suffix, distance in (("Forearm", 30.0), ("Arm", 80.0), ("Shoulder", 140.0)):
                pixels = wrist + distance * direction
                points = np.concatenate(
                    [pixels, np.ones((count, 1), dtype=np.float32)], axis=1
                )
                name = f"{side}{suffix}"
                handle.create_dataset(f"transforms/{name}", data=transforms(points))
                handle.create_dataset(
                    f"confidences/{name}", data=confidence[side]
                )
            handle.attrs[f"{side}_direct_detection_frames"] = int(direct[side].sum())
    print(
        f"wrote {args.output}: {count} frames; "
        + ", ".join(
            f"{side} direct={int(direct[side].sum())}" for side in active_sides
        )
    )


if __name__ == "__main__":
    main()
