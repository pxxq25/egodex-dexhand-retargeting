#!/usr/bin/env python3
"""Convert an EgoQuest aligned trajectory into the pipeline's HDF5 schema.

EgoQuest supplies camera/head/wrist poses and 21 hand landmarks, but no elbow
or shoulder tracking.  The generated arm markers are therefore used only as
SAM prompting guides; retargeting and robot IK still use the measured wrist
and hand landmarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


HAND_SUFFIXES = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
)


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.broadcast_to(np.eye(4), (len(translation), 4, 4)).copy()
    result[:, :3, :3] = rotation
    result[:, :3, 3] = translation
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--active-hand", choices=("left", "right", "both"), required=True
    )
    args = parser.parse_args()

    session = json.loads((args.recording / "session.json").read_text())
    table = pq.read_table(args.recording / "aligned_frames.parquet").slice(
        args.start_frame,
        None if args.end_frame is None else args.end_frame - args.start_frame,
    )
    rows = table.to_pydict()
    count = table.num_rows
    if count <= 0:
        raise ValueError("selected frame interval is empty")

    world_basis = np.diag([1.0, 1.0, -1.0])
    camera_basis = np.diag([1.0, -1.0, 1.0])
    camera_position_unity = np.asarray(
        rows["camera_position_world"], dtype=np.float64
    )
    camera_position = np.einsum(
        "ij,tj->ti", world_basis, camera_position_unity
    )
    camera_rotation_unity = Rotation.from_quat(
        np.asarray(rows["camera_quaternion_world"], dtype=np.float64)
    ).as_matrix()
    # Convert both Unity's left-handed world and its right/up/forward camera
    # basis.  Applying one reflection on each side produces a proper rotation.
    camera_rotation_cv = np.einsum(
        "ij,tjk,kl->til", world_basis, camera_rotation_unity, camera_basis
    )
    world_from_camera = transform(camera_rotation_cv, camera_position)

    calibration = session["camera_calibration"]
    source_width, source_height = calibration["current_resolution"]
    video_width = int(session["video_resolution"]["width"])
    video_height = int(session["video_resolution"]["height"])
    scale_x, scale_y = video_width / source_width, video_height / source_height
    fx, fy = calibration["focal_length"]
    cx, cy = calibration["principal_point"]
    intrinsic = np.asarray(
        [[fx * scale_x, 0.0, cx * scale_x],
         [0.0, fy * scale_y, cy * scale_y],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    hand_world: dict[str, np.ndarray] = {}
    hand_rotation: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        landmarks_unity = np.asarray(
            rows[f"{side}_landmarks_world"], dtype=np.float64
        ).reshape(count, 21, 3)
        hand_world[side] = np.einsum(
            "ij,tkj->tki", world_basis, landmarks_unity
        )
        rotation_unity = Rotation.from_quat(
            np.asarray(rows[f"{side}_wrist_quaternion"], dtype=np.float64)
        ).as_matrix()
        hand_rotation[side] = np.einsum(
            "ij,tjk,kl->til", world_basis, rotation_unity, world_basis
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("camera/intrinsic", data=intrinsic)
        handle.create_dataset("transforms/camera", data=world_from_camera)
        handle.attrs["source"] = "EgoQuest"
        handle.attrs["source_recording"] = str(args.recording.resolve())
        handle.attrs["source_start_frame"] = args.start_frame
        handle.attrs["source_end_frame"] = args.start_frame + count
        handle.attrs["active_hand"] = args.active_hand

        active_sides = (
            ("left", "right") if args.active_hand == "both" else (args.active_hand,)
        )
        for side in ("left", "right"):
            confidence_value = 1.0 if side in active_sides else 0.0
            confidence = np.full(count, confidence_value, dtype=np.float32)
            for joint_index, suffix in enumerate(HAND_SUFFIXES):
                name = f"{side}{suffix}"
                handle.create_dataset(
                    f"transforms/{name}",
                    data=transform(
                        hand_rotation[side], hand_world[side][:, joint_index]
                    ),
                )
                handle.create_dataset(f"confidences/{name}", data=confidence)

        # EgoQuest has no elbow/shoulder observations.  Construct camera-space
        # guide points extending from the measured wrist toward the image's
        # lower-right entry border.  They are consumed only by SAM prompting;
        # robot pose comes from the measured hand trajectory above.
        camera_from_world_rotation = np.swapaxes(camera_rotation_cv, 1, 2)
        for current_side in ("left", "right"):
            confidence_value = 1.0 if current_side in active_sides else 0.0
            confidence = np.full(count, confidence_value, dtype=np.float32)
            wrist_world = hand_world[current_side][:, 0]
            wrist_camera = np.einsum(
                "tij,tj->ti",
                camera_from_world_rotation,
                wrist_world - camera_position,
            )
            # Each sleeve enters from its corresponding lower image border.
            direction = -1.0 if current_side == "left" else 1.0
            offsets_cv = {
                "Hand": np.asarray([0.0, 0.0, 0.0]),
                "Forearm": np.asarray([0.14 * direction, 0.09, 0.05]),
                "Arm": np.asarray([0.28 * direction, 0.18, 0.10]),
                "Shoulder": np.asarray([0.42 * direction, 0.27, 0.15]),
            }
            for suffix in ("Shoulder", "Arm", "Forearm", "Hand"):
                if current_side in active_sides:
                    point_camera = wrist_camera + offsets_cv[suffix]
                    point_world = (
                        np.einsum("tij,tj->ti", camera_rotation_cv, point_camera)
                        + camera_position
                    )
                    rotation = hand_rotation[current_side]
                else:
                    point_world = hand_world[current_side][:, 0]
                    rotation = hand_rotation[current_side]
                name = f"{current_side}{suffix}"
                # Hand already exists in the 21-landmark group.
                if suffix != "Hand":
                    handle.create_dataset(
                        f"transforms/{name}", data=transform(rotation, point_world)
                    )
                handle.create_dataset(
                    f"confidences/{name}", data=confidence
                ) if f"confidences/{name}" not in handle else None

    print(
        f"wrote {args.output}: {count} frames, active={args.active_hand}, "
        f"K={intrinsic.tolist()}"
    )


if __name__ == "__main__":
    main()
