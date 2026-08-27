#!/usr/bin/env python3
"""Render selected frames from a saved UR5e + Shadow retargeting run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from egodex_dexhand.data import load_egodex_sequence, scaled_intrinsic
from egodex_dexhand.render import render_ur5e_shadow_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    frame_group = parser.add_mutually_exclusive_group(required=True)
    frame_group.add_argument("--frames", type=int, nargs="+")
    frame_group.add_argument("--all-frames", action="store_true")
    parser.add_argument("--temporal-samples", type=int, default=1)
    parser.add_argument("--render-device", required=True)
    parser.add_argument(
        "--hdf5",
        type=Path,
        help="override the trajectory HDF5 recorded in metadata.json",
    )
    parser.add_argument(
        "--hidden-arm-visual-link",
        action="append",
        default=None,
        help="override saved hidden arm links; repeat as needed",
    )
    parser.add_argument("--show-full-arm", action="store_true")
    parser.add_argument("--skip-arm-visibility-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.run / "metadata.json").read_text())
    hand = np.load(args.run / "retarget.npz")
    arm = np.load(args.run / "arm_retarget.npz")
    combined_urdf = Path(str(arm["urdf_path"]))
    if not combined_urdf.exists():
        relocated_urdf = args.run / "derived" / combined_urdf.name
        if not relocated_urdf.exists():
            raise FileNotFoundError(combined_urdf)
        combined_urdf = relocated_urdf
    frame_count = len(hand["qpos"])
    indices = (
        np.arange(frame_count, dtype=np.int64)
        if args.all_frames
        else np.asarray(args.frames, dtype=np.int64)
    )
    if np.any(indices < 0) or np.any(indices >= frame_count):
        raise ValueError(f"frame indices must be in [0, {frame_count - 1}]")

    trajectory = load_egodex_sequence(
        args.hdf5 if args.hdf5 is not None else metadata["hdf5"],
        metadata["hand"],
    )
    intrinsic = scaled_intrinsic(
        trajectory.intrinsic,
        int(metadata["width"]) / int(metadata["source_width"]),
        int(metadata["height"]) / int(metadata["source_height"]),
    )
    render_ur5e_shadow_sequence(
        arm_qpos=arm["qpos"][indices],
        arm_joint_names=tuple(str(value) for value in arm["joint_names"]),
        hand_qpos=hand["qpos"][indices],
        hand_joint_names=tuple(str(value) for value in hand["joint_names"]),
        combined_urdf_path=combined_urdf,
        base_translation_world=arm["base_translation_world"],
        base_rotation_world=arm["base_rotation_world"],
        world_from_camera=arm["world_from_camera"][indices],
        intrinsic=intrinsic,
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        output_dir=args.output,
        fps=float(metadata["fps"]),
        temporal_samples=args.temporal_samples,
        hidden_arm_visual_links=tuple(
            ()
            if args.show_full_arm
            else (
                metadata.get("hidden_arm_visual_links", ())
                if args.hidden_arm_visual_link is None
                else args.hidden_arm_visual_link
            )
        ),
        render_device=args.render_device,
        require_arm_visibility=not args.skip_arm_visibility_validation,
    )
    (args.output / "source_frame_indices.json").write_text(
        json.dumps([int(index) for index in indices], indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
