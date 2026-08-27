#!/usr/bin/env python3
"""Run SAM2 propagation and retain its pre-cleanup arm/hand union masks."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import cv2
import numpy as np

from egodex_dexhand.data import load_egodex_arm_sequence, load_egodex_sequence
from egodex_dexhand.segment import SAM2_CONFIGS, _arm_prompt_for_frame, _prompt_for_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--hand", required=True, choices=("left", "right"))
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-stride", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    hand = load_egodex_sequence(args.hdf5, args.hand)
    arm = load_egodex_arm_sequence(args.hdf5, args.hand)
    frame_paths = sorted(args.frames.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"no JPEG frames in {args.frames}")
    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    if hand.frame_count != len(frame_paths) or arm.frame_count != len(frame_paths):
        raise RuntimeError("frame and annotation counts do not match")

    sam2_root = args.sam2_root.resolve()
    sys.path.insert(0, str(sam2_root))
    try:
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            SAM2_CONFIGS["small"], str(args.checkpoint.resolve()), device="cuda"
        )
        state = predictor.init_state(
            video_path=str(args.frames.resolve()),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=True,
        )
        predictor.reset_state(state)
        prompt_frames = list(range(0, len(frame_paths), max(1, args.prompt_stride)))
        if prompt_frames[-1] != len(frame_paths) - 1:
            prompt_frames.append(len(frame_paths) - 1)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for frame_index in prompt_frames:
                hand_points, hand_labels, hand_box = _prompt_for_frame(
                    hand.joints_camera_cv[frame_index],
                    hand.intrinsic,
                    width,
                    height,
                    18,
                    hand.joint_confidence[frame_index],
                )
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_index,
                    obj_id=1,
                    points=hand_points,
                    labels=hand_labels,
                    box=hand_box,
                )
                arm_points, arm_labels, arm_box = _arm_prompt_for_frame(
                    hand.joints_camera_cv[frame_index],
                    arm.joints_camera_cv[frame_index],
                    hand.intrinsic,
                    width,
                    height,
                    hand_points[hand_labels == 0],
                )
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_index,
                    obj_id=2,
                    points=arm_points,
                    labels=arm_labels,
                    box=arm_box,
                )

            masks: dict[int, np.ndarray] = {}
            for frame_index, object_ids, logits in predictor.propagate_in_video(state):
                ids = list(object_ids)
                if 1 not in ids or 2 not in ids:
                    raise RuntimeError(f"SAM2 lost an object at frame {frame_index}: {ids}")
                hand_mask = (logits[ids.index(1), 0] > 0).cpu().numpy()
                arm_mask = (logits[ids.index(2), 0] > 0).cpu().numpy()
                masks[int(frame_index)] = hand_mask | arm_mask
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(sam2_root))

    missing = sorted(set(range(len(frame_paths))) - set(masks))
    if missing:
        raise RuntimeError(f"SAM2 did not return frames: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    for frame_index, mask in sorted(masks.items()):
        if not cv2.imwrite(
            str(args.output / f"{frame_index:05d}.png"), mask.astype(np.uint8) * 255
        ):
            raise RuntimeError(f"failed to write raw mask {frame_index}")


if __name__ == "__main__":
    main()
