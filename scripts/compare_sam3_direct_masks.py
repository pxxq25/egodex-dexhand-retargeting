#!/usr/bin/env python3
"""Compare official direct SAM3 video tracking against an existing SAM2 mask run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from egodex_dexhand.data import (
    load_egodex_arm_sequence,
    load_egodex_sequence,
    project_camera_points,
)
from egodex_dexhand.segment import (
    _arm_prompt_for_frame,
    _clean_arm_hand_mask,
    _prompt_for_frame,
    adaptive_arm_hand_envelopes,
    appearance_refine_arm_hand_masks,
    stabilize_binary_mask_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--hand", required=True, choices=("left", "right"))
    parser.add_argument("--sam2-masks", required=True, type=Path)
    parser.add_argument("--sam3-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-stride", type=int, default=5)
    parser.add_argument(
        "--prompt-mode",
        choices=("geometry", "keypoints"),
        default="geometry",
        help="use the tuned geometry prompt or only real projected landmarks",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--production-output",
        action="store_true",
        help=(
            "write only the stabilized masks and metrics; skip raw-mask PNGs "
            "and the four-panel comparison video"
        ),
    )
    return parser.parse_args()


def normalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray([width, height], dtype=np.float32)
    return np.clip(np.asarray(points, dtype=np.float32) / scale, 0.0, 1.0)


def normalize_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    return np.clip(np.asarray(box, dtype=np.float32) / scale, 0.0, 1.0)


def keypoint_prompt(
    joints_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    confidence: np.ndarray,
    width: int,
    height: int,
    padding: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a positive-only prompt from real projected landmarks."""

    pixels = project_camera_points(joints_camera_cv, intrinsic)
    geometric = np.isfinite(pixels).all(axis=1)
    geometric &= joints_camera_cv[:, 2] > 1e-4
    geometric &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    geometric &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    valid = geometric & (np.asarray(confidence) >= 0.3)
    if not valid.any():
        valid = geometric
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        raise RuntimeError("no visible real keypoints for prompt")
    # Direct SAM3 accepts at most 16 prompt tokens, including two box corners.
    # Evenly sample at most 14 real landmarks rather than letting it silently
    # retain only the first and last points.
    if len(valid_indices) > max_points:
        selection = np.linspace(0, len(valid_indices) - 1, max_points)
        valid_indices = valid_indices[np.rint(selection).astype(int)]
    points = pixels[valid_indices].astype(np.float32)
    all_valid = pixels[valid]
    box = np.asarray(
        [
            max(0.0, float(all_valid[:, 0].min() - padding)),
            max(0.0, float(all_valid[:, 1].min() - padding)),
            min(float(width - 1), float(all_valid[:, 0].max() + padding)),
            min(float(height - 1), float(all_valid[:, 1].max() + padding)),
        ],
        dtype=np.float32,
    )
    return points, np.ones(len(points), dtype=np.int32), box


def masks_to_union(masks, height: int, width: int) -> np.ndarray:
    array = masks.detach().float().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    while array.ndim > 3 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise RuntimeError(f"unexpected direct SAM3 mask shape {array.shape}")
    union = np.any(array > 0.0, axis=0)
    if union.shape != (height, width):
        union = cv2.resize(
            union.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return union


def overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
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


def label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (310, 38), (0, 0, 0), -1)
    cv2.putText(
        result, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2
    )
    return result


def temporal_change(masks: list[np.ndarray]) -> float:
    if len(masks) < 2:
        return 0.0
    return float(np.mean([np.mean(a != b) for a, b in zip(masks, masks[1:])]))


def sequence_metrics(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict[str, float]:
    ious = []
    for ref, cand in zip(reference, candidate):
        union = np.count_nonzero(ref | cand)
        ious.append(float(np.count_nonzero(ref & cand) / union) if union else 1.0)
    return {
        "mean_iou_with_sam2": float(np.mean(ious)),
        "minimum_iou_with_sam2": float(np.min(ious)),
        "mean_area_fraction": float(np.mean([mask.mean() for mask in candidate])),
        "temporal_change": temporal_change(candidate),
    }


def run(args: argparse.Namespace, *, model=None) -> None:
    import torch

    frame_paths = sorted(args.frames.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"no JPEG frames in {args.frames}")
    frames = [cv2.imread(str(path)) for path in frame_paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("failed to load one or more comparison frames")
    height, width = frames[0].shape[:2]

    hand = load_egodex_sequence(args.hdf5, args.hand)
    arm = load_egodex_arm_sequence(args.hdf5, args.hand)
    if hand.frame_count != len(frames) or arm.frame_count != len(frames):
        raise RuntimeError("frame and annotation counts do not match")

    if model is None:
        from sam3.model_builder import build_sam3_video_model

        model = build_sam3_video_model(
            checkpoint_path=str(args.sam3_checkpoint),
            load_from_HF=False,
            device="cuda",
            compile=False,
        )
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    state = predictor.init_state(video_path=str(args.video))
    if int(state["num_frames"]) != len(frames):
        raise RuntimeError(
            f"SAM3 decoded {state['num_frames']} frames, expected {len(frames)}"
        )

    stride = max(1, int(args.prompt_stride))
    prompt_frames = set(range(0, len(frames), stride))
    positive_seeds: list[np.ndarray] = []
    previous_hand_valid = False
    previous_arm_valid = False
    prompt_count = 0
    with torch.inference_mode():
        for frame_index in range(len(frames)):
            hand_prompt = None
            arm_prompt = None
            if args.prompt_mode == "keypoints":
                try:
                    hand_prompt = keypoint_prompt(
                        hand.joints_camera_cv[frame_index],
                        hand.intrinsic,
                        hand.joint_confidence[frame_index],
                        width,
                        height,
                        18,
                        14,
                    )
                except RuntimeError:
                    pass
                try:
                    arm_prompt = keypoint_prompt(
                        arm.joints_camera_cv[frame_index],
                        hand.intrinsic,
                        arm.joint_confidence[frame_index],
                        width,
                        height,
                        82,
                        4,
                    )
                except RuntimeError:
                    pass
            else:
                try:
                    hand_prompt = _prompt_for_frame(
                        hand.joints_camera_cv[frame_index],
                        hand.intrinsic,
                        width,
                        height,
                        18,
                        hand.joint_confidence[frame_index],
                    )
                except RuntimeError:
                    pass
                try:
                    hand_negatives = (
                        hand_prompt[0][hand_prompt[1] == 0]
                        if hand_prompt is not None
                        else np.empty((0, 2), dtype=np.float32)
                    )
                    arm_prompt = _arm_prompt_for_frame(
                        hand.joints_camera_cv[frame_index],
                        arm.joints_camera_cv[frame_index],
                        hand.intrinsic,
                        width,
                        height,
                        hand_negatives,
                    )
                except RuntimeError:
                    pass

            hand_valid = hand_prompt is not None
            arm_valid = arm_prompt is not None
            seed_groups = []
            for prompt in (hand_prompt, arm_prompt):
                if prompt is not None:
                    points, labels, _ = prompt
                    seed_groups.append(points[labels > 0])
            positive_seeds.append(
                np.concatenate(seed_groups, axis=0)
                if seed_groups
                else np.empty((0, 2), dtype=np.float32)
            )
            scheduled = frame_index in prompt_frames
            prompt_hand = hand_valid and (scheduled or not previous_hand_valid)
            prompt_arm = arm_valid and (scheduled or not previous_arm_valid)
            if prompt_hand:
                hand_points, hand_labels, hand_box = hand_prompt
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_index,
                    obj_id=1,
                    points=normalize_points(hand_points, width, height),
                    labels=hand_labels.astype(np.int32),
                    box=normalize_box(hand_box, width, height),
                    clear_old_points=True,
                )
                prompt_count += 1
            if prompt_arm:
                arm_points, arm_labels, arm_box = arm_prompt
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_index,
                    obj_id=2,
                    points=normalize_points(arm_points, width, height),
                    labels=arm_labels.astype(np.int32),
                    box=normalize_box(arm_box, width, height),
                    clear_old_points=True,
                )
                prompt_count += 1
            previous_hand_valid = hand_valid
            previous_arm_valid = arm_valid

        if prompt_count == 0:
            # RGB fusion can keep a bimanual interval active even when one
            # side has no projected HTS geometry at all. That side must
            # contribute an explicit empty mask, not abort the other hand or
            # receive a synthetic wrist prompt.
            args.output.mkdir(parents=True, exist_ok=True)
            stable_dir = args.output / "sam3_stabilized_mask"
            stable_dir.mkdir(exist_ok=True)
            raw_dir = args.output / "sam3_raw_mask"
            if not args.production_output:
                raw_dir.mkdir(exist_ok=True)
            empty = np.zeros((height, width), dtype=np.uint8)
            for index in range(len(frames)):
                if not args.production_output:
                    cv2.imwrite(str(raw_dir / f"{index:05d}.png"), empty)
                cv2.imwrite(str(stable_dir / f"{index:05d}.png"), empty)
            metrics = {
                "frames": len(frames),
                "prompt_stride": stride,
                "prompt_mode": args.prompt_mode,
                "status": "empty_no_projected_geometry",
                "sam3_direct_raw": {
                    "mean_area_fraction": 0.0,
                    "temporal_change": 0.0,
                },
                "sam3_direct_stabilized": {
                    "mean_area_fraction": 0.0,
                    "temporal_change": 0.0,
                },
            }
            (args.output / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            print(json.dumps(metrics, indent=2))
            return

        raw_by_frame: dict[int, np.ndarray] = {}
        for frame_index, _, _, video_masks, _ in predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=len(frames) - 1,
            reverse=False,
            propagate_preflight=True,
            tqdm_disable=True,
        ):
            raw_by_frame[int(frame_index)] = masks_to_union(video_masks, height, width)

    missing = sorted(set(range(len(frames))) - set(raw_by_frame))
    if missing:
        raise RuntimeError(f"direct SAM3 did not return frames: {missing}")
    raw_masks = [raw_by_frame[index] for index in range(len(frames))]
    geometry_fallback_reason = None
    try:
        _, reference_index, sleeve_radii = adaptive_arm_hand_envelopes(
            raw_masks,
            hand.joints_camera_cv,
            arm.joints_camera_cv,
            hand.intrinsic,
            width,
            height,
        )
    except RuntimeError as exc:
        # The geometric envelope is diagnostic only in this comparison script.
        # A foreshortened/offscreen sleeve can make its width unmeasurable even
        # when the official SAM3 tracker returned useful hand masks.
        reference_index = -1
        sleeve_radii = np.empty(0, dtype=np.float32)
        geometry_fallback_reason = str(exc)
    appearance_fallback_reason = None
    try:
        appearance_masks, appearance_reference, appearance_center, appearance_scale = (
            appearance_refine_arm_hand_masks(
                raw_masks,
                frames,
                hand.joints_camera_cv,
                arm.joints_camera_cv,
                hand.intrinsic,
                width,
                height,
            )
        )
    except RuntimeError as exc:
        appearance_masks = raw_masks
        appearance_reference = -1
        appearance_center = np.full(3, np.nan, dtype=np.float32)
        appearance_scale = np.full(3, np.nan, dtype=np.float32)
        appearance_fallback_reason = str(exc)
    cleaned_masks = []
    valid_for_stabilization = []
    for index, mask in enumerate(appearance_masks):
        seeds = positive_seeds[index]
        if len(seeds) == 0:
            # The replacement HTS projection deliberately does not invent an
            # image-space prompt when all real landmarks are outside the crop.
            cleaned = np.zeros((height, width), dtype=bool)
        else:
            cleaned = _clean_arm_hand_mask(mask, seeds)
            if not cleaned.any() or cleaned.all():
                raw = np.asarray(raw_masks[index], dtype=bool)
                cleaned = raw if raw.any() and not raw.all() else np.zeros_like(raw)
        cleaned_masks.append(cleaned)
        valid_for_stabilization.append(bool(len(seeds) and cleaned.any() and not cleaned.all()))

    # Stabilize each contiguous visible interval independently. This prevents a
    # propagated mask from bridging frames where the supplied geometry says the
    # limb is outside the image, and avoids start/reappearance timing flicker.
    stabilized_masks = [np.zeros((height, width), dtype=bool) for _ in frames]
    run_start = None
    for index in range(len(frames) + 1):
        valid = index < len(frames) and valid_for_stabilization[index]
        if valid and run_start is None:
            run_start = index
        if not valid and run_start is not None:
            run_end = index
            run_masks = cleaned_masks[run_start:run_end]
            run_frames = frames[run_start:run_end]
            if len(run_masks) == 1:
                stabilized = run_masks
            else:
                stabilized = stabilize_binary_mask_sequence(run_masks, run_frames)
            stabilized_masks[run_start:run_end] = stabilized
            run_start = None

    sam2_masks = []
    for index in range(len(frames)):
        mask_path = args.sam2_masks / f"{index:05d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None:
            # A SAM3-native run may intentionally skip the legacy SAM2 stage.
            # Keep the comparison panel/metrics structurally valid without
            # making SAM2 output a prerequisite for mask generation.
            sam2_masks.append(np.zeros((height, width), dtype=bool))
        else:
            sam2_masks.append(mask > 0)

    args.output.mkdir(parents=True, exist_ok=True)
    stable_dir = args.output / "sam3_stabilized_mask"
    stable_dir.mkdir(exist_ok=True)
    raw_dir = args.output / "sam3_raw_mask"
    writer = None
    if not args.production_output:
        raw_dir.mkdir(exist_ok=True)
        panel_width = 640
        panel_height = int(round(height * panel_width / width))
        writer = cv2.VideoWriter(
            str(args.output / "original_sam2_sam3_direct.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (panel_width * 4, panel_height),
        )
        if not writer.isOpened():
            raise RuntimeError("could not create SAM3 comparison video")
    for index, frame in enumerate(frames):
        if not args.production_output:
            cv2.imwrite(
                str(raw_dir / f"{index:05d}.png"),
                raw_masks[index].astype(np.uint8) * 255,
            )
        cv2.imwrite(
            str(stable_dir / f"{index:05d}.png"),
            stabilized_masks[index].astype(np.uint8) * 255,
        )
        if writer is None:
            continue
        panels = [
            label(frame, "Original"),
            label(overlay(frame, sam2_masks[index], (0, 0, 255)), "SAM2 stabilized"),
            label(overlay(frame, raw_masks[index], (255, 255, 0)), "SAM3 direct raw"),
            label(
                overlay(frame, stabilized_masks[index], (0, 255, 0)),
                "SAM3 direct stabilized",
            ),
        ]
        panels = [cv2.resize(panel, (panel_width, panel_height)) for panel in panels]
        writer.write(np.concatenate(panels, axis=1))
    if writer is not None:
        writer.release()

    metrics = {
        "frames": len(frames),
        "prompt_stride": stride,
        "prompt_mode": args.prompt_mode,
        "production_output": bool(args.production_output),
        "adaptive_sleeve_reference_frame": reference_index,
        "adaptive_sleeve_radii_px": [float(value) for value in sleeve_radii],
        "adaptive_sleeve_fallback_reason": geometry_fallback_reason,
        "appearance_reference_frame": appearance_reference,
        "appearance_lab_center": [float(value) for value in appearance_center],
        "appearance_lab_scale": [float(value) for value in appearance_scale],
        "appearance_fallback_reason": appearance_fallback_reason,
        "sam2": {
            "mean_area_fraction": float(np.mean([mask.mean() for mask in sam2_masks])),
            "temporal_change": temporal_change(sam2_masks),
        },
        "sam3_direct_raw": sequence_metrics(sam2_masks, raw_masks),
        "sam3_direct_stabilized": sequence_metrics(sam2_masks, stabilized_masks),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
