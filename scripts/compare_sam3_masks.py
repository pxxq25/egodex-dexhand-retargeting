#!/usr/bin/env python3
"""Compare SAM 3.1 hand/arm tracking with an existing SAM2 mask sequence."""

from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument("--sam2-masks", required=True, type=Path)
    parser.add_argument("--sam3-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-stride", type=int, default=5)
    parser.add_argument(
        "--text-prompt",
        help="use SAM3 text-driven detection/tracking instead of point prompts",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def relative_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray([width, height], dtype=np.float32)
    return np.clip(np.asarray(points, dtype=np.float32) / scale, 0.0, 1.0)


def extract_union(outputs: dict, height: int, width: int) -> np.ndarray:
    masks = outputs.get("out_binary_masks")
    if masks is None:
        masks = outputs.get("pred_masks")
    if masks is None:
        raise RuntimeError(f"SAM3 output has no mask field: {sorted(outputs)}")
    masks = np.asarray(masks)
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise RuntimeError(f"unexpected SAM3 mask shape {masks.shape}")
    union = np.any(masks.astype(bool), axis=0)
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
    cv2.rectangle(result, (0, 0), (260, 40), (0, 0, 0), -1)
    cv2.putText(
        result, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
    )
    return result


def temporal_change(masks: list[np.ndarray]) -> float:
    if len(masks) < 2:
        return 0.0
    return float(
        np.mean(
            [np.mean(a.astype(np.float32) != b.astype(np.float32)) for a, b in zip(masks, masks[1:])]
        )
    )


def main() -> None:
    args = parse_args()
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    frame_paths = sorted(args.frames.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"no JPEG frames in {args.frames}")
    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    hand = load_egodex_sequence(args.hdf5, args.hand)
    arm = load_egodex_arm_sequence(args.hdf5, args.hand)
    if hand.frame_count != len(frame_paths) or arm.frame_count != len(frame_paths):
        raise RuntimeError("frame and annotation counts do not match")

    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(args.sam3_checkpoint),
        max_num_objects=4,
        # The released SAM 3.1 checkpoint is parameterized for 16 multiplex
        # slots even when a session tracks fewer objects.
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        warm_up=False,
        async_loading_frames=True,
    )

    # SAM 3.1's current start_session wrapper forwards an unsupported
    # offload_state_to_cpu argument. Register the state through the model's
    # supported signature until the upstream fix lands.
    state = predictor.model.init_state(
        resource_path=str(args.frames),
        offload_video_to_cpu=True,
        async_loading_frames=True,
    )
    session_id = "egodex-sam3-comparison"
    now = time.time()
    predictor._all_inference_states[session_id] = {
        "state": state,
        "session_id": session_id,
        "start_time": now,
        "last_use_time": now,
    }

    stride = max(1, int(args.prompt_stride))
    prompt_frames = list(range(0, len(frame_paths), stride))

    prompted_outputs: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        if args.text_prompt:
            response = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": args.text_prompt,
                }
            )
            prompted_outputs[0] = extract_union(response["outputs"], height, width)
        for frame_index in ([] if args.text_prompt else prompt_frames):
            hand_points, hand_labels, _ = _prompt_for_frame(
                hand.joints_camera_cv[frame_index],
                hand.intrinsic,
                width,
                height,
                18,
                hand.joint_confidence[frame_index],
            )
            hand_response = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": frame_index,
                    "points": relative_points(hand_points, width, height),
                    "point_labels": hand_labels.astype(np.int32),
                    "obj_id": 1,
                }
            )
            arm_points, arm_labels, _ = _arm_prompt_for_frame(
                hand.joints_camera_cv[frame_index],
                arm.joints_camera_cv[frame_index],
                hand.intrinsic,
                width,
                height,
                hand_points[hand_labels == 0],
            )
            arm_response = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": frame_index,
                    "points": relative_points(arm_points, width, height),
                    "point_labels": arm_labels.astype(np.int32),
                    "obj_id": 2,
                }
            )
            prompted_outputs[frame_index] = extract_union(
                arm_response.get("outputs", hand_response["outputs"]), height, width
            )

        sam3_masks: dict[int, np.ndarray] = {}
        for response in predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                # SAM3.1's multiplex scheduler cannot infer a start frame from
                # point-only prompts because it inspects semantic-stage outputs.
                "start_frame_index": 0,
                "output_prob_thresh": 0.5,
            }
        ):
            sam3_masks[int(response["frame_index"])] = extract_union(
                response["outputs"], height, width
            )

    sam3_masks.update({k: v for k, v in prompted_outputs.items() if k not in sam3_masks})
    missing = sorted(set(range(len(frame_paths))) - set(sam3_masks))
    if missing:
        raise RuntimeError(f"SAM3 did not return frames: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output / "sam3_mask"
    mask_dir.mkdir(exist_ok=True)
    sam2_masks: list[np.ndarray] = []
    ordered_sam3: list[np.ndarray] = []
    writer = cv2.VideoWriter(
        str(args.output / "original_sam2_sam3.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width * 3, height),
    )
    ious: list[float] = []
    for frame_index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        sam2 = cv2.imread(
            str(args.sam2_masks / f"{frame_index:05d}.png"), cv2.IMREAD_GRAYSCALE
        )
        if sam2 is None:
            raise RuntimeError(f"missing SAM2 mask at frame {frame_index}")
        sam2 = sam2 > 0
        sam3 = sam3_masks[frame_index]
        cv2.imwrite(str(mask_dir / f"{frame_index:05d}.png"), sam3.astype(np.uint8) * 255)
        union = np.count_nonzero(sam2 | sam3)
        ious.append(float(np.count_nonzero(sam2 & sam3) / union) if union else 1.0)
        sam2_masks.append(sam2)
        ordered_sam3.append(sam3)
        panel = np.concatenate(
            [
                label(frame, "Original"),
                label(overlay(frame, sam2, (0, 0, 255)), "SAM2 mask"),
                label(overlay(frame, sam3, (255, 255, 0)), "SAM3.1 mask"),
            ],
            axis=1,
        )
        writer.write(panel)
    writer.release()

    metrics = {
        "frames": len(frame_paths),
        "prompt_stride": stride,
        "text_prompt": args.text_prompt,
        "mean_sam2_sam3_iou": float(np.mean(ious)),
        "minimum_sam2_sam3_iou": float(np.min(ious)),
        "sam2_temporal_change": temporal_change(sam2_masks),
        "sam3_temporal_change": temporal_change(ordered_sam3),
        "sam2_mean_area_fraction": float(np.mean([m.mean() for m in sam2_masks])),
        "sam3_mean_area_fraction": float(np.mean([m.mean() for m in ordered_sam3])),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictor.close_session(session_id)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
