#!/usr/bin/env python3
"""Run the generic screen-registered Shadow retarget + residual inpaint path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from egodex_dexhand.compose import composite_videos
from egodex_dexhand.data import (
    load_egodex_arm_sequence,
    load_egodex_sequence,
    project_camera_points,
    scaled_intrinsic,
)
from egodex_dexhand.inpaint import (
    build_robot_context_frames,
    build_robot_aware_removal_masks,
    evaluate_inpaint_change,
    run_propainter,
)
from egodex_dexhand.screen_registration import fit_shadow_screen_registration
from egodex_dexhand.screen_render import render_screen_registered_shadow_sequence
from egodex_dexhand.visual_forearm import (
    estimate_forearm_observation_sequence,
    refine_human_silhouette_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "fit a Shadow Hand to all 21 landmarks, infer forearm dimensions "
            "from the human silhouette, and inpaint only visible residuals"
        )
    )
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--propainter-root", required=True, type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--render-device", required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--allow-gate-failure", action="store_true")
    return parser.parse_args()


def _label(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(
        result,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        subtitle,
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return result


def _prepare_inputs(
    frames: list[Path], masks: list[Path], output: Path, fps: float
) -> tuple[Path, Path, Path, int, int]:
    frames_dir = output / "inputs/frames"
    masks_dir = output / "inputs/human_mask"
    frames_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"could not decode {frames[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output / "inputs/source.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("could not create source.mp4")
    for index, (frame_path, mask_path) in enumerate(zip(frames, masks)):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None or frame.shape[:2] != (height, width):
            raise RuntimeError(f"invalid input frame {index}")
        frame_link = frames_dir / f"{index:05d}.jpg"
        mask_link = masks_dir / f"{index:05d}.png"
        frame_link.symlink_to(frame_path.resolve())
        mask_link.symlink_to(mask_path.resolve())
        writer.write(frame)
    writer.release()
    return frames_dir, masks_dir, output / "inputs/source.mp4", width, height


def _write_review(
    source_video: Path,
    residual_masks: Path,
    render_dir: Path,
    final_video: Path,
    output: Path,
) -> None:
    source = cv2.VideoCapture(str(source_video))
    final = cv2.VideoCapture(str(final_video))
    fps = float(source.get(cv2.CAP_PROP_FPS))
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    masks = sorted(residual_masks.glob("*.png"))
    robot_frames = sorted((render_dir / "robot_rgb").glob("*.png"))
    if not masks or len(masks) != len(robot_frames):
        raise ValueError("review mask and robot frame counts differ")
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 4, height),
    )
    if not source.isOpened() or not final.isOpened() or not writer.isOpened():
        raise RuntimeError("could not create human review video")
    for index, (mask_path, robot_path) in enumerate(zip(masks, robot_frames)):
        ok_source, source_frame = source.read()
        ok_final, final_frame = final.read()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        robot = cv2.imread(str(robot_path), cv2.IMREAD_COLOR)
        if not ok_source or not ok_final or mask is None or robot is None:
            raise RuntimeError(f"review input ended at frame {index}")
        mask_preview = source_frame.copy()
        white = np.full_like(mask_preview, 255)
        alpha = (mask.astype(np.float32) / 255.0 * 0.72)[..., None]
        mask_preview = np.rint(
            mask_preview * (1.0 - alpha) + white * alpha
        ).astype(np.uint8)
        robot_preview = np.full_like(robot, 48)
        robot_mask = cv2.imread(
            str(render_dir / "robot_mask" / f"{index:05d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if robot_mask is None:
            raise RuntimeError(f"missing robot mask {index}")
        selector = robot_mask > 0
        robot_preview[selector] = robot[selector]
        panels = np.concatenate(
            [
                _label(source_frame, "1  Source", "original frame; no keypoints"),
                _label(
                    mask_preview,
                    "2  Pixels to remove",
                    "white = human visible outside opaque robot",
                ),
                _label(
                    robot_preview,
                    "3  Registered robot",
                    "21-point hand fit + silhouette-sized forearm",
                ),
                _label(
                    final_frame,
                    "4  Final result",
                    "residual inpaint + robot composite",
                ),
            ],
            axis=1,
        )
        writer.write(panels)
    extra_source, _ = source.read()
    extra_final, _ = final.read()
    source.release()
    final.release()
    writer.release()
    if extra_source or extra_final:
        raise ValueError("review source or final video has extra frames")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    metadata = json.loads((args.segment / "metadata.json").read_text())
    fps = float(metadata["fps"])
    source_frames = sorted((args.segment / "frames").glob("*.jpg"))
    human_masks = sorted((args.segment / "human_mask").glob("*.png"))
    if not source_frames or len(source_frames) != len(human_masks):
        raise ValueError("segment frames and human masks are missing or misaligned")
    frame_count = len(source_frames)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("max-frames must be positive")
        frame_count = min(frame_count, args.max_frames)
    source_frames = source_frames[:frame_count]
    human_masks = human_masks[:frame_count]
    frames_dir, _raw_human_mask_dir, source_video, width, height = _prepare_inputs(
        source_frames, human_masks, args.output, fps
    )

    hand_data = np.load(args.segment / "retarget.npz")
    arm_data = np.load(args.segment / "arm_retarget.npz")
    annotation_name = Path(
        str(metadata.get("hdf5", "annotations.hdf5"))
    ).name
    annotation_path = args.candidate / annotation_name
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"segment provenance requires missing annotation {annotation_path}"
        )
    hand_sequence = load_egodex_sequence(annotation_path, args.side)
    arm_sequence = load_egodex_arm_sequence(annotation_path, args.side)
    intrinsic = scaled_intrinsic(
        hand_sequence.intrinsic,
        width / int(metadata.get("source_width", width)),
        height / int(metadata.get("source_height", height)),
    )
    human_camera = np.asarray(
        hand_sequence.joints_camera_cv[:frame_count], dtype=np.float64
    )
    human_pixels = project_camera_points(human_camera, intrinsic)
    registration = fit_shadow_screen_registration(
        hand_data["qpos"][:frame_count],
        tuple(str(value) for value in hand_data["joint_names"]),
        str(hand_data["urdf_path"].item()),
        human_camera,
        confidence=hand_sequence.joint_confidence[:frame_count],
    )
    fitted_pixels = project_camera_points(
        registration.fitted_landmarks_camera, intrinsic
    )
    landmark_error = np.linalg.norm(fitted_pixels - human_pixels, axis=-1)

    frame_arrays = [
        cv2.imread(str(path), cv2.IMREAD_COLOR) for path in source_frames
    ]
    raw_mask_arrays = [
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in human_masks
    ]
    if any(value is None for value in frame_arrays) or any(
        value is None for value in raw_mask_arrays
    ):
        raise RuntimeError("could not decode source frames or human masks")
    palm_width = np.linalg.norm(human_pixels[:, 5] - human_pixels[:, 17], axis=1)
    mask_arrays, refinement_summary = refine_human_silhouette_sequence(
        frame_arrays,  # type: ignore[arg-type]
        raw_mask_arrays,  # type: ignore[arg-type]
        human_pixels[:, 0],
        palm_width,
    )
    human_mask_dir = args.output / "inputs/refined_human_mask"
    human_mask_dir.mkdir()
    for index, mask in enumerate(mask_arrays):
        output_mask = human_mask_dir / f"{index:05d}.png"
        if not cv2.imwrite(str(output_mask), mask):
            raise RuntimeError(f"could not write refined human mask {index}")
    observations = estimate_forearm_observation_sequence(
        mask_arrays,  # type: ignore[arg-type]
        human_camera[:, 0],
        human_pixels[:, 0],
        intrinsic,
        annotation_guide_camera=arm_sequence.joints_camera_cv[:frame_count, 2],
        palm_width_pixels=palm_width,
    )
    render_dir = args.output / "render"
    render_summary = render_screen_registered_shadow_sequence(
        registration,
        observations,
        hand_data["qpos"][:frame_count],
        tuple(str(value) for value in hand_data["joint_names"]),
        str(hand_data["urdf_path"].item()),
        str(arm_data["urdf_path"].item()),
        human_camera[:, 0],
        human_pixels[:, 0],
        intrinsic,
        width,
        height,
        render_dir,
        fps,
        args.render_device,
    )

    removal_masks = args.output / "robot_aware_removal_mask"
    mask_summary = build_robot_aware_removal_masks(
        human_mask_dir,
        render_dir / "robot_alpha",
        removal_masks,
    )
    inpaint_context = build_robot_context_frames(
        frames_dir,
        render_dir / "robot_rgb",
        render_dir / "robot_alpha",
        args.output / "robot_context_frames",
    )
    inpainted = run_propainter(
        frames_dir=inpaint_context,
        masks_dir=removal_masks,
        propainter_root=args.propainter_root,
        output_dir=args.output / "inpaint",
        python_executable=args.python_executable,
        fps=fps,
        fp16=True,
    )
    change_summary = evaluate_inpaint_change(frames_dir, removal_masks, inpainted)
    final_dir = args.output / "final"
    composite_videos(
        source_video=source_video,
        inpainted_video=inpainted,
        robot_rgb_dir=render_dir / "robot_rgb",
        robot_mask_dir=render_dir / "robot_mask",
        human_mask_dir=removal_masks,
        output_dir=final_dir,
    )
    _write_review(
        source_video,
        removal_masks,
        render_dir,
        final_dir / "composite_full.mp4",
        args.output / "human_review_four_panel.mp4",
    )

    diagonal = float(np.hypot(width, height))
    checks = {
        "proper_rotation": bool(np.min(registration.similarity.determinant) > 0.999),
        "wrist_locked": bool(np.max(landmark_error[:, 0]) <= 0.75),
        "hand_mean_error": bool(np.mean(landmark_error) <= 0.04 * diagonal),
        "hand_p95_error": bool(np.quantile(landmark_error, 0.95) <= 0.08 * diagonal),
        "forearm_direction_p95": bool(
            render_summary.forearm_direction_error_p95_degrees <= 20.0
        ),
        "forearm_direction_max": bool(
            render_summary.forearm_direction_error_max_degrees <= 35.0
        ),
        "forearm_pose_direction_p95": bool(
            render_summary.forearm_pose_direction_error_p95_degrees <= 0.5
        ),
        "forearm_pose_direction_max": bool(
            render_summary.forearm_pose_direction_error_max_degrees <= 1.0
        ),
        "forearm_expected_visibility": bool(
            render_summary.forearm_expected_visibility_ratio >= 0.90
        ),
        "forearm_direction_coverage": bool(
            render_summary.forearm_direction_observable_frames
            >= max(8, round(0.05 * frame_count))
            and render_summary.forearm_direction_evaluated_ratio >= 0.80
        ),
        "robot_area": bool(render_summary.robot_frame_ratio_max <= 0.35),
        "residual_human_area": bool(mask_summary.residual_human_ratio_p95 <= 0.70),
        "removal_hole_area": bool(mask_summary.removal_frame_ratio_max <= 0.12),
        "inpaint_changed_removed_pixels": bool(change_summary.low_change_fraction <= 0.55),
        "inpaint_changed_each_frame": bool(
            change_summary.frame_low_change_fraction_p95 <= 0.65
        ),
    }
    report = {
        "schema_version": 1,
        "method": (
            "wrist_locked_21point_registration+bounded_appearance_mask_refinement+"
            "silhouette_sized_forearm+robot_aware_residual_inpaint"
        ),
        "side": args.side,
        "annotation": annotation_name,
        "frame_count": frame_count,
        "resolution": [width, height],
        "hand_error_pixels": {
            "mean": float(np.mean(landmark_error)),
            "p95": float(np.quantile(landmark_error, 0.95)),
            "max": float(np.max(landmark_error)),
            "wrist_max": float(np.max(landmark_error[:, 0])),
        },
        "render": asdict(render_summary),
        "human_mask_refinement": asdict(refinement_summary),
        "robot_aware_mask": asdict(mask_summary),
        "inpaint_change": asdict(change_summary),
        "checks": checks,
        "technical_pass": bool(all(checks.values())),
        "human_review": "human_review_four_panel.mp4",
    }
    (args.output / "quality_gate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not report["technical_pass"] and not args.allow_gate_failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
