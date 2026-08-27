from __future__ import annotations

import argparse
import json
from pathlib import Path

from .provenance import sha256_file as _sha256

import cv2
import numpy as np

from .data import load_egodex_sequence, project_camera_points, scaled_intrinsic


def _numbered_paths(directory: Path, expected: int) -> list[Path]:
    paths = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))
    try:
        indices = sorted(int(path.stem) for path in paths)
    except ValueError as exc:
        raise RuntimeError(f"non-numeric frame name in {directory}") from exc
    if indices != list(range(expected)):
        raise RuntimeError(f"non-contiguous frame sequence in {directory}")
    return paths


def _video_info(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    count = 0
    shape = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        count += 1
        shape = list(frame.shape)
    capture.release()
    return {"path": str(path), "frames": count, "shape": shape, "bytes": path.stat().st_size}


def verify_run(run_dir: str | Path) -> dict[str, object]:
    run_dir = Path(run_dir).resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    expected = int(metadata["frame_count"])
    expected_shape = [int(metadata["height"]), int(metadata["width"]), 3]
    if metadata.get("real_backends_only") is not True:
        raise RuntimeError("run does not declare real_backends_only=true")
    for path_key, hash_key in (("video", "video_sha256"), ("hdf5", "hdf5_sha256")):
        source = Path(str(metadata[path_key]))
        if not source.is_file() or _sha256(source) != metadata.get(hash_key):
            raise RuntimeError(f"source provenance check failed: {path_key}")
    if metadata.get("combined_urdfs"):
        for side, raw_path in metadata["combined_urdfs"].items():
            combined_urdf = Path(str(raw_path))
            expected_hash = metadata.get("combined_urdf_sha256", {}).get(side)
            if not combined_urdf.is_file() or _sha256(combined_urdf) != expected_hash:
                raise RuntimeError(f"source provenance check failed: {side}_combined_urdf")
    elif metadata.get("combined_urdf"):
        combined_urdf = Path(str(metadata["combined_urdf"]))
        if (
            not combined_urdf.is_file()
            or _sha256(combined_urdf) != metadata.get("combined_urdf_sha256")
        ):
            raise RuntimeError("source provenance check failed: combined_urdf")
    checkpoint = Path(str(metadata["sam2_checkpoint"]))
    if (
        not checkpoint.is_file()
        or _sha256(checkpoint) != metadata.get("sam2_checkpoint_sha256")
    ):
        raise RuntimeError("SAM2 checkpoint provenance check failed")
    for name, expected_hash in metadata.get("propainter_weight_sha256", {}).items():
        weight = Path(str(metadata["propainter_root"])) / "weights" / name
        if not weight.is_file() or _sha256(weight) != expected_hash:
            raise RuntimeError(f"ProPainter checkpoint provenance check failed: {name}")

    bimanual = bool(metadata.get("bimanual"))
    result_sides = ("left", "right") if bimanual else ("target",)
    hand_reports: dict[str, dict[str, object]] = {}
    for side in result_sides:
        hand_path = (
            run_dir / f"retarget_{side}.npz"
            if bimanual
            else run_dir / "retarget.npz"
        )
        if not hand_path.is_file():
            raise RuntimeError(f"missing hand retarget result: {hand_path}")
        qpos_data = np.load(hand_path)
        qpos = qpos_data["qpos"]
        joint_names = qpos_data["joint_names"]
        joint_limits = qpos_data["joint_limits"]
        if (
            qpos.shape != (expected, len(joint_names))
            or joint_limits.shape != (qpos.shape[1], 2)
            or not np.isfinite(qpos).all()
            or np.any(qpos < joint_limits[:, 0] - 1e-5)
            or np.any(qpos > joint_limits[:, 1] + 1e-5)
        ):
            raise RuntimeError(f"invalid {side} hand qpos trajectory: {qpos.shape}")
        hand_dq = np.linalg.norm(np.diff(qpos, axis=0), axis=1)
        hand_reports[side] = {
            "qpos_shape": list(qpos.shape),
            "qpos_finite": True,
            "qpos_within_recorded_limits": True,
            "qpos_step_norm": {
                "median": float(np.median(hand_dq)),
                "max": float(np.max(hand_dq)),
            },
        }

    arm_reports: dict[str, dict[str, object]] = {}
    if metadata.get("whole_arm"):
        for side in result_sides:
            arm_path = (
                run_dir / f"arm_retarget_{side}.npz"
                if bimanual
                else run_dir / "arm_retarget.npz"
            )
            if not arm_path.is_file():
                raise RuntimeError(f"whole-arm run is missing {arm_path.name}")
            arm_data = np.load(arm_path)
            arm_qpos = arm_data["qpos"]
            arm_joint_names = arm_data["joint_names"]
            arm_joint_limits = arm_data["joint_limits"]
            position_error = arm_data["position_error"]
            orientation_error = arm_data["orientation_error_degrees"]
            base_translation = arm_data["base_translation_world"]
            base_rotation = arm_data["base_rotation_world"]
            camera_poses = arm_data["world_from_camera"]
            if (
                arm_qpos.shape != (expected, len(arm_joint_names))
                or arm_joint_limits.shape != (arm_qpos.shape[1], 2)
                or not np.isfinite(arm_qpos).all()
                or np.any(arm_qpos < arm_joint_limits[:, 0] - 1e-5)
                or np.any(arm_qpos > arm_joint_limits[:, 1] + 1e-5)
            ):
                raise RuntimeError(f"invalid {side} arm qpos trajectory: {arm_qpos.shape}")
            if (
                position_error.shape != (expected,)
                or orientation_error.shape != (expected,)
                or not np.isfinite(position_error).all()
                or not np.isfinite(orientation_error).all()
                or np.any(position_error < 0)
                or np.any(orientation_error < 0)
                or float(np.max(position_error)) > 0.01
                or float(np.max(orientation_error)) > 5.0
            ):
                raise RuntimeError(f"invalid or excessive {side} arm IK residual")
            if (
                base_translation.shape != (3,)
                or base_rotation.shape != (3, 3)
                or camera_poses.shape != (expected, 4, 4)
                or not np.isfinite(base_translation).all()
                or not np.isfinite(base_rotation).all()
                or not np.isfinite(camera_poses).all()
                or not np.allclose(base_rotation.T @ base_rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(base_rotation), 1.0, atol=1e-4)
            ):
                raise RuntimeError(f"invalid {side} arm base or camera transforms")
            derived_urdf = Path(str(arm_data["urdf_path"].item()))
            if not derived_urdf.is_file():
                raise RuntimeError(f"derived arm-hand URDF is missing: {derived_urdf}")
            arm_dq = np.linalg.norm(np.diff(arm_qpos, axis=0), axis=1)
            arm_reports[side] = {
                "qpos_shape": list(arm_qpos.shape),
                "qpos_finite": True,
                "qpos_within_recorded_limits": True,
                "position_error_mm": {
                    "median": float(np.median(position_error) * 1000.0),
                    "max": float(np.max(position_error) * 1000.0),
                },
                "orientation_error_degrees": {
                    "median": float(np.median(orientation_error)),
                    "max": float(np.max(orientation_error)),
                },
                "qpos_step_norm": {
                    "median": float(np.median(arm_dq)),
                    "max": float(np.max(arm_dq)),
                },
                "target_adjusted_frames": [
                    int(value) for value in arm_data["target_adjusted_frames"]
                ],
                "derived_urdf": str(derived_urdf),
            }
    hand_report: dict[str, object] = (
        hand_reports if bimanual else hand_reports["target"]
    )
    arm_report: dict[str, object] | None = (
        arm_reports if bimanual else arm_reports.get("target")
    )

    directories = {
        "source_frames": run_dir / "frames",
        "robot_rgb": run_dir / "render/robot_rgb",
        "robot_mask": run_dir / "render/robot_mask",
        "human_mask": run_dir / "human_mask",
        "inpainted_frames": run_dir / "inpaint/frames/frames",
    }
    if metadata.get("whole_arm"):
        directories.update(
            {
                "robot_arm_mask": run_dir / "render/arm_mask",
                "robot_hand_mask": run_dir / "render/hand_mask",
            }
        )
    if bimanual:
        for side in ("left", "right"):
            directories.update(
                {
                    f"human_mask_{side}": run_dir / f"human_mask_{side}",
                    f"robot_mask_{side}": run_dir / f"render/{side}_robot_mask",
                    f"robot_arm_mask_{side}": run_dir / f"render/{side}_arm_mask",
                    f"robot_hand_mask_{side}": run_dir / f"render/{side}_hand_mask",
                }
            )
    frame_counts = {}
    for name, directory in directories.items():
        paths = _numbered_paths(directory, expected)
        frame_counts[name] = len(paths)

    mask_areas = []
    border_touch_frames = []
    for path in sorted((run_dir / "human_mask").glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"could not decode {path}")
        binary = mask > 127
        mask_areas.append(int(np.count_nonzero(binary)))
        if binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any():
            border_touch_frames.append(int(path.stem))
    if min(mask_areas) < 64:
        raise RuntimeError("at least one human mask is empty or implausibly small")

    primary_hand = "right" if bimanual else str(metadata["hand"])
    intrinsic = scaled_intrinsic(
        load_egodex_sequence(metadata["hdf5"], primary_hand).intrinsic,
        int(metadata["width"]) / int(metadata["source_width"]),
        int(metadata["height"]) / int(metadata["source_height"]),
    )
    point_coverage: dict[str, list[float]] = {}
    for hand in ("left", "right"):
        sequence = load_egodex_sequence(metadata["hdf5"], hand)
        projected = project_camera_points(sequence.joints_camera_cv, intrinsic)
        coverage = []
        for frame_index, points in enumerate(projected):
            mask = cv2.imread(
                str(run_dir / "human_mask" / f"{frame_index:05d}.png"),
                cv2.IMREAD_GRAYSCALE,
            ) > 127
            valid = np.isfinite(points).all(axis=1)
            valid &= (points[:, 0] >= 0) & (points[:, 0] < int(metadata["width"]))
            valid &= (points[:, 1] >= 0) & (points[:, 1] < int(metadata["height"]))
            pixel = np.floor(points[valid]).astype(np.int64)
            coverage.append(float(mask[pixel[:, 1], pixel[:, 0]].mean()) if len(pixel) else 0.0)
        point_coverage[str(hand)] = coverage

    if bimanual:
        low_target_frames: object = {
            side: [
                index
                for index, value in enumerate(point_coverage[side])
                if value < 0.4
            ]
            for side in ("left", "right")
        }
        leakage_frames: object = None
        target_coverage_report: object = {
            side: {
                "min": min(point_coverage[side]),
                "median": float(np.median(point_coverage[side])),
                "max": max(point_coverage[side]),
            }
            for side in ("left", "right")
        }
        non_target_leakage_report: object = None
    else:
        target_coverage = point_coverage[primary_hand]
        other_hand = "left" if primary_hand == "right" else "right"
        non_target_leakage = point_coverage[other_hand]
        low_target_frames = [
            index for index, value in enumerate(target_coverage) if value < 0.4
        ]
        leakage_frames = [
            index for index, value in enumerate(non_target_leakage) if value > 0.2
        ]
        target_coverage_report = {
            "min": min(target_coverage),
            "median": float(np.median(target_coverage)),
            "max": max(target_coverage),
        }
        non_target_leakage_report = {
            "min": min(non_target_leakage),
            "median": float(np.median(non_target_leakage)),
            "max": max(non_target_leakage),
        }
    area_jump_frames = []
    for index in range(1, len(mask_areas)):
        ratio = max(mask_areas[index], mask_areas[index - 1]) / max(
            1, min(mask_areas[index], mask_areas[index - 1])
        )
        if ratio > 1.8:
            area_jump_frames.append(index)

    robot_mask_areas = []
    arm_mask_areas = []
    hand_mask_areas = []
    side_robot_areas: dict[str, list[int]] = {"left": [], "right": []}
    side_arm_areas: dict[str, list[int]] = {"left": [], "right": []}
    side_hand_areas: dict[str, list[int]] = {"left": [], "right": []}
    for frame_index, path in enumerate(
        sorted((run_dir / "render/robot_mask").glob("*.png"))
    ):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"could not decode {path}")
        robot_binary = mask > 127
        robot_mask_areas.append(int(np.count_nonzero(robot_binary)))
        if metadata.get("whole_arm"):
            arm_mask = cv2.imread(
                str(run_dir / "render/arm_mask" / f"{frame_index:05d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            hand_mask = cv2.imread(
                str(run_dir / "render/hand_mask" / f"{frame_index:05d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if arm_mask is None or hand_mask is None:
                raise RuntimeError(f"could not decode split robot masks at {frame_index}")
            arm_binary = arm_mask > 127
            hand_binary = hand_mask > 127
            if np.any(arm_binary & hand_binary) or not np.array_equal(
                robot_binary, arm_binary | hand_binary
            ):
                raise RuntimeError(f"inconsistent split robot masks at {frame_index}")
            arm_mask_areas.append(int(np.count_nonzero(arm_binary)))
            hand_mask_areas.append(int(np.count_nonzero(hand_binary)))
            if bimanual:
                side_binaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                for side in ("left", "right"):
                    decoded = []
                    for kind in ("robot", "arm", "hand"):
                        side_mask = cv2.imread(
                            str(
                                run_dir
                                / "render"
                                / f"{side}_{kind}_mask"
                                / f"{frame_index:05d}.png"
                            ),
                            cv2.IMREAD_GRAYSCALE,
                        )
                        if side_mask is None:
                            raise RuntimeError(
                                f"could not decode {side} {kind} mask at {frame_index}"
                            )
                        decoded.append(side_mask > 127)
                    side_robot, side_arm, side_hand = decoded
                    if np.any(side_arm & side_hand) or not np.array_equal(
                        side_robot, side_arm | side_hand
                    ):
                        raise RuntimeError(
                            f"inconsistent {side} split masks at {frame_index}"
                        )
                    side_binaries[side] = (side_robot, side_arm, side_hand)
                    side_robot_areas[side].append(int(np.count_nonzero(side_robot)))
                    side_arm_areas[side].append(int(np.count_nonzero(side_arm)))
                    side_hand_areas[side].append(int(np.count_nonzero(side_hand)))
                if (
                    np.any(side_binaries["left"][0] & side_binaries["right"][0])
                    or not np.array_equal(
                        robot_binary,
                        side_binaries["left"][0] | side_binaries["right"][0],
                    )
                    or not np.array_equal(
                        arm_binary,
                        side_binaries["left"][1] | side_binaries["right"][1],
                    )
                    or not np.array_equal(
                        hand_binary,
                        side_binaries["left"][2] | side_binaries["right"][2],
                    )
                ):
                    raise RuntimeError(
                        f"inconsistent bimanual renderer masks at {frame_index}"
                    )
    if not any(area >= 16 for area in robot_mask_areas):
        raise RuntimeError("the robot is absent from the complete interval")
    if metadata.get("whole_arm") and (
        not any(area >= 16 for area in hand_mask_areas)
        or not any(area >= 16 for area in arm_mask_areas)
    ):
        raise RuntimeError("split arm/hand masks fail visibility validation")
    if bimanual:
        for side in ("left", "right"):
            if (
                not any(area >= 32 for area in side_robot_areas[side])
                or not any(area >= 16 for area in side_hand_areas[side])
                or not any(area >= 16 for area in side_arm_areas[side])
            ):
                raise RuntimeError(f"{side} renderer masks fail visibility validation")

    video_paths = (
        run_dir / "render/robot_rgb.mp4",
        run_dir / "inpaint/frames/inpaint_out_exact.mp4",
        run_dir / "final/composite_full.mp4",
        run_dir / "final/composite_conservative.mp4",
        run_dir / "final/qa_side_by_side.mp4",
    )
    videos = [_video_info(path) for path in video_paths]
    for video in videos[:-1]:
        if video["frames"] != expected or video["shape"] != expected_shape:
            raise RuntimeError(f"video validation failed: {video}")
    qa_shape = [expected_shape[0], expected_shape[1] * 4, 3]
    if videos[-1]["frames"] != expected or videos[-1]["shape"] != qa_shape:
        raise RuntimeError(f"QA video validation failed: {videos[-1]}")

    report = {
        "status": "structural_validation_passed",
        "backend_policy": "pipeline has no mock or fallback adapters",
        "metadata": metadata,
        "qpos_shape": (
            {side: hand_reports[side]["qpos_shape"] for side in ("left", "right")}
            if bimanual
            else hand_reports["target"]["qpos_shape"]
        ),
        "qpos_finite": True,
        "qpos_within_recorded_limits": True,
        "hand_retarget": hand_report,
        "arm_retarget": arm_report,
        "frame_counts": frame_counts,
        "human_mask_area": {
            "min": min(mask_areas),
            "median": float(np.median(mask_areas)),
            "max": max(mask_areas),
        },
        "robot_mask_area": {
            "min": min(robot_mask_areas),
            "median": float(np.median(robot_mask_areas)),
            "max": max(robot_mask_areas),
        },
        "split_robot_mask_area": (
            {
                "arm": {
                    "min": min(arm_mask_areas),
                    "median": float(np.median(arm_mask_areas)),
                    "max": max(arm_mask_areas),
                    "visible_frames": sum(area >= 16 for area in arm_mask_areas),
                },
                "hand": {
                    "min": min(hand_mask_areas),
                    "median": float(np.median(hand_mask_areas)),
                    "max": max(hand_mask_areas),
                },
            }
            if metadata.get("whole_arm")
            else None
        ),
        "side_robot_mask_area": (
            {
                side: {
                    "robot": {
                        "min": min(side_robot_areas[side]),
                        "median": float(np.median(side_robot_areas[side])),
                        "max": max(side_robot_areas[side]),
                    },
                    "arm": {
                        "min": min(side_arm_areas[side]),
                        "median": float(np.median(side_arm_areas[side])),
                        "max": max(side_arm_areas[side]),
                        "visible_frames": sum(
                            area >= 16 for area in side_arm_areas[side]
                        ),
                    },
                    "hand": {
                        "min": min(side_hand_areas[side]),
                        "median": float(np.median(side_hand_areas[side])),
                        "max": max(side_hand_areas[side]),
                    },
                }
                for side in ("left", "right")
            }
            if bimanual
            else None
        ),
        "semantic_review": {
            "required": True,
            "reason": "shape/count checks cannot prove semantic mask correctness",
            "human_mask_border_touch_frames": border_touch_frames,
            "low_target_joint_coverage_frames": low_target_frames,
            "non_target_joint_leakage_frames": leakage_frames,
            "large_mask_area_jump_frames": area_jump_frames,
            "target_joint_coverage": target_coverage_report,
            "non_target_joint_leakage": non_target_leakage_report,
        },
        "videos": videos,
    }
    (run_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify every artifact in a completed run")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
