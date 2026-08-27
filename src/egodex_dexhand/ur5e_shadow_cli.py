from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .chunk_continuity import (
    last_state,
    load_chunk_continuity_state,
    save_chunk_continuity_state,
    smooth_wrapped_joint_boundary,
    trailing_context,
)

from .compose import composite_videos
from .data import (
    extract_video_frames,
    fuse_hand_visibility,
    load_egodex_arm_sequence,
    load_egodex_sequence,
    projected_hand_visibility,
    scaled_intrinsic,
)
from .inpaint import run_propainter
from .provenance import sha256_file as _sha256
from .render import render_ur5e_shadow_sequence
from .retarget import retarget_position_sequence, save_retarget_result
from .segment import segment_arm_hand_video
from .shadow_ur5e import (
    extract_shadow_forearm_targets,
    prepare_ur5e_shadow_urdf,
    save_ur5e_arm_result,
    solve_ur5e_arm_sequence,
)
from .temporal_smoothing import (
    add_temporal_smoothing_arguments,
    smooth_se3_trajectory,
    smooth_shadow_qpos,
    temporal_smoothing_config_from_args,
)


STAGES = ("prepare", "retarget", "render", "segment", "inpaint", "compose")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-free EgoDex to integrated UR5e + Shadow Hand video"
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument(
        "--visibility-hdf5",
        type=Path,
        help=(
            "optional unaligned world-coordinate trajectory used only for "
            "frame-level camera-frustum visibility gating"
        ),
    )
    parser.add_argument(
        "--mask-hdf5",
        type=Path,
        help="optional image-aligned landmarks used only for SAM2 masking",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dex-assets", required=True, type=Path)
    parser.add_argument("--combined-urdf", required=True, type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument("--propainter-root", required=True, type=Path)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--sam2-size", choices=("tiny", "small", "base_plus", "large"), default="small"
    )
    parser.add_argument("--prompt-stride", type=int, default=1)
    parser.add_argument(
        "--render-device",
        required=True,
        help="explicit SAPIEN Vulkan device, for example cuda:2",
    )
    parser.add_argument(
        "--hide-arm-visual-link",
        action="append",
        default=[],
        help="UR5e link whose visual mesh is omitted (repeatable)",
    )
    parser.add_argument(
        "--allow-hidden-arm",
        action="store_true",
        help=(
            "accept a visible Shadow hand when intentionally hidden proximal "
            "arm visuals produce no arm-mask pixels"
        ),
    )
    parser.add_argument(
        "--arm-q-reference",
        type=float,
        nargs=6,
        help="optional six-joint UR5e reference posture used to choose the IK branch",
    )
    parser.add_argument("--start-stage", choices=STAGES, default="prepare")
    parser.add_argument("--stop-stage", choices=STAGES, default="compose")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--initial-continuity-state", type=Path)
    parser.add_argument("--write-continuity-state", type=Path)
    parser.add_argument("--continuity-source-frame", type=int, default=-1)
    parser.add_argument("--continuity-margin-frames", type=int, default=12)
    add_temporal_smoothing_arguments(parser)
    return parser.parse_args()


def _enabled(name: str, args: argparse.Namespace) -> bool:
    index = STAGES.index(name)
    return STAGES.index(args.start_stage) <= index <= STAGES.index(args.stop_stage)


def _provenance(args: argparse.Namespace) -> dict[str, object]:
    smoothing_config = temporal_smoothing_config_from_args(args)
    weight_names = (
        "ProPainter.pth",
        "raft-things.pth",
        "recurrent_flow_completion.pth",
    )
    mask_hdf5 = (args.mask_hdf5 or args.hdf5).resolve()
    visibility_hdf5 = (args.visibility_hdf5 or args.hdf5).resolve()
    return {
        "video": str(args.video.resolve()),
        "video_sha256": _sha256(args.video.resolve()),
        "hdf5": str(args.hdf5.resolve()),
        "hdf5_sha256": _sha256(args.hdf5.resolve()),
        "mask_hdf5": str(mask_hdf5),
        "mask_hdf5_sha256": _sha256(mask_hdf5),
        "visibility_hdf5": str(visibility_hdf5),
        "visibility_hdf5_sha256": _sha256(visibility_hdf5),
        "robot": "shadow",
        "arm": "ur5e",
        "embodiment": f"ur5e+shadow_{args.hand}",
        "whole_arm": True,
        "integrated_articulation": True,
        "hand": args.hand,
        "scale": args.scale,
        "dex_assets": str(args.dex_assets.resolve()),
        "combined_urdf": str(args.combined_urdf.resolve()),
        "combined_urdf_sha256": _sha256(args.combined_urdf.resolve()),
        "sam2_root": str(args.sam2_root.resolve()),
        "sam2_checkpoint": str(args.sam2_checkpoint.resolve()),
        "sam2_checkpoint_sha256": _sha256(args.sam2_checkpoint.resolve()),
        "sam2_size": args.sam2_size,
        "prompt_stride": args.prompt_stride,
        "propainter_root": str(args.propainter_root.resolve()),
        "propainter_weight_sha256": {
            name: _sha256((args.propainter_root / "weights" / name).resolve())
            for name in weight_names
        },
        "real_backends_only": True,
        "contact_aware_occlusion": False,
        "shadow_thumb_collision_path_repaired": True,
        "shadow_optimizer_preroll": 10,
        "temporal_smoothing": smoothing_config.to_dict(),
        "shadow_root_smoothing": "preserve_dummy_q_and_smooth_forearm_se3",
        "hidden_arm_visual_links": list(args.hide_arm_visual_link),
        "require_arm_visibility": not args.allow_hidden_arm,
        "arm_q_reference": args.arm_q_reference,
    }


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required stage artifact is missing: {path}")


def _stop(name: str, args: argparse.Namespace, output: Path) -> bool:
    if args.stop_stage == name:
        print(f"stopped after {name}: {output}")
        return True
    return False


def main() -> None:
    args = _parse_args()
    continuity_state = load_chunk_continuity_state(args.initial_continuity_state)
    smoothing_config = temporal_smoothing_config_from_args(args)
    if STAGES.index(args.start_stage) > STAGES.index(args.stop_stage):
        raise ValueError("--start-stage must not come after --stop-stage")
    output = args.output.resolve()
    if (
        args.start_stage == "prepare"
        and output.exists()
        and any(output.iterdir())
        and not args.force
    ):
        raise FileExistsError(
            f"output directory is not empty: {output}; pass --force to replace it "
            "or resume with --start-stage"
        )
    frames_dir = output / "frames"
    render_dir = output / "render"
    human_masks = output / "human_mask"
    hand_file = output / "retarget.npz"
    arm_file = output / "arm_retarget.npz"
    derived_urdf = output / "derived" / "ur5e_shadow_patched.urdf"
    metadata_file = output / "metadata.json"
    output.mkdir(parents=True, exist_ok=True)

    hand_sequence = load_egodex_sequence(args.hdf5, hand=args.hand)
    arm_sequence = load_egodex_arm_sequence(args.hdf5, hand=args.hand)
    mask_hdf5 = args.mask_hdf5 or args.hdf5
    mask_hand_sequence = load_egodex_sequence(mask_hdf5, hand=args.hand)
    mask_arm_sequence = load_egodex_arm_sequence(mask_hdf5, hand=args.hand)
    visibility_sequence = load_egodex_sequence(
        args.visibility_hdf5 or args.hdf5, hand=args.hand
    )
    other_hand = "left" if args.hand == "right" else "right"
    negative_sequence = load_egodex_sequence(args.hdf5, hand=other_hand)
    provenance = _provenance(args)

    if _enabled("prepare", args):
        _reset(frames_dir)
        count, fps, (width, height), (source_width, source_height) = extract_video_frames(
            args.video, frames_dir, scale=args.scale
        )
        if (
            count != hand_sequence.frame_count
            or count != arm_sequence.frame_count
            or count != mask_hand_sequence.frame_count
            or count != mask_arm_sequence.frame_count
            or count != visibility_sequence.frame_count
        ):
            raise ValueError("video, hand, and arm frame counts do not match")
        metadata_file.write_text(
            json.dumps(
                {
                    "frame_count": count,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "source_width": source_width,
                    "source_height": source_height,
                    **provenance,
                },
                indent=2,
            )
            + "\n"
        )
    _require(metadata_file)
    if _stop("prepare", args, output):
        return
    metadata = json.loads(metadata_file.read_text())
    if not _enabled("prepare", args):
        mismatches = [key for key, value in provenance.items() if metadata.get(key) != value]
        if mismatches:
            raise RuntimeError(
                "resume inputs/config do not match metadata.json: " + ", ".join(mismatches)
            )
    fps = float(metadata["fps"])
    width, height = int(metadata["width"]), int(metadata["height"])
    intrinsic = scaled_intrinsic(
        hand_sequence.intrinsic,
        width / int(metadata["source_width"]),
        height / int(metadata["source_height"]),
    )
    mask_intrinsic = scaled_intrinsic(
        mask_hand_sequence.intrinsic,
        width / int(metadata["source_width"]),
        height / int(metadata["source_height"]),
    )
    require_robot_visibility = bool(
        projected_hand_visibility(
            visibility_sequence.joints_camera_cv,
            scaled_intrinsic(
                visibility_sequence.intrinsic,
                width / int(metadata["source_width"]),
                height / int(metadata["source_height"]),
            ),
            width,
            height,
            joint_confidence=visibility_sequence.joint_confidence,
        ).any()
    )

    if _enabled("retarget", args):
        hand_result = retarget_position_sequence(
            hand_sequence.joints_camera_sapien,
            assets_root=args.dex_assets,
            robot="shadow",
            hand=args.hand,
            preroll=10,
            initial_qpos=(
                None if continuity_state is None else continuity_state.hand_qpos
                if continuity_state.hand_qpos.ndim == 1
                else last_state(continuity_state.hand_qpos)
            ),
        )
        hand_result = replace(
            hand_result,
            qpos=smooth_shadow_qpos(
                hand_result.qpos,
                hand_result.joint_names,
                hand_result.joint_limits,
                fps=fps,
                config=smoothing_config,
            ),
        )
        if continuity_state is not None:
            hand_result = replace(
                hand_result,
                qpos=smooth_wrapped_joint_boundary(
                    continuity_state.hand_qpos,
                    hand_result.qpos,
                    margin_frames=args.continuity_margin_frames,
                    lower=hand_result.joint_limits[:, 0],
                    upper=hand_result.joint_limits[:, 1],
                ),
            )
        save_retarget_result(hand_file, hand_result)
        prepare_ur5e_shadow_urdf(args.combined_urdf, derived_urdf)
        targets = extract_shadow_forearm_targets(
            hand_result.qpos,
            hand_result.joint_names,
            hand_result.urdf_path,
            arm_sequence.world_from_camera,
        )
        target_positions, target_rotations = smooth_se3_trajectory(
            targets.position_world,
            targets.rotation_world,
            fps=fps,
            config=smoothing_config,
        )
        targets = replace(
            targets,
            position_world=target_positions,
            rotation_world=target_rotations,
        )
        arm_result = solve_ur5e_arm_sequence(
            targets=targets,
            human_arm_joints_world=arm_sequence.joints_world,
            world_from_camera=arm_sequence.world_from_camera,
            combined_urdf=derived_urdf,
            q_reference=args.arm_q_reference,
            initial_qpos=(
                None if continuity_state is None else last_state(continuity_state.arm_qpos)
            ),
            base_translation_world=(
                None
                if continuity_state is None
                else continuity_state.base_translation_world
            ),
            base_rotation_world=(
                None
                if continuity_state is None
                else continuity_state.base_rotation_world
            ),
        )
        if continuity_state is not None:
            arm_result = replace(
                arm_result,
                qpos=smooth_wrapped_joint_boundary(
                    continuity_state.arm_qpos,
                    arm_result.qpos,
                    margin_frames=args.continuity_margin_frames,
                    lower=arm_result.joint_limits[:, 0],
                    upper=arm_result.joint_limits[:, 1],
                ),
            )
        save_ur5e_arm_result(arm_file, arm_result)
        if args.write_continuity_state is not None:
            save_chunk_continuity_state(
                args.write_continuity_state,
                hand_qpos=trailing_context(
                    None if continuity_state is None else continuity_state.hand_qpos,
                    hand_result.qpos,
                    args.continuity_margin_frames,
                ),
                arm_qpos=trailing_context(
                    None if continuity_state is None else continuity_state.arm_qpos,
                    arm_result.qpos,
                    args.continuity_margin_frames,
                ),
                base_translation_world=arm_result.base_translation_world,
                base_rotation_world=arm_result.base_rotation_world,
                source_frame=args.continuity_source_frame,
            )
        print(
            f"retargeted {len(hand_result.qpos)} frames; UR5e forearm error "
            f"median={np.median(arm_result.position_error) * 1000:.2f} mm, "
            f"max={np.max(arm_result.position_error) * 1000:.2f} mm; "
            f"orientation median={np.median(arm_result.orientation_error_degrees):.2f} deg"
        )
    _require(hand_file)
    _require(arm_file)
    _require(derived_urdf)
    if _stop("retarget", args, output):
        return

    if _enabled("render", args):
        _reset(render_dir)
        hand_data = np.load(hand_file)
        arm_data = np.load(arm_file)
        render_ur5e_shadow_sequence(
            arm_qpos=arm_data["qpos"],
            arm_joint_names=tuple(str(value) for value in arm_data["joint_names"]),
            hand_qpos=hand_data["qpos"],
            hand_joint_names=tuple(str(value) for value in hand_data["joint_names"]),
            combined_urdf_path=str(arm_data["urdf_path"]),
            base_translation_world=arm_data["base_translation_world"],
            base_rotation_world=arm_data["base_rotation_world"],
            world_from_camera=arm_data["world_from_camera"],
            intrinsic=intrinsic,
            width=width,
            height=height,
            output_dir=render_dir,
            fps=fps,
            hidden_arm_visual_links=tuple(args.hide_arm_visual_link),
            render_device=args.render_device,
            require_robot_visibility=require_robot_visibility,
            require_arm_visibility=not args.allow_hidden_arm,
        )
    _require(render_dir / "robot_rgb.mp4")
    if _stop("render", args, output):
        return

    if _enabled("segment", args):
        _reset(human_masks)
        segment_arm_hand_video(
            frames_dir=frames_dir,
            hand_joints_camera_cv=mask_hand_sequence.joints_camera_cv,
            arm_joints_camera_cv=mask_arm_sequence.joints_camera_cv,
            intrinsic=mask_intrinsic,
            sam2_root=args.sam2_root,
            checkpoint=args.sam2_checkpoint,
            output_dir=human_masks,
            model_size=args.sam2_size,
            prompt_stride=args.prompt_stride,
            hand_joint_confidence=mask_hand_sequence.joint_confidence,
            negative_joints_camera_cv=negative_sequence.joints_camera_cv,
            negative_joint_confidence=negative_sequence.joint_confidence,
        )
    _require(human_masks / "00000.png")
    if _stop("segment", args, output):
        return

    inpainted = output / "inpaint" / frames_dir.name / "inpaint_out_exact.mp4"
    if _enabled("inpaint", args):
        inpaint_stage_dir = output / "inpaint" / frames_dir.name
        if inpaint_stage_dir.exists():
            shutil.rmtree(inpaint_stage_dir)
        inpainted = run_propainter(
            frames_dir=frames_dir,
            masks_dir=human_masks,
            propainter_root=args.propainter_root,
            output_dir=output / "inpaint",
            python_executable=sys.executable,
            fps=fps,
            fp16=True,
        )
    _require(inpainted)
    if _stop("inpaint", args, output):
        return

    if _enabled("compose", args):
        _reset(output / "final")
        composite_videos(
            source_video=args.video,
            inpainted_video=inpainted,
            robot_rgb_dir=render_dir / "robot_rgb",
            robot_mask_dir=render_dir / "robot_mask",
            human_mask_dir=human_masks,
            output_dir=output / "final",
            human_visibility_by_side={
                args.hand: fuse_hand_visibility(
                    projected_hand_visibility(
                        visibility_sequence.joints_camera_cv,
                        scaled_intrinsic(
                            visibility_sequence.intrinsic,
                            width / int(metadata["source_width"]),
                            height / int(metadata["source_height"]),
                        ),
                        width,
                        height,
                        joint_confidence=visibility_sequence.joint_confidence,
                    ),
                    (
                        mask_hand_sequence.joint_confidence
                        if args.mask_hdf5 is not None
                        else None
                    ),
                )
            },
            robot_mask_dirs_by_side={args.hand: render_dir / "robot_mask"},
        )
    print(f"complete: {output}")


if __name__ == "__main__":
    main()
