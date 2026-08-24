from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from .compose import composite_videos
from .data import (
    extract_video_frames,
    load_egodex_arm_sequence,
    load_egodex_sequence,
    scaled_intrinsic,
)
from .inpaint import run_propainter
from .render import render_arm_hand_sequence
from .retarget import retarget_position_sequence, save_retarget_result
from .segment import segment_arm_hand_video
from .whole_arm import (
    PANDA_DISTAL_SCALE,
    prepare_panda_arm_urdf,
    save_panda_arm_result,
    solve_panda_arm_sequence,
)


STAGES = ("prepare", "retarget", "render", "segment", "inpaint", "compose")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EgoDex to Panda + dexterous-hand video replacement without training"
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dex-assets", required=True, type=Path)
    parser.add_argument("--panda-asset", required=True, type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument("--propainter-root", required=True, type=Path)
    parser.add_argument("--robot", default="allegro")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--sam2-size", choices=("tiny", "small", "base_plus", "large"), default="small"
    )
    parser.add_argument("--prompt-stride", type=int, default=5)
    parser.add_argument("--start-stage", choices=STAGES, default="prepare")
    parser.add_argument("--stop-stage", choices=STAGES, default="compose")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _enabled(name: str, args: argparse.Namespace) -> bool:
    index = STAGES.index(name)
    return STAGES.index(args.start_stage) <= index <= STAGES.index(args.stop_stage)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _panda_source(path: Path) -> Path:
    path = path.resolve()
    return path if path.is_file() else path / "panda.urdf"


def _provenance(args: argparse.Namespace) -> dict[str, object]:
    panda_source = _panda_source(args.panda_asset)
    weight_names = (
        "ProPainter.pth",
        "raft-things.pth",
        "recurrent_flow_completion.pth",
    )
    return {
        "video": str(args.video.resolve()),
        "video_sha256": _sha256(args.video.resolve()),
        "hdf5": str(args.hdf5.resolve()),
        "hdf5_sha256": _sha256(args.hdf5.resolve()),
        "robot": args.robot,
        "arm": "franka_panda",
        "embodiment": f"franka_panda+{args.robot}_{args.hand}",
        "whole_arm": True,
        "hand": args.hand,
        "scale": args.scale,
        "dex_assets": str(args.dex_assets.resolve()),
        "panda_asset": str(args.panda_asset.resolve()),
        "panda_urdf_sha256": _sha256(panda_source),
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
        "panda_distal_scale": PANDA_DISTAL_SCALE,
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
    if STAGES.index(args.start_stage) > STAGES.index(args.stop_stage):
        raise ValueError("--start-stage must not come after --stop-stage")
    output = args.output.resolve()
    frames_dir = output / "frames"
    render_dir = output / "render"
    human_masks = output / "human_mask"
    hand_file = output / "retarget.npz"
    arm_file = output / "arm_retarget.npz"
    derived_arm_urdf = output / "derived" / "panda_arm_only.urdf"
    metadata_file = output / "metadata.json"
    output.mkdir(parents=True, exist_ok=True)

    hand_sequence = load_egodex_sequence(args.hdf5, hand=args.hand)
    arm_sequence = load_egodex_arm_sequence(args.hdf5, hand=args.hand)
    other_hand = "left" if args.hand == "right" else "right"
    negative_sequence = load_egodex_sequence(args.hdf5, hand=other_hand)
    provenance = _provenance(args)

    if _enabled("prepare", args):
        _reset(frames_dir)
        count, fps, (width, height), (source_width, source_height) = extract_video_frames(
            args.video, frames_dir, scale=args.scale
        )
        if count != hand_sequence.frame_count or count != arm_sequence.frame_count:
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

    if _enabled("retarget", args):
        hand_result = retarget_position_sequence(
            hand_sequence.joints_camera_sapien,
            assets_root=args.dex_assets,
            robot=args.robot,
            hand=args.hand,
        )
        save_retarget_result(hand_file, hand_result)
        prepare_panda_arm_urdf(args.panda_asset, derived_arm_urdf)
        arm_result = solve_panda_arm_sequence(
            arm_sequence.joints_world,
            arm_sequence.world_from_camera,
            derived_arm_urdf,
        )
        save_panda_arm_result(arm_file, arm_result)
        print(
            f"retargeted {len(hand_result.qpos)} frames; Panda wrist error "
            f"median={np.median(arm_result.wrist_error) * 1000:.2f} mm, "
            f"max={np.max(arm_result.wrist_error) * 1000:.2f} mm"
        )
    _require(hand_file)
    _require(arm_file)
    _require(derived_arm_urdf)
    if _stop("retarget", args, output):
        return

    if _enabled("render", args):
        _reset(render_dir)
        hand_data = np.load(hand_file)
        arm_data = np.load(arm_file)
        render_arm_hand_sequence(
            hand_qpos=hand_data["qpos"],
            hand_joint_names=tuple(str(value) for value in hand_data["joint_names"]),
            hand_urdf_path=str(hand_data["urdf_path"]),
            arm_qpos=arm_data["qpos"],
            arm_joint_names=tuple(str(value) for value in arm_data["joint_names"]),
            arm_urdf_path=str(arm_data["urdf_path"]),
            base_translation_world=arm_data["base_translation_world"],
            base_rotation_world=arm_data["base_rotation_world"],
            world_from_camera=arm_data["world_from_camera"],
            intrinsic=intrinsic,
            width=width,
            height=height,
            output_dir=render_dir,
            fps=fps,
        )
    _require(render_dir / "robot_rgb.mp4")
    if _stop("render", args, output):
        return

    if _enabled("segment", args):
        _reset(human_masks)
        segment_arm_hand_video(
            frames_dir=frames_dir,
            hand_joints_camera_cv=hand_sequence.joints_camera_cv,
            arm_joints_camera_cv=arm_sequence.joints_camera_cv,
            intrinsic=intrinsic,
            sam2_root=args.sam2_root,
            checkpoint=args.sam2_checkpoint,
            output_dir=human_masks,
            model_size=args.sam2_size,
            prompt_stride=args.prompt_stride,
            hand_joint_confidence=hand_sequence.joint_confidence,
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
        )
    print(f"complete: {output}")


if __name__ == "__main__":
    main()
