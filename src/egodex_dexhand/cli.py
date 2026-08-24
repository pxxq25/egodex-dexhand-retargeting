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
    load_egodex_sequence,
    scaled_intrinsic,
)
from .inpaint import run_propainter
from .render import render_robot_sequence
from .retarget import retarget_position_sequence, save_retarget_result
from .segment import segment_hand_video


STAGES = ("prepare", "retarget", "render", "segment", "inpaint", "compose")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EgoDex to dexterous-hand video replacement without training"
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dex-assets", required=True, type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument("--propainter-root", required=True, type=Path)
    parser.add_argument("--robot", default="allegro")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--sam2-size", choices=("tiny", "small", "base_plus", "large"), default="small"
    )
    parser.add_argument("--prompt-stride", type=int, default=10)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--sam2-box-prompt",
        action="store_true",
        dest="sam2_box_prompt",
        default=True,
        help="use joint-derived boxes plus points (default)",
    )
    prompt_group.add_argument(
        "--no-sam2-box-prompt",
        action="store_false",
        dest="sam2_box_prompt",
        help="use point prompts only; may absorb the forearm in contact clips",
    )
    parser.add_argument("--start-stage", choices=STAGES, default="prepare")
    parser.add_argument("--stop-stage", choices=STAGES, default="compose")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _stage_enabled(name: str, args: argparse.Namespace) -> bool:
    index = STAGES.index(name)
    return STAGES.index(args.start_stage) <= index <= STAGES.index(args.stop_stage)


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required stage artifact is missing: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_provenance(args: argparse.Namespace) -> dict[str, object]:
    weight_names = (
        "ProPainter.pth",
        "raft-things.pth",
        "recurrent_flow_completion.pth",
    )
    weight_hashes = {
        name: _sha256((args.propainter_root / "weights" / name).resolve())
        for name in weight_names
    }
    return {
        "video": str(args.video.resolve()),
        "video_sha256": _sha256(args.video.resolve()),
        "hdf5": str(args.hdf5.resolve()),
        "hdf5_sha256": _sha256(args.hdf5.resolve()),
        "robot": args.robot,
        "hand": args.hand,
        "scale": args.scale,
        "dex_assets": str(args.dex_assets.resolve()),
        "sam2_root": str(args.sam2_root.resolve()),
        "sam2_checkpoint": str(args.sam2_checkpoint.resolve()),
        "sam2_checkpoint_sha256": _sha256(args.sam2_checkpoint.resolve()),
        "sam2_size": args.sam2_size,
        "prompt_stride": args.prompt_stride,
        "sam2_box_prompt": args.sam2_box_prompt,
        "propainter_root": str(args.propainter_root.resolve()),
        "propainter_weight_sha256": weight_hashes,
        "real_backends_only": True,
    }


def _check_resume_provenance(
    metadata: dict[str, object], current: dict[str, object]
) -> None:
    mismatches = [
        key for key, value in current.items() if metadata.get(key) != value
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise RuntimeError(
            "resume inputs/config do not match metadata.json "
            f"({joined}); restart from --start-stage prepare"
        )


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _stopped_after(name: str, args: argparse.Namespace, output: Path) -> bool:
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
    retarget_file = output / "retarget.npz"
    metadata_file = output / "metadata.json"
    output.mkdir(parents=True, exist_ok=True)

    sequence = load_egodex_sequence(args.hdf5, hand=args.hand)
    other_hand = "left" if args.hand == "right" else "right"
    negative_sequence = load_egodex_sequence(args.hdf5, hand=other_hand)
    provenance = _run_provenance(args)
    if _stage_enabled("prepare", args):
        _reset_dir(frames_dir)
        count, fps, (width, height), (source_width, source_height) = extract_video_frames(
            args.video, frames_dir, scale=args.scale
        )
        if count != sequence.frame_count:
            raise ValueError(
                f"video/HDF5 frame mismatch: {count} vs {sequence.frame_count}"
            )
        metadata = {
            "frame_count": count,
            "fps": fps,
            "width": width,
            "height": height,
            "source_width": source_width,
            "source_height": source_height,
            **provenance,
        }
        metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    _require(metadata_file)
    if _stopped_after("prepare", args, output):
        return
    metadata = json.loads(metadata_file.read_text())
    if not _stage_enabled("prepare", args):
        _check_resume_provenance(metadata, provenance)
    fps = float(metadata["fps"])
    width, height = int(metadata["width"]), int(metadata["height"])
    intrinsic = scaled_intrinsic(
        sequence.intrinsic,
        width / int(metadata["source_width"]),
        height / int(metadata["source_height"]),
    )

    if _stage_enabled("retarget", args):
        result = retarget_position_sequence(
            sequence.joints_camera_sapien,
            assets_root=args.dex_assets,
            robot=args.robot,
            hand=args.hand,
        )
        save_retarget_result(retarget_file, result)
        print(
            f"retarget: {len(result.qpos)} frames, "
            f"median objective={np.median(result.objective_values):.6f}"
        )
    _require(retarget_file)
    if _stopped_after("retarget", args, output):
        return

    if _stage_enabled("render", args):
        _reset_dir(render_dir)
        data = np.load(retarget_file)
        render_robot_sequence(
            qpos=data["qpos"],
            joint_names=tuple(str(value) for value in data["joint_names"]),
            urdf_path=str(data["urdf_path"]),
            intrinsic=intrinsic,
            width=width,
            height=height,
            output_dir=render_dir,
            fps=fps,
        )
    _require(render_dir / "robot_rgb.mp4")
    if _stopped_after("render", args, output):
        return

    if _stage_enabled("segment", args):
        _reset_dir(human_masks)
        scaled_joints = sequence.joints_camera_cv
        segment_hand_video(
            frames_dir=frames_dir,
            joints_camera_cv=scaled_joints,
            intrinsic=intrinsic,
            sam2_root=args.sam2_root,
            checkpoint=args.sam2_checkpoint,
            output_dir=human_masks,
            model_size=args.sam2_size,
            prompt_stride=args.prompt_stride,
            joint_confidence=sequence.joint_confidence,
            negative_joints_camera_cv=negative_sequence.joints_camera_cv,
            negative_joint_confidence=negative_sequence.joint_confidence,
            use_box_prompt=args.sam2_box_prompt,
        )
    _require(human_masks / "00000.png")
    if _stopped_after("segment", args, output):
        return

    inpainted = output / "inpaint" / frames_dir.name / "inpaint_out_exact.mp4"
    if _stage_enabled("inpaint", args):
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
    if _stopped_after("inpaint", args, output):
        return

    if _stage_enabled("compose", args):
        _reset_dir(output / "final")
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
