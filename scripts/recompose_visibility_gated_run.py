#!/usr/bin/env python3
"""Recompose an adaptive run with projected-human/rendered-robot gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from egodex_dexhand.compose import composite_videos
from egodex_dexhand.data import (
    fuse_hand_visibility,
    load_egodex_sequence,
    projected_hand_visibility,
    scaled_intrinsic,
)

from run_egoquest_adaptive_recording import assemble_recording, command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--output-run", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int)
    return parser.parse_args()


def recompose_segment(
    source_segment: Path,
    candidate: Path,
    output_segment: Path,
    mode: str,
) -> Path:
    metadata = json.loads((source_segment / "metadata.json").read_text())
    width, height = int(metadata["width"]), int(metadata["height"])
    sides = ("left", "right") if mode == "both" else (mode,)
    visibility = {}
    robot_masks = {}
    for side in sides:
        sequence = load_egodex_sequence(candidate / "annotations.hdf5", hand=side)
        intrinsic = scaled_intrinsic(
            sequence.intrinsic,
            width / int(metadata["source_width"]),
            height / int(metadata["source_height"]),
        )
        projected = projected_hand_visibility(
            sequence.joints_camera_cv,
            intrinsic,
            width,
            height,
            joint_confidence=sequence.joint_confidence,
        )
        rgb_sequence = load_egodex_sequence(
            candidate / "rgb_mask_annotations.hdf5", hand=side
        )
        visibility[side] = fuse_hand_visibility(
            projected, rgb_sequence.joint_confidence
        )
        robot_masks[side] = (
            source_segment / "render" / f"{side}_robot_mask"
            if mode == "both"
            else source_segment / "render" / "robot_mask"
        )
    final = output_segment / "final"
    if final.exists():
        shutil.rmtree(final)
    composite_videos(
        source_video=candidate / "source.mp4",
        inpainted_video=(
            source_segment / "inpaint" / "frames" / "inpaint_out_exact.mp4"
        ),
        robot_rgb_dir=source_segment / "render" / "robot_rgb",
        robot_mask_dir=source_segment / "render" / "robot_mask",
        human_mask_dir=source_segment / "human_mask",
        output_dir=final,
        human_visibility_by_side=visibility,
        robot_mask_dirs_by_side=robot_masks,
    )
    return final / "composite_full.mp4"


def main() -> None:
    args = parse_args()
    episode = json.loads((args.source_run / "episode.json").read_text())
    chunks = [
        (item["mode"], int(item["start"]), int(item["end"]))
        for item in episode["chunks"]
    ]
    if args.worker_index is None and args.workers > 1:
        children = []
        for index in range(args.workers):
            children.append(subprocess.Popen([
                sys.executable,
                str(Path(__file__).resolve()),
                "--source-run", str(args.source_run),
                "--candidate-root", str(args.candidate_root),
                "--recording", str(args.recording),
                "--output-run", str(args.output_run),
                "--workers", str(args.workers),
                "--worker-index", str(index),
            ]))
        failures = [child.wait() for child in children]
        if any(failures):
            raise RuntimeError(f"recomposition workers failed: {failures}")
    else:
        worker = args.worker_index or 0
        for index, (mode, start, end) in enumerate(chunks):
            if index % args.workers != worker:
                continue
            name = f"{index:03d}_{mode}_{start:05d}_{end:05d}"
            recompose_segment(
                args.source_run / "segments" / name,
                args.candidate_root / name,
                args.output_run / "segments" / name,
                mode,
            )
            print(f"recomposed {name}", flush=True)
        if args.worker_index is not None:
            return

    outputs = [
        args.output_run / "segments"
        / f"{index:03d}_{mode}_{start:05d}_{end:05d}"
        / "final/composite_full.mp4"
        for index, (mode, start, end) in enumerate(chunks)
    ]
    assemble_recording(
        args.recording,
        chunks,
        outputs,
        args.output_run / "final/composite_full_recording.mp4",
        int(episode["source_start_frame"]),
        int(episode["source_end_frame"]),
    )
    command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(args.recording / "camera.mp4"),
        "-i", str(args.output_run / "final/composite_full_recording.mp4"),
        "-filter_complex",
        f"[0:v]trim=start_frame={episode['source_start_frame']}:"
        f"end_frame={episode['source_end_frame']},"
        "setpts=PTS-STARTPTS,scale=640:352[left];"
        "[1:v]scale=640:352[right];[left][right]hstack=inputs=2[out]",
        "-map", "[out]", "-an", "-r", "30", "-c:v", "libx264",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output_run / "final/human_left_robot_right_full_recording.mp4"),
    ])
    (args.output_run / "episode.json").write_text(
        json.dumps(episode, indent=2) + "\n"
    )
    print(f"complete: {args.output_run}")


if __name__ == "__main__":
    main()
