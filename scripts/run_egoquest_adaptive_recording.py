#!/usr/bin/env python3
"""Robotize an EgoQuest recording with left/right/both adaptive chunks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


STATE_NAMES = {0: "none", 1: "left", 2: "right", 3: "both"}


def expand_visibility_intervals(
    visible: np.ndarray,
    *,
    entry_padding_frames: int,
    exit_padding_frames: int,
) -> np.ndarray:
    """Expand every visible interval toward earlier and later video frames.

    The earlier expansion is intentionally larger: projected 3-D landmarks
    become valid only after a hand has entered the image, while human removal
    must already cover the first partially visible fingers and sleeve pixels.
    """

    values = np.asarray(visible, dtype=bool)
    if values.ndim != 1:
        raise ValueError("visibility must be a one-dimensional sequence")
    if entry_padding_frames < 0 or exit_padding_frames < 0:
        raise ValueError("visibility padding must be non-negative")
    expanded = np.zeros_like(values)
    if not values.any():
        return expanded
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False]) + 1
    for start, end in zip(starts, ends):
        expanded[
            max(0, int(start) - entry_padding_frames) :
            min(len(values), int(end) + exit_padding_frames)
        ] = True
    return expanded


def command(args: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, check=True, env=env)


def hand_visibility(
    recording: Path,
    side: str,
    *,
    minimum_visible_landmarks: int = 1,
    entry_padding_frames: int = 8,
    exit_padding_frames: int = 4,
) -> np.ndarray:
    table = pq.read_table(
        recording / "aligned_frames.parquet",
        columns=[
            "camera_position_world",
            "camera_quaternion_world",
            f"{side}_landmarks_world",
        ],
    )
    rows = table.to_pydict()
    session = json.loads((recording / "session.json").read_text())
    calibration = session["camera_calibration"]
    source_width, source_height = calibration["current_resolution"]
    video_resolution = session.get("video_resolution")
    if video_resolution is None:
        width, height = int(source_width), int(source_height)
    else:
        width = int(video_resolution["width"])
        height = int(video_resolution["height"])
    fx, fy = calibration["focal_length"]
    cx, cy = calibration["principal_point"]
    fx *= width / source_width
    fy *= height / source_height
    cx *= width / source_width
    cy *= height / source_height

    world_basis = np.diag([1.0, 1.0, -1.0])
    camera_basis = np.diag([1.0, -1.0, 1.0])
    camera_position = (
        np.asarray(rows["camera_position_world"], dtype=np.float64)
        @ world_basis.T
    )
    camera_rotation = np.einsum(
        "ij,tjk,kl->til",
        world_basis,
        Rotation.from_quat(
            np.asarray(rows["camera_quaternion_world"], dtype=np.float64)
        ).as_matrix(),
        camera_basis,
    )
    landmarks = (
        np.asarray(rows[f"{side}_landmarks_world"], dtype=np.float64)
        .reshape(table.num_rows, 21, 3)
        @ world_basis.T
    )
    camera_points = np.einsum(
        "tij,tkj->tki",
        np.swapaxes(camera_rotation, 1, 2),
        landmarks - camera_position[:, None, :],
    )
    depth = camera_points[..., 2]
    u = fx * camera_points[..., 0] / np.maximum(depth, 1e-6) + cx
    v = fy * camera_points[..., 1] / np.maximum(depth, 1e-6) + cy
    if not 1 <= minimum_visible_landmarks <= 21:
        raise ValueError("minimum_visible_landmarks must be in [1, 21]")
    visible = (
        (depth > 0.05)
        & (u >= -20)
        & (u < width + 20)
        & (v >= -20)
        & (v < height + 20)
    ).sum(axis=1) >= minimum_visible_landmarks
    false_indices = np.flatnonzero(~visible)
    for before, after in zip(
        np.r_[-1, false_indices], np.r_[false_indices, len(visible)]
    ):
        if 0 < after - before - 1 <= 5:
            visible[before + 1 : after] = True
    return expand_visibility_intervals(
        visible,
        entry_padding_frames=entry_padding_frames,
        exit_padding_frames=exit_padding_frames,
    )


def adaptive_chunks(
    recording: Path,
    *,
    min_frames: int,
    max_frames: int,
    start_frame: int = 0,
    end_frame: int | None = None,
    minimum_visible_landmarks: int = 1,
    entry_padding_frames: int = 8,
    exit_padding_frames: int = 4,
) -> list[tuple[str, int, int]]:
    visibility_options = {
        "minimum_visible_landmarks": minimum_visible_landmarks,
        "entry_padding_frames": entry_padding_frames,
        "exit_padding_frames": exit_padding_frames,
    }
    left = hand_visibility(recording, "left", **visibility_options)
    right = hand_visibility(recording, "right", **visibility_options)
    state = left.astype(np.uint8) + 2 * right.astype(np.uint8)
    end_frame = len(state) if end_frame is None else min(end_frame, len(state))
    if not 0 <= start_frame < end_frame:
        raise ValueError("invalid requested frame range")
    state = state[start_frame:end_frame]
    starts = np.r_[0, np.flatnonzero(state[1:] != state[:-1]) + 1]
    ends = np.r_[starts[1:], len(state)]
    chunks: list[tuple[str, int, int]] = []
    for start, end in zip(starts, ends):
        state_name = STATE_NAMES[int(state[start])]
        if state_name == "none" or end - start < min_frames:
            continue
        cursor = int(start) + start_frame
        absolute_end = int(end) + start_frame
        while cursor < absolute_end:
            next_end = min(cursor + max_frames, absolute_end)
            chunks.append((state_name, cursor, next_end))
            cursor = next_end
    return chunks


def common_flags(
    python: str, workspace: Path, video: Path, hdf5: Path, output: Path
) -> list[str]:
    return [
        python,
        "--video", str(video),
        "--hdf5", str(hdf5),
        "--output", str(output),
        "--dex-assets",
        str(workspace / "third_party/dex-retargeting/assets/robots/hands"),
        "--sam2-root", str(workspace / "third_party/sam2"),
        "--sam2-checkpoint",
        str(workspace / "third_party/sam2/checkpoints/sam2.1_hiera_small.pt"),
        "--sam2-size", "small",
        "--propainter-root", str(workspace / "third_party/ProPainter"),
        "--scale", "1.0",
        # Match the validated EgoDex removal path: recondition SAM2 on every
        # frame instead of letting a ten-frame propagation absorb nearby
        # furniture or manipulated objects.
        "--prompt-stride", "1",
        "--smoothing-window", "9",
        "--smoothing-passes", "2",
        "--forearm-max-angular-velocity", "2.0",
    ]


def single_cli(
    side: str, python: str, workspace: Path, video: Path, hdf5: Path, output: Path
) -> list[str]:
    flags = common_flags(python, workspace, video, hdf5, output)
    return [
        flags[0], "-m", "egodex_dexhand.ur5e_shadow_cli", *flags[1:],
        "--combined-urdf",
        str(
            workspace
            / "third_party/dex-retargeting/assets/robots/assembly/ur5e_shadow"
            / f"ur5e_shadow_{side}_hand_glb.urdf"
        ),
        "--hand", side,
        *[
            item
            for link in (
                "base_link_inertia", "shoulder_link", "upper_arm_link",
                "forearm_link", "wrist_1_link", "wrist_2_link",
                "wrist_3_link",
            )
            for item in ("--hide-arm-visual-link", link)
        ],
    ]


def bimanual_cli(
    python: str, workspace: Path, video: Path, hdf5: Path, output: Path
) -> list[str]:
    flags = common_flags(python, workspace, video, hdf5, output)
    assembly = workspace / "third_party/dex-retargeting/assets/robots/assembly/ur5e_shadow"
    result = [
        flags[0], "-m", "egodex_dexhand.bimanual_cli", *flags[1:],
        "--left-combined-urdf", str(assembly / "ur5e_shadow_left_hand_glb.urdf"),
        "--right-combined-urdf", str(assembly / "ur5e_shadow_right_hand_glb.urdf"),
    ]
    for side in ("left", "right"):
        for link in (
            "base_link_inertia", "shoulder_link", "upper_arm_link",
            "forearm_link", "wrist_1_link", "wrist_2_link",
            "wrist_3_link",
        ):
            result.extend([f"--{side}-hide-arm-visual-link", link])
    return result


RIGHT_BRANCHES: tuple[tuple[float, ...] | None, ...] = (
    None,
    (-2.2535082, -1.5632967, 2.5382182, -2.4822033, 2.1051646, -2.3255349),
    (0.0, -1.8, 2.2, -2.55, 2.3, -2.05),
    (0.0, -1.9415193, -2.3834678, -1.3048847, -0.6362524, -2.2988197),
)
LEFT_BRANCHES: tuple[tuple[float, ...] | None, ...] = (
    None,
    (0.75, -1.35, 1.70, -1.92, -1.57, -0.75),
    (-0.75, -1.35, 1.70, -1.92, -1.57, 0.75),
    (0.0, -1.8, 2.2, -2.55, -2.3, 2.05),
)


def run_with_branches(
    mode: str, base: list[str], environment: dict[str, str]
) -> None:
    attempts: list[list[str]] = []
    if mode in ("left", "right"):
        branches = LEFT_BRANCHES if mode == "left" else RIGHT_BRANCHES
        for branch in branches:
            args = base[:]
            if branch is not None:
                args.extend(["--arm-q-reference", *(str(v) for v in branch)])
            attempts.append(args)
    else:
        branch_pairs = (
            (LEFT_BRANCHES[0], RIGHT_BRANCHES[0]),
            (LEFT_BRANCHES[1], RIGHT_BRANCHES[1]),
            (LEFT_BRANCHES[2], RIGHT_BRANCHES[2]),
            (LEFT_BRANCHES[3], RIGHT_BRANCHES[3]),
        )
        for left, right in branch_pairs:
            args = base[:]
            if left is not None:
                args.extend(["--left-arm-reference-qpos", *(str(v) for v in left)])
            if right is not None:
                args.extend(["--right-arm-reference-qpos", *(str(v) for v in right)])
            attempts.append(args)
    last_error: subprocess.CalledProcessError | None = None
    for args in attempts:
        try:
            command(args + ["--force"], env=environment)
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            # The renderer writes all lossless frames before applying its
            # conservative "arm must be visible" guard.  A hand entering from
            # the image edge can legitimately have its hidden proximal UR5e
            # chain outside the crop.  If every robot RGB frame exists, retain
            # that complete render and continue from segmentation rather than
            # forcing a bulky upper-arm visual back into the image.
            output = Path(args[args.index("--output") + 1])
            metadata = output / "metadata.json"
            if metadata.exists():
                expected = int(json.loads(metadata.read_text())["frame_count"])
                rendered = len(list((output / "render/robot_rgb").glob("*.png")))
                if rendered == expected and expected > 0:
                    try:
                        command(args + ["--start-stage", "segment"], env=environment)
                        return
                    except subprocess.CalledProcessError as resume_error:
                        last_error = resume_error
    assert last_error is not None
    raise last_error


def process_chunk(
    recording: Path,
    mode: str,
    start: int,
    end: int,
    candidate: Path,
    output: Path,
    project: Path,
    workspace: Path,
    environment: dict[str, str],
    detection_brightness_gain: float,
    detection_brightness_offset: float,
) -> Path:
    final = output / "final/composite_full.mp4"
    if final.exists():
        return final
    candidate.mkdir(parents=True, exist_ok=True)
    video = candidate / "source.mp4"
    hdf5 = candidate / "annotations.hdf5"
    mask_hdf5 = candidate / "rgb_mask_annotations.hdf5"
    aligned_hdf5 = candidate / "annotations_rgb_aligned.hdf5"
    python = os.environ.get(
        "EGODEX_PYTHON", str(workspace / ".venv/bin/python")
    )
    mediapipe_python = os.environ.get(
        "EGODEX_MEDIAPIPE_PYTHON",
        str(workspace / ".venv-mediapipe/bin/python"),
    )
    command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(recording / "camera.mp4"),
        "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
        "-an", "-r", "30", "-c:v", "libx264", "-crf", "12",
        "-pix_fmt", "yuv420p", str(video),
    ])
    command([
        python, str(project / "scripts/convert_egoquest_to_egodex_hdf5.py"),
        str(recording), str(hdf5),
        "--start-frame", str(start), "--end-frame", str(end),
        "--active-hand", mode,
    ], env=environment)
    command([
        mediapipe_python,
        str(project / "scripts/build_rgb_mask_landmarks.py"),
        str(video), str(mask_hdf5), "--mode", mode,
        "--brightness-gain", str(detection_brightness_gain),
        "--brightness-offset", str(detection_brightness_offset),
    ])
    alignment_sides = ("left", "right") if mode == "both" else (mode,)
    command([
        python, str(project / "scripts/align_egoquest_3d_to_rgb.py"),
        str(hdf5), str(mask_hdf5), str(aligned_hdf5),
        *[item for side in alignment_sides for item in ("--side", side)],
    ], env=environment)
    base = (
        bimanual_cli(python, workspace, video, aligned_hdf5, output)
        if mode == "both"
        else single_cli(mode, python, workspace, video, aligned_hdf5, output)
    )
    base.extend(["--mask-hdf5", str(mask_hdf5)])
    run_with_branches(mode, base, environment)
    return final


def assemble_recording(
    recording: Path,
    chunks: list[tuple[str, int, int]],
    outputs: list[Path],
    destination: Path,
    start_frame: int,
    end_frame: int,
) -> None:
    inputs = ["-i", str(recording / "camera.mp4")]
    for output in outputs:
        inputs.extend(["-i", str(output)])
    filters: list[str] = []
    streams: list[str] = []
    cursor = start_frame
    for input_index, (_, start, end) in enumerate(chunks, start=1):
        if cursor < start:
            label = f"s{len(streams)}"
            filters.append(
                f"[0:v]trim=start_frame={cursor}:end_frame={start},"
                f"setpts=PTS-STARTPTS[{label}]"
            )
            streams.append(f"[{label}]")
        label = f"s{len(streams)}"
        filters.append(f"[{input_index}:v]setpts=PTS-STARTPTS[{label}]")
        streams.append(f"[{label}]")
        cursor = end
    if cursor < end_frame:
        label = f"s{len(streams)}"
        filters.append(
            f"[0:v]trim=start_frame={cursor}:end_frame={end_frame},"
            f"setpts=PTS-STARTPTS[{label}]"
        )
        streams.append(f"[{label}]")
    filters.append(
        "".join(streams)
        + f"concat=n={len(streams)}:v=1:a=0,format=yuv420p[full]"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
        "-filter_complex", ";".join(filters), "-map", "[full]", "-r", "30",
        "-c:v", "libx264", "-crf", "18", "-movflags", "+faststart",
        str(destination),
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=Path(
        "/data/shared/xxq/egodex_dexhand_pipeline"
    ))
    parser.add_argument("--run-name", default="egoquest_adaptive")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--min-frames", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--minimum-visible-landmarks", type=int, default=1)
    parser.add_argument("--entry-padding-frames", type=int, default=8)
    parser.add_argument("--exit-padding-frames", type=int, default=4)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--detection-brightness-gain", type=float, default=1.0)
    parser.add_argument("--detection-brightness-offset", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.workspace / "project"
    episode = args.recording.name
    run_root = args.workspace / "runs" / args.run_name / episode
    candidate_root = args.workspace / "candidates" / args.run_name / episode
    chunks = adaptive_chunks(
        args.recording,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        minimum_visible_landmarks=args.minimum_visible_landmarks,
        entry_padding_frames=args.entry_padding_frames,
        exit_padding_frames=args.exit_padding_frames,
    )
    total_source_frames = pq.read_metadata(
        args.recording / "aligned_frames.parquet"
    ).num_rows
    end_frame = total_source_frames if args.end_frame is None else args.end_frame

    if args.worker_index is None:
        children = []
        for index in range(args.workers):
            children.append(subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()),
                "--recording", str(args.recording),
                "--workspace", str(args.workspace),
                "--run-name", args.run_name,
                "--workers", str(args.workers),
                "--worker-index", str(index), "--gpu", str(index),
                "--min-frames", str(args.min_frames),
                "--max-frames", str(args.max_frames),
                "--minimum-visible-landmarks",
                str(args.minimum_visible_landmarks),
                "--entry-padding-frames", str(args.entry_padding_frames),
                "--exit-padding-frames", str(args.exit_padding_frames),
                "--start-frame", str(args.start_frame),
                "--end-frame", str(end_frame),
                "--detection-brightness-gain", str(args.detection_brightness_gain),
                "--detection-brightness-offset", str(args.detection_brightness_offset),
            ]))
        failures = [child.wait() for child in children]
        if any(failures):
            raise RuntimeError(f"adaptive workers failed: {failures}")
        outputs = [
            run_root / "segments" / f"{i:03d}_{mode}_{start:05d}_{end:05d}"
            / "final/composite_full.mp4"
            for i, (mode, start, end) in enumerate(chunks)
        ]
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            raise FileNotFoundError("missing chunk outputs: " + ", ".join(missing))
        full = run_root / "final/composite_full_recording.mp4"
        assemble_recording(
            args.recording,
            chunks,
            outputs,
            full,
            args.start_frame,
            end_frame,
        )
        side_by_side = run_root / "final/human_left_robot_right_full_recording.mp4"
        command([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(args.recording / "camera.mp4"), "-i", str(full),
            "-filter_complex",
            f"[0:v]trim=start_frame={args.start_frame}:end_frame={end_frame},"
            "setpts=PTS-STARTPTS,scale=640:352[left];"
            "[1:v]scale=640:352[right];"
            "[left][right]hstack=inputs=2[out]",
            "-map", "[out]", "-an", "-r", "30", "-c:v", "libx264",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(side_by_side),
        ])
        (run_root / "episode.json").write_text(json.dumps({
            "episode": episode,
            "total_frames": end_frame - args.start_frame,
            "source_start_frame": args.start_frame,
            "source_end_frame": end_frame,
            "chunks": [
                {"index": i, "mode": mode, "start": start, "end": end}
                for i, (mode, start, end) in enumerate(chunks)
            ],
        }, indent=2) + "\n")
        print(f"COMPLETE {episode}: {len(chunks)} adaptive chunks")
        return

    if args.gpu is None:
        raise ValueError("worker mode requires --gpu")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(project / "src"), str(args.workspace / "third_party/sam2")]
    )
    for index, (mode, start, end) in enumerate(chunks):
        if index % args.workers != args.worker_index:
            continue
        name = f"{index:03d}_{mode}_{start:05d}_{end:05d}"
        print(f"GPU {args.gpu}: {name}", flush=True)
        process_chunk(
            args.recording, mode, start, end,
            candidate_root / name, run_root / "segments" / name,
            project, args.workspace, environment,
            args.detection_brightness_gain,
            args.detection_brightness_offset,
        )


if __name__ == "__main__":
    main()
