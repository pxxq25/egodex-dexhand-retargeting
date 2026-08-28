#!/usr/bin/env python3
"""Robotize an EgoQuest recording with left/right/both adaptive chunks."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import traceback

import numpy as np


STATE_NAMES = {0: "none", 1: "left", 2: "right", 3: "both"}
PERSISTENT_RENDER_MODULES = {
    "egodex_dexhand.ur5e_shadow_cli",
    "egodex_dexhand.bimanual_cli",
}


def in_process_render_command(
    args: list[str], env: dict[str, str] | None
) -> bool:
    return bool(
        env
        and env.get("EGODEX_IN_PROCESS_RENDER") == "1"
        and len(args) >= 3
        and args[1] == "-m"
        and args[2] in PERSISTENT_RENDER_MODULES
    )


def run_render_module(args: list[str], env: dict[str, str]) -> None:
    """Execute a render-only CLI in this worker's persistent Vulkan process."""

    module = args[2]
    previous_argv = list(sys.argv)
    for key, value in env.items():
        os.environ[key] = value
    for path in reversed(env.get("PYTHONPATH", "").split(os.pathsep)):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    try:
        sys.argv = [module, *args[3:]]
        runpy.run_module(module, run_name="__main__")
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else int(bool(error.code))
        if code:
            raise subprocess.CalledProcessError(code, args) from error
    except subprocess.CalledProcessError:
        raise
    except Exception as error:
        traceback.print_exc()
        raise subprocess.CalledProcessError(1, args) from error
    finally:
        sys.argv = previous_argv
        gc.collect()


def hts_projection_params(
    session: dict, width: int, height: int
) -> tuple[float, float, float, float]:
    calibration = session.get("camera_calibration") or {}
    defaults = session.get("projection_defaults", {}) or {}
    focal = calibration.get("focal_length") or [
        defaults.get("fx", 854.844970703125),
        defaults.get("fy", 854.844970703125),
    ]
    principal = calibration.get("principal_point") or [
        defaults.get("cx", 642.5776977539062),
        defaults.get("cy", 645.1429443359375),
    ]
    sensor = (
        calibration.get("sensor_resolution")
        or calibration.get("current_resolution")
        or [width, height]
    )
    current = calibration.get("current_resolution") or [width, height]
    sensor_array = np.asarray(sensor, dtype=np.float64)
    scale = np.asarray(current, dtype=np.float64) / sensor_array
    scale /= max(float(scale[0]), float(scale[1]))
    crop_xy = sensor_array * (1.0 - scale) * 0.5
    crop_wh = sensor_array * scale
    fx = float(focal[0]) * width / float(crop_wh[0])
    fy = float(focal[1]) * height / float(crop_wh[1])
    cx = (float(principal[0]) - float(crop_xy[0])) * width / float(crop_wh[0])
    cy = height - (
        (float(principal[1]) - float(crop_xy[1])) * height / float(crop_wh[1])
    )
    return fx, fy, cx, cy


def bridge_short_visibility_gaps(
    visible: np.ndarray, *, maximum_gap_frames: int = 5
) -> np.ndarray:
    """Fill short interior false runs without inventing visibility at edges."""

    values = np.asarray(visible, dtype=bool).copy()
    if values.ndim != 1:
        raise ValueError("visibility must be a one-dimensional sequence")
    if maximum_gap_frames < 0:
        raise ValueError("maximum gap must be non-negative")
    missing = ~values
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False]) + 1
    for start, end in zip(starts, ends):
        if (
            start > 0
            and end < len(values)
            and end - start <= maximum_gap_frames
        ):
            values[start:end] = True
    return values


def balanced_chunk_ranges(
    start: int, end: int, *, min_frames: int, max_frames: int
) -> list[tuple[int, int]]:
    """Split an interval without producing a tiny final remainder."""

    if min_frames < 1 or max_frames < min_frames:
        raise ValueError("chunk limits must satisfy 1 <= min_frames <= max_frames")
    length = end - start
    if length < min_frames:
        return []
    part_count = max(1, math.ceil(length / max_frames))
    base, remainder = divmod(length, part_count)
    if base < min_frames:
        raise ValueError("interval cannot be partitioned within chunk limits")
    result = []
    cursor = start
    for index in range(part_count):
        size = base + (1 if index < remainder else 0)
        result.append((cursor, cursor + size))
        cursor += size
    return result


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


def command(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> None:
    if timeout is None and in_process_render_command(args, env):
        assert env is not None
        run_render_module(args, env)
        return
    if timeout is None:
        subprocess.run(args, check=True, env=env)
        return
    process = subprocess.Popen(args, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, args)


def video_is_readable(path: Path) -> bool:
    """Return true only for a finalized video that ffprobe can decode."""

    if not path.is_file() or path.stat().st_size == 0:
        return False
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,nb_frames",
            "-of", "default=noprint_wrappers=1", str(path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )
    return result.returncode == 0


def resume_skipped_visual(
    phase: str, skip_unrenderable: bool, skip_reason: Path
) -> bool:
    """Keep a trajectory rejection skipped when later phases resume."""

    return phase == "visual" and skip_unrenderable and skip_reason.exists()


def renderer_environment(
    project: Path, workspace: Path, base: dict[str, str] | None = None
) -> dict[str, str]:
    """Build the common Python/Vulkan environment used by probes and workers."""

    environment = dict(os.environ if base is None else base)
    python_paths = [
        str(project / "src"),
        str(workspace / "third_party/sam2"),
    ]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_paths.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    # The H100 images do not install the NVIDIA ICD in one uniform location:
    # h100-01 uses /etc while h100-03/07/08 use /usr/share. Pin the first
    # available NVIDIA ICD so SAPIEN cannot select a software or Intel driver.
    nvidia_icds = (
        Path("/etc/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/local/share/vulkan/icd.d/nvidia_icd.json"),
    )
    nvidia_icd = next((path for path in nvidia_icds if path.is_file()), None)
    if nvidia_icd is not None:
        environment["VK_ICD_FILENAMES"] = str(nvidia_icd)
    return environment


def preflight_render_device(
    python: str,
    project: Path,
    device: str,
    environment: dict[str, str],
    *,
    timeout: float,
    retries: int,
) -> bool:
    """Return only after a bounded, successful one-frame Vulkan capture."""

    for attempt in range(1, retries + 1):
        try:
            command(
                [
                    python,
                    str(project / "scripts/preflight_sapien_vulkan.py"),
                    "--device",
                    device,
                ],
                env=environment,
                timeout=timeout,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            print(
                f"Vulkan preflight {attempt}/{retries} failed for {device}: {error}",
                flush=True,
            )
    return False


def slice_frame_hdf5(source: Path, destination: Path, start: int, end: int) -> None:
    """Slice every frame-major dataset while preserving metadata and groups."""

    import h5py

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}"
    )
    with h5py.File(source, "r") as source_handle, h5py.File(temporary, "w") as output:
        frame_count = int(source_handle["transforms/camera"].shape[0])
        if not 0 <= start < end <= frame_count:
            raise ValueError(
                f"invalid HDF5 slice [{start}, {end}) for {frame_count} frames"
            )
        for key, value in source_handle.attrs.items():
            output.attrs[key] = value

        def copy_item(name: str, item: object) -> None:
            if isinstance(item, h5py.Group):
                group = output.require_group(name)
                for key, value in item.attrs.items():
                    group.attrs[key] = value
                return
            assert isinstance(item, h5py.Dataset)
            data = item[start:end] if item.ndim > 0 and item.shape[0] == frame_count else item[()]
            dataset = output.create_dataset(name, data=data)
            for key, value in item.attrs.items():
                dataset.attrs[key] = value

        source_handle.visititems(copy_item)
    os.replace(temporary, destination)


def hand_visibility(
    recording: Path,
    side: str,
    *,
    minimum_visible_landmarks: int = 1,
    entry_padding_frames: int = 8,
    exit_padding_frames: int = 8,
    rgb_direct_visibility: np.ndarray | None = None,
) -> np.ndarray:
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation

    table = pq.read_table(
        recording / "aligned_frames.parquet",
        columns=[
            "camera_frame_index",
            "camera_position_world",
            "camera_quaternion_world",
            f"{side}_landmarks_world",
        ],
    ).sort_by([("camera_frame_index", "ascending")])
    rows = table.to_pydict()
    session = json.loads((recording / "session.json").read_text())
    video_resolution = session.get("video_resolution")
    if video_resolution is None:
        width, height = map(
            int, session["camera_calibration"]["current_resolution"]
        )
    else:
        width = int(video_resolution["width"])
        height = int(video_resolution["height"])
    fx, fy, cx, cy = hts_projection_params(session, width, height)

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
    if rgb_direct_visibility is not None:
        rgb_visible = np.asarray(rgb_direct_visibility, dtype=bool)
        if rgb_visible.shape != visible.shape:
            raise ValueError("RGB visibility length does not match pose stream")
        visible |= rgb_visible
    visible = bridge_short_visibility_gaps(visible, maximum_gap_frames=5)
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
    exit_padding_frames: int = 8,
    rgb_visibility: dict[str, np.ndarray] | None = None,
) -> list[tuple[str, int, int]]:
    visibility_options: dict[str, object] = {
        "minimum_visible_landmarks": minimum_visible_landmarks,
        "entry_padding_frames": entry_padding_frames,
        "exit_padding_frames": exit_padding_frames,
    }
    left = hand_visibility(
        recording,
        "left",
        **visibility_options,
        rgb_direct_visibility=None if rgb_visibility is None else rgb_visibility["left"],
    )
    right = hand_visibility(
        recording,
        "right",
        **visibility_options,
        rgb_direct_visibility=None if rgb_visibility is None else rgb_visibility["right"],
    )
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
        absolute_start = int(start) + start_frame
        absolute_end = int(end) + start_frame
        chunks.extend(
            (state_name, chunk_start, chunk_end)
            for chunk_start, chunk_end in balanced_chunk_ranges(
                absolute_start,
                absolute_end,
                min_frames=min_frames,
                max_frames=max_frames,
            )
        )
    return chunks


def load_rgb_direct_visibility(path: Path) -> dict[str, np.ndarray]:
    """Read only genuine per-frame RGB detections, not interpolated points."""

    import h5py

    with h5py.File(path, "r") as handle:
        return {
            side: np.asarray(
                handle[f"confidences/{side}Hand"], dtype=np.float32
            ) >= 0.9
            for side in ("left", "right")
        }


def ensure_full_rgb_visibility(
    recording: Path,
    output: Path,
    *,
    project: Path,
    mediapipe_python: str,
    brightness_gain: float,
    brightness_offset: float,
) -> dict[str, np.ndarray]:
    """Run a full-video RGB prepass once and return direct detection gates."""

    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(
            f".{output.stem}.{os.getpid()}.tmp{output.suffix}"
        )
        command([
            mediapipe_python,
            str(project / "scripts/build_rgb_mask_landmarks.py"),
            str(recording / "camera.mp4"),
            str(temporary),
            "--mode", "both",
            "--allow-missing-hands",
            "--brightness-gain", str(brightness_gain),
            "--brightness-offset", str(brightness_offset),
        ])
        os.replace(temporary, output)
    return load_rgb_direct_visibility(output)


def common_flags(
    python: str, workspace: Path, video: Path, hdf5: Path, output: Path
) -> list[str]:
    result = [
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
        # The wearer can translate through the room.  Keep the embodiment base
        # rigidly attached to the egocentric camera/body frame instead of
        # leaving it behind at the first chunk's world position.
        "--camera-relative-base",
    ]
    stop_stage = os.environ.get("EGODEX_BASELINE_STOP_STAGE")
    if stop_stage:
        result.extend(["--stop-stage", stop_stage])
    start_stage = os.environ.get("EGODEX_PIPELINE_START_STAGE")
    if start_stage:
        result.extend(["--start-stage", start_stage])
    render_device = os.environ.get("EGODEX_RENDER_DEVICE")
    if not render_device:
        raise RuntimeError(
            "EGODEX_RENDER_DEVICE must explicitly select the worker Vulkan device"
        )
    result.extend(["--render-device", render_device])
    return result


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
        "--allow-hidden-arm",
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
        "--allow-hidden-arm",
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
    continuity_run = any(
        value.endswith("initial-continuity-state")
        or value.endswith("write-continuity-state")
        for value in base
    )
    frozen_trajectory = "--start-stage" in base
    if continuity_run or frozen_trajectory:
        # The stateful IK solver performs wrapped-distance branch selection.
        # Retrying legacy fixed references would overwrite the saved state and
        # reintroduce the exact chunk flip this path is designed to prevent.
        attempts.append(base[:])
    elif mode in ("left", "right"):
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
                    # Render-only baseline jobs intentionally stop before
                    # segmentation.  A complete frame sequence is therefore
                    # already a successful result even if the renderer's
                    # conservative arm-visibility guard exits non-zero.
                    if os.environ.get("EGODEX_BASELINE_STOP_STAGE") == "render":
                        return
                    resume_args = [
                        value
                        for index, value in enumerate(args)
                        if value != "--start-stage"
                        and (index == 0 or args[index - 1] != "--start-stage")
                    ]
                    attempted_stages: set[str] = set()
                    for _ in range(4):
                        output = Path(resume_args[resume_args.index("--output") + 1])
                        expected = int(json.loads((output / "metadata.json").read_text())["frame_count"])
                        final = output / "final/composite_full.mp4"
                        if final.exists():
                            return
                        masks = len(list((output / "human_mask").glob("*.png")))
                        inpainted = output / "inpaint/frames/inpaint_out.mp4"
                        if inpainted.exists() and inpainted.stat().st_size > 0:
                            stage = "compose"
                        elif masks == expected:
                            stage = "inpaint"
                        else:
                            stage = "segment"
                        # One retry is useful for transient CUDA OOM after the
                        # failed process releases its reserved allocator cache.
                        retry_key = f"{stage}:{stage in attempted_stages}"
                        if retry_key in attempted_stages:
                            break
                        attempted_stages.add(stage)
                        attempted_stages.add(retry_key)
                        try:
                            command(
                                resume_args + ["--start-stage", stage],
                                env=environment,
                            )
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
    continuity_root: Path | None = None,
    prepare_inputs: bool = True,
) -> Path:
    final = output / "final/composite_full.mp4"
    if final.exists():
        return final
    if os.environ.get("EGODEX_BASELINE_STOP_STAGE") == "render":
        rendered = output / "render" / "robot_rgb"
        if (output / "metadata.json").exists() and len(list(rendered.glob("*.png"))) == end - start:
            return output / "render" / "robot_rgb.mp4"
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
    if prepare_inputs:
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
        if not mask_hdf5.exists():
            full_rgb_visibility = candidate.parent / "full_rgb_visibility.hdf5"
            if full_rgb_visibility.exists():
                slice_frame_hdf5(full_rgb_visibility, mask_hdf5, start, end)
            else:
                command([
                    mediapipe_python,
                    str(project / "scripts/build_rgb_mask_landmarks.py"),
                    str(video), str(mask_hdf5), "--mode", mode,
                    "--brightness-gain", str(detection_brightness_gain),
                    "--brightness-offset", str(detection_brightness_offset),
                ])
        alignment_sides = ("left", "right") if mode == "both" else (mode,)
        import h5py

        with h5py.File(hdf5, "r") as handle:
            camera_processing = str(
                handle.attrs.get("camera_coordinate_processing", "")
            )
        if camera_processing.startswith("hts_calibration_viewport"):
            # The frozen Object Interaction route is already calibrated to the RGB
            # viewport.  A MediaPipe translation would alter those supplied pixels.
            shutil.copy2(hdf5, aligned_hdf5)
            with h5py.File(aligned_hdf5, "r+") as handle:
                handle.attrs["rgb_projection_alignment"] = "disabled_hts_direct"
        else:
            command([
                python, str(project / "scripts/align_egoquest_3d_to_rgb.py"),
                str(hdf5), str(mask_hdf5), str(aligned_hdf5),
                *[item for side in alignment_sides for item in ("--side", side)],
            ], env=environment)
    else:
        for required in (video, hdf5, mask_hdf5, aligned_hdf5):
            if not required.exists():
                raise FileNotFoundError(f"frozen trajectory input is missing: {required}")
    base = (
        bimanual_cli(python, workspace, video, aligned_hdf5, output)
        if mode == "both"
        else single_cli(mode, python, workspace, video, aligned_hdf5, output)
    )
    base.extend([
        "--mask-hdf5", str(mask_hdf5),
        "--visibility-hdf5", str(hdf5),
    ])
    if continuity_root is not None:
        continuity_root.mkdir(parents=True, exist_ok=True)
        active_sides = ("left", "right") if mode == "both" else (mode,)
        for side in active_sides:
            state_path = continuity_root / f"{side}.npz"
            if state_path.exists():
                if mode == "both":
                    base.extend([f"--{side}-initial-continuity-state", str(state_path)])
                else:
                    base.extend(["--initial-continuity-state", str(state_path)])
            if mode == "both":
                base.extend([f"--{side}-write-continuity-state", str(state_path)])
            else:
                base.extend(["--write-continuity-state", str(state_path)])
        base.extend(["--continuity-source-frame", str(end - 1)])
        base.extend([
            "--continuity-margin-frames",
            os.environ.get("EGODEX_CONTINUITY_MARGIN_FRAMES", "12"),
        ])
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
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("EGODEX_WORKSPACE", "workspace")),
    )
    parser.add_argument("--run-name", default="egoquest_adaptive")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument(
        "--phase",
        choices=("all", "trajectory", "visual"),
        default="all",
        help="internal two-phase execution selector",
    )
    parser.add_argument(
        "--gpu-devices",
        type=int,
        nargs="+",
        help="physical GPU indices available to coordinator workers",
    )
    parser.add_argument("--render-device")
    parser.add_argument("--render-preflight-timeout", type=float, default=60.0)
    parser.add_argument("--render-preflight-retries", type=int, default=2)
    parser.add_argument(
        "--skip-unrenderable",
        action="store_true",
        help=(
            "after every IK branch fails to render, preserve the attempt under "
            "skipped_unrenderable and leave that interval as source video"
        ),
    )
    parser.add_argument("--min-frames", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--minimum-visible-landmarks", type=int, default=1)
    parser.add_argument("--entry-padding-frames", type=int, default=8)
    parser.add_argument("--exit-padding-frames", type=int, default=8)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--trajectory-continuity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "process chunks chronologically with persistent per-hand Shadow/UR5e "
            "state (default: enabled)"
        ),
    )
    parser.add_argument(
        "--persistent-render-worker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "reuse one Python/Vulkan process across render-only chunks; "
            "experimental and opt-in"
        ),
    )
    parser.add_argument(
        "--chunk-overlap-frames",
        type=int,
        default=12,
        help="centered cross-boundary smoothing margin in frames",
    )
    parser.add_argument("--detection-brightness-gain", type=float, default=1.0)
    parser.add_argument("--detection-brightness-offset", type=float, default=0.0)
    parser.add_argument(
        "--disable-rgb-visibility-fusion",
        action="store_true",
        help="use only projected 3-D landmarks when selecting intervals",
    )
    return parser.parse_args()


def main() -> None:
    import pyarrow.parquet as pq

    args = parse_args()
    project = args.workspace / "project"
    episode = args.recording.name
    run_root = args.workspace / "runs" / args.run_name / episode
    candidate_root = args.workspace / "candidates" / args.run_name / episode
    mediapipe_python = os.environ.get(
        "EGODEX_MEDIAPIPE_PYTHON",
        str(args.workspace / ".venv-mediapipe/bin/python"),
    )
    rgb_visibility = None
    if not args.disable_rgb_visibility_fusion:
        rgb_visibility = ensure_full_rgb_visibility(
            args.recording,
            candidate_root / "full_rgb_visibility.hdf5",
            project=project,
            mediapipe_python=mediapipe_python,
            brightness_gain=args.detection_brightness_gain,
            brightness_offset=args.detection_brightness_offset,
        )
    chunks = adaptive_chunks(
        args.recording,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        minimum_visible_landmarks=args.minimum_visible_landmarks,
        entry_padding_frames=args.entry_padding_frames,
        exit_padding_frames=args.exit_padding_frames,
        rgb_visibility=rgb_visibility,
    )
    total_source_frames = pq.read_metadata(
        args.recording / "aligned_frames.parquet"
    ).num_rows
    end_frame = total_source_frames if args.end_frame is None else args.end_frame

    if args.worker_index is None:
        python = os.environ.get(
            "EGODEX_PYTHON", str(args.workspace / ".venv/bin/python")
        )
        probe_environment = renderer_environment(project, args.workspace)
        candidate_gpus = (
            args.gpu_devices
            if args.gpu_devices is not None
            else list(range(args.workers))
        )
        healthy_gpus = []
        for gpu in candidate_gpus:
            # CUDA visibility gives every worker a private, one-device view.
            # SAPIEN must therefore bind that physical GPU as worker-local
            # cuda:0, rather than using its host index after CUDA remapping.
            device_environment = probe_environment.copy()
            device_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            if preflight_render_device(
                python,
                project,
                "cuda:0",
                device_environment,
                timeout=args.render_preflight_timeout,
                retries=args.render_preflight_retries,
            ):
                healthy_gpus.append(gpu)
        if not healthy_gpus:
            raise RuntimeError("no SAPIEN Vulkan device passed one-frame preflight")
        active_workers = min(args.workers, len(healthy_gpus))
        if active_workers < args.workers:
            print(
                f"Using {active_workers} preflighted Vulkan devices instead of "
                f"{args.workers}: {healthy_gpus[:active_workers]}",
                flush=True,
            )
        def worker_command(
            index: int, gpu: int, workers: int, phase: str
        ) -> list[str]:
            return [
                sys.executable, str(Path(__file__).resolve()),
                "--recording", str(args.recording),
                "--workspace", str(args.workspace),
                "--run-name", args.run_name,
                "--workers", str(workers),
                "--worker-index", str(index), "--gpu", str(gpu),
                "--phase", phase,
                "--render-device", "cuda:0",
                "--render-preflight-timeout", str(args.render_preflight_timeout),
                "--render-preflight-retries", str(args.render_preflight_retries),
                "--gpu-devices", *(str(value) for value in candidate_gpus),
                *(["--skip-unrenderable"] if args.skip_unrenderable else []),
                "--min-frames", str(args.min_frames),
                "--max-frames", str(args.max_frames),
                "--minimum-visible-landmarks",
                str(args.minimum_visible_landmarks),
                "--entry-padding-frames", str(args.entry_padding_frames),
                "--exit-padding-frames", str(args.exit_padding_frames),
                "--start-frame", str(args.start_frame),
                "--end-frame", str(end_frame),
                *( ["--trajectory-continuity"] if args.trajectory_continuity
                   else ["--no-trajectory-continuity"] ),
                *(
                    ["--persistent-render-worker"]
                    if args.persistent_render_worker
                    else ["--no-persistent-render-worker"]
                ),
                "--chunk-overlap-frames", str(args.chunk_overlap_frames),
                "--detection-brightness-gain", str(args.detection_brightness_gain),
                "--detection-brightness-offset", str(args.detection_brightness_offset),
                *(["--disable-rgb-visibility-fusion"]
                  if args.disable_rgb_visibility_fusion else []),
            ]

        if args.trajectory_continuity and args.phase in ("all", "trajectory"):
            trajectory = subprocess.Popen(
                worker_command(0, healthy_gpus[0], 1, "trajectory")
            )
            trajectory_failure = trajectory.wait()
            if trajectory_failure:
                raise RuntimeError(
                    f"chronological trajectory phase failed: {trajectory_failure}"
                )
            if args.phase == "trajectory":
                print(f"TRAJECTORY PHASE COMPLETE {episode}", flush=True)
                return
        if args.trajectory_continuity:
            children = [
                subprocess.Popen(
                    worker_command(index, gpu, active_workers, "visual")
                )
                for index, gpu in enumerate(healthy_gpus[:active_workers])
            ]
        else:
            children = [
                subprocess.Popen(worker_command(index, gpu, active_workers, "all"))
                for index, gpu in enumerate(healthy_gpus[:active_workers])
            ]
        failures = [child.wait() for child in children]
        if any(failures):
            raise RuntimeError(f"adaptive workers failed: {failures}")
        if os.environ.get("EGODEX_BASELINE_STOP_STAGE") == "render":
            completed = len(list((run_root / "segments").glob("*/metadata.json")))
            skipped = len(
                list((run_root / "skipped_unrenderable").glob("*/skip_reason.json"))
            )
            (run_root / "render_baseline.json").write_text(
                json.dumps(
                    {
                        "episode": episode,
                        "completed_render_segments": completed,
                        "skipped_unrenderable_segments": skipped,
                        "render_devices": healthy_gpus[:active_workers],
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"RENDER BASELINE COMPLETE {episode}: {completed} rendered, "
                f"{skipped} unrenderable",
                flush=True,
            )
            return
        chunk_outputs = [
            (
                chunk,
                run_root / "segments"
                / f"{i:03d}_{chunk[0]}_{chunk[1]:05d}_{chunk[2]:05d}"
                / "final/composite_full.mp4",
            )
            for i, chunk in enumerate(chunks)
        ]
        completed_pairs = [pair for pair in chunk_outputs if pair[1].exists()]
        completed_chunks = [pair[0] for pair in completed_pairs]
        outputs = [pair[1] for pair in completed_pairs]
        full = run_root / "final/composite_full_recording.mp4"
        assemble_recording(
            args.recording,
            completed_chunks,
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
    environment = renderer_environment(project, args.workspace)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    render_device = args.render_device or "cuda:0"
    environment["EGODEX_RENDER_DEVICE"] = render_device
    environment["EGODEX_CONTINUITY_MARGIN_FRAMES"] = str(
        args.chunk_overlap_frames
    )
    # CLI construction occurs in this worker process before the child command
    # receives ``environment``. Keep the same explicit values visible to both.
    os.environ["EGODEX_RENDER_DEVICE"] = render_device
    os.environ["EGODEX_CONTINUITY_MARGIN_FRAMES"] = str(
        args.chunk_overlap_frames
    )
    if args.phase == "trajectory":
        environment["EGODEX_BASELINE_STOP_STAGE"] = "retarget"
        os.environ["EGODEX_BASELINE_STOP_STAGE"] = "retarget"
    elif args.phase == "visual":
        environment["EGODEX_PIPELINE_START_STAGE"] = "render"
        os.environ["EGODEX_PIPELINE_START_STAGE"] = "render"
        environment["EGODEX_INPAINT_CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if args.persistent_render_worker:
            environment["EGODEX_IN_PROCESS_RENDER"] = "1"
    for index, (mode, start, end) in enumerate(chunks):
        if index % args.workers != args.worker_index:
            continue
        name = f"{index:03d}_{mode}_{start:05d}_{end:05d}"
        print(f"GPU {args.gpu}: {name}", flush=True)
        output = run_root / "segments" / name
        completed_visual = output / "final/composite_full.mp4"
        if args.phase == "visual" and video_is_readable(completed_visual):
            print(f"RESUME COMPLETE {name}", flush=True)
            continue
        prior_skip = run_root / "skipped_unrenderable" / name / "skip_reason.json"
        if resume_skipped_visual(
            args.phase, args.skip_unrenderable, prior_skip
        ):
            print(f"RESUME SKIPPED {name}", flush=True)
            continue
        try:
            process_chunk(
                args.recording, mode, start, end,
                candidate_root / name, output,
                project, args.workspace, environment,
                args.detection_brightness_gain,
                args.detection_brightness_offset,
                (run_root / "continuity_state")
                if args.trajectory_continuity and args.phase != "visual"
                else None,
                prepare_inputs=args.phase != "visual",
            )
            stale_skip = run_root / "skipped_unrenderable" / name
            if stale_skip.exists():
                shutil.rmtree(stale_skip)
        except subprocess.CalledProcessError as error:
            if not args.skip_unrenderable:
                raise
            failed_stage = "trajectory_or_render"
            metadata = output / "metadata.json"
            if metadata.exists():
                expected = int(json.loads(metadata.read_text())["frame_count"])
                rendered = len(list((output / "render/robot_rgb").glob("*.png")))
                masks = len(list((output / "human_mask").glob("*.png")))
                if masks == expected:
                    failed_stage = "inpaint_or_compose"
                elif rendered == expected:
                    failed_stage = "segment"
            skipped = run_root / "skipped_unrenderable" / name
            skipped.parent.mkdir(parents=True, exist_ok=True)
            if skipped.exists():
                shutil.rmtree(skipped)
            # A visual-only retry consumes a frozen trajectory.  Never move
            # that resumable state out of the segment directory when a later
            # render/segmentation/inpaint stage fails.
            if output.exists() and args.phase != "visual":
                shutil.move(str(output), str(skipped))
            else:
                skipped.mkdir(parents=True)
            (skipped / "skip_reason.json").write_text(
                json.dumps(
                    {
                        "reason": f"{failed_stage}_failed_after_retries",
                        "returncode": error.returncode,
                        "mode": mode,
                        "start_frame": start,
                        "end_frame": end,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"SKIP UNRENDERABLE {name}", flush=True)


if __name__ == "__main__":
    main()
