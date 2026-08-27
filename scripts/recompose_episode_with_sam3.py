#!/usr/bin/env python3
"""Apply saved SAM3 masks, rerun ProPainter, and recompose adaptive segments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from egodex_dexhand.compose import composite_videos
from egodex_dexhand.data import (
    fuse_hand_visibility,
    load_egodex_sequence,
    projected_hand_visibility,
    scaled_intrinsic,
)
from egodex_dexhand.inpaint import (
    finalize_propainter_output,
    propainter_command,
    run_propainter,
)


SEGMENT_RE = re.compile(r"^(\d+)_(left|right|both)_(\d+)_(\d+)$")
PROPAINTER_MINIMUM_FREE_MIB = 24 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--segments-root", type=Path, required=True)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--propainter-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--segment-name")
    parser.add_argument("--propainter-ready", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    return parser.parse_args()


def numbered(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"))


def gpu_free_memory_mib() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    free: dict[str, int] = {}
    for line in result.stdout.splitlines():
        index, value = (part.strip() for part in line.split(",", 1))
        free[index] = int(value)
    return free


def choose_persistent_gpus(
    requested: list[str],
    free_mib: dict[str, int],
    *,
    minimum_free_mib: int = PROPAINTER_MINIMUM_FREE_MIB,
) -> list[str]:
    """Prefer assigned devices, but move a heavy stage away from contention."""

    eligible = {
        gpu for gpu, available in free_mib.items() if available >= minimum_free_mib
    }
    preferred = [gpu for gpu in requested if gpu in eligible]
    alternatives = sorted(
        eligible.difference(preferred),
        key=lambda gpu: free_mib[gpu],
        reverse=True,
    )
    candidates = preferred + alternatives
    return candidates[: min(len(requested), len(candidates))]


def wait_for_persistent_gpus(
    requested: list[str],
    *,
    timeout_seconds: float = 3600.0,
    poll_seconds: float = 30.0,
) -> list[str]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        free = gpu_free_memory_mib()
        selected = choose_persistent_gpus(requested, free)
        if selected:
            if selected != requested[: len(selected)]:
                print(
                    "PROPAINTER GPU REASSIGN "
                    f"requested={requested} selected={selected} free_mib={free}",
                    flush=True,
                )
            return selected
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "no GPU reached the ProPainter free-memory threshold "
                f"({PROPAINTER_MINIMUM_FREE_MIB} MiB)"
            )
        print(
            "WAIT ProPainter GPU memory "
            f"minimum={PROPAINTER_MINIMUM_FREE_MIB} MiB free_mib={free}",
            flush=True,
        )
        time.sleep(poll_seconds)


def install_sam3_masks(segment: Path, sam3_union: Path) -> None:
    source = numbered(sam3_union)
    target = segment / "human_mask"
    current = numbered(target)
    if not source:
        raise RuntimeError(f"no SAM3 masks for {segment.name}: {sam3_union}")
    if current and len(source) != len(current):
        raise RuntimeError(
            f"SAM3/current mask count mismatch for {segment.name}: "
            f"{len(source)} vs {len(current)}"
        )
    backup = segment / "human_mask_sam2"
    if current and not backup.exists():
        shutil.copytree(target, backup)
    target.mkdir(parents=True, exist_ok=True)
    for index, source_path in enumerate(source):
        shutil.copy2(source_path, target / f"{index:05d}.png")


def visibility_for_side(
    candidate: Path,
    side: str,
    metadata: dict[str, object],
) -> object:
    pose = load_egodex_sequence(candidate / "annotations.hdf5", side)
    rgb = load_egodex_sequence(candidate / "rgb_mask_annotations.hdf5", side)
    width, height = int(metadata["width"]), int(metadata["height"])
    intrinsic = scaled_intrinsic(
        pose.intrinsic,
        width / int(metadata["source_width"]),
        height / int(metadata["source_height"]),
    )
    return fuse_hand_visibility(
        projected_hand_visibility(
            pose.joints_camera_cv,
            intrinsic,
            width,
            height,
            joint_confidence=pose.joint_confidence,
        ),
        rgb.joint_confidence,
    )


def process_one(
    args: argparse.Namespace,
    name: str,
    *,
    propainter_ready: bool = False,
) -> None:
    match = SEGMENT_RE.match(name)
    if match is None:
        raise ValueError(name)
    mode = match.group(2)
    sides = ("left", "right") if mode == "both" else (mode,)
    segment = args.segments_root / name
    candidate = args.candidates_root / name
    metadata = json.loads((segment / "metadata.json").read_text())
    install_sam3_masks(segment, args.sam3_root / name / "union_mask")

    if propainter_ready:
        inpainted = finalize_propainter_output(
            frames_dir=segment / "frames",
            masks_dir=segment / "human_mask",
            output_dir=segment / "inpaint",
            fps=float(metadata["fps"]),
        )
    else:
        inpaint_stage = segment / "inpaint" / "frames"
        if inpaint_stage.exists():
            shutil.rmtree(inpaint_stage)
        inpainted = run_propainter(
            frames_dir=segment / "frames",
            masks_dir=segment / "human_mask",
            propainter_root=args.propainter_root,
            output_dir=segment / "inpaint",
            python_executable=sys.executable,
            fps=float(metadata["fps"]),
            fp16=True,
        )
    visibility = {
        side: visibility_for_side(candidate, side, metadata) for side in sides
    }
    side_masks = {
        side: (
            segment / "render" / f"{side}_robot_mask"
            if mode == "both"
            else segment / "render" / "robot_mask"
        )
        for side in sides
    }
    composite_videos(
        source_video=candidate / "source.mp4",
        inpainted_video=inpainted,
        robot_rgb_dir=segment / "render" / "robot_rgb",
        robot_mask_dir=segment / "render" / "robot_mask",
        human_mask_dir=segment / "human_mask",
        output_dir=segment / "final",
        human_visibility_by_side=visibility,
        robot_mask_dirs_by_side=side_masks,
    )
    (segment / "sam3_recompose.json").write_text(
        json.dumps(
            {
                "mask_source": str((args.sam3_root / name / "union_mask").resolve()),
                "sides": list(sides),
                "frame_count": int(metadata["frame_count"]),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"DONE {name}", flush=True)


def persistent_coordinator(
    args: argparse.Namespace,
    jobs: list[str],
    gpus: list[str],
) -> None:
    assignments: dict[str, list[tuple[str, int]]] = {gpu: [] for gpu in gpus}
    loads = {gpu: 0 for gpu in gpus}
    for name in jobs:
        metadata = json.loads(
            (args.segments_root / name / "metadata.json").read_text()
        )
        gpu = min(gpus, key=lambda value: loads[value])
        frames = int(metadata["frame_count"])
        assignments[gpu].append((name, frames))
        loads[gpu] += frames

    running: list[tuple[subprocess.Popen, object, str, Path]] = []
    for gpu, assigned in assignments.items():
        if not assigned:
            continue
        worker_jobs = []
        for name, _ in assigned:
            segment = args.segments_root / name
            metadata = json.loads((segment / "metadata.json").read_text())
            install_sam3_masks(segment, args.sam3_root / name / "union_mask")
            inpaint_stage = segment / "inpaint" / "frames"
            if inpaint_stage.exists():
                shutil.rmtree(inpaint_stage)
            command = propainter_command(
                frames_dir=segment / "frames",
                masks_dir=segment / "human_mask",
                propainter_root=args.propainter_root,
                output_dir=segment / "inpaint",
                python_executable=sys.executable,
                fps=float(metadata["fps"]),
                fp16=True,
            )
            postprocess = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--project-root", str(args.project_root),
                "--candidates-root", str(args.candidates_root),
                "--segments-root", str(args.segments_root),
                "--sam3-root", str(args.sam3_root),
                "--propainter-root", str(args.propainter_root),
                "--segment-name", name,
                "--propainter-ready",
            ]
            worker_jobs.append(
                {
                    "name": name,
                    "arguments": command[2:],
                    "postprocess": postprocess,
                    "postprocess_cwd": str(args.project_root),
                }
            )
        jobs_path = args.segments_root / f".propainter_jobs_gpu_{gpu}.json"
        jobs_path.write_text(json.dumps(worker_jobs, indent=2) + "\n")
        log = (args.segments_root / f"propainter_persistent_gpu_{gpu}.log").open("w")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["EGODEX_INPAINT_CUDA_VISIBLE_DEVICES"] = gpu
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("run_propainter_jobs.py")),
                "--propainter-root", str(args.propainter_root),
                "--jobs", str(jobs_path),
            ],
            cwd=args.project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        running.append((process, log, gpu, jobs_path))
        print(
            f"START persistent gpu={gpu} jobs={len(worker_jobs)} "
            f"frames={loads[gpu]}",
            flush=True,
        )
    failures = []
    for process, log, gpu, jobs_path in running:
        code = process.wait()
        log.close()
        if code:
            failures.append((gpu, code, str(jobs_path)))
        else:
            print(f"DONE persistent gpu={gpu}", flush=True)
    if failures:
        raise RuntimeError(f"persistent ProPainter failures: {failures}")


def coordinator(args: argparse.Namespace) -> None:
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    jobs = []
    for segment in sorted(path for path in args.segments_root.iterdir() if path.is_dir()):
        if SEGMENT_RE.match(segment.name) is None:
            continue
        metadata_path = segment / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"baseline render incomplete: {segment}")
        expected = int(json.loads(metadata_path.read_text())["frame_count"])
        if len(numbered(segment / "render" / "robot_rgb")) != expected:
            raise FileNotFoundError(f"baseline render incomplete: {segment}")
        if (segment / "sam3_recompose.json").is_file():
            continue
        jobs.append(segment.name)
    if args.persistent_workers:
        gpus = wait_for_persistent_gpus(gpus)
        persistent_coordinator(args, jobs, gpus)
        return
    running: dict[str, tuple[subprocess.Popen, object, str]] = {}
    available = list(gpus)
    failures = []
    while jobs or running:
        while jobs and available:
            name = jobs.pop(0)
            gpu = available.pop(0)
            log = (args.segments_root / name / "sam3_recompose.log").open("w")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--project-root", str(args.project_root),
                "--candidates-root", str(args.candidates_root),
                "--segments-root", str(args.segments_root),
                "--sam3-root", str(args.sam3_root),
                "--propainter-root", str(args.propainter_root),
                "--segment-name", name,
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            # Keep inpainting on this scheduler-assigned GPU. Otherwise
            # ProPainter probes the whole host and can collide with another
            # episode lane on whichever device is momentarily emptiest.
            environment["EGODEX_INPAINT_CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command, cwd=args.project_root, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
            running[gpu] = (process, log, name)
            print(f"START gpu={gpu} {name}", flush=True)
        completed = []
        for gpu, (process, log, name) in running.items():
            code = process.poll()
            if code is None:
                continue
            log.close()
            completed.append(gpu)
            if code:
                failures.append((name, code))
                print(f"FAIL gpu={gpu} {name} code={code}", flush=True)
            else:
                print(f"DONE gpu={gpu} {name}", flush=True)
        for gpu in completed:
            del running[gpu]
            available.append(gpu)
        if running and not completed:
            time.sleep(1)
    if failures:
        raise RuntimeError(f"SAM3 recomposition failures: {failures}")


def main() -> None:
    args = parse_args()
    if args.segment_name:
        process_one(
            args,
            args.segment_name,
            propainter_ready=args.propainter_ready,
        )
    else:
        coordinator(args)


if __name__ == "__main__":
    main()
