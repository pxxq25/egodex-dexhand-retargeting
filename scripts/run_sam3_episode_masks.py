#!/usr/bin/env python3
"""Generate per-side SAM3 masks for every adaptive segment of one episode."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


SEGMENT_RE = re.compile(r"^(\d+)_(left|right|both)_(\d+)_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--segments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--prompt-stride", type=int, default=5)
    parser.add_argument(
        "--prompt-mode", choices=("geometry", "keypoints"), default="geometry"
    )
    parser.add_argument("--production-output", action="store_true")
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="load SAM3 once per GPU and reuse it across that GPU's jobs",
    )
    return parser.parse_args()


def mask_count(path: Path) -> int:
    return sum(1 for _ in path.glob("*.png")) if path.is_dir() else 0


def validate_direct_video_checkpoint(path: Path) -> None:
    """Reject multiplex checkpoints before dispatching many doomed GPU jobs."""

    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"unsupported SAM3 checkpoint container: {type(state)!r}")
    required = (
        "detector.backbone.vision_backbone.trunk.pos_embed",
        "tracker.maskmem_tpos_enc",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(
            f"{path} is not compatible with build_sam3_video_model; "
            f"missing keys: {missing}. Use the direct-video sam3.pt checkpoint, "
            "not sam3.1_multiplex.pt."
        )


def job_arguments(args: argparse.Namespace, job: tuple) -> dict[str, object]:
    name, side, _, _, output = job
    segment = args.segments_root / name
    candidate = args.candidates_root / name
    return {
        "video": str(segment / "frames"),
        "frames": str(segment / "frames"),
        "hdf5": str(candidate / "annotations_rgb_aligned.hdf5"),
        "hand": side,
        "sam2_masks": str(segment / "human_mask"),
        "sam3_checkpoint": str(args.checkpoint),
        "output": str(output),
        "prompt_stride": args.prompt_stride,
        "prompt_mode": args.prompt_mode,
        "fps": 30.0,
        "production_output": bool(args.production_output),
    }


def run_persistent_workers(
    args: argparse.Namespace, jobs: list[tuple], gpus: list[str]
) -> None:
    """Greedily balance frames, then run one long-lived model per GPU."""

    assignments: list[list[tuple]] = [[] for _ in gpus]
    loads = [0 for _ in gpus]
    for job in sorted(jobs, key=lambda item: (item[3] - item[2]), reverse=True):
        worker_index = min(range(len(gpus)), key=loads.__getitem__)
        assignments[worker_index].append(job)
        loads[worker_index] += job[3] - job[2]

    running = []
    for worker_index, (gpu, worker_jobs) in enumerate(zip(gpus, assignments)):
        if not worker_jobs:
            continue
        plan = args.output_root / f"worker_{gpu}_plan.json"
        plan.write_text(json.dumps([
            {
                "name": f"{job[0]}/{job[1]}",
                "frames": job[3] - job[2],
                "arguments": job_arguments(args, job),
            }
            for job in worker_jobs
        ], indent=2) + "\n")
        log_path = args.output_root / f"worker_{gpu}.log"
        log = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (
                str(args.project_root / "src"),
                str(args.checkpoint.parent.parent),
                str(args.project_root / "scripts"),
                existing_pythonpath,
            )
            if item
        )
        command = [
            str(args.python),
            str(args.project_root / "scripts/run_sam3_mask_worker.py"),
            "--checkpoint", str(args.checkpoint),
            "--plan", str(plan),
        ]
        process = subprocess.Popen(
            command, cwd=args.project_root, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        running.append((gpu, process, log, log_path))
        print(
            f"START worker gpu={gpu} jobs={len(worker_jobs)} "
            f"frames={loads[worker_index]}",
            flush=True,
        )

    failures = []
    for gpu, process, log, log_path in running:
        code = process.wait()
        log.close()
        if code:
            failures.append((gpu, code, str(log_path)))
        else:
            print(f"DONE worker gpu={gpu}", flush=True)
    if failures:
        raise RuntimeError(f"persistent SAM3 workers failed: {failures}")


def main() -> None:
    args = parse_args()
    validate_direct_video_checkpoint(args.checkpoint)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise RuntimeError("at least one GPU is required")
    jobs = []
    segments = []
    for directory in sorted(path for path in args.segments_root.iterdir() if path.is_dir()):
        match = SEGMENT_RE.match(directory.name)
        if match is None:
            continue
        active = match.group(2)
        start, end = int(match.group(3)), int(match.group(4))
        sides = ("left", "right") if active == "both" else (active,)
        segments.append((directory.name, start, end, sides))
        for side in sides:
            output = args.output_root / directory.name / side
            stable = output / "sam3_stabilized_mask"
            if mask_count(stable) == end - start:
                print(f"SKIP {directory.name}/{side}: {end - start} masks already exist", flush=True)
                continue
            jobs.append((directory.name, side, start, end, output))

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.persistent_workers:
        run_persistent_workers(args, jobs, gpus)
        jobs = []
    running: dict[str, tuple[subprocess.Popen, object, str, str]] = {}
    available = list(gpus)
    failures = []
    while jobs or running:
        while jobs and available:
            name, side, start, end, output = jobs.pop(0)
            gpu = available.pop(0)
            output.mkdir(parents=True, exist_ok=True)
            log_path = output / "sam3.log"
            log = log_path.open("w", encoding="utf-8")
            segment = args.segments_root / name
            candidate = args.candidates_root / name
            command = [
                str(args.python),
                str(args.project_root / "scripts" / "compare_sam3_direct_masks.py"),
                "--video", str(segment / "frames"),
                "--frames", str(segment / "frames"),
                "--hdf5", str(candidate / "annotations_rgb_aligned.hdf5"),
                "--hand", side,
                "--sam2-masks", str(segment / "human_mask"),
                "--sam3-checkpoint", str(args.checkpoint),
                "--output", str(output),
                "--prompt-stride", str(args.prompt_stride),
                "--prompt-mode", args.prompt_mode,
                *(["--production-output"] if args.production_output else []),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                item
                for item in (
                    str(args.project_root / "src"),
                    str(args.checkpoint.parent.parent),
                    existing_pythonpath,
                )
                if item
            )
            process = subprocess.Popen(
                command, cwd=args.project_root, env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
            running[gpu] = (process, log, name, side)
            print(f"START gpu={gpu} {name}/{side} frames={end-start}", flush=True)

        completed = []
        for gpu, (process, log, name, side) in running.items():
            code = process.poll()
            if code is None:
                continue
            log.close()
            completed.append(gpu)
            if code != 0:
                failures.append((name, side, code))
                print(f"FAIL gpu={gpu} {name}/{side} code={code}", flush=True)
            else:
                print(f"DONE gpu={gpu} {name}/{side}", flush=True)
        for gpu in completed:
            del running[gpu]
            available.append(gpu)
        if running and not completed:
            time.sleep(1.0)

    if failures:
        raise RuntimeError(f"SAM3 jobs failed: {failures}")

    for name, start, end, sides in segments:
        union_dir = args.output_root / name / "union_mask"
        union_dir.mkdir(parents=True, exist_ok=True)
        for index in range(end - start):
            union = None
            for side in sides:
                path = args.output_root / name / side / "sam3_stabilized_mask" / f"{index:05d}.png"
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(path)
                union = mask > 0 if union is None else union | (mask > 0)
            cv2.imwrite(str(union_dir / f"{index:05d}.png"), union.astype(np.uint8) * 255)
        print(f"UNION {name}: {end-start} frames", flush=True)


if __name__ == "__main__":
    main()
