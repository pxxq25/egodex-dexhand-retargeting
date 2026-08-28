#!/usr/bin/env python3
"""Run the frozen HTS → SAM3.1 → ProPainter → Shadow production path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


BACKEND_NAME = "sam3.1_direct_geometry_refined_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("RUN " + json.dumps(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def timed_run(
    timings: dict[str, float],
    name: str,
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> None:
    started = time.monotonic()
    try:
        run(command, environment=environment)
    finally:
        elapsed = time.monotonic() - started
        timings[name] = elapsed
        print(f"TIMING {name}={elapsed:.3f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-devices", type=int, nargs="+", required=True)
    parser.add_argument("--chunk-overlap-frames", type=int, default=12)
    parser.add_argument("--minimum-visible-landmarks", type=int, default=1)
    parser.add_argument("--entry-padding-frames", type=int, default=8)
    parser.add_argument("--exit-padding-frames", type=int, default=8)
    parser.add_argument(
        "--rgb-visibility-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "allow direct RGB detections to create processing intervals; disable "
            "when other people's hands can appear in the scene"
        ),
    )
    parser.add_argument("--skip-unrenderable", action="store_true")
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--sam3-prompt-stride", type=int, default=5)
    parser.add_argument(
        "--sam3-prompt-mode",
        choices=("geometry", "keypoints"),
        default="geometry",
    )
    parser.add_argument(
        "--sam3-persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse one loaded SAM3 model per GPU (pixel-equivalent fast path)",
    )
    parser.add_argument(
        "--propainter-persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reuse loaded RAFT/flow-completion/ProPainter models across logical "
            "chunks (frame-equivalent fast path)"
        ),
    )
    return parser.parse_args()


def adaptive_command(
    args: argparse.Namespace, python: Path, project: Path, *, phase: str | None = None
) -> list[str]:
    command = [
        str(python),
        str(project / "scripts/run_egoquest_adaptive_recording.py"),
        "--recording", str(args.recording),
        "--workspace", str(args.workspace),
        "--run-name", args.run_name,
        "--workers", str(args.workers),
        "--gpu-devices", *(str(value) for value in args.gpu_devices),
        "--chunk-overlap-frames", str(args.chunk_overlap_frames),
        "--minimum-visible-landmarks", str(args.minimum_visible_landmarks),
        "--entry-padding-frames", str(args.entry_padding_frames),
        "--exit-padding-frames", str(args.exit_padding_frames),
    ]
    if not args.rgb_visibility_fusion:
        command.append("--disable-rgb-visibility-fusion")
    if phase is not None:
        command.extend(["--phase", phase])
    if args.skip_unrenderable:
        command.append("--skip-unrenderable")
    return command


def main() -> None:
    args = parse_args()
    project = args.workspace / "project"
    python = args.workspace / ".venv/bin/python"
    checkpoint = args.sam3_checkpoint or (
        args.workspace / "third_party/sam3/checkpoints/sam3.pt"
    )
    for required in (
        args.recording / "camera.mp4",
        project / "scripts/run_egoquest_adaptive_recording.py",
        project / "scripts/run_sam3_episode_masks.py",
        project / "scripts/recompose_episode_with_sam3.py",
        checkpoint,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    episode = args.recording.name
    run_root = args.workspace / "runs" / args.run_name / episode
    candidates = args.workspace / "candidates" / args.run_name / episode
    segments = run_root / "segments"
    sam3_root = run_root / "sam3_masks"
    gpu_csv = ",".join(str(value) for value in args.gpu_devices)
    base_environment = dict(os.environ)
    timings: dict[str, float] = {}

    # The standard renderer remains the source of the exact robot trajectory,
    # RGB, alpha, and masks. Stop before its legacy SAM2 stage.
    render_environment = dict(base_environment)
    render_environment["EGODEX_BASELINE_STOP_STAGE"] = "render"
    timed_run(
        timings,
        "render",
        adaptive_command(args, python, project),
        environment=render_environment,
    )

    # Reuse the exact SAM3 path that produced the approved diagnostics. The
    # production flag skips only raw-mask PNGs and comparison-video encoding.
    timed_run(
        timings,
        "sam3_masks",
        [
            str(python),
            str(project / "scripts/run_sam3_episode_masks.py"),
            "--project-root", str(project),
            "--python", str(python),
            "--checkpoint", str(checkpoint),
            "--candidates-root", str(candidates),
            "--segments-root", str(segments),
            "--output-root", str(sam3_root),
            "--gpus", gpu_csv,
            "--prompt-stride", str(args.sam3_prompt_stride),
            "--prompt-mode", args.sam3_prompt_mode,
            "--production-output",
            *(["--persistent-workers"] if args.sam3_persistent_workers else []),
        ],
        environment=base_environment,
    )
    timed_run(
        timings,
        "propainter_and_composite",
        [
            str(python),
            str(project / "scripts/recompose_episode_with_sam3.py"),
            "--project-root", str(project),
            "--candidates-root", str(candidates),
            "--segments-root", str(segments),
            "--sam3-root", str(sam3_root),
            "--propainter-root", str(args.workspace / "third_party/ProPainter"),
            "--gpus", gpu_csv,
            *(
                ["--persistent-workers"]
                if args.propainter_persistent_workers
                else []
            ),
        ],
        environment=base_environment,
    )

    # Completed chunk videos make the visual workers no-ops; this final pass
    # only performs the existing lossless timeline assembly and QA outputs.
    timed_run(
        timings,
        "final_assembly",
        adaptive_command(args, python, project, phase="visual"),
        environment=base_environment,
    )

    episode_metadata_path = run_root / "episode.json"
    episode_metadata = json.loads(episode_metadata_path.read_text())
    episode_metadata.update(
        {
            "mask_backend": BACKEND_NAME,
            "sam3_checkpoint": str(checkpoint.resolve()),
            "sam3_prompt_stride": args.sam3_prompt_stride,
            "sam3_prompt_mode": args.sam3_prompt_mode,
            "sam3_persistent_workers": bool(args.sam3_persistent_workers),
            "propainter_persistent_workers": bool(
                args.propainter_persistent_workers
            ),
            "stage_seconds": timings,
        }
    )
    episode_metadata_path.write_text(json.dumps(episode_metadata, indent=2) + "\n")
    (run_root / "pipeline.json").write_text(
        json.dumps(
            {
                "backend": BACKEND_NAME,
                "completed_at": utc_now(),
                "checkpoint": str(checkpoint.resolve()),
                "prompt_stride": args.sam3_prompt_stride,
                "prompt_mode": args.sam3_prompt_mode,
                "production_output": True,
                "persistent_workers": bool(args.sam3_persistent_workers),
                "sam3_persistent_workers": bool(args.sam3_persistent_workers),
                "propainter_persistent_workers": bool(
                    args.propainter_persistent_workers
                ),
                "stage_seconds": timings,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"COMPLETE {episode}: {BACKEND_NAME}", flush=True)


if __name__ == "__main__":
    main()
