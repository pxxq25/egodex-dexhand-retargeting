#!/usr/bin/env python3
"""Prepare and execute a resumable, multi-host Interact episode batch.

The manifest is immutable after preparation.  Mutable state is kept in one
JSON file and one atomic claim directory per episode, which avoids concurrent
manifest rewrites when several H100 hosts consume the same batch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Callable, Iterator


REQUIRED_INPUTS = ("camera.mp4", "aligned_frames.parquet", "session.json")
FINAL_VIDEO = Path("final/composite_full_recording.mp4")
SIDE_BY_SIDE_VIDEO = Path("final/human_left_robot_right_full_recording.mp4")
DIAGNOSTIC_VIDEO = Path("final/keypoints_mask_retarget_full_recording.mp4")
MANIFEST_VERSION = 2
SAM3_BACKEND = "sam3.1_direct_geometry_refined_v1"
SAM3_CHECKPOINT = "sam3.pt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return sys.maxsize, path.name


def parse_episode_selector(value: str | None) -> set[str] | None:
    """Parse ``1,3-5`` into canonical numeric episode names."""

    if value is None:
        return None
    selected: set[str] = set()
    for field in value.split(","):
        field = field.strip()
        if not field:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", field)
        if match is None:
            raise ValueError(f"invalid episode selector: {field!r}")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if last < first:
            raise ValueError(f"descending episode range: {field!r}")
        selected.update(str(index) for index in range(first, last + 1))
    return selected


def ffprobe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,"
            "nb_frames,duration", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}")
    return streams[0]


def integer_frames(probe: dict[str, object]) -> int:
    raw = str(probe.get("nb_frames", ""))
    if raw.isdigit():
        return int(raw)
    rate = str(probe.get("r_frame_rate", "0/1")).split("/", 1)
    fps = float(rate[0]) / float(rate[1])
    return int(round(float(probe.get("duration", 0.0)) * fps))


def parquet_frames(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "manifest preparation requires pyarrow; use the pipeline venv"
        ) from error
    return int(pq.read_metadata(path).num_rows)


def estimate_scratch_bytes(width: int, height: int, frames: int) -> int:
    """Conservative estimate calibrated on completed episodes 1 and 6."""

    bytes_per_frame = 2_800_000 if width * height > 640 * 480 else 1_100_000
    return frames * bytes_per_frame


def inspect_episode(recording: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED_INPUTS if not (recording / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{recording}: missing {', '.join(missing)}")
    probe = ffprobe(recording / "camera.mp4")
    video_frames = integer_frames(probe)
    aligned_frames = parquet_frames(recording / "aligned_frames.parquet")
    if video_frames != aligned_frames:
        raise RuntimeError(
            f"{recording}: camera has {video_frames} frames but parquet has "
            f"{aligned_frames}"
        )
    width = int(probe["width"])
    height = int(probe["height"])
    return {
        "episode": recording.name,
        "recording": str(recording.resolve()),
        "frames": aligned_frames,
        "width": width,
        "height": height,
        "fps": str(probe.get("r_frame_rate", "30/1")),
        "duration_seconds": float(probe.get("duration", aligned_frames / 30.0)),
        "input_bytes": sum(
            item.stat().st_size for item in recording.rglob("*") if item.is_file()
        ),
        "estimated_scratch_bytes": estimate_scratch_bytes(
            width, height, aligned_frames
        ),
    }


def git_revision(project: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True,
        text=True, check=False,
    )
    if result.returncode != 0:
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=project, capture_output=True,
        text=True, check=False,
    )
    suffix = "+dirty" if dirty.stdout.strip() else ""
    return result.stdout.strip() + suffix


def pipeline_fingerprint(project: Path) -> str:
    """Hash the executable Python pipeline, including uncommitted changes."""

    files = list((project / "src").rglob("*.py"))
    files.extend([
        project / "scripts/run_egoquest_adaptive_recording.py",
        project / "scripts/run_egoquest_sam3_recording.py",
        project / "scripts/run_sam3_episode_masks.py",
        project / "scripts/run_sam3_mask_worker.py",
        project / "scripts/run_propainter_jobs.py",
        project / "scripts/recompose_episode_with_sam3.py",
        project / "scripts/compare_sam3_direct_masks.py",
        project / "scripts/assemble_keypoint_mask_retarget_episode.py",
        project / "scripts/preflight_sapien_vulkan.py",
        project / "scripts/run_interact_batch.py",
    ])
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        if not path.is_file():
            raise FileNotFoundError(f"pipeline source missing: {path}")
        digest.update(str(path.relative_to(project)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_manifest(args: argparse.Namespace) -> Path:
    selected = parse_episode_selector(args.episodes)
    recordings = sorted(
        (item for item in args.dataset_root.iterdir() if item.is_dir()),
        key=numeric_key,
    )
    if selected is not None:
        recordings = [item for item in recordings if item.name in selected]
        missing = selected - {item.name for item in recordings}
        if missing:
            raise FileNotFoundError(f"episodes not found: {sorted(missing, key=int)}")
    episodes = [inspect_episode(recording) for recording in recordings]
    if not episodes:
        raise RuntimeError("no episodes selected")
    batch_root = args.workspace / "batches" / args.batch_name
    manifest_path = batch_root / "manifest.json"
    manifest = {
        "version": MANIFEST_VERSION,
        "created_at": utc_now(),
        "batch_name": args.batch_name,
        "dataset_root": str(args.dataset_root.resolve()),
        "workspace": str(args.workspace.resolve()),
        "project_revision": git_revision(args.project),
        "pipeline_sha256": pipeline_fingerprint(args.project),
        "settings": {
            "workers_per_episode": args.workers,
            "minimum_gpu_free_gib": args.minimum_gpu_free_gib,
            "maximum_gpus_per_host": args.maximum_gpus,
            "chunk_overlap_frames": args.chunk_overlap_frames,
            "diagnostic": not args.no_diagnostic,
            "strict_zero_skips": True,
            "scheduler": "one_episode_lane_per_preflighted_gpu",
            "host_local_dependencies": True,
            "host_runtime_cache_scope": "shared_across_batch_versions",
            "incremental_shared_checkpoints": True,
            "mask_backend": SAM3_BACKEND,
            "sam3_checkpoint": SAM3_CHECKPOINT,
            "sam3_prompt_stride": args.sam3_prompt_stride,
            "sam3_prompt_mode": args.sam3_prompt_mode,
            "sam3_production_output": True,
            "sam3_persistent_workers": True,
            "propainter_persistent_workers": True,
        },
        "summary": {
            "episodes": len(episodes),
            "frames": sum(int(item["frames"]) for item in episodes),
            "source_seconds": sum(
                float(item["duration_seconds"]) for item in episodes
            ),
            "input_bytes": sum(int(item["input_bytes"]) for item in episodes),
            "estimated_scratch_bytes": sum(
                int(item["estimated_scratch_bytes"]) for item in episodes
            ),
        },
        "episodes": episodes,
    }
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"{manifest_path} already exists; use a new batch name or --force"
        )
    atomic_json(manifest_path, manifest)
    for episode in episodes:
        status_path = batch_root / "status" / f"{episode['episode']}.json"
        if not status_path.exists():
            atomic_json(status_path, {
                "episode": episode["episode"], "state": "pending",
                "updated_at": utc_now(),
            })
    print(json.dumps(manifest["summary"], indent=2))
    print(manifest_path)
    return manifest_path


def load_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if value.get("version") not in (1, MANIFEST_VERSION):
        raise RuntimeError(f"unsupported manifest version in {path}")
    return value


def validate_video(path: Path, expected_frames: int) -> list[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return [f"missing video: {path}"]
    try:
        actual_frames = integer_frames(ffprobe(path))
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        return [f"unreadable video {path}: {error}"]
    if actual_frames != expected_frames:
        return [
            f"frame mismatch {path}: expected {expected_frames}, got {actual_frames}"
        ]
    return []


def validate_episode(
    run: Path,
    expected_frames: int,
    *,
    require_diagnostic: bool,
    mask_backend: str | None = None,
) -> list[str]:
    errors: list[str] = []
    episode_json = run / "episode.json"
    if not episode_json.is_file():
        return [f"missing metadata: {episode_json}"]
    try:
        metadata = json.loads(episode_json.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return [f"invalid metadata {episode_json}: {error}"]
    if int(metadata.get("total_frames", -1)) != expected_frames:
        errors.append("episode.json total_frames does not match the manifest")
    skipped = list((run / "skipped_unrenderable").glob("*/skip_reason.json"))
    if skipped:
        errors.append(f"{len(skipped)} skipped/unrenderable segments remain")
    chunks = metadata.get("chunks", [])
    if mask_backend is not None:
        pipeline_path = run / "pipeline.json"
        if not pipeline_path.is_file():
            errors.append(f"missing pipeline provenance: {pipeline_path}")
        else:
            try:
                actual_backend = json.loads(pipeline_path.read_text()).get("backend")
            except (json.JSONDecodeError, OSError) as error:
                errors.append(f"invalid pipeline provenance {pipeline_path}: {error}")
            else:
                if actual_backend != mask_backend:
                    errors.append(
                        f"mask backend mismatch: expected {mask_backend}, "
                        f"got {actual_backend}"
                    )
    segment_root = run / "segments"
    for chunk in chunks:
        name = (
            f"{int(chunk['index']):03d}_{chunk['mode']}_"
            f"{int(chunk['start']):05d}_{int(chunk['end']):05d}"
        )
        length = int(chunk["end"]) - int(chunk["start"])
        segment = segment_root / name
        if mask_backend == SAM3_BACKEND and not (
            segment / "sam3_recompose.json"
        ).is_file():
            errors.append(f"missing SAM3 provenance for {name}")
        errors.extend(validate_video(segment / "final/composite_full.mp4", length))
        mask_count = len(list((segment / "human_mask").glob("*.png")))
        if mask_count != length:
            errors.append(
                f"mask count mismatch {name}: expected {length}, got {mask_count}"
            )
    errors.extend(validate_video(run / FINAL_VIDEO, expected_frames))
    errors.extend(validate_video(run / SIDE_BY_SIDE_VIDEO, expected_frames))
    if require_diagnostic:
        errors.extend(validate_video(run / DIAGNOSTIC_VIDEO, expected_frames))
    return errors


def gpu_free_mib() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, check=True, timeout=15,
    )
    available = {}
    for line in result.stdout.splitlines():
        index, free = (field.strip() for field in line.split(",", 1))
        available[int(index)] = int(free)
    return available


def select_gpus(args: argparse.Namespace) -> list[int]:
    free = gpu_free_mib()
    requested = args.gpu_devices
    candidates = requested if requested is not None else sorted(free)
    minimum = int(args.minimum_gpu_free_gib * 1024)
    selected = [gpu for gpu in candidates if free.get(gpu, 0) >= minimum]
    selected = selected[: args.maximum_gpus]
    if len(selected) < args.minimum_gpus:
        details = ", ".join(f"{gpu}:{mib / 1024:.1f}GiB" for gpu, mib in free.items())
        raise RuntimeError(
            f"only {len(selected)} eligible GPUs; need {args.minimum_gpus}; "
            f"free memory is {details}"
        )
    return selected


def ensure_link(path: Path, target: Path) -> None:
    if not target.exists():
        raise FileNotFoundError(f"scratch dependency target is missing: {target}")
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise RuntimeError(f"unexpected scratch link: {path} -> {path.resolve()}")
        return
    if path.exists():
        raise FileExistsError(f"scratch dependency path already exists: {path}")
    try:
        path.symlink_to(target.resolve(), target_is_directory=True)
    except FileExistsError:
        # Multiple GPU lanes initialize the same host workspace concurrently.
        if not path.is_symlink() or path.resolve() != target.resolve():
            raise


def dependency_signature(
    shared: Path, *, include_sam3: bool
) -> dict[str, object]:
    paths = [
        shared / "third_party/sam2/checkpoints/sam2.1_hiera_small.pt",
        shared / "third_party/sam2/sam2/sam2_video_predictor.py",
        shared / "third_party/ProPainter/weights/ProPainter.pth",
        shared / "third_party/ProPainter/weights/raft-things.pth",
        shared / "third_party/ProPainter/weights/recurrent_flow_completion.pth",
        shared / "third_party/ProPainter/inference_propainter.py",
        shared / "third_party/dex-retargeting/assets/robots/assembly/ur5e_shadow/ur5e_shadow_left_hand_glb.urdf",
        shared / "third_party/dex-retargeting/assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb.urdf",
    ]
    if include_sam3:
        paths.extend(
            [
                shared / f"third_party/sam3/checkpoints/{SAM3_CHECKPOINT}",
                shared / "third_party/sam3/sam3/model_builder.py",
            ]
        )
    signature = {}
    for path in paths:
        stat = path.stat()
        signature[str(path.relative_to(shared))] = {
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        }
    return signature


def copy_sam3_runtime(source: Path, destination: Path) -> None:
    """Stage SAM3 code and only the direct-video checkpoint on local NVMe."""

    destination.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "rsync", "-a", "--exclude", "checkpoints/",
            f"{source}/", f"{destination}/",
        ],
        check=False,
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)
    source_checkpoint = source / f"checkpoints/{SAM3_CHECKPOINT}"
    destination_checkpoint = destination / f"checkpoints/{SAM3_CHECKPOINT}"
    destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if (
        not destination_checkpoint.is_file()
        or destination_checkpoint.stat().st_size != source_checkpoint.stat().st_size
        or destination_checkpoint.stat().st_mtime_ns
        != source_checkpoint.stat().st_mtime_ns
    ):
        temporary = destination_checkpoint.with_name(
            f".{destination_checkpoint.name}.{os.getpid()}.tmp"
        )
        shutil.copy2(source_checkpoint, temporary)
        os.replace(temporary, destination_checkpoint)
    # This is a disposable host-local cache. Retaining an incompatible 3.3 GB
    # multiplex checkpoint wastes NVMe and makes accidental selection easier.
    for cached_checkpoint in destination_checkpoint.parent.glob("*.pt"):
        if cached_checkpoint != destination_checkpoint:
            cached_checkpoint.unlink()


def prepare_local_dependencies(
    scratch_root: Path, shared: Path, *, include_sam3: bool
) -> Path:
    """Copy immutable runtime assets to local NVMe once per host and batch."""

    cache_root = scratch_root / "host_cache"
    third_party = cache_root / "third_party"
    marker = cache_root / "dependencies.json"
    signature = dependency_signature(shared, include_sam3=include_sam3)
    required_directories = ["sam2", "ProPainter", "dex-retargeting"]
    if include_sam3:
        required_directories.append("sam3")

    def ready() -> bool:
        try:
            value = json.loads(marker.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        return value.get("signature") == signature and all(
            (third_party / name).is_dir()
            for name in required_directories
        )

    if ready():
        return third_party
    lock = cache_root / "dependencies.lock"
    cache_root.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if (time.time() - lock.stat().st_mtime) > 3600:
                shutil.rmtree(lock)
                continue
            time.sleep(1)
            if ready():
                return third_party
    try:
        if not ready():
            for name in ("sam2", "ProPainter", "dex-retargeting"):
                rsync(shared / "third_party" / name, third_party / name)
            if include_sam3:
                copy_sam3_runtime(
                    shared / "third_party/sam3", third_party / "sam3"
                )
            atomic_json(marker, {
                "prepared_at": utc_now(), "host": socket.gethostname(),
                "signature": signature,
            })
    finally:
        shutil.rmtree(lock)
    return third_party


def prewarm_provenance_cache(
    third_party: Path, cache_root: Path, *, include_sam3: bool
) -> None:
    from egodex_dexhand.provenance import sha256_file

    paths = [
        third_party / "sam2/checkpoints/sam2.1_hiera_small.pt",
        third_party / "ProPainter/weights/ProPainter.pth",
        third_party / "ProPainter/weights/raft-things.pth",
        third_party / "ProPainter/weights/recurrent_flow_completion.pth",
    ]
    if include_sam3:
        paths.append(third_party / f"sam3/checkpoints/{SAM3_CHECKPOINT}")
    previous = os.environ.get("EGODEX_SHA256_CACHE_DIR")
    os.environ["EGODEX_SHA256_CACHE_DIR"] = str(cache_root)
    try:
        for path in paths:
            sha256_file(path)
    finally:
        if previous is None:
            os.environ.pop("EGODEX_SHA256_CACHE_DIR", None)
        else:
            os.environ["EGODEX_SHA256_CACHE_DIR"] = previous


def prepare_scratch_workspace(
    scratch: Path, shared: Path, local_third_party: Path
) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    targets = {
        "project": shared / "project",
        "third_party": local_third_party,
        ".venv": shared / ".venv",
        ".venv-mediapipe": shared / ".venv-mediapipe",
    }
    for name, target in targets.items():
        ensure_link(scratch / name, target)


def rsync(
    source: Path, destination: Path, *, tolerate_vanished: bool = False
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["rsync", "-a", "--partial", f"{source}/", f"{destination}/"],
        check=False,
    )
    # Exit 24 means a source file vanished while rsync enumerated a live
    # directory. That is expected for periodic snapshots while a stage swaps
    # temporary frames. Final promotion remains strict and never ignores it.
    if result.returncode and not (
        tolerate_vanished and result.returncode == 24
    ):
        raise subprocess.CalledProcessError(result.returncode, result.args)


def read_status(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"state": "pending"}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"state": "invalid"}


@contextmanager
def claim_episode(
    batch_root: Path, episode: str, *, stale_hours: float
) -> Iterator[bool]:
    claim = batch_root / "claims" / f"{episode}.lock"
    claim.parent.mkdir(parents=True, exist_ok=True)
    if claim.exists():
        age_hours = (time.time() - claim.stat().st_mtime) / 3600.0
        if age_hours >= stale_hours:
            shutil.rmtree(claim)
    try:
        claim.mkdir()
    except FileExistsError:
        yield False
        return
    atomic_json(claim / "owner.json", {
        "host": socket.gethostname(), "pid": os.getpid(), "claimed_at": utc_now(),
    })
    try:
        yield True
    finally:
        if claim.exists():
            shutil.rmtree(claim)


def stage_recording(source: Path, scratch_dataset: Path) -> Path:
    destination = scratch_dataset / source.name
    rsync(source, destination)
    return destination


def run_logged(
    command: list[str],
    log: Path,
    environment: dict[str, str],
    *,
    checkpoint: Callable[[], None] | None = None,
    checkpoint_interval_seconds: float = 300.0,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, env=environment,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    returncode = process.wait(timeout=checkpoint_interval_seconds)
                    break
                except subprocess.TimeoutExpired:
                    if checkpoint is not None:
                        checkpoint()
        except BaseException:
            # Never orphan a GPU-heavy renderer/inpainter if orchestration or
            # checkpointing fails. Every command owns a dedicated process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if checkpoint is not None:
            checkpoint()
        handle.write(f"[{utc_now()}] EXIT {returncode}\n")
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def process_episode(
    args: argparse.Namespace,
    manifest: dict[str, object],
    episode: dict[str, object],
    gpus: list[int],
) -> None:
    batch_name = str(manifest["batch_name"])
    settings = manifest["settings"]
    mask_backend = str(settings.get("mask_backend", "sam2.1_small"))
    include_sam3 = mask_backend == SAM3_BACKEND
    shared = Path(str(manifest["workspace"]))
    project = shared / "project"
    project_src = str(project / "src")
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    scratch_root = args.scratch_root / batch_name
    scratch_workspace = scratch_root / "workspace"
    runtime_cache_root = args.scratch_root / "_host_runtime"
    local_third_party = prepare_local_dependencies(
        runtime_cache_root, shared, include_sam3=include_sam3
    )
    prewarm_provenance_cache(
        local_third_party,
        runtime_cache_root / "host_cache/sha256",
        include_sam3=include_sam3,
    )
    prepare_scratch_workspace(scratch_workspace, shared, local_third_party)
    scratch_recording = stage_recording(
        Path(str(episode["recording"])), scratch_root / "datasets" / "interact"
    )
    local_run = scratch_workspace / "runs" / batch_name / str(episode["episode"])
    local_candidates = (
        scratch_workspace / "candidates" / batch_name / str(episode["episode"])
    )
    shared_run = shared / "runs" / batch_name / str(episode["episode"])
    shared_candidates = shared / "candidates" / batch_name / str(episode["episode"])
    if shared_run.exists() and not local_run.exists():
        rsync(shared_run, local_run)
    if shared_candidates.exists() and not local_candidates.exists():
        rsync(shared_candidates, local_candidates)
    python = shared / ".venv/bin/python"
    environment = dict(os.environ)
    environment["EGODEX_PYTHON"] = str(python)
    environment["EGODEX_MEDIAPIPE_PYTHON"] = str(
        shared / ".venv-mediapipe/bin/python"
    )
    environment["EGODEX_SHA256_CACHE_DIR"] = str(
        runtime_cache_root / "host_cache/sha256"
    )
    runner = (
        project / "scripts/run_egoquest_sam3_recording.py"
        if include_sam3
        else project / "scripts/run_egoquest_adaptive_recording.py"
    )
    command = [
        str(python), str(runner),
        "--recording", str(scratch_recording),
        "--workspace", str(scratch_workspace),
        "--run-name", batch_name,
        "--workers", str(min(args.workers, len(gpus))),
        "--gpu-devices", *(str(gpu) for gpu in gpus),
        "--chunk-overlap-frames", str(args.chunk_overlap_frames),
        "--skip-unrenderable",
    ]
    if include_sam3:
        command.extend(
            [
                "--sam3-checkpoint",
                str(
                    scratch_workspace
                    / "third_party/sam3/checkpoints"
                    / str(settings["sam3_checkpoint"])
                ),
                "--sam3-prompt-stride", str(settings["sam3_prompt_stride"]),
                "--sam3-prompt-mode", str(settings["sam3_prompt_mode"]),
                "--sam3-persistent-workers",
                "--propainter-persistent-workers",
            ]
        )
    batch_root = shared / "batches" / batch_name
    log = batch_root / "logs" / f"{episode['episode']}_{socket.gethostname()}.log"

    def checkpoint() -> None:
        # Make completed chunks resumable from another host while the episode
        # is still running. An interrupted MP4 may be copied, but validation
        # never accepts it and the adaptive runner regenerates it on resume.
        if local_candidates.exists():
            rsync(
                local_candidates, shared_candidates, tolerate_vanished=True
            )
        if local_run.exists():
            rsync(local_run, shared_run, tolerate_vanished=True)

    run_logged(
        command, log, environment, checkpoint=checkpoint,
        checkpoint_interval_seconds=args.checkpoint_interval_seconds,
    )
    if not args.no_diagnostic:
        diagnostic = local_run / DIAGNOSTIC_VIDEO
        diagnostic_command = [
            str(python), str(project / "scripts/assemble_keypoint_mask_retarget_episode.py"),
            "--recording", str(scratch_recording), "--run", str(local_run),
            "--candidates", str(local_candidates), "--output", str(diagnostic),
        ]
        run_logged(diagnostic_command, log, environment)
    errors = validate_episode(
        local_run,
        int(episode["frames"]),
        require_diagnostic=not args.no_diagnostic,
        mask_backend=mask_backend if include_sam3 else None,
    )
    if errors:
        raise RuntimeError("local validation failed:\n" + "\n".join(errors))
    rsync(local_candidates, shared_candidates)
    rsync(local_run, shared_run)
    errors = validate_episode(
        shared_run,
        int(episode["frames"]),
        require_diagnostic=not args.no_diagnostic,
        mask_backend=mask_backend if include_sam3 else None,
    )
    if errors:
        raise RuntimeError("promoted validation failed:\n" + "\n".join(errors))
    if not args.keep_scratch:
        shutil.rmtree(local_run)
        shutil.rmtree(local_candidates)
        shutil.rmtree(scratch_recording)


def worker(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    shared = Path(str(manifest["workspace"]))
    actual_fingerprint = pipeline_fingerprint(shared / "project")
    if actual_fingerprint != manifest.get("pipeline_sha256"):
        raise RuntimeError(
            "pipeline source changed after manifest preparation; prepare a new "
            "batch name so every episode uses one exact pipeline version"
        )
    settings = manifest["settings"]
    mask_backend = str(settings.get("mask_backend", "sam2.1_small"))
    args.workers = int(settings["workers_per_episode"])
    args.chunk_overlap_frames = int(settings["chunk_overlap_frames"])
    args.no_diagnostic = not bool(settings["diagnostic"])
    batch_root = args.manifest.parent
    selected = parse_episode_selector(args.episodes)
    gpus = select_gpus(args)
    print(f"{socket.gethostname()}: selected physical GPUs {gpus}", flush=True)
    completed_this_run = 0
    for episode in manifest["episodes"]:
        name = str(episode["episode"])
        if selected is not None and name not in selected:
            continue
        if args.limit is not None and completed_this_run >= args.limit:
            break
        status_path = batch_root / "status" / f"{name}.json"
        status = read_status(status_path)
        shared_run = Path(str(manifest["workspace"])) / "runs" / str(
            manifest["batch_name"]
        ) / name
        if status.get("state") == "completed":
            errors = validate_episode(
                shared_run, int(episode["frames"]),
                require_diagnostic=not args.no_diagnostic,
                mask_backend=(
                    mask_backend if mask_backend == SAM3_BACKEND else None
                ),
            )
            if not errors:
                continue
        if status.get("state") == "failed" and not args.retry_failed:
            continue
        with claim_episode(batch_root, name, stale_hours=args.stale_lock_hours) as claimed:
            if not claimed:
                continue
            started = time.monotonic()
            atomic_json(status_path, {
                "episode": name, "state": "running", "host": socket.gethostname(),
                "pid": os.getpid(), "gpus": gpus, "started_at": utc_now(),
                "updated_at": utc_now(),
            })
            try:
                process_episode(args, manifest, episode, gpus)
            except Exception as error:
                atomic_json(status_path, {
                    "episode": name, "state": "failed", "host": socket.gethostname(),
                    "gpus": gpus, "updated_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "error": f"{type(error).__name__}: {error}",
                })
                print(f"FAILED episode {name}: {error}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    raise
            else:
                atomic_json(status_path, {
                    "episode": name, "state": "completed", "host": socket.gethostname(),
                    "gpus": gpus, "updated_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "run": str(shared_run),
                })
                completed_this_run += 1
                print(f"COMPLETE episode {name}", flush=True)


def nvidia_icd() -> Path:
    candidates = (
        Path("/etc/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/local/share/vulkan/icd.d/nvidia_icd.json"),
    )
    try:
        return next(path for path in candidates if path.is_file())
    except StopIteration as error:
        raise RuntimeError("no NVIDIA Vulkan ICD is installed on this host") from error


def preflight_physical_gpu(
    shared: Path, gpu: int, *, timeout: float, retries: int
) -> tuple[int, bool, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["VK_ICD_FILENAMES"] = str(nvidia_icd())
    environment["PYTHONPATH"] = str(shared / "project/src")
    command = [
        str(shared / ".venv/bin/python"),
        str(shared / "project/scripts/preflight_sapien_vulkan.py"),
        "--device", "cuda:0",
    ]
    messages = []
    for attempt in range(1, retries + 1):
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
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
            messages.append(f"attempt {attempt}: timeout")
        else:
            if process.returncode == 0:
                return gpu, True, stdout.strip()
            detail = (stderr or stdout or "failed").strip().splitlines()
            messages.append(f"attempt {attempt}: {detail[-1] if detail else 'failed'}")
    return gpu, False, "; ".join(messages)


def host_worker(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    shared = Path(str(manifest["workspace"]))
    project_src = str(shared / "project/src")
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    actual_fingerprint = pipeline_fingerprint(shared / "project")
    if actual_fingerprint != manifest.get("pipeline_sha256"):
        raise RuntimeError(
            "pipeline source changed after manifest preparation; prepare a new batch"
        )
    selected = select_gpus(args)
    settings = manifest["settings"]
    include_sam3 = settings.get("mask_backend") == SAM3_BACKEND
    scratch_root = args.scratch_root / str(manifest["batch_name"])
    runtime_cache_root = args.scratch_root / "_host_runtime"
    local_third_party = prepare_local_dependencies(
        runtime_cache_root, shared, include_sam3=include_sam3
    )
    prewarm_provenance_cache(
        local_third_party,
        runtime_cache_root / "host_cache/sha256",
        include_sam3=include_sam3,
    )
    prepare_scratch_workspace(
        scratch_root / "workspace", shared, local_third_party
    )
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        preflights = list(pool.map(
            lambda gpu: preflight_physical_gpu(
                shared, gpu, timeout=args.preflight_timeout,
                retries=args.preflight_retries,
            ),
            selected,
        ))
    healthy = [gpu for gpu, passed, _ in preflights if passed]
    report = {
        "host": socket.gethostname(),
        "candidate_gpus": selected,
        "healthy_gpus": healthy,
        "vulkan_icd": str(nvidia_icd()),
        "local_third_party": str(local_third_party),
        "preflight": {
            str(gpu): {"passed": passed, "detail": detail}
            for gpu, passed, detail in preflights
        },
    }
    report_path = args.manifest.parent / "hosts" / f"{socket.gethostname()}.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    if not healthy:
        raise RuntimeError("no eligible GPU passed the SAPIEN Vulkan preflight")
    if args.check_only:
        return
    children = []
    for gpu in healthy:
        command = [
            sys.executable, str(Path(__file__).resolve()), "worker",
            "--manifest", str(args.manifest),
            "--scratch-root", str(args.scratch_root),
            "--gpu-devices", str(gpu),
            "--minimum-gpus", "1", "--maximum-gpus", "1",
            "--minimum-gpu-free-gib", str(args.minimum_gpu_free_gib),
            "--stale-lock-hours", str(args.stale_lock_hours),
            "--checkpoint-interval-seconds",
            str(args.checkpoint_interval_seconds),
        ]
        if args.episodes:
            command.extend(["--episodes", args.episodes])
        if args.limit_per_lane is not None:
            command.extend(["--limit", str(args.limit_per_lane)])
        if args.retry_failed:
            command.append("--retry-failed")
        if args.keep_scratch:
            command.append("--keep-scratch")
        if args.fail_fast:
            command.append("--fail-fast")
        children.append(subprocess.Popen(command))
    failures = [child.wait() for child in children]
    if any(failures):
        raise RuntimeError(f"one or more GPU episode lanes failed: {failures}")


def print_status(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    counts: dict[str, int] = {}
    rows = []
    for episode in manifest["episodes"]:
        name = str(episode["episode"])
        status = read_status(args.manifest.parent / "status" / f"{name}.json")
        state = str(status.get("state", "pending"))
        counts[state] = counts.get(state, 0) + 1
        if args.verbose:
            rows.append({"episode": name, **status})
    print(json.dumps({"counts": counts, "episodes": rows}, indent=2))


def add_common_settings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-gpu-free-gib", type=float, default=36.0)
    parser.add_argument("--maximum-gpus", type=int, default=4)
    parser.add_argument("--chunk-overlap-frames", type=int, default=12)
    parser.add_argument("--no-diagnostic", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--project", type=Path, required=True)
    prepare.add_argument("--batch-name", required=True)
    prepare.add_argument("--episodes", help="numeric list/ranges, for example 1-100")
    prepare.add_argument("--sam3-prompt-stride", type=int, default=5)
    prepare.add_argument(
        "--sam3-prompt-mode",
        choices=("geometry", "keypoints"),
        default="geometry",
    )
    prepare.add_argument("--force", action="store_true")
    add_common_settings(prepare)
    run = subparsers.add_parser("worker")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--scratch-root", type=Path, default=Path("/tmp/egodex_batch"))
    run.add_argument("--episodes", help="optional numeric list/ranges for this worker")
    run.add_argument("--gpu-devices", type=int, nargs="+")
    run.add_argument("--minimum-gpus", type=int, default=2)
    run.add_argument("--limit", type=int)
    run.add_argument("--stale-lock-hours", type=float, default=12.0)
    run.add_argument("--checkpoint-interval-seconds", type=float, default=300.0)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--keep-scratch", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    add_common_settings(run)
    host = subparsers.add_parser("host")
    host.add_argument("--manifest", type=Path, required=True)
    host.add_argument("--scratch-root", type=Path, default=Path("/tmp/egodex_batch"))
    host.add_argument("--episodes", help="optional numeric list/ranges for this host")
    host.add_argument("--gpu-devices", type=int, nargs="+")
    host.add_argument("--minimum-gpus", type=int, default=1)
    host.add_argument("--limit-per-lane", type=int)
    host.add_argument("--stale-lock-hours", type=float, default=12.0)
    host.add_argument("--checkpoint-interval-seconds", type=float, default=300.0)
    host.add_argument("--preflight-timeout", type=float, default=60.0)
    host.add_argument("--preflight-retries", type=int, default=2)
    host.add_argument("--retry-failed", action="store_true")
    host.add_argument("--keep-scratch", action="store_true")
    host.add_argument("--fail-fast", action="store_true")
    host.add_argument("--check-only", action="store_true")
    add_common_settings(host)
    status = subparsers.add_parser("status")
    status.add_argument("--manifest", type=Path, required=True)
    status.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_manifest(args)
    elif args.command == "worker":
        worker(args)
    elif args.command == "host":
        host_worker(args)
    else:
        print_status(args)


if __name__ == "__main__":
    main()
