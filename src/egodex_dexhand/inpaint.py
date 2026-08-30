from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MASK_DILATION = 10
DEFAULT_SEAM_FEATHER = 4


@dataclass(frozen=True)
class RobotAwareMaskSummary:
    frame_count: int
    human_pixels: int
    opaque_robot_covered_human_pixels: int
    residual_human_pixels: int
    removal_pixels: int
    residual_human_ratio_mean: float
    residual_human_ratio_p95: float
    residual_human_ratio_max: float
    removal_frame_ratio_max: float
    empty_removal_frames: int


@dataclass(frozen=True)
class InpaintChangeSummary:
    evaluated_pixels: int
    mean_absolute_change: float
    low_change_fraction: float
    frame_low_change_fraction_p95: float


def build_robot_context_frames(
    source_frame_dir: str | Path,
    robot_rgb_dir: str | Path,
    robot_alpha_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Hide covered human pixels from the inpainter's source context.

    A residual-only mask is insufficient when the unmasked pixels underneath
    the future robot still contain the original arm: a temporal inpainter can
    copy that foreground straight back into the visible residual. Feeding it a
    robot-precomposited context removes that leakage source. The final robot is
    composited again after inpainting, so this is only context preparation and
    does not change the renderer's ownership of final pixels.
    """

    source_paths = sorted(Path(source_frame_dir).glob("*.jpg"))
    rgb_paths = _numbered_images(Path(robot_rgb_dir))
    alpha_paths = _numbered_images(Path(robot_alpha_dir))
    if not source_paths:
        raise FileNotFoundError(f"no JPG frames in {source_frame_dir}")
    if not (len(source_paths) == len(rgb_paths) == len(alpha_paths)):
        raise ValueError("source, robot-RGB, and robot-alpha frame counts differ")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {destination}")
    for index, (source_path, rgb_path, alpha_path) in enumerate(
        zip(source_paths, rgb_paths, alpha_paths)
    ):
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        robot = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        alpha_raw = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
        if source is None or robot is None or alpha_raw is None:
            raise RuntimeError(f"could not decode robot-context input {index}")
        if alpha_raw.ndim == 3:
            alpha_raw = alpha_raw[..., 0]
        if source.shape != robot.shape or source.shape[:2] != alpha_raw.shape:
            raise ValueError(f"robot-context shape mismatch at frame {index}")
        alpha_max = float(np.iinfo(alpha_raw.dtype).max) if np.issubdtype(
            alpha_raw.dtype, np.integer
        ) else 1.0
        alpha = (
            alpha_raw.astype(np.float32) / max(alpha_max, 1.0)
        )[..., None]
        context = np.rint(
            robot.astype(np.float32) * alpha
            + source.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        output = destination / f"{index:05d}.jpg"
        if not cv2.imwrite(
            str(output), context, [cv2.IMWRITE_JPEG_QUALITY, 98]
        ):
            raise RuntimeError(f"could not write {output}")
    return destination


def _numbered_images(path: Path) -> list[Path]:
    values = sorted(path.glob("*.png"))
    if not values:
        raise FileNotFoundError(f"no PNG frames in {path}")
    return values


def build_robot_aware_removal_masks(
    human_mask_dir: str | Path,
    robot_alpha_dir: str | Path,
    output_dir: str | Path,
    *,
    opaque_threshold: float = 0.98,
    seam_dilation_pixels: int | None = None,
) -> RobotAwareMaskSummary:
    """Remove only human pixels that the opaque robot will not cover.

    Background under an opaque robot is unobservable and also irrelevant to
    the final composite. Keeping it out of the inpaint mask turns a persistent
    arm-sized hole into the usually narrow silhouette residual that is truly
    visible. A small seam band covers antialiased robot edges.
    """

    human_paths = _numbered_images(Path(human_mask_dir))
    alpha_paths = _numbered_images(Path(robot_alpha_dir))
    if len(human_paths) != len(alpha_paths):
        raise ValueError("human-mask and robot-alpha frame counts differ")
    if not 0.0 < opaque_threshold <= 1.0:
        raise ValueError("opaque_threshold must be in (0, 1]")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {destination}")

    human_counts = []
    covered_counts = []
    residual_counts = []
    removal_counts = []
    image_area = None
    for index, (human_path, alpha_path) in enumerate(zip(human_paths, alpha_paths)):
        human_raw = cv2.imread(str(human_path), cv2.IMREAD_GRAYSCALE)
        alpha_raw = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
        if human_raw is None or alpha_raw is None:
            raise RuntimeError(f"could not decode robot-aware mask input {index}")
        if alpha_raw.ndim == 3:
            alpha_raw = alpha_raw[..., 0]
        if human_raw.shape != alpha_raw.shape:
            raise ValueError(f"mask shape mismatch at frame {index}")
        if image_area is None:
            image_area = int(human_raw.size)
        alpha_max = float(np.iinfo(alpha_raw.dtype).max) if np.issubdtype(
            alpha_raw.dtype, np.integer
        ) else 1.0
        alpha = alpha_raw.astype(np.float32) / max(alpha_max, 1.0)
        human = human_raw > 0
        opaque_robot = alpha >= opaque_threshold
        residual = human & ~opaque_robot
        seam = (
            max(2, int(round(min(human.shape) * 0.00625)))
            if seam_dilation_pixels is None
            else int(seam_dilation_pixels)
        )
        if seam < 0:
            raise ValueError("seam_dilation_pixels must be non-negative")
        if seam:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * seam + 1, 2 * seam + 1)
            )
            removal = cv2.dilate(residual.astype(np.uint8), kernel) > 0
            human_envelope = cv2.dilate(human.astype(np.uint8), kernel) > 0
            removal &= human_envelope
        else:
            removal = residual
        output = destination / f"{index:05d}.png"
        if not cv2.imwrite(str(output), removal.astype(np.uint8) * 255):
            raise RuntimeError(f"could not write {output}")
        human_counts.append(int(np.count_nonzero(human)))
        covered_counts.append(int(np.count_nonzero(human & opaque_robot)))
        residual_counts.append(int(np.count_nonzero(residual)))
        removal_counts.append(int(np.count_nonzero(removal)))

    human_values = np.asarray(human_counts, dtype=np.float64)
    residual_values = np.asarray(residual_counts, dtype=np.float64)
    ratios = np.divide(
        residual_values,
        human_values,
        out=np.zeros_like(residual_values),
        where=human_values > 0,
    )
    assert image_area is not None
    summary = RobotAwareMaskSummary(
        frame_count=len(human_paths),
        human_pixels=int(np.sum(human_values)),
        opaque_robot_covered_human_pixels=int(np.sum(covered_counts)),
        residual_human_pixels=int(np.sum(residual_values)),
        removal_pixels=int(np.sum(removal_counts)),
        residual_human_ratio_mean=float(np.mean(ratios)),
        residual_human_ratio_p95=float(np.quantile(ratios, 0.95)),
        residual_human_ratio_max=float(np.max(ratios)),
        removal_frame_ratio_max=float(np.max(removal_counts) / image_area),
        empty_removal_frames=int(np.count_nonzero(np.asarray(removal_counts) == 0)),
    )
    # ProPainter treats every file in its mask directory as an image. Keep
    # provenance beside that directory so a JSON receipt cannot be decoded as
    # a mask frame.
    (destination.parent / f"{destination.name}_summary.json").write_text(
        json.dumps(asdict(summary), indent=2) + "\n"
    )
    return summary


def evaluate_inpaint_change(
    frames_dir: str | Path,
    masks_dir: str | Path,
    inpainted_video: str | Path,
    *,
    low_change_threshold: float = 10.0,
) -> InpaintChangeSummary:
    """Detect a failed inpainter that copied the removed foreground back."""

    frame_paths = sorted(Path(frames_dir).glob("*.jpg"))
    mask_paths = _numbered_images(Path(masks_dir))
    if len(frame_paths) != len(mask_paths):
        raise ValueError("source-frame and removal-mask counts differ")
    if low_change_threshold < 0:
        raise ValueError("low_change_threshold must be non-negative")
    reader = cv2.VideoCapture(str(inpainted_video))
    if not reader.isOpened():
        raise RuntimeError(f"could not open {inpainted_video}")
    changes = []
    low_change = 0
    evaluated = 0
    frame_fractions = []
    for index, (frame_path, mask_path) in enumerate(zip(frame_paths, mask_paths)):
        source = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        ok, result = reader.read()
        if source is None or mask is None or not ok:
            raise RuntimeError(f"could not decode inpaint-change frame {index}")
        if source.shape != result.shape or mask.shape != source.shape[:2]:
            raise ValueError(f"inpaint-change shape mismatch at frame {index}")
        selected = mask > 0
        count = int(np.count_nonzero(selected))
        if count == 0:
            frame_fractions.append(0.0)
            continue
        difference = np.mean(
            np.abs(source.astype(np.float32) - result.astype(np.float32)), axis=2
        )[selected]
        changes.append(difference)
        low = int(np.count_nonzero(difference < low_change_threshold))
        low_change += low
        evaluated += count
        frame_fractions.append(low / count)
    reader.release()
    if evaluated == 0:
        return InpaintChangeSummary(0, 0.0, 0.0, 0.0)
    all_changes = np.concatenate(changes)
    return InpaintChangeSummary(
        evaluated_pixels=evaluated,
        mean_absolute_change=float(np.mean(all_changes)),
        low_change_fraction=float(low_change / evaluated),
        frame_low_change_fraction_p95=float(np.quantile(frame_fractions, 0.95)),
    )


def propainter_environment() -> dict[str, str]:
    """Put ProPainter on the least-loaded physical CUDA device.

    SAPIEN workers intentionally expose one GPU through CUDA_VISIBLE_DEVICES,
    but inpainting is an independent subprocess and can safely use another
    device. This prevents unrelated jobs on the render GPU from turning a
    completed retarget/render into a falsely "unrenderable" chunk.
    """

    environment = dict(os.environ)
    requested = environment.get("EGODEX_INPAINT_CUDA_VISIBLE_DEVICES")
    if requested:
        environment["CUDA_VISIBLE_DEVICES"] = requested
    else:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            candidates = []
            for line in result.stdout.splitlines():
                index, free_memory = (int(value.strip()) for value in line.split(","))
                candidates.append((free_memory, index))
            if candidates:
                environment["CUDA_VISIBLE_DEVICES"] = str(max(candidates)[1])
        except (OSError, ValueError, subprocess.SubprocessError):
            # CPU-only tests and single-GPU hosts retain their supplied device.
            pass
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def soft_inpaint_alpha(
    mask: np.ndarray,
    mask_dilation: int = DEFAULT_MASK_DILATION,
    seam_feather: int = DEFAULT_SEAM_FEATHER,
) -> np.ndarray:
    """Create an alpha matte with solid coverage and an outward soft seam.

    ProPainter internally dilates its binary input mask.  Its raw output then
    changes pixels up to that hard dilation boundary and leaves the next pixel
    untouched, which exposes a visible contour.  This matte keeps the inner
    ``mask_dilation - seam_feather`` pixels fully inpainted and linearly blends
    only the remaining generated margin back to the source frame.

    The 10-pixel dilation and 4-pixel seam defaults are for 960x536 episode
    1029 frames: six pixels of solid safety coverage followed by a four-pixel
    transition.  Both values are explicit controls for other resolutions.
    """

    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {binary.shape}")
    mask_dilation = int(mask_dilation)
    seam_feather = int(seam_feather)
    if mask_dilation < 0:
        raise ValueError("mask_dilation must be non-negative")
    if not 0 <= seam_feather <= mask_dilation:
        raise ValueError("seam_feather must be between 0 and mask_dilation")
    if not binary.any():
        # Visibility padding and per-side fusion can legitimately produce an
        # empty union mask. ProPainter leaves that frame untouched, so the
        # exact seam-blending counterpart is a zero-alpha matte.
        return np.zeros(binary.shape, dtype=np.float32)

    # scipy.ndimage.binary_dilation, used by ProPainter, applies one-connected
    # pixel iteration at a time.  A 3x3 cross reproduces that footprint.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    solid_iterations = mask_dilation - seam_feather
    solid = binary.astype(np.uint8)
    if solid_iterations:
        solid = cv2.dilate(solid, kernel, iterations=solid_iterations)
    alpha = solid.astype(np.float32)
    frontier = solid > 0
    for step in range(1, seam_feather + 1):
        expanded = cv2.dilate(
            frontier.astype(np.uint8), kernel, iterations=1
        ) > 0
        ring = expanded & ~frontier
        alpha[ring] = 1.0 - step / float(seam_feather + 1)
        frontier = expanded
    return np.clip(alpha, 0.0, 1.0)


def blend_inpainted_frame(
    source_frame: np.ndarray,
    inpainted_frame: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Blend a raw inpaint result back to its source with a prepared matte."""

    source = np.asarray(source_frame)
    inpainted = np.asarray(inpainted_frame)
    alpha = np.asarray(alpha, dtype=np.float32)
    if source.shape != inpainted.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source and inpainted frames must be aligned BGR images")
    if alpha.shape != source.shape[:2]:
        raise ValueError("alpha and frame shapes do not match")
    if not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
        raise ValueError("alpha must be finite and within [0, 1]")
    weight = alpha[..., None]
    return np.clip(
        inpainted.astype(np.float32) * weight
        + source.astype(np.float32) * (1.0 - weight),
        0,
        255,
    ).astype(np.uint8)


def run_propainter(
    frames_dir: str | Path,
    masks_dir: str | Path,
    propainter_root: str | Path,
    output_dir: str | Path,
    python_executable: str,
    fps: float,
    fp16: bool = True,
    mask_dilation: int = DEFAULT_MASK_DILATION,
    seam_feather: int = DEFAULT_SEAM_FEATHER,
) -> Path:
    """Run ProPainter and return a seam-blended, exact-resolution video."""

    command = propainter_command(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        propainter_root=propainter_root,
        output_dir=output_dir,
        python_executable=python_executable,
        fps=fps,
        fp16=fp16,
        mask_dilation=mask_dilation,
        seam_feather=seam_feather,
    )
    subprocess.run(
        command,
        cwd=Path(propainter_root).resolve(),
        check=True,
        env=propainter_environment(),
    )
    return finalize_propainter_output(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        fps=fps,
        mask_dilation=mask_dilation,
        seam_feather=seam_feather,
    )


def propainter_command(
    frames_dir: str | Path,
    masks_dir: str | Path,
    propainter_root: str | Path,
    output_dir: str | Path,
    python_executable: str,
    fps: float,
    fp16: bool = True,
    mask_dilation: int = DEFAULT_MASK_DILATION,
    seam_feather: int = DEFAULT_SEAM_FEATHER,
) -> list[str]:
    """Build the exact upstream ProPainter invocation used by the pipeline."""

    frames_dir = Path(frames_dir).resolve()
    masks_dir = Path(masks_dir).resolve()
    propainter_root = Path(propainter_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script = propainter_root / "inference_propainter.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    # Validate once before starting the expensive model process.
    if int(mask_dilation) < 0:
        raise ValueError("mask_dilation must be non-negative")
    if not 0 <= int(seam_feather) <= int(mask_dilation):
        raise ValueError("seam_feather must be between 0 and mask_dilation")

    command = [
        python_executable,
        str(script),
        "--video",
        str(frames_dir),
        "--mask",
        str(masks_dir),
        "--output",
        str(output_dir),
        "--save_fps",
        str(max(1, round(fps))),
        "--mask_dilation",
        str(int(mask_dilation)),
        "--subvideo_length",
        "80",
        "--neighbor_length",
        "10",
        "--ref_stride",
        "10",
        "--raft_iter",
        "20",
        "--resize_ratio",
        "1.0",
        "--save_frames",
    ]
    if fp16:
        command.append("--fp16")
    return command


def finalize_propainter_output(
    frames_dir: str | Path,
    masks_dir: str | Path,
    output_dir: str | Path,
    fps: float,
    mask_dilation: int = DEFAULT_MASK_DILATION,
    seam_feather: int = DEFAULT_SEAM_FEATHER,
) -> Path:
    """Validate and seam-blend an already generated ProPainter result."""

    frames_dir = Path(frames_dir).resolve()
    masks_dir = Path(masks_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if int(mask_dilation) < 0:
        raise ValueError("mask_dilation must be non-negative")
    if not 0 <= int(seam_feather) <= int(mask_dilation):
        raise ValueError("seam_feather must be between 0 and mask_dilation")

    # With a frame directory input, ProPainter names the result after that dir.
    result_root = output_dir / frames_dir.name
    upstream_video = result_root / "inpaint_out.mp4"
    result_frames = sorted((result_root / "frames").glob("*.png"))
    input_frames = sorted(frames_dir.glob("*.jpg"))
    input_masks = sorted(masks_dir.glob("*.png"))
    if not upstream_video.is_file() or upstream_video.stat().st_size == 0:
        raise RuntimeError(f"ProPainter returned without producing {upstream_video}")
    if len(result_frames) != len(input_frames):
        raise RuntimeError(
            f"ProPainter frame mismatch: {len(result_frames)} vs {len(input_frames)}"
        )
    if len(input_masks) != len(input_frames):
        raise RuntimeError(
            f"ProPainter mask mismatch: {len(input_masks)} vs {len(input_frames)}"
        )

    # Preserve upstream's raw PNGs for debugging, but prepare a soft seam for
    # the exact video consumed by the compositor.  This prevents a hard change
    # at the outermost dilated pixel without reintroducing the original hand.
    alpha_dir = result_root / "seam_alpha"
    blended_dir = result_root / "blended_frames"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    blended_dir.mkdir(parents=True, exist_ok=True)
    blended_frames: list[Path] = []
    for frame_index, (source_path, mask_path, result_path) in enumerate(
        zip(input_frames, input_masks, result_frames)
    ):
        source_frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        result_frame = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
        if source_frame is None or mask is None or result_frame is None:
            raise RuntimeError(f"could not decode inpaint input {frame_index}")
        if (
            source_frame.shape != result_frame.shape
            or mask.shape != source_frame.shape[:2]
        ):
            raise RuntimeError(f"inpaint shape mismatch at frame {frame_index}")
        alpha = soft_inpaint_alpha(mask, mask_dilation, seam_feather)
        blended = blend_inpainted_frame(source_frame, result_frame, alpha)
        alpha_path = alpha_dir / f"{frame_index:05d}.png"
        blended_path = blended_dir / f"{frame_index:05d}.png"
        if not cv2.imwrite(
            str(alpha_path), np.round(alpha * 255.0).astype(np.uint8)
        ) or not cv2.imwrite(str(blended_path), blended):
            raise RuntimeError(f"could not write seam preparation {frame_index}")
        blended_frames.append(blended_path)

    # Upstream ImageIO silently pads dimensions to multiples of 16. Re-encode
    # our exact, seam-blended PNG results so compositing stays aligned.
    first = cv2.imread(str(blended_frames[0]))
    if first is None:
        raise RuntimeError(f"could not decode {result_frames[0]}")
    height, width = first.shape[:2]
    exact_video = result_root / "inpaint_out_exact.mp4"
    writer = cv2.VideoWriter(
        str(exact_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create {exact_video}")
    for frame_path in blended_frames:
        frame = cv2.imread(str(frame_path))
        if frame is None or frame.shape[:2] != (height, width):
            raise RuntimeError(f"invalid ProPainter frame: {frame_path}")
        writer.write(frame)
    writer.release()
    if not exact_video.is_file() or exact_video.stat().st_size == 0:
        raise RuntimeError(f"failed to encode {exact_video}")
    return exact_video
