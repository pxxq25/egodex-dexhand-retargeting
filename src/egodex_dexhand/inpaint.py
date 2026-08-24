from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MASK_DILATION = 10
DEFAULT_SEAM_FEATHER = 4


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
    if not binary.any():
        raise ValueError("mask has no foreground")
    mask_dilation = int(mask_dilation)
    seam_feather = int(seam_feather)
    if mask_dilation < 0:
        raise ValueError("mask_dilation must be non-negative")
    if not 0 <= seam_feather <= mask_dilation:
        raise ValueError("seam_feather must be between 0 and mask_dilation")

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
    subprocess.run(command, cwd=propainter_root, check=True)

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
