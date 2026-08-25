from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .data import read_numbered_images
from .render import _decontaminate_foreground


def _open_video(path: Path) -> tuple[cv2.VideoCapture, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return capture, fps, (width, height)


def _unit_float(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 image, or retain an already-normalized float image."""

    array = np.asarray(image)
    result = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer) or (
        result.size and float(np.nanmax(result)) > 1.0
    ):
        result /= 255.0
    return np.clip(result, 0.0, 1.0)


def _fallback_premultiplied_matte(
    robot_rgb: np.ndarray, robot_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build a clean premultiplied matte for renders made by older versions."""

    straight = _decontaminate_foreground(
        _unit_float(robot_rgb), np.asarray(robot_mask, dtype=bool)
    )
    alpha = np.asarray(robot_mask, dtype=np.float32)
    return straight * alpha[..., None], alpha


def _filter_premultiplied(
    premultiplied: np.ndarray, alpha: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Feather a matte without mixing renderer-background RGB into its edge."""

    premultiplied = np.asarray(premultiplied, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if premultiplied.shape != (*alpha.shape, 3):
        raise ValueError("premultiplied RGB and alpha shapes differ")
    if sigma > 0:
        # Filtering color and coverage in premultiplied space is essential:
        # blurring alpha alone and multiplying raw SAPIEN RGB produces a gray
        # fringe wherever the antialiased render contains its clear color.
        premultiplied = cv2.GaussianBlur(premultiplied, (0, 0), float(sigma))
        alpha = cv2.GaussianBlur(alpha, (0, 0), float(sigma))
    alpha = np.clip(alpha, 0.0, 1.0)
    premultiplied = np.clip(premultiplied, 0.0, 1.0)
    premultiplied = np.minimum(premultiplied, alpha[..., None])
    return premultiplied, alpha


def _composite_premultiplied(
    background: np.ndarray, premultiplied: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Composite normalized premultiplied foreground over an 8-bit frame."""

    background_float = _unit_float(background)
    if background_float.shape != premultiplied.shape:
        raise ValueError("background and foreground sizes differ")
    result = premultiplied + background_float * (1.0 - alpha[..., None])
    return np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)


def _discover_render_mattes(
    robot_rgb_dir: Path, frame_count: int
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Load the renderer's soft alpha and premultiplied RGB when available."""

    parent = robot_rgb_dir.parent
    alpha_dir = parent / "robot_alpha"
    premultiplied_dir = parent / "robot_premultiplied"
    if not alpha_dir.is_dir() or not premultiplied_dir.is_dir():
        return None
    alphas = read_numbered_images(alpha_dir, grayscale=True)
    premultiplied = read_numbered_images(premultiplied_dir)
    if len(alphas) != frame_count or len(premultiplied) != frame_count:
        raise ValueError("soft-alpha/premultiplied render frame counts differ")
    return premultiplied, alphas


def _frame_visibility_selector(
    index: int,
    robot_union_mask: np.ndarray,
    human_visibility_by_side: dict[str, np.ndarray],
    side_robot_masks: dict[str, list[np.ndarray]],
    minimum_robot_pixels: int,
) -> tuple[np.ndarray, bool, dict[str, tuple[bool, bool]]]:
    """Select only robot sides backed by human and rendered visibility."""

    permitted = np.zeros_like(robot_union_mask, dtype=bool)
    use_inpainted_background = False
    decisions: dict[str, tuple[bool, bool]] = {}
    for side, visibility in human_visibility_by_side.items():
        human_visible = bool(visibility[index])
        rendered = side_robot_masks[side][index] > 127
        robot_visible = int(np.count_nonzero(rendered)) >= minimum_robot_pixels
        decisions[side] = (human_visible, robot_visible)
        if human_visible:
            use_inpainted_background = True
        if human_visible and robot_visible:
            permitted |= rendered
    permitted &= np.asarray(robot_union_mask, dtype=bool)
    return permitted, use_inpainted_background, decisions


def composite_videos(
    source_video: str | Path,
    inpainted_video: str | Path,
    robot_rgb_dir: str | Path,
    robot_mask_dir: str | Path,
    human_mask_dir: str | Path,
    output_dir: str | Path,
    human_mask_dilation: int = 18,
    feather_sigma: float = 1.5,
    human_visibility_by_side: dict[str, np.ndarray] | None = None,
    robot_mask_dirs_by_side: dict[str, str | Path] | None = None,
    robot_visibility_min_pixels: int = 16,
) -> None:
    """Write halo-free full/conservative composites and a full-mask QA video."""

    source_video = Path(source_video)
    inpainted_video = Path(inpainted_video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    robot_rgb_dir = Path(robot_rgb_dir)
    robot_rgbs = read_numbered_images(robot_rgb_dir)
    robot_masks = read_numbered_images(robot_mask_dir, grayscale=True)
    human_masks = read_numbered_images(human_mask_dir, grayscale=True)
    frame_count = len(robot_rgbs)
    if not (len(robot_masks) == len(human_masks) == frame_count):
        raise ValueError("robot RGB/mask and human-mask frame counts differ")

    side_robot_masks: dict[str, list[np.ndarray]] = {}
    if (human_visibility_by_side is None) != (robot_mask_dirs_by_side is None):
        raise ValueError(
            "human visibility and side robot-mask directories must be provided together"
        )
    if human_visibility_by_side is not None:
        if set(human_visibility_by_side) != set(robot_mask_dirs_by_side or {}):
            raise ValueError("human visibility and robot-mask sides differ")
        for side, visibility in human_visibility_by_side.items():
            values = np.asarray(visibility, dtype=bool)
            if values.shape != (frame_count,):
                raise ValueError(f"{side} visibility length differs from render")
            human_visibility_by_side[side] = values
            side_robot_masks[side] = read_numbered_images(
                (robot_mask_dirs_by_side or {})[side], grayscale=True
            )
            if len(side_robot_masks[side]) != frame_count:
                raise ValueError(f"{side} robot-mask frame count differs")
    if robot_visibility_min_pixels < 1:
        raise ValueError("robot_visibility_min_pixels must be positive")

    render_mattes = _discover_render_mattes(robot_rgb_dir, frame_count)
    source, source_fps, source_size = _open_video(source_video)
    inpainted, _, inpaint_size = _open_video(inpainted_video)
    height, width = robot_rgbs[0].shape[:2]
    if inpaint_size != (width, height):
        raise ValueError(
            f"inpaint size {inpaint_size} does not match render {(width, height)}"
        )

    writers = {}
    for name, size in {
        "composite_full.mp4": (width, height),
        "composite_conservative.mp4": (width, height),
        "qa_side_by_side.mp4": (width * 4, height),
    }.items():
        writer = cv2.VideoWriter(
            str(output_dir / name),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not create {name}")
        writers[name] = writer

    kernel_size = max(1, int(human_mask_dilation) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    processed = 0
    gate_summary: dict[str, object] = {
        "enabled": human_visibility_by_side is not None,
        "robot_visibility_min_pixels": robot_visibility_min_pixels,
        "sides": {},
    }
    if human_visibility_by_side is not None:
        gate_summary["sides"] = {
            side: {
                "human_visible_frames": int(np.count_nonzero(visibility)),
                "robot_visible_frames": 0,
                "composited_frames": 0,
                "human_without_robot_frames": [],
                "robot_without_human_frames": [],
            }
            for side, visibility in human_visibility_by_side.items()
        }
    for index in range(frame_count):
        ok_source, source_frame = source.read()
        ok_inpaint, background = inpainted.read()
        if not ok_source or not ok_inpaint:
            raise RuntimeError(f"video ended before expected frame {index}")
        if source_size != (width, height):
            source_frame = cv2.resize(source_frame, (width, height), cv2.INTER_AREA)

        robot_mask = robot_masks[index] > 127
        human_mask = human_masks[index] > 127
        if robot_mask.shape != (height, width) or human_mask.shape != (height, width):
            raise ValueError(f"mask size mismatch at frame {index}")
        if render_mattes is None:
            premultiplied_raw, alpha_raw = _fallback_premultiplied_matte(
                robot_rgbs[index], robot_mask
            )
        else:
            premultiplied_frames, alpha_frames = render_mattes
            premultiplied_raw = _unit_float(premultiplied_frames[index])
            alpha_raw = _unit_float(alpha_frames[index])
            # The binary union is a safety bound for incomplete/stale mattes.
            alpha_raw *= robot_mask.astype(np.float32)
            premultiplied_raw *= robot_mask[..., None]

        use_inpainted_background = True
        if human_visibility_by_side is not None:
            permitted_robot, use_inpainted_background, decisions = (
                _frame_visibility_selector(
                    index,
                    robot_mask,
                    human_visibility_by_side,
                    side_robot_masks,
                    robot_visibility_min_pixels,
                )
            )
            side_summary = gate_summary["sides"]
            assert isinstance(side_summary, dict)
            for side, (human_visible, robot_visible) in decisions.items():
                stats = side_summary[side]
                if robot_visible:
                    stats["robot_visible_frames"] += 1
                if human_visible and robot_visible:
                    stats["composited_frames"] += 1
                elif human_visible:
                    stats["human_without_robot_frames"].append(index)
                elif robot_visible:
                    stats["robot_without_human_frames"].append(index)
            alpha_raw *= permitted_robot.astype(np.float32)
            premultiplied_raw *= permitted_robot[..., None]

        composite_background = background if use_inpainted_background else source_frame

        expanded_human = cv2.dilate(human_mask.astype(np.uint8), kernel) > 0
        full_premultiplied, full_alpha = _filter_premultiplied(
            premultiplied_raw, alpha_raw, feather_sigma
        )
        conservative_selector = expanded_human.astype(np.float32)
        conservative_premultiplied, conservative_alpha = _filter_premultiplied(
            premultiplied_raw * conservative_selector[..., None],
            alpha_raw * conservative_selector,
            feather_sigma,
        )

        full = _composite_premultiplied(
            composite_background, full_premultiplied, full_alpha
        )
        conservative = _composite_premultiplied(
            composite_background, conservative_premultiplied, conservative_alpha
        )
        mask_preview = source_frame.copy()
        red = np.zeros_like(mask_preview)
        red[..., 2] = 255
        mask_alpha = (human_mask.astype(np.float32) * 0.55)[..., None]
        mask_preview = (
            mask_preview * (1.0 - mask_alpha) + red * mask_alpha
        ).astype(np.uint8)
        # Full robot coverage is the default while object/robot occlusion is
        # explicitly out of scope. The conservative result remains available
        # as a secondary diagnostic artifact.
        qa = np.concatenate([source_frame, mask_preview, background, full], axis=1)
        writers["composite_full.mp4"].write(full)
        writers["composite_conservative.mp4"].write(conservative)
        writers["qa_side_by_side.mp4"].write(qa)
        processed += 1

    source.release()
    inpainted.release()
    for writer in writers.values():
        writer.release()
    if processed != frame_count:
        raise RuntimeError(f"composited {processed} of {frame_count} frames")
    (output_dir / "visibility_gate.json").write_text(
        json.dumps(gate_summary, indent=2) + "\n"
    )
