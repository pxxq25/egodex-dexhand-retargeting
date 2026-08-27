from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ChunkContinuityState:
    hand_qpos: np.ndarray
    arm_qpos: np.ndarray
    base_translation_world: np.ndarray
    base_rotation_world: np.ndarray
    source_frame: int = -1


def load_chunk_continuity_state(path: str | Path | None) -> ChunkContinuityState | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    with np.load(source) as values:
        return ChunkContinuityState(
            hand_qpos=np.asarray(values["hand_qpos"], dtype=np.float32),
            arm_qpos=np.asarray(values["arm_qpos"], dtype=np.float32),
            base_translation_world=np.asarray(
                values["base_translation_world"], dtype=np.float32
            ),
            base_rotation_world=np.asarray(
                values["base_rotation_world"], dtype=np.float32
            ),
            source_frame=int(values.get("source_frame", np.asarray(-1)).item()),
        )


def save_chunk_continuity_state(
    path: str | Path,
    *,
    hand_qpos: np.ndarray,
    arm_qpos: np.ndarray,
    base_translation_world: np.ndarray,
    base_rotation_world: np.ndarray,
    source_frame: int = -1,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        hand_qpos=np.asarray(hand_qpos, dtype=np.float32),
        arm_qpos=np.asarray(arm_qpos, dtype=np.float32),
        base_translation_world=np.asarray(base_translation_world, dtype=np.float32),
        base_rotation_world=np.asarray(base_rotation_world, dtype=np.float32),
        source_frame=np.asarray(source_frame, dtype=np.int64),
    )
    temporary.replace(destination)


def wrapped_joint_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("joint states must have the same shape")
    delta = np.arctan2(np.sin(first - second), np.cos(first - second))
    return float(np.linalg.norm(delta))


def last_state(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and len(array):
        return array[-1]
    raise ValueError("continuity state must be a vector or non-empty trajectory")


def trailing_context(previous: np.ndarray | None, current: np.ndarray, frames: int) -> np.ndarray:
    current = np.asarray(current)
    if frames < 1:
        return current[-1:].copy()
    if previous is None:
        combined = current
    else:
        old = np.asarray(previous)
        if old.ndim == 1:
            old = old[None]
        combined = np.concatenate([old, current], axis=0)
    return combined[-frames:].copy()


def smooth_wrapped_joint_boundary(
    previous: np.ndarray | None,
    current: np.ndarray,
    *,
    margin_frames: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Centered moving average across a chunk boundary on the angle torus."""

    values = np.asarray(current, dtype=np.float64)
    if previous is None or margin_frames < 1 or len(values) < 2:
        return values.astype(np.asarray(current).dtype, copy=False)
    prefix = np.asarray(previous, dtype=np.float64)
    if prefix.ndim == 1:
        prefix = prefix[None]
    prefix = prefix[-margin_frames:]
    joined = np.concatenate([prefix, values], axis=0)
    unwrapped = np.unwrap(joined, axis=0)
    radius = min(margin_frames, max(1, (len(joined) - 1) // 2))
    window = 2 * radius + 1
    padded = np.pad(unwrapped, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    filtered = np.stack(
        [np.convolve(padded[:, joint], kernel, mode="valid") for joint in range(values.shape[1])],
        axis=1,
    )
    result = filtered[len(prefix):]
    # Choose an equivalent 2pi representation nearest to the raw current
    # values, then enforce the robot's actual limits.
    result += 2.0 * np.pi * np.round((values - result) / (2.0 * np.pi))
    result = np.clip(result, np.asarray(lower), np.asarray(upper))
    return result.astype(np.asarray(current).dtype, copy=False)
