from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np


# Shadow Hand frame names in the same anatomical order as EgoDex/MediaPipe.
SHADOW_LANDMARK_FRAMES = (
    "wrist",
    "thbase",
    "thproximal",
    "thmiddle",
    "thtip",
    "ffknuckle",
    "ffproximal",
    "ffmiddle",
    "fftip",
    "mfknuckle",
    "mfproximal",
    "mfmiddle",
    "mftip",
    "rfknuckle",
    "rfproximal",
    "rfmiddle",
    "rftip",
    "lfknuckle",
    "lfproximal",
    "lfmiddle",
    "lftip",
)

# The wrist is an exact anchor rather than a least-squares observation. MCPs
# stabilize palm orientation; tips retain the visible articulated pose.
DEFAULT_LANDMARK_WEIGHTS = np.asarray(
    [0, 2, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2],
    dtype=np.float64,
)


@dataclass(frozen=True)
class WristLockedSimilaritySequence:
    """Per-frame proper rotations, one clip scale, and an exact wrist anchor."""

    scale: float
    per_frame_optimal_scale: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    determinant: np.ndarray
    effective_landmarks: np.ndarray

    def transform(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] != len(self.rotation):
            raise ValueError("points must have shape [frames, landmarks, 3]")
        return (
            self.scale * np.einsum("tij,tkj->tki", self.rotation, values)
            + self.translation[:, None, :]
        )


@dataclass(frozen=True)
class ShadowScreenRegistration:
    """Visual Shadow Hand placement derived from all observed landmarks."""

    similarity: WristLockedSimilaritySequence
    robot_landmarks: np.ndarray
    fitted_landmarks_camera: np.ndarray
    forearm_position_camera: np.ndarray
    forearm_rotation_camera: np.ndarray
    wrist_in_forearm: np.ndarray


def fit_wrist_locked_similarity_sequence(
    robot_points: np.ndarray,
    observed_points: np.ndarray,
    weights: np.ndarray = DEFAULT_LANDMARK_WEIGHTS,
    *,
    confidence: np.ndarray | None = None,
    wrist_index: int = 0,
    shared_scale: float | None = None,
    minimum_confidence: float = 0.0,
) -> WristLockedSimilaritySequence:
    """Fit a reflection-free visual registration without trusting one point.

    Confidence is applied independently per frame. Invalid or low-confidence
    landmarks are ignored, but every frame must retain three non-collinear
    observations and a finite wrist. A single median scale prevents frame-wise
    breathing while translation keeps the wrist exact.
    """

    robot = np.asarray(robot_points, dtype=np.float64)
    observed = np.asarray(observed_points, dtype=np.float64)
    base_weights = np.asarray(weights, dtype=np.float64)
    if robot.ndim != 3 or robot.shape[-1] != 3:
        raise ValueError("robot points must have shape [frames, landmarks, 3]")
    if observed.shape != robot.shape:
        raise ValueError("observed and robot point shapes differ")
    frame_count, landmark_count, _ = robot.shape
    if base_weights.shape != (landmark_count,):
        raise ValueError("weights must have one value per landmark")
    if not 0 <= wrist_index < landmark_count:
        raise ValueError("wrist index is outside the landmark range")
    if not np.isfinite(base_weights).all() or np.any(base_weights < 0):
        raise ValueError("weights must be finite and non-negative")
    if minimum_confidence < 0:
        raise ValueError("minimum_confidence must be non-negative")

    if confidence is None:
        confidence_values = np.ones((frame_count, landmark_count), dtype=np.float64)
    else:
        confidence_values = np.asarray(confidence, dtype=np.float64)
        if confidence_values.shape != (frame_count, landmark_count):
            raise ValueError("confidence must have shape [frames, landmarks]")
        confidence_values = np.where(
            np.isfinite(confidence_values), np.maximum(confidence_values, 0.0), 0.0
        )

    rotations = np.empty((frame_count, 3, 3), dtype=np.float64)
    determinants = np.empty(frame_count, dtype=np.float64)
    optimal_scales = np.empty(frame_count, dtype=np.float64)
    effective_counts = np.empty(frame_count, dtype=np.int32)
    for frame_index in range(frame_count):
        finite = np.isfinite(robot[frame_index]).all(axis=1) & np.isfinite(
            observed[frame_index]
        ).all(axis=1)
        if not finite[wrist_index]:
            raise ValueError(f"wrist is invalid at frame {frame_index}")
        effective = (
            base_weights
            * confidence_values[frame_index]
            * finite.astype(np.float64)
        )
        effective[confidence_values[frame_index] < minimum_confidence] = 0.0
        positive = effective > 0
        effective_counts[frame_index] = int(np.count_nonzero(positive))
        if effective_counts[frame_index] < 3 or float(effective.sum()) <= 0:
            raise ValueError(
                f"fewer than three trusted landmarks at frame {frame_index}"
            )
        normalized = effective / float(effective.sum())
        robot_centered = robot[frame_index] - robot[frame_index, wrist_index]
        observed_centered = (
            observed[frame_index] - observed[frame_index, wrist_index]
        )
        covariance = (normalized[:, None] * robot_centered).T @ observed_centered
        left, singular_values, right = np.linalg.svd(covariance)
        rotation = right.T @ left.T
        if np.linalg.det(rotation) < 0:
            right[-1] *= -1.0
            rotation = right.T @ left.T
        variance = float(
            np.sum(normalized * np.sum(robot_centered * robot_centered, axis=1))
        )
        if variance < 1e-12:
            raise ValueError(f"robot landmarks are degenerate at frame {frame_index}")
        rotations[frame_index] = rotation
        determinants[frame_index] = np.linalg.det(rotation)
        optimal_scales[frame_index] = float(singular_values.sum() / variance)

    scale = (
        float(np.median(optimal_scales))
        if shared_scale is None
        else float(shared_scale)
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("shared scale must be finite and positive")
    rotated_wrist = np.einsum("tij,tj->ti", rotations, robot[:, wrist_index])
    translations = observed[:, wrist_index] - scale * rotated_wrist
    return WristLockedSimilaritySequence(
        scale=scale,
        per_frame_optimal_scale=optimal_scales.astype(np.float32),
        rotation=rotations.astype(np.float32),
        translation=translations.astype(np.float32),
        determinant=determinants.astype(np.float32),
        effective_landmarks=effective_counts,
    )


def fit_shadow_screen_registration(
    hand_qpos: np.ndarray,
    hand_joint_names: tuple[str, ...] | list[str],
    standalone_shadow_urdf: str | Path,
    observed_landmarks_camera: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    weights: np.ndarray = DEFAULT_LANDMARK_WEIGHTS,
) -> ShadowScreenRegistration:
    """Place a Shadow Hand from its 21 rendered landmarks, not its wrist alone."""

    import pinocchio as pin
    from dex_retargeting import yourdfpy as urdf

    qpos_values = np.asarray(hand_qpos, dtype=np.float64)
    observed = np.asarray(observed_landmarks_camera, dtype=np.float64)
    if qpos_values.ndim != 2 or qpos_values.shape[1] != len(hand_joint_names):
        raise ValueError("hand qpos and joint names do not match")
    if observed.shape != (len(qpos_values), len(SHADOW_LANDMARK_FRAMES), 3):
        raise ValueError("observed landmarks must have shape [frames, 21, 3]")

    source = Path(standalone_shadow_urdf).resolve()
    if "glb" not in source.stem:
        candidate = source.with_stem(source.stem + "_glb")
        if candidate.is_file():
            source = candidate
    description = urdf.URDF.load(
        str(source), add_dummy_free_joints=True, build_scene_graph=False
    )
    with tempfile.TemporaryDirectory(prefix="egodex-screen-registration-") as temp:
        augmented = Path(temp) / source.name
        description.write_xml_file(str(augmented))
        model = pin.buildModelFromUrdf(str(augmented))
        model_names = tuple(str(name) for name in model.names[1:])
        input_lookup = {
            str(name): index for index, name in enumerate(hand_joint_names)
        }
        try:
            input_order = np.asarray(
                [input_lookup[name] for name in model_names], dtype=np.int64
            )
        except KeyError as exc:
            raise RuntimeError(f"Shadow registration joint is missing: {exc}") from exc
        frame_ids = [model.getFrameId(name) for name in SHADOW_LANDMARK_FRAMES]
        if any(frame_id >= len(model.frames) for frame_id in frame_ids):
            raise RuntimeError("Shadow URDF is missing a landmark frame")
        forearm_frame = model.getFrameId("forearm")
        wrist_frame = model.getFrameId("wrist")
        if forearm_frame >= len(model.frames) or wrist_frame >= len(model.frames):
            raise RuntimeError("Shadow URDF is missing forearm or wrist frame")

        data = model.createData()
        robot_landmarks = []
        forearm_positions = []
        forearm_rotations = []
        wrist_in_forearm = []
        for input_qpos in qpos_values:
            pin.framesForwardKinematics(model, data, input_qpos[input_order])
            robot_landmarks.append(
                np.stack(
                    [
                        np.asarray(data.oMf[frame_id].translation).copy()
                        for frame_id in frame_ids
                    ]
                )
            )
            forearm_pose = data.oMf[forearm_frame]
            wrist_pose = data.oMf[wrist_frame]
            forearm_positions.append(np.asarray(forearm_pose.translation).copy())
            forearm_rotations.append(np.asarray(forearm_pose.rotation).copy())
            wrist_in_forearm.append(
                np.asarray(forearm_pose.inverse().act(wrist_pose).translation).copy()
            )

    robot = np.asarray(robot_landmarks, dtype=np.float64)
    similarity = fit_wrist_locked_similarity_sequence(
        robot,
        observed,
        weights,
        confidence=confidence,
    )
    fitted = similarity.transform(robot)
    forearm_position = (
        similarity.scale
        * np.einsum(
            "tij,tj->ti",
            similarity.rotation,
            np.asarray(forearm_positions, dtype=np.float64),
        )
        + similarity.translation
    )
    forearm_rotation = np.einsum(
        "tij,tjk->tik",
        similarity.rotation,
        np.asarray(forearm_rotations, dtype=np.float64),
    )
    return ShadowScreenRegistration(
        similarity=similarity,
        robot_landmarks=robot.astype(np.float32),
        fitted_landmarks_camera=fitted.astype(np.float32),
        forearm_position_camera=forearm_position.astype(np.float32),
        forearm_rotation_camera=forearm_rotation.astype(np.float32),
        wrist_in_forearm=np.median(
            np.asarray(wrist_in_forearm, dtype=np.float64), axis=0
        ).astype(np.float32),
    )
