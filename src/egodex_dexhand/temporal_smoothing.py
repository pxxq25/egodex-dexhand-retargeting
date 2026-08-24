from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import numpy as np


SHADOW_ROOT_JOINT_NAMES = (
    "dummy_x_translation_joint",
    "dummy_y_translation_joint",
    "dummy_z_translation_joint",
    "dummy_x_rotation_joint",
    "dummy_y_rotation_joint",
    "dummy_z_rotation_joint",
)
SHADOW_ACTUATED_DOF = 24


@dataclass(frozen=True)
class TemporalSmoothingConfig:
    """Offline, centered smoothing and physical derivative limits.

    The centered filter is zero-phase: it uses past and future samples equally.
    The derivative projection also updates both endpoints of every constraint,
    rather than imposing a causal forward clamp.
    """

    enabled: bool = True
    window_size: int = 7
    filter_passes: int = 1
    outlier_sigma: float = 3.5
    hand_max_velocity: float = 6.0
    hand_max_acceleration: float = 60.0
    forearm_max_linear_velocity: float = 1.5
    forearm_max_linear_acceleration: float = 12.0
    forearm_max_angular_velocity: float = 8.0
    constraint_iterations: int = 96

    def __post_init__(self) -> None:
        if self.window_size < 3 or self.window_size % 2 != 1:
            raise ValueError("smoothing window must be an odd integer >= 3")
        if self.filter_passes < 1:
            raise ValueError("smoothing filter passes must be >= 1")
        if self.outlier_sigma <= 0:
            raise ValueError("smoothing outlier sigma must be positive")
        derivative_limits = (
            self.hand_max_velocity,
            self.hand_max_acceleration,
            self.forearm_max_linear_velocity,
            self.forearm_max_linear_acceleration,
            self.forearm_max_angular_velocity,
        )
        if any(value <= 0 for value in derivative_limits):
            raise ValueError("smoothing derivative limits must be positive")
        if self.constraint_iterations < 1:
            raise ValueError("smoothing constraint iterations must be >= 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def add_temporal_smoothing_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("offline temporal smoothing")
    group.add_argument(
        "--temporal-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="smooth Shadow joints and forearm SE(3) targets (default: enabled)",
    )
    group.add_argument("--smoothing-window", type=int, default=7)
    group.add_argument("--smoothing-passes", type=int, default=1)
    group.add_argument("--smoothing-outlier-sigma", type=float, default=3.5)
    group.add_argument("--hand-max-velocity", type=float, default=6.0)
    group.add_argument("--hand-max-acceleration", type=float, default=60.0)
    group.add_argument("--forearm-max-linear-velocity", type=float, default=1.5)
    group.add_argument("--forearm-max-linear-acceleration", type=float, default=12.0)
    group.add_argument("--forearm-max-angular-velocity", type=float, default=8.0)


def temporal_smoothing_config_from_args(
    args: argparse.Namespace,
) -> TemporalSmoothingConfig:
    return TemporalSmoothingConfig(
        enabled=bool(args.temporal_smoothing),
        window_size=int(args.smoothing_window),
        filter_passes=int(args.smoothing_passes),
        outlier_sigma=float(args.smoothing_outlier_sigma),
        hand_max_velocity=float(args.hand_max_velocity),
        hand_max_acceleration=float(args.hand_max_acceleration),
        forearm_max_linear_velocity=float(args.forearm_max_linear_velocity),
        forearm_max_linear_acceleration=float(
            args.forearm_max_linear_acceleration
        ),
        forearm_max_angular_velocity=float(args.forearm_max_angular_velocity),
    )


def shadow_actuated_joint_indices(
    joint_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    names = tuple(str(name) for name in joint_names)
    if len(set(names)) != len(names):
        raise ValueError("Shadow trajectory contains duplicate joint names")
    missing_root = sorted(set(SHADOW_ROOT_JOINT_NAMES) - set(names))
    if missing_root:
        raise ValueError(f"Shadow floating root joints are missing: {missing_root}")
    indices = np.asarray(
        [index for index, name in enumerate(names) if name not in SHADOW_ROOT_JOINT_NAMES],
        dtype=np.int64,
    )
    if len(indices) != SHADOW_ACTUATED_DOF:
        raise ValueError(
            f"expected {SHADOW_ACTUATED_DOF} actuated Shadow joints, got {len(indices)}"
        )
    return indices


def smooth_shadow_qpos(
    qpos: np.ndarray,
    joint_names: tuple[str, ...] | list[str],
    joint_limits: np.ndarray,
    fps: float,
    config: TemporalSmoothingConfig,
) -> np.ndarray:
    """Smooth the 24 physical Shadow joints while preserving its dummy root.

    The floating root is represented by three translations and three sequential
    Euler joints. Independent scalar filtering of those rotation joints can
    create an invalid pose, so they are retained here. The pose they induce is
    smoothed geometrically by :func:`smooth_se3_trajectory` before arm IK.
    """

    values = _finite_trajectory(qpos, "Shadow qpos")
    limits = np.asarray(joint_limits, dtype=np.float64)
    if values.shape[1] != len(joint_names):
        raise ValueError("Shadow qpos and joint names do not match")
    if limits.shape != (values.shape[1], 2):
        raise ValueError("Shadow joint limits must have shape [Q,2]")
    if not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("Shadow joint limits are invalid")
    _validate_fps(fps)
    actuated = shadow_actuated_joint_indices(joint_names)
    result = values.copy()
    if not config.enabled or len(values) < 2:
        return result.astype(np.asarray(qpos).dtype, copy=False)

    hand = np.clip(
        values[:, actuated], limits[actuated, 0], limits[actuated, 1]
    )
    hand = _hampel_filter(
        hand,
        window_size=config.window_size,
        sigma=config.outlier_sigma,
        minimum_deviation=0.08,
    )
    for _ in range(config.filter_passes):
        hand = _centered_filter(hand, config.window_size)
    hand = _project_scalar_derivative_limits(
        hand,
        fps=fps,
        max_velocity=config.hand_max_velocity,
        max_acceleration=config.hand_max_acceleration,
        lower=limits[actuated, 0],
        upper=limits[actuated, 1],
        iterations=config.constraint_iterations,
    )
    result[:, actuated] = hand
    return result.astype(np.asarray(qpos).dtype, copy=False)


def smooth_se3_trajectory(
    positions: np.ndarray,
    rotations: np.ndarray,
    fps: float,
    config: TemporalSmoothingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly smooth an offline SE(3) trajectory without temporal lag."""

    from scipy.spatial.transform import Rotation

    xyz = _finite_trajectory(positions, "SE(3) positions")
    matrices = np.asarray(rotations, dtype=np.float64)
    if xyz.shape[1] != 3:
        raise ValueError("SE(3) positions must have shape [T,3]")
    if matrices.shape != (len(xyz), 3, 3):
        raise ValueError("SE(3) rotations must have shape [T,3,3]")
    if not np.isfinite(matrices).all():
        raise ValueError("SE(3) rotations contain non-finite values")
    orthogonality = np.matmul(matrices.transpose(0, 2, 1), matrices)
    if np.max(np.abs(orthogonality - np.eye(3))) > 1e-3:
        raise ValueError("SE(3) rotation matrices are not orthonormal")
    if np.any(np.linalg.det(matrices) < 0.0):
        raise ValueError("SE(3) rotation matrices must be proper rotations")
    _validate_fps(fps)
    if not config.enabled or len(xyz) < 2:
        return (
            xyz.astype(np.asarray(positions).dtype, copy=False),
            matrices.astype(np.asarray(rotations).dtype, copy=False),
        )

    xyz = _hampel_filter(
        xyz,
        window_size=config.window_size,
        sigma=config.outlier_sigma,
        minimum_deviation=0.01,
    )
    for _ in range(config.filter_passes):
        xyz = _centered_filter(xyz, config.window_size)
    xyz = _project_scalar_derivative_limits(
        xyz,
        fps=fps,
        max_velocity=config.forearm_max_linear_velocity,
        max_acceleration=config.forearm_max_linear_acceleration,
        iterations=config.constraint_iterations,
    )

    quaternions = _continuous_quaternions(Rotation.from_matrix(matrices).as_quat())
    quaternions = _reject_quaternion_outliers(
        quaternions,
        window_size=config.window_size,
        sigma=config.outlier_sigma,
        minimum_angle=np.deg2rad(20.0),
    )
    for _ in range(config.filter_passes):
        quaternions = _centered_quaternion_filter(
            quaternions, config.window_size
        )
    quaternions = _project_angular_velocity(
        quaternions,
        max_step=config.forearm_max_angular_velocity / float(fps),
        iterations=config.constraint_iterations,
    )
    smoothed_rotations = Rotation.from_quat(quaternions).as_matrix()
    return (
        xyz.astype(np.asarray(positions).dtype, copy=False),
        smoothed_rotations.astype(np.asarray(rotations).dtype, copy=False),
    )


def _finite_trajectory(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{label} must have shape [T,D] with T > 0")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def _validate_fps(fps: float) -> None:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")


def _window_bounds(index: int, count: int, half: int) -> tuple[int, int, slice]:
    start = max(0, index - half)
    stop = min(count, index + half + 1)
    weight_start = half - (index - start)
    return start, stop, slice(weight_start, weight_start + stop - start)


def _kernel(window_size: int) -> np.ndarray:
    half = window_size // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    sigma = max(1.0, window_size / 4.0)
    weights = np.exp(-0.5 * np.square(offsets / sigma))
    return weights / weights.sum()


def _centered_filter(values: np.ndarray, window_size: int) -> np.ndarray:
    count = len(values)
    if count < 2:
        return values.copy()
    half = window_size // 2
    weights = _kernel(window_size)
    output = np.empty_like(values)
    for index in range(count):
        start, stop, weight_slice = _window_bounds(index, count, half)
        local_weights = weights[weight_slice].copy()
        local_weights /= local_weights.sum()
        output[index] = np.sum(
            values[start:stop] * local_weights[:, None], axis=0
        )
    return output


def _hampel_filter(
    values: np.ndarray,
    window_size: int,
    sigma: float,
    minimum_deviation: float,
) -> np.ndarray:
    count = len(values)
    if count < 3:
        return values.copy()
    half = window_size // 2
    output = values.copy()
    for index in range(count):
        start = max(0, index - half)
        stop = min(count, index + half + 1)
        local = values[start:stop]
        median = np.median(local, axis=0)
        mad = np.median(np.abs(local - median), axis=0)
        threshold = np.maximum(minimum_deviation, sigma * 1.4826 * mad)
        replace = np.abs(values[index] - median) > threshold
        output[index, replace] = median[replace]
    return output


def _project_scalar_derivative_limits(
    values: np.ndarray,
    fps: float,
    max_velocity: float,
    max_acceleration: float,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    iterations: int = 96,
) -> np.ndarray:
    """Project component trajectories onto symmetric derivative constraints."""

    output = values.copy()
    count, dimensions = output.shape
    if lower is None:
        lower_bound = np.full(dimensions, -np.inf, dtype=np.float64)
    else:
        lower_bound = np.broadcast_to(
            np.asarray(lower, dtype=np.float64), (dimensions,)
        )
    if upper is None:
        upper_bound = np.full(dimensions, np.inf, dtype=np.float64)
    else:
        upper_bound = np.broadcast_to(
            np.asarray(upper, dtype=np.float64), (dimensions,)
        )
    velocity_step = float(max_velocity) / float(fps)
    acceleration_step = float(max_acceleration) / float(fps) ** 2
    tolerance = 1e-10

    for _ in range(iterations):
        output = np.clip(output, lower_bound, upper_bound)
        for order in (range(1, count), range(count - 1, 0, -1)):
            for index in order:
                delta = output[index] - output[index - 1]
                excess = delta - np.clip(delta, -velocity_step, velocity_step)
                output[index - 1] += 0.5 * excess
                output[index] -= 0.5 * excess
        if count >= 3:
            for order in (range(1, count - 1), range(count - 2, 0, -1)):
                for index in order:
                    second_difference = (
                        output[index - 1]
                        - 2.0 * output[index]
                        + output[index + 1]
                    )
                    excess = second_difference - np.clip(
                        second_difference,
                        -acceleration_step,
                        acceleration_step,
                    )
                    output[index - 1] -= excess / 6.0
                    output[index] += excess / 3.0
                    output[index + 1] -= excess / 6.0
        output = np.clip(output, lower_bound, upper_bound)
        velocity_violation = (
            np.max(np.abs(np.diff(output, axis=0))) - velocity_step
            if count >= 2
            else 0.0
        )
        acceleration_violation = (
            np.max(np.abs(np.diff(output, n=2, axis=0))) - acceleration_step
            if count >= 3
            else 0.0
        )
        if max(velocity_violation, acceleration_violation) <= tolerance:
            break
    return output


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("zero quaternion in orientation trajectory")
    return quaternions / norms


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    output = _normalize_quaternions(np.asarray(quaternions, dtype=np.float64).copy())
    for index in range(1, len(output)):
        if np.dot(output[index - 1], output[index]) < 0.0:
            output[index] *= -1.0
    return output


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def _reject_quaternion_outliers(
    quaternions: np.ndarray,
    window_size: int,
    sigma: float,
    minimum_angle: float,
) -> np.ndarray:
    count = len(quaternions)
    if count < 3:
        return quaternions.copy()
    half = window_size // 2
    output = quaternions.copy()
    for index in range(count):
        start = max(0, index - half)
        stop = min(count, index + half + 1)
        local = quaternions[start:stop]
        pairwise = np.empty((len(local), len(local)), dtype=np.float64)
        for row in range(len(local)):
            for column in range(len(local)):
                pairwise[row, column] = _quaternion_angle(
                    local[row], local[column]
                )
        medoid = local[int(np.argmin(pairwise.sum(axis=1)))]
        distances = np.asarray(
            [_quaternion_angle(candidate, medoid) for candidate in local]
        )
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        threshold = max(minimum_angle, median + sigma * 1.4826 * mad)
        if _quaternion_angle(quaternions[index], medoid) > threshold:
            output[index] = medoid
    return _continuous_quaternions(output)


def _centered_quaternion_filter(
    quaternions: np.ndarray, window_size: int
) -> np.ndarray:
    count = len(quaternions)
    half = window_size // 2
    weights = _kernel(window_size)
    output = np.empty_like(quaternions)
    for index in range(count):
        start, stop, weight_slice = _window_bounds(index, count, half)
        local = quaternions[start:stop].copy()
        local[local @ quaternions[index] < 0.0] *= -1.0
        local_weights = weights[weight_slice].copy()
        local_weights /= local_weights.sum()
        average = np.sum(local * local_weights[:, None], axis=0)
        output[index] = average / np.linalg.norm(average)
    return _continuous_quaternions(output)


def _slerp_pair(first: np.ndarray, second: np.ndarray, amount: float) -> np.ndarray:
    second_aligned = second.copy()
    dot = float(np.dot(first, second_aligned))
    if dot < 0.0:
        second_aligned *= -1.0
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = (1.0 - amount) * first + amount * second_aligned
        return result / np.linalg.norm(result)
    theta = float(np.arccos(dot))
    denominator = float(np.sin(theta))
    result = (
        np.sin((1.0 - amount) * theta) / denominator * first
        + np.sin(amount * theta) / denominator * second_aligned
    )
    return result / np.linalg.norm(result)


def _project_angular_velocity(
    quaternions: np.ndarray, max_step: float, iterations: int
) -> np.ndarray:
    output = quaternions.copy()
    count = len(output)
    for _ in range(iterations):
        maximum_excess = 0.0
        for order in (range(count - 1), range(count - 2, -1, -1)):
            for index in order:
                angle = _quaternion_angle(output[index], output[index + 1])
                excess = angle - max_step
                maximum_excess = max(maximum_excess, excess)
                if excess <= 1e-10:
                    continue
                amount = excess / (2.0 * angle)
                first = output[index].copy()
                second = output[index + 1].copy()
                output[index] = _slerp_pair(first, second, amount)
                output[index + 1] = _slerp_pair(first, second, 1.0 - amount)
        output = _continuous_quaternions(output)
        if maximum_excess <= 1e-10:
            break
    return output
