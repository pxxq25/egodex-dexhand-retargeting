from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .data import CV_TO_SAPIEN


UR5E_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class ShadowForearmTargets:
    position_world: np.ndarray
    rotation_world: np.ndarray


@dataclass(frozen=True)
class UR5eArmResult:
    qpos: np.ndarray
    joint_names: tuple[str, ...]
    joint_limits: np.ndarray
    base_translation_world: np.ndarray
    base_rotation_world: np.ndarray
    world_from_camera: np.ndarray
    position_error: np.ndarray
    orientation_error_degrees: np.ndarray
    target_adjusted_frames: np.ndarray
    urdf_path: Path


def prepare_ur5e_shadow_urdf(
    source_urdf: str | Path, output_path: str | Path
) -> Path:
    """Resolve mesh paths and repair the bundled thumb collision reference."""

    source_urdf = Path(source_urdf).resolve()
    output_path = Path(output_path).resolve()
    if not source_urdf.is_file():
        raise FileNotFoundError(source_urdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        raw = mesh.attrib.get("filename")
        if not raw:
            continue
        candidate = (source_urdf.parent / raw).resolve()
        if not candidate.is_file() and raw.endswith(
            "hands/shadow_hand/meshes/collision/th_distal_pst.obj"
        ):
            candidate = Path(
                str(candidate).replace(
                    "/meshes/collision/th_distal_pst.obj",
                    "/meshes/visual/th_distal_pst.obj",
                )
            )
        if not candidate.is_file():
            raise FileNotFoundError(f"URDF mesh is missing: {raw} -> {candidate}")
        mesh.attrib["filename"] = str(candidate)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def extract_shadow_forearm_targets(
    hand_qpos: np.ndarray,
    hand_joint_names: tuple[str, ...] | list[str],
    standalone_shadow_urdf: str | Path,
    world_from_camera: np.ndarray,
) -> ShadowForearmTargets:
    """Recover the floating Shadow forearm pose chosen by dex-retargeting."""

    import pinocchio as pin
    from dex_retargeting import yourdfpy as urdf

    hand_qpos = np.asarray(hand_qpos, dtype=np.float64)
    world_from_camera = np.asarray(world_from_camera, dtype=np.float64)
    if hand_qpos.ndim != 2 or hand_qpos.shape[1] != len(hand_joint_names):
        raise ValueError("hand qpos and joint names do not match")
    if world_from_camera.shape != (len(hand_qpos), 4, 4):
        raise ValueError("camera transforms do not match the hand trajectory")

    source = Path(standalone_shadow_urdf).resolve()
    if "glb" not in source.stem:
        candidate = source.with_stem(source.stem + "_glb")
        if candidate.is_file():
            source = candidate
    description = urdf.URDF.load(
        str(source), add_dummy_free_joints=True, build_scene_graph=False
    )
    with tempfile.TemporaryDirectory(prefix="egodex-shadow-target-") as temp_dir:
        augmented = Path(temp_dir) / source.name
        description.write_xml_file(str(augmented))
        model = pin.buildModelFromUrdf(str(augmented))
        model_names = tuple(str(name) for name in model.names[1:])
        name_to_input = {str(name): index for index, name in enumerate(hand_joint_names)}
        try:
            input_order = np.asarray(
                [name_to_input[name] for name in model_names], dtype=np.int64
            )
        except KeyError as exc:
            raise RuntimeError(f"floating Shadow target joint is missing: {exc}") from exc
        data = model.createData()
        frame_id = model.getFrameId("forearm")
        if frame_id >= len(model.frames):
            raise RuntimeError("standalone Shadow URDF has no forearm frame")

        positions = []
        rotations = []
        sapien_to_cv = CV_TO_SAPIEN.T.astype(np.float64)
        for frame_index, input_qpos in enumerate(hand_qpos):
            qpos = input_qpos[input_order]
            pin.framesForwardKinematics(model, data, qpos)
            pose = data.oMf[frame_id]
            position_cv = sapien_to_cv @ np.asarray(pose.translation)
            rotation_cv = sapien_to_cv @ np.asarray(pose.rotation)
            camera_pose = world_from_camera[frame_index]
            positions.append(
                camera_pose[:3, :3] @ position_cv + camera_pose[:3, 3]
            )
            rotations.append(camera_pose[:3, :3] @ rotation_cv)

    position_world = np.stack(positions).astype(np.float32)
    rotation_world = np.stack(rotations).astype(np.float32)
    if not np.isfinite(position_world).all() or not np.isfinite(rotation_world).all():
        raise RuntimeError("non-finite Shadow forearm targets")
    return ShadowForearmTargets(
        position_world=position_world,
        rotation_world=rotation_world,
    )


def solve_ur5e_arm_sequence(
    targets: ShadowForearmTargets,
    human_arm_joints_world: np.ndarray,
    world_from_camera: np.ndarray,
    combined_urdf: str | Path,
    q_reference: np.ndarray | None = None,
    initial_qpos: np.ndarray | None = None,
    base_translation_world: np.ndarray | None = None,
    base_rotation_world: np.ndarray | None = None,
) -> UR5eArmResult:
    """Solve UR5e IK so its mounted Shadow forearm matches retargeting exactly."""

    import pinocchio as pin
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    positions_world = np.asarray(targets.position_world, dtype=np.float64).copy()
    rotations_world = np.asarray(targets.rotation_world, dtype=np.float64).copy()
    human_arm_joints_world = np.asarray(human_arm_joints_world, dtype=np.float64)
    world_from_camera = np.asarray(world_from_camera, dtype=np.float64)
    frame_count = len(positions_world)
    if rotations_world.shape != (frame_count, 3, 3):
        raise ValueError("Shadow target rotations must have shape [T,3,3]")
    if human_arm_joints_world.shape != (frame_count, 4, 3):
        raise ValueError("human arm joints must have shape [T,4,3]")
    if world_from_camera.shape != (frame_count, 4, 4):
        raise ValueError("camera trajectory must have shape [T,4,4]")

    # The offline hand optimizer's unconstrained floating root has a 9.8 cm
    # first-frame initialization outlier in episode 1029, while EgoDex's tracked
    # wrist moves only 3.9 mm.  Replace only that boundary outlier; keep all
    # subsequent raw motion, including the genuine manipulation rotation.
    adjusted_frames: list[int] = []
    if frame_count >= 2:
        target_step = float(np.linalg.norm(positions_world[1] - positions_world[0]))
        tracked_step = float(
            np.linalg.norm(
                human_arm_joints_world[1, 3] - human_arm_joints_world[0, 3]
            )
        )
        if target_step > 0.05 and tracked_step < 0.02:
            positions_world[0] = positions_world[1]
            rotations_world[0] = rotations_world[1]
            adjusted_frames.append(0)

    combined_urdf = Path(combined_urdf).resolve()
    model = pin.buildModelFromUrdf(str(combined_urdf))
    data = model.createData()
    frame_id = model.getFrameId("forearm")
    if frame_id >= len(model.frames):
        raise RuntimeError("integrated UR5e + Shadow URDF has no forearm frame")
    joint_ids = [model.getJointId(name) for name in UR5E_JOINT_NAMES]
    if any(joint_id == 0 for joint_id in joint_ids):
        raise RuntimeError("integrated URDF is missing a UR5e joint")
    q_indices = np.asarray([model.joints[joint_id].idx_q for joint_id in joint_ids])
    lower = np.asarray(model.lowerPositionLimit[q_indices], dtype=np.float64)
    upper = np.asarray(model.upperPositionLimit[q_indices], dtype=np.float64)
    if q_reference is None:
        q_reference = np.asarray(
            [0.0, -1.35, 1.70, -1.92, -1.57, 0.0], dtype=np.float64
        )
    else:
        q_reference = np.asarray(q_reference, dtype=np.float64)
        if q_reference.shape != (6,) or not np.isfinite(q_reference).all():
            raise ValueError("UR5e reference posture must be a finite 6-vector")
    q_reference = np.clip(q_reference, lower + 1e-4, upper - 1e-4)
    if initial_qpos is not None:
        initial_qpos = np.asarray(initial_qpos, dtype=np.float64)
        if initial_qpos.shape != (6,) or not np.isfinite(initial_qpos).all():
            raise ValueError("initial UR5e qpos must be a finite 6-vector")
        initial_qpos = np.clip(initial_qpos, lower + 1e-4, upper - 1e-4)
    if (base_translation_world is None) != (base_rotation_world is None):
        raise ValueError("persisted UR5e base translation and rotation are a pair")
    neutral = np.asarray(pin.neutral(model), dtype=np.float64)

    def forearm_pose(arm_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        full_qpos = neutral.copy()
        full_qpos[q_indices] = arm_qpos
        pin.framesForwardKinematics(model, data, full_qpos)
        pose = data.oMf[frame_id]
        return (
            np.asarray(pose.translation, dtype=np.float64).copy(),
            np.asarray(pose.rotation, dtype=np.float64).copy(),
        )

    # Anchor the model at the middle of the clip.  This reference posture keeps
    # the UR5e base and proximal links offscreen while its distal links enter
    # through the same lower-right border as the captured human sleeve.
    if base_translation_world is None:
        anchor = (frame_count - 1) // 2
        position_base_reference, rotation_base_reference = forearm_pose(q_reference)
        base_rotation_world = rotations_world[anchor] @ rotation_base_reference.T
        base_translation_world = (
            positions_world[anchor] - base_rotation_world @ position_base_reference
        )
    else:
        # Keeping the same physical base is essential: changing it at every
        # chunk lets the redundant elbow/wrist chain visibly flip even when
        # the end-effector target is continuous.
        anchor = 0
        base_translation_world = np.asarray(
            base_translation_world, dtype=np.float64
        )
        base_rotation_world = np.asarray(base_rotation_world, dtype=np.float64)
        if base_translation_world.shape != (3,):
            raise ValueError("persisted UR5e base translation must be a 3-vector")
        if base_rotation_world.shape != (3, 3):
            raise ValueError("persisted UR5e base rotation must be 3x3")
        if not (
            np.isfinite(base_translation_world).all()
            and np.isfinite(base_rotation_world).all()
        ):
            raise ValueError("persisted UR5e base pose must be finite")

    positions_base = (
        base_rotation_world.T
        @ (positions_world - base_translation_world).T
    ).T
    rotations_base = np.einsum(
        "ij,tjk->tik", base_rotation_world.T, rotations_world
    )

    q_frames = np.empty((frame_count, 6), dtype=np.float64)
    q_frames[anchor] = q_reference

    def solve_frame(frame_index: int, previous: np.ndarray) -> np.ndarray:
        def residual(qpos: np.ndarray) -> np.ndarray:
            position, rotation = forearm_pose(qpos)
            orientation_error = Rotation.from_matrix(
                rotations_base[frame_index].T @ rotation
            ).as_rotvec()
            return np.concatenate(
                [
                    50.0 * (position - positions_base[frame_index]),
                    8.0 * orientation_error,
                    0.12 * (qpos - previous),
                    0.015 * (qpos - q_reference),
                ]
            )

        solution = least_squares(
            residual,
            previous,
            bounds=(lower + 1e-5, upper - 1e-5),
            max_nfev=300,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        if not np.isfinite(solution.x).all():
            raise RuntimeError(f"UR5e IK returned non-finite values at {frame_index}")
        return np.clip(solution.x, lower, upper)

    def wrapped_distance(candidate: np.ndarray, previous: np.ndarray) -> float:
        delta = np.arctan2(np.sin(candidate - previous), np.cos(candidate - previous))
        return float(np.linalg.norm(delta))

    if initial_qpos is not None:
        # Evaluate the persisted branch and the configured fallback, then keep
        # the feasible solution nearest to the prior state on the angle torus.
        candidates = [solve_frame(anchor, initial_qpos)]
        if wrapped_distance(q_reference, initial_qpos) > 1e-6:
            candidates.append(solve_frame(anchor, q_reference))
        q_frames[anchor] = min(
            candidates, key=lambda value: wrapped_distance(value, initial_qpos)
        )

    previous = q_frames[anchor]
    for frame_index in range(anchor + 1, frame_count):
        q_frames[frame_index] = solve_frame(frame_index, previous)
        previous = q_frames[frame_index]
    previous = q_frames[anchor]
    for frame_index in range(anchor - 1, -1, -1):
        q_frames[frame_index] = solve_frame(frame_index, previous)
        previous = q_frames[frame_index]

    position_errors = []
    orientation_errors = []
    for frame_index, qpos in enumerate(q_frames):
        position, rotation = forearm_pose(qpos)
        position_errors.append(
            float(np.linalg.norm(position - positions_base[frame_index]))
        )
        orientation_errors.append(
            float(
                np.linalg.norm(
                    Rotation.from_matrix(
                        rotations_base[frame_index].T @ rotation
                    ).as_rotvec()
                )
                * 180.0
                / np.pi
            )
        )

    return UR5eArmResult(
        qpos=q_frames.astype(np.float32),
        joint_names=UR5E_JOINT_NAMES,
        joint_limits=np.stack([lower, upper], axis=1).astype(np.float32),
        base_translation_world=base_translation_world.astype(np.float32),
        base_rotation_world=base_rotation_world.astype(np.float32),
        world_from_camera=world_from_camera.astype(np.float32),
        position_error=np.asarray(position_errors, dtype=np.float32),
        orientation_error_degrees=np.asarray(orientation_errors, dtype=np.float32),
        target_adjusted_frames=np.asarray(adjusted_frames, dtype=np.int64),
        urdf_path=combined_urdf,
    )


def save_ur5e_arm_result(path: str | Path, result: UR5eArmResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=result.qpos,
        joint_names=np.asarray(result.joint_names),
        joint_limits=result.joint_limits,
        base_translation_world=result.base_translation_world,
        base_rotation_world=result.base_rotation_world,
        world_from_camera=result.world_from_camera,
        position_error=result.position_error,
        orientation_error_degrees=result.orientation_error_degrees,
        target_adjusted_frames=result.target_adjusted_frames,
        urdf_path=np.asarray(str(result.urdf_path)),
    )
