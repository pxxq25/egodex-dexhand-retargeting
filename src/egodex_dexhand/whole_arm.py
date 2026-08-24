from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


PANDA_HOME = np.asarray(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64
)
PANDA_DISTAL_SCALE = 1.0
PANDA_BASE_ROLL = 2.0
PANDA_BASE_LOCAL_OFFSET = np.asarray([-0.005, -0.047, -0.014], dtype=np.float64)


@dataclass(frozen=True)
class PandaArmResult:
    qpos: np.ndarray
    joint_names: tuple[str, ...]
    joint_limits: np.ndarray
    base_translation_world: np.ndarray
    base_rotation_world: np.ndarray
    world_from_camera: np.ndarray
    wrist_error: np.ndarray
    guide_error: np.ndarray
    urdf_path: Path
    shoulder_height: float


def prepare_panda_arm_urdf(
    panda_asset: str | Path,
    output_path: str | Path,
    distal_scale: float = PANDA_DISTAL_SCALE,
) -> Path:
    """Create an arm-only Panda URDF with resolved mesh paths.

    The Bullet Panda asset includes its parallel gripper.  We remove that
    gripper because the floating Allegro articulation is rendered separately.
    Absolute mesh paths also make the derived URDF load consistently in both
    Pinocchio and SAPIEN.  ``distal_scale`` is available for morphology studies;
    the validated Panda + Allegro sample keeps the original scale.
    """

    panda_asset = Path(panda_asset).resolve()
    source = panda_asset if panda_asset.is_file() else panda_asset / "panda.urdf"
    if not source.is_file():
        raise FileNotFoundError(f"Panda URDF not found: {source}")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source)
    root = tree.getroot()
    removed_links = {
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
        "panda_grasptarget",
    }
    for link in list(root.findall("link")):
        if link.attrib.get("name") in removed_links:
            root.remove(link)
    for joint in list(root.findall("joint")):
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = "" if parent is None else parent.attrib.get("link", "")
        child_name = "" if child is None else child.attrib.get("link", "")
        if parent_name in removed_links or child_name in removed_links:
            root.remove(joint)

    if not 0.25 <= distal_scale <= 1.0:
        raise ValueError(f"distal_scale must be in [0.25, 1.0], got {distal_scale}")

    def scale_xyz(element: ET.Element) -> None:
        raw = element.attrib.get("xyz")
        if not raw:
            return
        values = np.fromstring(raw, sep=" ", dtype=np.float64)
        if values.shape == (3,):
            element.attrib["xyz"] = " ".join(
                f"{value * distal_scale:.9g}" for value in values
            )

    # joint5 begins the elbow-to-flange chain.  Scale every following fixed or
    # actuated offset; Pinocchio and SAPIEN then agree on the shortened chain.
    for joint in root.findall("joint"):
        if joint.attrib.get("name") in {
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_joint8",
        }:
            origin = joint.find("origin")
            if origin is not None:
                scale_xyz(origin)

    distal_links = {"panda_link4", "panda_link5", "panda_link6", "panda_link7"}
    for link in root.findall("link"):
        if link.attrib.get("name") not in distal_links:
            continue
        for branch_name in ("visual", "collision"):
            for branch in link.findall(branch_name):
                origin = branch.find("origin")
                if origin is not None:
                    scale_xyz(origin)
                mesh = branch.find("geometry/mesh")
                if mesh is not None:
                    existing = np.fromstring(
                        mesh.attrib.get("scale", "1 1 1"), sep=" ", dtype=np.float64
                    )
                    if existing.shape != (3,):
                        existing = np.ones(3, dtype=np.float64)
                    mesh.attrib["scale"] = " ".join(
                        f"{value * distal_scale:.9g}" for value in existing
                    )

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        if filename.startswith("package://"):
            filename = filename[len("package://") :]
        candidate = Path(filename)
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        mesh.attrib["filename"] = str(candidate.resolve())

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _frame_position(model, data, frame_id: int, qpos: np.ndarray) -> np.ndarray:
    import pinocchio as pin

    pin.framesForwardKinematics(model, data, qpos)
    return np.asarray(data.oMf[frame_id].translation, dtype=np.float64).copy()


def _arm_positions(model, data, wrist_frame: int, guide_frame: int, qpos: np.ndarray):
    import pinocchio as pin

    pin.framesForwardKinematics(model, data, qpos)
    wrist = np.asarray(data.oMf[wrist_frame].translation, dtype=np.float64).copy()
    guide = np.asarray(data.oMf[guide_frame].translation, dtype=np.float64).copy()
    return wrist, guide


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("cannot normalize a zero-length vector")
    return np.asarray(vector, dtype=np.float64) / norm


def solve_panda_arm_sequence(
    joints_world: np.ndarray,
    world_from_camera: np.ndarray,
    arm_urdf: str | Path,
    shoulder_height: float = 0.333,
    wrist_weight: float = 40.0,
    guide_weight: float = 8.0,
    smooth_weight: float = 0.06,
    home_weight: float = 0.01,
) -> PandaArmResult:
    """Fit a smooth Panda trajectory to EgoDex's wrist track.

    EgoDex's ``Arm``, ``Forearm``, and ``Hand`` markers correspond to shoulder,
    elbow, and wrist.  The Panda's first actuated joint is 0.333 m above its URDF
    root, so the root is offset along local -Z to put joint 1 at the tracked
    shoulder.  The wrist is the strong constraint and link 4 guides the elbow.
    """

    import pinocchio as pin
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    joints_world = np.asarray(joints_world, dtype=np.float64)
    world_from_camera = np.asarray(world_from_camera, dtype=np.float64)
    if joints_world.ndim != 3 or joints_world.shape[1:] != (4, 3):
        raise ValueError(f"expected arm joints [T,4,3], got {joints_world.shape}")
    if world_from_camera.shape != (len(joints_world), 4, 4):
        raise ValueError(
            f"expected camera transforms [T,4,4], got {world_from_camera.shape}"
        )
    arm_urdf = Path(arm_urdf).resolve()
    model = pin.buildModelFromUrdf(str(arm_urdf))
    if model.nq != 7:
        raise RuntimeError(f"expected a 7-DoF Panda arm, got nq={model.nq}")
    data = model.createData()
    wrist_frame = model.getFrameId("panda_link8")
    guide_frame = model.getFrameId("panda_link4")
    if wrist_frame >= len(model.frames) or guide_frame >= len(model.frames):
        raise RuntimeError("Panda URDF is missing link4/link8 frames")

    lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)
    upper = np.asarray(model.upperPositionLimit, dtype=np.float64)
    home = np.clip(PANDA_HOME, lower + 1e-5, upper - 1e-5)
    wrist_home = _frame_position(model, data, wrist_frame, home)
    guide_home = _frame_position(model, data, guide_frame, home)

    # rightShoulder is a torso-side marker.  rightArm is the actual shoulder
    # joint for the kinematic chain, followed by rightForearm and rightHand.
    shoulder = np.median(joints_world[:, 1], axis=0)
    upper_arm_directions = joints_world[:, 2] - joints_world[:, 1]
    upper_arm_directions /= np.linalg.norm(
        upper_arm_directions, axis=1, keepdims=True
    )
    z_axis = _normalized(np.median(upper_arm_directions, axis=0))
    reference_up = np.asarray([0.0, 1.0, 0.0])
    if abs(float(np.dot(reference_up, z_axis))) > 0.98:
        reference_up = np.asarray([1.0, 0.0, 0.0])
    x_axis = _normalized(np.cross(reference_up, z_axis))
    y_axis = _normalized(np.cross(z_axis, x_axis))
    aligned = np.column_stack([x_axis, y_axis, z_axis])
    # This fixed roll and small local translation select a smooth, collision-free
    # redundant-IK branch for the sample while preserving upper-arm alignment.
    roll = Rotation.from_euler("z", PANDA_BASE_ROLL).as_matrix()
    base_rotation = aligned @ roll
    base_translation = shoulder - base_rotation @ np.asarray(
        [0.0, 0.0, float(shoulder_height)]
    ) + base_rotation @ PANDA_BASE_LOCAL_OFFSET

    wrist_targets = (base_rotation.T @ (joints_world[:, 3] - base_translation).T).T
    guide_targets = (base_rotation.T @ (joints_world[:, 2] - base_translation).T).T

    q_frames: list[np.ndarray] = []
    wrist_errors: list[float] = []
    guide_errors: list[float] = []
    previous = home.copy()
    for wrist_target, guide_target in zip(wrist_targets, guide_targets):
        def residual(qpos: np.ndarray) -> np.ndarray:
            wrist, guide = _arm_positions(
                model, data, wrist_frame, guide_frame, qpos
            )
            return np.concatenate(
                [
                    wrist_weight * (wrist - wrist_target),
                    guide_weight * (guide - guide_target),
                    smooth_weight * (qpos - previous),
                    home_weight * (qpos - home),
                ]
            )

        solution = least_squares(
            residual,
            previous,
            bounds=(lower + 1e-4, upper - 1e-4),
            max_nfev=250,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        qpos = np.clip(solution.x, lower, upper)
        if not solution.success or not np.isfinite(qpos).all():
            raise RuntimeError(f"Panda IK failed: {solution.message}")
        wrist, guide = _arm_positions(model, data, wrist_frame, guide_frame, qpos)
        q_frames.append(qpos)
        wrist_errors.append(float(np.linalg.norm(wrist - wrist_target)))
        guide_errors.append(float(np.linalg.norm(guide - guide_target)))
        previous = qpos

    # Joint-by-joint warm starts can occasionally switch redundant IK branches.
    # Refine all frames together with velocity and acceleration penalties.
    from scipy.sparse import lil_matrix

    initial = np.stack(q_frames)
    frame_residual_size = 13  # wrist 3 + guide 3 + home 7
    residual_count = (
        len(initial) * frame_residual_size
        + (len(initial) - 1) * 7
        + max(0, len(initial) - 2) * 7
    )
    sparsity = lil_matrix((residual_count, initial.size), dtype=np.int8)
    row = 0
    for frame_index in range(len(initial)):
        column = slice(frame_index * 7, (frame_index + 1) * 7)
        sparsity[row : row + frame_residual_size, column] = 1
        row += frame_residual_size
    for frame_index in range(len(initial) - 1):
        sparsity[row : row + 7, frame_index * 7 : (frame_index + 2) * 7] = 1
        row += 7
    for frame_index in range(len(initial) - 2):
        sparsity[row : row + 7, frame_index * 7 : (frame_index + 3) * 7] = 1
        row += 7

    def sequence_residual(flat_qpos: np.ndarray) -> np.ndarray:
        trajectory = flat_qpos.reshape((-1, 7))
        values: list[np.ndarray] = []
        for frame_index, qpos in enumerate(trajectory):
            wrist, guide = _arm_positions(
                model, data, wrist_frame, guide_frame, qpos
            )
            values.extend(
                [
                    40.0 * (wrist - wrist_targets[frame_index]),
                    4.0 * (guide - guide_targets[frame_index]),
                    0.005 * (qpos - home),
                ]
            )
        values.append((trajectory[1:] - trajectory[:-1]).reshape(-1))
        if len(trajectory) > 2:
            acceleration = trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]
            values.append(2.0 * acceleration.reshape(-1))
        return np.concatenate(values)

    smooth_solution = least_squares(
        sequence_residual,
        initial.reshape(-1),
        bounds=(np.tile(lower + 1e-4, len(initial)), np.tile(upper - 1e-4, len(initial))),
        jac_sparsity=sparsity.tocsr(),
        max_nfev=250,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    initial_residual = sequence_residual(initial.reshape(-1))
    initial_cost = 0.5 * float(np.dot(initial_residual, initial_residual))
    if (
        np.isfinite(smooth_solution.x).all()
        and float(smooth_solution.cost) <= initial_cost
    ):
        q_frames = list(smooth_solution.x.reshape((-1, 7)))
        wrist_errors = []
        guide_errors = []
        for frame_index, qpos in enumerate(q_frames):
            wrist, guide = _arm_positions(
                model, data, wrist_frame, guide_frame, qpos
            )
            wrist_errors.append(
                float(np.linalg.norm(wrist - wrist_targets[frame_index]))
            )
            guide_errors.append(
                float(np.linalg.norm(guide - guide_targets[frame_index]))
            )

    return PandaArmResult(
        qpos=np.stack(q_frames).astype(np.float32),
        joint_names=tuple(model.names[1:]),
        joint_limits=np.stack([lower, upper], axis=1).astype(np.float32),
        base_translation_world=base_translation.astype(np.float32),
        base_rotation_world=base_rotation.astype(np.float32),
        world_from_camera=world_from_camera.astype(np.float32),
        wrist_error=np.asarray(wrist_errors, dtype=np.float32),
        guide_error=np.asarray(guide_errors, dtype=np.float32),
        urdf_path=arm_urdf,
        shoulder_height=float(shoulder_height),
    )


def save_panda_arm_result(path: str | Path, result: PandaArmResult) -> None:
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
        wrist_error=result.wrist_error,
        guide_error=result.guide_error,
        urdf_path=np.asarray(str(result.urdf_path)),
        shoulder_height=np.asarray(result.shoulder_height, dtype=np.float32),
    )
