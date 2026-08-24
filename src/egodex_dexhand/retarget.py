from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RetargetResult:
    qpos: np.ndarray
    joint_names: tuple[str, ...]
    urdf_path: Path
    target_human_indices: np.ndarray
    objective_values: np.ndarray
    joint_limits: np.ndarray


def retarget_position_sequence(
    joints_sapien: np.ndarray,
    assets_root: str | Path,
    robot: str = "allegro",
    hand: str = "right",
    preroll: int = 0,
) -> RetargetResult:
    """Run dex-retargeting's real offline POSITION optimizer.

    Six dummy joints are added at the root, so the returned trajectory includes
    wrist translation/orientation as well as finger joints.
    """

    from dex_retargeting.constants import (
        HandType,
        RetargetingType,
        RobotName,
        get_default_config_path,
    )
    from dex_retargeting.retargeting_config import RetargetingConfig

    try:
        robot_enum = RobotName[robot.lower()]
    except KeyError as exc:
        choices = ", ".join(item.name for item in RobotName)
        raise ValueError(f"unsupported robot {robot!r}; choose from {choices}") from exc
    try:
        hand_enum = HandType[hand.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported hand {hand!r}; choose left or right") from exc

    joints_sapien = np.asarray(joints_sapien, dtype=np.float32)
    if joints_sapien.ndim != 3 or joints_sapien.shape[1:] != (21, 3):
        raise ValueError(f"expected joints [T,21,3], got {joints_sapien.shape}")

    assets_root = Path(assets_root).resolve()
    RetargetingConfig.set_default_urdf_dir(assets_root)
    config_path = get_default_config_path(
        robot_enum, RetargetingType.position, hand_enum
    )
    config = RetargetingConfig.load_from_file(
        config_path, override={"add_dummy_free_joint": True, "low_pass_alpha": 1.0}
    )
    retargeting = config.build()

    # The optimizer will refine all six free-root coordinates. Identity is only
    # a stable initial guess; it is not used as the final wrist orientation.
    retargeting.warm_start(
        joints_sapien[0, 0],
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        hand_type=hand_enum,
        is_mano_convention=True,
    )
    target_indices = np.asarray(
        retargeting.optimizer.target_link_human_indices, dtype=np.int64
    )

    # The floating-root optimizer can otherwise expose its cold-start pose as
    # frame 0. Repeating the first observation converges that internal state
    # without changing any input data or requiring training.
    for _ in range(max(0, int(preroll))):
        retargeting.retarget(joints_sapien[0, target_indices])

    qpos_frames = []
    objectives = []
    joint_limits = np.asarray(
        retargeting.optimizer.robot.joint_limits, dtype=np.float32
    )
    for frame in joints_sapien:
        qpos = np.asarray(retargeting.retarget(frame[target_indices]), dtype=np.float32)
        # dex-retargeting intentionally permits epsilon around the URDF limits.
        # Clip before handing the trajectory to a simulator.
        qpos = np.clip(qpos, joint_limits[:, 0], joint_limits[:, 1])
        if not np.isfinite(qpos).all():
            raise RuntimeError("retargeting produced non-finite qpos")
        qpos_frames.append(qpos)
        objectives.append(float(retargeting.optimizer.opt.last_optimum_value()))

    urdf_path = Path(config.urdf_path)
    return RetargetResult(
        qpos=np.stack(qpos_frames),
        joint_names=tuple(retargeting.joint_names),
        urdf_path=urdf_path,
        target_human_indices=target_indices,
        objective_values=np.asarray(objectives, dtype=np.float32),
        joint_limits=joint_limits,
    )


def save_retarget_result(path: str | Path, result: RetargetResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=result.qpos,
        joint_names=np.asarray(result.joint_names),
        urdf_path=np.asarray(str(result.urdf_path)),
        target_human_indices=result.target_human_indices,
        objective_values=result.objective_values,
        joint_limits=result.joint_limits,
    )
