from __future__ import annotations

from dataclasses import dataclass
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .data import CV_TO_SAPIEN


DEFAULT_TEMPORAL_SAMPLES = 5
DEFAULT_SHUTTER_FRACTION = 0.5
DEFAULT_DECONTAMINATE_RADIUS = 2
HIDEABLE_ARM_VISUAL_LINKS = frozenset(
    {
        "base_link_inertia",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "forearm",
    }
)


def _create_sapien_scene(render_device: str):
    """Create a render scene only on an explicitly selected Vulkan device."""

    if not render_device or not render_device.strip():
        raise ValueError("an explicit SAPIEN Vulkan render_device is required")
    import sapien

    return sapien.Scene(
        [
            sapien.physx.PhysxCpuSystem(),
            sapien.render.RenderSystem(render_device),
        ]
    )


def _temporal_sample_positions(
    frame_index: int,
    frame_count: int,
    temporal_samples: int = DEFAULT_TEMPORAL_SAMPLES,
    shutter_fraction: float = DEFAULT_SHUTTER_FRACTION,
) -> np.ndarray:
    """Return centered sub-frame times without changing output timing.

    Each output frame still corresponds to exactly one input frame. Multiple
    poses are sampled inside a centered shutter interval and accumulated into
    that frame, which adds motion blur without changing frame count or FPS.
    """

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if temporal_samples <= 0:
        raise ValueError("temporal_samples must be positive")
    if not 0.0 <= shutter_fraction <= 1.0:
        raise ValueError("shutter_fraction must be in [0, 1]")
    if frame_count == 1 or temporal_samples == 1 or shutter_fraction == 0:
        return np.asarray([float(frame_index)], dtype=np.float64)
    centers = (np.arange(temporal_samples, dtype=np.float64) + 0.5) / temporal_samples
    offsets = (centers - 0.5) * shutter_fraction
    return np.clip(float(frame_index) + offsets, 0.0, float(frame_count - 1))


def _interpolate_sequence(values: np.ndarray, frame_position: float) -> np.ndarray:
    """Linearly interpolate an array along its leading frame dimension."""

    values = np.asarray(values)
    if len(values) == 0:
        raise ValueError("cannot interpolate an empty sequence")
    position = float(np.clip(frame_position, 0.0, len(values) - 1))
    lower = int(np.floor(position))
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return (1.0 - weight) * values[lower] + weight * values[upper]


def _interpolate_rigid_transform(
    transforms: np.ndarray, frame_position: float
) -> np.ndarray:
    """Interpolate translation and project blended rotation back onto SO(3)."""

    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"expected transforms [T,4,4], got {transforms.shape}")
    position = float(np.clip(frame_position, 0.0, len(transforms) - 1))
    lower = int(np.floor(position))
    upper = min(lower + 1, len(transforms) - 1)
    weight = position - lower
    if lower == upper or weight == 0.0:
        return transforms[lower].copy()

    blended_rotation = (
        (1.0 - weight) * transforms[lower, :3, :3]
        + weight * transforms[upper, :3, :3]
    )
    left, _, right = np.linalg.svd(blended_rotation)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    translation = (
        (1.0 - weight) * transforms[lower, :3, 3]
        + weight * transforms[upper, :3, 3]
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _decontaminate_foreground(
    rgb: np.ndarray, mask: np.ndarray, radius: int = DEFAULT_DECONTAMINATE_RADIUS
) -> np.ndarray:
    """Replace renderer-background-contaminated boundary colors.

    The returned straight RGB is zero outside ``mask``. For the boundary band,
    colors are propagated from eroded, trusted foreground pixels. This is done
    before premultiplication; blurring a binary mask over the raw renderer RGB
    would otherwise pull its gray background into the robot silhouette.
    """

    rgb = np.asarray(rgb, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[:2] != mask.shape or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 and mask must match its first dimensions")
    clean = np.zeros_like(rgb, dtype=np.float32)
    if not mask.any():
        return clean
    if radius <= 0:
        clean[mask] = rgb[mask]
        return clean

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    trusted = cv2.erode(
        mask.astype(np.uint8), kernel, iterations=int(radius)
    ).astype(bool)
    component_count, labels = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8
    )
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    known = np.zeros_like(mask, dtype=bool)
    background_color: np.ndarray | None = None
    for component_index in range(1, component_count):
        component = labels == component_index
        component_trusted = component & trusted
        if component_trusted.any():
            clean[component_trusted] = rgb[component_trusted]
            known[component_trusted] = True
            continue

        # A finger/link thinner than the erosion band has no trusted interior.
        # Use its deepest pixel that differs most from the local clear color as
        # a representative, rather than retaining every contaminated edge
        # pixel. The entire thin component becomes a seed for propagation.
        maximum_distance = float(np.max(distance[component]))
        candidates = component & (distance >= maximum_distance - 1e-6)
        if background_color is None:
            background = rgb[~mask]
            background_color = (
                np.median(background, axis=0)
                if len(background)
                else np.zeros(3, dtype=np.float32)
            )
        candidate_colors = rgb[candidates]
        color_distances = np.sum(
            np.square(candidate_colors - background_color[None]), axis=1
        )
        representative = candidate_colors[int(np.argmax(color_distances))]
        clean[component] = representative
        known[component] = True

    target = mask.copy()
    # A 3x3 propagation reaches at least one pixel per pass. Extra passes cover
    # diagonal corners of the eroded boundary band.
    for _ in range(max(2, int(radius) * 3)):
        remaining = target & ~known
        if not remaining.any():
            break
        counts = cv2.boxFilter(
            known.astype(np.float32),
            ddepth=-1,
            ksize=(3, 3),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        sums = cv2.boxFilter(
            clean * known[..., None],
            ddepth=-1,
            ksize=(3, 3),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        fill = remaining & (counts > 0)
        if not fill.any():
            break
        clean[fill] = sums[fill] / counts[fill][:, None]
        known[fill] = True

    # This is only a numerical safety fallback; every component is seeded
    # above, so ordinary erosion bands resolve during the propagation passes.
    unresolved = target & ~known
    if unresolved.any():
        for component_index in np.unique(labels[unresolved]):
            component = labels == component_index
            source = component & known
            clean[component & unresolved] = np.median(clean[source], axis=0)
    return clean


def _finalize_temporal_accumulation(
    premultiplied_sum: np.ndarray,
    alpha_sum: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return straight RGB, premultiplied RGB, and alpha from sub-frame sums."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    alpha = np.clip(alpha_sum / float(sample_count), 0.0, 1.0).astype(np.float32)
    premultiplied = np.clip(
        premultiplied_sum / float(sample_count), 0.0, 1.0
    ).astype(np.float32)
    straight = np.zeros_like(premultiplied, dtype=np.float32)
    visible = alpha > 1e-6
    straight[visible] = premultiplied[visible] / alpha[visible][:, None]
    return np.clip(straight, 0.0, 1.0), premultiplied, alpha


def _render_output_directories(
    output_dir: Path, extra_names: tuple[str, ...] = ()
) -> dict[str, Path]:
    names = (
        "robot_rgb",
        "robot_premultiplied",
        "robot_alpha",
        "robot_mask",
        *extra_names,
    )
    directories = {name: output_dir / name for name in names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _write_color_matte_frame(
    directories: dict[str, Path],
    frame_index: int,
    straight_rgb: np.ndarray,
    premultiplied_rgb: np.ndarray,
    alpha: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Write lossless straight/premultiplied color and soft/binary mattes."""

    straight_u8 = np.rint(np.clip(straight_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    premultiplied_u8 = np.rint(
        np.clip(premultiplied_rgb, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    alpha_u8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(
        str(directories["robot_rgb"] / f"{frame_index:05d}.png"),
        straight_u8[..., ::-1],
    )
    cv2.imwrite(
        str(directories["robot_premultiplied"] / f"{frame_index:05d}.png"),
        premultiplied_u8[..., ::-1],
    )
    cv2.imwrite(
        str(directories["robot_alpha"] / f"{frame_index:05d}.png"), alpha_u8
    )
    cv2.imwrite(
        str(directories["robot_mask"] / f"{frame_index:05d}.png"),
        np.asarray(binary_mask, dtype=np.uint8) * 255,
    )
    return premultiplied_u8[..., ::-1]


def _capture_foreground(
    camera, mask: np.ndarray, decontaminate_radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Capture one sample and return straight foreground RGB plus binary alpha."""

    color = np.asarray(camera.get_picture("Color")[..., :3], dtype=np.float32)
    color = np.clip(color, 0.0, 1.0)
    binary_alpha = np.asarray(mask, dtype=bool)
    return (
        _decontaminate_foreground(color, binary_alpha, decontaminate_radius),
        binary_alpha.astype(np.float32),
    )


def _accumulate_foreground(
    premultiplied_sum: np.ndarray,
    alpha_sum: np.ndarray,
    straight_rgb: np.ndarray,
    alpha: np.ndarray,
) -> None:
    premultiplied_sum += straight_rgb * alpha[..., None]
    alpha_sum += alpha


def _partition_temporal_masks(
    coverages: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Make temporal class unions disjoint while preserving their full union.

    A pixel can belong to an arm at one shutter sample and a hand (or the
    other robot) at another. Diagnostic masks must still form an exact,
    non-overlapping partition, so each covered pixel is assigned to the class
    visible for the most sub-frame samples. Dict insertion order breaks ties.
    """

    if not coverages:
        raise ValueError("at least one temporal coverage map is required")
    names = tuple(coverages)
    shapes = {np.asarray(coverages[name]).shape for name in names}
    if len(shapes) != 1:
        raise ValueError("temporal coverage map shapes differ")
    stacked = np.stack(
        [np.asarray(coverages[name], dtype=np.uint16) for name in names], axis=-1
    )
    covered = np.max(stacked, axis=-1) > 0
    winner = np.argmax(stacked, axis=-1)
    return {
        name: covered & (winner == class_index)
        for class_index, name in enumerate(names)
    }


@dataclass(frozen=True)
class UR5eShadowRenderTrajectory:
    side: str
    arm_qpos: np.ndarray
    arm_joint_names: tuple[str, ...]
    hand_qpos: np.ndarray
    hand_joint_names: tuple[str, ...]
    combined_urdf_path: Path
    base_translation_world: np.ndarray
    base_rotation_world: np.ndarray
    hidden_arm_visual_links: tuple[str, ...] = ()


def _normalize_hidden_arm_visual_links(
    names: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Validate the proximal UR5e visuals that may be omitted from rendering."""

    normalized = tuple(sorted(set(str(name) for name in names)))
    unsupported = set(normalized) - HIDEABLE_ARM_VISUAL_LINKS
    if unsupported:
        raise ValueError(
            "only configured UR5e arm visual links may be hidden; unsupported: "
            + ", ".join(sorted(unsupported))
        )
    return normalized


def _hide_arm_visual_links(robot: object, names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Hide selected SAPIEN render bodies without changing kinematics/collisions."""

    normalized = _normalize_hidden_arm_visual_links(names)
    if not normalized:
        return normalized
    links = {link.name: link for link in robot.get_links()}
    missing = set(normalized) - set(links)
    if missing:
        raise ValueError("combined URDF is missing links: " + ", ".join(sorted(missing)))
    for name in normalized:
        render_components = [
            component
            for component in links[name].entity.components
            if hasattr(component, "visibility") and hasattr(component, "render_shapes")
        ]
        if not render_components:
            raise RuntimeError(f"{name} has no SAPIEN render component")
        for component in render_components:
            component.visibility = 0.0
    return normalized


def _glb_urdf_path(urdf_path: Path) -> Path:
    if "glb" in urdf_path.stem:
        return urdf_path
    candidate = urdf_path.with_stem(urdf_path.stem + "_glb")
    return candidate if candidate.exists() else urdf_path


def render_robot_sequence(
    qpos: np.ndarray,
    joint_names: tuple[str, ...] | list[str],
    urdf_path: str | Path,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    output_dir: str | Path,
    fps: float,
    render_device: str,
    temporal_samples: int = DEFAULT_TEMPORAL_SAMPLES,
    shutter_fraction: float = DEFAULT_SHUTTER_FRACTION,
    decontaminate_radius: int = DEFAULT_DECONTAMINATE_RADIUS,
) -> None:
    """Render robot color and mattes with temporal supersampling in SAPIEN."""

    import sapien
    from dex_retargeting import yourdfpy as urdf

    qpos = np.asarray(qpos, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    output_dir = Path(output_dir)
    directories = _render_output_directories(output_dir)

    scene = _create_sapien_scene(render_device)
    scene.set_ambient_light([0.55, 0.55, 0.55])
    scene.add_directional_light([0.4, -0.4, -1.0], [1.8, 1.8, 1.8], shadow=True)
    scene.add_directional_light([-0.4, 0.4, -0.2], [0.8, 0.8, 0.8], shadow=False)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True

    source_urdf = _glb_urdf_path(Path(urdf_path).resolve())
    robot_urdf = urdf.URDF.load(
        str(source_urdf), add_dummy_free_joints=True, build_scene_graph=False
    )

    with tempfile.TemporaryDirectory(prefix="egodex-dexhand-") as temp_dir:
        augmented_path = Path(temp_dir) / source_urdf.name
        robot_urdf.write_xml_file(str(augmented_path))
        robot = loader.load(str(augmented_path))
        if robot is None:
            raise RuntimeError(f"SAPIEN failed to load {source_urdf}")

        sapien_names = [joint.name for joint in robot.get_active_joints()]
        name_to_index = {name: index for index, name in enumerate(joint_names)}
        try:
            retarget_to_sapien = np.asarray(
                [name_to_index[name] for name in sapien_names], dtype=np.int64
            )
        except KeyError as exc:
            raise RuntimeError(f"renderer joint missing from retarget result: {exc}") from exc

        camera = scene.add_camera("egodex", width, height, 1.0, 0.01, 10.0)
        camera.set_local_pose(sapien.Pose())
        camera.set_perspective_parameters(
            0.01,
            10.0,
            float(intrinsic[0, 0]),
            float(intrinsic[1, 1]),
            float(intrinsic[0, 2]),
            float(intrinsic[1, 2]),
            float(intrinsic[0, 1]),
        )

        writer = cv2.VideoWriter(
            str(output_dir / "robot_rgb.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("could not create robot_rgb.mp4")

        mask_pixels = []
        for frame_index in range(len(qpos)):
            premultiplied_sum = np.zeros((height, width, 3), dtype=np.float32)
            alpha_sum = np.zeros((height, width), dtype=np.float32)
            mask_union = np.zeros((height, width), dtype=bool)
            sample_positions = _temporal_sample_positions(
                frame_index,
                len(qpos),
                temporal_samples=temporal_samples,
                shutter_fraction=shutter_fraction,
            )
            for sample_position in sample_positions:
                frame_qpos = _interpolate_sequence(qpos, sample_position)
                robot.set_qpos(frame_qpos[retarget_to_sapien])
                scene.update_render()
                camera.take_picture()
                segmentation = camera.get_picture("Segmentation")
                mask = segmentation[..., 0] > 0
                straight_sample, alpha_sample = _capture_foreground(
                    camera, mask, decontaminate_radius
                )
                _accumulate_foreground(
                    premultiplied_sum,
                    alpha_sum,
                    straight_sample,
                    alpha_sample,
                )
                mask_union |= mask

            if not mask_union.any():
                raise RuntimeError(f"empty robot render at frame {frame_index}")
            straight, premultiplied, alpha = _finalize_temporal_accumulation(
                premultiplied_sum, alpha_sum, len(sample_positions)
            )
            mask_pixels.append(int(np.count_nonzero(mask_union)))
            writer.write(
                _write_color_matte_frame(
                    directories,
                    frame_index,
                    straight,
                    premultiplied,
                    alpha,
                    mask_union,
                )
            )
        writer.release()

    mask_pixels = np.asarray(mask_pixels)
    if np.any(mask_pixels < 16):
        raise RuntimeError("one or more rendered robot masks are implausibly small")


def _sapien_pose_from_matrix(rotation: np.ndarray, translation: np.ndarray):
    import sapien
    from scipy.spatial.transform import Rotation

    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    return sapien.Pose(np.asarray(translation), quaternion_wxyz)


def render_arm_hand_sequence(
    hand_qpos: np.ndarray,
    hand_joint_names: tuple[str, ...] | list[str],
    hand_urdf_path: str | Path,
    arm_qpos: np.ndarray,
    arm_joint_names: tuple[str, ...] | list[str],
    arm_urdf_path: str | Path,
    base_translation_world: np.ndarray,
    base_rotation_world: np.ndarray,
    world_from_camera: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    output_dir: str | Path,
    fps: float,
    render_device: str,
    temporal_samples: int = DEFAULT_TEMPORAL_SAMPLES,
    shutter_fraction: float = DEFAULT_SHUTTER_FRACTION,
    decontaminate_radius: int = DEFAULT_DECONTAMINATE_RADIUS,
) -> None:
    """Render a Panda and floating hand with a soft premultiplied matte."""

    import sapien
    from dex_retargeting import yourdfpy as urdf

    hand_qpos = np.asarray(hand_qpos, dtype=np.float32)
    arm_qpos = np.asarray(arm_qpos, dtype=np.float32)
    world_from_camera = np.asarray(world_from_camera, dtype=np.float64)
    if len(hand_qpos) != len(arm_qpos) or len(arm_qpos) != len(world_from_camera):
        raise ValueError("hand, arm, and camera trajectories must have equal length")

    output_dir = Path(output_dir)
    directories = _render_output_directories(
        output_dir, extra_names=("arm_mask", "hand_mask")
    )

    scene = _create_sapien_scene(render_device)
    scene.set_ambient_light([0.55, 0.55, 0.55])
    scene.add_directional_light([0.4, -0.4, -1.0], [1.8, 1.8, 1.8], shadow=True)
    scene.add_directional_light([-0.4, 0.4, -0.2], [0.8, 0.8, 0.8], shadow=False)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True

    arm = loader.load(str(Path(arm_urdf_path).resolve()))
    if arm is None:
        raise RuntimeError(f"SAPIEN failed to load Panda URDF: {arm_urdf_path}")
    arm_sapien_names = [joint.name for joint in arm.get_active_joints()]
    arm_name_to_index = {name: index for index, name in enumerate(arm_joint_names)}
    try:
        arm_order = np.asarray(
            [arm_name_to_index[name] for name in arm_sapien_names], dtype=np.int64
        )
    except KeyError as exc:
        raise RuntimeError(f"Panda renderer joint missing from IK result: {exc}") from exc

    source_hand_urdf = _glb_urdf_path(Path(hand_urdf_path).resolve())
    hand_description = urdf.URDF.load(
        str(source_hand_urdf), add_dummy_free_joints=True, build_scene_graph=False
    )
    with tempfile.TemporaryDirectory(prefix="egodex-arm-hand-") as temp_dir:
        augmented_path = Path(temp_dir) / source_hand_urdf.name
        hand_description.write_xml_file(str(augmented_path))
        hand = loader.load(str(augmented_path))
        if hand is None:
            raise RuntimeError(f"SAPIEN failed to load dexterous hand: {hand_urdf_path}")
        hand_sapien_names = [joint.name for joint in hand.get_active_joints()]
        hand_name_to_index = {name: index for index, name in enumerate(hand_joint_names)}
        try:
            hand_order = np.asarray(
                [hand_name_to_index[name] for name in hand_sapien_names], dtype=np.int64
            )
        except KeyError as exc:
            raise RuntimeError(f"hand renderer joint missing from retarget result: {exc}") from exc

        arm_actor_ids = np.asarray(
            [link.entity.per_scene_id for link in arm.get_links()], dtype=np.int64
        )
        hand_actor_ids = np.asarray(
            [link.entity.per_scene_id for link in hand.get_links()], dtype=np.int64
        )

        camera = scene.add_camera("egodex", width, height, 1.0, 0.01, 10.0)
        camera.set_local_pose(sapien.Pose())
        camera.set_perspective_parameters(
            0.01,
            10.0,
            float(intrinsic[0, 0]),
            float(intrinsic[1, 1]),
            float(intrinsic[0, 2]),
            float(intrinsic[1, 2]),
            float(intrinsic[0, 1]),
        )

        writer = cv2.VideoWriter(
            str(output_dir / "robot_rgb.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("could not create robot_rgb.mp4")

        combined_areas = []
        arm_areas = []
        for frame_index in range(len(arm_qpos)):
            premultiplied_sum = np.zeros((height, width, 3), dtype=np.float32)
            alpha_sum = np.zeros((height, width), dtype=np.float32)
            arm_coverage = np.zeros((height, width), dtype=np.uint16)
            hand_coverage = np.zeros((height, width), dtype=np.uint16)
            sample_positions = _temporal_sample_positions(
                frame_index,
                len(arm_qpos),
                temporal_samples=temporal_samples,
                shutter_fraction=shutter_fraction,
            )
            for sample_position in sample_positions:
                sample_world_from_camera = _interpolate_rigid_transform(
                    world_from_camera, sample_position
                )
                camera_from_world = np.linalg.inv(sample_world_from_camera)
                rotation_cv = camera_from_world[:3, :3] @ base_rotation_world
                translation_cv = (
                    camera_from_world[:3, :3] @ base_translation_world
                    + camera_from_world[:3, 3]
                )
                arm.set_root_pose(
                    _sapien_pose_from_matrix(
                        CV_TO_SAPIEN @ rotation_cv,
                        CV_TO_SAPIEN @ translation_cv,
                    )
                )
                arm_sample = _interpolate_sequence(arm_qpos, sample_position)
                hand_sample = _interpolate_sequence(hand_qpos, sample_position)
                arm.set_qpos(arm_sample[arm_order])
                hand.set_qpos(hand_sample[hand_order])

                scene.update_render()
                camera.take_picture()
                segmentation = camera.get_picture("Segmentation")
                visual_mask = segmentation[..., 0] > 0
                actor_ids = segmentation[..., 1]
                arm_mask = visual_mask & np.isin(actor_ids, arm_actor_ids)
                hand_mask = visual_mask & np.isin(actor_ids, hand_actor_ids)
                combined_mask = arm_mask | hand_mask
                straight_sample, alpha_sample = _capture_foreground(
                    camera, combined_mask, decontaminate_radius
                )
                _accumulate_foreground(
                    premultiplied_sum,
                    alpha_sum,
                    straight_sample,
                    alpha_sample,
                )
                arm_coverage += arm_mask
                hand_coverage += hand_mask

            partition = _partition_temporal_masks(
                {"arm": arm_coverage, "hand": hand_coverage}
            )
            arm_union = partition["arm"]
            hand_union = partition["hand"]
            combined_union = arm_union | hand_union
            combined_area = int(np.count_nonzero(combined_union))
            if combined_area < 64:
                raise RuntimeError(f"empty combined robot render at frame {frame_index}")
            straight, premultiplied, alpha = _finalize_temporal_accumulation(
                premultiplied_sum, alpha_sum, len(sample_positions)
            )
            writer.write(
                _write_color_matte_frame(
                    directories,
                    frame_index,
                    straight,
                    premultiplied,
                    alpha,
                    combined_union,
                )
            )
            cv2.imwrite(
                str(directories["arm_mask"] / f"{frame_index:05d}.png"),
                arm_union.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                str(directories["hand_mask"] / f"{frame_index:05d}.png"),
                hand_union.astype(np.uint8) * 255,
            )
            combined_areas.append(combined_area)
            arm_areas.append(int(np.count_nonzero(arm_union)))
        writer.release()

    # The captured wrist exits through the lower-right edge late in this clip,
    # so the arm can legitimately be fully outside while the hand remains in
    # frame.  Require a visible arm for most of the sequence, not every frame.
    visible_arm_frames = sum(area >= 16 for area in arm_areas)
    if min(combined_areas) < 64 or visible_arm_frames < len(arm_areas) // 2:
        raise RuntimeError("one or more whole-arm robot masks are implausibly small")


def render_ur5e_shadow_sequence(
    arm_qpos: np.ndarray,
    arm_joint_names: tuple[str, ...] | list[str],
    hand_qpos: np.ndarray,
    hand_joint_names: tuple[str, ...] | list[str],
    combined_urdf_path: str | Path,
    base_translation_world: np.ndarray,
    base_rotation_world: np.ndarray,
    world_from_camera: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    output_dir: str | Path,
    fps: float,
    render_device: str,
    temporal_samples: int = DEFAULT_TEMPORAL_SAMPLES,
    shutter_fraction: float = DEFAULT_SHUTTER_FRACTION,
    decontaminate_radius: int = DEFAULT_DECONTAMINATE_RADIUS,
    hidden_arm_visual_links: tuple[str, ...] | list[str] = (),
    require_robot_visibility: bool = True,
    require_arm_visibility: bool = True,
) -> None:
    """Render integrated UR5e + Shadow with a soft premultiplied matte."""

    import sapien

    arm_qpos = np.asarray(arm_qpos, dtype=np.float32)
    hand_qpos = np.asarray(hand_qpos, dtype=np.float32)
    world_from_camera = np.asarray(world_from_camera, dtype=np.float64)
    base_translation_world = np.asarray(base_translation_world, dtype=np.float64)
    base_rotation_world = np.asarray(base_rotation_world, dtype=np.float64)
    if len(arm_qpos) != len(hand_qpos) or len(arm_qpos) != len(world_from_camera):
        raise ValueError("arm, hand, and camera trajectories must have equal length")

    output_dir = Path(output_dir)
    directories = _render_output_directories(
        output_dir, extra_names=("arm_mask", "hand_mask")
    )

    scene = _create_sapien_scene(render_device)
    scene.set_ambient_light([0.55, 0.55, 0.55])
    scene.add_directional_light([0.4, -0.4, -1.0], [1.8, 1.8, 1.8], shadow=True)
    scene.add_directional_light([-0.4, 0.4, -0.2], [0.8, 0.8, 0.8], shadow=False)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(str(Path(combined_urdf_path).resolve()))
    if robot is None:
        raise RuntimeError(f"SAPIEN failed to load {combined_urdf_path}")
    hidden_arm_visual_links = _hide_arm_visual_links(
        robot, hidden_arm_visual_links
    )

    arm_lookup = {str(name): index for index, name in enumerate(arm_joint_names)}
    hand_lookup = {str(name): index for index, name in enumerate(hand_joint_names)}
    sapien_names = [joint.name for joint in robot.get_active_joints()]
    sources: list[tuple[str, int]] = []
    for name in sapien_names:
        if name in arm_lookup:
            sources.append(("arm", arm_lookup[name]))
        elif name in hand_lookup:
            sources.append(("hand", hand_lookup[name]))
        else:
            raise RuntimeError(f"combined URDF joint has no trajectory value: {name}")

    arm_link_names = {
        "base_link",
        "base_link_inertia",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        # Shadow's 0.213 m forearm housing is the visible distal arm in this
        # egocentric crop, even when the UR5e links themselves are offscreen.
        "forearm",
        "base",
        "flange",
        "tool0",
        "ee_link",
    }
    arm_actor_ids = np.asarray(
        [
            link.entity.per_scene_id
            for link in robot.get_links()
            if link.name in arm_link_names
            and link.name not in hidden_arm_visual_links
        ],
        dtype=np.int64,
    )
    hand_actor_ids = np.asarray(
        [
            link.entity.per_scene_id
            for link in robot.get_links()
            if link.name not in arm_link_names
        ],
        dtype=np.int64,
    )

    camera = scene.add_camera("egodex", width, height, 1.0, 0.01, 10.0)
    camera.set_local_pose(sapien.Pose())
    camera.set_perspective_parameters(
        0.01,
        10.0,
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
        float(intrinsic[0, 1]),
    )
    writer = cv2.VideoWriter(
        str(output_dir / "robot_rgb.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("could not create robot_rgb.mp4")

    combined_areas = []
    arm_areas = []
    for frame_index in range(len(arm_qpos)):
        premultiplied_sum = np.zeros((height, width, 3), dtype=np.float32)
        alpha_sum = np.zeros((height, width), dtype=np.float32)
        arm_coverage = np.zeros((height, width), dtype=np.uint16)
        hand_coverage = np.zeros((height, width), dtype=np.uint16)
        sample_positions = _temporal_sample_positions(
            frame_index,
            len(arm_qpos),
            temporal_samples=temporal_samples,
            shutter_fraction=shutter_fraction,
        )
        for sample_position in sample_positions:
            sample_world_from_camera = _interpolate_rigid_transform(
                world_from_camera, sample_position
            )
            camera_from_world = np.linalg.inv(sample_world_from_camera)
            rotation_cv = camera_from_world[:3, :3] @ base_rotation_world
            translation_cv = (
                camera_from_world[:3, :3] @ base_translation_world
                + camera_from_world[:3, 3]
            )
            robot.set_root_pose(
                _sapien_pose_from_matrix(
                    CV_TO_SAPIEN @ rotation_cv,
                    CV_TO_SAPIEN @ translation_cv,
                )
            )
            arm_sample = _interpolate_sequence(arm_qpos, sample_position)
            hand_sample = _interpolate_sequence(hand_qpos, sample_position)
            frame_qpos = np.empty(len(sources), dtype=np.float32)
            for output_index, (source, source_index) in enumerate(sources):
                frame_qpos[output_index] = (
                    arm_sample[source_index]
                    if source == "arm"
                    else hand_sample[source_index]
                )
            robot.set_qpos(frame_qpos)
            scene.update_render()
            camera.take_picture()
            segmentation = camera.get_picture("Segmentation")
            visual_mask = segmentation[..., 0] > 0
            actor_ids = segmentation[..., 1]
            arm_mask = visual_mask & np.isin(actor_ids, arm_actor_ids)
            hand_mask = visual_mask & np.isin(actor_ids, hand_actor_ids)
            combined_mask = arm_mask | hand_mask
            straight_sample, alpha_sample = _capture_foreground(
                camera, combined_mask, decontaminate_radius
            )
            _accumulate_foreground(
                premultiplied_sum,
                alpha_sum,
                straight_sample,
                alpha_sample,
            )
            arm_coverage += arm_mask
            hand_coverage += hand_mask

        partition = _partition_temporal_masks(
            {"arm": arm_coverage, "hand": hand_coverage}
        )
        arm_union = partition["arm"]
        hand_union = partition["hand"]
        combined_union = arm_union | hand_union
        combined_area = int(np.count_nonzero(combined_union))
        # A hand can legitimately enter or leave through the image boundary.
        # Preserve frame alignment with a transparent render instead of
        # aborting the complete segment on that recoverable frame.
        straight, premultiplied, alpha = _finalize_temporal_accumulation(
            premultiplied_sum, alpha_sum, len(sample_positions)
        )
        writer.write(
            _write_color_matte_frame(
                directories,
                frame_index,
                straight,
                premultiplied,
                alpha,
                combined_union,
            )
        )
        cv2.imwrite(
            str(directories["arm_mask"] / f"{frame_index:05d}.png"),
            arm_union.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(directories["hand_mask"] / f"{frame_index:05d}.png"),
            hand_union.astype(np.uint8) * 255,
        )
        combined_areas.append(combined_area)
        arm_areas.append(int(np.count_nonzero(arm_union)))
    writer.release()

    _validate_single_robot_visibility(
        combined_areas,
        arm_areas,
        require_robot_visibility=require_robot_visibility,
        require_arm_visibility=require_arm_visibility,
    )


def _validate_single_robot_visibility(
    combined_areas: list[int],
    arm_areas: list[int],
    *,
    require_robot_visibility: bool,
    require_arm_visibility: bool,
) -> None:
    visible_robot_frames = sum(area >= 64 for area in combined_areas)
    visible_arm_frames = sum(area >= 16 for area in arm_areas)
    if (require_robot_visibility and visible_robot_frames == 0) or (
        require_arm_visibility and visible_arm_frames == 0
    ):
        raise RuntimeError("UR5e + Shadow visibility validation failed")


def _validate_bimanual_robot_visibility(
    combined_areas: list[int],
    side_robot_areas: dict[str, list[int]],
    side_arm_areas: dict[str, list[int]],
    *,
    require_robot_visibility_by_side: dict[str, bool],
    require_arm_visibility: bool,
) -> None:
    if any(require_robot_visibility_by_side.values()) and not any(
        area >= 128 for area in combined_areas
    ):
        raise RuntimeError("bimanual combined robot mask is implausibly small")
    for side in ("left", "right"):
        side_required = bool(require_robot_visibility_by_side[side])
        if side_required and not any(area >= 32 for area in side_robot_areas[side]):
            raise RuntimeError(f"{side} robot is never visible")
        if side_required and require_arm_visibility and max(side_arm_areas[side]) < 16:
            raise RuntimeError(f"{side} arm is never visible")


def render_bimanual_ur5e_shadow_sequence(
    trajectories: tuple[UR5eShadowRenderTrajectory, UR5eShadowRenderTrajectory],
    world_from_camera: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    output_dir: str | Path,
    fps: float,
    render_device: str,
    temporal_samples: int = DEFAULT_TEMPORAL_SAMPLES,
    shutter_fraction: float = DEFAULT_SHUTTER_FRACTION,
    decontaminate_radius: int = DEFAULT_DECONTAMINATE_RADIUS,
    require_robot_visibility_by_side: dict[str, bool] | None = None,
    require_arm_visibility: bool = True,
) -> None:
    """Render both UR5e + Shadow arms with a soft premultiplied matte."""

    import sapien

    if {trajectory.side for trajectory in trajectories} != {"left", "right"}:
        raise ValueError("bimanual rendering requires one left and one right trajectory")
    if require_robot_visibility_by_side is None:
        require_robot_visibility_by_side = {"left": True, "right": True}
    elif set(require_robot_visibility_by_side) != {"left", "right"}:
        raise ValueError(
            "require_robot_visibility_by_side must contain left and right"
        )
    world_from_camera = np.asarray(world_from_camera, dtype=np.float64)
    frame_count = len(world_from_camera)
    if world_from_camera.shape != (frame_count, 4, 4):
        raise ValueError("camera trajectory must have shape [T,4,4]")

    output_dir = Path(output_dir)
    extra_directory_names = (
        "arm_mask",
        "hand_mask",
        "left_robot_mask",
        "right_robot_mask",
        "left_arm_mask",
        "right_arm_mask",
        "left_hand_mask",
        "right_hand_mask",
    )
    directories = _render_output_directories(
        output_dir, extra_names=extra_directory_names
    )

    scene = _create_sapien_scene(render_device)
    scene.set_ambient_light([0.55, 0.55, 0.55])
    scene.add_directional_light([0.4, -0.4, -1.0], [1.8, 1.8, 1.8], shadow=True)
    scene.add_directional_light([-0.4, 0.4, -0.2], [0.8, 0.8, 0.8], shadow=False)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True

    arm_link_names = {
        "base_link",
        "base_link_inertia",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "forearm",
        "base",
        "flange",
        "tool0",
        "ee_link",
    }
    states: dict[str, dict[str, object]] = {}
    for trajectory in trajectories:
        arm_qpos = np.asarray(trajectory.arm_qpos, dtype=np.float32)
        hand_qpos = np.asarray(trajectory.hand_qpos, dtype=np.float32)
        if len(arm_qpos) != frame_count or len(hand_qpos) != frame_count:
            raise ValueError(f"{trajectory.side} trajectory length does not match camera")
        robot = loader.load(str(Path(trajectory.combined_urdf_path).resolve()))
        if robot is None:
            raise RuntimeError(f"SAPIEN failed to load {trajectory.combined_urdf_path}")
        hidden_arm_visual_links = _hide_arm_visual_links(
            robot, trajectory.hidden_arm_visual_links
        )

        arm_lookup = {
            str(name): index for index, name in enumerate(trajectory.arm_joint_names)
        }
        hand_lookup = {
            str(name): index for index, name in enumerate(trajectory.hand_joint_names)
        }
        sources: list[tuple[str, int]] = []
        for joint in robot.get_active_joints():
            if joint.name in arm_lookup:
                sources.append(("arm", arm_lookup[joint.name]))
            elif joint.name in hand_lookup:
                sources.append(("hand", hand_lookup[joint.name]))
            else:
                raise RuntimeError(
                    f"{trajectory.side} combined URDF joint has no trajectory value: "
                    f"{joint.name}"
                )

        arm_actor_ids = np.asarray(
            [
                link.entity.per_scene_id
                for link in robot.get_links()
                if link.name in arm_link_names
                and link.name not in hidden_arm_visual_links
            ],
            dtype=np.int64,
        )
        hand_actor_ids = np.asarray(
            [
                link.entity.per_scene_id
                for link in robot.get_links()
                if link.name not in arm_link_names
            ],
            dtype=np.int64,
        )
        if not len(arm_actor_ids) or not len(hand_actor_ids):
            raise RuntimeError(f"could not split {trajectory.side} arm and hand links")
        states[trajectory.side] = {
            "robot": robot,
            "sources": sources,
            "arm_qpos": arm_qpos,
            "hand_qpos": hand_qpos,
            "base_translation": np.asarray(
                trajectory.base_translation_world, dtype=np.float64
            ),
            "base_rotation": np.asarray(
                trajectory.base_rotation_world, dtype=np.float64
            ),
            "arm_actor_ids": arm_actor_ids,
            "hand_actor_ids": hand_actor_ids,
        }

    camera = scene.add_camera("egodex", width, height, 1.0, 0.01, 10.0)
    camera.set_local_pose(sapien.Pose())
    camera.set_perspective_parameters(
        0.01,
        10.0,
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
        float(intrinsic[0, 1]),
    )
    writer = cv2.VideoWriter(
        str(output_dir / "robot_rgb.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("could not create bimanual robot_rgb.mp4")

    combined_areas: list[int] = []
    side_robot_areas: dict[str, list[int]] = {"left": [], "right": []}
    side_arm_areas: dict[str, list[int]] = {"left": [], "right": []}
    for frame_index in range(frame_count):
        premultiplied_sum = np.zeros((height, width, 3), dtype=np.float32)
        alpha_sum = np.zeros((height, width), dtype=np.float32)
        class_coverages = {
            f"{side}_{kind}": np.zeros((height, width), dtype=np.uint16)
            for side in states
            for kind in ("arm", "hand")
        }
        sample_positions = _temporal_sample_positions(
            frame_index,
            frame_count,
            temporal_samples=temporal_samples,
            shutter_fraction=shutter_fraction,
        )
        for sample_position in sample_positions:
            sample_world_from_camera = _interpolate_rigid_transform(
                world_from_camera, sample_position
            )
            camera_from_world = np.linalg.inv(sample_world_from_camera)
            for state in states.values():
                base_rotation = state["base_rotation"]
                base_translation = state["base_translation"]
                rotation_cv = camera_from_world[:3, :3] @ base_rotation
                translation_cv = (
                    camera_from_world[:3, :3] @ base_translation
                    + camera_from_world[:3, 3]
                )
                robot = state["robot"]
                robot.set_root_pose(
                    _sapien_pose_from_matrix(
                        CV_TO_SAPIEN @ rotation_cv,
                        CV_TO_SAPIEN @ translation_cv,
                    )
                )
                sources = state["sources"]
                arm_sample = _interpolate_sequence(
                    state["arm_qpos"], sample_position
                )
                hand_sample = _interpolate_sequence(
                    state["hand_qpos"], sample_position
                )
                frame_qpos = np.empty(len(sources), dtype=np.float32)
                for output_index, (source, source_index) in enumerate(sources):
                    frame_qpos[output_index] = (
                        arm_sample[source_index]
                        if source == "arm"
                        else hand_sample[source_index]
                    )
                robot.set_qpos(frame_qpos)

            scene.update_render()
            camera.take_picture()
            segmentation = camera.get_picture("Segmentation")
            visual_mask = segmentation[..., 0] > 0
            actor_ids = segmentation[..., 1]
            sample_combined = np.zeros((height, width), dtype=bool)
            for side, state in states.items():
                arm_mask = visual_mask & np.isin(actor_ids, state["arm_actor_ids"])
                hand_mask = visual_mask & np.isin(actor_ids, state["hand_actor_ids"])
                robot_mask = arm_mask | hand_mask
                class_coverages[f"{side}_arm"] += arm_mask
                class_coverages[f"{side}_hand"] += hand_mask
                sample_combined |= robot_mask
            straight_sample, alpha_sample = _capture_foreground(
                camera, sample_combined, decontaminate_radius
            )
            _accumulate_foreground(
                premultiplied_sum,
                alpha_sum,
                straight_sample,
                alpha_sample,
            )

        class_partition = _partition_temporal_masks(class_coverages)
        side_unions = {
            side: {
                "arm": class_partition[f"{side}_arm"],
                "hand": class_partition[f"{side}_hand"],
            }
            for side in states
        }
        for side in states:
            side_unions[side]["robot"] = (
                side_unions[side]["arm"] | side_unions[side]["hand"]
            )
            robot_union = side_unions[side]["robot"]
            arm_union = side_unions[side]["arm"]
            side_robot_areas[side].append(int(np.count_nonzero(robot_union)))
            side_arm_areas[side].append(int(np.count_nonzero(arm_union)))
            for kind in ("robot", "arm", "hand"):
                cv2.imwrite(
                    str(
                        directories[f"{side}_{kind}_mask"]
                        / f"{frame_index:05d}.png"
                    ),
                    side_unions[side][kind].astype(np.uint8) * 255,
                )

        combined_arm = side_unions["left"]["arm"] | side_unions["right"]["arm"]
        combined_hand = side_unions["left"]["hand"] | side_unions["right"]["hand"]
        combined_mask = combined_arm | combined_hand
        area = int(np.count_nonzero(combined_mask))
        # Do not make a transient off-screen frame fatal.  The zero-alpha
        # image is a valid, frame-aligned render and later compositing leaves
        # the source pixels unchanged there.
        combined_areas.append(area)
        straight, premultiplied, alpha = _finalize_temporal_accumulation(
            premultiplied_sum, alpha_sum, len(sample_positions)
        )
        writer.write(
            _write_color_matte_frame(
                directories,
                frame_index,
                straight,
                premultiplied,
                alpha,
                combined_mask,
            )
        )
        for name, mask in (("arm_mask", combined_arm), ("hand_mask", combined_hand)):
            cv2.imwrite(
                str(directories[name] / f"{frame_index:05d}.png"),
                mask.astype(np.uint8) * 255,
            )
    writer.release()

    _validate_bimanual_robot_visibility(
        combined_areas,
        side_robot_areas,
        side_arm_areas,
        require_robot_visibility_by_side=require_robot_visibility_by_side,
        require_arm_visibility=require_arm_visibility,
    )
