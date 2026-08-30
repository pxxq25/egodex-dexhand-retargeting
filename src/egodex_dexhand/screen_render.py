from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .data import CV_TO_SAPIEN
from .render import (
    _capture_foreground,
    _create_sapien_scene,
    _glb_urdf_path,
    _hide_arm_visual_links,
    _render_output_directories,
    _sapien_pose_from_matrix,
    _write_color_matte_frame,
)
from .screen_registration import ShadowScreenRegistration
from .visual_forearm import ForearmObservationSequence, estimate_forearm_silhouette


@dataclass(frozen=True)
class ScreenRenderSummary:
    frame_count: int
    hand_scale: float
    forearm_scale_xyz: tuple[float, float, float]
    robot_frame_ratio_max: float
    robot_frame_ratio_p95: float
    forearm_expected_frames: int
    forearm_visible_frames: int
    forearm_expected_visibility_ratio: float
    forearm_direction_observable_frames: int
    forearm_direction_evaluated_frames: int
    forearm_direction_evaluated_ratio: float
    forearm_direction_error_mean_degrees: float
    forearm_direction_error_p95_degrees: float
    forearm_direction_error_max_degrees: float
    forearm_pose_direction_error_mean_degrees: float
    forearm_pose_direction_error_p95_degrees: float
    forearm_pose_direction_error_max_degrees: float


def _point_image_distance(point: np.ndarray, width: int, height: int) -> float:
    """Return zero in-frame and Euclidean distance to the nearest border outside."""

    value = np.asarray(point, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("image point must be finite [2]")
    horizontal = max(0.0, -value[0], value[0] - (width - 1.0))
    vertical = max(0.0, -value[1], value[1] - (height - 1.0))
    return float(np.hypot(horizontal, vertical))


def _directed_segment_image_length(
    origin: np.ndarray,
    direction: np.ndarray,
    maximum_length: float,
    width: int,
    height: int,
) -> float:
    """Length of a directed 2-D segment that is actually visible in-frame."""

    point = np.asarray(origin, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)
    length = float(maximum_length)
    if point.shape != (2,) or vector.shape != (2,):
        raise ValueError("origin and direction must be 2-vectors")
    if not np.isfinite(point).all() or not np.isfinite(vector).all():
        raise ValueError("origin and direction must be finite")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8 or not np.isfinite(length) or length <= 0:
        raise ValueError("direction and maximum_length must be positive")
    vector /= norm
    lower = 0.0
    upper = length
    for coordinate, delta, limit in zip(
        point, vector, (float(width - 1), float(height - 1))
    ):
        if abs(float(delta)) < 1e-12:
            if coordinate < 0.0 or coordinate > limit:
                return 0.0
            continue
        first = (0.0 - coordinate) / delta
        second = (limit - coordinate) / delta
        entry, exit_ = sorted((float(first), float(second)))
        lower = max(lower, entry)
        upper = min(upper, exit_)
        if upper <= lower:
            return 0.0
    return float(upper - lower)


def _shadow_forearm_mesh(combined_urdf: Path) -> tuple[Path, np.ndarray]:
    root = ET.parse(combined_urdf).getroot()
    link = root.find("./link[@name='forearm']")
    mesh = None if link is None else link.find("./visual/geometry/mesh")
    if mesh is None or "filename" not in mesh.attrib:
        raise RuntimeError("combined URDF has no Shadow forearm visual mesh")
    mesh_path = Path(mesh.attrib["filename"])
    if not mesh_path.is_absolute():
        mesh_path = (combined_urdf.parent / mesh_path).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    raw_scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ")
    if raw_scale.shape != (3,) or not np.all(raw_scale > 0):
        raise RuntimeError("invalid Shadow forearm mesh scale")
    return mesh_path, raw_scale.astype(np.float64)


def automatic_forearm_scale(
    mesh_path: str | Path,
    urdf_mesh_scale: np.ndarray,
    wrist_in_forearm: np.ndarray,
    observations: ForearmObservationSequence,
    *,
    coverage_margin: float = 1.08,
) -> np.ndarray:
    """Size the mechanical forearm from the observed sleeve, not a clip knob."""

    wrist_local = np.asarray(wrist_in_forearm, dtype=np.float64)
    mesh_scale = np.asarray(urdf_mesh_scale, dtype=np.float64)
    if wrist_local.shape != (3,) or mesh_scale.shape != (3,):
        raise ValueError("wrist and mesh scale must be 3-vectors")
    if coverage_margin < 1.0:
        raise ValueError("coverage_margin must be at least one")
    extents = _mesh_extents(Path(mesh_path).resolve()) * mesh_scale
    if extents.shape != (3,) or not np.isfinite(extents).all() or np.any(extents <= 0):
        raise RuntimeError("could not measure Shadow forearm mesh")

    longitudinal_axis = int(np.argmax(np.abs(wrist_local)))
    cross_axes = [axis for axis in range(3) if axis != longitudinal_axis]
    observed_length = float(np.median(observations.length_camera))
    observed_width = float(np.quantile(observations.width_camera, 0.75))
    if observed_length <= 0 or observed_width <= 0:
        raise ValueError("observed forearm dimensions must be positive")
    result = np.ones(3, dtype=np.float64)
    local_length = abs(float(wrist_local[longitudinal_axis]))
    if local_length < 1e-6:
        local_length = float(extents[longitudinal_axis])
    result[longitudinal_axis] = coverage_margin * observed_length / local_length
    for axis in cross_axes:
        result[axis] = coverage_margin * observed_width / extents[axis]
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise RuntimeError("automatic forearm scale is invalid")
    return result.astype(np.float32)


def _mesh_extents(path: Path) -> np.ndarray:
    """Read bounds from the GLB/OBJ assets used by the supported robot URDFs."""

    if path.suffix.lower() == ".obj":
        vertices = []
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if line.startswith("v "):
                    values = np.fromstring(line[2:], sep=" ", dtype=np.float64)
                    if values.shape[0] >= 3:
                        vertices.append(values[:3])
        if not vertices:
            raise RuntimeError(f"OBJ has no vertices: {path}")
        array = np.asarray(vertices, dtype=np.float64)
        return np.max(array, axis=0) - np.min(array, axis=0)
    if path.suffix.lower() == ".glb":
        payload = path.read_bytes()
        if len(payload) < 20 or payload[:4] != b"glTF":
            raise RuntimeError(f"invalid GLB header: {path}")
        _, version, total_length = struct.unpack_from("<4sII", payload, 0)
        if version != 2 or total_length != len(payload):
            raise RuntimeError(f"unsupported GLB container: {path}")
        offset = 12
        document = None
        while offset + 8 <= len(payload):
            chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
            offset += 8
            chunk = payload[offset : offset + chunk_length]
            offset += chunk_length
            if chunk_type == 0x4E4F534A:
                document = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\0"))
                break
        if document is None:
            raise RuntimeError(f"GLB has no JSON chunk: {path}")
        accessors = document.get("accessors", [])
        minimums = []
        maximums = []
        for mesh in document.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                accessor_index = primitive.get("attributes", {}).get("POSITION")
                if accessor_index is None:
                    continue
                accessor = accessors[int(accessor_index)]
                if "min" in accessor and "max" in accessor:
                    minimums.append(np.asarray(accessor["min"], dtype=np.float64))
                    maximums.append(np.asarray(accessor["max"], dtype=np.float64))
        if not minimums:
            raise RuntimeError(f"GLB position accessors have no bounds: {path}")
        return np.max(maximums, axis=0) - np.min(minimums, axis=0)
    raise RuntimeError(f"unsupported robot mesh format: {path.suffix}")


def detached_forearm_transform_camera(
    wrist_camera: np.ndarray,
    guide_camera: np.ndarray,
    registered_forearm_rotation_camera: np.ndarray,
    wrist_in_forearm: np.ndarray,
    scale_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the camera-space transform for the sleeve-aligned forearm."""

    wrist = np.asarray(wrist_camera, dtype=np.float64)
    guide = np.asarray(guide_camera, dtype=np.float64)
    hand_rotation = np.asarray(registered_forearm_rotation_camera, dtype=np.float64)
    wrist_local = np.asarray(wrist_in_forearm, dtype=np.float64)
    scale = np.asarray(scale_xyz, dtype=np.float64)
    # The mesh is scaled before it is rotated. With anisotropic XYZ scale the
    # actual actor-space origin-to-wrist vector is scale * wrist_local, not the
    # unscaled vector. Aligning the latter introduces a clip-dependent visual
    # angle even though the requested target direction is correct.
    scaled_wrist_local = scale * wrist_local
    local_primary = scaled_wrist_local / np.linalg.norm(scaled_wrist_local)
    target_primary = wrist - guide
    target_primary /= np.linalg.norm(target_primary)

    candidate_axes = np.eye(3)
    local_secondary = candidate_axes[
        int(np.argmin(np.abs(candidate_axes @ local_primary)))
    ]
    local_secondary -= local_primary * np.dot(local_primary, local_secondary)
    local_secondary /= np.linalg.norm(local_secondary)
    local_third = np.cross(local_primary, local_secondary)
    local_basis = np.stack([local_primary, local_secondary, local_third], axis=1)

    target_secondary = hand_rotation @ local_secondary
    target_secondary -= target_primary * np.dot(target_primary, target_secondary)
    if np.linalg.norm(target_secondary) < 1e-8:
        target_secondary = hand_rotation @ local_third
        target_secondary -= target_primary * np.dot(target_primary, target_secondary)
    target_secondary /= np.linalg.norm(target_secondary)
    target_third = np.cross(target_primary, target_secondary)
    target_basis = np.stack([target_primary, target_secondary, target_third], axis=1)
    rotation_camera = target_basis @ local_basis.T
    origin_camera = wrist - rotation_camera @ scaled_wrist_local
    return rotation_camera, origin_camera


def detached_forearm_pose(
    wrist_camera: np.ndarray,
    guide_camera: np.ndarray,
    registered_forearm_rotation_camera: np.ndarray,
    wrist_in_forearm: np.ndarray,
    scale_xyz: np.ndarray,
):
    """Attach a visual forearm to the fitted wrist while following the sleeve."""

    rotation_camera, origin_camera = detached_forearm_transform_camera(
        wrist_camera,
        guide_camera,
        registered_forearm_rotation_camera,
        wrist_in_forearm,
        scale_xyz,
    )
    return _sapien_pose_from_matrix(
        CV_TO_SAPIEN @ rotation_camera,
        CV_TO_SAPIEN @ origin_camera,
    )


def render_screen_registered_shadow_sequence(
    registration: ShadowScreenRegistration,
    observations: ForearmObservationSequence,
    hand_qpos: np.ndarray,
    hand_joint_names: tuple[str, ...] | list[str],
    standalone_shadow_urdf: str | Path,
    combined_urdf: str | Path,
    wrist_camera: np.ndarray,
    wrist_pixels: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    output_dir: str | Path,
    fps: float,
    render_device: str,
) -> ScreenRenderSummary:
    """Render a fitted hand plus an independently sleeve-aligned forearm."""

    import sapien

    qpos_values = np.asarray(hand_qpos, dtype=np.float32)
    wrist_values = np.asarray(wrist_camera, dtype=np.float64)
    wrist_pixel_values = np.asarray(wrist_pixels, dtype=np.float64)
    frame_count = len(qpos_values)
    if observations.direction_pixels.shape != (frame_count, 2):
        raise ValueError("forearm observations do not match hand trajectory")
    if registration.forearm_position_camera.shape != (frame_count, 3):
        raise ValueError("screen registration does not match hand trajectory")
    if wrist_values.shape != (frame_count, 3) or wrist_pixel_values.shape != (
        frame_count,
        2,
    ):
        raise ValueError("wrist trajectory does not match hand trajectory")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output}")
    directories = _render_output_directories(
        output, extra_names=("arm_mask", "hand_mask")
    )
    combined_path = Path(combined_urdf).resolve()
    mesh_path, mesh_scale = _shadow_forearm_mesh(combined_path)
    automatic_scale = automatic_forearm_scale(
        mesh_path,
        mesh_scale,
        registration.wrist_in_forearm,
        observations,
    )

    scene = _create_sapien_scene(render_device)
    scene.set_ambient_light([0.55, 0.55, 0.55])
    scene.add_directional_light([0.4, -0.4, -1.0], [1.8, 1.8, 1.8], shadow=True)
    scene.add_directional_light([-0.4, 0.4, -0.2], [0.8, 0.8, 0.8], shadow=False)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = float(registration.similarity.scale)
    hand_path = _glb_urdf_path(Path(standalone_shadow_urdf).resolve())
    hand = loader.load(str(hand_path))
    if hand is None:
        raise RuntimeError(f"SAPIEN could not load Shadow Hand {hand_path}")
    _hide_arm_visual_links(hand, ("forearm",))
    hand_lookup = {str(name): index for index, name in enumerate(hand_joint_names)}
    try:
        hand_order = np.asarray(
            [hand_lookup[joint.name] for joint in hand.get_active_joints()],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RuntimeError(f"rendered Shadow joint is missing from qpos: {exc}") from exc

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(
        str(mesh_path),
        scale=tuple((mesh_scale * automatic_scale).tolist()),
        name="observed_shadow_forearm",
    )
    forearm = builder.build_kinematic("observed_shadow_forearm")
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
        str(output / "robot_rgb.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("could not create robot_rgb.mp4")

    robot_areas = []
    direction_errors = []
    pose_direction_errors = []
    expected_forearm_frames = 0
    direction_observable_frames = 0
    visible_forearm_frames = 0
    visible_expected_forearm_frames = 0
    for frame_index in range(frame_count):
        hand.set_root_pose(
            _sapien_pose_from_matrix(
                CV_TO_SAPIEN
                @ np.asarray(
                    registration.forearm_rotation_camera[frame_index],
                    dtype=np.float64,
                ),
                CV_TO_SAPIEN
                @ np.asarray(
                    registration.forearm_position_camera[frame_index],
                    dtype=np.float64,
                ),
            )
        )
        hand.set_qpos(qpos_values[frame_index, hand_order])
        forearm_rotation_camera, forearm_origin_camera = (
            detached_forearm_transform_camera(
                wrist_values[frame_index],
                observations.guide_camera[frame_index],
                registration.forearm_rotation_camera[frame_index],
                registration.wrist_in_forearm,
                automatic_scale,
            )
        )
        forearm.set_pose(
            _sapien_pose_from_matrix(
                CV_TO_SAPIEN @ forearm_rotation_camera,
                CV_TO_SAPIEN @ forearm_origin_camera,
            )
        )
        scene.update_render()
        camera.take_picture()
        segmentation = camera.get_picture("Segmentation")
        visual = segmentation[..., 0] > 0
        actor_ids = segmentation[..., 1]
        hand_mask = visual & np.isin(actor_ids, hand_actor_ids)
        forearm_mask = visual & (actor_ids == forearm.per_scene_id)
        robot_mask = hand_mask | forearm_mask
        if np.count_nonzero(robot_mask) < 64:
            raise RuntimeError(f"robot render is too small at frame {frame_index}")
        straight, alpha = _capture_foreground(
            camera, robot_mask, decontaminate_radius=2
        )
        writer.write(
            _write_color_matte_frame(
                directories,
                frame_index,
                straight,
                straight * alpha[..., None],
                alpha,
                robot_mask,
            )
        )
        cv2.imwrite(
            str(directories["hand_mask"] / f"{frame_index:05d}.png"),
            hand_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(directories["arm_mask"] / f"{frame_index:05d}.png"),
            forearm_mask.astype(np.uint8) * 255,
        )
        visible_centerline_length = _directed_segment_image_length(
            wrist_pixel_values[frame_index],
            observations.direction_pixels[frame_index],
            float(observations.length_pixels[frame_index]),
            width,
            height,
        )
        observed_width = float(observations.width_pixels[frame_index])
        expected_forearm = visible_centerline_length >= max(
            8.0, 0.15 * observed_width
        )
        direction_observable = visible_centerline_length >= max(
            24.0, 0.70 * observed_width
        )
        expected_forearm_frames += int(expected_forearm)
        direction_observable_frames += int(direction_observable)
        forearm_visible = int(np.count_nonzero(forearm_mask)) >= 64
        visible_forearm_frames += int(forearm_visible)
        visible_expected_forearm_frames += int(expected_forearm and forearm_visible)
        projected_origin = intrinsic @ forearm_origin_camera
        projected_origin = projected_origin[:2] / projected_origin[2]
        pose_direction = projected_origin - wrist_pixel_values[frame_index]
        pose_direction /= np.linalg.norm(pose_direction)
        pose_direction_errors.append(
            float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(
                                pose_direction,
                                observations.direction_pixels[frame_index],
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        )
        if forearm_visible and direction_observable:
            try:
                rendered_direction, _, _, _ = estimate_forearm_silhouette(
                    forearm_mask,
                    wrist_pixel_values[frame_index],
                    fallback_direction=observations.direction_pixels[frame_index],
                    fallback_width_pixels=float(
                        observations.width_pixels[frame_index]
                    ),
                )
            except ValueError:
                # A clipped sliver can be visibly correct yet too short for a
                # stable axis. Visibility remains counted and independently
                # gated; no direction value is fabricated for this frame.
                pass
            else:
                direction_errors.append(
                    float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    np.dot(
                                        rendered_direction,
                                        observations.direction_pixels[frame_index],
                                    ),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                )
        robot_areas.append(int(np.count_nonzero(robot_mask)))
    writer.release()

    area_ratios = np.asarray(robot_areas, dtype=np.float64) / float(width * height)
    errors = np.asarray(direction_errors, dtype=np.float64)
    pose_errors = np.asarray(pose_direction_errors, dtype=np.float64)
    expected_visibility_ratio = (
        visible_expected_forearm_frames / expected_forearm_frames
        if expected_forearm_frames
        else 1.0
    )
    direction_evaluated_ratio = len(direction_errors) / max(
        direction_observable_frames, 1
    )
    summary = ScreenRenderSummary(
        frame_count=frame_count,
        hand_scale=float(registration.similarity.scale),
        forearm_scale_xyz=tuple(float(value) for value in automatic_scale),
        robot_frame_ratio_max=float(np.max(area_ratios)),
        robot_frame_ratio_p95=float(np.quantile(area_ratios, 0.95)),
        forearm_expected_frames=expected_forearm_frames,
        forearm_visible_frames=visible_forearm_frames,
        forearm_expected_visibility_ratio=float(expected_visibility_ratio),
        forearm_direction_observable_frames=direction_observable_frames,
        forearm_direction_evaluated_frames=len(direction_errors),
        forearm_direction_evaluated_ratio=float(direction_evaluated_ratio),
        forearm_direction_error_mean_degrees=(
            float(np.mean(errors)) if len(errors) else 0.0
        ),
        forearm_direction_error_p95_degrees=(
            float(np.quantile(errors, 0.95)) if len(errors) else 0.0
        ),
        forearm_direction_error_max_degrees=(
            float(np.max(errors)) if len(errors) else 0.0
        ),
        forearm_pose_direction_error_mean_degrees=float(np.mean(pose_errors)),
        forearm_pose_direction_error_p95_degrees=float(
            np.quantile(pose_errors, 0.95)
        ),
        forearm_pose_direction_error_max_degrees=float(np.max(pose_errors)),
    )
    (output / "screen_render_summary.json").write_text(
        json.dumps(asdict(summary), indent=2) + "\n"
    )
    return summary
