from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


# EgoDex names in the 21-landmark order used by dex-retargeting/MediaPipe.
# The non-thumb metacarpals are deliberately omitted.
HAND_SUFFIXES = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
)

# Change of basis from OpenCV camera coordinates (right, down, forward) to the
# SAPIEN camera convention used by this project (forward, left, up).
CV_TO_SAPIEN = np.asarray(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class EgoDexSequence:
    intrinsic: np.ndarray
    joints_camera_cv: np.ndarray
    joints_camera_sapien: np.ndarray
    joint_confidence: np.ndarray
    hand_transforms_camera: np.ndarray
    hand: str

    @property
    def frame_count(self) -> int:
        return int(self.joints_camera_cv.shape[0])


@dataclass(frozen=True)
class EgoDexArmSequence:
    """Sparse upper-limb tracking supplied by EgoDex.

    Joint order is shoulder, upper-arm marker, forearm marker, and hand/wrist.
    The world-space trajectory is useful for fitting a robot base once, while
    the camera-space trajectory is used for rendering and SAM2 prompts.
    """

    intrinsic: np.ndarray
    world_from_camera: np.ndarray
    joints_world: np.ndarray
    joints_camera_cv: np.ndarray
    joint_confidence: np.ndarray
    hand: str

    @property
    def frame_count(self) -> int:
        return int(self.joints_world.shape[0])


def _hand_prefix(hand: str) -> str:
    normalized = hand.lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
    return normalized


def load_egodex_sequence(hdf5_path: str | Path, hand: str = "right") -> EgoDexSequence:
    """Load one hand in camera coordinates from an EgoDex episode HDF5.

    EgoDex stores world transforms. The documented conversion is
    ``inv(T_world_camera) @ T_world_joint`` for each frame.
    """

    hand = _hand_prefix(hand)
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)

    with h5py.File(hdf5_path, "r") as handle:
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float32)
        world_from_camera = np.asarray(handle["transforms/camera"], dtype=np.float32)
        camera_from_world = np.linalg.inv(world_from_camera)

        transforms_camera = []
        confidence = []
        for suffix in HAND_SUFFIXES:
            name = f"{hand}{suffix}"
            world_from_joint = np.asarray(
                handle[f"transforms/{name}"], dtype=np.float32
            )
            transforms_camera.append(camera_from_world @ world_from_joint)
            confidence.append(
                np.asarray(handle[f"confidences/{name}"], dtype=np.float32)
            )

    # Input lists are joint-major; move the joint dimension behind time.
    hand_transforms_camera = np.stack(transforms_camera, axis=1)
    joint_confidence = np.stack(confidence, axis=1)
    joints_camera_cv = hand_transforms_camera[..., :3, 3].copy()
    joints_camera_sapien = np.einsum(
        "ij,tkj->tki", CV_TO_SAPIEN, joints_camera_cv
    ).astype(np.float32)

    if not np.isfinite(joints_camera_cv).all():
        raise ValueError(f"non-finite hand coordinates in {hdf5_path}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"expected a 3x3 camera intrinsic, got {intrinsic.shape}")

    return EgoDexSequence(
        intrinsic=intrinsic,
        joints_camera_cv=joints_camera_cv,
        joints_camera_sapien=joints_camera_sapien,
        joint_confidence=joint_confidence,
        hand_transforms_camera=hand_transforms_camera,
        hand=hand,
    )


def load_egodex_arm_sequence(
    hdf5_path: str | Path, hand: str = "right"
) -> EgoDexArmSequence:
    """Load EgoDex shoulder-to-wrist markers in world and camera coordinates."""

    hand = _hand_prefix(hand)
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)

    suffixes = ("Shoulder", "Arm", "Forearm", "Hand")
    with h5py.File(hdf5_path, "r") as handle:
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float32)
        world_from_camera = np.asarray(handle["transforms/camera"], dtype=np.float32)
        camera_from_world = np.linalg.inv(world_from_camera)
        transforms_world = []
        confidence = []
        for suffix in suffixes:
            name = f"{hand}{suffix}"
            transforms_world.append(
                np.asarray(handle[f"transforms/{name}"], dtype=np.float32)
            )
            confidence.append(
                np.asarray(handle[f"confidences/{name}"], dtype=np.float32)
            )

    transforms_world_array = np.stack(transforms_world, axis=1)
    joints_world = transforms_world_array[..., :3, 3].copy()
    transforms_camera = camera_from_world[:, None] @ transforms_world_array
    joints_camera_cv = transforms_camera[..., :3, 3].copy()
    joint_confidence = np.stack(confidence, axis=1)

    if intrinsic.shape != (3, 3):
        raise ValueError(f"expected a 3x3 camera intrinsic, got {intrinsic.shape}")
    if not np.isfinite(joints_world).all() or not np.isfinite(joints_camera_cv).all():
        raise ValueError(f"non-finite arm coordinates in {hdf5_path}")
    return EgoDexArmSequence(
        intrinsic=intrinsic,
        world_from_camera=world_from_camera,
        joints_world=joints_world,
        joints_camera_cv=joints_camera_cv,
        joint_confidence=joint_confidence,
        hand=hand,
    )


def project_camera_points(points_cv: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Project OpenCV-camera 3-D points to image pixels."""

    points_cv = np.asarray(points_cv, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    z = points_cv[..., 2:3]
    safe_z = np.where(np.abs(z) > 1e-6, z, np.nan)
    normalized = points_cv[..., :2] / safe_z
    pixels = np.empty_like(normalized)
    pixels[..., 0] = intrinsic[0, 0] * normalized[..., 0] + intrinsic[0, 2]
    pixels[..., 1] = intrinsic[1, 1] * normalized[..., 1] + intrinsic[1, 2]
    return pixels


def scaled_intrinsic(
    intrinsic: np.ndarray, scale_x: float, scale_y: float | None = None
) -> np.ndarray:
    if scale_y is None:
        scale_y = scale_x
    if not 0 < scale_x <= 1 or not 0 < scale_y <= 1:
        raise ValueError(f"scales must be in (0, 1], got {(scale_x, scale_y)}")
    result = np.asarray(intrinsic, dtype=np.float32).copy()
    result[0, :] *= scale_x
    result[1, :] *= scale_y
    result[2, :] = (0.0, 0.0, 1.0)
    return result


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    scale: float = 1.0,
) -> tuple[int, float, tuple[int, int], tuple[int, int]]:
    """Return count, fps, output size, and source size after extracting JPEGs."""

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = max(8, int(round(source_width * scale)) // 8 * 8)
    height = max(8, int(round(source_height * scale)) // 8 * 8)

    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        target = output_dir / f"{count:05d}.jpg"
        if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 96]):
            raise RuntimeError(f"failed to write {target}")
        count += 1
    capture.release()

    if count == 0:
        raise RuntimeError(f"video contained no decodable frames: {video_path}")
    return count, fps, (width, height), (source_width, source_height)


def read_numbered_images(directory: str | Path, grayscale: bool = False) -> list[np.ndarray]:
    directory = Path(directory)
    paths = sorted(directory.glob("*.png"))
    if not paths:
        paths = sorted(directory.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no PNG/JPEG frames in {directory}")
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    images = [cv2.imread(str(path), flag) for path in paths]
    if any(image is None for image in images):
        raise RuntimeError(f"one or more frames could not be decoded in {directory}")
    return images
