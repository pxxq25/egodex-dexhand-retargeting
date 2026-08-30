from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ForearmObservationSequence:
    """Image-derived sleeve axis and size with explicit confidence."""

    direction_pixels: np.ndarray
    guide_camera: np.ndarray
    length_pixels: np.ndarray
    width_pixels: np.ndarray
    length_camera: np.ndarray
    width_camera: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class HumanMaskRefinementSummary:
    """Bounded appearance refinement statistics for reviewer provenance."""

    frame_count: int
    refined_frames: int
    rejected_frames: int
    area_growth_mean: float
    area_growth_p95: float
    area_growth_max: float


def _nearest_component(
    mask: np.ndarray,
    wrist: np.ndarray,
    *,
    max_distance_pixels: float | None = None,
) -> np.ndarray:
    # Segment padding legitimately contains a visible forearm while its wrist
    # is just outside the image. Use the closest image-boundary point only to
    # choose the component; callers retain the real, unclamped wrist for all
    # direction and camera geometry.
    selection_wrist = np.clip(
        np.asarray(wrist, dtype=np.float64),
        np.asarray([0.0, 0.0]),
        np.asarray([mask.shape[1] - 1.0, mask.shape[0] - 1.0]),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        raise ValueError("human mask is empty")
    best_label = -1
    best_distance = np.inf
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) < 32:
            continue
        rows, columns = np.nonzero(labels == label)
        distance = float(
            np.min(
                np.hypot(
                    columns - float(selection_wrist[0]),
                    rows - float(selection_wrist[1]),
                )
            )
        )
        if distance < best_distance:
            best_label = label
            best_distance = distance
    image_diagonal = float(np.hypot(mask.shape[1], mask.shape[0]))
    distance_limit = max(24.0, 0.06 * image_diagonal)
    if max_distance_pixels is not None:
        if not np.isfinite(max_distance_pixels) or max_distance_pixels <= 0:
            raise ValueError("max_distance_pixels must be finite and positive")
        distance_limit = max(distance_limit, float(max_distance_pixels))
    if best_label < 0 or best_distance > distance_limit:
        raise ValueError("no human-mask component reaches the tracked wrist")
    return labels == best_label


def refine_human_silhouette(
    frame: np.ndarray,
    mask: np.ndarray,
    wrist_pixel: np.ndarray,
    palm_width_pixels: float,
    *,
    max_area_growth: float = 2.25,
) -> tuple[np.ndarray, float, bool]:
    """Complete a wrist-connected mask without scene or skin assumptions.

    Segmenters commonly return only one side of a low-contrast sleeve. The
    existing wrist-connected component is a high-precision foreground seed;
    GrabCut may extend it only inside a palm-scaled neighbourhood. A connected
    component check and hard area-growth bound prevent unrelated foreground
    from being absorbed.
    """

    image = np.asarray(frame)
    binary = np.asarray(mask) > 0
    wrist = np.asarray(wrist_pixel, dtype=np.float64)
    palm_width = float(palm_width_pixels)
    if image.ndim != 3 or image.shape[2] != 3 or binary.shape != image.shape[:2]:
        raise ValueError("frame and human mask shapes differ")
    if wrist.shape != (2,) or not np.isfinite(wrist).all():
        raise ValueError("wrist_pixel must be finite [2]")
    if not np.isfinite(palm_width) or palm_width <= 2:
        raise ValueError("palm_width_pixels must be finite and positive")
    if max_area_growth <= 1.0:
        raise ValueError("max_area_growth must exceed one")

    component_distance = 3.0 * palm_width
    base = _nearest_component(
        binary, wrist, max_distance_pixels=component_distance
    )
    base_area = int(np.count_nonzero(base))
    diagonal = float(np.hypot(image.shape[1], image.shape[0]))
    radius = int(
        np.clip(round(0.65 * palm_width), 8, max(8, round(0.08 * diagonal)))
    )
    candidate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    candidate = cv2.dilate(base.astype(np.uint8), candidate_kernel) > 0
    grabcut_mask = np.full(binary.shape, cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[candidate] = cv2.GC_PR_BGD
    grabcut_mask[base] = cv2.GC_PR_FGD
    core_radius = max(1, int(round(0.08 * palm_width)))
    core_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * core_radius + 1, 2 * core_radius + 1)
    )
    core = cv2.erode(base.astype(np.uint8), core_kernel) > 0
    if np.count_nonzero(core) < 32:
        core = base
    grabcut_mask[core] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(
        image,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )
    foreground = (grabcut_mask == cv2.GC_FGD) | (
        grabcut_mask == cv2.GC_PR_FGD
    )
    foreground |= base
    selected = _nearest_component(
        foreground, wrist, max_distance_pixels=component_distance
    )
    refined_area = int(np.count_nonzero(selected))
    growth = refined_area / max(base_area, 1)
    rejected = bool(growth > max_area_growth)
    if rejected:
        selected = base
        growth = 1.0
    return selected.astype(np.uint8) * 255, float(growth), rejected


def refine_human_silhouette_sequence(
    frames: list[np.ndarray] | tuple[np.ndarray, ...],
    masks: list[np.ndarray] | tuple[np.ndarray, ...],
    wrist_pixels: np.ndarray,
    palm_width_pixels: np.ndarray,
    *,
    max_area_growth: float = 2.25,
) -> tuple[list[np.ndarray], HumanMaskRefinementSummary]:
    """Apply bounded wrist-seeded appearance refinement to a frame sequence."""

    wrists = np.asarray(wrist_pixels, dtype=np.float64)
    palm_widths = np.asarray(palm_width_pixels, dtype=np.float64)
    frame_count = len(frames)
    if len(masks) != frame_count or wrists.shape != (frame_count, 2):
        raise ValueError("frame, mask, and wrist counts differ")
    if palm_widths.shape != (frame_count,):
        raise ValueError("palm_width_pixels must have shape [frames]")
    results: list[np.ndarray] = []
    growth_values = []
    rejected_frames = 0
    for index in range(frame_count):
        try:
            refined, growth, rejected = refine_human_silhouette(
                frames[index],
                masks[index],
                wrists[index],
                float(palm_widths[index]),
                max_area_growth=max_area_growth,
            )
        except ValueError as error:
            raise ValueError(
                f"human-mask refinement failed at frame {index}: {error}"
            ) from error
        results.append(refined)
        growth_values.append(growth)
        rejected_frames += int(rejected)
    growth_array = np.asarray(growth_values, dtype=np.float64)
    summary = HumanMaskRefinementSummary(
        frame_count=frame_count,
        refined_frames=int(np.count_nonzero(growth_array > 1.01)),
        rejected_frames=rejected_frames,
        area_growth_mean=float(np.mean(growth_array)),
        area_growth_p95=float(np.quantile(growth_array, 0.95)),
        area_growth_max=float(np.max(growth_array)),
    )
    return results, summary


def estimate_forearm_silhouette(
    mask: np.ndarray,
    wrist_pixel: np.ndarray,
    *,
    fallback_direction: np.ndarray | None = None,
    fallback_width_pixels: float | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Estimate the proximal sleeve direction, length, width and confidence.

    The selected component must touch the tracked wrist. The direction comes
    from its distant pixels, so the hand blob near the wrist cannot dominate.
    No scene class, plane, skin colour or camera-specific constant is used.
    """

    binary = np.asarray(mask) > 0
    wrist = np.asarray(wrist_pixel, dtype=np.float64)
    if binary.ndim != 2 or wrist.shape != (2,) or not np.isfinite(wrist).all():
        raise ValueError("mask must be 2-D and wrist_pixel must be finite [2]")
    component = _nearest_component(
        binary,
        wrist,
        max_distance_pixels=(
            None
            if fallback_width_pixels is None
            else 3.0 * float(fallback_width_pixels)
        ),
    )
    rows, columns = np.nonzero(component)
    offsets = np.stack([columns - wrist[0], rows - wrist[1]], axis=1)
    radii = np.linalg.norm(offsets, axis=1)
    if len(radii) < 64:
        raise ValueError("wrist component is too small for a forearm estimate")

    far_threshold = max(12.0, float(np.quantile(radii, 0.68)))
    far = offsets[radii >= far_threshold]
    if len(far) < 24:
        raise ValueError("too few distal sleeve pixels")
    normalized_far = far / np.maximum(
        np.linalg.norm(far, axis=1, keepdims=True), 1e-8
    )
    direction = np.median(normalized_far, axis=0)
    if fallback_direction is not None:
        fallback = np.asarray(fallback_direction, dtype=np.float64)
        if fallback.shape == (2,) and np.isfinite(fallback).all():
            fallback_norm = float(np.linalg.norm(fallback))
            if fallback_norm > 1e-8 and np.dot(direction, fallback) < 0:
                direction *= -1.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        raise ValueError("forearm direction is degenerate")
    direction /= norm

    # Element-wise reductions avoid platform BLAS warnings observed for a
    # highly rectangular [pixels, 2] matmul while producing the same values.
    along = np.sum(offsets * direction[None], axis=1)
    perpendicular_direction = np.asarray([-direction[1], direction[0]])
    perpendicular = np.sum(offsets * perpendicular_direction[None], axis=1)
    positive = along > max(6.0, float(np.quantile(along, 0.15)))
    if np.count_nonzero(positive) < 32:
        raise ValueError("forearm component has insufficient proximal support")
    length = float(np.quantile(along[positive], 0.95))
    middle = positive & (along >= 0.25 * length) & (along <= 0.80 * length)
    if np.count_nonzero(middle) >= 32:
        width = float(
            np.quantile(perpendicular[middle], 0.95)
            - np.quantile(perpendicular[middle], 0.05)
        )
    elif fallback_width_pixels is not None:
        width = float(fallback_width_pixels)
    else:
        raise ValueError("forearm width is unobservable")
    if not np.isfinite(length) or not np.isfinite(width) or length <= 8 or width <= 2:
        raise ValueError("forearm dimensions are implausible")

    concentration = float(np.linalg.norm(np.mean(normalized_far, axis=0)))
    support = min(1.0, len(far) / 256.0)
    aspect = min(1.0, max(0.0, length / max(width, 1e-6) / 2.0))
    confidence = float(np.clip(concentration * support * aspect, 0.0, 1.0))
    return direction, length, width, confidence


def _smooth_directions(values: np.ndarray, confidence: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("temporal window must be a positive odd integer")
    radius = window // 2
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        local = values[start:end].copy()
        signs = np.sign(local @ values[index])
        local[signs < 0] *= -1.0
        weights = np.maximum(confidence[start:end], 1e-3)
        direction = np.sum(local * weights[:, None], axis=0)
        norm = float(np.linalg.norm(direction))
        result[index] = values[index] if norm < 1e-8 else direction / norm
    return result


def _running_median(values: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    return np.asarray(
        [
            np.median(values[max(0, index - radius) : index + radius + 1])
            for index in range(len(values))
        ],
        dtype=np.float64,
    )


def estimate_forearm_observation_sequence(
    masks: list[np.ndarray] | tuple[np.ndarray, ...],
    wrist_camera: np.ndarray,
    wrist_pixels: np.ndarray,
    intrinsic: np.ndarray,
    *,
    annotation_guide_camera: np.ndarray | None = None,
    palm_width_pixels: np.ndarray | None = None,
    temporal_window: int = 9,
) -> ForearmObservationSequence:
    """Fuse sleeve silhouettes with annotation only as a signed fallback."""

    wrists_camera = np.asarray(wrist_camera, dtype=np.float64)
    wrists_pixels = np.asarray(wrist_pixels, dtype=np.float64)
    intrinsic_values = np.asarray(intrinsic, dtype=np.float64)
    frame_count = len(wrists_camera)
    if len(masks) != frame_count or wrists_camera.shape != (frame_count, 3):
        raise ValueError("mask and wrist-camera frame counts differ")
    if wrists_pixels.shape != (frame_count, 2) or intrinsic_values.shape != (3, 3):
        raise ValueError("invalid wrist pixels or intrinsic")

    annotation_directions: np.ndarray | None = None
    if annotation_guide_camera is not None:
        guides = np.asarray(annotation_guide_camera, dtype=np.float64)
        if guides.shape != (frame_count, 3):
            raise ValueError("annotation guide must have shape [frames, 3]")
        delta = guides - wrists_camera
        endpoint_pixels = np.empty((frame_count, 2), dtype=np.float64)
        safe_depth = np.where(np.abs(guides[:, 2]) > 1e-6, guides[:, 2], np.nan)
        endpoint_pixels[:, 0] = (
            intrinsic_values[0, 0] * guides[:, 0] / safe_depth
            + intrinsic_values[0, 2]
        )
        endpoint_pixels[:, 1] = (
            intrinsic_values[1, 1] * guides[:, 1] / safe_depth
            + intrinsic_values[1, 2]
        )
        annotation_directions = endpoint_pixels - wrists_pixels
        annotation_directions /= np.maximum(
            np.linalg.norm(annotation_directions, axis=1, keepdims=True), 1e-8
        )

    if palm_width_pixels is not None:
        fallback_width = np.asarray(palm_width_pixels, dtype=np.float64)
        if fallback_width.shape != (frame_count,):
            raise ValueError("palm_width_pixels must have shape [frames]")
    else:
        fallback_width = np.full(frame_count, np.nan, dtype=np.float64)

    directions = []
    lengths = []
    widths = []
    confidences = []
    for frame_index, mask in enumerate(masks):
        fallback_direction = (
            None
            if annotation_directions is None
            else annotation_directions[frame_index]
        )
        try:
            direction, length, width, confidence = estimate_forearm_silhouette(
                mask,
                wrists_pixels[frame_index],
                fallback_direction=fallback_direction,
                fallback_width_pixels=(
                    None
                    if not np.isfinite(fallback_width[frame_index])
                    else float(fallback_width[frame_index])
                ),
            )
        except ValueError:
            if fallback_direction is None or not np.isfinite(fallback_direction).all():
                raise
            direction = fallback_direction
            length = float(max(24.0, 2.5 * fallback_width[frame_index]))
            width = float(fallback_width[frame_index])
            confidence = 0.05
        directions.append(direction)
        lengths.append(length)
        widths.append(width)
        confidences.append(confidence)

    raw_directions = np.asarray(directions, dtype=np.float64)
    confidence_values = np.asarray(confidences, dtype=np.float64)
    smoothed = _smooth_directions(raw_directions, confidence_values, temporal_window)
    length_pixels = _running_median(
        np.asarray(lengths, dtype=np.float64), temporal_window
    )
    width_pixels = _running_median(
        np.asarray(widths, dtype=np.float64), temporal_window
    )

    endpoint_pixels = wrists_pixels + smoothed * length_pixels[:, None]
    depth = wrists_camera[:, 2]
    guide = np.empty_like(wrists_camera)
    guide[:, 0] = (
        endpoint_pixels[:, 0] - intrinsic_values[0, 2]
    ) * depth / intrinsic_values[0, 0]
    guide[:, 1] = (
        endpoint_pixels[:, 1] - intrinsic_values[1, 2]
    ) * depth / intrinsic_values[1, 1]
    guide[:, 2] = depth

    delta_pixels = smoothed * length_pixels[:, None]
    length_camera = depth * np.sqrt(
        np.square(delta_pixels[:, 0] / intrinsic_values[0, 0])
        + np.square(delta_pixels[:, 1] / intrinsic_values[1, 1])
    )
    perpendicular = np.stack([-smoothed[:, 1], smoothed[:, 0]], axis=1)
    half_width_pixels = 0.5 * width_pixels[:, None] * perpendicular
    half_width_camera = depth * np.sqrt(
        np.square(half_width_pixels[:, 0] / intrinsic_values[0, 0])
        + np.square(half_width_pixels[:, 1] / intrinsic_values[1, 1])
    )
    return ForearmObservationSequence(
        direction_pixels=smoothed.astype(np.float32),
        guide_camera=guide.astype(np.float32),
        length_pixels=length_pixels.astype(np.float32),
        width_pixels=width_pixels.astype(np.float32),
        length_camera=length_camera.astype(np.float32),
        width_camera=(2.0 * half_width_camera).astype(np.float32),
        confidence=confidence_values.astype(np.float32),
    )
