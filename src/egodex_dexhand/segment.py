from __future__ import annotations

import contextlib
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from .data import project_camera_points


SAM2_CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

HAND_CONNECTIONS = tuple(
    (chain[index], chain[index + 1])
    for chain in (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    for index in range(len(chain) - 1)
)


def tracked_hand_support_mask(
    joints_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    joint_confidence: np.ndarray | None = None,
    radius: int | None = None,
) -> np.ndarray:
    """Rasterize a conservative support around the measured 21-joint hand.

    SAM can occasionally select a sleeve while omitting a thin pointing finger.
    The tracked skeleton is stronger evidence that those pixels belong to the
    actor, so this support is unioned into the removal mask after stabilization.
    """

    joints = np.asarray(joints_camera_cv, dtype=np.float32)
    if joints.shape != (21, 3):
        raise ValueError(f"expected 21 hand joints, got {joints.shape}")
    pixels = project_camera_points(joints, intrinsic)
    valid = np.isfinite(pixels).all(axis=1) & (joints[:, 2] > 1e-4)
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    if joint_confidence is not None:
        trusted = valid & (np.asarray(joint_confidence) >= 0.3)
        if np.count_nonzero(trusted) >= 4:
            valid = trusted
    support = np.zeros((height, width), dtype=np.uint8)
    if radius is None:
        radius = max(4, int(round(min(width, height) * 0.026)))
    thickness = 2 * int(radius)
    rounded = np.rint(pixels).astype(np.int32)
    for first, second in HAND_CONNECTIONS:
        if valid[first] and valid[second]:
            cv2.line(
                support,
                tuple(rounded[first]),
                tuple(rounded[second]),
                1,
                thickness,
                cv2.LINE_AA,
            )
    for point in rounded[valid]:
        cv2.circle(support, tuple(point), int(radius), 1, -1, cv2.LINE_AA)
    return support.astype(bool)


def arm_hand_geometry_envelope(
    hand_joints_camera_cv: np.ndarray,
    arm_joints_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    *,
    corridor_radius: int | None = None,
) -> np.ndarray:
    """Return a conservative per-frame support envelope for a hand and sleeve.

    This is the arm analogue of PHANTOM's per-frame hand-box clamp. A broad,
    oriented capsule follows the measured wrist-to-elbow ray to the image
    boundary, while the projected hand skeleton preserves fingers outside that
    capsule. Intersecting a SAM mask with this envelope prevents an ambiguous
    border-connected sleeve prompt from absorbing a table or work surface.
    """

    hand = np.asarray(hand_joints_camera_cv, dtype=np.float32)
    arm = np.asarray(arm_joints_camera_cv, dtype=np.float32)
    if hand.shape != (21, 3):
        raise ValueError(f"expected 21 hand joints, got {hand.shape}")
    if arm.shape != (4, 3):
        raise ValueError(f"expected 4 arm joints, got {arm.shape}")

    hand_pixels = project_camera_points(hand, intrinsic)
    arm_pixels = project_camera_points(arm, intrinsic)

    def in_frame(point: np.ndarray) -> bool:
        return bool(
            np.isfinite(point).all()
            and 0 <= point[0] <= width - 1
            and 0 <= point[1] <= height - 1
        )

    wrist = arm_pixels[3]
    elbow = arm_pixels[2]
    if in_frame(wrist):
        base = wrist.astype(np.float32)
    else:
        knuckles = hand_pixels[[5, 9, 13, 17]]
        visible_knuckles = np.asarray([in_frame(point) for point in knuckles])
        if np.any(visible_knuckles):
            base = np.mean(knuckles[visible_knuckles], axis=0).astype(np.float32)
        else:
            visible_hand = np.asarray([in_frame(point) for point in hand_pixels])
            if not np.any(visible_hand):
                raise RuntimeError("no visible hand anchor for geometry envelope")
            base = np.mean(hand_pixels[visible_hand], axis=0).astype(np.float32)

    direction = elbow - wrist
    endpoint = _ray_to_image_border(base, direction, width, height)
    if corridor_radius is None:
        visible_hand = np.asarray([in_frame(point) for point in hand_pixels])
        if np.count_nonzero(visible_hand) >= 2:
            extent = hand_pixels[visible_hand]
            palm_scale = float(np.linalg.norm(extent.max(axis=0) - extent.min(axis=0)))
        else:
            palm_scale = 0.0
        corridor_radius = int(
            round(
                np.clip(
                    max(min(width, height) * 0.13, palm_scale * 0.55),
                    58.0,
                    92.0,
                )
            )
        )

    envelope = np.zeros((height, width), dtype=np.uint8)
    start = tuple(np.rint(base).astype(np.int32))
    end = tuple(np.rint(endpoint).astype(np.int32))
    cv2.line(
        envelope,
        start,
        end,
        1,
        2 * int(corridor_radius),
        cv2.LINE_AA,
    )
    cv2.circle(envelope, start, int(corridor_radius), 1, -1, cv2.LINE_AA)
    cv2.circle(envelope, end, int(corridor_radius), 1, -1, cv2.LINE_AA)

    # Keep a generously dilated hand skeleton so fingertips and an open palm
    # are never lost merely because they lie perpendicular to the forearm ray.
    hand_support = tracked_hand_support_mask(
        hand,
        intrinsic,
        width,
        height,
        radius=max(8, int(round(min(width, height) * 0.04))),
    )
    return (envelope > 0) | hand_support


def _arm_axis_for_frame(
    hand_joints_camera_cv: np.ndarray,
    arm_joints_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the visible wrist-to-image-border arm axis for one frame."""

    hand_pixels = project_camera_points(hand_joints_camera_cv, intrinsic)
    arm_pixels = project_camera_points(arm_joints_camera_cv, intrinsic)

    def in_frame(point: np.ndarray) -> bool:
        return bool(
            np.isfinite(point).all()
            and 0 <= point[0] <= width - 1
            and 0 <= point[1] <= height - 1
        )

    wrist = arm_pixels[3]
    elbow = arm_pixels[2]
    if in_frame(wrist):
        base = wrist.astype(np.float32)
    else:
        knuckles = hand_pixels[[5, 9, 13, 17]]
        valid_knuckles = np.asarray([in_frame(point) for point in knuckles])
        if np.any(valid_knuckles):
            base = np.mean(knuckles[valid_knuckles], axis=0).astype(np.float32)
        else:
            valid_hand = np.asarray([in_frame(point) for point in hand_pixels])
            if not np.any(valid_hand):
                raise RuntimeError("no visible hand anchor for arm axis")
            base = np.mean(hand_pixels[valid_hand], axis=0).astype(np.float32)
    endpoint = _ray_to_image_border(base, elbow - wrist, width, height)
    return base, endpoint


def adaptive_arm_hand_envelopes(
    raw_masks: list[np.ndarray] | tuple[np.ndarray, ...],
    hand_joints_camera_cv: np.ndarray,
    arm_joints_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    *,
    profile_bins: int = 12,
    profile_margin: float = 12.0,
) -> tuple[list[np.ndarray], int, np.ndarray]:
    """Measure sleeve width once and transfer it through projected arm motion.

    The reference is the frame whose projected hand center is farthest from an
    image boundary, following PHANTOM's seed selection. Cross-sections of that
    frame's SAM mask provide an image-derived sleeve radius profile. The profile
    is then rendered along every frame's measured arm axis and mildly rescaled
    by projected hand size. Hand keypoints are protected separately.
    """

    masks = [np.asarray(mask, dtype=bool) for mask in raw_masks]
    count = len(masks)
    if count == 0:
        raise ValueError("at least one raw mask is required")
    if len(hand_joints_camera_cv) != count or len(arm_joints_camera_cv) != count:
        raise ValueError("mask and joint sequence lengths differ")
    if profile_bins < 4:
        raise ValueError("profile_bins must be at least four")

    axes: list[tuple[np.ndarray, np.ndarray]] = []
    edge_margins: list[float] = []
    hand_scales: list[float] = []
    for frame_index in range(count):
        base, endpoint = _arm_axis_for_frame(
            hand_joints_camera_cv[frame_index],
            arm_joints_camera_cv[frame_index],
            intrinsic,
            width,
            height,
        )
        axes.append((base, endpoint))
        pixels = project_camera_points(hand_joints_camera_cv[frame_index], intrinsic)
        valid = np.isfinite(pixels).all(axis=1)
        valid &= hand_joints_camera_cv[frame_index, :, 2] > 1e-4
        valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        if np.any(valid):
            extent = pixels[valid]
            center = np.mean(extent, axis=0)
            edge_margins.append(
                float(
                    min(
                        center[0],
                        width - 1 - center[0],
                        center[1],
                        height - 1 - center[1],
                    )
                )
            )
            hand_scales.append(
                max(1.0, float(np.linalg.norm(extent.max(axis=0) - extent.min(axis=0))))
            )
        else:
            edge_margins.append(-1.0)
            hand_scales.append(1.0)

    # PHANTOM uses only maximum distance from the image edge. That can still
    # pick an already oversegmented frame when the hand is central. Retain all
    # frames within 85% of the best visibility, then choose the most compact
    # nonempty SAM mask as the reference. A leaked table/board mask has a much
    # larger area than a correctly isolated sleeve at comparable visibility.
    maximum_margin = float(np.max(edge_margins))
    reference_candidates = [
        index
        for index, margin in enumerate(edge_margins)
        if margin >= 0.85 * maximum_margin
        and 0.002 <= float(np.mean(masks[index])) <= 0.20
    ]
    if not reference_candidates:
        reference_index = int(np.argmax(edge_margins))
    else:
        reference_index = min(
            reference_candidates, key=lambda index: float(np.mean(masks[index]))
        )
    base, endpoint = axes[reference_index]
    axis = endpoint - base
    axis_length = float(np.linalg.norm(axis))
    if axis_length < 1e-5:
        raise RuntimeError("reference arm axis is degenerate")
    unit = axis / axis_length
    perpendicular = np.asarray([-unit[1], unit[0]], dtype=np.float32)
    ys, xs = np.nonzero(masks[reference_index])
    coordinates = np.stack([xs, ys], axis=1).astype(np.float32)
    relative = coordinates - base[None]
    longitudinal = relative @ unit / axis_length
    lateral = np.abs(relative @ perpendicular)

    radii = np.full(profile_bins, np.nan, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, profile_bins + 1)
    for bin_index in range(profile_bins):
        selected = (longitudinal >= edges[bin_index]) & (
            longitudinal < edges[bin_index + 1]
        )
        if np.count_nonzero(selected) >= 24:
            radii[bin_index] = float(np.percentile(lateral[selected], 92.0))
    valid_bins = np.flatnonzero(np.isfinite(radii))
    # A clean seed may show only the distal sleeve; requiring a mask all the
    # way to the image border forces selection of the very leaked masks we are
    # trying to reject. Three cross-sections are enough to estimate width, and
    # np.interp deliberately extrapolates the nearest clean radius.
    if len(valid_bins) < max(3, profile_bins // 4):
        raise RuntimeError("reference SAM mask does not contain a measurable sleeve")
    radii = np.interp(
        np.arange(profile_bins), valid_bins, radii[valid_bins]
    ).astype(np.float32)
    radii = np.clip(radii + float(profile_margin), 28.0, 88.0)
    radii = np.convolve(radii, np.asarray([0.25, 0.5, 0.25]), mode="same")
    radii[0] = 0.75 * radii[0] + 0.25 * radii[1]
    radii[-1] = 0.75 * radii[-1] + 0.25 * radii[-2]

    reference_scale = hand_scales[reference_index]
    envelopes: list[np.ndarray] = []
    for frame_index, (frame_base, frame_endpoint) in enumerate(axes):
        frame_axis = frame_endpoint - frame_base
        scale = float(np.clip(hand_scales[frame_index] / reference_scale, 0.78, 1.28))
        envelope = np.zeros((height, width), dtype=np.uint8)
        for fraction in np.linspace(0.0, 1.0, profile_bins * 4):
            profile_position = fraction * (profile_bins - 1)
            lower = int(np.floor(profile_position))
            upper = min(profile_bins - 1, lower + 1)
            blend = profile_position - lower
            radius = int(round(((1.0 - blend) * radii[lower] + blend * radii[upper]) * scale))
            center = frame_base + float(fraction) * frame_axis
            cv2.circle(
                envelope,
                tuple(np.rint(center).astype(np.int32)),
                max(1, radius),
                1,
                -1,
                cv2.LINE_AA,
            )
        hand_support = tracked_hand_support_mask(
            hand_joints_camera_cv[frame_index],
            intrinsic,
            width,
            height,
            radius=max(8, int(round(min(width, height) * 0.04))),
        )
        envelopes.append((envelope > 0) | hand_support)
    return envelopes, reference_index, radii


def appearance_refine_arm_hand_masks(
    raw_masks: list[np.ndarray] | tuple[np.ndarray, ...],
    frames: list[np.ndarray] | tuple[np.ndarray, ...],
    hand_joints_camera_cv: np.ndarray,
    arm_joints_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    *,
    appearance_threshold: float = 3.0,
    update_rate: float = 0.12,
) -> tuple[list[np.ndarray], int, np.ndarray, np.ndarray]:
    """Suppress SAM leakage using a temporally tracked sleeve appearance model.

    A compact, high-visibility reference frame supplies robust LAB statistics
    for sleeve pixels. The model is tracked independently forward and backward
    in time, but it can only accept pixels inside the conservative geometric
    arm envelope and the raw SAM mask. The hand is recovered separately with a
    keypoint-skeleton support, so skin color need not resemble the sleeve.
    """

    masks = [np.asarray(mask, dtype=bool) for mask in raw_masks]
    bgr_frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    count = len(masks)
    if count == 0 or len(bgr_frames) != count:
        raise ValueError("raw mask and frame sequences must be nonempty and aligned")
    if not 0 < appearance_threshold:
        raise ValueError("appearance_threshold must be positive")
    if not 0 <= update_rate <= 1:
        raise ValueError("update_rate must be in [0, 1]")

    # Reuse the compact-reference selection validated by the width experiment.
    _, reference_index, _ = adaptive_arm_hand_envelopes(
        masks,
        hand_joints_camera_cv,
        arm_joints_camera_cv,
        intrinsic,
        width,
        height,
    )
    broad_envelopes = [
        arm_hand_geometry_envelope(
            hand_joints_camera_cv[index],
            arm_joints_camera_cv[index],
            intrinsic,
            width,
            height,
        )
        for index in range(count)
    ]
    lab_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) for frame in bgr_frames]

    def hand_region(frame_index: int) -> np.ndarray:
        return tracked_hand_support_mask(
            hand_joints_camera_cv[frame_index],
            intrinsic,
            width,
            height,
            radius=max(8, int(round(min(width, height) * 0.04))),
        )

    reference_base, reference_endpoint = _arm_axis_for_frame(
        hand_joints_camera_cv[reference_index],
        arm_joints_camera_cv[reference_index],
        intrinsic,
        width,
        height,
    )
    reference_axis = reference_endpoint - reference_base
    reference_length = float(np.linalg.norm(reference_axis))
    reference_unit = reference_axis / max(reference_length, 1e-5)
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    relative_x = grid_x - reference_base[0]
    relative_y = grid_y - reference_base[1]
    longitudinal = (
        relative_x * reference_unit[0] + relative_y * reference_unit[1]
    ) / max(reference_length, 1e-5)
    reference_hand_region = hand_region(reference_index)
    reference_sleeve = masks[reference_index] & broad_envelopes[reference_index]
    reference_sleeve &= ~reference_hand_region
    # Prefer the clean distal sleeve seen in the compact reference. If too few
    # pixels survive, relax the longitudinal gate but never include the hand.
    distal = reference_sleeve & (longitudinal >= 0.02) & (longitudinal <= 0.55)
    if np.count_nonzero(distal) >= 256:
        reference_sleeve = distal
    if np.count_nonzero(reference_sleeve) < 128:
        raise RuntimeError("too few reference sleeve pixels for appearance tracking")
    reference_colors = lab_frames[reference_index][reference_sleeve].astype(np.float32)
    center = np.median(reference_colors, axis=0)
    mad = np.median(np.abs(reference_colors - center[None]), axis=0)
    scale = np.maximum(1.4826 * mad, np.asarray([10.0, 7.0, 7.0], np.float32))

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined: list[np.ndarray | None] = [None] * count

    def process_frame(
        frame_index: int, current_center: np.ndarray, current_scale: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        difference = (
            lab_frames[frame_index].astype(np.float32) - current_center[None, None]
        ) / current_scale[None, None]
        appearance = np.sqrt(np.mean(difference * difference, axis=2))
        sleeve_candidate = masks[frame_index] & broad_envelopes[frame_index]
        sleeve_candidate &= appearance <= appearance_threshold
        sleeve_candidate = cv2.morphologyEx(
            sleeve_candidate.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel
        )
        sleeve_candidate = cv2.morphologyEx(
            sleeve_candidate, cv2.MORPH_OPEN, open_kernel
        ) > 0

        # Keep appearance-consistent components connected to an image boundary
        # or to the wrist neighborhood. Detached board islands are discarded.
        base, _ = _arm_axis_for_frame(
            hand_joints_camera_cv[frame_index],
            arm_joints_camera_cv[frame_index],
            intrinsic,
            width,
            height,
        )
        count_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            sleeve_candidate.astype(np.uint8), connectivity=8
        )
        sleeve = np.zeros((height, width), dtype=bool)
        bx = int(np.clip(round(float(base[0])), 0, width - 1))
        by = int(np.clip(round(float(base[1])), 0, height - 1))
        wrist_radius = max(16, int(round(min(width, height) * 0.055)))
        for component_id in range(1, count_components):
            if int(stats[component_id, cv2.CC_STAT_AREA]) < 96:
                continue
            component = labels == component_id
            touches_border = bool(
                component[:5].any()
                or component[-5:].any()
                or component[:, :5].any()
                or component[:, -5:].any()
            )
            y0, y1 = max(0, by - wrist_radius), min(height, by + wrist_radius + 1)
            x0, x1 = max(0, bx - wrist_radius), min(width, bx + wrist_radius + 1)
            touches_wrist = bool(component[y0:y1, x0:x1].any())
            if touches_border or touches_wrist:
                sleeve |= component

        support = hand_region(frame_index)
        hand = support
        result = sleeve | hand
        if np.count_nonzero(sleeve) >= 256:
            colors = lab_frames[frame_index][sleeve].astype(np.float32)
            observed_center = np.median(colors, axis=0)
            observed_mad = np.median(
                np.abs(colors - observed_center[None]), axis=0
            )
            observed_scale = np.maximum(
                1.4826 * observed_mad,
                np.asarray([10.0, 7.0, 7.0], np.float32),
            )
            current_center = (
                (1.0 - update_rate) * current_center + update_rate * observed_center
            )
            current_scale = (
                (1.0 - update_rate) * current_scale + update_rate * observed_scale
            )
        return result, current_center, current_scale

    for indices in (
        range(reference_index, count),
        range(reference_index - 1, -1, -1),
    ):
        tracked_center = center.copy()
        tracked_scale = scale.copy()
        for frame_index in indices:
            result, tracked_center, tracked_scale = process_frame(
                frame_index, tracked_center, tracked_scale
            )
            refined[frame_index] = result
    if any(mask is None for mask in refined):
        raise RuntimeError("appearance refinement did not process every frame")
    return [np.asarray(mask, dtype=bool) for mask in refined], reference_index, center, scale


def _prompt_for_frame(
    joints_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    padding: int,
    joint_confidence: np.ndarray | None = None,
    negative_joints_camera_cv: np.ndarray | None = None,
    negative_joint_confidence: np.ndarray | None = None,
    confidence_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = project_camera_points(joints_camera_cv, intrinsic)
    geometric_valid = np.isfinite(pixels).all(axis=1)
    geometric_valid &= joints_camera_cv[:, 2] > 1e-4
    geometric_valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    geometric_valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    valid = geometric_valid.copy()
    if joint_confidence is not None:
        valid &= np.asarray(joint_confidence) >= confidence_threshold
        # Some otherwise usable EgoDex episodes assign every hand landmark a
        # confidence just below the nominal 0.3 cutoff.  Rejecting the whole
        # frame makes batch processing impossible even when the projected
        # skeleton is finite and visibly inside the image.  Prefer trusted
        # landmarks, but fall back to geometric validity when fewer than four
        # survive; SAM2 still has the video appearance to refine the prompt.
        if np.count_nonzero(valid) < 4:
            valid = geometric_valid.copy()
    # The EgoQuest visibility scanner deliberately retains a hand whose
    # skeleton is at most 20 pixels outside the crop: those are real partial
    # hands entering from an image edge.  SAM accepts points on the boundary,
    # so when strict projection leaves too few prompts, clamp that same narrow
    # tolerance band instead of rejecting the whole chunk.
    if np.count_nonzero(valid) < 4:
        near_valid = np.isfinite(pixels).all(axis=1)
        near_valid &= joints_camera_cv[:, 2] > 1e-4
        near_valid &= (pixels[:, 0] >= -20) & (pixels[:, 0] < width + 20)
        near_valid &= (pixels[:, 1] >= -20) & (pixels[:, 1] < height + 20)
        if joint_confidence is not None:
            trusted_near = near_valid & (
                np.asarray(joint_confidence) >= confidence_threshold
            )
            if np.count_nonzero(trusted_near) >= 4:
                near_valid = trusted_near
        if np.count_nonzero(near_valid) >= 4:
            valid = near_valid
            pixels = pixels.copy()
            pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    positive = pixels[valid]
    if len(positive) < 4:
        raise RuntimeError("too few projected hand landmarks for a SAM2 prompt")

    x0 = max(0.0, float(positive[:, 0].min() - padding))
    y0 = max(0.0, float(positive[:, 1].min() - padding))
    x1 = min(float(width - 1), float(positive[:, 0].max() + padding))
    y1 = min(float(height - 1), float(positive[:, 1].max() + padding))
    box = np.asarray([x0, y0, x1, y1], dtype=np.float32)

    # Points on all 21 bones are redundant. A spread of landmarks is more stable
    # and gives SAM2 both the palm and fingertips without overweighting a finger.
    sample_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    selected_indices = [index for index in sample_indices if valid[index]]
    # Confidence filtering can invalidate every preferred point even when other
    # landmarks are valid. Fill from the actual valid set rather than passing an
    # empty positive prompt to SAM2.
    for index in np.flatnonzero(valid):
        if len(selected_indices) >= 4:
            break
        if int(index) not in selected_indices:
            selected_indices.append(int(index))
    if not selected_indices:
        raise RuntimeError("no valid positive SAM2 prompt points")
    selected = [pixels[index] for index in selected_indices]
    points = np.asarray(selected, dtype=np.float32)
    labels = np.ones(len(points), dtype=np.int32)

    # In bimanual clips, points on the non-target hand are valuable negatives,
    # especially while both hands touch the same object. This reduces SAM2's
    # tendency to merge the two hands or absorb the manipulated object.
    if negative_joints_camera_cv is not None:
        negative_pixels = project_camera_points(negative_joints_camera_cv, intrinsic)
        negative_valid = np.isfinite(negative_pixels).all(axis=1)
        negative_valid &= negative_joints_camera_cv[:, 2] > 1e-4
        negative_valid &= (negative_pixels[:, 0] >= 0) & (negative_pixels[:, 0] < width)
        negative_valid &= (negative_pixels[:, 1] >= 0) & (negative_pixels[:, 1] < height)
        if negative_joint_confidence is not None:
            negative_valid &= (
                np.asarray(negative_joint_confidence) >= confidence_threshold
            )
        negative_indices = [0, 2, 6, 10, 14, 18]
        negative = [
            negative_pixels[index]
            for index in negative_indices
            if negative_valid[index]
            and np.min(np.linalg.norm(points - negative_pixels[index], axis=1)) > 10.0
        ]
        if negative:
            points = np.concatenate(
                [points, np.asarray(negative, dtype=np.float32)], axis=0
            )
            labels = np.concatenate(
                [labels, np.zeros(len(negative), dtype=np.int32)], axis=0
            )
    return points, labels, box


def segment_hand_video(
    frames_dir: str | Path,
    joints_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    sam2_root: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    model_size: str = "small",
    prompt_stride: int = 10,
    box_padding: int = 18,
    joint_confidence: np.ndarray | None = None,
    negative_joints_camera_cv: np.ndarray | None = None,
    negative_joint_confidence: np.ndarray | None = None,
    use_box_prompt: bool = False,
) -> None:
    """Segment a hand with real SAM2, automatically prompted by EgoDex joints."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 stage requires CUDA; refusing to use a fallback")
    if model_size not in SAM2_CONFIGS:
        raise ValueError(f"model_size must be one of {sorted(SAM2_CONFIGS)}")

    frames_dir = Path(frames_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if len(frame_paths) != len(joints_camera_cv):
        raise ValueError(
            f"frame/annotation mismatch: {len(frame_paths)} vs {len(joints_camera_cv)}"
        )
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"could not decode {frame_paths[0]}")
    height, width = first.shape[:2]

    sam2_root = Path(sam2_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(sam2_root))
    try:
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            SAM2_CONFIGS[model_size], str(checkpoint), device="cuda"
        )
        inference_state = predictor.init_state(video_path=str(frames_dir))
        predictor.reset_state(inference_state)

        stride = max(1, int(prompt_stride))
        prompt_frames = list(range(0, len(frame_paths), stride))
        if prompt_frames[-1] != len(frame_paths) - 1:
            prompt_frames.append(len(frame_paths) - 1)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for frame_index in prompt_frames:
                points, labels, box = _prompt_for_frame(
                    joints_camera_cv[frame_index],
                    intrinsic,
                    width,
                    height,
                    box_padding,
                    None
                    if joint_confidence is None
                    else joint_confidence[frame_index],
                    None
                    if negative_joints_camera_cv is None
                    else negative_joints_camera_cv[frame_index],
                    None
                    if negative_joint_confidence is None
                    else negative_joint_confidence[frame_index],
                )
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_index,
                    obj_id=1,
                    points=points,
                    labels=labels,
                    box=box if use_box_prompt else None,
                )

            masks: dict[int, np.ndarray] = {}
            for frame_index, object_ids, logits in predictor.propagate_in_video(
                inference_state
            ):
                if 1 not in object_ids:
                    raise RuntimeError(f"SAM2 lost object 1 at frame {frame_index}")
                object_index = list(object_ids).index(1)
                masks[int(frame_index)] = (
                    logits[object_index, 0] > 0.0
                ).cpu().numpy()
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(sam2_root))

    missing = sorted(set(range(len(frame_paths))) - set(masks))
    if missing:
        raise RuntimeError(f"SAM2 did not return frames: {missing}")
    for frame_index in range(len(frame_paths)):
        mask = masks[frame_index].astype(np.uint8) * 255
        if np.count_nonzero(mask) < 64:
            raise RuntimeError(f"SAM2 mask too small at frame {frame_index}")
        cv2.imwrite(str(output_dir / f"{frame_index:05d}.png"), mask)


def _ray_to_image_border(
    base: np.ndarray, direction: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Return the first image-border intersection along a 2-D ray."""

    base = np.asarray(base, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if not np.isfinite(base).all() or not np.isfinite(direction).all():
        raise RuntimeError("non-finite arm prompt ray")
    if np.linalg.norm(direction) < 1e-5:
        raise RuntimeError("degenerate arm prompt ray")
    candidates: list[float] = []
    for axis, boundary in ((0, 0.0), (0, width - 1.0), (1, 0.0), (1, height - 1.0)):
        if abs(direction[axis]) < 1e-8:
            continue
        scale = (boundary - base[axis]) / direction[axis]
        if scale <= 0:
            continue
        point = base + scale * direction
        if -1e-3 <= point[0] <= width - 1 + 1e-3 and -1e-3 <= point[1] <= height - 1 + 1e-3:
            candidates.append(float(scale))
    if not candidates:
        raise RuntimeError("arm prompt ray does not intersect the image")
    endpoint = base + min(candidates) * direction
    endpoint[0] = np.clip(endpoint[0], 0, width - 1)
    endpoint[1] = np.clip(endpoint[1], 0, height - 1)
    return endpoint.astype(np.float32)


def _arm_prompt_for_frame(
    hand_joints_camera_cv: np.ndarray,
    arm_joints_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    other_hand_negative_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a sleeve prompt from the tracked wrist-to-elbow ray."""

    hand_pixels = project_camera_points(hand_joints_camera_cv, intrinsic)
    arm_pixels = project_camera_points(arm_joints_camera_cv, intrinsic)
    wrist_projected = arm_pixels[3]
    elbow_projected = arm_pixels[2]

    def in_frame(point: np.ndarray) -> bool:
        # Image coordinates denote pixel centers.  Values between the final
        # center (width/height - 1) and the outer image extent are not valid
        # ray anchors: an outward wrist-to-elbow direction would already have
        # crossed the border and therefore have no positive intersection.
        return bool(
            np.isfinite(point).all()
            and 0 <= point[0] <= width - 1
            and 0 <= point[1] <= height - 1
        )

    if in_frame(wrist_projected):
        base = wrist_projected.astype(np.float32)
    else:
        knuckles = hand_pixels[[5, 9, 13, 17]]
        valid_knuckles = np.asarray([in_frame(point) for point in knuckles])
        if np.any(valid_knuckles):
            base = np.mean(knuckles[valid_knuckles], axis=0).astype(np.float32)
        else:
            valid_hand = np.asarray([in_frame(point) for point in hand_pixels])
            if not np.any(valid_hand):
                raise RuntimeError("no visible palm anchor for arm SAM2 prompt")
            base = np.mean(hand_pixels[valid_hand], axis=0).astype(np.float32)

    direction = elbow_projected - wrist_projected
    endpoint = _ray_to_image_border(base, direction, width, height)
    fractions = np.asarray([0.04, 0.26, 0.50, 0.74, 0.94], dtype=np.float32)
    axis_points = base[None] + fractions[:, None] * (endpoint - base)[None]

    negatives: list[np.ndarray] = [
        np.asarray(point, dtype=np.float32) for point in other_hand_negative_points
    ]
    axis = endpoint - base
    axis_length = float(np.linalg.norm(axis))
    if axis_length > 1e-5:
        perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float32) / axis_length
        for fraction in (0.28, 0.58, 0.86):
            center = base + fraction * axis
            radius = 78.0 + 38.0 * fraction
            for sign in (-1.0, 1.0):
                point = center + sign * radius * perpendicular
                if in_frame(point):
                    negatives.append(point.astype(np.float32))

    points = axis_points.astype(np.float32)
    labels = np.ones(len(points), dtype=np.int32)
    if negatives:
        points = np.concatenate([points, np.stack(negatives)], axis=0)
        labels = np.concatenate(
            [labels, np.zeros(len(negatives), dtype=np.int32)], axis=0
        )

    box_points = np.concatenate([base[None], axis_points], axis=0)
    padding = 82.0
    box = np.asarray(
        [
            max(0.0, float(box_points[:, 0].min() - padding)),
            max(0.0, float(box_points[:, 1].min() - padding)),
            min(float(width - 1), float(box_points[:, 0].max() + padding)),
            min(float(height - 1), float(box_points[:, 1].max() + padding)),
        ],
        dtype=np.float32,
    )
    return points, labels, box


def _clean_arm_hand_mask(mask: np.ndarray, positive_seeds: np.ndarray) -> np.ndarray:
    """Remove sparse SAM2 streaks while retaining the seeded limb component."""

    binary = np.asarray(mask, dtype=np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    keep = np.zeros_like(cleaned, dtype=bool)
    height, width = cleaned.shape
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < 128:
            continue
        component = component_labels == component_id
        border_connected = bool(component[-3:, :].any() or component[:, -3:].any())
        seed_connected = False
        for seed in positive_seeds:
            if not np.isfinite(seed).all():
                continue
            x = int(np.clip(round(float(seed[0])), 0, width - 1))
            y = int(np.clip(round(float(seed[1])), 0, height - 1))
            x0, x1 = max(0, x - 5), min(width, x + 6)
            y0, y1 = max(0, y - 5), min(height, y + 6)
            if component[y0:y1, x0:x1].any():
                seed_connected = True
                break
        if border_connected or seed_connected:
            keep |= component
    return keep


def _signed_mask_distance(mask: np.ndarray, clip_distance: float) -> np.ndarray:
    """Return a clipped signed distance field, positive inside ``mask``."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"binary masks must be 2-D, got {binary.shape}")
    if not binary.any() or binary.all():
        raise ValueError("binary masks must contain foreground and background")
    if clip_distance <= 0:
        raise ValueError("clip_distance must be positive")
    inside = cv2.distanceTransform(
        binary.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    outside = cv2.distanceTransform(
        (~binary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return np.clip(inside - outside, -clip_distance, clip_distance)


def _farneback_flow(
    source_gray: np.ndarray, target_gray: np.ndarray, scale: float
) -> np.ndarray:
    """Estimate dense source-to-target flow, optionally at reduced resolution."""

    source_gray = np.asarray(source_gray)
    target_gray = np.asarray(target_gray)
    if source_gray.shape != target_gray.shape or source_gray.ndim != 2:
        raise ValueError("optical-flow frames must be aligned 2-D grayscale images")
    if not 0 < scale <= 1:
        raise ValueError("flow scale must be in (0, 1]")
    height, width = source_gray.shape
    if scale < 1:
        flow_size = (
            max(16, int(round(width * scale))),
            max(16, int(round(height * scale))),
        )
        source_work = cv2.resize(source_gray, flow_size, interpolation=cv2.INTER_AREA)
        target_work = cv2.resize(target_gray, flow_size, interpolation=cv2.INTER_AREA)
    else:
        source_work = source_gray
        target_work = target_gray
    flow = cv2.calcOpticalFlowFarneback(
        source_work,
        target_work,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    if scale < 1:
        scale_x = width / float(source_work.shape[1])
        scale_y = height / float(source_work.shape[0])
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= scale_x
        flow[..., 1] *= scale_y
    return flow.astype(np.float32)


def _warp_mask_to_reference(
    source_mask: np.ndarray, reference_to_source_flow: np.ndarray
) -> np.ndarray:
    """Sample ``source_mask`` at coordinates corresponding to a reference frame."""

    source_mask = np.asarray(source_mask, dtype=np.uint8)
    flow = np.asarray(reference_to_source_flow, dtype=np.float32)
    if source_mask.ndim != 2 or flow.shape != (*source_mask.shape, 2):
        raise ValueError("mask and reference-to-source flow shapes do not match")
    height, width = source_mask.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    warped = cv2.remap(
        source_mask,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def stabilize_binary_mask_sequence(
    masks: list[np.ndarray] | tuple[np.ndarray, ...],
    grayscale_frames: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    *,
    neighbor_weight: float = 0.35,
    distance_clip: float = 12.0,
    flow_scale: float = 0.5,
    minimum_area_ratio: float = 0.65,
    maximum_area_ratio: float = 1.35,
) -> list[np.ndarray]:
    """Stabilize binary video masks with flow-aligned signed-distance consensus.

    A current-frame signed distance field is averaged with its warped immediate
    neighbors.  This suppresses one-frame SAM2 spikes and fills transient edge
    holes without temporally averaging RGB or blindly smearing a fast-moving
    hand.  The current mask retains the largest weight, and an area guard falls
    back to it when optical flow is unreliable.

    The defaults were selected for the 960x536 EgoDex episode 1029.  Dense flow
    is estimated at half resolution for a useful speed/quality compromise.
    """

    binary_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    if not binary_masks:
        raise ValueError("at least one mask is required")
    shape = binary_masks[0].shape
    if len(shape) != 2 or any(mask.shape != shape for mask in binary_masks):
        raise ValueError("all masks must be aligned 2-D arrays")
    if any(not mask.any() or mask.all() for mask in binary_masks):
        raise ValueError("each mask must contain foreground and background")
    if not 0 <= neighbor_weight < 1:
        raise ValueError("neighbor_weight must be in [0, 1)")
    if not 0 < minimum_area_ratio <= 1:
        raise ValueError("minimum_area_ratio must be in (0, 1]")
    if maximum_area_ratio < 1:
        raise ValueError("maximum_area_ratio must be at least 1")

    forward_flows: list[np.ndarray] = []
    backward_flows: list[np.ndarray] = []
    if grayscale_frames is not None:
        gray = []
        if len(grayscale_frames) != len(binary_masks):
            raise ValueError("mask and grayscale-frame counts differ")
        for frame in grayscale_frames:
            array = np.asarray(frame)
            if array.ndim == 3:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            if array.shape != shape:
                raise ValueError("mask and grayscale-frame shapes differ")
            gray.append(array.astype(np.uint8, copy=False))
        for index in range(len(gray) - 1):
            forward_flows.append(
                _farneback_flow(gray[index], gray[index + 1], flow_scale)
            )
            backward_flows.append(
                _farneback_flow(gray[index + 1], gray[index], flow_scale)
            )

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stabilized: list[np.ndarray] = []
    for index, current in enumerate(binary_masks):
        field = _signed_mask_distance(current, distance_clip)
        total_weight = 1.0
        if index > 0:
            previous = binary_masks[index - 1]
            if grayscale_frames is not None:
                # backward_flows[index - 1] maps current coordinates to the
                # preceding frame and therefore samples the previous mask.
                previous = _warp_mask_to_reference(
                    previous, backward_flows[index - 1]
                )
            field += neighbor_weight * _signed_mask_distance(
                previous, distance_clip
            )
            total_weight += neighbor_weight
        if index + 1 < len(binary_masks):
            following = binary_masks[index + 1]
            if grayscale_frames is not None:
                # forward_flows[index] maps current coordinates to next-frame
                # coordinates and therefore samples the following mask.
                following = _warp_mask_to_reference(following, forward_flows[index])
            field += neighbor_weight * _signed_mask_distance(
                following, distance_clip
            )
            total_weight += neighbor_weight
        candidate = field / total_weight > 0
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel
        )
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_kernel) > 0
        area_ratio = float(np.count_nonzero(candidate)) / float(
            np.count_nonzero(current)
        )
        if not minimum_area_ratio <= area_ratio <= maximum_area_ratio:
            candidate = current.copy()
        stabilized.append(candidate)
    return stabilized


def segment_arm_hand_video(
    frames_dir: str | Path,
    hand_joints_camera_cv: np.ndarray,
    arm_joints_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    sam2_root: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    model_size: str = "small",
    prompt_stride: int = 5,
    hand_joint_confidence: np.ndarray | None = None,
    negative_joints_camera_cv: np.ndarray | None = None,
    negative_joint_confidence: np.ndarray | None = None,
    temporal_stabilization: bool = True,
    temporal_neighbor_weight: float = 0.35,
    temporal_flow_scale: float = 0.5,
) -> None:
    """Segment the target hand and sleeve, then stabilize their video contour."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 stage requires CUDA; refusing to use a fallback")
    if model_size not in SAM2_CONFIGS:
        raise ValueError(f"model_size must be one of {sorted(SAM2_CONFIGS)}")

    frames_dir = Path(frames_dir).resolve()
    output_dir = Path(output_dir)
    debug_root = output_dir.parent / f"{output_dir.name}_debug"
    if debug_root.exists():
        shutil.rmtree(debug_root)
    hand_component_dir = debug_root / "hand_component"
    arm_component_dir = debug_root / "arm_component"
    raw_union_dir = debug_root / "raw_union"
    pre_temporal_dir = debug_root / "clean_pre_temporal"
    output_dir.mkdir(parents=True, exist_ok=True)
    hand_component_dir.mkdir(parents=True, exist_ok=True)
    arm_component_dir.mkdir(parents=True, exist_ok=True)
    raw_union_dir.mkdir(parents=True, exist_ok=True)
    pre_temporal_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    count = len(frame_paths)
    if count != len(hand_joints_camera_cv) or count != len(arm_joints_camera_cv):
        raise ValueError("frame, hand, and arm annotation counts do not match")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"could not decode {frame_paths[0]}")
    height, width = first.shape[:2]

    sam2_root = Path(sam2_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(sam2_root))
    try:
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            SAM2_CONFIGS[model_size], str(checkpoint), device="cuda"
        )
        # A complete EgoDex trajectory is commonly 300 frames.  Keeping the
        # resized video and all per-frame tracking state on CUDA scales GPU
        # memory linearly with sequence length and can exhaust even an H100.
        # SAM2 documents only a small tracking-speed penalty for CPU state
        # offload, while asynchronous frame loading hides most video-transfer
        # latency.  The model and active features remain on CUDA.
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=True,
        )
        predictor.reset_state(state)
        stride = max(1, int(prompt_stride))
        prompt_frames = list(range(0, count, stride))
        if prompt_frames[-1] != count - 1:
            prompt_frames.append(count - 1)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            positive_seeds: dict[int, np.ndarray] = {}
            for frame_index in prompt_frames:
                hand_points, hand_labels, hand_box = _prompt_for_frame(
                    hand_joints_camera_cv[frame_index],
                    intrinsic,
                    width,
                    height,
                    18,
                    None
                    if hand_joint_confidence is None
                    else hand_joint_confidence[frame_index],
                    None
                    if negative_joints_camera_cv is None
                    else negative_joints_camera_cv[frame_index],
                    None
                    if negative_joint_confidence is None
                    else negative_joint_confidence[frame_index],
                )
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_index,
                    obj_id=1,
                    points=hand_points,
                    labels=hand_labels,
                    box=hand_box,
                )
                other_hand_negatives = hand_points[hand_labels == 0]
                arm_points, arm_labels, arm_box = _arm_prompt_for_frame(
                    hand_joints_camera_cv[frame_index],
                    arm_joints_camera_cv[frame_index],
                    intrinsic,
                    width,
                    height,
                    other_hand_negatives,
                )
                positive_seeds[frame_index] = np.concatenate(
                    [hand_points[hand_labels == 1], arm_points[arm_labels == 1]], axis=0
                )
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_index,
                    obj_id=2,
                    points=arm_points,
                    labels=arm_labels,
                    box=arm_box,
                )

            masks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for frame_index, object_ids, logits in predictor.propagate_in_video(state):
                ids = list(object_ids)
                if 1 not in ids or 2 not in ids:
                    raise RuntimeError(
                        f"SAM2 lost a hand/arm object at frame {frame_index}: {ids}"
                    )
                hand_mask = (logits[ids.index(1), 0] > 0).cpu().numpy()
                arm_mask = (logits[ids.index(2), 0] > 0).cpu().numpy()
                masks[int(frame_index)] = (hand_mask, arm_mask)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(sam2_root))

    missing = sorted(set(range(count)) - set(masks))
    if missing:
        raise RuntimeError(f"SAM2 did not return frames: {missing}")
    clean_masks: list[np.ndarray] = []
    for frame_index in range(count):
        hand_mask, arm_mask = masks[frame_index]
        union = hand_mask | arm_mask
        # Build exact per-frame seeds even when SAM2 was conditioned sparsely.
        # The sample wrapper uses stride 1 because future prompts do not repair
        # an earlier miss during forward propagation.
        if frame_index not in positive_seeds:
            hand_points, hand_labels, _ = _prompt_for_frame(
                hand_joints_camera_cv[frame_index],
                intrinsic,
                width,
                height,
                18,
                None
                if hand_joint_confidence is None
                else hand_joint_confidence[frame_index],
                None
                if negative_joints_camera_cv is None
                else negative_joints_camera_cv[frame_index],
                None
                if negative_joint_confidence is None
                else negative_joint_confidence[frame_index],
            )
            arm_points, arm_labels, _ = _arm_prompt_for_frame(
                hand_joints_camera_cv[frame_index],
                arm_joints_camera_cv[frame_index],
                intrinsic,
                width,
                height,
                hand_points[hand_labels == 0],
            )
            positive_seeds[frame_index] = np.concatenate(
                [hand_points[hand_labels == 1], arm_points[arm_labels == 1]], axis=0
            )
        geometry_envelope = arm_hand_geometry_envelope(
            hand_joints_camera_cv[frame_index],
            arm_joints_camera_cv[frame_index],
            intrinsic,
            width,
            height,
        )
        # PHANTOM hard-crops hand masks to tracked boxes. For a whole arm, an
        # axis-aligned box is too destructive, so use an oriented per-frame
        # wrist-to-elbow envelope instead. Apply it before component cleanup:
        # otherwise any oversized SAM region touching the image border survives.
        clean_union = _clean_arm_hand_mask(
            union & geometry_envelope, positive_seeds[frame_index]
        )
        if np.count_nonzero(clean_union) < 256:
            raise RuntimeError(f"SAM2 arm+hand mask too small at frame {frame_index}")
        clean_masks.append(clean_union)
        cv2.imwrite(
            str(hand_component_dir / f"{frame_index:05d}.png"),
            hand_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(arm_component_dir / f"{frame_index:05d}.png"),
            arm_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(raw_union_dir / f"{frame_index:05d}.png"),
            union.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(pre_temporal_dir / f"{frame_index:05d}.png"),
            clean_union.astype(np.uint8) * 255,
        )

    if temporal_stabilization and count > 1:
        grayscale_frames = [
            cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            for frame_path in frame_paths
        ]
        if any(frame is None for frame in grayscale_frames):
            raise RuntimeError("could not decode a frame for mask stabilization")
        final_masks = stabilize_binary_mask_sequence(
            clean_masks,
            grayscale_frames,
            neighbor_weight=temporal_neighbor_weight,
            flow_scale=temporal_flow_scale,
        )
    else:
        final_masks = clean_masks

    for frame_index, stabilized in enumerate(final_masks):
        hand_support = tracked_hand_support_mask(
            hand_joints_camera_cv[frame_index],
            intrinsic,
            width,
            height,
            None
            if hand_joint_confidence is None
            else hand_joint_confidence[frame_index],
        )
        stabilized = stabilized | hand_support
        # Re-apply seeded-component filtering after temporal consensus so flow
        # cannot introduce a detached foreground component from the other hand.
        final_mask = _clean_arm_hand_mask(
            stabilized, positive_seeds[frame_index]
        )
        if np.count_nonzero(final_mask) < 256:
            raise RuntimeError(
                f"stabilized arm+hand mask too small at frame {frame_index}"
            )
        if not cv2.imwrite(
            str(output_dir / f"{frame_index:05d}.png"),
            final_mask.astype(np.uint8) * 255,
        ):
            raise RuntimeError(f"could not write stabilized mask {frame_index}")


def union_mask_directories(
    input_directories: tuple[str | Path, ...], output_directory: str | Path
) -> None:
    """Combine aligned binary-mask sequences without changing their geometry."""

    if len(input_directories) < 2:
        raise ValueError("at least two mask directories are required")
    sequences = [sorted(Path(path).glob("*.png")) for path in input_directories]
    frame_count = len(sequences[0])
    if frame_count == 0 or any(len(sequence) != frame_count for sequence in sequences):
        raise ValueError("mask directories are empty or have different frame counts")
    expected_names = [f"{index:05d}.png" for index in range(frame_count)]
    for sequence in sequences:
        if [path.name for path in sequence] != expected_names:
            raise ValueError("mask directory does not contain a contiguous frame sequence")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for frame_index in range(frame_count):
        masks = [
            cv2.imread(str(sequence[frame_index]), cv2.IMREAD_GRAYSCALE)
            for sequence in sequences
        ]
        if any(mask is None for mask in masks):
            raise RuntimeError(f"could not decode mask frame {frame_index}")
        shape = masks[0].shape
        if any(mask.shape != shape for mask in masks):
            raise ValueError(f"mask resolution mismatch at frame {frame_index}")
        union = np.logical_or.reduce([mask > 127 for mask in masks])
        if np.count_nonzero(union) < 256:
            raise RuntimeError(f"combined mask too small at frame {frame_index}")
        if not cv2.imwrite(
            str(output_directory / f"{frame_index:05d}.png"),
            union.astype(np.uint8) * 255,
        ):
            raise RuntimeError(f"could not write combined mask frame {frame_index}")
