#!/usr/bin/env python3
"""Translate EgoQuest 3-D hands to their RGB-aligned landmark projections."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from egodex_dexhand.data import (
    HAND_SUFFIXES,
    load_egodex_sequence,
    project_camera_points,
)


ANCHOR_INDICES = (0, 4, 8, 12, 16, 20)  # wrist and five fingertips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("rgb_landmarks", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--side", action="append", choices=("left", "right"), required=True)
    parser.add_argument("--median-window", type=int, default=7)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    parser.add_argument(
        "--shared-translation",
        action="store_true",
        help=(
            "apply one camera-space translation to every requested side; "
            "this preserves Quest hand identity when RGB handedness flips"
        ),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input, args.output)

    corrections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    detection_confidence: dict[str, np.ndarray] = {}
    for side in args.side:
        sequence = load_egodex_sequence(args.input, side)
        projected = project_camera_points(
            sequence.joints_camera_cv[:, ANCHOR_INDICES], sequence.intrinsic
        )
        with h5py.File(args.rgb_landmarks, "r") as rgb_handle:
            rgb_pixels = np.stack(
                [
                    np.asarray(
                        rgb_handle[f"transforms/{side}{HAND_SUFFIXES[index]}"][:, :2, 3],
                        dtype=np.float32,
                    )
                    for index in ANCHOR_INDICES
                ],
                axis=1,
            )
            detection_confidence[side] = np.asarray(
                rgb_handle[f"confidences/{side}Hand"], dtype=np.float32
            )
        pixel_delta = np.median(rgb_pixels - projected, axis=1)
        pixel_delta = median_filter(
            pixel_delta,
            size=(max(1, int(args.median_window) | 1), 1),
            mode="nearest",
        )
        pixel_delta = gaussian_filter1d(
            pixel_delta,
            sigma=max(0.0, float(args.gaussian_sigma)),
            axis=0,
            mode="nearest",
        ).astype(np.float32)
        depth = np.median(
            sequence.joints_camera_cv[:, ANCHOR_INDICES, 2], axis=1
        )
        camera_delta = np.zeros((sequence.frame_count, 3), dtype=np.float32)
        camera_delta[:, 0] = pixel_delta[:, 0] * depth / sequence.intrinsic[0, 0]
        camera_delta[:, 1] = pixel_delta[:, 1] * depth / sequence.intrinsic[1, 1]
        corrections[side] = pixel_delta, camera_delta

    if args.shared_translation and len(corrections) > 1:
        sides = tuple(corrections)
        translations = np.stack([corrections[side][1] for side in sides], axis=1)
        confidence = np.stack([detection_confidence[side] for side in sides], axis=1)
        direct = confidence >= 0.9
        has_direct = direct.any(axis=1, keepdims=True)
        weights = np.where(has_direct, direct.astype(np.float32), confidence)
        shared = np.sum(translations * weights[..., None], axis=1) / np.maximum(
            np.sum(weights, axis=1, keepdims=True), 1e-6
        )
        shared = gaussian_filter1d(
            shared.astype(np.float32),
            sigma=max(0.0, float(args.gaussian_sigma)),
            axis=0,
            mode="nearest",
        )
        corrections = {
            side: (pixel_delta, shared.copy())
            for side, (pixel_delta, _) in corrections.items()
        }

    with h5py.File(args.output, "r+") as handle:
        world_from_camera = np.asarray(handle["transforms/camera"], dtype=np.float32)
        for side, (pixel_delta, camera_delta) in corrections.items():
            world_delta = np.einsum(
                "tij,tj->ti", world_from_camera[:, :3, :3], camera_delta
            )
            for name in tuple(handle["transforms"].keys()):
                if not name.startswith(side):
                    continue
                dataset = handle[f"transforms/{name}"]
                values = np.asarray(dataset, dtype=np.float32)
                values[:, :3, 3] += world_delta
                dataset[...] = values
            group = handle.require_group(f"rgb_alignment/{side}")
            group.create_dataset("pixel_delta", data=pixel_delta)
            group.create_dataset("camera_translation", data=camera_delta)
        handle.attrs["rgb_projection_alignment"] = "wrist_and_fingertip_translation"

    for side in args.side:
        corrected = load_egodex_sequence(args.output, side)
        corrected_pixels = project_camera_points(
            corrected.joints_camera_cv[:, ANCHOR_INDICES], corrected.intrinsic
        )
        with h5py.File(args.rgb_landmarks, "r") as rgb_handle:
            rgb_pixels = np.stack(
                [
                    rgb_handle[f"transforms/{side}{HAND_SUFFIXES[index]}"][:, :2, 3]
                    for index in ANCHOR_INDICES
                ],
                axis=1,
            )
        residual = rgb_pixels - corrected_pixels
        print(
            f"{side}: residual median dx={np.median(residual[..., 0]):.2f}px "
            f"dy={np.median(residual[..., 1]):.2f}px"
        )


if __name__ == "__main__":
    main()
