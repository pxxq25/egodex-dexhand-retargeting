#!/usr/bin/env python3
"""Visualize and measure raw SAM2 and raw direct-SAM3 masks without post-processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--sam2-raw", required=True, type=Path)
    parser.add_argument("--sam3-raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def read_mask(directory: Path, index: int) -> np.ndarray:
    mask = cv2.imread(str(directory / f"{index:05d}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"missing mask {index:05d}.png in {directory}")
    return mask > 0


def overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = frame.copy()
    tint = np.empty_like(frame)
    tint[:] = color
    alpha = (mask.astype(np.float32) * 0.55)[..., None]
    result = np.clip(result * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, color, 2, cv2.LINE_AA)
    return result


def label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(
        result, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2
    )
    return result


def temporal_change(masks: list[np.ndarray]) -> float:
    if len(masks) < 2:
        return 0.0
    return float(np.mean([np.mean(a != b) for a, b in zip(masks, masks[1:])]))


def main() -> None:
    args = parse_args()
    frame_paths = sorted(args.frames.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"no JPEG frames in {args.frames}")
    frames = [cv2.imread(str(path)) for path in frame_paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("failed to decode one or more frames")

    sam2 = [read_mask(args.sam2_raw, i) for i in range(len(frames))]
    sam3 = [read_mask(args.sam3_raw, i) for i in range(len(frames))]
    height, width = frames[0].shape[:2]
    if any(mask.shape != (height, width) for mask in sam2 + sam3):
        raise RuntimeError("frame and mask dimensions differ")

    args.output.mkdir(parents=True, exist_ok=True)
    panel_width = 640
    panel_height = int(round(height * panel_width / width))
    writer = cv2.VideoWriter(
        str(args.output / "original_raw_sam2_raw_sam3.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (panel_width * 3, panel_height),
    )
    ious = []
    for frame, sam2_mask, sam3_mask in zip(frames, sam2, sam3):
        union = np.count_nonzero(sam2_mask | sam3_mask)
        ious.append(
            float(np.count_nonzero(sam2_mask & sam3_mask) / union) if union else 1.0
        )
        panels = [
            label(frame, "Original"),
            label(overlay(frame, sam2_mask, (0, 0, 255)), "Raw SAM2"),
            label(overlay(frame, sam3_mask, (255, 255, 0)), "Raw SAM3"),
        ]
        panels = [cv2.resize(panel, (panel_width, panel_height)) for panel in panels]
        writer.write(np.concatenate(panels, axis=1))
    writer.release()

    metrics = {
        "frames": len(frames),
        "mean_raw_sam2_sam3_iou": float(np.mean(ious)),
        "minimum_raw_sam2_sam3_iou": float(np.min(ious)),
        "raw_sam2_mean_area_fraction": float(np.mean([mask.mean() for mask in sam2])),
        "raw_sam3_mean_area_fraction": float(np.mean([mask.mean() for mask in sam3])),
        "raw_sam2_temporal_change": temporal_change(sam2),
        "raw_sam3_temporal_change": temporal_change(sam3),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
