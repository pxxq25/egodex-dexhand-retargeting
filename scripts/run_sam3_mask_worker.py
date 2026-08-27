#!/usr/bin/env python3
"""Run a fixed list of SAM3 mask jobs while loading the model only once."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

from compare_sam3_direct_masks import run


PATH_FIELDS = {
    "video",
    "frames",
    "hdf5",
    "sam2_masks",
    "sam3_checkpoint",
    "output",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sam3.model_builder import build_sam3_video_model

    jobs = json.loads(args.plan.read_text())
    model = build_sam3_video_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device="cuda",
        compile=False,
    )
    failures = []
    for job in jobs:
        values = {
            key: Path(value) if key in PATH_FIELDS else value
            for key, value in job["arguments"].items()
        }
        output = Path(values["output"])
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "sam3.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    run(SimpleNamespace(**values), model=model)
        except Exception:
            failures.append(job["name"])
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            print(f"FAIL {job['name']}", flush=True)
        else:
            print(f"DONE {job['name']}", flush=True)
    if failures:
        raise RuntimeError(f"SAM3 jobs failed: {failures}")


if __name__ == "__main__":
    main()
