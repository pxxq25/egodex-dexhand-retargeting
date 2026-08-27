#!/usr/bin/env python3
"""Run multiple unchanged ProPainter invocations in one model process."""

from __future__ import annotations

import argparse
import gc
import json
import os
import runpy
import subprocess
import sys
import traceback
import types
from pathlib import Path
from typing import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--propainter-root", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    return parser.parse_args()


def cached_factory(factory: Callable[..., object]) -> Callable[..., object]:
    """Return one lazily constructed model for every later invocation."""

    instance: object | None = None

    def construct(*args: object, **kwargs: object) -> object:
        nonlocal instance
        if instance is None:
            instance = factory(*args, **kwargs)
        return instance

    return construct


def install_model_cache(propainter_root: Path) -> None:
    """Patch the constructors imported by upstream without changing inference."""

    os.chdir(propainter_root)
    sys.path.insert(0, str(propainter_root))
    import model.modules.flow_comp_raft as flow_comp_raft
    import model.propainter as propainter
    import model.recurrent_flow_completion as recurrent_flow_completion

    # Do not replace the class inside its defining module. Several upstream
    # classes use ``super(ClassName, self)``, which resolves ClassName through
    # that module's globals. Instead, let the inference script import a proxy
    # module whose constructor is cached while the real class module remains
    # intact for its method globals.
    for module_name, module, attribute in (
        ("model.modules.flow_comp_raft", flow_comp_raft, "RAFT_bi"),
        (
            "model.recurrent_flow_completion",
            recurrent_flow_completion,
            "RecurrentFlowCompleteNet",
        ),
        ("model.propainter", propainter, "InpaintGenerator"),
    ):
        proxy = types.ModuleType(module_name)
        proxy.__dict__.update(vars(module))
        proxy.__dict__[attribute] = cached_factory(getattr(module, attribute))
        sys.modules[module_name] = proxy


def run_upstream(script: Path, arguments: list[str]) -> None:
    previous = list(sys.argv)
    try:
        sys.argv = [str(script), *arguments]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous


def release_transient_cuda_memory() -> None:
    """Release per-job tensors while retaining the three cached models."""

    gc.collect()
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def is_cuda_out_of_memory(error: BaseException) -> bool:
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def main() -> None:
    args = parse_args()
    propainter_root = args.propainter_root.resolve()
    script = propainter_root / "inference_propainter.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    jobs = json.loads(args.jobs.read_text())
    if not isinstance(jobs, list):
        raise TypeError("jobs must be a JSON list")
    install_model_cache(propainter_root)

    failures: list[str] = []
    for index, job in enumerate(jobs):
        name = str(job["name"])
        print(f"START {index + 1}/{len(jobs)} {name}", flush=True)
        for attempt in range(2):
            try:
                run_upstream(script, [str(value) for value in job["arguments"]])
                postprocess = job.get("postprocess")
                if postprocess:
                    subprocess.run(
                        [str(value) for value in postprocess],
                        check=True,
                        cwd=job.get("postprocess_cwd"),
                    )
            except Exception as error:
                traceback.print_exc()
                if attempt == 0 and is_cuda_out_of_memory(error):
                    print(f"RETRY CUDA OOM {name}", flush=True)
                    continue
                failures.append(name)
                print(f"FAIL {name}", flush=True)
            else:
                print(f"DONE {name}", flush=True)
            finally:
                release_transient_cuda_memory()
            break
    if failures:
        raise RuntimeError(f"persistent ProPainter failures: {failures}")


if __name__ == "__main__":
    main()
