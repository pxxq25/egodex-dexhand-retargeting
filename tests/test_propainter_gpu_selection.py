from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/recompose_episode_with_sam3.py"
SPEC = importlib.util.spec_from_file_location("recompose_episode_with_sam3", SCRIPT)
RECOMPOSE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RECOMPOSE)


def test_prefers_requested_gpu_when_it_has_memory() -> None:
    assert RECOMPOSE.choose_persistent_gpus(
        ["2"], {"0": 70_000, "2": 30_000}
    ) == ["2"]


def test_reassigns_to_freest_gpu_under_contention() -> None:
    assert RECOMPOSE.choose_persistent_gpus(
        ["2"], {"0": 40_000, "2": 17_000, "6": 80_000}
    ) == ["6"]


def test_reduces_parallelism_when_only_one_gpu_is_safe() -> None:
    assert RECOMPOSE.choose_persistent_gpus(
        ["2", "7"], {"2": 17_000, "6": 80_000, "7": 17_000}
    ) == ["6"]


def test_returns_empty_when_every_gpu_is_below_threshold() -> None:
    assert RECOMPOSE.choose_persistent_gpus(
        ["2"], {"2": 17_000, "7": 20_000}
    ) == []
