from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_propainter_jobs.py"
SPEC = importlib.util.spec_from_file_location("run_propainter_jobs", SCRIPT)
WORKER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WORKER)


def test_cached_factory_constructs_only_once() -> None:
    calls = []

    def factory(value: int) -> object:
        calls.append(value)
        return object()

    cached = WORKER.cached_factory(factory)
    first = cached(1)
    second = cached(2)

    assert first is second
    assert calls == [1]


def test_cuda_oom_detection_is_specific() -> None:
    assert WORKER.is_cuda_out_of_memory(
        RuntimeError("CUDA out of memory. Tried to allocate 414 MiB")
    )
    assert not WORKER.is_cuda_out_of_memory(RuntimeError("CUDA kernel failed"))
    assert not WORKER.is_cuda_out_of_memory(MemoryError("out of memory"))


def test_release_transient_cuda_memory_empties_torch_cache(monkeypatch) -> None:
    calls: list[str] = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append("empty_cache"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(WORKER.gc, "collect", lambda: calls.append("gc"))

    WORKER.release_transient_cuda_memory()

    assert calls == ["gc", "empty_cache"]
