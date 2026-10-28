"""Resolved CUDA runs spread over several GPUs (opt-in: ``MWIR_CUDA_DEVICES``).

``MWIR_CUDA_DEVICES=0,1,2,3`` (or ``all``) runs the fine-grid chunks of a
large resolved run on one worker process per listed GPU; unset, a CUDA run
uses one GPU (``MWIR_CUDA_DEVICE``, default the current device) in this
process. Each worker is **spawned** (a forked child cannot use CUDA once the
parent has initialised it) and pins itself to its GPU through
``MWIR_CUDA_DEVICE`` before anything touches CUDA; every chunk then runs the
same code as on one GPU (:func:`mwir_atmos.api._resolved_chunk`), so results
match a single-GPU run to rounding (chunk boundaries move).

Workers start in a few seconds (imports, line lists, CUDA context), which is
more than a full default band takes on one GPU (~0.55 s on an RTX 4090), so
the pool is kept for the life of the process and reused by later runs
(sweeps, scripts). Chunks are handed out as workers free up, so a slower GPU
(the work server's Ada 6000 beside its Blackwells) takes fewer. After each
chunk a worker returns its cached GPU memory to the driver (the GPUs may be
shared) and keeps only the uploaded line lists.

Scripts on Windows need the usual ``if __name__ == "__main__":`` guard, as
for the CPU backend's worker processes.
"""
from __future__ import annotations

import atexit
import os
import threading
import warnings
from typing import Optional, Sequence

__all__ = ["cuda_devices", "run_chunks", "shutdown"]

_ENV = "MWIR_CUDA_DEVICES"
_POOL = None
_POOL_DEVICES: Optional[tuple] = None
_POOL_LOCK = threading.Lock()
#: The GPU this worker process is pinned to (worker processes only).
_WORKER_DEVICE: Optional[int] = None


def cuda_devices(device_count: Optional[int] = None) -> Optional[tuple[int, ...]]:
    """The GPUs ``MWIR_CUDA_DEVICES`` lists, or None when unset (one GPU).

    ``all`` means every device CuPy sees; otherwise a comma-separated list of
    distinct device indices. ``device_count`` defaults to CuPy's count.
    """
    value = os.environ.get(_ENV, "").strip().lower()
    if not value:
        return None
    if device_count is None:
        from .optics import lbl_cuda
        device_count = lbl_cuda.device_count()
    if value == "all":
        return tuple(range(device_count))
    try:
        devices = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{_ENV} must be 'all' or comma-separated GPU indices, "
                         f"got {value!r}") from exc
    if len(set(devices)) != len(devices) or not devices:
        raise ValueError(f"{_ENV} lists a GPU twice or none: {value!r}")
    bad = [d for d in devices if not 0 <= d < device_count]
    if bad:
        raise ValueError(f"{_ENV}: no GPU {bad} (this machine has {device_count})")
    return devices


def _init_worker(queue) -> None:
    global _WORKER_DEVICE
    _WORKER_DEVICE = int(queue.get())
    os.environ["MWIR_CUDA_DEVICE"] = str(_WORKER_DEVICE)
    os.environ.pop(_ENV, None)      # a worker never fans out again


def _worker_chunk(spectral_pass, job):
    from .api import _resolved_chunk
    try:
        return _resolved_chunk(spectral_pass, *job), _WORKER_DEVICE
    finally:
        if spectral_pass.backend == "cuda":
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()


def _pool(devices: tuple):
    """The persistent spawned pool for ``devices`` (recreated if they change)."""
    global _POOL, _POOL_DEVICES
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with _POOL_LOCK:
        if _POOL is not None and _POOL_DEVICES == devices:
            return _POOL
        shutdown()
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        for device in devices:
            queue.put(device)
        _POOL = ProcessPoolExecutor(max_workers=len(devices), mp_context=context,
                                    initializer=_init_worker, initargs=(queue,))
        _POOL_DEVICES = devices
        return _POOL


def shutdown() -> None:
    """Stop the worker processes (they also stop when this process exits)."""
    global _POOL, _POOL_DEVICES
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=True)
    _POOL, _POOL_DEVICES = None, None


atexit.register(shutdown)


def run_chunks(spectral_pass, jobs: Sequence[tuple], devices: Sequence[int],
               with_devices: bool = False):
    """``_resolved_chunk(spectral_pass, *job)`` for every job, on the worker GPUs, in order.

    Returns None (after a warning) when the workers cannot run, so the caller
    continues on one GPU in this process. ``with_devices`` also returns the
    GPU each chunk ran on.
    """
    from concurrent.futures.process import BrokenProcessPool

    try:
        pool = _pool(tuple(devices))
        done = list(pool.map(_worker_chunk, [spectral_pass] * len(jobs), jobs))
    except (BrokenProcessPool, OSError) as exc:
        shutdown()
        warnings.warn(f"multi-GPU: worker processes unavailable ({exc!r}); "
                      "continuing on one GPU in this process", RuntimeWarning, stacklevel=3)
        return None
    parts = [part for part, _ in done]
    return (parts, [device for _, device in done]) if with_devices else parts
