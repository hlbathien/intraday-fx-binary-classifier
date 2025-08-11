"""Parallel execution utilities.

Provides a simple abstraction for mapping a function across an iterable
using a ProcessPool with an automatically determined worker count.

Design goals:
    * Safe on Windows (caller must guard with if __name__ == "__main__")
    * Preserve input order in returned results
    * Lightweight dependency footprint (std lib + optional tqdm)
    * Reasonable default worker count (reserve 1 core for OS)

If an exception occurs in a worker, it's re-raised after cancelling
remaining futures so the caller can handle/log appropriately.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import Iterable, List, Sequence, TypeVar, Callable, Any
import os

try:  # optional
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

T = TypeVar("T")
R = TypeVar("R")


def determine_workers(preferred: int | None = None) -> int:
    """Return an appropriate worker count.

    Reserves one core for the OS when possible to keep the machine responsive.
    Clamps preferred inside [1, cpu_count-1]. Falls back to 1 if cpu_count unknown.
    """
    cpu = os.cpu_count() or 1
    max_allow = max(1, cpu - 1)
    if preferred is None:
        return max_allow
    return max(1, min(preferred, max_allow))


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    workers: int | None = None,
    desc: str | None = None,
    chunksize: int = 1,
    show_progress: bool = True,
) -> List[R]:
    """Execute fn over items in parallel returning results in input order.

    Parameters
    ----------
    fn : picklable top-level function executed in each worker.
    items : work items (materialized to list for order reconstruction).
    workers : process count (auto if None).
    desc : optional progress bar description when tqdm available.
    chunksize : placeholder for future batching (unused for now).
    show_progress : disable to suppress progress bar even if tqdm present.
    """
    seq: List[T] = list(items) if not isinstance(items, list) else items
    n = len(seq)
    if n == 0:
        return []
    w = determine_workers(workers)

    results: List[Any] = [None] * n  # type: ignore
    with ProcessPoolExecutor(max_workers=w) as ex:
        future_to_idx: dict[Future[R], int] = {}
        for idx, item in enumerate(seq):
            future = ex.submit(fn, item)
            future_to_idx[future] = idx

        iterator = as_completed(future_to_idx)
        if tqdm and show_progress:
            iterator = tqdm(iterator, total=n, desc=desc)

        first_exc: BaseException | None = None
        for fut in iterator:  # type: ignore
            idx = future_to_idx[fut]
            if first_exc is not None:
                continue
            try:
                results[idx] = fut.result()
            except BaseException as e:  # capture & cancel all
                first_exc = e
                for f2 in future_to_idx:
                    if f2 is not fut:
                        f2.cancel()
        if first_exc:
            raise first_exc
    return results  # type: ignore


__all__ = ["determine_workers", "parallel_map"]
