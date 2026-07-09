"""
Measurement helpers: latency percentiles, Python-heap sampling for
soak scenarios, and small polling utilities.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from types import TracebackType
from typing import Callable, List, Optional, Type


def percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1, max(0, round(pct / 100.0 * (len(ordered) - 1)))
    )
    return ordered[index]


def latency_summary_ms(latencies: List[float]) -> dict:
    """p50/p95/p99/max of a latency series, in milliseconds."""
    return {
        "latency_p50_ms": round(percentile(latencies, 50) * 1000, 3),
        "latency_p95_ms": round(percentile(latencies, 95) * 1000, 3),
        "latency_p99_ms": round(percentile(latencies, 99) * 1000, 3),
        "latency_max_ms": round(max(latencies) * 1000, 3)
        if latencies
        else 0.0,
    }


class HeapSampler:
    """
    Periodic sampler of the traced Python heap (``tracemalloc``).

    Use as an async context manager around the load phase; afterwards
    :meth:`growth_bytes` compares the early samples with the late ones
    to estimate steady-state growth (leaks), ignoring the warmup.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.samples: List[int] = []
        self._task: Optional[asyncio.Task[None]] = None
        self._started_tracing = False

    async def __aenter__(self) -> "HeapSampler":
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracing = True
        self._task = asyncio.get_running_loop().create_task(self._run())
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._started_tracing:
            tracemalloc.stop()

    async def _run(self) -> None:
        while True:
            self.samples.append(tracemalloc.get_traced_memory()[0])
            await asyncio.sleep(self.interval_s)

    def growth_bytes(self) -> int:
        """Median of the last quarter minus median of the second one.

        The first quarter is treated as warmup and ignored entirely.
        """
        if len(self.samples) < 8:
            return 0
        quarter = len(self.samples) // 4
        early = self.samples[quarter:2 * quarter]
        late = self.samples[-quarter:]
        return int(statistics.median(late) - statistics.median(early))


async def wait_until(
    condition: Callable[[], bool],
    timeout_s: float,
    poll_s: float = 0.002,
) -> Optional[float]:
    """
    Poll until ``condition()`` holds; returns the elapsed seconds or
    None on timeout.
    """
    started = time.perf_counter()
    deadline = started + timeout_s
    while time.perf_counter() < deadline:
        if condition():
            return time.perf_counter() - started
        await asyncio.sleep(poll_s)
    return None
