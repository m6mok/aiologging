"""
Load generators: concurrent async producers, paced (fixed-rate)
producers, and stdlib-record factories for bridge/thread scenarios.

Every produced record carries ``extra={"seq": i, "producer": p}`` so
sinks can verify ordering and identify exactly which records were
dropped or lost.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

from aiologging.logger import AsyncLogger


async def produce(
    logger: AsyncLogger,
    producer: int,
    count: int,
    payload_bytes: int = 32,
    measure: bool = False,
    yield_every: int = 0,
) -> List[float]:
    """
    Send ``count`` records as fast as the queue allows.

    Returns per-call latencies (seconds) when ``measure`` is set,
    otherwise an empty list.
    """
    latencies: List[float] = []
    pad = "x" * payload_bytes
    for i in range(count):
        if measure:
            started = time.perf_counter()
        await logger.info(
            "stress %d %s",
            i,
            pad,
            extra={"seq": i, "producer": producer},
        )
        if measure:
            latencies.append(time.perf_counter() - started)
        if yield_every and (i + 1) % yield_every == 0:
            await asyncio.sleep(0)
    return latencies


async def produce_many(
    logger: AsyncLogger,
    producers: int,
    count_each: int,
    payload_bytes: int = 32,
    yield_every: int = 64,
) -> int:
    """Run ``producers`` concurrent producers; returns records sent."""
    await asyncio.gather(
        *(
            produce(
                logger,
                producer,
                count_each,
                payload_bytes=payload_bytes,
                yield_every=yield_every,
            )
            for producer in range(producers)
        )
    )
    return producers * count_each


async def produce_paced(
    logger: AsyncLogger,
    producer: int,
    rate_per_s: int,
    duration_s: float,
    payload_bytes: int = 32,
) -> int:
    """
    Send records at a fixed rate for a fixed duration.

    Records are sent in 10 ms ticks; if the queue applies
    backpressure the effective rate simply drops. Returns the number
    of records actually sent.
    """
    tick = 0.01
    per_tick = max(1, int(rate_per_s * tick))
    deadline = time.perf_counter() + duration_s
    seq = 0
    while time.perf_counter() < deadline:
        tick_started = time.perf_counter()
        for _ in range(per_tick):
            await logger.info(
                "paced %d",
                seq,
                extra={"seq": seq, "producer": producer},
            )
            seq += 1
        elapsed = time.perf_counter() - tick_started
        if elapsed < tick:
            await asyncio.sleep(tick - elapsed)
    return seq


def make_stdlib_record(
    name: str,
    seq: int,
    producer: int,
    level: int = logging.INFO,
    message: Optional[str] = None,
) -> logging.LogRecord:
    """A stdlib record tagged like the async producers tag theirs."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message if message is not None else f"bridge {seq}",
        args=(),
        exc_info=None,
    )
    record.seq = seq
    record.producer = producer
    return record
