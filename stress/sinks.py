"""
Instrumented in-memory handlers used as scenario sinks.

They subclass the real ``AsyncHandler`` so records travel the exact
production path (filters, retry loop, metrics) — the only fake part
is the I/O at the very end: counting instead of writing.
"""

from __future__ import annotations

import asyncio
from logging import LogRecord
from typing import Any, Dict, List, Set, Tuple

from aiologging.handlers.base import AsyncHandler


class CollectorHandler(AsyncHandler):
    """
    Counting sink with optional artificial latency and injected
    failures.

    Args:
        delay: Seconds to sleep per record (simulates slow I/O)
        fail_every: Raise on every N-th emit attempt (0 disables);
            the base-class retry loop then redelivers the record
        track: Record ``(producer, seq)`` pairs from the ``extra``
            payload for ordering / loss analysis
    """

    def __init__(
        self,
        delay: float = 0.0,
        fail_every: int = 0,
        track: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("retry_delay", 0.001)
        super().__init__(**kwargs)
        self.delay = delay
        self.fail_every = fail_every
        self.track = track
        self.received = 0
        self.attempts = 0
        self.injected_failures = 0
        self.by_producer: Dict[int, List[int]] = {}

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        self.attempts += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.fail_every and self.attempts % self.fail_every == 0:
            self.injected_failures += 1
            raise RuntimeError("injected sink failure")
        self.received += 1
        if self.track:
            seq = record.__dict__.get("seq")
            if seq is not None:
                producer = record.__dict__.get("producer", 0)
                self.by_producer.setdefault(producer, []).append(seq)

    def pairs(self) -> List[Tuple[int, int]]:
        """Every received ``(producer, seq)`` in arrival order."""
        return [
            (producer, seq)
            for producer, seqs in self.by_producer.items()
            for seq in seqs
        ]

    def unique_pairs(self) -> Set[Tuple[int, int]]:
        return set(self.pairs())

    def ordered_per_producer(self) -> bool:
        """Whether each producer's records arrived in send order."""
        return all(
            all(a < b for a, b in zip(seqs, seqs[1:]))
            for seqs in self.by_producer.values()
        )

    def max_seq(self, producer: int) -> int:
        seqs = self.by_producer.get(producer)
        return max(seqs) if seqs else -1


class HangingHandler(AsyncHandler):
    """Sink whose first record never completes (stuck I/O)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.started = 0

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        self.started += 1
        await asyncio.sleep(3600.0)
