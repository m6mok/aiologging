"""
Delivery guarantees under stress (the D1/D2/D3 features):

- D1 ``flush_sync``: repeated loop churn with a backlog, concurrent
  callers from foreign threads, and records stuck in handler buffers
  when the loop dies.
- D2 inline delivery: an ERROR burst through the bridge must be
  bounded by the token bucket, a hanging send (the DNS hole) must not
  stall the caller past the deadline, and concurrent threads must
  neither deadlock nor lose or duplicate records.
- D3 ``LevelAwareDrop``: under sustained overload every ERROR
  survives, only low-severity records are sacrificed, and the drop
  accounting still balances.

Everything runs offline: the queue path uses ``httpx.MockTransport``,
the inline path replaces ``urllib.request.urlopen`` with a local fake
(restored afterwards).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..metrics import wait_until
from ..runner import Context, Skip, scenario
from ..sinks import CollectorHandler
from ..workload import make_stdlib_record, produce, produce_many

# Slow-sink delay for the overload scenario, matching overload.py
_SLOW = 0.002


# ----------------------------------------------------------------------
# Offline stand-ins for the inline (urllib) and queue (httpx) paths
# ----------------------------------------------------------------------


class _FakeResponse:
    status = 200

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _FakeUrlopen:
    """Replacement for urllib.request.urlopen; counts inline posts."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.posts: List[str] = []
        self._lock = threading.Lock()

    def __call__(
        self,
        request: Any,
        timeout: Optional[float] = None,
        context: Any = None,
    ) -> _FakeResponse:
        if self.delay:
            time.sleep(self.delay)
        payload = json.loads(request.data.decode("utf-8"))
        with self._lock:
            self.posts.append(payload["text"])
        return _FakeResponse()

    @property
    def records(self) -> int:
        with self._lock:
            return sum(len(text.splitlines()) for text in self.posts)


class _PatchedUrlopen:
    """Context manager swapping urllib.request.urlopen for a fake."""

    def __init__(self, fake: Callable[..., Any]) -> None:
        self.fake = fake

    def __enter__(self) -> None:
        import urllib.request

        self._original = urllib.request.urlopen
        urllib.request.urlopen = self.fake  # type: ignore[assignment]

    def __exit__(self, *exc: Any) -> None:
        import urllib.request

        urllib.request.urlopen = self._original


def _make_telegram_pager(
    ctx: Context, manager: Any, **handler_kwargs: Any
) -> Tuple[Any, Dict[str, int]]:
    """
    A "pager" logger with a Telegram handler whose queue path counts
    delivered records through an httpx mock transport.
    """
    try:
        import httpx
    except ImportError:
        raise Skip("httpx not installed")
    from aiologging.handlers.telegram import AsyncTelegramHandler
    from aiologging.types import BatchConfig

    counters = {"queue_records": 0}

    def respond(request: "httpx.Request") -> "httpx.Response":
        text = json.loads(request.content)["text"]
        counters["queue_records"] += len(text.splitlines())
        return httpx.Response(200)

    handler = AsyncTelegramHandler(
        token="123456:stress-token",
        chat_id="42",
        backend="httpx",
        batch_config=BatchConfig(batch_size=10_000, flush_interval=3600.0),
        **handler_kwargs,
    )
    handler._create_session = (  # type: ignore[method-assign]
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond))
    )

    pager = manager.getLogger("pager")
    pager.setLevel("INFO")
    pager.propagate = False
    pager.addHandler(handler)
    return handler, counters


# ----------------------------------------------------------------------
# D1: flush_sync
# ----------------------------------------------------------------------


@scenario("delivery.flush_sync_loop_churn")
def flush_sync_loop_churn(ctx: Context) -> None:
    """D1: repeated loop death + flush_sync cycles lose nothing."""
    cycles = ctx.n(10, 3)
    count_each = ctx.n(200, 50)
    manager = ctx.new_manager()
    sink = CollectorHandler(delay=0.001, track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    drains_ok = 0
    worst_drain_s = 0.0
    for cycle in range(cycles):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(produce(logger, cycle, count_each))
        finally:
            loop.close()
        started = time.perf_counter()
        if manager.flush_sync(timeout=30.0):
            drains_ok += 1
        worst_drain_s = max(worst_drain_s, time.perf_counter() - started)

    sent = cycles * count_each
    expected: Set[Tuple[int, int]] = {
        (cycle, seq)
        for cycle in range(cycles)
        for seq in range(count_each)
    }
    pairs = sink.pairs()
    unique = set(pairs)
    missing = expected - unique
    duplicates = len(pairs) - len(unique)

    ctx.metrics.update(
        {
            "cycles": cycles,
            "records_sent": sent,
            "records_delivered": sink.received,
            "drains_ok": drains_ok,
            "worst_drain_s": round(worst_drain_s, 3),
            "duplicates": duplicates,
        }
    )
    ctx.check(
        "every flush_sync cycle drained fully",
        drains_ok == cycles,
        f"{drains_ok}/{cycles}",
    )
    ctx.check(
        "no record lost across the churn",
        not missing,
        f"missing={len(missing)}",
    )
    ctx.check(
        "at most the in-flight record duplicated per cycle",
        duplicates <= cycles,
        f"duplicates={duplicates}",
    )
    ctx.check(
        "nothing left undelivered", manager.undelivered() == 0
    )


@scenario("delivery.flush_sync_from_threads")
async def flush_sync_from_threads(ctx: Context) -> None:
    """D1: flush_sync from foreign threads while the loop is live."""
    manager = ctx.new_manager()
    sink = CollectorHandler(track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)
    producers = 3
    count_each = ctx.n(2_000, 300)

    # Register the consumer loop first so every threaded flush_sync
    # takes the run_coroutine_threadsafe path. Without records the
    # manager has no loop yet, and a thread would spin up a private
    # drain loop that fights the just-started consumer loop — a
    # legitimate best-effort cold-start race, but not this scenario.
    await logger.info("warmup", extra={"seq": -1, "producer": 0})

    producing = asyncio.get_running_loop().create_task(
        produce_many(logger, producers, count_each)
    )

    results: List[bool] = []

    def flusher() -> None:
        for _ in range(ctx.n(5, 2)):
            results.append(manager.flush_sync(timeout=30.0))

    threads = 3
    await asyncio.gather(
        *(asyncio.to_thread(flusher) for _ in range(threads))
    )
    sent = await producing + 1  # + the warmup record
    await manager.flush()

    ctx.metrics.update(
        {
            "records_sent": sent,
            "records_delivered": sink.received,
            "flush_sync_calls": len(results),
            "flush_sync_ok": sum(results),
        }
    )
    ctx.check(
        "every threaded flush_sync succeeded",
        all(results) and results,
        f"{sum(results)}/{len(results)}",
    )
    ctx.check(
        "every record delivered",
        sink.received == sent,
        f"received={sink.received} sent={sent}",
    )
    ctx.check("ordering preserved", sink.ordered_per_producer())


@scenario("delivery.flush_sync_buffered_http")
def flush_sync_buffered_http(ctx: Context) -> None:
    """D1: records buffered in an HTTP handler survive loop death."""
    try:
        import httpx
    except ImportError:
        raise Skip("httpx not installed")
    from aiologging.handlers.http import AsyncHttpJsonHandler
    from aiologging.types import BatchConfig

    delivered: List[dict] = []

    def respond(request: "httpx.Request") -> "httpx.Response":
        delivered.extend(json.loads(request.content))
        return httpx.Response(200)

    handler = AsyncHttpJsonHandler(
        "https://stress.invalid/logs",
        backend="httpx",
        batch_config=BatchConfig(batch_size=10_000, flush_interval=3600.0),
    )
    handler._create_session = (  # type: ignore[method-assign]
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond))
    )

    manager = ctx.new_manager()
    logger = ctx.new_logger(manager)
    logger.addHandler(handler)
    count = ctx.n(500, 100)

    async def fill_buffer() -> None:
        await produce(logger, 0, count)
        # wait until the records sit in the handler buffer (nothing
        # posts them: the batch is huge and the interval is an hour)
        arrived = await wait_until(
            lambda: len(handler._buffer) >= count, timeout_s=15.0
        )
        assert arrived is not None, "records never reached the buffer"

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(fill_buffer())
    finally:
        loop.close()
    buffered_at_death = len(handler._buffer)

    drained = manager.flush_sync(timeout=30.0)

    ctx.metrics.update(
        {
            "records_sent": count,
            "buffered_at_loop_death": buffered_at_death,
            "records_delivered_http": len(delivered),
            "undelivered_after": manager.undelivered(),
        }
    )
    ctx.check(
        "the backlog really was in the buffer",
        buffered_at_death == count,
        f"buffered={buffered_at_death}",
    )
    ctx.check("flush_sync reported success", drained)
    ctx.check(
        "every buffered record delivered without a loop",
        len(delivered) == count,
        f"delivered={len(delivered)} sent={count}",
    )
    ctx.check(
        "nothing left undelivered", manager.undelivered() == 0
    )


# ----------------------------------------------------------------------
# D2: inline delivery through the bridge
# ----------------------------------------------------------------------


@scenario("delivery.inline_error_burst")
def inline_error_burst(ctx: Context) -> None:
    """D2: an ERROR burst is bounded by the bucket, delivered once."""
    from aiologging.bridge import StdlibBridgeHandler

    manager = ctx.new_manager(queue_size=50_000)
    _, counters = _make_telegram_pager(ctx, manager)
    bridge = StdlibBridgeHandler(
        manager=manager,
        inline_level=logging.ERROR,
        inline_timeout=1.0,
    )
    count = ctx.n(500, 100)
    fake = _FakeUrlopen(delay=0.005)

    with _PatchedUrlopen(fake):
        started = time.perf_counter()
        for seq in range(count):
            bridge.handle(
                make_stdlib_record(
                    "pager",
                    seq=seq,
                    producer=0,
                    level=logging.ERROR,
                    message=f"err {seq}",
                )
            )
        burst_elapsed = time.perf_counter() - started
        drained = manager.flush_sync(timeout=60.0)

    inline_records = fake.records
    total = inline_records + counters["queue_records"]
    ctx.metrics.update(
        {
            "records_sent": count,
            "burst_elapsed_s": round(burst_elapsed, 3),
            "inline_records": inline_records,
            "queue_records": counters["queue_records"],
        }
    )
    ctx.check("drain succeeded", drained)
    ctx.check(
        "the token bucket let at least one inline send through",
        inline_records >= 1,
    )
    ctx.check(
        "the bucket bounded inline sends (burst=1, 2/min)",
        inline_records <= 2,
        f"inline={inline_records}",
    )
    ctx.check(
        "the burst never stalled on the inline path",
        burst_elapsed < 3.0,
        f"elapsed={burst_elapsed:.3f}s",
    )
    ctx.check(
        "exactly-once overall (inline + queue == sent)",
        total == count,
        f"inline={inline_records} queue={counters['queue_records']} "
        f"sent={count}",
    )


@scenario("delivery.inline_hanging_send")
def inline_hanging_send(ctx: Context) -> None:
    """D2: a hanging send (the DNS hole) cannot stall the caller."""
    from aiologging.bridge import StdlibBridgeHandler

    manager = ctx.new_manager()
    _, counters = _make_telegram_pager(ctx, manager)
    bridge = StdlibBridgeHandler(
        manager=manager,
        inline_level=logging.ERROR,
        inline_timeout=0.3,
    )
    count = 10
    fake = _FakeUrlopen(delay=30.0)  # every send hangs way past 0.3s

    with _PatchedUrlopen(fake):
        started = time.perf_counter()
        for seq in range(count):
            bridge.handle(
                make_stdlib_record(
                    "pager",
                    seq=seq,
                    producer=0,
                    level=logging.ERROR,
                    message=f"err {seq}",
                )
            )
        burst_elapsed = time.perf_counter() - started
        drained = manager.flush_sync(timeout=30.0)

    ctx.metrics.update(
        {
            "records_sent": count,
            "burst_elapsed_s": round(burst_elapsed, 3),
            "queue_records": counters["queue_records"],
        }
    )
    ctx.check(
        "the caller escaped the hang within the deadline",
        burst_elapsed < 2.0,
        f"elapsed={burst_elapsed:.3f}s",
    )
    ctx.check("drain succeeded", drained)
    ctx.check(
        "every record fell back to the queue path",
        counters["queue_records"] == count,
        f"queue={counters['queue_records']} sent={count}",
    )


@scenario("delivery.inline_thread_swarm")
def inline_thread_swarm(ctx: Context) -> None:
    """D2: concurrent threads: no deadlock, no loss, no duplicates."""
    from aiologging.bridge import StdlibBridgeHandler

    manager = ctx.new_manager(queue_size=50_000)
    _, counters = _make_telegram_pager(ctx, manager)
    bridge = StdlibBridgeHandler(
        manager=manager,
        inline_level=logging.ERROR,
        inline_timeout=1.0,
    )
    threads = 4
    count_each = ctx.n(400, 80)
    fake = _FakeUrlopen(delay=0.002)

    def producer(worker: int) -> None:
        for seq in range(count_each):
            bridge.handle(
                make_stdlib_record(
                    "pager",
                    seq=seq,
                    producer=worker,
                    level=logging.ERROR,
                    message=f"err {worker} {seq}",
                )
            )

    with _PatchedUrlopen(fake):
        started = time.perf_counter()
        workers = [
            threading.Thread(target=producer, args=(worker,))
            for worker in range(threads)
        ]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(timeout=60.0)
        hung = any(thread.is_alive() for thread in workers)
        swarm_elapsed = time.perf_counter() - started
        drained = manager.flush_sync(timeout=60.0)

    sent = threads * count_each
    inline_records = fake.records
    total = inline_records + counters["queue_records"]
    ctx.metrics.update(
        {
            "threads": threads,
            "records_sent": sent,
            "swarm_elapsed_s": round(swarm_elapsed, 3),
            "inline_records": inline_records,
            "queue_records": counters["queue_records"],
        }
    )
    ctx.check("no producer thread deadlocked", not hung)
    ctx.check("drain succeeded", drained)
    ctx.check(
        "the bucket bounded inline sends across threads",
        inline_records <= 3,
        f"inline={inline_records}",
    )
    ctx.check(
        "exactly-once overall (inline + queue == sent)",
        total == sent,
        f"inline={inline_records} queue={counters['queue_records']} "
        f"sent={sent}",
    )


# ----------------------------------------------------------------------
# D3: level-aware drop under overload
# ----------------------------------------------------------------------


@scenario("delivery.drop_policy_errors_survive")
async def drop_policy_errors_survive(ctx: Context) -> None:
    """D3: under overload every ERROR survives, only INFO is shed."""
    from aiologging.logger import LevelAwareDrop

    manager = ctx.new_manager(queue_size=200, overflow="drop_old")
    manager.drop_policy = LevelAwareDrop(
        discard_below=logging.WARNING, watermark=0.7
    )
    sink = CollectorHandler(delay=_SLOW, track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    info_producers = 6
    info_each = ctx.n(3_000, 500)
    errors = ctx.n(200, 50)
    error_producer = 999

    async def error_stream() -> None:
        for seq in range(errors):
            await logger.error(
                "critical %d",
                seq,
                extra={"seq": seq, "producer": error_producer},
            )
            await asyncio.sleep(0)

    _, sent_info = await asyncio.gather(
        error_stream(),
        produce_many(logger, info_producers, info_each),
    )
    await manager.flush()

    sent = sent_info + errors
    dropped = int(manager.get_metrics()["records_dropped"])
    errors_delivered = len(sink.by_producer.get(error_producer, []))
    ctx.metrics.update(
        {
            "records_sent": sent,
            "errors_sent": errors,
            "errors_delivered": errors_delivered,
            "records_delivered": sink.received,
            "records_dropped": dropped,
            "drop_ratio": round(dropped / sent, 3),
        }
    )
    ctx.check(
        "every ERROR survived the overload",
        errors_delivered == errors,
        f"delivered={errors_delivered} sent={errors}",
    )
    ctx.check(
        "low-severity records were actually sacrificed", dropped > 0
    )
    ctx.check(
        "accounting balances (sent == delivered + dropped)",
        sent == sink.received + dropped,
        f"sent={sent} delivered={sink.received} dropped={dropped}",
    )
    ctx.check(
        "delivered records stay ordered", sink.ordered_per_producer()
    )
