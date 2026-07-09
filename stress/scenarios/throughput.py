"""
Throughput and latency benchmarks on the happy path: how fast the
call site returns, how fast records reach the sinks, and what a
delivery-"await" round trip costs under concurrency.
"""

from __future__ import annotations

import asyncio
import time

from ..metrics import latency_summary_ms
from ..runner import Context, scenario
from ..sinks import CollectorHandler
from ..workload import produce, produce_many


@scenario("throughput.enqueue")
async def enqueue(ctx: Context) -> None:
    """Call-site and end-to-end rate with concurrent producers."""
    producers = 6
    count_each = ctx.n(30_000, 2_000)
    manager = ctx.new_manager()
    sink = CollectorHandler(track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    started = time.perf_counter()
    sent = await produce_many(logger, producers, count_each)
    call_site_done = time.perf_counter()
    await manager.flush()
    drained = time.perf_counter()

    ctx.metrics.update(
        {
            "records_sent": sent,
            "call_site_rate_per_s": int(
                sent / (call_site_done - started)
            ),
            "end_to_end_rate_per_s": int(sent / (drained - started)),
            "drain_tail_s": round(drained - call_site_done, 3),
        }
    )
    dropped = manager.get_metrics()["records_dropped"]
    ctx.check(
        "all records delivered",
        sink.received == sent,
        f"received={sink.received} sent={sent}",
    )
    ctx.check("no drops with block policy", dropped == 0)
    ctx.check(
        "per-producer ordering preserved", sink.ordered_per_producer()
    )


@scenario("throughput.await_latency")
async def await_latency(ctx: Context) -> None:
    """Per-record latency of delivery mode \"await\" under load."""
    producers = 4
    count_each = ctx.n(3_000, 300)
    manager = ctx.new_manager(delivery="await")
    sink = CollectorHandler()
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    started = time.perf_counter()
    latency_lists = await asyncio.gather(
        *(
            produce(logger, producer, count_each, measure=True)
            for producer in range(producers)
        )
    )
    elapsed = time.perf_counter() - started
    latencies = [value for series in latency_lists for value in series]
    sent = producers * count_each

    ctx.metrics.update(
        {
            "records_sent": sent,
            "rate_per_s": int(sent / elapsed),
            **latency_summary_ms(latencies),
        }
    )
    ctx.check(
        "every await implies delivery",
        sink.received == sent,
        f"received={sink.received} sent={sent}",
    )


@scenario("throughput.fanout")
async def fanout(ctx: Context) -> None:
    """One logger fanning out to several handler workers."""
    handlers = 4
    producers = 2
    count_each = ctx.n(10_000, 1_000)
    manager = ctx.new_manager()
    sinks = [CollectorHandler(track=True) for _ in range(handlers)]
    logger = ctx.new_logger(manager)
    for sink in sinks:
        logger.addHandler(sink)

    started = time.perf_counter()
    sent = await produce_many(logger, producers, count_each)
    await manager.flush()
    elapsed = time.perf_counter() - started

    ctx.metrics.update(
        {
            "records_sent": sent,
            "handlers": handlers,
            "end_to_end_rate_per_s": int(sent / elapsed),
            "deliveries_per_s": int(sent * handlers / elapsed),
        }
    )
    for index, sink in enumerate(sinks):
        ctx.check(
            f"handler {index} got every record",
            sink.received == sent,
            f"received={sink.received} sent={sent}",
        )
    ctx.check(
        "ordering preserved in every fanout branch",
        all(sink.ordered_per_producer() for sink in sinks),
    )
