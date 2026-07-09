"""
Correctness under overload: tiny queues, slow sinks, producers far
faster than consumers. The interesting outputs are not rates but
invariants — drop accounting must balance, drop_old must keep the
newest records, block must lose nothing, flush/shutdown timeouts
must hold under a backlog.
"""

from __future__ import annotations

import asyncio
import time

from ..metrics import wait_until
from ..runner import Context, scenario
from ..sinks import CollectorHandler
from ..workload import produce_many

# Slow-sink delay used by the overflow scenarios: long enough that
# producers overrun the queue by orders of magnitude, short enough
# that draining the accepted records stays fast.
_SLOW = 0.002


@scenario("overload.drop_new_accounting")
async def drop_new_accounting(ctx: Context) -> None:
    """drop_new: sent == delivered + dropped, exactly."""
    manager = ctx.new_manager(queue_size=200, overflow="drop_new")
    sink = CollectorHandler(delay=_SLOW, track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    sent = await produce_many(logger, 6, ctx.n(3_000, 500))
    await manager.flush()

    dropped = int(manager.get_metrics()["records_dropped"])
    ctx.metrics.update(
        {
            "records_sent": sent,
            "records_delivered": sink.received,
            "records_dropped": dropped,
            "drop_ratio": round(dropped / sent, 3),
        }
    )
    ctx.check(
        "accounting balances (sent == delivered + dropped)",
        sent == sink.received + dropped,
        f"sent={sent} delivered={sink.received} dropped={dropped}",
    )
    ctx.check("some records were actually dropped", dropped > 0)
    ctx.check(
        "delivered records stay ordered", sink.ordered_per_producer()
    )


@scenario("overload.drop_old_newest_survive")
async def drop_old_newest_survive(ctx: Context) -> None:
    """drop_old: accounting balances and the newest records survive."""
    producers = 6
    count_each = ctx.n(3_000, 500)
    manager = ctx.new_manager(queue_size=200, overflow="drop_old")
    sink = CollectorHandler(delay=_SLOW, track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    sent = await produce_many(logger, producers, count_each)
    # Sentinels are the newest records in the system; with drop_old
    # (evict the oldest) every one of them must be delivered.
    for producer in range(producers):
        await logger.info(
            "sentinel",
            extra={"seq": count_each, "producer": producer},
        )
    sent += producers
    await manager.flush()

    dropped = int(manager.get_metrics()["records_dropped"])
    sentinels_delivered = sum(
        1
        for producer in range(producers)
        if sink.max_seq(producer) == count_each
    )
    ctx.metrics.update(
        {
            "records_sent": sent,
            "records_delivered": sink.received,
            "records_dropped": dropped,
            "sentinels_delivered": sentinels_delivered,
        }
    )
    ctx.check(
        "accounting balances (sent == delivered + dropped)",
        sent == sink.received + dropped,
        f"sent={sent} delivered={sink.received} dropped={dropped}",
    )
    ctx.check(
        "newest record of every producer survived",
        sentinels_delivered == producers,
        f"{sentinels_delivered}/{producers}",
    )
    ctx.check(
        "delivered records stay ordered", sink.ordered_per_producer()
    )


@scenario("overload.block_backpressure")
async def block_backpressure(ctx: Context) -> None:
    """block: producers are slowed down but nothing is ever lost."""
    manager = ctx.new_manager(queue_size=50, overflow="block")
    sink = CollectorHandler(delay=0.001, track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    started = time.perf_counter()
    sent = await produce_many(logger, 4, ctx.n(1_000, 200))
    producers_done = time.perf_counter()
    await manager.flush()
    drained = time.perf_counter()

    dropped = int(manager.get_metrics()["records_dropped"])
    ctx.metrics.update(
        {
            "records_sent": sent,
            "producer_wall_s": round(producers_done - started, 3),
            "total_wall_s": round(drained - started, 3),
            "backpressure_share": round(
                (producers_done - started) / (drained - started), 3
            ),
        }
    )
    ctx.check("nothing dropped", dropped == 0, f"dropped={dropped}")
    ctx.check(
        "every record delivered",
        sink.received == sent,
        f"received={sink.received} sent={sent}",
    )
    ctx.check("ordering preserved", sink.ordered_per_producer())


@scenario("overload.slow_handler_isolation")
async def slow_handler_isolation(ctx: Context) -> None:
    """A slow handler must not delay a fast one (0.2.3 guarantee)."""
    count_each = ctx.n(1_000, 200)
    producers = 2
    sent = producers * count_each
    # Queue larger than the workload so the block policy never
    # propagates the slow handler's backpressure to the fast one.
    manager = ctx.new_manager(queue_size=sent * 2)
    fast = CollectorHandler()
    slow = CollectorHandler(delay=_SLOW)
    logger = ctx.new_logger(manager)
    logger.addHandler(fast)
    logger.addHandler(slow)

    started = time.perf_counter()
    await produce_many(logger, producers, count_each)
    fast_done = await wait_until(
        lambda: fast.received >= sent, timeout_s=30.0
    )
    await manager.flush()
    slow_done = time.perf_counter() - started

    ctx.metrics.update(
        {
            "records_sent": sent,
            "fast_handler_done_s": round(fast_done or -1.0, 3),
            "all_handlers_done_s": round(slow_done, 3),
        }
    )
    ctx.check(
        "fast handler finished", fast_done is not None
    )
    ctx.check(
        "fast handler unaffected by the slow one",
        fast_done is not None and fast_done < slow_done * 0.5,
        f"fast={fast_done} total={slow_done}",
    )
    ctx.check(
        "slow handler still got everything", slow.received == sent
    )


@scenario("overload.flush_timeout_under_backlog")
async def flush_timeout_under_backlog(ctx: Context) -> None:
    """flush(timeout) must give up on time, a later flush delivers."""
    manager = ctx.new_manager(queue_size=10_000)
    sink = CollectorHandler(delay=0.005)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    sent = await produce_many(logger, 2, ctx.n(300, 60))

    started = time.perf_counter()
    timed_out = False
    try:
        await manager.flush(timeout=0.2)
    except asyncio.TimeoutError:
        timed_out = True
    timeout_elapsed = time.perf_counter() - started

    await manager.flush()

    ctx.metrics.update(
        {
            "records_sent": sent,
            "flush_timeout_elapsed_s": round(timeout_elapsed, 3),
            "delivered_after_full_flush": sink.received,
        }
    )
    ctx.check("bounded flush raised TimeoutError", timed_out)
    ctx.check(
        "bounded flush returned promptly",
        timeout_elapsed < 1.0,
        f"elapsed={timeout_elapsed:.3f}s",
    )
    ctx.check(
        "records survive an expired flush",
        sink.received == sent,
        f"received={sink.received} sent={sent}",
    )


@scenario("overload.shutdown_under_load")
async def shutdown_under_load(ctx: Context) -> None:
    """shutdown(timeout) under a backlog: prompt and accounted for."""
    manager = ctx.new_manager(queue_size=10_000)
    sink = CollectorHandler(delay=0.005)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    sent = await produce_many(logger, 2, ctx.n(400, 80))
    started = time.perf_counter()
    await manager.shutdown(timeout=0.3)
    elapsed = time.perf_counter() - started

    ctx.metrics.update(
        {
            "records_sent": sent,
            "shutdown_elapsed_s": round(elapsed, 3),
            "delivered_before_shutdown": sink.received,
            "undelivered_after_shutdown": manager.undelivered(),
        }
    )
    ctx.check(
        "shutdown honoured its timeout",
        elapsed < 2.0,
        f"elapsed={elapsed:.3f}s",
    )
    ctx.check(
        "manager is pristine afterwards",
        manager.undelivered() == 0
        and not manager.loggerDict,
    )
