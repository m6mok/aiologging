"""
Fault injection and hostile environments: failing and hanging sinks,
an HTTP endpoint that errors, event-loop churn with rescue, the
sync flush path, and a thread swarm feeding the bridge entry point.

Note: the loop-churn scenarios close event loops with tasks still
pending on purpose; asyncio may print "Task was destroyed but it is
pending!" — that noise is part of the scenario, not a failure.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import List, Set, Tuple

from ..metrics import wait_until
from ..runner import Context, Skip, scenario
from ..sinks import CollectorHandler, HangingHandler
from ..workload import make_stdlib_record, produce, produce_many


@scenario("chaos.flaky_handler")
async def flaky_handler(ctx: Context) -> None:
    """A sink failing every 3rd attempt: retries mask it fully."""
    manager = ctx.new_manager()
    flaky = CollectorHandler(fail_every=3, track=True)
    healthy = CollectorHandler(track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(flaky)
    logger.addHandler(healthy)

    sent = await produce_many(logger, 2, ctx.n(1_000, 150))
    await manager.flush()

    ctx.metrics.update(
        {
            "records_sent": sent,
            "injected_failures": flaky.injected_failures,
            "flaky_attempts": flaky.attempts,
        }
    )
    ctx.check("failures were actually injected", flaky.injected_failures > 0)
    ctx.check(
        "flaky sink recovered every record via retries",
        flaky.received == sent,
        f"received={flaky.received} sent={sent}",
    )
    ctx.check(
        "healthy sink was never disturbed",
        healthy.received == sent,
        f"received={healthy.received} sent={sent}",
    )
    ctx.check(
        "retries kept per-producer ordering",
        flaky.ordered_per_producer(),
    )


@scenario("chaos.hanging_handler")
async def hanging_handler(ctx: Context) -> None:
    """A stuck sink must not stall others nor block shutdown."""
    manager = ctx.new_manager()
    hanging = HangingHandler()
    healthy = CollectorHandler(track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(hanging)
    logger.addHandler(healthy)

    sent = await produce_many(logger, 2, ctx.n(400, 80))
    healthy_done = await wait_until(
        lambda: healthy.received >= sent, timeout_s=15.0
    )

    started = time.perf_counter()
    await manager.shutdown(timeout=0.3)
    shutdown_elapsed = time.perf_counter() - started

    ctx.metrics.update(
        {
            "records_sent": sent,
            "healthy_done_s": round(healthy_done or -1.0, 3),
            "hanging_started": hanging.started,
            "shutdown_elapsed_s": round(shutdown_elapsed, 3),
        }
    )
    ctx.check(
        "healthy sink delivered everything despite the hang",
        healthy_done is not None and healthy.received == sent,
        f"received={healthy.received} sent={sent}",
    )
    ctx.check(
        "hanging sink is truly stuck on its first record",
        hanging.started <= 1,
        f"started={hanging.started}",
    )
    ctx.check(
        "shutdown escaped the hang within its timeout",
        shutdown_elapsed < 2.0,
        f"elapsed={shutdown_elapsed:.3f}s",
    )


@scenario("chaos.http_mock_endpoint")
async def http_mock_endpoint(ctx: Context) -> None:
    """Batched HTTP sink against an endpoint failing every 4th call."""
    try:
        import httpx
    except ImportError:
        raise Skip("httpx not installed")
    from collections import Counter
    from aiologging.handlers.http import AsyncHttpJsonHandler
    from aiologging.types import BatchConfig

    delivered: List[dict] = []
    state = {"requests": 0, "rejected": 0}

    def respond(request: "httpx.Request") -> "httpx.Response":
        state["requests"] += 1
        if state["requests"] % 4 == 0:
            state["rejected"] += 1
            return httpx.Response(500, text="injected server error")
        delivered.extend(json.loads(request.content))
        return httpx.Response(200)

    handler = AsyncHttpJsonHandler(
        "https://stress.invalid/logs",
        backend="httpx",
        batch_config=BatchConfig(
            batch_size=50,
            flush_interval=0.2,
            max_retries=5,
            retry_delay=0.01,
        ),
    )
    handler._session = httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    )

    manager = ctx.new_manager()
    logger = ctx.new_logger(manager)
    logger.addHandler(handler)

    producers = 2
    count_each = ctx.n(1_000, 150)
    sent = await produce_many(logger, producers, count_each)
    await manager.flush()

    # Producers share the message text per seq, so with no loss and
    # no duplicated batch every distinct message arrives exactly
    # ``producers`` times.
    counts = Counter(item["message"] for item in delivered)
    off_count = sum(
        1 for count in counts.values() if count != producers
    )
    ctx.metrics.update(
        {
            "records_sent": sent,
            "records_delivered_http": len(delivered),
            "http_requests": state["requests"],
            "http_rejections": state["rejected"],
        }
    )
    ctx.check("server errors were injected", state["rejected"] > 0)
    ctx.check(
        "every record eventually reached the endpoint",
        len(delivered) == sent,
        f"delivered={len(delivered)} sent={sent}",
    )
    ctx.check(
        "no record lost or duplicated by batch retries",
        len(counts) == count_each and off_count == 0,
        f"distinct={len(counts)} expected={count_each} "
        f"off_count={off_count}",
    )


@scenario("chaos.http_aiohttp_backend")
async def http_aiohttp_backend(ctx: Context) -> None:
    """The aiohttp backend against an endpoint failing every 4th call."""
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        raise Skip("aiohttp not installed")
    from collections import Counter

    from aiologging.handlers.http import AsyncHttpJsonHandler
    from aiologging.types import BatchConfig

    delivered: List[dict] = []
    state = {"requests": 0, "rejected": 0}

    class _FakeResponse:
        def __init__(self, status: int) -> None:
            self.status = status

        async def text(self) -> str:
            return "injected server error" if self.status != 200 else ""

    class _FakeAiohttpSession:
        """Duck-typed aiohttp.ClientSession: request/closed/close."""

        closed = False

        async def request(self, **kwargs: object) -> _FakeResponse:
            state["requests"] += 1
            if state["requests"] % 4 == 0:
                state["rejected"] += 1
                return _FakeResponse(500)
            payload = kwargs.get("json")
            assert isinstance(payload, list)
            delivered.extend(payload)
            return _FakeResponse(200)

        async def close(self) -> None:
            self.closed = True

    handler = AsyncHttpJsonHandler(
        "https://stress.invalid/logs",
        backend="aiohttp",
        batch_config=BatchConfig(
            batch_size=50,
            flush_interval=0.2,
            max_retries=5,
            retry_delay=0.01,
        ),
    )
    handler._create_session = (  # type: ignore[method-assign]
        lambda: _FakeAiohttpSession()
    )

    manager = ctx.new_manager()
    logger = ctx.new_logger(manager)
    logger.addHandler(handler)

    producers = 2
    count_each = ctx.n(1_000, 150)
    sent = await produce_many(logger, producers, count_each)
    await manager.flush()

    counts = Counter(item["message"] for item in delivered)
    off_count = sum(
        1 for count in counts.values() if count != producers
    )
    ctx.metrics.update(
        {
            "records_sent": sent,
            "records_delivered_http": len(delivered),
            "http_requests": state["requests"],
            "http_rejections": state["rejected"],
        }
    )
    ctx.check("server errors were injected", state["rejected"] > 0)
    ctx.check(
        "every record eventually reached the endpoint",
        len(delivered) == sent,
        f"delivered={len(delivered)} sent={sent}",
    )
    ctx.check(
        "no record lost or duplicated by batch retries",
        len(counts) == count_each and off_count == 0,
        f"distinct={len(counts)} expected={count_each} "
        f"off_count={off_count}",
    )


@scenario("chaos.topology_churn")
async def topology_churn(ctx: Context) -> None:
    """Handlers added and removed mid-stream: the stable one is whole."""
    manager = ctx.new_manager()
    stable = CollectorHandler(track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(stable)

    producers = 2
    count_each = ctx.n(2_000, 400)
    churn_cycles = ctx.n(40, 10)
    transient_received = 0

    async def churner() -> None:
        nonlocal transient_received
        for _ in range(churn_cycles):
            transient = CollectorHandler()
            logger.addHandler(transient)
            await asyncio.sleep(0.002)
            logger.removeHandler(transient)
            transient_received += transient.received
            await asyncio.sleep(0)

    producing = asyncio.gather(
        produce_many(logger, producers, count_each), churner()
    )
    sent, _ = await producing
    await manager.flush()

    dropped = int(manager.get_metrics()["records_dropped"])
    ctx.metrics.update(
        {
            "records_sent": sent,
            "churn_cycles": churn_cycles,
            "stable_received": stable.received,
            "transient_received": transient_received,
            "records_dropped": dropped,
        }
    )
    ctx.check(
        "the stable handler got every record",
        stable.received == sent,
        f"received={stable.received} sent={sent}",
    )
    ctx.check(
        "transient handlers saw some records",
        transient_received > 0,
    )
    ctx.check("nothing dropped", dropped == 0)
    ctx.check(
        "ordering preserved through the churn",
        stable.ordered_per_producer(),
    )


@scenario("chaos.manager_reuse_after_shutdown")
async def manager_reuse_after_shutdown(ctx: Context) -> None:
    """shutdown() then logging again: the second generation works."""
    manager = ctx.new_manager()
    generations = ctx.n(5, 2)
    count_each = ctx.n(500, 100)
    results = []

    for generation in range(generations):
        sink = CollectorHandler(track=True)
        logger = manager.getLogger(f"gen{generation}")
        logger.setLevel("INFO")
        logger.propagate = False
        logger.addHandler(sink)
        sent = await produce_many(logger, 2, count_each)
        await manager.flush()
        results.append((sink.received, sent))
        await manager.shutdown(timeout=10.0)
        pristine = (
            manager.undelivered() == 0 and not manager.loggerDict
        )
        results[-1] += (pristine,)

    ctx.metrics.update(
        {
            "generations": generations,
            "records_each": 2 * count_each,
        }
    )
    ctx.check(
        "every generation delivered everything",
        all(received == sent for received, sent, _ in results),
        str([(r, s) for r, s, _ in results]),
    )
    ctx.check(
        "the manager is pristine after every shutdown",
        all(pristine for _, _, pristine in results),
    )


@scenario("chaos.loop_switch")
def loop_switch(ctx: Context) -> None:
    """Kill the event loop mid-stream; a new loop must rescue all."""
    manager = ctx.new_manager()
    sink = CollectorHandler(delay=0.001, track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)
    count_each = ctx.n(300, 80)

    # Phase A: produce, then close the loop with the queue undrained
    # and the consumer/worker tasks still alive.
    loop_a = asyncio.new_event_loop()
    try:
        loop_a.run_until_complete(produce(logger, 0, count_each))
    finally:
        loop_a.close()
    delivered_phase_a = sink.received

    # Phase B: a fresh loop; the first record triggers the consumer
    # rebuild which rescues everything stuck on the dead loop.
    loop_b = asyncio.new_event_loop()
    try:

        async def phase_b() -> None:
            await produce(logger, 1, count_each)
            await manager.flush()

        loop_b.run_until_complete(phase_b())
    finally:
        loop_b.close()

    expected: Set[Tuple[int, int]] = {
        (producer, seq)
        for producer in (0, 1)
        for seq in range(count_each)
    }
    received_pairs = sink.pairs()
    unique = set(received_pairs)
    missing = expected - unique
    duplicates = len(received_pairs) - len(unique)
    dropped = int(manager.get_metrics()["records_dropped"])

    ctx.metrics.update(
        {
            "records_sent": 2 * count_each,
            "delivered_before_switch": delivered_phase_a,
            "records_delivered": sink.received,
            "duplicates_after_rescue": duplicates,
            "records_dropped": dropped,
        }
    )
    ctx.check(
        "no record lost across the loop switch",
        not missing,
        f"missing={len(missing)}",
    )
    ctx.check(
        "at most the in-flight records were duplicated",
        duplicates <= 2,
        f"duplicates={duplicates}",
    )
    ctx.check("nothing counted as dropped", dropped == 0)


@scenario("chaos.flush_sync_rescue")
def flush_sync_rescue(ctx: Context) -> None:
    """flush_sync after the loop died must deliver the backlog."""
    manager = ctx.new_manager()
    sink = CollectorHandler(delay=0.001, track=True)
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)
    count = ctx.n(300, 80)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(produce(logger, 0, count))
    finally:
        loop.close()
    backlog = count - sink.received

    started = time.perf_counter()
    drained = manager.flush_sync(timeout=30.0)
    elapsed = time.perf_counter() - started

    ctx.metrics.update(
        {
            "records_sent": count,
            "backlog_at_flush_sync": backlog,
            "flush_sync_elapsed_s": round(elapsed, 3),
            "records_delivered": sink.received,
            "undelivered_after": manager.undelivered(),
        }
    )
    ctx.check("there was an actual backlog to drain", backlog > 0)
    ctx.check("flush_sync reported success", drained)
    ctx.check(
        "every record delivered without an event loop",
        sink.received == count,
        f"received={sink.received} sent={count}",
    )
    ctx.check(
        "nothing left undelivered", manager.undelivered() == 0
    )


@scenario("chaos.bridge_thread_swarm")
async def bridge_thread_swarm(ctx: Context) -> None:
    """Foreign threads flooding enqueue_from_thread concurrently."""
    threads = 6
    count_each = ctx.n(1_500, 250)
    manager = ctx.new_manager(queue_size=50_000)
    sink = CollectorHandler(track=True)
    bridge_root = manager.getLogger("bridge")
    bridge_root.setLevel("INFO")
    bridge_root.propagate = False
    bridge_root.addHandler(sink)

    # Start the consumer via the public path so the thread producers
    # find a live loop to hop onto.
    warmup = ctx.new_logger(manager, name="warmup")
    await warmup.info("warmup")
    await manager.flush()

    def thread_producer(producer: int) -> None:
        for seq in range(count_each):
            manager.enqueue_from_thread(
                make_stdlib_record(
                    f"bridge.t{producer}", seq=seq, producer=producer
                )
            )

    started = time.perf_counter()
    await asyncio.gather(
        *(
            asyncio.to_thread(thread_producer, producer)
            for producer in range(threads)
        )
    )
    await manager.flush()
    elapsed = time.perf_counter() - started

    sent = threads * count_each
    dropped = int(manager.get_metrics()["records_dropped"])
    ctx.metrics.update(
        {
            "threads": threads,
            "records_sent": sent,
            "records_delivered": sink.received,
            "records_dropped": dropped,
            "bridge_rate_per_s": int(sent / elapsed),
        }
    )
    ctx.check(
        "every bridged record delivered",
        sink.received == sent,
        f"received={sink.received} sent={sent}",
    )
    ctx.check("no drops with a roomy queue", dropped == 0)
    ctx.check(
        "per-thread ordering preserved", sink.ordered_per_producer()
    )
