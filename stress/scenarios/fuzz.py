"""
Seeded chaos-mix fuzzing: every round draws a random combination of
queue size, overflow policy, drop policy, delivery mode, sink faults,
bridge threads and (sometimes) loop churn, then verifies the exact
accounting identities that must hold for that combination. The seed
is fixed (override with ``STRESS_SEED``) and reported in the metrics,
and every failing check names the round's full parameter set, so a
failure is reproducible.

Interaction bugs live at the seams between features; the targeted
scenarios cannot see them by construction — this one can.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import List, Optional

from aiologging.logger import LevelAwareDrop

from ..runner import Context, scenario
from ..sinks import CollectorHandler
from ..workload import make_stdlib_record

_DEFAULT_SEED = 20260709
_LEVELS = [logging.INFO, logging.WARNING, logging.ERROR]


@scenario("fuzz.chaos_mix", timeout=240.0)
def chaos_mix(ctx: Context) -> None:
    """Seeded random mix of policies, faults, threads and churn."""
    seed = int(os.environ.get("STRESS_SEED", str(_DEFAULT_SEED)))
    rng = random.Random(seed)
    rounds = ctx.n(12, 4)

    totals = {"sent": 0, "delivered": 0, "dropped": 0}
    accounting_bad: List[str] = []
    loss_bad: List[str] = []
    order_bad: List[str] = []
    dup_bad: List[str] = []
    drain_bad: List[str] = []

    for rnd in range(rounds):
        # Churn rounds kill the loop with a backlog and rescue via
        # flush_sync; they pin block + no policy so the invariant is
        # exact (lose nothing, duplicate at most the in-flight).
        churn = rng.random() < 0.35
        if churn:
            overflow, delivery = "block", "enqueue"
            policy: Optional[LevelAwareDrop] = None
            queue_size = rng.choice([200, 2_000])
        else:
            overflow = rng.choice(["block", "drop_new", "drop_old"])
            policy = rng.choice(
                [
                    None,
                    LevelAwareDrop(
                        discard_below=logging.WARNING,
                        watermark=rng.choice([0.5, 0.8]),
                    ),
                ]
            )
            delivery = rng.choice(["enqueue", "await"])
            queue_size = (
                rng.choice([50, 200, 2_000])
                if overflow != "block"
                else rng.choice([200, 2_000])
            )

        manager = ctx.new_manager(
            queue_size=queue_size,
            overflow=overflow,  # type: ignore[arg-type]
            delivery=delivery,  # type: ignore[arg-type]
            drop_policy=policy,
        )
        sink = CollectorHandler(
            delay=rng.choice([0.0, 0.0005, 0.002]),
            fail_every=rng.choice([0, 0, 5, 9]),
            track=True,
        )
        logger = ctx.new_logger(manager)
        logger.addHandler(sink)

        # Pre-draw every plan so the parameters depend only on the
        # seed, not on task scheduling.
        producers = rng.randint(2, 4)
        count_each = (
            rng.randint(80, 200) if ctx.quick else rng.randint(300, 800)
        )
        plans = [
            [rng.choice(_LEVELS) for _ in range(count_each)]
            for _ in range(producers)
        ]
        threads = 0 if churn else rng.randint(0, 2)
        thread_plans = [
            [rng.choice(_LEVELS) for _ in range(count_each)]
            for _ in range(threads)
        ]
        with_flusher = not churn and rng.random() < 0.3
        sent = (producers + threads) * count_each + 1  # + warmup

        detail = (
            f"round={rnd} seed={seed} churn={churn} "
            f"overflow={overflow} policy={'on' if policy else 'off'} "
            f"delivery={delivery} q={queue_size} "
            f"fail_every={sink.fail_every} delay={sink.delay} "
            f"producers={producers} threads={threads} "
            f"count_each={count_each}"
        )

        async def one_producer(worker: int, plan: List[int]) -> None:
            for seq, level in enumerate(plan):
                await logger.log(
                    level,
                    "fz %d",
                    seq,
                    extra={"seq": seq, "producer": worker},
                )

        def thread_producer(worker: int, plan: List[int]) -> None:
            for seq, level in enumerate(plan):
                manager.enqueue_from_thread(
                    make_stdlib_record(
                        "stress",
                        seq=seq,
                        producer=worker,
                        level=level,
                        message=f"fz {seq}",
                    )
                )

        async def run_round() -> None:
            # Warmup registers the consumer loop so bridge threads
            # and threaded flush_sync have a live loop to hop onto.
            await logger.info(
                "warmup", extra={"seq": 0, "producer": 999}
            )
            jobs = [
                one_producer(worker, plan)
                for worker, plan in enumerate(plans)
            ]
            jobs += [
                asyncio.to_thread(thread_producer, 100 + i, plan)
                for i, plan in enumerate(thread_plans)
            ]
            if with_flusher:
                jobs.append(
                    asyncio.to_thread(manager.flush_sync, 30.0)
                )
            await asyncio.gather(*jobs)
            if not churn:
                await manager.flush()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_round())
        finally:
            loop.close()

        drained = True
        if churn:
            drained = manager.flush_sync(timeout=30.0)

        dropped = int(manager.get_metrics()["records_dropped"])
        pairs = sink.pairs()
        unique = len(set(pairs))
        duplicates = len(pairs) - unique
        totals["sent"] += sent
        totals["delivered"] += sink.received
        totals["dropped"] += dropped

        if churn:
            if not drained:
                drain_bad.append(detail)
            if dropped != 0 or unique != sent:
                loss_bad.append(
                    f"{detail} unique={unique} dropped={dropped} "
                    f"sent={sent}"
                )
            if duplicates > 2:
                dup_bad.append(f"{detail} duplicates={duplicates}")
        else:
            if sink.received + dropped != sent:
                accounting_bad.append(
                    f"{detail} delivered={sink.received} "
                    f"dropped={dropped} sent={sent}"
                )
            if duplicates != 0:
                dup_bad.append(f"{detail} duplicates={duplicates}")
            if overflow == "block" and policy is None:
                # block loses nothing for async producers; bridge
                # threads degrade to drop_new (a sync producer cannot
                # await), so only arriving bridged records may be
                # shed — and never more than the bridge sent
                unique_async = sum(
                    1
                    for producer, _ in sink.unique_pairs()
                    if producer < 100 or producer == 999
                )
                async_sent = producers * count_each + 1  # + warmup
                if unique_async != async_sent:
                    loss_bad.append(
                        f"{detail} async_delivered={unique_async} "
                        f"async_sent={async_sent}"
                    )
                if dropped > threads * count_each:
                    loss_bad.append(
                        f"{detail} dropped={dropped} "
                        f"bridge_sent={threads * count_each}"
                    )
            if not sink.ordered_per_producer():
                order_bad.append(detail)

    ctx.metrics.update(
        {
            "seed": seed,
            "rounds": rounds,
            "records_sent": totals["sent"],
            "records_delivered": totals["delivered"],
            "records_dropped": totals["dropped"],
        }
    )
    ctx.check(
        "accounting balanced in every clean round",
        not accounting_bad,
        "; ".join(accounting_bad[:2]),
    )
    ctx.check(
        "no loss where the combination forbids it",
        not loss_bad,
        "; ".join(loss_bad[:2]),
    )
    ctx.check(
        "every churn round drained",
        not drain_bad,
        "; ".join(drain_bad[:2]),
    )
    ctx.check(
        "duplicates only from churn, and bounded",
        not dup_bad,
        "; ".join(dup_bad[:2]),
    )
    ctx.check(
        "per-producer ordering held in clean rounds",
        not order_bad,
        "; ".join(order_bad[:2]),
    )
