"""
Sustained-load scenarios: memory growth at a steady rate, task
hygiene across produce/flush cycles, and file rotation churn.
"""

from __future__ import annotations

import asyncio

from ..metrics import HeapSampler
from ..runner import Context, Skip, scenario
from ..sinks import CollectorHandler
from ..workload import produce_many, produce_paced

_GROWTH_LIMIT_BYTES = 4 * 1024 * 1024


@scenario("soak.sustained_memory", timeout=300.0)
async def sustained_memory(ctx: Context) -> None:
    """Steady 2000 rec/s; the Python heap must reach a plateau."""
    duration_s = float(ctx.n(40, 5))
    manager = ctx.new_manager()
    sink = CollectorHandler()
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    async with HeapSampler(interval_s=0.1) as sampler:
        sent = await produce_paced(
            logger, 0, rate_per_s=2_000, duration_s=duration_s
        )
        await manager.flush()

    growth = sampler.growth_bytes()
    ctx.metrics.update(
        {
            "duration_s": duration_s,
            "records_sent": sent,
            "achieved_rate_per_s": int(sent / duration_s),
            "heap_growth_kib": growth // 1024,
            "heap_peak_kib": max(sampler.samples) // 1024,
        }
    )
    ctx.check(
        "every record delivered",
        sink.received == sent,
        f"received={sink.received} sent={sent}",
    )
    ctx.check(
        "heap growth stays under 4 MiB after warmup",
        growth < _GROWTH_LIMIT_BYTES,
        f"growth={growth // 1024} KiB",
    )


@scenario("soak.task_hygiene", timeout=300.0)
async def task_hygiene(ctx: Context) -> None:
    """Produce/flush cycles must not accumulate asyncio tasks."""
    cycles = ctx.n(30, 6)
    manager = ctx.new_manager()
    sink = CollectorHandler()
    logger = ctx.new_logger(manager)
    logger.addHandler(sink)

    task_counts = []
    for _ in range(cycles):
        await produce_many(logger, 2, ctx.n(500, 100))
        await manager.flush()
        task_counts.append(len(asyncio.all_tasks()))

    ctx.metrics.update(
        {
            "cycles": cycles,
            "records_delivered": sink.received,
            "tasks_min": min(task_counts),
            "tasks_max": max(task_counts),
        }
    )
    ctx.check(
        "task count is stable across cycles",
        max(task_counts) - min(task_counts) <= 3,
        f"min={min(task_counts)} max={max(task_counts)}",
    )
    ctx.check(
        "exactly one dispatcher per handler",
        len(manager._dispatchers) == 1,
    )


@scenario("soak.rotation_churn", timeout=300.0)
async def rotation_churn(ctx: Context) -> None:
    """Size-based rotation under a continuous stream of records."""
    try:
        import aiofiles  # noqa: F401
    except ImportError:
        raise Skip("aiofiles not installed")
    from aiologging.handlers.rotating import AsyncRotatingFileHandler

    errors = []

    async def error_handler(record: object, error: Exception) -> None:
        errors.append(error)

    max_bytes = 64 * 1024
    backup_count = 4
    log_path = ctx.tmpdir / "churn.log"
    handler = AsyncRotatingFileHandler(
        filename=log_path,
        max_bytes=max_bytes,
        backup_count=backup_count,
        error_handler=error_handler,
    )
    manager = ctx.new_manager()
    logger = ctx.new_logger(manager)
    logger.addHandler(handler)

    sent = await produce_many(
        logger, 2, ctx.n(4_000, 500), payload_bytes=96
    )
    await manager.flush()

    produced_files = sorted(
        path.name for path in ctx.tmpdir.glob("churn.log*")
    )
    total_bytes = sum(
        path.stat().st_size for path in ctx.tmpdir.glob("churn.log*")
    )
    ctx.metrics.update(
        {
            "records_sent": sent,
            "handler_errors": len(errors),
            "files": ", ".join(produced_files),
            "total_bytes": total_bytes,
        }
    )
    ctx.check("no handler errors", not errors, f"errors={len(errors)}")
    ctx.check(
        "rotation actually happened",
        len(produced_files) > 1,
        f"files={len(produced_files)}",
    )
    ctx.check(
        "backup count respected",
        len(produced_files) <= backup_count + 1,
        f"files={len(produced_files)}",
    )
    ctx.check(
        "no file grossly exceeds max_bytes",
        all(
            path.stat().st_size <= max_bytes * 2
            for path in ctx.tmpdir.glob("churn.log*")
        ),
    )
