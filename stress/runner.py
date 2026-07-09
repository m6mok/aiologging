"""
Scenario registry and orchestration.

A scenario is a coroutine function (or a plain function when it needs
to manage event loops itself, e.g. loop-switch scenarios) taking a
:class:`Context`. It reports numbers through ``ctx.metrics`` and
correctness invariants through ``ctx.check``; raising :class:`Skip`
marks it as skipped (e.g. a missing optional dependency).

Every scenario runs on a fresh event loop with its own
``AsyncLoggerManager`` instances (created via ``ctx.new_manager``),
so scenarios cannot contaminate each other or the global manager.
"""

from __future__ import annotations

import asyncio
import inspect
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from aiologging.logger import (
    AsyncLogger,
    AsyncLoggerManager,
    DeliveryMode,
    OverflowPolicy,
)

MetricValue = Union[int, float, str, bool, None]


class Skip(Exception):
    """Raised by a scenario to mark itself as skipped."""


@dataclass
class Check:
    """One correctness invariant verified by a scenario."""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class ScenarioResult:
    """Outcome of a single scenario run."""

    name: str
    category: str
    duration_s: float = 0.0
    metrics: Dict[str, MetricValue] = field(default_factory=dict)
    checks: List[Check] = field(default_factory=list)
    error: Optional[str] = None
    skipped: Optional[str] = None

    @property
    def status(self) -> str:
        if self.skipped is not None:
            return "SKIP"
        if self.error is not None:
            return "ERROR"
        if all(check.ok for check in self.checks):
            return "OK"
        return "FAIL"


class Context:
    """
    Per-scenario toolbox: scale selection, isolated managers,
    a temporary directory, and the metrics/checks sinks.
    """

    def __init__(self, quick: bool, tmpdir: Path) -> None:
        self.quick = quick
        self.tmpdir = tmpdir
        self.metrics: Dict[str, MetricValue] = {}
        self.checks: List[Check] = []
        self.managers: List[AsyncLoggerManager] = []

    def n(self, full: int, quick: int) -> int:
        """Pick a workload size depending on the run mode."""
        return quick if self.quick else full

    def new_manager(
        self,
        queue_size: int = 10_000,
        overflow: OverflowPolicy = "block",
        delivery: DeliveryMode = "enqueue",
    ) -> AsyncLoggerManager:
        """Create an isolated manager; the runner shuts it down."""
        manager = AsyncLoggerManager(
            queue_size=queue_size, overflow=overflow, delivery=delivery
        )
        self.managers.append(manager)
        return manager

    def new_logger(
        self, manager: AsyncLoggerManager, name: str = "stress"
    ) -> AsyncLogger:
        """A non-propagating INFO logger on an isolated manager."""
        logger = manager.getLogger(name)
        logger.setLevel("INFO")
        logger.propagate = False
        return logger

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        """Record a correctness invariant."""
        self.checks.append(Check(name=name, ok=ok, detail=detail))

    async def aclose(self) -> None:
        """Shut down every manager the scenario created."""
        for manager in self.managers:
            try:
                await manager.shutdown(timeout=10.0)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass


ScenarioFn = Callable[[Context], object]


@dataclass
class Scenario:
    """A registered scenario."""

    name: str
    category: str
    fn: ScenarioFn
    timeout: float
    doc: str


SCENARIOS: Dict[str, Scenario] = {}


def scenario(
    name: str, timeout: float = 180.0
) -> Callable[[ScenarioFn], ScenarioFn]:
    """Register a scenario under ``category.name``."""

    def decorator(fn: ScenarioFn) -> ScenarioFn:
        category = name.split(".", 1)[0]
        doc = inspect.getdoc(fn) or ""
        SCENARIOS[name] = Scenario(
            name=name,
            category=category,
            fn=fn,
            timeout=timeout,
            doc=doc.splitlines()[0] if doc else "",
        )
        return fn

    return decorator


def select(patterns: List[str]) -> List[Scenario]:
    """Scenarios whose name contains any pattern (all when empty)."""
    ordered = list(SCENARIOS.values())
    if not patterns:
        return ordered
    return [
        item
        for item in ordered
        if any(pattern in item.name for pattern in patterns)
    ]


async def _drive_async(item: Scenario, ctx: Context) -> None:
    """Run an async scenario under its timeout, then tear down."""
    try:
        await asyncio.wait_for(
            ctx_call_async(item, ctx), timeout=item.timeout
        )
    finally:
        await ctx.aclose()


async def ctx_call_async(item: Scenario, ctx: Context) -> None:
    result = item.fn(ctx)
    assert inspect.isawaitable(result)
    await result


def _drive_sync(item: Scenario, ctx: Context) -> None:
    """
    Run a sync scenario in a daemon thread so its timeout holds.

    Sync scenarios exercise exactly the paths that can wedge the
    whole process (``flush_sync``, loop churn), so they must not run
    on the runner's thread. On timeout the thread is abandoned (it
    cannot be killed) but, being a daemon, it will not keep the
    process alive.
    """
    failure: List[BaseException] = []

    def target() -> None:
        try:
            item.fn(ctx)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            failure.append(exc)

    thread = threading.Thread(
        target=target, name=f"stress-{item.name}", daemon=True
    )
    thread.start()
    thread.join(timeout=item.timeout)
    if thread.is_alive():
        raise asyncio.TimeoutError
    if failure:
        raise failure[0]


def _run_one(item: Scenario, quick: bool) -> ScenarioResult:
    result = ScenarioResult(name=item.name, category=item.category)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="aiologging-stress-") as tmp:
        ctx = Context(quick=quick, tmpdir=Path(tmp))
        try:
            if asyncio.iscoroutinefunction(item.fn):
                asyncio.run(_drive_async(item, ctx))
            else:
                # Sync scenarios manage event loops themselves
                # (loop churn is exactly what they stress); clean up
                # their managers on a dedicated loop afterwards.
                try:
                    _drive_sync(item, ctx)
                finally:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(ctx.aclose())
                    finally:
                        loop.close()
        except Skip as skip:
            result.skipped = str(skip)
        except asyncio.TimeoutError:
            result.error = (
                f"scenario exceeded its {item.timeout:.0f}s timeout"
            )
        except Exception:  # noqa: BLE001 - report, keep the run going
            result.error = traceback.format_exc()
        result.metrics = ctx.metrics
        result.checks = ctx.checks
    result.duration_s = time.perf_counter() - started
    return result


def run(
    patterns: List[str],
    quick: bool = False,
    on_start: Optional[Callable[[Scenario], None]] = None,
    on_done: Optional[Callable[[ScenarioResult], None]] = None,
) -> List[ScenarioResult]:
    """Run the selected scenarios sequentially."""
    results: List[ScenarioResult] = []
    for item in select(patterns):
        if on_start is not None:
            on_start(item)
        result = _run_one(item, quick=quick)
        results.append(result)
        if on_done is not None:
            on_done(result)
    return results
