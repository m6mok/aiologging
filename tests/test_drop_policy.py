"""
Level-aware queue drop policy (LevelAwareDrop).

Opt-in, mirroring rsyslog discardSeverity / logback
discardingThreshold: under pressure low-severity records are
sacrificed before high-severity ones. Above the watermark arriving
expendable records are shed; on a full queue the oldest expendable
queued record is evicted to make room; with no victims the configured
overflow policy applies unchanged.

The tests exploit that ``enqueue`` with a non-blocking overflow
policy never yields to the event loop, so the consumer cannot drain
the queue between calls and fill levels stay deterministic.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import aiologging
from aiologging import AsyncLoggerManager, LevelAwareDrop
from aiologging.logger import AsyncLogger

from .test_logger import BlockingHandler, RecordingHandler


def make_manager(
    queue_size: int,
    watermark: float = 0.8,
    discard_below: int = logging.WARNING,
    overflow: str = "drop_old",
    handler: RecordingHandler | None = None,
) -> "tuple[AsyncLoggerManager, AsyncLogger, RecordingHandler]":
    manager = AsyncLoggerManager(
        queue_size=queue_size,
        overflow=overflow,  # type: ignore[arg-type]
        drop_policy=LevelAwareDrop(
            discard_below=discard_below, watermark=watermark
        ),
    )
    if handler is None:
        handler = RecordingHandler()
    logger = manager.getLogger("app")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return manager, logger, handler


class TestLevelAwareDrop:
    def test_expendable_below_the_threshold(self) -> None:
        policy = LevelAwareDrop()
        assert policy.is_expendable(
            logging.LogRecord("x", logging.DEBUG, "f", 1, "m", None, None)
        )
        assert policy.is_expendable(
            logging.LogRecord("x", logging.INFO, "f", 1, "m", None, None)
        )
        assert not policy.is_expendable(
            logging.LogRecord("x", logging.ERROR, "f", 1, "m", None, None)
        )

    def test_watermark_gates_the_shedding(self) -> None:
        policy = LevelAwareDrop(watermark=0.5)
        info = logging.LogRecord(
            "x", logging.INFO, "f", 1, "m", None, None
        )
        error = logging.LogRecord(
            "x", logging.ERROR, "f", 1, "m", None, None
        )
        assert not policy.should_discard_arriving(info, 4, 10)
        assert policy.should_discard_arriving(info, 5, 10)
        assert not policy.should_discard_arriving(error, 10, 10)
        # unbounded queue: never shed
        assert not policy.should_discard_arriving(info, 100, 0)

    def test_accepts_level_names(self) -> None:
        policy = LevelAwareDrop(discard_below="ERROR")
        assert policy.discard_below == logging.ERROR

    def test_rejects_bad_watermark(self) -> None:
        with pytest.raises(ValueError):
            LevelAwareDrop(watermark=0.0)
        with pytest.raises(ValueError):
            LevelAwareDrop(watermark=1.5)


class TestMainQueuePolicy:
    def test_errors_survive_a_full_queue_of_info(self) -> None:
        async def scenario() -> None:
            manager, logger, handler = make_manager(
                queue_size=2, watermark=1.0
            )

            # no awaits yield here: the consumer cannot drain between
            # calls, so the queue really fills up
            await logger.info("context-1")
            await logger.info("context-2")  # queue full
            await logger.error("boom-1")  # evicts context-1
            await logger.error("boom-2")  # evicts context-2

            await manager.flush()
            assert handler.messages == ["boom-1", "boom-2"]
            await manager.shutdown()

        asyncio.run(scenario())

    def test_watermark_sheds_arriving_low_severity(self) -> None:
        async def scenario() -> None:
            manager, logger, handler = make_manager(
                queue_size=4, watermark=0.5
            )

            await logger.info("kept")  # qsize 0 < 2: queued
            await logger.error("boom")  # qsize 1 < 2: queued
            await logger.info("shed-1")  # qsize 2 >= 2: shed
            await logger.info("shed-2")  # qsize 2 >= 2: shed

            await manager.flush()
            assert handler.messages == ["kept", "boom"]
            await manager.shutdown()

        asyncio.run(scenario())

    def test_no_victims_falls_back_to_overflow_policy(self) -> None:
        async def scenario() -> None:
            manager, logger, handler = make_manager(
                queue_size=2, watermark=1.0, overflow="drop_old"
            )

            await logger.error("old-1")
            await logger.error("old-2")
            # full, nothing expendable: drop_old drops the oldest
            await logger.error("new")

            await manager.flush()
            assert handler.messages == ["old-2", "new"]
            await manager.shutdown()

        asyncio.run(scenario())

    def test_await_delivery_resolves_for_shed_records(self) -> None:
        async def scenario() -> None:
            manager, logger, _ = make_manager(queue_size=2, watermark=0.5)
            manager.delivery = "await"

            await logger.error("boom")  # qsize 1 >= 1 afterwards
            # must resolve promptly even though the record is shed
            await asyncio.wait_for(logger.info("shed"), timeout=1.0)

            await manager.shutdown()

        asyncio.run(scenario())

    def test_sync_producer_path_applies_the_policy(self) -> None:
        """enqueue_from_thread (the stdlib bridge) sheds and evicts."""

        async def scenario() -> None:
            manager, logger, handler = make_manager(
                queue_size=2, watermark=1.0
            )

            manager._put_nowait_sync(
                logger,
                logging.LogRecord(
                    "app", logging.INFO, "f", 1, "context", None, None
                ),
            )
            manager._put_nowait_sync(
                logger,
                logging.LogRecord(
                    "app", logging.INFO, "f", 1, "shed", None, None
                ),
            )  # full (watermark 1.0): second INFO queued
            manager._put_nowait_sync(
                logger,
                logging.LogRecord(
                    "app", logging.ERROR, "f", 1, "boom", None, None
                ),
            )  # evicts the oldest INFO

            await manager.flush()
            assert handler.messages == ["shed", "boom"]
            await manager.shutdown()

        asyncio.run(scenario())


class TestDispatchQueuePolicy:
    def test_dispatch_queue_evicts_expendables(self) -> None:
        async def scenario() -> None:
            handler = BlockingHandler()
            manager, logger, _ = make_manager(
                queue_size=2,
                watermark=1.0,
                overflow="block",
                handler=handler,  # type: ignore[arg-type]
            )

            await logger.error("first")  # taken in-flight by the worker
            for _ in range(20):  # let the worker pick it up
                await asyncio.sleep(0.001)
                if manager._queue is not None and (
                    manager._queue.qsize() == 0
                ):
                    break

            # dispatch queue (maxsize 2) fills, then ERROR evicts INFO
            await logger.info("ctx-1")
            await logger.info("ctx-2")
            await asyncio.sleep(0.01)  # consumer forwards to dispatch
            await logger.error("boom")
            await asyncio.sleep(0.01)

            handler.release.set()
            await manager.flush()
            assert "boom" in handler.messages
            assert "first" in handler.messages
            await manager.shutdown()

        asyncio.run(scenario())


class TestBasicConfig:
    def test_basic_config_sets_the_policy(self) -> None:
        policy = LevelAwareDrop()
        aiologging.basicConfig(drop_policy=policy)
        try:
            assert aiologging._logger_manager.drop_policy is policy
        finally:
            aiologging._logger_manager.drop_policy = None

    def test_default_manager_has_no_policy(self) -> None:
        assert AsyncLoggerManager().drop_policy is None


if __name__ == "__main__":
    pytest.main([__file__])
