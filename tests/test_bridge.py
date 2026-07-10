"""
Tests for the stdlib logging bridge.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from logging import LogRecord
from typing import Any, List

import pytest

import aiologging
from aiologging import captureStdlib
from aiologging.bridge import StdlibBridgeHandler
from aiologging.handlers.base import AsyncHandler
from aiologging.logger import AsyncLoggerManager


class RecordingHandler(AsyncHandler):
    """Handler that stores handled records in memory."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.records: List[LogRecord] = []

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        self.records.append(record)

    @property
    def messages(self) -> List[str]:
        return [record.getMessage() for record in self.records]


@pytest.fixture
def stdlib_logger():
    """An isolated stdlib logger that reaches the stdlib root."""
    logger = logging.getLogger("bridged.lib")
    logger.setLevel(logging.DEBUG)
    yield logger
    logger.setLevel(logging.NOTSET)


class TestCaptureStdlib:
    """captureStdlib installs/removes the bridge on the stdlib root."""

    def test_install_and_remove(self) -> None:
        stdlib_root = logging.getLogger()
        before = list(stdlib_root.handlers)

        captureStdlib(True)
        installed = [
            h
            for h in stdlib_root.handlers
            if isinstance(h, StdlibBridgeHandler)
        ]
        assert len(installed) == 1

        captureStdlib(True)  # idempotent
        installed = [
            h
            for h in stdlib_root.handlers
            if isinstance(h, StdlibBridgeHandler)
        ]
        assert len(installed) == 1

        captureStdlib(False)
        assert stdlib_root.handlers == before

    async def test_stdlib_records_reach_async_handlers(
        self, stdlib_logger: logging.Logger
    ) -> None:
        handler = RecordingHandler()
        root = aiologging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        captureStdlib(True)

        stdlib_logger.warning("from stdlib: %s", "library")
        await aiologging.flush()

        assert handler.messages == ["from stdlib: library"]
        record = handler.records[0]
        assert record.name == "bridged.lib"
        assert record.levelno == logging.WARNING

    async def test_routing_respects_aiologging_hierarchy(
        self, stdlib_logger: logging.Logger
    ) -> None:
        # a handler attached to the matching aiologging logger name
        # receives bridged records without root involvement
        named_handler = RecordingHandler()
        named = aiologging.getLogger("bridged.lib")
        named.addHandler(named_handler)
        named.propagate = False
        captureStdlib(True)

        stdlib_logger.error("routed by name")
        await aiologging.flush()

        assert named_handler.messages == ["routed by name"]

    async def test_bridge_level_filters(
        self, stdlib_logger: logging.Logger
    ) -> None:
        handler = RecordingHandler()
        root = aiologging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        captureStdlib(True, level=logging.ERROR)

        stdlib_logger.warning("below bridge level")
        stdlib_logger.error("above bridge level")
        await aiologging.flush()

        assert handler.messages == ["above bridge level"]


class TestBridgeRobustness:
    """Threading, feedback loops and no-loop buffering."""

    async def test_emit_from_foreign_thread(
        self, stdlib_logger: logging.Logger
    ) -> None:
        handler = RecordingHandler()
        root = aiologging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        captureStdlib(True)

        # make sure the consumer loop exists before the thread logs
        await aiologging.getLogger("warmup").info("warmup")

        done = threading.Event()

        def worker() -> None:
            stdlib_logger.warning("from a thread")
            done.set()

        threading.Thread(target=worker).start()
        await asyncio.get_running_loop().run_in_executor(None, done.wait)
        await asyncio.sleep(0.05)  # let call_soon_threadsafe land
        await aiologging.flush()

        assert "from a thread" in handler.messages

    def test_records_before_any_loop_are_buffered(
        self, stdlib_logger: logging.Logger
    ) -> None:
        handler = RecordingHandler()
        root = aiologging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        captureStdlib(True)

        stdlib_logger.warning("logged before the app started")

        async def app() -> None:
            await aiologging.flush()

        asyncio.run(app())
        assert handler.messages == ["logged before the app started"]

    async def test_feedback_loop_is_broken(
        self, stdlib_logger: logging.Logger
    ) -> None:
        # a handler whose I/O library logs via stdlib logging: without
        # protection every handled record would enqueue a new one
        class ChattyHandler(RecordingHandler):
            async def _emit(
                self, record: LogRecord, formatted_message: str
            ) -> None:
                logging.getLogger("chatty.io").warning("internal chatter")
                self.records.append(record)

        handler = ChattyHandler()
        root = aiologging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        captureStdlib(True)

        stdlib_logger.warning("outer record")
        await aiologging.flush()
        await aiologging.flush()

        assert handler.messages == ["outer record"]

    async def test_block_full_queue_sheds_arriving_bridge_record(
        self,
    ) -> None:
        # "block" degrades to drop_new for sync producers: the
        # arriving bridged record is shed with accounting; records
        # already accepted in the queue are never evicted
        manager = AsyncLoggerManager(queue_size=2, overflow="block")
        handler = RecordingHandler()
        logger = manager.getLogger("app")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        # no awaits between the puts: the consumer cannot drain, so
        # the queue really fills up
        for message in ("first", "second", "shed"):
            manager._put_nowait_sync(
                logger,
                logging.LogRecord(
                    "app", logging.INFO, __file__, 1, message, None, None
                ),
            )

        await manager.flush()
        assert handler.messages == ["first", "second"]
        assert manager.get_metrics()["records_dropped"] == 1
        await manager.shutdown()

    async def test_drop_old_sync_path_still_evicts_oldest(self) -> None:
        manager = AsyncLoggerManager(queue_size=2, overflow="drop_old")
        handler = RecordingHandler()
        logger = manager.getLogger("app")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        for message in ("evicted", "second", "third"):
            manager._put_nowait_sync(
                logger,
                logging.LogRecord(
                    "app", logging.INFO, __file__, 1, message, None, None
                ),
            )

        await manager.flush()
        assert handler.messages == ["second", "third"]
        assert manager.get_metrics()["records_dropped"] == 1
        await manager.shutdown()

    def test_cold_buffer_block_sheds_arriving_record(self) -> None:
        # before any loop exists the cold buffer holds queue_size
        # records; under "block" (and "drop_new") the arriving record
        # is shed — buffered ones (which may include records parked by
        # the dead-loop rescue) survive
        manager = AsyncLoggerManager(queue_size=2, overflow="block")
        handler = RecordingHandler()
        logger = manager.getLogger("app")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        for message in ("first", "second", "shed"):
            manager.enqueue_from_thread(
                logging.LogRecord(
                    "app", logging.INFO, __file__, 1, message, None, None
                )
            )

        async def app() -> None:
            await logger.info("started")
            await manager.flush()

        asyncio.run(app())
        assert handler.messages == ["first", "second", "started"]
        assert manager.get_metrics()["records_dropped"] == 1

        asyncio.run(manager.shutdown())

    def test_cold_buffer_drop_old_evicts_oldest(self) -> None:
        manager = AsyncLoggerManager(queue_size=2, overflow="drop_old")
        handler = RecordingHandler()
        logger = manager.getLogger("app")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        for message in ("evicted", "second", "third"):
            manager.enqueue_from_thread(
                logging.LogRecord(
                    "app", logging.INFO, __file__, 1, message, None, None
                )
            )

        async def app() -> None:
            await manager.flush()

        asyncio.run(app())
        assert handler.messages == ["second", "third"]
        assert manager.get_metrics()["records_dropped"] == 1

        asyncio.run(manager.shutdown())

    async def test_handle_error_on_broken_manager(self) -> None:
        class BrokenManager(AsyncLoggerManager):
            def enqueue_from_thread(self, record: LogRecord) -> None:
                raise RuntimeError("broken")

        bridge = StdlibBridgeHandler(manager=BrokenManager())
        record = logging.LogRecord(
            "x", logging.WARNING, __file__, 1, "msg", None, None
        )
        # handleError must swallow the failure (raiseExceptions aside)
        bridge.emit(record)


if __name__ == "__main__":
    pytest.main([__file__])
