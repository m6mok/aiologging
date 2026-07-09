"""
Tests for the AsyncLogger class, the manager queue and the module API.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from logging import LogRecord
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

import aiologging
from aiologging import AsyncLogger, getLogger
from aiologging.exceptions import ContextError
from aiologging.handlers.base import AsyncHandler
from aiologging.handlers.stream import AsyncStreamHandler
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


class BlockingHandler(RecordingHandler):
    """Handler that blocks until released; used for overflow tests.

    Uses a threading.Event because on Python 3.9 an asyncio.Event
    created outside a running loop binds to the wrong loop.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.release = threading.Event()

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        while not self.release.is_set():
            await asyncio.sleep(0.001)
        self.records.append(record)


def make_logger(
    name: str = "test",
    level: int = logging.DEBUG,
    delivery: Optional[str] = "await",
    **kwargs: Any,
) -> tuple[AsyncLogger, RecordingHandler]:
    """Create a standalone logger with a recording handler."""
    handler = RecordingHandler()
    logger = AsyncLogger(
        name,
        level=level,
        handlers=[handler],
        propagate=False,
        delivery=delivery,  # type: ignore[arg-type]
        **kwargs,
    )
    return logger, handler


async def drain_until_blocked(manager: AsyncLoggerManager) -> None:
    """Yield until the consumer has taken everything off the queue."""
    for _ in range(50):
        queue = manager._queue
        if queue is not None and queue.qsize() == 0:
            return
        await asyncio.sleep(0)


class TestLoggerConfiguration:
    """Sync configuration API, mirroring logging.Logger."""

    def test_initialization_defaults(self) -> None:
        logger = AsyncLogger("test_logger")
        assert logger.name == "test_logger"
        assert logger.level == logging.NOTSET
        assert logger.handlers == []
        assert logger.filters == []
        assert logger.parent is None
        assert logger.propagate is True
        assert logger.disabled is False

    def test_set_level_accepts_int_and_name(self) -> None:
        logger = AsyncLogger("test_logger")
        logger.setLevel(logging.DEBUG)
        assert logger.level == logging.DEBUG
        logger.setLevel("ERROR")
        assert logger.level == logging.ERROR

    def test_set_level_rejects_garbage(self) -> None:
        logger = AsyncLogger("test_logger")
        with pytest.raises(ValueError):
            logger.setLevel("NO_SUCH_LEVEL")
        with pytest.raises(TypeError):
            logger.setLevel(3.5)  # type: ignore[arg-type]

    def test_get_effective_level_walks_hierarchy(self) -> None:
        parent = AsyncLogger("parent", level=logging.ERROR)
        child = AsyncLogger("parent.child")
        child.parent = parent

        assert child.getEffectiveLevel() == logging.ERROR
        child.setLevel(logging.INFO)
        assert child.getEffectiveLevel() == logging.INFO
        assert AsyncLogger("orphan").getEffectiveLevel() == logging.NOTSET

    def test_is_enabled_for(self) -> None:
        logger = AsyncLogger("test_logger", level=logging.INFO)
        assert logger.isEnabledFor(logging.INFO)
        assert logger.isEnabledFor(logging.WARNING)
        assert not logger.isEnabledFor(logging.DEBUG)

        logger.disabled = True
        assert not logger.isEnabledFor(logging.CRITICAL)

    def test_module_level_disable(self) -> None:
        logger = AsyncLogger("test_logger", level=logging.DEBUG)
        aiologging.disable(logging.ERROR)
        assert not logger.isEnabledFor(logging.ERROR)
        assert logger.isEnabledFor(logging.CRITICAL)

    def test_handler_management(self) -> None:
        logger = AsyncLogger("test_logger")
        handler1 = AsyncStreamHandler()
        handler2 = AsyncStreamHandler()

        logger.addHandler(handler1)
        logger.addHandler(handler1)  # no duplicates
        logger.addHandler(handler2)
        assert logger.handlers == [handler1, handler2]

        logger.removeHandler(handler1)
        assert logger.handlers == [handler2]

    def test_has_handlers_walks_hierarchy(self) -> None:
        parent = AsyncLogger("parent")
        child = AsyncLogger("parent.child")
        child.parent = parent

        assert not child.hasHandlers()
        parent.addHandler(AsyncStreamHandler())
        assert child.hasHandlers()

        child.propagate = False
        assert not child.hasHandlers()

    def test_filter_management(self) -> None:
        logger = AsyncLogger("test_logger")
        filter1 = MagicMock()
        filter2 = MagicMock()

        logger.addFilter(filter1)
        logger.addFilter(filter2)
        assert logger.filters == [filter1, filter2]

        logger.removeFilter(filter1)
        assert logger.filters == [filter2]

    def test_repr(self) -> None:
        logger = AsyncLogger("test_logger", level=logging.INFO)
        assert repr(logger) == "<AsyncLogger test_logger (INFO)>"


class TestLogMethods:
    """The async logging methods."""

    async def test_all_levels(self) -> None:
        logger, handler = make_logger()

        await logger.debug("debug")
        await logger.info("info")
        await logger.warning("warning")
        await logger.error("error")
        await logger.critical("critical")
        await logger.warn("warn alias")
        await logger.fatal("fatal alias")
        await logger.log(logging.INFO, "explicit level")

        assert handler.messages == [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "warn alias",
            "fatal alias",
            "explicit level",
        ]
        levels = [record.levelno for record in handler.records]
        assert levels[:5] == [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]
        assert levels[5] == logging.WARNING
        assert levels[6] == logging.CRITICAL

    async def test_log_rejects_non_int_level(self) -> None:
        logger, _ = make_logger()
        with pytest.raises(TypeError):
            await logger.log("INFO", "nope")  # type: ignore[arg-type]

    async def test_level_filtering(self) -> None:
        logger, handler = make_logger(level=logging.WARNING)
        await logger.debug("dropped")
        await logger.info("dropped")
        await logger.warning("kept")
        assert handler.messages == ["kept"]

    async def test_percent_formatting_is_eager(self) -> None:
        logger, handler = make_logger()
        payload = {"key": "before"}
        await logger.info("value: %s", payload)
        payload["key"] = "after"

        record = handler.records[0]
        assert record.getMessage() == "value: {'key': 'before'}"
        assert record.args is None  # frozen for the queue

    async def test_extra_attributes(self) -> None:
        logger, handler = make_logger()
        await logger.info("msg", extra={"custom_attr": "custom_value"})
        assert handler.records[0].custom_attr == "custom_value"

    async def test_extra_rejects_reserved_keys(self) -> None:
        logger, _ = make_logger()
        with pytest.raises(KeyError):
            await logger.info("msg", extra={"message": "clash"})
        with pytest.raises(KeyError):
            await logger.info("msg", extra={"lineno": 1})

    async def test_exception_captures_current_exc(self) -> None:
        logger, handler = make_logger()
        try:
            raise ValueError("boom")
        except ValueError:
            await logger.exception("failed")

        record = handler.records[0]
        assert record.levelno == logging.ERROR
        # exc_info is rendered into exc_text when the record is frozen
        assert record.exc_info is None
        assert record.exc_text is not None
        assert "ValueError: boom" in record.exc_text

    async def test_exc_info_instance(self) -> None:
        logger, handler = make_logger()
        await logger.error("failed", exc_info=RuntimeError("direct"))
        assert "RuntimeError: direct" in (handler.records[0].exc_text or "")

    async def test_caller_info(self) -> None:
        logger, handler = make_logger()
        await logger.info("who called")
        record = handler.records[0]
        assert record.funcName == "test_caller_info"
        assert record.pathname == __file__

    async def test_stack_info(self) -> None:
        logger, handler = make_logger()
        await logger.info("with stack", stack_info=True)
        stack = handler.records[0].stack_info
        assert stack is not None
        assert stack.startswith("Stack (most recent call last):")

    async def test_disabled_logger_drops_records(self) -> None:
        logger, handler = make_logger()
        logger.disabled = True
        await logger.info("dropped")
        assert handler.records == []

    async def test_closed_logger_drops_records(self) -> None:
        logger, handler = make_logger()
        await logger.close()
        await logger.info("dropped")
        assert handler.records == []

    async def test_logger_filter_drops_records(self) -> None:
        logger, handler = make_logger()
        reject = MagicMock()
        reject.filter.return_value = False
        logger.addFilter(reject)

        await logger.info("dropped")
        assert handler.records == []
        reject.filter.assert_called_once()

    async def test_handler_error_reported_to_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class ExplodingHandler(AsyncHandler):
            async def handle(self, record: LogRecord) -> None:
                raise RuntimeError("handler error")

            async def _emit(
                self, record: LogRecord, formatted_message: str
            ) -> None:  # pragma: no cover
                pass

        logger = AsyncLogger(
            "test",
            level=logging.DEBUG,
            handlers=[ExplodingHandler()],
            propagate=False,
            delivery="await",
        )
        await logger.info("message")
        assert "Error in handler ExplodingHandler" in capsys.readouterr().err

    async def test_rate_limiter_drops_records(self) -> None:
        async def deny() -> bool:
            return False

        logger, handler = make_logger(
            rate_limiter=deny, enable_metrics=True
        )
        await logger.info("dropped")
        assert handler.records == []
        assert logger.get_metrics()["records_dropped"] == 1


class TestPropagation:
    """Record propagation through the hierarchy."""

    async def test_propagates_to_parent_handlers(self) -> None:
        parent, parent_handler = make_logger("parent")
        child = AsyncLogger(
            "parent.child", level=logging.DEBUG, delivery="await"
        )
        child.parent = parent

        await child.info("propagated")
        assert parent_handler.messages == ["propagated"]

    async def test_propagate_false_stops_walk(self) -> None:
        parent, parent_handler = make_logger("parent")
        child, child_handler = make_logger("parent.child")
        child.parent = parent
        child.propagate = False

        await child.info("local only")
        assert child_handler.messages == ["local only"]
        assert parent_handler.messages == []

    async def test_closed_parent_is_skipped(self) -> None:
        parent, parent_handler = make_logger("parent")
        await parent.close()

        child, child_handler = make_logger("parent.child")
        child.propagate = True
        child.parent = parent

        await child.info("still works")
        assert child_handler.messages == ["still works"]

    async def test_last_resort_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        logger = AsyncLogger(
            "test.nohandlers",
            level=logging.DEBUG,
            propagate=False,
            delivery="await",
        )
        await logger.warning("nowhere to go")
        await logger.debug("below WARNING is silent")
        err = capsys.readouterr().err
        assert "nowhere to go" in err
        assert "below WARNING is silent" not in err


class TestDeliveryModes:
    """The delivery extension: what the await guarantees."""

    async def test_enqueue_returns_before_handling(self) -> None:
        handler = BlockingHandler()
        logger = AsyncLogger(
            "test.enqueue",
            level=logging.DEBUG,
            handlers=[handler],
            propagate=False,
            delivery="enqueue",
        )

        await logger.info("queued")  # must not block on the handler
        assert handler.records == []

        handler.release.set()
        await aiologging.flush()
        assert handler.messages == ["queued"]

    async def test_await_resolves_after_handling(self) -> None:
        logger, handler = make_logger(delivery="await")
        await logger.info("handled")
        assert handler.messages == ["handled"]

    async def test_manager_default_delivery(self) -> None:
        manager = AsyncLoggerManager(delivery="await")
        handler = RecordingHandler()
        logger = manager.getLogger("app")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        await logger.info("handled")
        assert handler.messages == ["handled"]
        await manager.shutdown()

    async def test_context_manager_flushes_without_closing(self) -> None:
        handler = RecordingHandler()
        logger = getLogger("test.ctx")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        async with logger:
            await logger.info("inside")

        assert handler.messages == ["inside"]
        assert not logger._closed
        # the same logger keeps working after the with-block
        await logger.info("after")
        await aiologging.flush()
        assert handler.messages == ["inside", "after"]

    async def test_context_manager_rejects_closed_logger(self) -> None:
        logger, _ = make_logger()
        await logger.close()
        with pytest.raises(ContextError):
            async with logger:
                pass  # pragma: no cover


class TestOverflowPolicies:
    """Bounded queue behaviour."""

    async def _stuck_manager(
        self, overflow: str
    ) -> tuple[AsyncLoggerManager, AsyncLogger, BlockingHandler]:
        """Manager with queue_size=1 whose consumer is blocked."""
        manager = AsyncLoggerManager(
            queue_size=1, overflow=overflow  # type: ignore[arg-type]
        )
        handler = BlockingHandler()
        logger = manager.getLogger("app")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        await logger.info("r1")  # consumer takes it and blocks
        await drain_until_blocked(manager)
        await logger.info("r2")  # fills the single queue slot
        return manager, logger, handler

    async def test_drop_new(self) -> None:
        manager, logger, handler = await self._stuck_manager("drop_new")
        await logger.info("r3")  # queue full -> discarded

        handler.release.set()
        await manager.flush()
        assert handler.messages == ["r1", "r2"]
        assert manager.get_metrics()["records_dropped"] == 1
        await manager.shutdown()

    async def test_drop_old(self) -> None:
        manager, logger, handler = await self._stuck_manager("drop_old")
        await logger.info("r3")  # queue full -> r2 discarded

        handler.release.set()
        await manager.flush()
        assert handler.messages == ["r1", "r3"]
        assert manager.get_metrics()["records_dropped"] == 1
        await manager.shutdown()

    async def test_block_waits_for_space(self) -> None:
        manager, logger, handler = await self._stuck_manager("block")

        blocked = asyncio.ensure_future(logger.info("r3"))
        await asyncio.sleep(0.01)
        assert not blocked.done()  # queue is full, producer waits

        handler.release.set()
        await asyncio.wait_for(blocked, timeout=2)
        await manager.flush()
        assert handler.messages == ["r1", "r2", "r3"]
        assert manager.get_metrics()["records_dropped"] == 0
        await manager.shutdown()


class TestLifecycle:
    """Lazy start, flush, shutdown, loop change."""

    async def test_consumer_starts_lazily(self) -> None:
        manager = AsyncLoggerManager()
        assert manager.get_metrics()["consumer_running"] == 0

        logger = manager.getLogger("app")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(RecordingHandler())
        await logger.info("first record")

        assert manager.get_metrics()["consumer_running"] == 1
        await manager.shutdown()
        assert manager.get_metrics()["consumer_running"] == 0

    async def test_flush_drains_queue(self) -> None:
        handler = RecordingHandler()
        logger = getLogger("test.flush")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        for i in range(100):
            await logger.info("msg %d", i)
        await aiologging.flush()
        assert len(handler.records) == 100

    async def test_shutdown_closes_handlers_and_resets(self) -> None:
        handler = RecordingHandler()
        logger = getLogger("test.shutdown")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        await logger.info("before shutdown")

        await aiologging.shutdown()

        assert handler.messages == ["before shutdown"]
        assert handler.get_metrics()["closed"] is True
        assert logger._closed

        # the hierarchy is reset: fresh logger under the same name
        fresh = getLogger("test.shutdown")
        assert fresh is not logger
        assert not fresh._closed

    def test_loop_change_is_survived(self) -> None:
        manager = AsyncLoggerManager()
        handler = RecordingHandler()
        logger = manager.getLogger("app")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        async def first_loop() -> None:
            await logger.info("from loop 1")
            await manager.flush()

        async def second_loop() -> None:
            await logger.info("from loop 2")
            await manager.flush()

        asyncio.run(first_loop())
        asyncio.run(second_loop())

        assert handler.messages == ["from loop 1", "from loop 2"]
        asyncio.run(manager.shutdown())

    def test_records_stuck_in_dead_loop_are_rescued(self) -> None:
        manager = AsyncLoggerManager()
        handler = BlockingHandler()
        logger = manager.getLogger("app")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        async def first_loop() -> None:
            # both records enqueued; the consumer blocks on the first,
            # the second stays in the queue when the loop dies
            await logger.info("r1")
            await logger.info("r2")

        async def second_loop() -> None:
            handler.release.set()
            await manager.flush()

        asyncio.run(first_loop())
        asyncio.run(second_loop())

        assert sorted(handler.messages) == ["r1", "r2"]
        asyncio.run(manager.shutdown())


class TestLoggerManager:
    """Hierarchy management."""

    def test_get_logger_singleton(self) -> None:
        assert getLogger("test_manager") is getLogger("test_manager")

    def test_get_logger_requires_string(self) -> None:
        with pytest.raises(TypeError):
            getLogger(123)  # type: ignore[arg-type]

    def test_root_logger(self) -> None:
        root = getLogger()
        assert root.name == "root"
        assert root.level == logging.WARNING
        assert getLogger("root") is root
        assert getLogger("") is root

    def test_hierarchy_links(self) -> None:
        parent = getLogger("parent")
        child = getLogger("parent.child")
        grandchild = getLogger("parent.child.grandchild")

        assert child.parent is parent
        assert grandchild.parent is child
        assert parent.parent is getLogger()

    def test_intermediate_loggers_created_eagerly(self) -> None:
        leaf = getLogger("a.b.c")
        assert leaf.parent is getLogger("a.b")
        assert getLogger("a.b").parent is getLogger("a")

    def test_get_child(self) -> None:
        assert getLogger("app").getChild("db") is getLogger("app.db")
        assert getLogger("app").getChild("db.pool") is getLogger(
            "app.db.pool"
        )
        assert getLogger().getChild("top") is getLogger("top")

    def test_get_children(self) -> None:
        parent = getLogger("gc_parent")
        child_a = getLogger("gc_parent.a")
        child_b = getLogger("gc_parent.b")
        getLogger("gc_parent.a.deep")

        assert parent.getChildren() == {child_a, child_b}


class TestModuleLevelFunctions:
    """Root-logger convenience coroutines."""

    async def test_info_auto_configures_and_writes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        aiologging.getLogger().setLevel(logging.INFO)
        await aiologging.info("module-level %s", "info")
        await aiologging.flush()
        assert "module-level info" in capsys.readouterr().err

    async def test_default_root_level_is_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        await aiologging.info("hidden")
        await aiologging.warning("visible")
        await aiologging.flush()
        err = capsys.readouterr().err
        assert "hidden" not in err
        assert "visible" in err

    async def test_exception_helper(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        try:
            raise ValueError("root boom")
        except ValueError:
            await aiologging.exception("root failed")
        await aiologging.flush()
        err = capsys.readouterr().err
        assert "root failed" in err
        assert "ValueError: root boom" in err


class TestBasicConfig:
    """basicConfig, like logging.basicConfig."""

    def test_configures_root_once(self) -> None:
        handler = AsyncStreamHandler()
        aiologging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
            handlers=[handler],
        )

        root = getLogger()
        assert root.getEffectiveLevel() == logging.DEBUG
        assert root.handlers == [handler]
        assert handler.formatter is not None

        # a second call must not stack more handlers (stdlib behaviour)
        aiologging.basicConfig(handlers=[AsyncStreamHandler()])
        assert root.handlers == [handler]

    def test_force_replaces_handlers(self) -> None:
        first = AsyncStreamHandler()
        second = AsyncStreamHandler()
        aiologging.basicConfig(handlers=[first])
        aiologging.basicConfig(handlers=[second], force=True)
        assert getLogger().handlers == [second]

    def test_default_handler_created(self) -> None:
        aiologging.basicConfig(level="INFO")
        root = getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], AsyncStreamHandler)
        assert root.getEffectiveLevel() == logging.INFO

    def test_queue_parameters_applied(self) -> None:
        from aiologging.logger import _logger_manager

        aiologging.basicConfig(
            queue_size=123, overflow="drop_old", delivery="await"
        )
        assert _logger_manager.queue_size == 123
        assert _logger_manager.overflow == "drop_old"
        assert _logger_manager.delivery == "await"
        # restore defaults for other tests
        aiologging.basicConfig(
            queue_size=10_000, overflow="block", delivery="enqueue"
        )


if __name__ == "__main__":
    pytest.main([__file__])
