"""
Synchronous emergency drain (flush_sync) and bounded flush/shutdown.

The motivating scenario: critical records are emitted right before the
event loop dies (a crash escaping ``asyncio.run``, a SIGTERM-initiated
teardown), leaving them in the queue or in a handler buffer. flush_sync
must deliver them without a running loop, on a bounded budget, and the
atexit hook must do so automatically.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, List, Optional

import httpx
import pytest

import aiologging
from aiologging import AsyncLoggerManager
from aiologging.handlers import AsyncHandler, AsyncHttpJsonHandler
from aiologging.logger import _drain_at_exit
from aiologging.types import BatchConfig

from .test_logger import BlockingHandler, RecordingHandler


def make_manager_with_recorder(
    handler: Optional[AsyncHandler] = None,
) -> "tuple[AsyncLoggerManager, Any]":
    manager = AsyncLoggerManager()
    if handler is None:
        handler = RecordingHandler()
    logger = manager.getLogger("app")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return manager, logger


class TestFlushSync:
    def test_delivers_records_stuck_after_loop_death(self) -> None:
        manager, logger = make_manager_with_recorder()
        handler = logger.handlers[0]

        async def crash() -> None:
            # enqueue only: the loop dies before the consumer runs
            await logger.error("Service crashed")

        asyncio.run(crash())

        assert manager.flush_sync(timeout=5.0) is True
        assert handler.messages == ["Service crashed"]
        asyncio.run(manager.shutdown())

    def test_delivers_buffered_http_records_without_a_loop(self) -> None:
        """The pager case: the record sits in the HTTP handler buffer."""
        requests: List[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200)

        http_handler = AsyncHttpJsonHandler(
            "https://example.com/api/logs",
            backend="httpx",
            batch_config=BatchConfig(batch_size=100, flush_interval=3600.0),
        )
        http_handler._create_session = (  # type: ignore[method-assign]
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )
        manager, logger = make_manager_with_recorder(http_handler)

        async def crash() -> None:
            await logger.error("Service crashed")
            # let the consumer hand the record to the handler buffer
            await manager.flush()

        asyncio.run(crash())
        # flushed on the first loop already
        assert len(requests) == 1

        async def enqueue_only() -> None:
            await logger.error("dying words")

        asyncio.run(enqueue_only())
        assert manager.flush_sync(timeout=5.0) is True
        assert len(requests) == 2
        asyncio.run(manager.shutdown())

    def test_returns_false_on_timeout_and_keeps_records(self) -> None:
        handler = BlockingHandler()
        manager, logger = make_manager_with_recorder(handler)

        async def enqueue_only() -> None:
            await logger.error("stuck")

        asyncio.run(enqueue_only())
        assert manager.flush_sync(timeout=0.2) is False
        assert handler.messages == []

        # the record survived and is delivered by a later drain
        handler.release.set()
        assert manager.flush_sync(timeout=5.0) is True
        assert handler.messages == ["stuck"]
        asyncio.run(manager.shutdown())

    def test_raises_inside_running_loop(self) -> None:
        manager, _ = make_manager_with_recorder()

        async def inside() -> None:
            with pytest.raises(RuntimeError, match="running event loop"):
                manager.flush_sync(timeout=1.0)

        asyncio.run(inside())
        asyncio.run(manager.shutdown())

    def test_submits_to_live_loop_in_other_thread(self) -> None:
        manager, logger = make_manager_with_recorder()
        handler = logger.handlers[0]
        result: List[bool] = []

        async def main() -> None:
            await logger.error("from the loop thread")
            thread = threading.Thread(
                target=lambda: result.append(manager.flush_sync(timeout=5.0))
            )
            thread.start()
            while thread.is_alive():
                await asyncio.sleep(0.001)
            thread.join()

        asyncio.run(main())
        assert result == [True]
        assert handler.messages == ["from the loop thread"]
        asyncio.run(manager.shutdown())

    def test_module_level_wrapper_uses_global_manager(self) -> None:
        handler = RecordingHandler()
        root = aiologging._logger_manager.root
        root.addHandler(handler)

        async def enqueue_only() -> None:
            await aiologging.error("last words")

        asyncio.run(enqueue_only())
        assert aiologging.flush_sync(timeout=5.0) is True
        assert handler.messages == ["last words"]


class TestBoundedFlushAndShutdown:
    def test_flush_timeout_raises(self) -> None:
        handler = BlockingHandler()
        manager, logger = make_manager_with_recorder(handler)

        async def scenario() -> None:
            await logger.error("stuck")
            with pytest.raises(asyncio.TimeoutError):
                await manager.flush(timeout=0.2)

        asyncio.run(scenario())
        handler.release.set()
        asyncio.run(manager.shutdown())

    def test_shutdown_timeout_completes_teardown(self) -> None:
        handler = BlockingHandler()
        manager, logger = make_manager_with_recorder(handler)

        async def scenario() -> None:
            await logger.error("stuck")
            # must not hang: the grace period expires, teardown runs
            await manager.shutdown(timeout=0.2)

        asyncio.run(scenario())
        assert manager._consumer_task is None
        assert manager.getLogger("app").handlers == []


class TestAtexitDrain:
    def test_drains_by_default(self, capsys: Any) -> None:
        handler = RecordingHandler()
        aiologging._logger_manager.root.addHandler(handler)

        async def enqueue_only() -> None:
            await aiologging.error("exit words")

        asyncio.run(enqueue_only())
        _drain_at_exit()

        assert handler.messages == ["exit words"]
        assert "never delivered" not in capsys.readouterr().err

    def test_disabled_drain_only_warns(self, capsys: Any) -> None:
        handler = RecordingHandler()
        aiologging._logger_manager.root.addHandler(handler)
        aiologging.set_atexit_flush(0)
        try:

            async def enqueue_only() -> None:
                await aiologging.error("exit words")

            asyncio.run(enqueue_only())
            _drain_at_exit()

            assert handler.messages == []
            assert "never delivered" in capsys.readouterr().err
        finally:
            aiologging.set_atexit_flush(2.0)

    def test_noop_when_nothing_undelivered(self, capsys: Any) -> None:
        _drain_at_exit()
        assert capsys.readouterr().err == ""

    def test_rejects_negative_timeout(self) -> None:
        with pytest.raises(ValueError):
            aiologging.set_atexit_flush(-1.0)

    def test_basic_config_forwards_atexit_flush(self) -> None:
        aiologging.basicConfig(atexit_flush=0.5)
        try:
            from aiologging import logger as logger_module

            assert logger_module._atexit_flush_timeout == 0.5
        finally:
            aiologging.set_atexit_flush(2.0)
