"""
Loop-affinity of handler resources.

The manager's consumer and dispatchers already survive an event-loop
change (see test_logger.py); these tests cover the handler-owned
resources that are bound to the loop they were created on: LazyLock's
inner lock, the HTTP client session, the aiofiles file wrapper and the
buffered handler's auto-flush task. Each must be recreated — never
reused — when records flow again on a fresh loop, which is exactly
what an emergency drain on a private loop (flush_sync) relies on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import httpx
import pytest

from aiologging import AsyncLoggerManager
from aiologging.types import BatchConfig
from aiologging.handlers import AsyncFileHandler, AsyncHttpJsonHandler
from aiologging.utils import LazyLock


def make_record(message: str) -> logging.LogRecord:
    """Create an INFO record for direct handler calls."""
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


class TestLazyLock:
    def test_lock_is_recreated_on_loop_change(self) -> None:
        lock = LazyLock()
        inner: List[Optional[asyncio.Lock]] = []

        async def use() -> None:
            async with lock:
                inner.append(lock._lock)

        asyncio.run(use())
        asyncio.run(use())

        assert inner[0] is not None
        assert inner[1] is not None
        assert inner[0] is not inner[1]

    @pytest.mark.asyncio
    async def test_lock_is_reused_on_same_loop(self) -> None:
        lock = LazyLock()
        async with lock:
            first = lock._lock
        async with lock:
            assert lock._lock is first


class TestHttpSessionAffinity:
    def _make_handler(
        self, requests: List[httpx.Request]
    ) -> AsyncHttpJsonHandler:
        """Handler whose real session factory yields a mock transport."""

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200)

        handler = AsyncHttpJsonHandler(
            "https://example.com/api/logs",
            backend="httpx",
            batch_config=BatchConfig(flush_interval=3600.0),
        )
        handler._create_session = (  # type: ignore[method-assign]
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )
        return handler

    def test_session_is_recreated_and_delivers_on_new_loop(self) -> None:
        requests: List[httpx.Request] = []
        handler = self._make_handler(requests)
        sessions: List[object] = []

        async def emit_and_flush(message: str) -> None:
            await handler.handle(make_record(message))
            await handler.force_flush()
            sessions.append(handler._session)

        asyncio.run(emit_and_flush("from loop 1"))
        asyncio.run(emit_and_flush("from loop 2"))

        assert len(requests) == 2
        assert sessions[0] is not sessions[1]
        asyncio.run(handler.close())

    def test_injected_session_is_left_untouched(self) -> None:
        requests: List[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200)

        handler = AsyncHttpJsonHandler(
            "https://example.com/api/logs",
            backend="httpx",
            batch_config=BatchConfig(flush_interval=3600.0),
        )
        injected = httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        )
        handler._session = injected

        async def emit_and_flush(message: str) -> None:
            await handler.handle(make_record(message))
            await handler.force_flush()

        asyncio.run(emit_and_flush("one"))
        assert handler._session is injected
        assert len(requests) == 1
        asyncio.run(handler.close())

    def test_close_on_new_loop_abandons_dead_loop_session(self) -> None:
        requests: List[httpx.Request] = []
        handler = self._make_handler(requests)

        async def emit_and_flush() -> None:
            await handler.handle(make_record("x"))
            await handler.force_flush()

        asyncio.run(emit_and_flush())
        assert handler._session is not None
        # Closing on a different loop must not raise and must drop
        # the session created on the (now dead) first loop
        asyncio.run(handler.close())
        assert handler._session is None


class TestFileAffinity:
    def test_file_is_reopened_and_appends_on_new_loop(
        self, tmp_path: object
    ) -> None:
        logfile = tmp_path / "app.log"  # type: ignore[operator]
        handler = AsyncFileHandler(str(logfile), mode="w", delay=True)
        files: List[object] = []

        async def emit(message: str) -> None:
            await handler.handle(make_record(message))
            files.append(handler._file)

        asyncio.run(emit("from loop 1"))
        asyncio.run(emit("from loop 2"))
        asyncio.run(handler.close())

        assert files[0] is not files[1]
        content = logfile.read_text()  # type: ignore[attr-defined]
        assert "from loop 1" in content
        assert "from loop 2" in content


class TestAutoFlushTaskAffinity:
    def test_auto_flush_task_is_restarted_on_new_loop(self) -> None:
        requests: List[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200)

        handler = AsyncHttpJsonHandler(
            "https://example.com/api/logs",
            backend="httpx",
            batch_config=BatchConfig(flush_interval=3600.0),
        )
        handler._create_session = (  # type: ignore[method-assign]
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )
        tasks: List[object] = []

        async def emit(message: str) -> None:
            await handler.handle(make_record(message))
            tasks.append(handler._flush_task)

        asyncio.run(emit("one"))
        asyncio.run(emit("two"))

        assert tasks[0] is not None
        assert tasks[1] is not None
        assert tasks[0] is not tasks[1]
        asyncio.run(handler.close())
        assert handler._flush_task is None


class TestEndToEndAcrossLoops:
    def test_pipeline_delivers_over_http_after_loop_change(self) -> None:
        """The motivating case: log, loop dies, log + flush again."""
        requests: List[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200)

        manager = AsyncLoggerManager()
        handler = AsyncHttpJsonHandler(
            "https://example.com/api/logs",
            backend="httpx",
            batch_config=BatchConfig(flush_interval=3600.0),
        )
        handler._create_session = (  # type: ignore[method-assign]
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )
        logger = manager.getLogger("app")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        async def first_loop() -> None:
            await logger.error("boom on loop 1")
            await manager.flush()

        async def second_loop() -> None:
            await logger.error("boom on loop 2")
            await manager.flush()

        asyncio.run(first_loop())
        asyncio.run(second_loop())

        assert len(requests) == 2
        asyncio.run(manager.shutdown())
