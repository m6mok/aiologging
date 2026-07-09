"""
Regression tests for fixed bugs.

Each test here pins down a specific bug that was found and fixed:
- exc_info=True was passed to LogRecord as-is and broke formatters
- sync text streams (StringIO, real files) matched the async writer
  protocol and were written with a spurious await + retries
- closing a stream handler closed sys.stderr / sys.stdout
- propagation to an already closed parent logger raised
- AiologgingError mutated the caller's details dictionary
- GC of consumer/worker coroutines stranded on a closed loop spilled
  noise into stderr (never-awaited RuntimeWarning, ContextVar reset
  ValueError, "Exception ignored" from queue.get cleanup)
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import sys
import warnings
from typing import Any, List

from aiologging import AsyncHandler, AsyncLogger, AsyncLoggerManager
from aiologging.exceptions import AiologgingError
from aiologging.handlers.stream import AsyncStreamHandler


class TestExcInfoNormalization:
    """exc_info must be normalized the way standard logging does it."""

    async def test_exc_info_true_renders_traceback(self) -> None:
        buf = io.StringIO()
        handler = AsyncStreamHandler(
            buf, formatter=logging.Formatter("%(message)s")
        )
        logger = AsyncLogger(
            "test.excinfo",
            level=logging.DEBUG,
            handlers=[handler],
            propagate=False,
            delivery="await",
        )

        try:
            raise ValueError("boom")
        except ValueError:
            await logger.exception("failed")

        out = buf.getvalue()
        assert "failed" in out
        assert "Traceback" in out
        assert "ValueError: boom" in out

    async def test_exc_info_exception_instance(self) -> None:
        buf = io.StringIO()
        handler = AsyncStreamHandler(
            buf, formatter=logging.Formatter("%(message)s")
        )
        logger = AsyncLogger(
            "test.excinfo.instance",
            level=logging.DEBUG,
            handlers=[handler],
            propagate=False,
            delivery="await",
        )

        error = RuntimeError("direct instance")
        await logger.error("failed", exc_info=error)

        assert "RuntimeError: direct instance" in buf.getvalue()


class TestSyncStreamDispatch:
    """Sync text streams must not be treated as async writers."""

    async def test_stringio_written_exactly_once(self) -> None:
        buf = io.StringIO()
        handler = AsyncStreamHandler(buf)
        logger = AsyncLogger(
            "test.stringio",
            level=logging.DEBUG,
            handlers=[handler],
            propagate=False,
            delivery="await",
        )

        await logger.info("single write")

        assert buf.getvalue().count("single write") == 1


class TestStandardStreamsSurviveClose:
    """Closing a handler must never close the interpreter's std streams."""

    async def test_stderr_not_closed(self) -> None:
        handler = AsyncStreamHandler()  # defaults to sys.stderr
        await handler.close()
        assert not sys.stderr.closed

    async def test_stdout_not_closed(self) -> None:
        handler = AsyncStreamHandler(sys.stdout)
        await handler.close()
        assert not sys.stdout.closed


class TestClosedParentPropagation:
    """Logging via a child of a closed parent must not raise."""

    async def test_child_created_after_parent_closed(self) -> None:
        # Closing a parent cascades to already registered children, so
        # the scenario is a child attached to an already closed parent
        # (e.g. getLogger("app.oneoff") after "app" was closed)
        parent = AsyncLogger("test.parent", level=logging.DEBUG)
        await parent.close()

        child = AsyncLogger(
            "test.parent.child", level=logging.DEBUG, delivery="await"
        )
        child.parent = parent

        buf = io.StringIO()
        child.addHandler(AsyncStreamHandler(buf))

        await child.info("still works")

        assert "still works" in buf.getvalue()


class TestExceptionDetailsNotMutated:
    """AiologgingError must not mutate the caller's details dict."""

    def test_details_dict_untouched(self) -> None:
        details = {"key": "value"}
        error = AiologgingError("msg", details=details, extra="added")

        assert details == {"key": "value"}
        assert error.details == {"key": "value", "extra": "added"}


class TestRealStreamWriter:
    """A real asyncio.StreamWriter (sync write + async drain) must work."""

    async def test_write_over_tcp(self) -> None:
        received: list[bytes] = []

        async def on_connect(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            received.append(await reader.read(1024))
            writer.close()

        server = await asyncio.start_server(on_connect, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        _, writer = await asyncio.open_connection("127.0.0.1", port)

        handler = AsyncStreamHandler(writer)
        logger = AsyncLogger(
            "test.tcp",
            level=logging.DEBUG,
            handlers=[handler],
            propagate=False,
            delivery="await",
        )
        await logger.info("over tcp")

        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
        server.close()
        await server.wait_closed()

        assert b"over tcp" in b"".join(received)


class _SlowSink(AsyncHandler):
    """Sink slow enough to leave a backlog when the loop dies."""

    async def _emit(self, record: logging.LogRecord, message: str) -> None:
        await asyncio.sleep(0.001)


class TestLoopDeathTeardownIsQuiet:
    """
    Closing an event loop with the consumer and handler workers still
    alive, then resuming on a fresh loop, must not spill cosmetic
    noise when the stranded coroutines are GC-finalized: no
    ``RuntimeWarning: coroutine ... was never awaited`` for a worker
    that never got its first step, no ``ValueError: token was created
    in a different Context`` from the ``_IN_CONSUMER`` reset, and no
    ``Exception ignored`` from ``queue.get()`` touching the closed
    loop.
    """

    def test_gc_after_loop_death_is_silent(self) -> None:
        manager = AsyncLoggerManager()
        logger = manager.getLogger("test.loopdeath")
        logger.setLevel(logging.INFO)
        logger.addHandler(_SlowSink())

        async def produce(count: int) -> None:
            for i in range(count):
                await logger.info("msg %d", i)

        async def rescue() -> None:
            await produce(1)
            await manager.flush()

        unraisable: List[Any] = []
        original_hook = sys.unraisablehook
        sys.unraisablehook = unraisable.append
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                # Kill the loop with the queue undrained and the
                # consumer/worker tasks still alive
                dying = asyncio.new_event_loop()
                try:
                    dying.run_until_complete(produce(200))
                finally:
                    dying.close()
                del dying
                gc.collect()

                # The rebuild on the fresh loop drops the stranded
                # tasks; their coroutines are finalized here
                asyncio.run(rescue())
                gc.collect()
        finally:
            sys.unraisablehook = original_hook

        runtime_warnings = [
            str(w.message)
            for w in caught
            if issubclass(w.category, RuntimeWarning)
        ]
        assert runtime_warnings == []
        assert [args.exc_value for args in unraisable] == []

        asyncio.run(manager.shutdown())
