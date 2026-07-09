"""
Regression tests for fixed bugs.

Each test here pins down a specific bug that was found and fixed:
- exc_info=True was passed to LogRecord as-is and broke formatters
- sync text streams (StringIO, real files) matched the async writer
  protocol and were written with a spurious await + retries
- closing a stream handler closed sys.stderr / sys.stdout
- propagation to an already closed parent logger raised
- AiologgingError mutated the caller's details dictionary
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys

from aiologging import AsyncLogger
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

        child = AsyncLogger("test.parent.child", level=logging.DEBUG)
        child.setParent(parent)

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
        )
        await logger.info("over tcp")

        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
        server.close()
        await server.wait_closed()

        assert b"over tcp" in b"".join(received)
