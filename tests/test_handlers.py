"""
Tests for async handlers.
"""

import asyncio
import logging
import pytest
import sys
from io import StringIO
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch, ANY

from aiologging.handlers.base import AsyncHandler, BufferedAsyncHandler
from aiologging.handlers.stream import (
    AsyncStreamHandler,
    AsyncStandardStreamHandler,
)
from aiologging.exceptions import HandlerError, FormatterError


# Concrete implementations for testing abstract classes
class AsyncHandlerImpl(AsyncHandler):
    """Concrete implementation of AsyncHandler for testing."""
    async def _emit(
        self, record: logging.LogRecord, formatted_message: str
    ) -> None:
        """Mock implementation for testing."""
        pass


class BufferedAsyncHandlerImpl(BufferedAsyncHandler):
    """Concrete implementation of BufferedAsyncHandler for testing."""
    async def _emit(
        self, record: logging.LogRecord, formatted_message: str
    ) -> None:
        """Mock implementation for testing."""
        pass

    async def flush(self, records: List[logging.LogRecord]) -> None:
        """Mock implementation for testing."""
        pass


class AsyncHandlerImplWithError(AsyncHandler):
    """Concrete implementation of AsyncHandler for testing error handling."""
    async def _emit(
        self, record: logging.LogRecord, formatted_message: str
    ) -> None:
        """Mock implementation that raises an error."""
        raise Exception("Emit error")


class TestAsyncHandler:
    """Test cases for AsyncHandler base class."""

    def test_handler_initialization(self) -> None:
        """Test handler initialization."""
        handler = AsyncHandlerImpl(level=logging.INFO)
        assert handler.level == logging.INFO
        assert handler.formatter is None
        assert handler.filters == []
        assert handler.error_handler is None
        assert handler._closed is False

    def test_handler_with_parameters(self) -> None:
        """Test handler initialization with parameters."""
        formatter = MagicMock()
        filter_mock = MagicMock()
        error_handler = AsyncMock()

        handler = AsyncHandlerImpl(
            level=logging.WARNING,
            formatter=formatter,
            filters=[filter_mock],
            error_handler=error_handler,
        )

        assert handler.level == logging.WARNING
        assert handler.formatter == formatter
        assert handler.filters == [filter_mock]
        assert handler.error_handler == error_handler

    def test_set_level(self) -> None:
        """Test setting handler level."""
        handler = AsyncHandlerImpl()
        handler.setLevel(logging.DEBUG)
        assert handler.level == logging.DEBUG

    def test_filter_management(self) -> None:
        """Test adding and removing filters."""
        handler = AsyncHandlerImpl()
        filter1 = MagicMock()
        filter2 = MagicMock()

        handler.addFilter(filter1)
        assert filter1 in handler.filters

        handler.addFilter(filter2)
        assert len(handler.filters) == 2

        handler.removeFilter(filter1)
        assert filter1 not in handler.filters
        assert filter2 in handler.filters

    def test_formatter_management(self) -> None:
        """Test setting formatter."""
        handler = AsyncHandlerImpl()
        formatter = MagicMock()

        handler.setFormatter(formatter)
        assert handler.formatter == formatter

    def test_format_without_formatter(self) -> None:
        """Test formatting without a formatter."""
        handler = AsyncHandlerImpl()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = handler.format(record)
        assert formatted == "Test message"

    def test_format_with_formatter(self) -> None:
        """Test formatting with a formatter."""
        handler = AsyncHandlerImpl()
        formatter = MagicMock()
        formatter.format.return_value = "Formatted message"
        handler.setFormatter(formatter)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = handler.format(record)
        assert formatted == "Formatted message"
        formatter.format.assert_called_once_with(record)

    def test_format_with_formatter_error(self) -> None:
        """Test formatting with a formatter that raises an exception."""
        handler = AsyncHandlerImpl()
        formatter = MagicMock()
        formatter.format.side_effect = Exception("Format error")
        handler.setFormatter(formatter)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        with pytest.raises(FormatterError):
            handler.format(record)

    def test_filter_application(self) -> None:
        """Test filter application."""
        handler = AsyncHandlerImpl()

        # Create filters
        filter1 = MagicMock()
        filter1.filter.return_value = True
        filter2 = MagicMock()
        filter2.filter.return_value = False

        handler.addFilter(filter1)
        handler.addFilter(filter2)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Should return False because filter2 rejects
        assert handler.filter(record) is False
        filter1.filter.assert_called_once_with(record)
        filter2.filter.assert_called_once_with(record)

    def test_is_enabled_for(self) -> None:
        """Test level checking."""
        handler = AsyncHandlerImpl(level=logging.INFO)

        assert handler.isEnabledFor(logging.INFO) is True
        assert handler.isEnabledFor(logging.WARNING) is True
        assert handler.isEnabledFor(logging.DEBUG) is False

    @pytest.mark.asyncio
    async def test_handle_closed_handler(self) -> None:
        """Test handling with closed handler."""
        handler = AsyncHandlerImpl()
        await handler.close()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        with pytest.raises(HandlerError):
            await handler.handle(record)

    @pytest.mark.asyncio
    async def test_handle_filtered_record(self) -> None:
        """Test handling a filtered record."""
        handler = AsyncHandlerImpl()

        # Create a filter that rejects all records
        filter_mock = MagicMock()
        filter_mock.filter.return_value = False
        handler.addFilter(filter_mock)

        # Create a mock emit method
        handler.emit = AsyncMock()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        await handler.handle(record)

        # Should not call emit because record is filtered
        handler.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_disabled_level(self) -> None:
        """Test handling a record with disabled level."""
        handler = AsyncHandlerImpl(level=logging.WARNING)
        handler.emit = AsyncMock()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        await handler.handle(record)

        # Should not call emit because level is too low
        handler.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_with_error_handler(self) -> None:
        """Test handling with error handler."""
        handler = AsyncHandlerImplWithError()
        error_handler = AsyncMock()
        handler.error_handler = error_handler

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        await handler.handle(record)
        error_handler.assert_called_once_with(record, ANY)

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing handler."""
        handler = AsyncHandlerImpl()
        await handler.close()
        assert handler._closed is True


class TestBufferedAsyncHandler:
    """Test cases for BufferedAsyncHandler."""

    def test_buffered_handler_initialization(self) -> None:
        """Test buffered handler initialization."""
        handler = BufferedAsyncHandlerImpl(
            buffer_size=50,
            flush_interval=2.0,
            auto_flush=False,
        )

        assert handler.buffer_size == 50
        assert handler.flush_interval == 2.0
        assert handler.auto_flush is False
        assert handler._buffer == []
        assert handler._flush_task is None

    @pytest.mark.asyncio
    async def test_buffered_handler_add_to_buffer(self) -> None:
        """Test adding records to buffer."""
        handler = BufferedAsyncHandlerImpl(buffer_size=2, auto_flush=False)
        handler.flush = AsyncMock()

        record1, record2 = self._create_test_records("Message 1", "Message 2")

        await handler.handle(record1)
        assert len(handler._buffer) == 1
        handler.flush.assert_not_called()

        await handler.handle(record2)
        assert len(handler._buffer) == 0  # Should be flushed
        handler.flush.assert_called_once_with([record1, record2])

    def _create_test_records(self, *messages: str) -> List[logging.LogRecord]:
        """Create test log records with the given messages."""
        records = []
        for msg in messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=msg,
                args=(),
                exc_info=None,
            )
            records.append(record)
        return records

    @pytest.mark.asyncio
    async def test_buffered_handler_auto_flush(self) -> None:
        """Test auto-flush functionality."""
        handler = BufferedAsyncHandlerImpl(
            buffer_size=10,
            flush_interval=0.1,  # Short interval for testing
            auto_flush=True,
        )
        handler.flush = AsyncMock()

        record = self._create_test_records("Test message")[0]

        await handler.handle(record)

        # Wait for auto-flush
        await asyncio.sleep(0.2)

        handler.flush.assert_called_once_with([record])

    @pytest.mark.asyncio
    async def test_buffered_handler_close(self) -> None:
        """Test closing buffered handler."""
        handler = BufferedAsyncHandlerImpl(auto_flush=True)
        handler.flush = AsyncMock()

        record = self._create_test_records("Test message")[0]

        await handler.handle(record)
        await handler.close()

        # Should flush remaining records
        handler.flush.assert_called_once_with([record])
        assert handler._closed is True


class TestAsyncStreamHandler:
    """Test cases for AsyncStreamHandler."""

    def test_stream_handler_initialization(self) -> None:
        """Test stream handler initialization."""
        stream = StringIO()
        handler = AsyncStreamHandler(stream)

        assert handler.stream == stream
        assert handler.encoding == "utf-8"
        assert handler.errors == "strict"
        assert handler._is_async_stream is False

    def test_stream_handler_with_async_stream(self) -> None:
        """Test stream handler with async stream."""
        async_stream = MagicMock()
        handler = AsyncStreamHandler(async_stream)

        assert handler.stream == async_stream
        assert handler._is_async_stream is True

    @pytest.mark.asyncio
    async def test_stream_handler_emit_sync_stream(self) -> None:
        """Test emitting to sync stream."""
        stream = StringIO()
        handler = AsyncStreamHandler(stream)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        await handler.emit(record)

        output = stream.getvalue()
        assert "Test message\n" in output

    @pytest.mark.asyncio
    async def test_stream_handler_emit_async_stream(self) -> None:
        """Test emitting to async stream."""
        async_stream = AsyncMock()
        async_stream.write = AsyncMock()
        async_stream.drain = AsyncMock()

        handler = AsyncStreamHandler(async_stream)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        await handler.emit(record)

        async_stream.write.assert_called_once()
        async_stream.drain.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_handler_emit_error(self) -> None:
        """Test stream handler emit error."""
        stream = MagicMock()
        stream.write.side_effect = Exception("Write error")

        handler = AsyncStreamHandler(stream)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # The handler should handle the error internally and write to stderr
        with patch("sys.stderr") as stderr:
            await handler.emit(record)
            # Should write error to stderr
            stderr.write.assert_called()

    @pytest.mark.asyncio
    async def test_stream_handler_close(self) -> None:
        """Test closing stream handler."""
        stream = MagicMock()
        stream.close = MagicMock()

        handler = AsyncStreamHandler(stream)
        await handler.close()

        stream.close.assert_called_once()


class TestAsyncStandardStreamHandler:
    """Test cases for AsyncStandardStreamHandler."""

    def test_stdout_handler(self) -> None:
        """Test stdout handler creation."""
        handler = AsyncStandardStreamHandler("stdout")
        assert handler.stream_name == "stdout"
        assert handler.stream == sys.stdout

    def test_stderr_handler(self) -> None:
        """Test stderr handler creation."""
        handler = AsyncStandardStreamHandler("stderr")
        assert handler.stream_name == "stderr"
        assert handler.stream == sys.stderr

    def test_invalid_stream_name(self) -> None:
        """Test invalid stream name."""
        with pytest.raises(HandlerError):
            AsyncStandardStreamHandler("invalid")


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_create_stream_handler(self) -> None:
        """Test creating stream handler."""
        from aiologging import create_stream_handler

        handler = create_stream_handler(level=logging.DEBUG)
        assert isinstance(handler, AsyncStreamHandler)
        assert handler.level == logging.DEBUG


if __name__ == "__main__":
    pytest.main([__file__])
