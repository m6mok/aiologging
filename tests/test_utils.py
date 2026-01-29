"""
Tests for the aiologging utils module.
"""

import logging
import pytest
import sys
from io import StringIO
from unittest.mock import AsyncMock, patch

from aiologging.utils import (
    log_error_to_stderr,
    handle_error_with_fallback,
    safe_format_exception,
    ErrorContext,
    create_error_handler,
)


class TestLogErrorToStderr:
    """Test cases for the log_error_to_stderr function."""

    def test_log_message_without_exception(self) -> None:
        """Test logging a message without an exception."""
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            log_error_to_stderr("Test message")
            assert stderr.getvalue() == "Test message\n"

    def test_log_message_with_exception(self) -> None:
        """Test logging a message with an exception."""
        exception = ValueError("Test error")
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            log_error_to_stderr("Test message", exception)
            output = stderr.getvalue()
            assert "Test message: Test error" in output
            assert "Traceback:" in output

    def test_log_message_with_none_exception(self) -> None:
        """Test logging a message with None exception."""
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            log_error_to_stderr("Test message", None)
            assert stderr.getvalue() == "Test message\n"


class TestHandleErrorWithFallback:
    """Test cases for the handle_error_with_fallback function."""

    @pytest.mark.asyncio
    async def test_with_error_handler(self) -> None:
        """Test error handling with a custom error handler."""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        error = ValueError("Test error")
        error_handler = AsyncMock()

        await handle_error_with_fallback(record, error, error_handler)

        error_handler.assert_called_once_with(record, error)

    @pytest.mark.asyncio
    async def test_with_failing_error_handler(self) -> None:
        """Test error handling with a failing error handler."""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        error = ValueError("Test error")
        error_handler = AsyncMock()
        error_handler.side_effect = RuntimeError("Handler error")

        with patch("sys.stderr", new_callable=StringIO) as stderr:
            await handle_error_with_fallback(record, error, error_handler)
            output = stderr.getvalue()
            assert "Error in error handler: Handler error" in output
            assert "Test error" in output

    @pytest.mark.asyncio
    async def test_without_error_handler(self) -> None:
        """Test error handling without a custom error handler."""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        error = ValueError("Test error")

        with patch("sys.stderr", new_callable=StringIO) as stderr:
            await handle_error_with_fallback(record, error)
            output = stderr.getvalue()
            assert "Error processing log record: Test error" in output

    @pytest.mark.asyncio
    async def test_with_custom_fallback_message(self) -> None:
        """Test error handling with a custom fallback message."""
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        error = ValueError("Test error")

        with patch("sys.stderr", new_callable=StringIO) as stderr:
            await handle_error_with_fallback(
                record, error, fallback_message="Custom error message"
            )
            output = stderr.getvalue()
            assert "Custom error message: Test error" in output


class TestSafeFormatException:
    """Test cases for the safe_format_exception function."""

    def test_format_none(self) -> None:
        """Test formatting None."""
        result = safe_format_exception(None)
        assert result == ""

    def test_format_exc_info_tuple(self) -> None:
        """Test formatting an exc_info tuple."""
        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()
            result = safe_format_exception(exc_info)
            assert "ValueError" in result
            assert "Test error" in result

    def test_format_exception_instance(self) -> None:
        """Test formatting an exception instance."""
        error = ValueError("Test error")
        result = safe_format_exception(error)
        assert "ValueError" in result
        assert "Test error" in result

    def test_format_other_object(self) -> None:
        """Test formatting a non-exception object."""
        obj = "not an exception"
        result = safe_format_exception(obj)
        assert result == "not an exception"

    def test_format_with_error_in_formatting(self) -> None:
        """Test handling errors during formatting."""
        # Create a mock object that causes an error during formatting
        class BadExcInfo:
            def __init__(self):
                self.__traceback__ = self

            def __getattr__(self, name):
                raise AttributeError("Bad attribute")

        bad_exc_info = BadExcInfo()
        result = safe_format_exception(bad_exc_info)
        assert "Failed to format exception" in result


class TestErrorContext:
    """Test cases for the ErrorContext class."""

    def test_initialization(self) -> None:
        """Test ErrorContext initialization."""
        context = ErrorContext("test operation")
        assert context.operation == "test operation"
        assert context.error_handler is None
        assert context.fallback_message == "Error during test operation"
        assert context.reraise is False
        assert context.record is None

    def test_initialization_with_parameters(self) -> None:
        """Test ErrorContext initialization with parameters."""
        error_handler = AsyncMock()
        context = ErrorContext(
            "test operation",
            error_handler=error_handler,
            fallback_message="Custom message",
            reraise=True
        )
        assert context.operation == "test operation"
        assert context.error_handler == error_handler
        assert context.fallback_message == "Custom message"
        assert context.reraise is True

    def test_set_record(self) -> None:
        """Test setting a record on the context."""
        context = ErrorContext("test operation")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        context.set_record(record)
        assert context.record == record

    @pytest.mark.asyncio
    async def test_context_manager_without_error(self) -> None:
        """Test context manager without an error."""
        context = ErrorContext("test operation")
        async with context as ctx:
            assert ctx is context

    @pytest.mark.asyncio
    async def test_context_manager_with_error_no_reraise(self) -> None:
        """Test context manager with an error and no reraise."""
        context = ErrorContext("test operation", reraise=False)
        error_handler = AsyncMock()
        context.error_handler = error_handler
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        context.set_record(record)

        error = ValueError("Test error")
        result = None
        try:
            async with context:
                raise error
        except Exception as e:
            result = e

        # Should not reraise the exception
        assert result is None
        error_handler.assert_called_once_with(record, error)

    @pytest.mark.asyncio
    async def test_context_manager_with_error_reraise(self) -> None:
        """Test context manager with an error and reraise."""
        context = ErrorContext("test operation", reraise=True)
        error_handler = AsyncMock()
        context.error_handler = error_handler
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        context.set_record(record)

        error = ValueError("Test error")
        result = None
        try:
            async with context:
                raise error
        except Exception as e:
            result = e

        # Should reraise the exception
        assert result is error
        error_handler.assert_called_once_with(record, error)

    @pytest.mark.asyncio
    async def test_context_manager_with_error_no_handler(self) -> None:
        """Test context manager with an error and no handler."""
        context = ErrorContext("test operation", reraise=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        context.set_record(record)

        error = ValueError("Test error")
        result = None
        try:
            async with context:
                raise error
        except Exception as e:
            result = e

        # Should not reraise the exception
        assert result is None

        # Should log to stderr
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            async with context:
                raise error
            output = stderr.getvalue()
            assert "Error during test operation: Test error" in output

    @pytest.mark.asyncio
    async def test_context_manager_with_non_exception_error(self) -> None:
        """Test context manager with a non-exception error."""
        context = ErrorContext("test operation", reraise=False)
        result = None
        try:
            async with context:
                raise ValueError("Test error")
        except Exception as e:
            result = e

        # Should not reraise the exception
        assert result is None


class TestCreateErrorHandler:
    """Test cases for the create_error_handler function."""

    def test_create_error_handler(self) -> None:
        """Test creating an error handler from a function."""
        async def my_error_handler(record, error):
            pass

        handler = create_error_handler(my_error_handler)
        assert handler is my_error_handler

    def test_create_error_handler_with_lambda(self) -> None:
        """Test creating an error handler from a lambda."""
        handler = create_error_handler(lambda record, error: None)
        assert callable(handler)


if __name__ == "__main__":
    pytest.main([__file__])
