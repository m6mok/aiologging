"""
Tests for the AsyncLogger class.
"""

import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import aiologging
from aiologging import AsyncLogger, getLogger, getLoggerContext
from aiologging.exceptions import ContextError
from aiologging.handlers.stream import AsyncStreamHandler


class TestAsyncLogger:
    """Test cases for AsyncLogger class."""

    def test_logger_initialization(self) -> None:
        """Test logger initialization."""
        logger = AsyncLogger("test_logger")
        assert logger.name == "test_logger"
        assert logger.level == logging.NOTSET
        assert logger.handlers == []
        assert logger.propagate is True
        assert logger.disabled is False
        assert logger._closed is False

    def test_logger_with_parameters(self) -> None:
        """Test logger initialization with parameters."""
        handler = AsyncStreamHandler()
        logger = AsyncLogger(
            "test_logger",
            level=logging.INFO,
            handlers=[handler],
            propagate=False,
            disabled=True,
        )
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert logger.handlers == [handler]
        assert logger.propagate is False
        assert logger.disabled is True

    def test_set_level(self) -> None:
        """Test setting logger level."""
        logger = AsyncLogger("test_logger")
        logger.setLevel(logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_get_effective_level(self) -> None:
        """Test getting effective level."""
        logger = AsyncLogger("test_logger")
        assert logger.getLevel() == logging.WARNING  # Default level

        logger.setLevel(logging.INFO)
        assert logger.getLevel() == logging.INFO

        # Test with parent
        parent = AsyncLogger("parent")
        parent.setLevel(logging.ERROR)
        logger.setParent(parent)
        logger.level = logging.NOTSET
        assert logger.getLevel() == logging.ERROR

    def test_is_enabled_for(self) -> None:
        """Test level checking."""
        logger = AsyncLogger("test_logger")
        logger.setLevel(logging.INFO)

        assert logger.isEnabledFor(logging.INFO) is True
        assert logger.isEnabledFor(logging.WARNING) is True
        assert logger.isEnabledFor(logging.DEBUG) is False

        # Test disabled logger
        logger.disable()
        assert logger.isEnabledFor(logging.CRITICAL) is False

    def test_handler_management(self) -> None:
        """Test adding and removing handlers."""
        logger = AsyncLogger("test_logger")
        handler1 = AsyncStreamHandler()
        handler2 = AsyncStreamHandler()

        logger.addHandler(handler1)
        assert handler1 in logger.handlers

        logger.addHandler(handler2)
        assert len(logger.handlers) == 2

        logger.removeHandler(handler1)
        assert handler1 not in logger.handlers
        assert handler2 in logger.handlers

    def test_filter_management(self) -> None:
        """Test adding and removing filters."""
        logger = AsyncLogger("test_logger")
        filter1 = MagicMock()
        filter2 = MagicMock()

        logger.addFilter(filter1)
        assert filter1 in logger._filters

        logger.addFilter(filter2)
        assert len(logger._filters) == 2

        logger.removeFilter(filter1)
        assert filter1 not in logger._filters
        assert filter2 in logger._filters

    def test_formatter_management(self) -> None:
        """Test setting and getting formatter."""
        logger = AsyncLogger("test_logger")
        formatter = MagicMock()

        logger.setFormatter(formatter)
        assert logger.getFormatter() == formatter

    def test_parent_child_relationship(self) -> None:
        """Test parent-child logger relationships."""
        parent = AsyncLogger("parent")
        child = AsyncLogger("parent.child")

        child.setParent(parent)
        assert child.getParent() == parent
        assert parent._children["parent.child"] == child
        assert child.propagateToParent() is True

        child.propagate = False
        assert child.propagateToParent() is False

    @pytest.mark.asyncio
    async def test_log_methods(self) -> None:
        """Test all log methods."""
        logger = AsyncLogger("test_logger")
        logger.setLevel(logging.DEBUG)

        handler = AsyncMock(spec=AsyncStreamHandler)
        handler.handle = AsyncMock()
        logger.addHandler(handler)

        # Test all log levels
        await self._test_log_levels(logger)

        # Test aliases
        await self._test_log_aliases(logger)

        # Test exception logging
        await self._test_exception_logging(logger)

        # Verify all calls were made
        assert handler.handle.call_count == 8

    async def _test_log_levels(self, logger: AsyncLogger) -> None:
        """Test all log levels."""
        await logger.debug("Debug message")
        await logger.info("Info message")
        await logger.warning("Warning message")
        await logger.error("Error message")
        await logger.critical("Critical message")

    async def _test_log_aliases(self, logger: AsyncLogger) -> None:
        """Test log method aliases."""
        await logger.warn("Warning message (alias)")
        await logger.fatal("Fatal message (alias)")

    async def _test_exception_logging(self, logger: AsyncLogger) -> None:
        """Test exception logging."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            await logger.exception("Exception occurred")

    @pytest.mark.asyncio
    async def test_log_with_args_and_kwargs(self) -> None:
        """Test logging with format arguments and keyword arguments."""
        logger = AsyncLogger("test_logger")
        logger.setLevel(logging.DEBUG)

        handler = AsyncMock(spec=AsyncStreamHandler)
        handler.handle = AsyncMock()
        logger.addHandler(handler)

        await logger.info("Message with %s", "argument")
        await logger.error("Error: %s", "details", extra={"key": "value"})

        assert handler.handle.call_count == 2

        # Check the log records
        records = [call.args[0] for call in handler.handle.call_args_list]
        assert records[0].getMessage() == "Message with argument"
        assert records[1].getMessage() == "Error: details"
        assert hasattr(records[1], "key")
        assert records[1].key == "value"

    @pytest.mark.asyncio
    async def test_disabled_logger(self) -> None:
        """Test that disabled logger doesn't process records."""
        logger = AsyncLogger("test_logger")
        logger.disable()

        handler = AsyncMock(spec=AsyncStreamHandler)
        handler.handle = AsyncMock()
        logger.addHandler(handler)

        await logger.info("This should not be logged")

        handler.handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_closed_logger_error(self) -> None:
        """Test that closed logger doesn't process records."""
        logger = AsyncLogger("test_logger")
        await logger.close()

        handler = AsyncMock(spec=AsyncStreamHandler)
        handler.handle = AsyncMock()
        logger.addHandler(handler)

        await logger.info("This should not be logged")
        handler.handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test logger as context manager."""
        logger = AsyncLogger("test_logger")

        async with logger as ctx:
            assert ctx is logger
            assert logger._closed is False

        assert logger._closed is True

    @pytest.mark.asyncio
    async def test_context_manager_error_on_closed(self) -> None:
        """Test context manager error on closed logger."""
        logger = AsyncLogger("test_logger")
        await logger.close()

        with pytest.raises(ContextError):
            async with logger:
                pass

    @pytest.mark.asyncio
    async def test_filter_application(self) -> None:
        """Test filter application."""
        logger = AsyncLogger("test_logger")
        logger.setLevel(logging.DEBUG)  # Set level to DEBUG to allow INFO messages

        # Create a filter that rejects all records
        filter_mock = MagicMock()
        filter_mock.filter.return_value = False
        logger.addFilter(filter_mock)

        handler = AsyncMock(spec=AsyncStreamHandler)
        handler.handle = AsyncMock()
        logger.addHandler(handler)

        await logger.info("This should be filtered")

        handler.handle.assert_not_called()
        filter_mock.filter.assert_called()

    @pytest.mark.asyncio
    async def test_handler_error_handling(self) -> None:
        """Test error handling in handlers."""
        logger = AsyncLogger("test_logger")
        logger.setLevel(logging.DEBUG)  # Set level to DEBUG to allow INFO messages

        # Create a handler that raises an exception
        handler = AsyncMock(spec=AsyncStreamHandler)
        handler.handle.side_effect = Exception("Handler error")
        logger.addHandler(handler)

        # This should not raise an exception, but should log to stderr
        with patch("sys.stderr") as stderr:
            await logger.info("Test message")
            # Check that stderr.write was called with error message
            stderr.write.assert_called()
            # Get the call arguments and verify it contains error info
            call_args = stderr.write.call_args[0][0]
            assert "Error in handler" in call_args

    @pytest.mark.asyncio
    async def test_log_record_creation(self) -> None:
        """Test log record creation with proper attributes."""
        logger = AsyncLogger("test_logger")
        logger.setLevel(logging.DEBUG)  # Set level to DEBUG to allow INFO messages

        handler = AsyncMock(spec=AsyncStreamHandler)
        handler.handle = AsyncMock()
        logger.addHandler(handler)

        await logger.info("Test message", extra={"custom_attr": "custom_value"})

        # Check that handle was called
        handler.handle.assert_called_once()
        # Get the record from the call
        record = handler.handle.call_args[0][0]
        assert record.name == "test_logger"
        assert record.levelno == logging.INFO
        assert record.levelname == "INFO"
        assert record.getMessage() == "Test message"
        assert hasattr(record, "custom_attr")
        assert record.custom_attr == "custom_value"

    def test_logger_repr(self) -> None:
        """Test logger string representation."""
        logger = AsyncLogger("test_logger")
        logger.setLevel(logging.INFO)
        handler = AsyncStreamHandler()
        logger.addHandler(handler)

        repr_str = repr(logger)
        assert "AsyncLogger" in repr_str
        assert "test_logger" in repr_str
        assert "INFO" in repr_str
        assert "handlers=1" in repr_str


class TestLoggerManager:
    """Test cases for AsyncLoggerManager."""

    def test_get_logger_singleton(self) -> None:
        """Test that getLogger returns the same instance for the same name."""
        logger1 = getLogger("test_manager")
        logger2 = getLogger("test_manager")

        assert logger1 is logger2

    def test_get_logger_hierarchy(self) -> None:
        """Test logger hierarchy creation."""
        parent = getLogger("parent")
        child = getLogger("parent.child")
        grandchild = getLogger("parent.child.grandchild")

        assert child.getParent() == parent
        assert grandchild.getParent() == child
        assert parent._children["parent.child"] == child
        assert child._children["parent.child.grandchild"] == grandchild

    def test_get_root_logger(self) -> None:
        """Test getting the root logger."""
        root = aiologging.getRootLogger()
        assert root.name == "root"

        # Should return the same instance
        root2 = aiologging.getRootLogger()
        assert root is root2

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Test logger manager shutdown."""
        logger = getLogger("test_shutdown")
        handler = AsyncMock(spec=AsyncStreamHandler)
        handler.close = AsyncMock()
        logger.addHandler(handler)

        await aiologging.shutdown()

        handler.close.assert_called_once()
        assert logger._closed is True


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @pytest.mark.asyncio
    async def test_get_logger_context(self) -> None:
        """Test getLoggerContext function."""
        async with getLoggerContext("test_context") as logger:
            assert logger.name == "test_context"
            assert logger._closed is False

        assert logger._closed is True

    @pytest.mark.asyncio
    async def test_log_async_function(self) -> None:
        """Test log_async convenience function."""
        with patch("aiologging.logger.getLoggerContext") as mock_context:
            mock_logger = AsyncMock()
            mock_context.return_value.__aenter__.return_value = mock_logger

            await aiologging.log_async("test", logging.INFO, "Test message")

            # Check that getLoggerContext was called with the right name
            mock_context.assert_called_once_with("test")
            # Check that the logger's log method was called
            mock_logger.log.assert_called_once_with(logging.INFO, "Test message")

    def test_basic_config(self) -> None:
        """Test basicConfig function."""
        handler = AsyncStreamHandler()

        aiologging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
            handlers=[handler],
        )

        root = aiologging.getRootLogger()
        assert root.getLevel() == logging.DEBUG
        assert handler in root.handlers
        assert handler.formatter is not None


if __name__ == "__main__":
    pytest.main([__file__])
