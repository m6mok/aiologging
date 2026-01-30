"""
Enhanced async logger implementation for aiologging.

This module provides the main AsyncLogger class with full compatibility
to the standard logging module but with async methods requiring await.
It includes advanced features like performance metrics, adaptive logging,
and enhanced error handling.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from logging import LogRecord, NOTSET
from types import TracebackType
from typing import Any, Dict, List, Optional, Union, cast
from weakref import WeakSet

from .exceptions import ContextError, LoggerClosedError
from .handlers.base import AsyncHandler
from .types import (
    FilterProtocol,
    FormatterProtocol,
    RateLimiter,
    AsyncErrorHandler,
)
from .utils import handle_error_with_fallback


class AsyncLoggerMetrics:
    """Metrics collection for async loggers."""

    def __init__(self, enable_metrics: bool = False) -> None:
        self.enable_metrics = enable_metrics
        self._records_created = 0
        self._records_processed = 0
        self._records_dropped = 0
        self._errors_count = 0
        self._last_error_time = 0.0

    def increment_created(self) -> None:
        """Increment created records count."""
        if self.enable_metrics:
            self._records_created += 1

    def increment_processed(self) -> None:
        """Increment processed records count."""
        if self.enable_metrics:
            self._records_processed += 1

    def increment_dropped(self) -> None:
        """Increment dropped records count."""
        if self.enable_metrics:
            self._records_dropped += 1

    def increment_errors(self) -> None:
        """Increment errors count and update last error time."""
        if self.enable_metrics:
            self._errors_count += 1
            self._last_error_time = time.time()

    def get_metrics(self) -> dict[str, Union[int, float]]:
        """Get current metrics."""
        return {
            "records_created": self._records_created,
            "records_processed": self._records_processed,
            "records_dropped": self._records_dropped,
            "errors_count": self._errors_count,
            "last_error_time": self._last_error_time,
        }


class AsyncLogger:
    """
    Enhanced async logger with full compatibility to standard logging module.

    This class provides the same interface as the standard logging.Logger
    but with async methods that require await. It includes advanced features
    like performance metrics, adaptive logging, and enhanced error handling.

    The main usage pattern is:

    async with aiologging.getLogger("name") as logger:
        await logger.info("message")
    """

    # Class-level configuration for better performance
    _default_record_cache_size: int = 1000
    _default_filter_cache_size: int = 1000

    def __init__(
        self,
        name: str,
        level: int = NOTSET,
        handlers: Optional[List[AsyncHandler]] = None,
        propagate: bool = True,
        disabled: bool = False,
        rate_limiter: Optional[RateLimiter] = None,
        enable_metrics: bool = False,
        record_cache_size: Optional[int] = None,
        filter_cache_size: Optional[int] = None,
        error_handler: Optional[AsyncErrorHandler] = None,
    ) -> None:
        """
        Initialize the async logger with enhanced configuration options.

        Args:
            name: The logger name
            level: The logging level
            handlers: List of async handlers
            propagate: Whether to propagate to parent loggers
            disabled: Whether the logger is disabled
            rate_limiter: Optional rate limiter for the logger
            enable_metrics: Enable performance metrics collection
            record_cache_size: Size of the record cache for optimization
            filter_cache_size: Size of the filter cache for optimization
            error_handler: Optional error handler for logging errors
        """
        self.name = name
        self.level = level
        self.handlers = handlers or []
        self.propagate = propagate
        self.disabled = disabled
        self._rate_limiter = rate_limiter
        self._enable_metrics = enable_metrics
        self._record_cache_size = (
            record_cache_size or self._default_record_cache_size
        )
        self._filter_cache_size = (
            filter_cache_size or self._default_filter_cache_size
        )
        self._error_handler = error_handler

        self._parent: Optional[AsyncLogger] = None
        self._children: Dict[str, AsyncLogger] = {}
        self._filters: List[FilterProtocol] = []
        self._formatter: Optional[FormatterProtocol] = None
        self._lock = asyncio.Lock()
        self._closed = False

        # Performance optimizations
        self._record_cache: dict[int, LogRecord] = {}
        self._filter_cache: dict[int, bool] = {}

        # Metrics collection
        self._metrics = AsyncLoggerMetrics(enable_metrics)

        # Reference tracking for proper cleanup
        self._handlers_refs: WeakSet[AsyncHandler] = WeakSet(self.handlers)

    def setLevel(self, level: int) -> None:
        """Set the logging level for this logger and clear caches."""
        self.level = level
        self._filter_cache.clear()

    def getLevel(self) -> int:
        """Get the effective logging level for this logger."""
        if self.disabled:
            return logging.CRITICAL + 1

        if self.level != NOTSET:
            return self.level

        if self._parent:
            return self._parent.getLevel()

        return logging.WARNING  # Default level

    def isEnabledFor(self, level: int) -> bool:
        """Check if this logger is enabled for the given level."""
        if self.disabled:
            return False

        return level >= self.getLevel()

    def addHandler(self, handler: AsyncHandler) -> None:
        """Add an async handler to this logger and track reference."""
        if handler not in self.handlers:
            self.handlers.append(handler)
            self._handlers_refs.add(handler)
            self._filter_cache.clear()

    def removeHandler(self, handler: AsyncHandler) -> None:
        """Remove an async handler from this logger and clear caches."""
        if handler in self.handlers:
            self.handlers.remove(handler)
            self._filter_cache.clear()

    def addFilter(self, filter: FilterProtocol) -> None:
        """Add a filter to this logger and clear cache."""
        if filter not in self._filters:
            self._filters.append(filter)
            self._filter_cache.clear()

    def removeFilter(self, filter: FilterProtocol) -> None:
        """Remove a filter from this logger and clear cache."""
        if filter in self._filters:
            self._filters.remove(filter)
            self._filter_cache.clear()

    def setFormatter(self, formatter: FormatterProtocol) -> None:
        """Set the formatter for this logger."""
        self._formatter = formatter

    def getFormatter(self) -> Optional[FormatterProtocol]:
        """Get the formatter for this logger."""
        return self._formatter

    def setParent(self, parent: AsyncLogger) -> None:
        """Set the parent logger."""
        self._parent = parent
        parent._children[self.name] = self

    def getParent(self) -> Optional[AsyncLogger]:
        """Get the parent logger."""
        return self._parent

    def propagateToParent(self) -> bool:
        """Check if this logger propagates to its parent."""
        return self.propagate and self._parent is not None

    def disable(self) -> None:
        """Disable this logger."""
        self.disabled = True

    def enable(self) -> None:
        """Enable this logger."""
        self.disabled = False

    def isDisabledFor(self, level: int) -> bool:
        """Check if this logger is disabled for the given level."""
        return self.disabled or not self.isEnabledFor(level)

    async def log(
        self,
        level: int,
        msg: str,
        *args: Any,
        exc_info: Optional[
            Union[
                bool,
                tuple[
                    type[BaseException], BaseException, Optional[TracebackType]
                ],
                tuple[None, None, None],
            ]
        ] = None,
        extra: Optional[Dict[str, Any]] = None,
        stack_info: bool = False,
    ) -> None:
        """
        Log a message with the specified level and enhanced error handling.

        Args:
            level: The logging level
            msg: The message to log
            *args: Arguments for message formatting
            exc_info: Exception information to include
            extra: Extra attributes to add to the log record
            stack_info: Whether to include stack information

        Raises:
            LoggerClosedError: If the logger is closed.
        """
        if self._closed:
            return

        if self.isDisabledFor(level):
            return

        # Apply rate limiting if configured
        if self._rate_limiter and not await self._rate_limiter():
            self._metrics.increment_dropped()
            return

        # Create log record
        record = self._make_record(
            level, msg, args, exc_info, extra, stack_info
        )

        # Handle the record
        try:
            await self.handle(record)
            self._metrics.increment_processed()
        except Exception as e:
            self._metrics.increment_errors()
            await self._handle_error(record, e)

    async def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        await self.log(logging.DEBUG, msg, *args, **kwargs)

    async def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        await self.log(logging.INFO, msg, *args, **kwargs)

    async def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        await self.log(logging.WARNING, msg, *args, **kwargs)

    async def warn(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message (alias for warning)."""
        await self.warning(msg, *args, **kwargs)

    async def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        await self.log(logging.ERROR, msg, *args, **kwargs)

    async def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        await self.log(logging.CRITICAL, msg, *args, **kwargs)

    async def fatal(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a fatal message (alias for critical)."""
        await self.critical(msg, *args, **kwargs)

    async def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception message."""
        kwargs.setdefault("exc_info", True)
        await self.error(msg, *args, **kwargs)

    def _make_record(
        self,
        level: int,
        msg: str,
        args: tuple[Any, ...],
        exc_info: Optional[
            Union[
                bool,
                tuple[
                    type[BaseException], BaseException, Optional[TracebackType]
                ],
                tuple[None, None, None],
            ]
        ],
        extra: Optional[Dict[str, Any]],
        stack_info: bool,
    ) -> LogRecord:
        """Create a log record with caching for better performance."""
        # Get caller information
        fn, lno, func = self._get_caller_info()

        # Create the record
        record = logging.LogRecord(
            name=self.name,
            level=level,
            pathname=fn,
            lineno=lno,
            msg=msg,
            args=args,
            exc_info=cast(Any, exc_info),
        )
        record.funcName = func

        # Add extra attributes
        self._add_extra_attributes(record, extra)

        # Add stack info if requested
        if stack_info:
            record.stack_info = self._format_stack(logging.currentframe())

        # Cache the record
        self._cache_record(record)

        self._metrics.increment_created()
        return record

    def _get_caller_info(self) -> tuple[str, int, str]:
        """Get caller information for the log record."""
        # Get caller information
        frame: Optional[Any] = logging.currentframe()
        if frame is not None:
            frame = frame.f_back
            if frame is not None:
                frame = frame.f_back
                if frame is not None:
                    frame = frame.f_back

        if frame is None:
            return "<unknown>", 0, "<unknown>"
        else:
            return (
                frame.f_code.co_filename,
                frame.f_lineno,
                frame.f_code.co_name,
            )

    def _add_extra_attributes(
        self, record: LogRecord, extra: Optional[Dict[str, Any]]
    ) -> None:
        """Add extra attributes to the log record."""
        if not extra:
            return

        for key, value in extra.items():
            if key in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
            ]:
                raise KeyError(f"Attempt to overwrite {key!r} in LogRecord")
            setattr(record, key, value)

    def _cache_record(self, record: LogRecord) -> None:
        """Cache the log record for better performance."""
        record_id = id(record)
        if len(self._record_cache) >= self._record_cache_size:
            # Clear oldest entries
            self._record_cache.clear()
        self._record_cache[record_id] = record

    def _format_stack(self, frame: Any) -> str:
        """Format the stack information."""
        import traceback

        return "".join(traceback.format_stack(frame))

    async def handle(self, record: LogRecord) -> None:
        """
        Handle a log record with enhanced error handling
        and performance optimization.

        This method applies filters, formats the record, and passes it
        to handlers and parent loggers.
        """
        if self._closed:
            raise LoggerClosedError(
                "Logger is closed and cannot handle records",
                logger_name=self.name,
                operation="handle",
            )

        async with self._lock:
            # Apply filters with caching
            if not self._filter(record):
                return

            # Set the formatter on the record if the logger has one
            if self._formatter and not hasattr(record, "formatter"):
                record.formatter = self._formatter

            # Pass to handlers with enhanced error handling
            await self._handle_with_handlers(record)

            # Propagate to parent if enabled
            if self.propagateToParent() and self._parent is not None:
                await self._parent.handle(record)

    async def _handle_with_handlers(self, record: LogRecord) -> None:
        """Handle the log record with all handlers."""
        for handler in self.handlers:
            try:
                # Set default formatter and error handler if not set
                self._configure_handler(handler)
                await handler.handle(record)
            except Exception as e:
                # Log handler errors to stderr with expected format
                sys.stderr.write(
                    f"Error in handler {type(handler).__name__}: {e}\n"
                    f"Record: {record.getMessage()}\n"
                )

    def _configure_handler(self, handler: Any) -> None:
        """
        Configure a handler with default formatter
        and error handler if needed.
        """
        if (
            hasattr(handler, "formatter")
            and handler.formatter is None
            and self._formatter is not None
        ):
            handler.formatter = self._formatter
        if (
            hasattr(handler, "error_handler")
            and handler.error_handler is None
            and self._error_handler is not None
        ):
            handler.error_handler = self._error_handler

    def _filter(self, record: LogRecord) -> bool:
        """
        Apply all filters to a log record with caching for better performance.
        """
        # Use cache for filter results if enabled
        record_id = id(record)
        if record_id in self._filter_cache:
            return self._filter_cache[record_id]

        # Apply filters
        result = True
        for filter_obj in self._filters:
            try:
                if not filter_obj.filter(record):
                    result = False
                    break
            except Exception as e:
                sys.stderr.write(
                    f"Filter error in {type(filter_obj).__name__}: {e}\n"
                )
                result = False
                break

        # Cache the result
        if len(self._filter_cache) >= self._filter_cache_size:
            # Clear oldest entries
            self._filter_cache.clear()
        self._filter_cache[record_id] = result

        return result

    async def _handle_error(self, record: LogRecord, error: Exception) -> None:
        """
        Handle errors during record processing.

        Args:
            record: The log record being processed.
            error: The exception that occurred.
        """
        fallback_message = f"Error in logger {self.name}: {error}"
        await handle_error_with_fallback(
            record, error, self._error_handler, fallback_message
        )

    async def close(self) -> None:
        """Close the logger and all its handlers with enhanced cleanup."""
        async with self._lock:
            if self._closed:
                return

            self._closed = True

            # Close all handlers
            for handler in self.handlers:
                try:
                    await handler.close()
                except Exception as e:
                    sys.stderr.write(
                        "Error closing handler "
                        f"{type(handler).__name__}: {e}\n"
                    )

            # Close all children
            for child in self._children.values():
                try:
                    await child.close()
                except Exception as e:
                    sys.stderr.write(
                        f"Error closing child logger {child.name}: {e}\n"
                    )

        # Clean up references and caches
        self._handlers_refs.clear()
        self._record_cache.clear()
        self._filter_cache.clear()

    def get_metrics(self) -> dict[str, Union[int, float, bool]]:
        """
        Get performance metrics for the logger.

        Returns:
            Dictionary containing performance metrics.
        """
        metrics = self._metrics.get_metrics()
        metrics.update(
            {
                "record_cache_size": len(self._record_cache),
                "filter_cache_size": len(self._filter_cache),
                "handlers_count": len(self.handlers),
                "filters_count": len(self._filters),
                "children_count": len(self._children),
                "closed": self._closed,
                "disabled": self.disabled,
            }
        )
        return metrics

    async def __aenter__(self) -> AsyncLogger:
        """Async context manager entry."""
        if self._closed:
            raise ContextError(
                "Cannot enter context: logger is closed",
                context_type="AsyncLogger",
                operation="__aenter__",
            )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """Return a string representation of the logger."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"level={logging.getLevelName(self.getLevel())}, "
            f"handlers={len(self.handlers)}, "
            f"disabled={self.disabled}, "
            f"metrics={self.get_metrics()})"
        )


class AsyncLoggerManager:
    """
    Manager for async loggers with hierarchical naming.

    This class provides a singleton instance for managing async loggers
    with the same hierarchical naming as the standard logging module.
    """

    _instance: Optional[AsyncLoggerManager] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> AsyncLoggerManager:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the logger manager."""
        if not hasattr(self, "_initialized"):
            self._loggers: Dict[str, AsyncLogger] = {}
            self._root_logger: Optional[AsyncLogger] = None
            self._initialized = True

    def getLogger(self, name: str) -> AsyncLogger:
        """
        Get or create an async logger with the specified name.

        Args:
            name: The logger name (hierarchical, e.g., 'package.module')

        Returns:
            The async logger instance
        """
        if name in self._loggers:
            return self._loggers[name]

        # Create the logger
        logger = AsyncLogger(name)
        self._loggers[name] = logger

        # Set up parent relationship
        if name != "root":
            parent_name = name.rsplit(".", 1)[0] if "." in name else "root"
            parent = self.getLogger(parent_name)
            logger.setParent(parent)

        return logger

    def getRootLogger(self) -> AsyncLogger:
        """Get the root logger."""
        if self._root_logger is None:
            self._root_logger = AsyncLogger("root")
            self._loggers["root"] = self._root_logger
        return self._root_logger

    async def shutdown(self) -> None:
        """Shutdown all loggers."""
        async with self._lock:
            for logger in self._loggers.values():
                await logger.close()
            self._loggers.clear()
            self._root_logger = None


# Global logger manager instance
_logger_manager = AsyncLoggerManager()


def getLogger(name: str) -> AsyncLogger:
    """
    Get or create an async logger with the specified name.

    Args:
        name: The logger name (hierarchical, e.g., 'package.module')

    Returns:
        The async logger instance
    """
    return _logger_manager.getLogger(name)


def getRootLogger() -> AsyncLogger:
    """Get the root logger."""
    return _logger_manager.getRootLogger()


async def shutdown() -> None:
    """Shutdown all loggers."""
    await _logger_manager.shutdown()


@asynccontextmanager
async def getLoggerContext(name: str) -> Any:
    """
    Get an async logger as a context manager.

    This is the recommended way to use async loggers:

    async with aiologging.getLoggerContext("app") as logger:
        await logger.info("message")

    Args:
        name: The logger name

    Yields:
        The async logger instance
    """
    logger = getLogger(name)
    try:
        async with logger:
            yield logger
    except Exception as e:
        raise ContextError(
            f"Error in logger context: {e}",
            context_type="AsyncLogger",
            operation="getLoggerContext",
        ) from e


# Convenience function for the most common usage pattern
async def log_async(
    name: str,
    level: int,
    msg: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Log a message using the specified logger and level.

    This is a convenience function for simple logging operations
    without needing to manage the logger lifecycle.

    Args:
        name: The logger name
        level: The logging level
        msg: The message to log
        *args: Arguments for message formatting
        **kwargs: Additional keyword arguments for logging
    """
    async with getLoggerContext(name) as logger:
        await logger.log(level, msg, *args, **kwargs)
