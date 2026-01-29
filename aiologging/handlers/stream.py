"""
Async stream handler for aiologging.

This module provides an async handler that writes log records to streams
like stdout, stderr, or any file-like stream.
"""

from __future__ import annotations

import asyncio
import sys
from logging import LogRecord, NOTSET
from typing import Optional, TextIO, Union, Protocol, runtime_checkable

from ..exceptions import HandlerError
from ..types import ErrorHandler, FilterProtocol, FormatterProtocol
from .base import AsyncHandler


@runtime_checkable
class AsyncStreamWriter(Protocol):
    """Protocol for async stream writers."""

    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


@runtime_checkable
class AsyncTextWriter(Protocol):
    """Protocol for async text writers."""

    async def write(self, data: str) -> None: ...


@runtime_checkable
class SyncWriter(Protocol):
    """Protocol for sync writers."""

    def write(self, data: str) -> None: ...

    def flush(self) -> None: ...


class AsyncStreamHandler(AsyncHandler):
    """
    Async handler that writes log records to a stream.

    This handler is similar to the standard logging.StreamHandler but
    provides async functionality for non-blocking I/O operations.
    """

    def __init__(
        self,
        stream: Optional[Union[TextIO, AsyncStreamWriter, AsyncTextWriter, SyncWriter]] = None,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[list[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
        encoding: str = "utf-8",
        errors: Optional[str] = None,
    ) -> None:
        """
        Initialize the async stream handler.

        Args:
            stream: The stream to write to (defaults to sys.stderr)
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions
            encoding: Encoding to use for the stream
            errors: Error handling strategy for encoding
        """
        super().__init__(level, formatter, filters, error_handler)

        if stream is None:
            stream = sys.stderr

        self.stream = stream
        self.encoding = encoding
        self.errors = errors or "strict"
        # Check if it's an asyncio.StreamWriter by checking for drain method
        self._is_async_stream = hasattr(stream, 'drain') and hasattr(stream, 'write')
        self._write_lock = asyncio.Lock()

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        """
        Emit a log record to the stream.

        Args:
            record: The log record to emit
            formatted_message: The formatted log message

        Raises:
            HandlerError: If writing to the stream fails
        """
        try:
            msg = formatted_message
            stream = self.stream

            # Ensure the message ends with a newline
            if not msg.endswith('\n'):
                msg += '\n'

            async with self._write_lock:
                if self._is_async_stream and isinstance(stream, AsyncStreamWriter):
                    # Handle asyncio.StreamWriter
                    msg_bytes = msg.encode(self.encoding, self.errors) if isinstance(msg, str) else msg

                    stream.write(msg_bytes)
                    await stream.drain()
                elif isinstance(stream, AsyncTextWriter):
                    # Handle async text writer
                    await stream.write(msg)
                elif isinstance(stream, SyncWriter):
                    # Handle sync writer - run in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, stream.write, msg)

                    # Flush if the stream has a flush method
                    if hasattr(stream, 'flush'):
                        await loop.run_in_executor(None, stream.flush)
                else:
                    raise HandlerError(
                        "Stream does not have a compatible write method",
                        handler_name=type(self).__name__,
                        operation="_emit",
                        details={"stream_type": type(stream).__name__},
                    )

        except Exception as e:
            raise HandlerError(
                f"Failed to emit log record to stream: {e}",
                handler_name=type(self).__name__,
                operation="_emit",
                details={
                    "record_level": record.levelname,
                    "record_message": record.getMessage(),
                    "stream_type": type(self.stream).__name__},
            ) from e

    async def _close_resources(self) -> None:
        """Close the stream if it supports closing."""
        if hasattr(self.stream, 'close'):
            try:
                if asyncio.iscoroutinefunction(self.stream.close):
                    await self.stream.close()
                else:
                    # For sync close, run in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.stream.close)
            except Exception as e:
                # Log the error but don't raise it during cleanup
                sys.stderr.write(f"Error closing stream: {e}\n")

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        stream_name = getattr(self.stream, 'name', type(self.stream).__name__)
        return (
            f"{self.__class__.__name__}(stream={stream_name}, "
            f"level={self.level}, "
            f"formatter={type(self.formatter).__name__ if self.formatter else None})"
        )


class AsyncStandardStreamHandler(AsyncStreamHandler):
    """
    Async handler for standard streams (stdout/stderr).

    This is a convenience class that provides predefined handlers
    for stdout and stderr with appropriate defaults.
    """

    def __init__(
        self,
        stream_name: str = "stderr",
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[list[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
    ) -> None:
        """
        Initialize the async standard stream handler.

        Args:
            stream_name: Either "stdout" or "stderr"
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions

        Raises:
            HandlerError: If stream_name is not "stdout" or "stderr"
        """
        if stream_name not in ("stdout", "stderr"):
            raise HandlerError(
                f"Invalid stream name: {stream_name}. Must be 'stdout' or 'stderr'",
                handler_name=type(self).__name__,
                operation="__init__",
                details={"stream_name": stream_name},
            )

        stream = sys.stdout if stream_name == "stdout" else sys.stderr
        super().__init__(stream, level, formatter, filters, error_handler)
        self.stream_name = stream_name

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        return (
            f"{self.__class__.__name__}(stream={self.stream_name}, "
            f"level={self.level}, "
            f"formatter={type(self.formatter).__name__ if self.formatter else None})"
        )


# Convenience functions for creating standard stream handlers
def async_stdout_handler(
    level: int = NOTSET,
    formatter: Optional[FormatterProtocol] = None,
    filters: Optional[list[FilterProtocol]] = None,
    error_handler: Optional[ErrorHandler] = None,
) -> AsyncStandardStreamHandler:
    """
    Create an async stdout handler.

    Args:
        level: The logging level for this handler
        formatter: The formatter to use for log records
        filters: List of filters to apply to log records
        error_handler: Optional error handler for exceptions

    Returns:
        An AsyncStandardStreamHandler configured for stdout
    """
    return AsyncStandardStreamHandler(
        "stdout", level, formatter, filters, error_handler
    )


def async_stderr_handler(
    level: int = NOTSET,
    formatter: Optional[FormatterProtocol] = None,
    filters: Optional[list[FilterProtocol]] = None,
    error_handler: Optional[ErrorHandler] = None,
) -> AsyncStandardStreamHandler:
    """
    Create an async stderr handler.

    Args:
        level: The logging level for this handler
        formatter: The formatter to use for log records
        filters: List of filters to apply to log records
        error_handler: Optional error handler for exceptions

    Returns:
        An AsyncStandardStreamHandler configured for stderr
    """
    return AsyncStandardStreamHandler(
        "stderr", level, formatter, filters, error_handler
    )
