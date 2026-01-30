"""
Utility functions for the aiologging library.

This module provides common utility functions used throughout the aiologging
library to ensure consistency and reduce code duplication.

The utilities include:
- Error handling helpers with consistent formatting
- Exception formatting utilities
- Context managers for error handling
- Type conversion helpers

Example:
    >>> import aiologging
    >>> from aiologging.utils import ErrorContext
    >>>
    >>> async with ErrorContext("file operation") as ctx:
    >>>     # Your code here
    >>>     pass
"""

from __future__ import annotations

import sys
import traceback
from logging import LogRecord
from typing import Any, Optional, Callable, Awaitable

from .types import AsyncErrorHandler


def log_error_to_stderr(
    message: str, exception: Optional[Exception] = None
) -> None:
    """
    Log an error message to stderr with optional exception details.

    This function provides a consistent way
    to log errors throughout the library.
    It formats the error message and optionally includes exception information
    and traceback.

    Args:
        message: The error message to log
        exception: Optional exception to include in the log

    Example:
        >>> try:
        ...     # Some operation that might fail
        ...     pass
        ... except Exception as e:
        ...     log_error_to_stderr("Operation failed", e)
    """
    error_msg = message
    if exception:
        error_msg += f": {exception}"

    sys.stderr.write(f"{error_msg}\n")

    if exception:
        sys.stderr.write(f"Traceback: {traceback.format_exc()}\n")


async def handle_error_with_fallback(
    record: LogRecord,
    error: Exception,
    error_handler: Optional[AsyncErrorHandler] = None,
    fallback_message: Optional[str] = None,
) -> None:
    """
    Handle an error using the provided error handler or fallback to stderr.

    This function provides a consistent error handling pattern throughout the
    library. It first tries to use the provided error handler, and if that
    fails or if no handler is provided, it falls back to logging to stderr.

    Args:
        record: The log record being processed when the error occurred
        error: The exception that occurred
        error_handler: Optional error handler to use for custom error handling
        fallback_message: Optional custom fallback message for stderr logging

    Example:
        >>> async def custom_error_handler(record, error):
        ...     # Custom error handling logic
        ...     pass
        >>>
        >>> try:
        ...     # Some operation that might fail
        ...     pass
        ... except Exception as e:
        ...     await handle_error_with_fallback(
        ...         record,
        ...         e,
        ...         custom_error_handler,
        ...     )
    """
    if error_handler:
        try:
            await error_handler(record, error)
        except Exception as handler_error:
            # Avoid infinite recursion if error handler fails
            log_error_to_stderr(
                f"Error in error handler: {handler_error}", error
            )
    else:
        # Default error handling
        message = fallback_message or f"Error processing log record: {error}"
        log_error_to_stderr(message, error)


def safe_format_exception(exc_info: Any) -> str:
    """
    Safely format exception information.

    This function handles different types of exception information and formats
    them safely, without raising exceptions during the formatting process.

    Args:
        exc_info: Exception information from sys.exc_info()
                  or LogRecord.exc_info

    Returns:
        Formatted exception string, or an error message if formatting fails

    Example:
        >>> try:
        ...     # Some operation that might fail
        ...     pass
        ... except Exception:
        ...     formatted = safe_format_exception(sys.exc_info())
        ...     print(formatted)
    """
    try:
        if exc_info is None:
            return ""

        if isinstance(exc_info, tuple) and len(exc_info) == 3:
            # Standard exc_info tuple (type, value, traceback)
            return traceback.format_exception(*exc_info)[-1]
        elif hasattr(exc_info, "__traceback__"):
            # Exception instance
            return "".join(
                traceback.format_exception(
                    type(exc_info), exc_info, exc_info.__traceback__
                )
            )
        else:
            # Fallback for other types
            return str(exc_info)
    except Exception:
        return f"Failed to format exception: {exc_info}"


class ErrorContext:
    """
    Context manager for consistent error handling.

    This context manager provides a consistent way to handle errors
    across different parts of the library. It can be used with or without
    a custom error handler and can optionally suppress exceptions.

    Attributes:
        operation: Description of the operation being performed
        error_handler: Optional error handler to use
        fallback_message: Custom fallback message for error logging
        reraise: Whether to reraise exceptions after handling them
        record: Optional log record associated with the operation

    Example:
        >>> async with ErrorContext("file operation", reraise=True) as ctx:
        ...     # Your code here
        ...     # If an exception occurs, it will be handled and then re-raised
        ...     pass
    """

    def __init__(
        self,
        operation: str,
        error_handler: Optional[AsyncErrorHandler] = None,
        fallback_message: Optional[str] = None,
        reraise: bool = False,
    ) -> None:
        """
        Initialize the error context.

        Args:
            operation: Description of the operation being performed
            error_handler: Optional error handler to use
                           for custom error handling
            fallback_message: Optional custom fallback message
                              for stderr logging
            reraise: Whether to reraise the exception after handling it
        """
        self.operation = operation
        self.error_handler = error_handler
        self.fallback_message = fallback_message or f"Error during {operation}"
        self.reraise = reraise
        self.record: Optional[LogRecord] = None

    def set_record(self, record: LogRecord) -> None:
        """
        Set the log record for error handling.

        This allows the error context to include the log record
        when handling errors.

        Args:
            record: The log record to associate with this context
        """
        self.record = record

    async def __aenter__(self) -> ErrorContext:
        """
        Enter the error context.

        Returns:
            The error context instance
        """
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> bool:
        """
        Exit the error context and handle any exceptions.

        Args:
            exc_type: The exception type (None if no exception)
            exc_val: The exception value (None if no exception)
            exc_tb: The exception traceback (None if no exception)

        Returns:
            True if the exception should be suppressed, False otherwise
        """
        if exc_type is not None and issubclass(exc_type, Exception):
            error = exc_val
            if self.record:
                await handle_error_with_fallback(
                    self.record,
                    error,
                    self.error_handler,
                    self.fallback_message,
                )
            else:
                log_error_to_stderr(self.fallback_message, error)

            if not self.reraise:
                return True  # Suppress the exception

        return False  # Don't suppress the exception


def create_error_handler(
    handler_func: Callable[[LogRecord, Exception], Awaitable[None]],
) -> AsyncErrorHandler:
    """
    Create a properly typed error handler from a function.

    This function is a convenience wrapper that ensures the provided
    function has the correct type signature for an error handler.

    Args:
        handler_func: Function to convert to an error handler

    Returns:
        Properly typed error handler

    Example:
        >>> async def my_error_handler(record, error):
        ...     print(f"Error {error} in record {record.name}")
        >>>
        >>> typed_handler = create_error_handler(my_error_handler)
    """
    return handler_func
