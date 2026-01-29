"""
Custom exceptions for the aiologging library.

This module defines all custom exceptions used throughout the aiologging
library to provide clear error handling and debugging information.

The exception hierarchy is designed to provide specific error types for different
components of the library, making it easier to handle errors appropriately.
All exceptions inherit from the base AiologgingError class, which provides
consistent error formatting and context information.

Exception hierarchy:
- AiologgingError (base)
  - HandlerError
    - HandlerClosedError
    - BufferFullError
  - ConfigurationError
  - DependencyError
  - AuthenticationError
  - NetworkError
  - FileError
    - RotationError
  - BatchError
  - FormatterError
  - LoggerError
    - LoggerClosedError
  - ContextError

Example:
    >>> try:
    ...     # Some operation that might fail
    ...     pass
    ... except aiologging.HandlerError as e:
    ...     print(f"Handler error: {e}")
    ...     print(f"Handler name: {e.handler_name}")
    ...     print(f"Operation: {e.operation}")
"""

from __future__ import annotations

from typing import ClassVar, Optional, Any

from .types import ConfigValue


class AiologgingError(Exception):
    """
    Base exception for all aiologging errors.

    This is the base class for all exceptions in the aiologging library.
    It provides consistent error formatting and context information through
    a flexible details system.

    Attributes:
        message: The error message
        details: Dictionary of additional context information
        _error_template: Template for formatting error messages
        _detail_separator: Separator between detail items
        _detail_format: Format for individual detail items

    Example:
        >>> raise AiologgingError(
        ...     "Something went wrong",
        ...     operation="logging",
        ...     level="ERROR",
        ...     component="handler"
        ... )
    """

    # Class-level configuration for error formatting
    _error_template: ClassVar[str] = "{message}"
    _detail_separator: ClassVar[str] = " | "
    _detail_format: ClassVar[str] = "{key}={value}"

    def __init__(self, message: str, details: Optional[dict[str, ConfigValue]] = None, **kwargs: Any) -> None:
        """
        Initialize the exception with a message and optional details.

        Args:
            message: The error message
            details: Optional dictionary of additional context information
            **kwargs: Additional context information that will be merged with details
        """
        super().__init__(message)
        self.message = message

        # Merge kwargs with details for more flexible initialization
        merged_details = details or {}
        merged_details.update(kwargs)
        self.details = merged_details

        # Set common attributes from kwargs for direct access
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _format_details(self) -> str:
        """
        Format details into a string.

        Returns:
            Formatted details string or empty string if no details
        """
        if not self.details:
            return ""
        # Filter out None values to avoid cluttering the output
        filtered_details = {k: v for k, v in self.details.items() if v is not None}
        if not filtered_details:
            return ""
        return self._detail_separator.join(
            self._detail_format.format(key=k, value=v) for k, v in filtered_details.items()
        )

    def __str__(self) -> str:
        """
        Return a string representation of the exception.

        Returns:
            Formatted error message with details if available
        """
        formatted_details = self._format_details()
        if formatted_details:
            return f"{self.message} ({formatted_details})"
        return self.message

    def get_message_only(self) -> str:
        """
        Return only the error message without details.

        This method is useful when you need just the error message
        without the additional context information.

        Returns:
            The error message only
        """
        return self.message

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the exception.

        Returns:
            Detailed representation including class name and details
        """
        return f"{self.__class__.__name__}('{self.message}', details={self.details})"


class HandlerError(AiologgingError):
    """
    Exception raised when handler operations fail.

    This exception is used for errors that occur in log handlers,
    such as file handlers, HTTP handlers, etc.

    Attributes:
        message: The error message
        handler_name: Name of the handler that raised the error
        operation: Operation that was being performed when the error occurred
        details: Additional context information

    Example:
        >>> raise HandlerError(
        ...     "Failed to write to file",
        ...     handler_name="file_handler",
        ...     operation="emit"
        ... )
    """

    _error_template: ClassVar[str] = "{message} | Handler: {handler_name} | Operation: {operation}"

    def __init__(
        self,
        message: str,
        handler_name: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the handler error.

        Args:
            message: The error message
            handler_name: Name of the handler that raised the error
            operation: Operation that was being performed when the error occurred
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(
            message,
            details,
            handler_name=handler_name,
            operation=operation,
            **kwargs
        )


class ConfigurationError(AiologgingError):
    """
    Exception raised when configuration is invalid.

    This exception is used for errors related to configuration,
    such as invalid configuration values, missing required settings, etc.

    Attributes:
        message: The error message
        config_key: The configuration key that caused the error
        config_value: The configuration value that caused the error
        details: Additional context information

    Example:
        >>> raise ConfigurationError(
        ...     "Invalid log level",
        ...     config_key="level",
        ...     config_value="INVALID"
        ... )
    """

    _error_template: ClassVar[str] = "{message} | Config key: {config_key} | Config value: {config_value}"

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[ConfigValue] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the configuration error.

        Args:
            message: The error message
            config_key: The configuration key that caused the error
            config_value: The configuration value that caused the error
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(
            message,
            details,
            config_key=config_key,
            config_value=config_value,
            **kwargs
        )


class DependencyError(AiologgingError):
    """
    Exception raised when required dependencies are missing.

    This exception is used when a required dependency is not available,
    such as when trying to use a feature that requires an optional package.

    Attributes:
        message: The error message
        dependency_name: Name of the missing dependency
        install_command: Command to install the missing dependency
        details: Additional context information

    Example:
        >>> raise DependencyError(
        ...     "aiohttp is required for HTTP handlers",
        ...     dependency_name="aiohttp",
        ...     install_command="pip install aiohttp"
        ... )
    """

    _error_template: ClassVar[str] = "{message} | Missing dependency: {dependency_name} | Install with: {install_command}"

    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        install_command: Optional[str] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the dependency error.

        Args:
            message: The error message
            dependency_name: Name of the missing dependency
            install_command: Command to install the missing dependency
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(
            message,
            details,
            dependency_name=dependency_name,
            install_command=install_command,
            **kwargs
        )


class AuthenticationError(AiologgingError):
    """
    Exception raised when authentication fails.

    This exception is used for authentication-related errors,
    such as invalid credentials, authentication server errors, etc.

    Attributes:
        message: The error message
        auth_type: Type of authentication that failed
        status_code: HTTP status code if applicable
        details: Additional context information

    Example:
        >>> raise AuthenticationError(
        ...     "Invalid credentials",
        ...     auth_type="basic",
        ...     status_code=401
        ... )
    """

    _error_template: ClassVar[str] = "{message} | Auth type: {auth_type} | Status code: {status_code}"

    def __init__(
        self,
        message: str,
        auth_type: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the authentication error.

        Args:
            message: The error message
            auth_type: Type of authentication that failed
            status_code: HTTP status code if applicable
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(
            message,
            details,
            auth_type=auth_type,
            status_code=status_code,
            **kwargs
        )


class NetworkError(AiologgingError):
    """
    Exception raised when network operations fail.

    This exception is used for network-related errors,
    such as connection failures, timeouts, HTTP errors, etc.

    Attributes:
        message: The error message
        url: URL that was being accessed when the error occurred
        status_code: HTTP status code if applicable
        timeout: Timeout value in seconds if applicable
        details: Additional context information

    Example:
        >>> raise NetworkError(
        ...     "Connection failed",
        ...     url="https://example.com/api/logs",
        ...     timeout=30.0
        ... )
    """

    _error_template: ClassVar[str] = "{message} | URL: {url} | Status code: {status_code} | Timeout: {timeout}s"

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        timeout: Optional[float] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the network error.

        Args:
            message: The error message
            url: URL that was being accessed when the error occurred
            status_code: HTTP status code if applicable
            timeout: Timeout value in seconds if applicable
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(
            message,
            details,
            url=url,
            status_code=status_code,
            timeout=timeout,
            **kwargs
        )


class FileError(AiologgingError):
    """
    Exception raised when file operations fail.

    This exception is used for file-related errors,
    such as permission errors, disk full errors, etc.

    Attributes:
        message: The error message
        filename: Name of the file that caused the error
        operation: Operation that was being performed when the error occurred
        errno: System error number if applicable
        details: Additional context information

    Example:
        >>> raise FileError(
        ...     "Permission denied",
        ...     filename="/var/log/app.log",
        ...     operation="write",
        ...     errno=13
        ... )
    """

    _error_template: ClassVar[str] = "{message} | File: {filename} | Operation: {operation} | Error number: {errno}"

    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        operation: Optional[str] = None,
        errno: Optional[int] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the file error.

        Args:
            message: The error message
            filename: Name of the file that caused the error
            operation: Operation that was being performed when the error occurred
            errno: System error number if applicable
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(
            message,
            details,
            filename=filename,
            operation=operation,
            errno=errno,
            **kwargs
        )


class RotationError(FileError):
    """
    Exception raised when file rotation fails.

    This exception is used for errors related to file rotation,
    such as inability to create new files, permission errors, etc.

    Attributes:
        message: The error message
        filename: Name of the file that was being rotated
        rotation_type: Type of rotation that failed
        details: Additional context information

    Example:
        >>> raise RotationError(
        ...     "Failed to rotate file",
        ...     filename="/var/log/app.log",
        ...     rotation_type="size"
        ... )
    """

    _error_template: ClassVar[str] = "{message} | File: {filename} | Rotation type: {rotation_type}"

    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        rotation_type: Optional[str] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the rotation error.

        Args:
            message: The error message
            filename: Name of the file that was being rotated
            rotation_type: Type of rotation that failed
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        # Don't pass operation to parent to avoid it appearing in basic error messages
        super().__init__(
            message,
            details=details,
            filename=filename,
            operation=None,  # Set to None to avoid appearing in basic error messages
            errno=None,
            rotation_type=rotation_type,
            **kwargs
        )


class BatchError(AiologgingError):
    """
    Exception raised when batch processing fails.

    This exception is used for errors related to batch processing,
    such as when a batch of records cannot be processed.

    Attributes:
        message: The error message
        batch_size: Size of the batch that failed
        processed_count: Number of items that were successfully processed
        details: Additional context information

    Example:
        >>> raise BatchError(
        ...     "Failed to process batch",
        ...     batch_size=100,
        ...     processed_count=50
        ... )
    """

    _error_template: ClassVar[str] = "{message} | Batch size: {batch_size} | Processed: {processed_count}"

    def __init__(
        self,
        message: str,
        batch_size: Optional[int] = None,
        processed_count: Optional[int] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the batch error.

        Args:
            message: The error message
            batch_size: Size of the batch that failed
            processed_count: Number of items that were successfully processed
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(
            message,
            details,
            batch_size=batch_size,
            processed_count=processed_count,
            **kwargs
        )


class FormatterError(AiologgingError):
    """
    Exception raised when formatting fails.

    This exception is used for errors related to log record formatting,
    such as invalid format strings, missing attributes, etc.

    Attributes:
        message: The error message
        formatter_name: Name of the formatter that failed
        record_summary: Summary of the record that caused the error
        record_details: Full details of the record that caused the error
        details: Additional context information

    Example:
        >>> raise FormatterError(
        ...     "Invalid format string",
        ...     formatter_name="json_formatter",
        ...     record_details={"level": "INFO", "message": "Test"}
        ... )
    """

    _error_template: ClassVar[str] = "{message} | Formatter: {formatter_name} | Record: {record_summary}"

    def __init__(
        self,
        message: str,
        formatter_name: Optional[str] = None,
        record_details: Optional[dict[str, ConfigValue]] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the formatter error.

        Args:
            message: The error message
            formatter_name: Name of the formatter that failed
            record_details: Details of the record that caused the error
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        # Create a summary of record details to avoid overly long error messages
        record_summary = None
        if record_details:
            record_summary = ", ".join(f"{k}={v}" for k, v in list(record_details.items())[:3])
            if len(record_details) > 3:
                record_summary += "..."

        super().__init__(
            message,
            details,
            formatter_name=formatter_name,
            record_summary=record_summary,
            **kwargs
        )
        self.record_details = record_details or {}


class LoggerError(AiologgingError):
    """
    Exception raised when logger operations fail.

    This exception is used for errors related to logger operations,
    such as configuration errors, initialization errors, etc.

    Attributes:
        message: The error message
        logger_name: Name of the logger that raised the error
        operation: Operation that was being performed when the error occurred
        details: Additional context information

    Example:
        >>> raise LoggerError(
        ...     "Invalid log level",
        ...     logger_name="my_logger",
        ...     operation="set_level"
        ... )
    """

    _error_template: ClassVar[str] = "{message} | Logger: {logger_name} | Operation: {operation}"

    def __init__(
        self,
        message: str,
        logger_name: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the logger error.

        Args:
            message: The error message
            logger_name: Name of the logger that raised the error
            operation: Operation that was being performed when the error occurred
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(
            message,
            details,
            logger_name=logger_name,
            operation=operation,
            **kwargs
        )


class ContextError(AiologgingError):
    """
    Exception raised when context manager operations fail.

    This exception is used for errors related to context manager operations,
    such as entering or exiting a context.

    Attributes:
        message: The error message
        context_type: Type of context that failed
        operation: Operation that was being performed when the error occurred
        details: Additional context information

    Example:
        >>> raise ContextError(
        ...     "Failed to enter context",
        ...     context_type="file_handler",
        ...     operation="enter"
        ... )
    """

    _error_template: ClassVar[str] = "{message} | Context type: {context_type} | Operation: {operation}"

    def __init__(
        self,
        message: str,
        context_type: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the context error.

        Args:
            message: The error message
            context_type: Type of context that failed
            operation: Operation that was being performed when the error occurred
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(
            message,
            details,
            context_type=context_type,
            operation=operation,
            **kwargs
        )


class HandlerClosedError(HandlerError):
    """
    Exception raised when attempting to use a closed handler.

    This exception is used when an operation is attempted on a handler
    that has been closed.

    Attributes:
        message: The error message
        handler_name: Name of the handler that is closed
        operation: Operation that was attempted on the closed handler
        details: Additional context information

    Example:
        >>> raise HandlerClosedError(
        ...     "Cannot emit to closed handler",
        ...     handler_name="file_handler",
        ...     operation="emit"
        ... )
    """

    def __init__(
        self,
        message: str,
        handler_name: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the handler closed error.

        Args:
            message: The error message
            handler_name: Name of the handler that is closed
            operation: Operation that was attempted on the closed handler
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(message, handler_name, operation, details, **kwargs)


class LoggerClosedError(LoggerError):
    """
    Exception raised when attempting to use a closed logger.

    This exception is used when an operation is attempted on a logger
    that has been closed.

    Attributes:
        message: The error message
        logger_name: Name of the logger that is closed
        operation: Operation that was attempted on the closed logger
        details: Additional context information

    Example:
        >>> raise LoggerClosedError(
        ...     "Cannot log to closed logger",
        ...     logger_name="my_logger",
        ...     operation="info"
        ... )
    """

    def __init__(
        self,
        message: str,
        logger_name: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the logger closed error.

        Args:
            message: The error message
            logger_name: Name of the logger that is closed
            operation: Operation that was attempted on the closed logger
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(message, logger_name, operation, details, **kwargs)


class BufferFullError(HandlerError):
    """
    Exception raised when a handler buffer is full.

    This exception is used when a handler's buffer is full and cannot
    accept more data.

    Attributes:
        message: The error message
        handler_name: Name of the handler with the full buffer
        operation: Operation that was attempted when the buffer was full
        buffer_size: Size of the buffer that is full
        details: Additional context information

    Example:
        >>> raise BufferFullError(
        ...     "Cannot add to full buffer",
        ...     handler_name="memory_handler",
        ...     buffer_size=1000
        ... )
    """

    _error_template: ClassVar[str] = "{message} | Handler: {handler_name} | Buffer size: {buffer_size}"

    def __init__(
        self,
        message: str,
        handler_name: Optional[str] = None,
        operation: Optional[str] = None,
        buffer_size: Optional[int] = None,
        details: Optional[dict[str, ConfigValue]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the buffer full error.

        Args:
            message: The error message
            handler_name: Name of the handler with the full buffer
            operation: Operation that was attempted when the buffer was full
            buffer_size: Size of the buffer that is full
            details: Optional dictionary of additional context information
            **kwargs: Additional context information
        """
        super().__init__(
            message,
            handler_name,
            operation,
            details,
            buffer_size=buffer_size,
            **kwargs
        )
