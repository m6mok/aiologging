"""
aiologging - Asynchronous logging library for Python.

This library provides asynchronous logging with full compatibility to the
standard logging module but with async methods requiring await.

Basic usage:
    import aiologging

    async with aiologging.getLogger("app") as logger:
        await logger.info("Hello, world!")

Features:
- Full compatibility with standard logging module API
- Async handlers for streams, files, and HTTP endpoints
- File rotation support (size and time-based)
- Extensible authentication for HTTP handlers
- Optional dependencies for different use cases
- Strict type checking with mypy support
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from .exceptions import (
    AiologgingError,
    AuthenticationError,
    BatchError,
    ConfigurationError,
    ContextError,
    DependencyError,
    FileError,
    FormatterError,
    HandlerError,
    LoggerError,
    NetworkError,
    RotationError,
)
from .logger import (
    AsyncLogger,
    AsyncLoggerManager,
    getLogger,
    getLoggerContext,
    getRootLogger,
    log_async,
    shutdown,
)
from .config import (
    ConfigManager,
    get_config_manager,
    configure_from_file,
    configure_from_dict,
    configure_from_env,
    get_configured_logger,
)
from .types import (
    AuthDataType,
    AuthenticatorProtocol,
    BatchConfig,
    ErrorHandler,
    FileConfig,
    FilterProtocol,
    FormatterProtocol,
    HandlerConfig,
    HeadersType,
    HttpConfig,
    HttpContentType,
    LoggerConfig,
    LogLevel,
    ParamsType,
    RotationConfig,
    TimeInterval,
)

# Import handlers with conditional imports
from .handlers import (
    AsyncHandler,
    BufferedAsyncHandler,
)

# Direct import for AsyncStreamHandler to avoid type issues
from .handlers.stream import AsyncStreamHandler

# Helper function to create stub classes with dependency errors
def _create_dependency_stub(class_name: str, dependency_name: str, install_command: str) -> type:
    """Create a stub class that raises a DependencyError when instantiated."""
    class StubClass:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            from .exceptions import DependencyError
            raise DependencyError(
                f"{dependency_name} is required for {class_name}. "
                f"Install with: {install_command}",
                dependency_name=dependency_name,
                install_command=install_command,
            )
    StubClass.__name__ = class_name
    return StubClass

# Import file handlers (require aiofiles)
if TYPE_CHECKING:
    from .handlers.file import AsyncFileHandler
    from .handlers.rotating import AsyncRotatingFileHandler, AsyncTimedRotatingFileHandler
    from .handlers.http import (
        AsyncHttpHandlerBase,
        AsyncHttpTextHandler,
        AsyncHttpJsonHandler,
        AsyncHttpProtoHandler,
        AsyncHttpHandler,
    )
else:
    # Try to import file handlers
    try:
        from .handlers import AsyncFileHandler
    except ImportError:
        AsyncFileHandler = _create_dependency_stub(
            "AsyncFileHandler",
            "aiofiles",
            "pip install aiologging[aiofiles]"
        )

    # Try to import rotating file handlers
    try:
        from .handlers import AsyncRotatingFileHandler, AsyncTimedRotatingFileHandler
    except ImportError:
        AsyncRotatingFileHandler = _create_dependency_stub(
            "AsyncRotatingFileHandler",
            "aiofiles",
            "pip install aiologging[aiofiles]"
        )
        AsyncTimedRotatingFileHandler = _create_dependency_stub(
            "AsyncTimedRotatingFileHandler",
            "aiofiles",
            "pip install aiologging[aiofiles]"
        )

    # Try to import HTTP handlers
    try:
        from .handlers import (
            AsyncHttpHandlerBase,
            AsyncHttpTextHandler,
            AsyncHttpJsonHandler,
            AsyncHttpProtoHandler,
            AsyncHttpHandler,
        )
    except ImportError:
        AsyncHttpHandlerBase = _create_dependency_stub(
            "AsyncHttpHandlerBase",
            "aiohttp",
            "pip install aiologging[aiohttp]"
        )
        AsyncHttpTextHandler = _create_dependency_stub(
            "AsyncHttpTextHandler",
            "aiohttp",
            "pip install aiologging[aiohttp]"
        )
        AsyncHttpJsonHandler = _create_dependency_stub(
            "AsyncHttpJsonHandler",
            "aiohttp",
            "pip install aiologging[aiohttp]"
        )
        AsyncHttpProtoHandler = _create_dependency_stub(
            "AsyncHttpProtoHandler",
            "aiohttp and protobuf",
            "pip install aiologging[aiohttp,protobuf]"
        )
        AsyncHttpHandler = _create_dependency_stub(
            "AsyncHttpHandler",
            "aiohttp",
            "pip install aiologging[aiohttp]"
        )

# Re-export logging levels for compatibility
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

# Version information
__version__ = "0.1.0"
__author__ = "Aiologging Contributors"
__email__ = "contributors@aiologging.dev"
__license__ = "MIT"

# Public API
__all__ = [
    # Core classes and functions
    "AsyncLogger",
    "AsyncLoggerManager",
    "getLogger",
    "getLoggerContext",
    "getRootLogger",
    "log_async",
    "shutdown",

    # Base handlers
    "AsyncHandler",
    "AsyncStreamHandler",
    "BufferedAsyncHandler",

    # File handlers (conditional)
    "AsyncFileHandler",
    "AsyncRotatingFileHandler",
    "AsyncTimedRotatingFileHandler",

    # HTTP handlers (conditional)
    "AsyncHttpHandlerBase",
    "AsyncHttpTextHandler",
    "AsyncHttpJsonHandler",
    "AsyncHttpProtoHandler",
    "AsyncHttpHandler",

    # Configuration management
    "ConfigManager",
    "get_config_manager",
    "configure_from_file",
    "configure_from_dict",
    "configure_from_env",
    "get_configured_logger",

    # Exceptions
    "AiologgingError",
    "AuthenticationError",
    "BatchError",
    "ConfigurationError",
    "ContextError",
    "DependencyError",
    "FileError",
    "FormatterError",
    "HandlerError",
    "LoggerError",
    "NetworkError",
    "RotationError",

    # Types and configurations
    "AuthDataType",
    "AuthenticatorProtocol",
    "BatchConfig",
    "ErrorHandler",
    "FileConfig",
    "FilterProtocol",
    "FormatterProtocol",
    "HandlerConfig",
    "HeadersType",
    "HttpConfig",
    "HttpContentType",
    "LoggerConfig",
    "LogLevel",
    "ParamsType",
    "RotationConfig",
    "TimeInterval",

    # Logging levels
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",

    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]


def _check_optional_dependencies() -> None:
    """Check for optional dependencies and provide helpful error messages."""
    import sys

    # Define optional dependencies with their install commands
    dependencies = {
        "aiofiles": "pip install aiologging[aiofiles]",
        "aiohttp": "pip install aiologging[aiohttp]",
        "google.protobuf": "pip install aiologging[protobuf]",
    }

    # Check each dependency and collect warnings
    warnings = []
    for dep_name, install_cmd in dependencies.items():
        try:
            __import__(dep_name)
        except ImportError:
            friendly_name = dep_name.split('.')[-1]  # Get the last part of the module name
            warnings.append(
                f"Warning: {friendly_name} not available. "
                f"Install with: {install_cmd}"
            )

    # Write all warnings at once for better performance
    if warnings:
        sys.stderr.write("\n".join(warnings) + "\n")


# Check dependencies on import
_check_optional_dependencies()


# Convenience functions for creating handlers
def create_stream_handler(
    level: int = NOTSET,
    formatter: Optional[FormatterProtocol] = None,
) -> AsyncStreamHandler:
    """
    Create an async stream handler.

    Args:
        level: The logging level for this handler
        formatter: The formatter to use for log records

    Returns:
        An AsyncStreamHandler instance
    """
    return AsyncStreamHandler(level=level, formatter=formatter)


def create_file_handler(
    filename: str,
    mode: str = "a",
    encoding: str = "utf-8",
    level: int = NOTSET,
    formatter: Optional[FormatterProtocol] = None,
) -> AsyncFileHandler:
    """
    Create an async file handler.

    Args:
        filename: The path to the log file
        mode: The file opening mode
        encoding: The file encoding
        level: The logging level for this handler
        formatter: The formatter to use for log records

    Returns:
        An AsyncFileHandler instance

    Raises:
        DependencyError: If aiofiles is not installed
    """
    # Check if aiofiles is available by trying to import it
    try:
        import aiofiles
    except ImportError:
        from .exceptions import DependencyError
        raise DependencyError(
            "aiofiles is required for file handlers. "
            "Install with: pip install aiologging[aiofiles]",
            dependency_name="aiofiles",
            install_command="pip install aiologging[aiofiles]",
        )

    return AsyncFileHandler(filename, mode, encoding, level=level, formatter=formatter)


def create_http_handler(
    url: str,
    method: str = "POST",
    headers: Optional[HeadersType] = None,
    level: int = NOTSET,
    formatter: Optional[FormatterProtocol] = None,
    authenticator: Optional[AuthenticatorProtocol[Any, Any]] = None,
) -> AsyncHttpHandler:
    """
    Create an async HTTP handler.

    Args:
        url: The HTTP endpoint URL
        method: The HTTP method
        headers: Additional HTTP headers
        level: The logging level for this handler
        formatter: The formatter to use for log records
        authenticator: Optional authentication function

    Returns:
        An AsyncHttpHandler instance

    Raises:
        DependencyError: If aiohttp is not installed
    """
    # Check if aiohttp is available by trying to import it
    try:
        import aiohttp
    except ImportError:
        from .exceptions import DependencyError
        raise DependencyError(
            "aiohttp is required for HTTP handlers. "
            "Install with: pip install aiologging[aiohttp]",
            dependency_name="aiohttp",
            install_command="pip install aiologging[aiohttp]",
        )

    return AsyncHttpHandler(
        url, method=method, headers=headers, level=level,
        formatter=formatter, authenticator=authenticator
    )


# Module-level configuration
class Config:
    """Global configuration for aiologging."""

    def __init__(self) -> None:
        self.default_level: int = INFO
        self.default_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.default_date_format: str = "%Y-%m-%d %H:%M:%S"

    def get_default_formatter(self) -> logging.Formatter:
        """Get the default formatter."""
        return logging.Formatter(self.default_format, self.default_date_format)


# Global configuration instance
config = Config()


def basicConfig(
    level: int = INFO,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    handlers: Optional[list[AsyncHandler]] = None,
) -> None:
    """
    Configure basic logging for aiologging.

    This function is similar to logging.basicConfig() but for async loggers.

    Args:
        level: The default logging level
        format: The log message format
        datefmt: The date format
        handlers: List of handlers to add to the root logger
    """
    # Update global config
    config.default_level = level
    config.default_format = format
    config.default_date_format = datefmt

    # Configure root logger
    root_logger = getRootLogger()
    root_logger.setLevel(level)

    # Create default formatter
    formatter = logging.Formatter(format, datefmt)

    # Add handlers
    if handlers:
        for handler in handlers:
            if handler.formatter is None:
                handler.setFormatter(formatter)
            root_logger.addHandler(handler)
    else:
        # Add default stream handler if none provided
        stream_handler = create_stream_handler(level=level, formatter=formatter)
        root_logger.addHandler(stream_handler)
