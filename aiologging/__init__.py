"""
aiologging - Asynchronous logging library for Python.

The API mirrors the standard logging module; the logging methods are
coroutines. Records are enqueued at the call site and written by a
background consumer, so ``await logger.info(...)`` never waits for
handler I/O (configurable via ``delivery``).

Basic usage:
    import aiologging

    logger = aiologging.getLogger(__name__)
    await logger.info("Hello, world!")

    # once, at application exit:
    await aiologging.shutdown()

Features:
- logging-compatible API (methods, hierarchy, levels, filters)
- Background consumer: handler I/O off the calling coroutine's path
- Configurable delivery ("enqueue"/"await") and overflow policy
  ("block"/"drop_new"/"drop_old")
- Bridge for stdlib logging records (captureStdlib) so third-party
  library logs flow through the same async handlers
- Async handlers for streams, files (with rotation) and HTTP endpoints
- Strict type checking with mypy support
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

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
    DeliveryMode,
    OverflowPolicy,
    _logger_manager,
    critical,
    debug,
    disable,
    error,
    exception,
    fatal,
    flush,
    getLogger,
    info,
    log,
    shutdown,
    warn,
    warning,
)
from .bridge import StdlibBridgeHandler, captureStdlib
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
    HttpBackendType,
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
    AsyncStreamHandler,
    AsyncFileHandler,
    AsyncRotatingFileHandler,
    AsyncTimedRotatingFileHandler,
    AsyncHttpHandlerBase,
    AsyncHttpTextHandler,
    AsyncHttpJsonHandler,
    AsyncHttpProtoHandler,
    AsyncHttpHandler,
    AsyncTelegramHandler,
)

# Re-export logging levels for compatibility
CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = logging.WARN
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

# Version information
__version__ = "0.2.3"
__author__ = "Evgenii Dementev (m6mok)"
__license__ = "MIT"

# Public API
__all__ = [
    # Core classes and functions
    "AsyncLogger",
    "AsyncLoggerManager",
    "getLogger",
    "basicConfig",
    "shutdown",
    "flush",
    "disable",
    # Root-logger convenience coroutines (mirror logging module funcs)
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "exception",
    "critical",
    "fatal",
    "log",
    # Stdlib bridge
    "StdlibBridgeHandler",
    "captureStdlib",
    # Delivery/overflow types
    "DeliveryMode",
    "OverflowPolicy",
    # Convenience factories
    "create_stream_handler",
    "create_file_handler",
    "create_http_handler",
    "create_telegram_handler",
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
    "AsyncTelegramHandler",
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
    "HttpBackendType",
    "HttpConfig",
    "HttpContentType",
    "LoggerConfig",
    "LogLevel",
    "ParamsType",
    "RotationConfig",
    "TimeInterval",
    # Logging levels
    "CRITICAL",
    "FATAL",
    "ERROR",
    "WARNING",
    "WARN",
    "INFO",
    "DEBUG",
    "NOTSET",
    # Version info
    "__version__",
    "__author__",
    "__license__",
]


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
    return AsyncFileHandler(
        filename, mode, encoding, level=level, formatter=formatter
    )


def create_http_handler(
    url: str,
    method: str = "POST",
    headers: Optional[HeadersType] = None,
    level: int = NOTSET,
    formatter: Optional[FormatterProtocol] = None,
    authenticator: Optional[AuthenticatorProtocol[Any, Any]] = None,
    backend: Optional[HttpBackendType] = None,
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
        backend: HTTP client backend to use ('aiohttp' or 'httpx');
                 if None, aiohttp is used when installed,
                 falling back to httpx

    Returns:
        An AsyncHttpHandler instance

    Raises:
        DependencyError: If neither aiohttp nor httpx is installed
    """
    return AsyncHttpHandler(
        url,
        method=method,
        headers=headers,
        level=level,
        formatter=formatter,
        authenticator=authenticator,
        backend=backend,
    )


def create_telegram_handler(
    token: str,
    chat_id: Union[str, int],
    parse_mode: Optional[str] = None,
    level: int = NOTSET,
    formatter: Optional[FormatterProtocol] = None,
    backend: Optional[HttpBackendType] = None,
) -> AsyncTelegramHandler:
    """
    Create an async Telegram handler.

    Args:
        token: The bot token from @BotFather
        chat_id: Target chat: unique identifier or "@channelname"
        parse_mode: Message formatting mode ("HTML", "Markdown",
                    "MarkdownV2"); plain text if None
        level: The logging level for this handler
        formatter: The formatter to use for log records
        backend: HTTP client backend to use ('aiohttp' or 'httpx');
                 if None, aiohttp is used when installed,
                 falling back to httpx

    Returns:
        An AsyncTelegramHandler instance

    Raises:
        ConfigurationError: If token or chat_id is invalid
        DependencyError: If neither aiohttp nor httpx is installed
    """
    return AsyncTelegramHandler(
        token,
        chat_id,
        parse_mode=parse_mode,
        level=level,
        formatter=formatter,
        backend=backend,
    )


def basicConfig(
    level: Optional[Union[int, str]] = None,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    handlers: Optional[List[AsyncHandler]] = None,
    force: bool = False,
    queue_size: Optional[int] = None,
    overflow: Optional[OverflowPolicy] = None,
    delivery: Optional[DeliveryMode] = None,
    capture_stdlib: Optional[bool] = None,
) -> None:
    """
    Configure the root logger, like ``logging.basicConfig``.

    Handlers are attached only when the root logger has none yet
    (or when ``force`` is True, which discards the existing ones);
    ``level`` is applied whenever given.

    Args:
        level: The root logger level (number or name)
        format: The log message format
        datefmt: The date format
        handlers: Handlers for the root logger; a stderr stream handler
            is created when omitted
        force: Remove existing root handlers before configuring
        queue_size: Capacity of the record queue (applies to queues
            created afterwards)
        overflow: What to do when the queue is full: "block" (default),
            "drop_new" or "drop_old"
        delivery: Default guarantee of ``await logger.info(...)``:
            "enqueue" (default) resolves once the record is queued,
            "await" resolves after handlers processed it
        capture_stdlib: True routes stdlib logging records through
            aiologging handlers (see :func:`captureStdlib`)
    """
    manager = _logger_manager
    if queue_size is not None:
        manager.queue_size = queue_size
    if overflow is not None:
        manager.overflow = overflow
    if delivery is not None:
        manager.delivery = delivery

    root = manager.root
    if force:
        for existing in list(root.handlers):
            root.removeHandler(existing)

    if not root.handlers:
        formatter = logging.Formatter(format, datefmt)
        if handlers is None:
            handlers = [create_stream_handler(formatter=formatter)]
        for handler in handlers:
            if handler.formatter is None:
                handler.setFormatter(formatter)
            root.addHandler(handler)

    if level is not None:
        root.setLevel(level)

    if capture_stdlib is not None:
        captureStdlib(capture_stdlib)
