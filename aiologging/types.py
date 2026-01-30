"""
Core type definitions for aiologging library.

This module contains all the type definitions used throughout the aiologging
library to ensure strict type checking and better IDE support.

The types defined here include:
- Type variables for generic handlers and loggers
- Type aliases for common values and configurations
- Protocol definitions for async interfaces
- Configuration classes for different components
- Type definitions for HTTP, file, and rotation handlers

Example:
    >>> from aiologging.types import HandlerConfig, AsyncHandlerProtocol
    >>>
    >>> config = HandlerConfig(
    ...     handler_class=MyHandler,
    ...     level=logging.INFO,
    ...     formatter=my_formatter
    ... )
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop
from collections.abc import Awaitable, Callable, Mapping
from logging import LogRecord
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

if sys.version_info >= (3, 10):
    from typing import ParamSpec, TypeAlias
else:
    from typing_extensions import ParamSpec, TypeAlias

# Type variables for generic handlers with proper variance
T = TypeVar("T")
RequestType = TypeVar("RequestType", contravariant=True)
ResponseType = TypeVar("ResponseType", covariant=True)
P = ParamSpec("P")

# More specific type variables for better type safety
HandlerType = TypeVar("HandlerType", bound="AsyncHandler")
LoggerType = TypeVar("LoggerType", bound="AsyncLogger")
FormatterType = TypeVar("FormatterType", bound="FormatterProtocol")
FilterType = TypeVar("FilterType", bound="FilterProtocol")


JsonValue = Union[str, int, float, bool, None, Dict[str, "JsonValue"], List["JsonValue"]]
ConfigValue = Union[str, int, float, bool, None, List["ConfigValue"], Dict[str, "ConfigValue"]]
HeaderValues = Union[str, List[str]]
ParamValues = Union[str, List[str]]


LogLevel: TypeAlias = Literal[
    "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"
]


LOG_LEVEL_MAP: Dict[LogLevel, int] = {
    "CRITICAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 0,
}


@runtime_checkable
class AuthenticatorProtocol(Protocol[RequestType, ResponseType]):
    """
    Protocol for authentication functions used in HTTP handlers.

    This protocol defines the interface for authentication functions
    that can be used with HTTP handlers to authenticate requests.

    Type Parameters:
        RequestType: The type of request data (contravariant)
        ResponseType: The type of authentication response (covariant)

    Example:
        >>> async def basic_auth(session, request_data):
        ...     # Perform authentication
        ...     return {"authenticated": True}
    """

    async def __call__(
        self,
        session: "ClientSession",
        request_data: RequestType,
    ) -> ResponseType:
        """Authenticate the request using the session and request data."""
        ...


@runtime_checkable
class FormatterProtocol(Protocol):
    """
    Protocol for log formatters.

    This protocol defines the interface for log formatters
    that can be used to format log records into strings.

    Methods:
        format: Format a log record into a string
        usesTime: Check if the formatter includes time information
        formatTime: Format the time from a log record

    Example:
        >>> class MyFormatter:
        ...     def format(self, record):
        ...         return f"{record.levelname}: {record.getMessage()}"
        ...
        ...     def usesTime(self):
        ...         return False
        ...
        ...     def formatTime(self, record, datefmt=None):
        ...         return record.created
    """

    def format(self, record: LogRecord) -> str:
        """Format a log record into a string."""
        ...

    def usesTime(self) -> bool:
        """Check if the formatter includes time information."""
        ...

    def formatTime(self, record: LogRecord, datefmt: Optional[str] = None) -> str:
        """Format the time from a log record."""
        ...


@runtime_checkable
class FilterProtocol(Protocol):
    """
    Protocol for log filters.

    This protocol defines the interface for log filters
    that can be used to filter log records based on various criteria.

    Methods:
        filter: Filter a log record
        filterName: Filter by logger name

    Example:
        >>> class MyFilter:
        ...     def filter(self, record):
        ...         return record.levelno >= logging.WARNING
        ...
        ...     def filterName(self, name):
        ...         return name.startswith("myapp")
    """

    def filter(self, record: LogRecord) -> bool:
        """Filter a log record."""
        ...

    def filterName(self, name: str) -> bool:
        """Filter by logger name."""
        ...


HeadersType: TypeAlias = Mapping[str, str]
ParamsType: TypeAlias = Mapping[str, Union[str, List[str]]]

AuthDataType: TypeAlias = Dict[str, Union[str, int, float, bool, List[Union[str, int, float]], Dict[str, Union[str, int, float]]]]
FilePath: TypeAlias = Union[str, Path]

AsyncCallable = Callable[..., Awaitable[Union[str, bytes, Dict[str, JsonValue], None]]]
AsyncErrorHandler = Callable[[LogRecord, Exception], Awaitable[None]]
AsyncFilterFunc = Callable[[LogRecord], Awaitable[bool]]

LogProcessor = Callable[[LogRecord], Awaitable[Optional[LogRecord]]]
BatchProcessor = Callable[[List[LogRecord]], Awaitable[None]]

# HTTP content types with validation
HttpContentType: TypeAlias = Literal[
    "text/plain",
    "application/json",
    "application/x-protobuf",
    "application/octet-stream",
    "application/xml",
    "text/xml",
]

# File rotation types with more options
RotationType: TypeAlias = Literal["time", "size", "both"]

# Time rotation intervals with better documentation
TimeInterval: TypeAlias = Literal[
    "S", "M", "H", "D", "midnight", "W0", "W1", "W2", "W3", "W4", "W5", "W6"
]

# Time interval descriptions for better UX
TIME_INTERVAL_DESCRIPTIONS: Dict[TimeInterval, str] = {
    "S": "Seconds",
    "M": "Minutes",
    "H": "Hours",
    "D": "Days",
    "midnight": "Midnight",
    "W0": "Monday",
    "W1": "Tuesday",
    "W2": "Wednesday",
    "W3": "Thursday",
    "W4": "Friday",
    "W5": "Saturday",
    "W6": "Sunday",
}

# Enhanced abstract base class for async handlers with more methods
class AsyncHandlerABC(ABC):
    """
    Abstract base class for all async handlers.

    This class defines the interface that all async handlers must implement.
    It provides a consistent API for handling log records asynchronously.

    Methods:
        emit: Process a log record asynchronously
        close: Close the handler and release resources asynchronously
        setLevel: Set the logging level for the handler
        addFilter: Add a filter to the handler
        removeFilter: Remove a filter from the handler

    Example:
        >>> class MyAsyncHandler(AsyncHandlerABC):
        ...     async def emit(self, record):
        ...         # Process the log record
        ...         pass
        ...
        ...     async def close(self):
        ...         # Close the handler
        ...         pass
        ...
        ...     # ... implement other methods
    """

    @abstractmethod
    async def emit(self, record: LogRecord) -> None:
        """Process a log record asynchronously."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the handler and release resources asynchronously."""
        ...

    @abstractmethod
    def setLevel(self, level: int) -> None:
        """Set the logging level for the handler."""
        ...

    @abstractmethod
    def addFilter(self, filter: FilterProtocol) -> None:
        """Add a filter to the handler."""
        ...

    @abstractmethod
    def removeFilter(self, filter: FilterProtocol) -> None:
        """Remove a filter from the handler."""
        ...

# More specific factory types with better type safety
if sys.version_info >= (3, 9):
    HandlerFactory = Callable[..., Awaitable[HandlerType]]
    LoggerFactory = Callable[..., LoggerType]
    FormatterFactory = Callable[..., FormatterType]
    FilterFactory = Callable[..., FilterType]
else:
    # For Python 3.8 compatibility - use simple Callable without type parameters
    HandlerFactory = Callable
    LoggerFactory = Callable
    FormatterFactory = Callable
    FilterFactory = Callable

# Forward declarations for type checking with better organization
if TYPE_CHECKING:
    from .handlers.base import AsyncHandler
    from .logger import AsyncLogger

# Enhanced context manager type with generics
if sys.version_info >= (3, 9):
    AsyncContextManager = Callable[..., Awaitable[Any]]
    AsyncContextManagerGeneric = Callable[..., Awaitable[T]]
    LoopGetter = Callable[[], Optional[AbstractEventLoop]]
    ConfigValidator = Callable[[Dict[str, ConfigValue]], bool]
    RetryStrategy = Callable[[int, Exception], bool]
    RateLimiter = Callable[[], Awaitable[bool]]
else:
    # For Python 3.8 compatibility - use simple Callable without type parameters
    AsyncContextManager = Callable
    AsyncContextManagerGeneric = Callable
    LoopGetter = Callable
    ConfigValidator = Callable
    RetryStrategy = Callable
    RateLimiter = Callable

# Error handler type with more specific name
ErrorHandler = AsyncErrorHandler

# Type for HTTP session (forward reference)
if TYPE_CHECKING:
    from aiohttp import ClientSession

# Configuration types
class HandlerConfig(Generic[T]):
    """
    Configuration for async handlers.

    This class provides a configuration container for async handlers,
    allowing for consistent initialization and configuration across
    different handler types.

    Type Parameters:
        T: The type of handler class

    Attributes:
        handler_class: The handler class to instantiate
        level: The logging level for the handler
        formatter: The formatter to use with the handler
        filters: List of filters to apply to the handler
        kwargs: Additional keyword arguments for handler initialization

    Example:
        >>> config = HandlerConfig(
        ...     handler_class=MyHandler,
        ...     level=logging.INFO,
        ...     formatter=my_formatter,
        ...     filters=[my_filter],
        ...     custom_arg="value"
        ... )
    """

    def __init__(
        self,
        handler_class: Type[T],
        level: Optional[int] = None,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[List[FilterProtocol]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the handler configuration.

        Args:
            handler_class: The handler class to instantiate
            level: The logging level for the handler
            formatter: The formatter to use with the handler
            filters: List of filters to apply to the handler
            **kwargs: Additional keyword arguments for handler initialization
        """
        self.handler_class = handler_class
        self.level = level
        self.formatter = formatter
        self.filters = filters or []
        self.kwargs = kwargs

class LoggerConfig:
    """
    Configuration for async loggers.

    This class provides a configuration container for async loggers,
    allowing for consistent initialization and configuration across
    different logger instances.

    Attributes:
        name: The name of the logger
        level: The logging level for the logger
        handlers: List of handler configurations
        propagate: Whether to propagate messages to ancestor loggers
        disabled: Whether the logger is disabled

    Example:
        >>> config = LoggerConfig(
        ...     name="myapp",
        ...     level=logging.INFO,
        ...     handlers=[handler_config1, handler_config2],
        ...     propagate=False
        ... )
    """

    def __init__(
        self,
        name: str,
        level: Optional[int] = None,
        handlers: Optional[List[HandlerConfig[Any]]] = None,
        propagate: bool = True,
        disabled: bool = False,
    ) -> None:
        """
        Initialize the logger configuration.

        Args:
            name: The name of the logger
            level: The logging level for the logger
            handlers: List of handler configurations
            propagate: Whether to propagate messages to ancestor loggers
            disabled: Whether the logger is disabled
        """
        self.name = name
        self.level = level
        self.handlers = handlers or []
        self.propagate = propagate
        self.disabled = disabled

# Batch configuration
class BatchConfig:
    """
    Configuration for batch processing.

    This class provides configuration options for batch processing
    in handlers that support batching, such as HTTP handlers.

    Attributes:
        batch_size: The maximum number of records in a batch
        flush_interval: The time interval in seconds between flushes
        max_retries: The maximum number of retry attempts
        retry_delay: The delay in seconds between retry attempts

    Example:
        >>> config = BatchConfig(
        ...     batch_size=200,
        ...     flush_interval=10.0,
        ...     max_retries=5,
        ...     retry_delay=2.0
        ... )
    """

    def __init__(
        self,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize the batch configuration.

        Args:
            batch_size: The maximum number of records in a batch
            flush_interval: The time interval in seconds between flushes
            max_retries: The maximum number of retry attempts
            retry_delay: The delay in seconds between retry attempts
        """
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay

# HTTP configuration
class HttpConfig:
    """
    Configuration for HTTP handlers.

    This class provides configuration options for HTTP handlers,
    allowing for customization of HTTP requests and batch processing.

    Attributes:
        url: The URL to send log records to
        method: The HTTP method to use
        headers: The HTTP headers to include
        params: The query parameters to include
        timeout: The request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
        batch_config: Configuration for batch processing

    Example:
        >>> config = HttpConfig(
        ...     url="https://example.com/api/logs",
        ...     method="POST",
        ...     headers={"Content-Type": "application/json"},
        ...     timeout=30.0,
        ...     batch_config=BatchConfig(batch_size=50)
        ... )
    """

    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: Optional[HeadersType] = None,
        params: Optional[ParamsType] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        batch_config: Optional[BatchConfig] = None,
    ) -> None:
        """
        Initialize the HTTP configuration.

        Args:
            url: The URL to send log records to
            method: The HTTP method to use
            headers: The HTTP headers to include
            params: The query parameters to include
            timeout: The request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            batch_config: Configuration for batch processing
        """
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.params = params or {}
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.batch_config = batch_config or BatchConfig()

# File configuration
class FileConfig:
    """
    Configuration for file handlers.

    This class provides configuration options for file handlers,
    allowing for customization of file operations.

    Attributes:
        filename: The name of the file to write to
        mode: The file mode to use
        encoding: The file encoding to use
        delay: Whether to delay file opening until the first emit
        errors: The error handling strategy

    Example:
        >>> config = FileConfig(
        ...     filename="/var/log/app.log",
        ...     mode="a",
        ...     encoding="utf-8",
        ...     delay=False
        ... )
    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        encoding: str = "utf-8",
        delay: bool = False,
        errors: Optional[str] = None,
    ) -> None:
        """
        Initialize the file configuration.

        Args:
            filename: The name of the file to write to
            mode: The file mode to use
            encoding: The file encoding to use
            delay: Whether to delay file opening until the first emit
            errors: The error handling strategy
        """
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        self.errors = errors

# Rotation configuration
class RotationConfig:
    """
    Configuration for rotating file handlers.

    This class provides configuration options for rotating file handlers,
    allowing for customization of rotation behavior based on size or time.

    Attributes:
        max_bytes: The maximum file size in bytes before rotation
        backup_count: The number of backup files to keep
        when: The type of time-based rotation
        interval: The interval for time-based rotation
        backup_time: The specific time for daily rotation
        utc: Whether to use UTC time for time-based rotation

    Example:
        >>> config = RotationConfig(
        ...     max_bytes=10485760,  # 10MB
        ...     backup_count=5,
        ...     when="midnight",
        ...     interval=1,
        ...     utc=True
        ... )
    """

    def __init__(
        self,
        max_bytes: int = 0,
        backup_count: int = 0,
        when: Optional[TimeInterval] = None,
        interval: int = 1,
        backup_time: Optional[str] = None,
        utc: bool = False,
    ) -> None:
        """
        Initialize the rotation configuration.

        Args:
            max_bytes: The maximum file size in bytes before rotation
            backup_count: The number of backup files to keep
            when: The type of time-based rotation
            interval: The interval for time-based rotation
            backup_time: The specific time for daily rotation
            utc: Whether to use UTC time for time-based rotation
        """
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.when = when
        self.interval = interval
        self.backup_time = backup_time
        self.utc = utc
