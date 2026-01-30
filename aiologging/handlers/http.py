"""
Async HTTP handlers for aiologging.

This module provides async handlers for sending log records to HTTP endpoints
with support for different formats and extensible authentication.
"""

from __future__ import annotations

import asyncio
import json
from abc import abstractmethod
from logging import LogRecord, NOTSET
from typing import Any, Dict, List, Optional, Union

from ..exceptions import AuthenticationError, DependencyError, NetworkError
from ..types import (
    AuthenticatorProtocol,
    BatchConfig,
    ErrorHandler,
    FilterProtocol,
    FormatterProtocol,
    HeadersType,
    HttpContentType,
    ParamsType,
)
from .base import BufferedAsyncHandler

# Try to import aiohttp
try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore

# Try to import protobuf
try:
    from google.protobuf import message

    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    message = None  # type: ignore


def _check_aiohttp() -> None:
    """Check if aiohttp is available."""
    if not AIOHTTP_AVAILABLE:
        raise DependencyError(
            "aiohttp is required for HTTP handlers. "
            "Install it with: pip install aiologging[aiohttp]",
            dependency_name="aiohttp",
            install_command="pip install aiologging[aiohttp]",
        )


def _check_protobuf() -> None:
    """Check if protobuf is available."""
    if not PROTOBUF_AVAILABLE:
        raise DependencyError(
            "protobuf is required for AsyncHttpProtoHandler. "
            "Install it with: pip install aiologging[protobuf]",
            dependency_name="protobuf",
            install_command="pip install aiologging[protobuf]",
        )


class AsyncHttpHandlerBase(BufferedAsyncHandler):
    """
    Base class for async HTTP handlers with authentication support.

    This handler sends log records to HTTP endpoints with support for
    custom authentication methods and different content formats.
    """

    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: Optional[HeadersType] = None,
        params: Optional[ParamsType] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        authenticator: Optional[AuthenticatorProtocol[Any, Any]] = None,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[List[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
        batch_config: Optional[BatchConfig] = None,
    ) -> None:
        """
        Initialize the async HTTP handler.

        Args:
            url: The HTTP endpoint URL
            method: The HTTP method (defaults to 'POST')
            headers: Additional HTTP headers
            params: URL query parameters
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            authenticator: Optional authentication function
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions
            batch_config: Configuration for batch processing

        Raises:
            DependencyError: If aiohttp is not installed
        """
        _check_aiohttp()

        super().__init__(
            level=level,
            formatter=formatter,
            filters=filters,
            error_handler=error_handler,
            buffer_size=batch_config.batch_size if batch_config else 100,
            flush_interval=(
                batch_config.flush_interval if batch_config else 5.0
            ),
            auto_flush=True,
        )

        self.url = url
        self.method = method.upper()
        self.headers = dict(headers) if headers else {}
        self.params = params
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.authenticator = authenticator or self._default_authenticator
        self.batch_config = batch_config or BatchConfig()
        self._session: Optional[ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = ClientTimeout(total=self.timeout)
                    connector = aiohttp.TCPConnector(
                        verify_ssl=self.verify_ssl,
                    )
                    self._session = ClientSession(
                        timeout=timeout,
                        connector=connector,
                    )
        return self._session

    async def _default_authenticator(
        self,
        session: ClientSession,
        request_data: Any,
    ) -> Dict[str, str]:
        """
        Default authentication method.

        This method can be overridden by subclasses to provide
        default authentication behavior.

        Args:
            session: The aiohttp session
            request_data: The data to be sent

        Returns:
            Authentication headers
        """
        return {}

    async def _authenticate_request(
        self,
        session: ClientSession,
        request_data: Any,
    ) -> Dict[str, str]:
        """
        Authenticate the request using the configured authenticator.

        Args:
            session: The aiohttp session
            request_data: The data to be sent

        Returns:
            Authentication headers

        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            return await self.authenticator(session, request_data)
        except Exception as e:
            raise AuthenticationError(
                f"Authentication failed: {e}",
                auth_type=(
                    type(self.authenticator).__name__
                    if self.authenticator
                    else "default"
                ),
            ) from e

    @abstractmethod
    async def _prepare_request_data(self, records: List[LogRecord]) -> Any:
        """
        Prepare request data from log records.

        This method must be implemented by subclasses to define
        how log records are converted to the request format.

        Args:
            records: The list of log records to send

        Returns:
            The prepared request data
        """
        raise NotImplementedError

    @abstractmethod
    def _get_content_type(self) -> str:
        """
        Get the content type for this handler.

        Returns:
            The content type string
        """
        raise NotImplementedError

    async def flush(self, records: Optional[List[LogRecord]] = None) -> None:
        """
        Flush a batch of log records to the HTTP endpoint.

        Args:
            records: The list of log records to send (if None, flush buffer)

        Raises:
            NetworkError: If the HTTP request fails
        """
        # If no records provided, use the parent's force_flush method
        if records is None:
            await super().force_flush()
            return

        if not records:
            return

        session = await self._get_session()

        try:
            # Prepare request data
            request_data = await self._prepare_request_data(records)

            # Get authentication headers
            auth_headers = await self._authenticate_request(
                session, request_data
            )

            # Merge headers
            headers = dict(self.headers)
            headers.update(auth_headers)
            headers["Content-Type"] = self._get_content_type()

            # Make the HTTP request with retries
            await self._make_request_with_retries(
                session, headers, request_data, records
            )

        except Exception as e:
            if isinstance(e, (NetworkError, AuthenticationError)):
                raise
            else:
                raise NetworkError(
                    f"Failed to send log records: {e}",
                    url=self.url,
                    details={"record_count": len(records)},
                ) from e

    async def _make_request_with_retries(
        self,
        session: ClientSession,
        headers: HeadersType,
        request_data: Any,
        records: List[LogRecord],
    ) -> None:
        """
        Make HTTP request with retry logic.

        Args:
            session: The aiohttp session
            headers: Request headers
            request_data: The data to send
            records: The original log records (for error reporting)

        Raises:
            NetworkError: If all retries fail
        """
        last_exception = None

        for attempt in range(self.batch_config.max_retries + 1):
            try:
                response = await self._make_single_request(
                    session, headers, request_data
                )
                if response.status < 400:
                    return  # Success

                # Handle HTTP error
                error_text = await response.text()
                last_exception = NetworkError(
                    (
                        "HTTP request failed with status "
                        f"{response.status}: {error_text}"
                    ),
                    url=self.url,
                    status_code=response.status,
                    details={"attempt": attempt + 1},
                )

                # Don't retry on client errors (4xx)
                if 400 <= response.status < 500:
                    break

            except Exception as e:
                last_exception = NetworkError(
                    f"HTTP request failed: {e}",
                    url=self.url,
                    details={"attempt": attempt + 1},
                )

            # Wait before retry (except on the last attempt)
            if attempt < self.batch_config.max_retries:
                await asyncio.sleep(
                    self.batch_config.retry_delay * (2**attempt)
                )

        # All retries failed
        raise last_exception or NetworkError(
            "HTTP request failed after all retries",
            url=self.url,
            details={"record_count": len(records)},
        )

    async def _make_single_request(
        self,
        session: ClientSession,
        headers: HeadersType,
        request_data: Any,
    ) -> Any:
        """
        Make a single HTTP request.

        Args:
            session: The aiohttp session
            headers: Request headers
            request_data: The data to send

        Returns:
            The HTTP response object
        """
        return await session.request(
            method=self.method,
            url=self.url,
            headers=headers,
            json=(
                request_data
                if isinstance(request_data, (dict, list))
                else None
            ),
            data=(
                request_data
                if not isinstance(request_data, (dict, list))
                else None
            ),
            params=self.params,
        )

    async def _close_resources(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        formatter: Union[str, None] = None
        if self.formatter:
            formatter = type(self.formatter).__name__
        return (
            f"{self.__class__.__name__}(url='{self.url}', "
            f"method='{self.method}', level={self.level}, "
            f"formatter={formatter})"
        )


class AsyncHttpTextHandler(AsyncHttpHandlerBase):
    """
    Async HTTP handler that sends log records as plain text.
    """

    async def emit(self, record: LogRecord) -> None:
        """
        Emit a log record by adding it to the buffer.

        Args:
            record: The log record to emit
        """
        await self.handle(record)

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        """Emit a log record (buffered handler uses flush instead)."""
        # This is a buffered handler, so individual records
        # are not emitted directly
        # They are collected and sent in batches via flush()
        pass

    async def _prepare_request_data(self, records: List[LogRecord]) -> str:
        """Prepare request data as plain text."""
        lines = []
        for record in records:
            formatted = self.format(record)
            lines.append(formatted)
        return "\n".join(lines)

    def _get_content_type(self) -> str:
        """Get the content type for plain text."""
        return "text/plain"


class AsyncHttpJsonHandler(AsyncHttpHandlerBase):
    """
    Async HTTP handler that sends log records as JSON.
    """

    async def emit(self, record: LogRecord) -> None:
        """
        Emit a log record by adding it to the buffer.

        Args:
            record: The log record to emit
        """
        await self.handle(record)

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        """Emit a log record (buffered handler uses flush instead)."""
        # This is a buffered handler, so individual records
        # are not emitted directly
        # They are collected and sent in batches via flush()
        pass

    def _convert_record_to_dict(self, record: LogRecord) -> Dict[str, Any]:
        """Convert a log record to a dictionary for JSON serialization."""
        json_record = {
            "name": record.name,
            "level": record.levelname,
            "levelno": record.levelno,
            "pathname": record.pathname,
            "filename": record.filename,
            "module": record.module,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "created": record.created,
            "msecs": record.msecs,
            "relativeCreated": record.relativeCreated,
            "thread": record.thread,
            "threadName": record.threadName,
            "processName": record.processName,
            "process": record.process,
            "message": record.getMessage(),
            "formatted": self.format(record),
        }

        # Add exception info if present
        if record.exc_info:
            if self.formatter and hasattr(self.formatter, "formatException"):
                json_record["exc_info"] = self.formatter.formatException(
                    record.exc_info
                )
            else:
                json_record["exc_info"] = str(record.exc_info)

        # Add stack trace if present
        if record.stack_info:
            json_record["stack_info"] = record.stack_info

        return json_record

    async def _prepare_request_data(
        self, records: List[LogRecord]
    ) -> List[Dict[str, Any]]:
        """Prepare request data as JSON."""
        return [self._convert_record_to_dict(record) for record in records]

    def _get_content_type(self) -> str:
        """Get the content type for JSON."""
        return "application/json"


class AsyncHttpProtoHandler(AsyncHttpHandlerBase):
    """
    Async HTTP handler that sends log records as Protocol Buffers.

    This handler requires the protobuf library to be installed.
    """

    def __init__(
        self,
        url: str,
        proto_message_class: Optional[type] = None,
        method: str = "POST",
        headers: Optional[HeadersType] = None,
        params: Optional[ParamsType] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        authenticator: Optional[AuthenticatorProtocol[Any, Any]] = None,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[List[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
        batch_config: Optional[BatchConfig] = None,
    ) -> None:
        """
        Initialize the async HTTP protobuf handler.

        Args:
            url: The HTTP endpoint URL
            proto_message_class: The protobuf message class to use
            method: The HTTP method (defaults to 'POST')
            headers: Additional HTTP headers
            params: URL query parameters
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            authenticator: Optional authentication function
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions
            batch_config: Configuration for batch processing

        Raises:
            DependencyError: If aiohttp or protobuf is not installed
        """
        _check_protobuf()

        super().__init__(
            url,
            method,
            headers,
            params,
            timeout,
            verify_ssl,
            authenticator,
            level,
            formatter,
            filters,
            error_handler,
            batch_config,
        )

        self.proto_message_class = proto_message_class

    async def emit(self, record: LogRecord) -> None:
        """
        Emit a log record by adding it to the buffer.

        Args:
            record: The log record to emit
        """
        await self.handle(record)

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        """Emit a log record (buffered handler uses flush instead)."""
        # This is a buffered handler, so individual records
        # are not emitted directly
        # They are collected and sent in batches via flush()
        pass

    async def _prepare_request_data(self, records: List[LogRecord]) -> bytes:
        """Prepare request data as Protocol Buffers."""
        if self.proto_message_class is None:
            # Create a simple protobuf message if no class is provided
            return self._create_simple_proto_message(records)
        else:
            # Use the provided protobuf message class
            return await self._create_custom_proto_message(records)

    def _create_simple_proto_message(self, records: List[LogRecord]) -> bytes:
        """Create a simple protobuf message from log records."""
        # This is a simplified implementation
        # In a real implementation, you would define proper protobuf messages
        json_data = []
        for record in records:
            json_record = {
                "timestamp": int(record.created * 1000),  # milliseconds
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "formatted": self.format(record),
            }
            json_data.append(json_record)

        # Convert to JSON and then to bytes (as a simple protobuf alternative)
        return json.dumps(json_data).encode("utf-8")

    async def _create_custom_proto_message(
        self, records: List[LogRecord]
    ) -> bytes:
        """Create a custom protobuf message from log records."""
        # This would be implemented based
        # on the specific protobuf message class
        # For now, fall back to the simple implementation
        return self._create_simple_proto_message(records)

    def _get_content_type(self) -> str:
        """Get the content type for Protocol Buffers."""
        return "application/x-protobuf"


class AsyncHttpHandler(AsyncHttpHandlerBase):
    """
    Universal async HTTP handler with automatic format detection.

    This handler automatically determines the format based on:
    - Content-Type header
    - Analysis of the data being sent
    - Explicit format specification
    """

    def __init__(
        self,
        url: str,
        format_type: Optional[HttpContentType] = None,
        method: str = "POST",
        headers: Optional[HeadersType] = None,
        params: Optional[ParamsType] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        authenticator: Optional[AuthenticatorProtocol[Any, Any]] = None,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[List[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
        batch_config: Optional[BatchConfig] = None,
    ) -> None:
        """
        Initialize the universal async HTTP handler.

        Args:
            url: The HTTP endpoint URL
            format_type: Explicit format type (auto-detected if None)
            method: The HTTP method (defaults to 'POST')
            headers: Additional HTTP headers
            params: URL query parameters
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            authenticator: Optional authentication function
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions
            batch_config: Configuration for batch processing
        """
        super().__init__(
            url,
            method,
            headers,
            params,
            timeout,
            verify_ssl,
            authenticator,
            level,
            formatter,
            filters,
            error_handler,
            batch_config,
        )

        self.format_type = format_type
        self._handlers: Dict[str, AsyncHttpHandlerBase] = {}

        # Create handlers for different formats
        self._handlers["text/plain"] = AsyncHttpTextHandler(
            url,
            method,
            headers,
            params,
            timeout,
            verify_ssl,
            authenticator,
            level,
            formatter,
            filters,
            error_handler,
            batch_config,
        )
        self._handlers["application/json"] = AsyncHttpJsonHandler(
            url,
            method,
            headers,
            params,
            timeout,
            verify_ssl,
            authenticator,
            level,
            formatter,
            filters,
            error_handler,
            batch_config,
        )

        # Only create proto handler if protobuf is available
        if PROTOBUF_AVAILABLE:
            self._handlers["application/x-protobuf"] = AsyncHttpProtoHandler(
                url,
                None,
                method,
                headers,
                params,
                timeout,
                verify_ssl,
                authenticator,
                level,
                formatter,
                filters,
                error_handler,
                batch_config,
            )

    async def emit(self, record: LogRecord) -> None:
        """
        Emit a log record by adding it to the buffer.

        Args:
            record: The log record to emit
        """
        await self.handle(record)

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        """Emit a log record (buffered handler uses flush instead)."""
        # This is a buffered handler, so individual records
        # are not emitted directly
        # They are collected and sent in batches via flush()
        pass

    def _detect_format_type(self) -> HttpContentType:
        """
        Detect the appropriate format type.

        Returns:
            The detected format type
        """
        if self.format_type:
            return self.format_type

        # Check Content-Type header
        content_type = self.headers.get("Content-Type", "").lower()
        if "json" in content_type:
            return "application/json"
        elif "protobuf" in content_type or "x-protobuf" in content_type:
            return "application/x-protobuf"
        else:
            # Default to text/plain
            return "text/plain"

    async def _prepare_request_data(self, records: List[LogRecord]) -> Any:
        """Prepare request data using the detected format."""
        format_type = self._detect_format_type()
        handler = self._handlers.get(format_type, self._handlers["text/plain"])
        return await handler._prepare_request_data(records)

    def _get_content_type(self) -> str:
        """Get the content type for the detected format."""
        return self._detect_format_type()

    async def _close_resources(self) -> None:
        """Close all handler resources."""
        await super()._close_resources()
        for handler in self._handlers.values():
            await handler._close_resources()
