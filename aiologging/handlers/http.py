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

from ..exceptions import (
    AuthenticationError,
    ConfigurationError,
    DependencyError,
    NetworkError,
)
from ..types import (
    AuthenticatorProtocol,
    BatchConfig,
    ErrorHandler,
    FilterProtocol,
    FormatterProtocol,
    HeadersType,
    HttpBackendType,
    HttpContentType,
    ParamsType,
)
from ..utils import LazyLock
from .base import BufferedAsyncHandler

# Try to import aiohttp
try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Try to import httpx
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Try to import protobuf
try:
    from google.protobuf import message
    assert message

    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False


def _check_aiohttp() -> None:
    """Check if aiohttp is available."""
    if not AIOHTTP_AVAILABLE:
        raise DependencyError(
            "aiohttp is required for the 'aiohttp' HTTP backend. "
            "Install it with: pip install aiologging[aiohttp]",
            dependency_name="aiohttp",
            install_command="pip install aiologging[aiohttp]",
        )


def _check_httpx() -> None:
    """Check if httpx is available."""
    if not HTTPX_AVAILABLE:
        raise DependencyError(
            "httpx is required for the 'httpx' HTTP backend. "
            "Install it with: pip install aiologging[httpx]",
            dependency_name="httpx",
            install_command="pip install aiologging[httpx]",
        )


def _resolve_backend(backend: Optional[HttpBackendType]) -> HttpBackendType:
    """
    Resolve the HTTP client backend to use.

    Args:
        backend: The explicitly requested backend, or None for
                 auto-detection (aiohttp is preferred when both
                 libraries are installed)

    Returns:
        The resolved backend name

    Raises:
        DependencyError: If the requested backend (or, with
                         auto-detection, any backend) is not installed
        ConfigurationError: If an unknown backend name is given
    """
    if backend is None:
        if AIOHTTP_AVAILABLE:
            return "aiohttp"
        if HTTPX_AVAILABLE:
            return "httpx"
        raise DependencyError(
            "An async HTTP client is required for HTTP handlers. "
            "Install one with: pip install aiologging[aiohttp] "
            "or pip install aiologging[httpx]",
            dependency_name="aiohttp|httpx",
            install_command="pip install aiologging[aiohttp]",
        )
    if backend == "aiohttp":
        _check_aiohttp()
        return backend
    if backend == "httpx":
        _check_httpx()
        return backend
    raise ConfigurationError(
        f"Unknown HTTP backend: {backend!r}. "
        "Supported backends: 'aiohttp', 'httpx'",
        config_key="backend",
        config_value=backend,
    )


class _HttpxResponseAdapter:
    """
    Adapts an httpx.Response to the aiohttp-style response interface
    used internally by the HTTP handlers (``status`` attribute and
    async ``text()`` method).
    """

    def __init__(self, response: "httpx.Response") -> None:
        self._response = response

    @property
    def status(self) -> int:
        """HTTP status code of the response."""
        return self._response.status_code

    async def text(self) -> str:
        """Response body decoded as text."""
        return self._response.text


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
        backend: Optional[HttpBackendType] = None,
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
            backend: HTTP client backend to use ('aiohttp' or 'httpx');
                     if None, aiohttp is used when installed,
                     falling back to httpx

        Raises:
            DependencyError: If the selected HTTP client is not installed
        """
        resolved_backend = _resolve_backend(backend)

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
        self.backend: HttpBackendType = resolved_backend
        # aiohttp.ClientSession or httpx.AsyncClient, per self.backend
        self._session: Optional[Any] = None
        self._session_lock = LazyLock()
        # Loop the session was created on; sessions cannot outlive it
        self._session_loop: Optional[asyncio.AbstractEventLoop] = None

    def _session_is_closed(self) -> bool:
        """Check whether the underlying HTTP client is missing or closed."""
        if self._session is None:
            return True
        if self.backend == "aiohttp":
            return bool(self._session.closed)
        return bool(self._session.is_closed)

    def _abandon_session(self) -> None:
        """
        Drop a session bound to a dead or foreign event loop.

        A proper ``close()`` needs the session's own loop, which is
        gone, so the best effort is releasing the transports
        synchronously and silencing the "unclosed" destructor warnings.
        """
        session = self._session
        self._session = None
        self._session_loop = None
        if session is None:
            return
        try:
            if self.backend == "aiohttp":
                connector = session.connector
                session.detach()
                if connector is not None:
                    connector._close()
        except Exception:
            pass

    def _create_session(self) -> Any:
        """Create a new HTTP client for the configured backend."""
        if self.backend == "aiohttp":
            timeout = ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                ssl=self.verify_ssl,
            )
            return ClientSession(
                timeout=timeout,
                connector=connector,
            )
        return httpx.AsyncClient(
            timeout=self.timeout,
            verify=self.verify_ssl,
        )

    async def _get_session(self) -> Any:
        """
        Get or create the HTTP client for the configured backend.

        A session created on a previous event loop (which has died or
        been replaced) is abandoned and a fresh one is created on the
        current loop. Externally injected sessions (``_session_loop``
        is None, e.g. a test transport) are left untouched.
        """
        loop = asyncio.get_running_loop()
        if (
            self._session is not None
            and self._session_loop is not None
            and self._session_loop is not loop
        ):
            self._abandon_session()
        if self._session_is_closed():
            async with self._session_lock:
                if self._session_is_closed():
                    self._session = self._create_session()
                    self._session_loop = loop
        return self._session

    async def _default_authenticator(
        self,
        session: Any,
        request_data: Any,
    ) -> Dict[str, str]:
        """
        Default authentication method.

        This method can be overridden by subclasses to provide
        default authentication behavior.

        Args:
            session: The HTTP client (aiohttp.ClientSession
                     or httpx.AsyncClient)
            request_data: The data to be sent

        Returns:
            Authentication headers
        """
        return {}

    async def _authenticate_request(
        self,
        session: Any,
        request_data: Any,
    ) -> Dict[str, str]:
        """
        Authenticate the request using the configured authenticator.

        Args:
            session: The HTTP client (aiohttp.ClientSession
                     or httpx.AsyncClient)
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

    def _redacted_url(self) -> str:
        """
        Get the URL as it may be shown in error messages and repr.

        Subclasses whose URL embeds a secret (e.g. a bot token)
        should override this to hide it.

        Returns:
            The URL safe for display
        """
        return self.url

    def _retry_delay_from_response(
        self, status: int, error_text: str
    ) -> Optional[float]:
        """
        Extract a server-provided retry delay from an error response.

        Called for every error response (status >= 400) before the
        retry decision is made. Returning a delay in seconds forces a
        retry after that delay — even for 4xx statuses that are
        otherwise not retried (e.g. 429 Too Many Requests). Returning
        None keeps the default policy: exponential backoff for 5xx,
        no retries for 4xx.

        Args:
            status: The HTTP status code of the response
            error_text: The response body as text

        Returns:
            The delay in seconds before the next attempt, or None
        """
        return None

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
                    url=self._redacted_url(),
                    details={"record_count": len(records)},
                ) from e

    async def _make_request_with_retries(
        self,
        session: Any,
        headers: HeadersType,
        request_data: Any,
        records: List[LogRecord],
    ) -> None:
        """
        Make HTTP request with retry logic.

        Args:
            session: The HTTP client (aiohttp.ClientSession
                     or httpx.AsyncClient)
            headers: Request headers
            request_data: The data to send
            records: The original log records (for error reporting)

        Raises:
            NetworkError: If all retries fail
        """
        last_exception = None

        for attempt in range(self.batch_config.max_retries + 1):
            retry_delay_override: Optional[float] = None
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
                    url=self._redacted_url(),
                    status_code=response.status,
                    details={"attempt": attempt + 1},
                )

                # A server-provided delay (e.g. 429 with retry-after)
                # forces a retry even for client errors
                retry_delay_override = self._retry_delay_from_response(
                    response.status, error_text
                )

                # Don't retry on client errors (4xx)
                if (
                    retry_delay_override is None
                    and 400 <= response.status < 500
                ):
                    break

            except Exception as e:
                last_exception = NetworkError(
                    f"HTTP request failed: {e}",
                    url=self._redacted_url(),
                    details={"attempt": attempt + 1},
                )

            # Wait before retry (except on the last attempt)
            if attempt < self.batch_config.max_retries:
                await asyncio.sleep(
                    retry_delay_override
                    if retry_delay_override is not None
                    else self.batch_config.retry_delay * (2**attempt)
                )

        # All retries failed
        raise last_exception or NetworkError(
            "HTTP request failed after all retries",
            url=self._redacted_url(),
            details={"record_count": len(records)},
        )

    async def _make_single_request(
        self,
        session: Any,
        headers: HeadersType,
        request_data: Any,
    ) -> Any:
        """
        Make a single HTTP request.

        Args:
            session: The HTTP client (aiohttp.ClientSession
                     or httpx.AsyncClient)
            headers: Request headers
            request_data: The data to send

        Returns:
            The HTTP response object with a ``status`` attribute
            and an async ``text()`` method
        """
        json_data = (
            request_data if isinstance(request_data, (dict, list)) else None
        )
        raw_data = (
            None if isinstance(request_data, (dict, list)) else request_data
        )

        if self.backend == "aiohttp":
            return await session.request(
                method=self.method,
                url=self.url,
                headers=headers,
                json=json_data,
                data=raw_data,
                params=self.params,
            )

        response = await session.request(
            method=self.method,
            url=self.url,
            headers=dict(headers),
            json=json_data,
            content=raw_data,
            params=self.params,
        )
        return _HttpxResponseAdapter(response)

    async def _close_resources(self) -> None:
        """Stop the auto-flush task and close the underlying client."""
        await super()._close_resources()
        if self._session is None or self._session_is_closed():
            return
        if (
            self._session_loop is not None
            and self._session_loop is not asyncio.get_running_loop()
        ):
            # The session's loop is gone; a graceful close is
            # impossible, release what can be released synchronously
            self._abandon_session()
            return
        if self.backend == "aiohttp":
            await self._session.close()
        else:
            await self._session.aclose()
        self._session_loop = None

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        formatter: Union[str, None] = None
        if self.formatter:
            formatter = type(self.formatter).__name__
        return (
            f"{self.__class__.__name__}(url='{self._redacted_url()}', "
            f"method='{self.method}', backend='{self.backend}', "
            f"level={self.level}, formatter={formatter})"
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
        backend: Optional[HttpBackendType] = None,
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
            backend: HTTP client backend to use ('aiohttp' or 'httpx')

        Raises:
            DependencyError: If the HTTP client or protobuf
                             is not installed
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
            backend=backend,
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
        backend: Optional[HttpBackendType] = None,
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
            backend: HTTP client backend to use ('aiohttp' or 'httpx');
                     if None, aiohttp is used when installed,
                     falling back to httpx
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
            backend=backend,
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
            backend=self.backend,
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
            backend=self.backend,
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
                backend=self.backend,
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
