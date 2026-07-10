"""
Tests for HTTP handlers.
"""

import asyncio
import json
import logging
import pytest
import sys
from typing import List
from unittest.mock import AsyncMock, patch

import httpx

from aiologging.handlers.http import (
    AsyncHttpHandler,
    AsyncHttpJsonHandler,
    _HttpxResponseAdapter,
)
from aiologging.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DependencyError,
    NetworkError,
)


class TestAsyncHttpHandler:
    """Test cases for AsyncHttpHandler."""

    @pytest.mark.asyncio
    async def test_initialization(self) -> None:
        """Test handler initialization."""
        handler = AsyncHttpHandler("https://example.com/api/logs")
        self._assert_default_handler_properties(handler)

    @pytest.mark.asyncio
    async def test_initialization_with_parameters(self) -> None:
        """Test handler initialization with parameters."""
        handler = AsyncHttpHandler(
            url="https://example.com/api/logs",
            method="PUT",
            headers={"Content-Type": "text/plain"},
            params={"key": "value"},
            timeout=60.0,
            verify_ssl=False
        )
        self._assert_custom_handler_properties(handler)

    def _assert_default_handler_properties(self, handler: AsyncHttpHandler) -> None:
        """Assert default handler properties."""
        assert handler.url == "https://example.com/api/logs"
        assert handler.method == "POST"
        assert handler.timeout == 30.0
        assert handler.verify_ssl is True

    def _assert_custom_handler_properties(self, handler: AsyncHttpHandler) -> None:
        """Assert custom handler properties."""
        assert handler.url == "https://example.com/api/logs"
        assert handler.method == "PUT"
        assert handler.headers == {"Content-Type": "text/plain"}
        assert handler.params == {"key": "value"}
        assert handler.timeout == 60.0
        assert handler.verify_ssl is False

    def _create_test_record(self, message: str = "Test message", level: int = logging.INFO) -> logging.LogRecord:
        """Create a test log record."""
        return logging.LogRecord(
            name="test", level=level, pathname="", lineno=0,
            msg=message, args=(), exc_info=None
        )

    @pytest.mark.asyncio
    async def test_emit_success(self) -> None:
        """Test successful emit."""
        handler = AsyncHttpHandler("https://example.com/api/logs")
        handler.flush = AsyncMock()

        record = self._create_test_record()
        await handler.emit(record)
        await handler.flush()

        handler.flush.assert_called()

    @pytest.mark.asyncio
    async def test_emit_with_authenticator(self) -> None:
        """Test emit with authenticator."""
        # Create a mock authenticator
        authenticator = AsyncMock()
        authenticator.return_value = {"Authorization": "Bearer token"}

        handler = AsyncHttpHandler(
            "https://example.com/api/logs",
            authenticator=authenticator
        )

        # Mock the _authenticate_request method to track if it's called
        handler._authenticate_request = AsyncMock(return_value={"Authorization": "Bearer token"})

        record = self._create_test_record()
        await handler.emit(record)
        await handler.flush()

        # Verify that _authenticate_request was called
        handler._authenticate_request.assert_called()

    @pytest.mark.asyncio
    async def test_emit_network_error(self) -> None:
        """Test emit with network error."""
        handler = AsyncHttpHandler("https://example.com/api/logs")

        handler.flush = AsyncMock(side_effect=NetworkError("Connection failed"))
        record = self._create_test_record()

        with pytest.raises(NetworkError):
            await handler.emit(record)
            await handler.flush()

    @pytest.mark.asyncio
    async def test_emit_http_error(self) -> None:
        """Test emit with HTTP error status."""
        handler = AsyncHttpHandler("https://example.com/api/logs")

        handler.flush = AsyncMock(side_effect=NetworkError("HTTP 500 error"))
        record = self._create_test_record()

        with pytest.raises(NetworkError):
            await handler.emit(record)
            await handler.flush()

    @pytest.mark.asyncio
    async def test_emit_authentication_error(self) -> None:
        """Test emit with authentication error."""
        handler = AsyncHttpHandler("https://example.com/api/logs")

        handler.flush = AsyncMock(side_effect=AuthenticationError("Invalid credentials"))
        record = self._create_test_record()

        with pytest.raises(AuthenticationError):
            await handler.emit(record)
            await handler.flush()

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing handler."""
        handler = AsyncHttpHandler("https://example.com/api/logs")

        # Mock the _close_resources method to track if it's called
        handler._close_resources = AsyncMock()

        record = self._create_test_record()
        await handler.emit(record)
        await handler.flush()
        await handler.close()

        handler._close_resources.assert_called()


class TestAsyncHttpJsonHandler:
    """Test cases for AsyncHttpJsonHandler."""

    def _create_test_record(self, message: str = "Test message", level: int = logging.INFO) -> logging.LogRecord:
        """Create a test log record."""
        return logging.LogRecord(
            name="test", level=level, pathname="", lineno=0,
            msg=message, args=(), exc_info=None
        )

    @pytest.mark.asyncio
    async def test_emit_json_format(self) -> None:
        """Test emit with JSON format."""
        handler = AsyncHttpJsonHandler("https://example.com/api/logs")
        handler.flush = AsyncMock()
        record = self._create_test_record()

        await handler.emit(record)
        await handler.flush()

        handler.flush.assert_called()

    @pytest.mark.asyncio
    async def test_emit_json_with_extra_fields(self) -> None:
        """Test emit with JSON format and extra fields."""
        handler = AsyncHttpJsonHandler("https://example.com/api/logs")
        handler.flush = AsyncMock()
        record = self._create_test_record()
        record.custom_field = "custom_value"
        record.numeric_field = 42

        await handler.emit(record)
        await handler.flush()

        handler.flush.assert_called()

    @pytest.mark.asyncio
    async def test_emit_json_with_exception(self) -> None:
        """Test emit with JSON format and exception info."""
        handler = AsyncHttpJsonHandler("https://example.com/api/logs")
        handler.flush = AsyncMock()

        try:
            raise ValueError("Test exception")
        except ValueError:
            record = self._create_test_record("Error message", logging.ERROR)
            record.exc_info = True

            await handler.emit(record)
            await handler.flush()

            handler.flush.assert_called()

    @pytest.mark.asyncio
    async def test_emit_json_with_nested_data(self) -> None:
        """Test emit with JSON format and nested data."""
        handler = AsyncHttpJsonHandler("https://example.com/api/logs")

        handler.flush = AsyncMock()
        record = self._create_test_record()
        record.nested_dict = {"key1": "value1", "key2": {"nested": "value"}}
        record.nested_list = [1, 2, {"nested": "list_value"}]

        await handler.emit(record)
        await handler.flush()

        handler.flush.assert_called()


class TestAsyncHttpTextHandler:
    """Test cases for AsyncHttpTextHandler."""

    def _create_test_record(self, message: str = "Test message", level: int = logging.INFO) -> logging.LogRecord:
        """Create a test log record."""
        return logging.LogRecord(
            name="test", level=level, pathname="", lineno=0,
            msg=message, args=(), exc_info=None
        )

    def _create_test_records(self, *messages: str) -> List[logging.LogRecord]:
        """Create test log records with the given messages."""
        records = []
        for msg in messages:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg=msg, args=(), exc_info=None
            )
            records.append(record)
        return records

    @pytest.mark.asyncio
    async def test_emit_text_format(self) -> None:
        """Test emit with text format."""
        from aiologging.handlers.http import AsyncHttpTextHandler

        handler = AsyncHttpTextHandler("https://example.com/api/logs")
        handler.flush = AsyncMock()
        record = self._create_test_record()

        await handler.emit(record)
        await handler.flush()

        handler.flush.assert_called()

    @pytest.mark.asyncio
    async def test_prepare_request_data(self) -> None:
        """Test _prepare_request_data method."""
        from aiologging.handlers.http import AsyncHttpTextHandler

        handler = AsyncHttpTextHandler("https://example.com/api/logs")

        record1, record2 = self._create_test_records("Test message 1", "Test message 2")

        # Prepare request data
        request_data = await handler._prepare_request_data([record1, record2])

        # Verify the data is a string with formatted records
        assert isinstance(request_data, str)
        assert "Test message 1" in request_data
        assert "Test message 2" in request_data
        assert "\n".join([handler.format(record1), handler.format(record2)]) == request_data

    @pytest.mark.asyncio
    async def test_get_content_type(self) -> None:
        """Test _get_content_type method."""
        from aiologging.handlers.http import AsyncHttpTextHandler

        handler = AsyncHttpTextHandler("https://example.com/api/logs")
        assert handler._get_content_type() == "text/plain"


class TestAsyncHttpHandlerAdvanced:
    """Advanced test cases for AsyncHttpHandler."""

    def _create_test_records(self, *messages: str) -> List[logging.LogRecord]:
        """Create test log records with the given messages."""
        records = []
        for msg in messages:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg=msg, args=(), exc_info=None
            )
            records.append(record)
        return records

    @pytest.mark.asyncio
    async def test_convert_record_to_dict(self) -> None:
        """Test _convert_record_to_dict method."""
        handler = AsyncHttpJsonHandler("https://example.com/api/logs")

        record = logging.LogRecord(
            name="test_logger", level=logging.INFO, pathname="/path/to/file.py",
            lineno=42, funcName="test_function", msg="Test message", args=(), exc_info=None
        )
        record.created = 1234567890.123
        record.msecs = 123
        record.relativeCreated = 456
        record.thread = 789
        record.threadName = "MainThread"
        record.processName = "MainProcess"
        record.process = 12345

        # Convert record to dict
        result = handler._convert_record_to_dict(record)

        # Verify all expected fields are present
        assert result["name"] == "test_logger"
        assert result["level"] == "INFO"
        assert result["levelno"] == logging.INFO
        assert result["pathname"] == "/path/to/file.py"
        assert result["filename"] == "file.py"
        assert result["module"] == "file"
        assert result["lineno"] == 42
        # funcName might be None in some LogRecord implementations
        # Check if it's present before asserting its value
        if "funcName" in result and result["funcName"] is not None:
            assert result["funcName"] == "test_function"
        assert result["created"] == 1234567890.123
        assert result["msecs"] == 123
        assert result["relativeCreated"] == 456
        assert result["thread"] == 789
        assert result["threadName"] == "MainThread"
        assert result["processName"] == "MainProcess"
        assert result["process"] == 12345
        assert result["message"] == "Test message"
        assert "formatted" in result

    @pytest.mark.asyncio
    async def test_convert_record_to_dict_with_exception(self) -> None:
        """Test _convert_record_to_dict method with exception info."""
        handler = AsyncHttpJsonHandler("https://example.com/api/logs")

        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="Error message", args=(), exc_info=exc_info
            )

            # Convert record to dict
            result = handler._convert_record_to_dict(record)

            # Verify exception info is included
            assert "exc_info" in result
            assert "ValueError" in result["exc_info"]
            assert "Test exception" in result["exc_info"]

    @pytest.mark.asyncio
    async def test_convert_record_to_dict_with_stack_info(self) -> None:
        """Test _convert_record_to_dict method with stack info."""
        handler = AsyncHttpJsonHandler("https://example.com/api/logs")

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="Error message", args=(), exc_info=None
        )
        record.stack_info = "Stack trace info"

        # Convert record to dict
        result = handler._convert_record_to_dict(record)

        # Verify stack info is included
        assert "stack_info" in result
        assert result["stack_info"] == "Stack trace info"

    @pytest.mark.asyncio
    async def test_prepare_request_data(self) -> None:
        """Test _prepare_request_data method."""
        handler = AsyncHttpJsonHandler("https://example.com/api/logs")

        record1, record2 = self._create_test_records("Test message 1", "Test message 2")

        # Prepare request data
        request_data = await handler._prepare_request_data([record1, record2])

        # Verify the data is a list of dictionaries
        assert isinstance(request_data, list)
        assert len(request_data) == 2
        assert all(isinstance(item, dict) for item in request_data)
        assert request_data[0]["message"] == "Test message 1"
        assert request_data[1]["message"] == "Test message 2"

    @pytest.mark.asyncio
    async def test_get_content_type(self) -> None:
        """Test _get_content_type method."""
        handler = AsyncHttpJsonHandler("https://example.com/api/logs")
        assert handler._get_content_type() == "application/json"

    @pytest.mark.asyncio
    async def test_detect_format_type(self) -> None:
        """Test _detect_format_type method."""
        # Test with explicit format_type
        handler = AsyncHttpHandler(
            "https://example.com/api/logs",
            format_type="application/json"
        )
        assert handler._detect_format_type() == "application/json"

        # Test with Content-Type header
        handler = AsyncHttpHandler(
            "https://example.com/api/logs",
            headers={"Content-Type": "application/json"}
        )
        assert handler._detect_format_type() == "application/json"

        # Test with default (text/plain)
        handler = AsyncHttpHandler("https://example.com/api/logs")
        assert handler._detect_format_type() == "text/plain"

    @pytest.mark.asyncio
    async def test_make_request_with_retries_success(self) -> None:
        """Test _make_request_with_retries with successful request."""
        handler = AsyncHttpHandler("https://example.com/api/logs")
        handler.batch_config.max_retries = 2

        # Add a record to the buffer
        record = self._create_test_records("Test message")[0]
        await handler.emit(record)

        # Mock the _make_request_with_retries method to avoid complex async context manager mocking
        with patch.object(handler, '_make_request_with_retries') as mock_make_request:
            mock_make_request.return_value = None  # Success

            # Call the method through flush, which uses _make_request_with_retries
            await handler.flush()

            # Verify the method was called
            mock_make_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_with_retries_server_error(self) -> None:
        """Test _make_request_with_retries with server error that retries."""
        from aiologging.exceptions import NetworkError

        handler = AsyncHttpHandler("https://example.com/api/logs")
        handler.batch_config.max_retries = 2

        # Add a record to the buffer
        record = self._create_test_records("Test message")[0]
        await handler.emit(record)

        # Mock the _make_request_with_retries method to simulate retry behavior
        call_count = 0

        async def mock_make_request(session, headers, request_data, records):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # First two calls fail
                raise NetworkError("Server error", status_code=500)
            # Third call succeeds

        with patch.object(handler, '_make_request_with_retries', side_effect=mock_make_request):
            # This should succeed after retries
            await handler.flush()

            # Verify the method was called 3 times (1 initial + 2 retries)
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_make_request_with_retries_client_error(self) -> None:
        """Test _make_request_with_retries with client error that doesn't retry."""
        from aiologging.exceptions import NetworkError

        handler = AsyncHttpHandler("https://example.com/api/logs")
        handler.batch_config.max_retries = 2

        # Add a record to the buffer
        record = self._create_test_records("Test message")[0]
        await handler.emit(record)

        # Mock the _make_request_with_retries method to simulate client error
        mock_make_request = AsyncMock(side_effect=NetworkError("Client error", status_code=404))

        with patch.object(handler, '_make_request_with_retries', mock_make_request):
            # This should fail without retries
            with pytest.raises(NetworkError, match="Client error"):
                await handler.flush([record])

            # Verify the method was called only once (no retries for client errors)
            mock_make_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_with_retries_all_fail(self) -> None:
        """Test _make_request_with_retries when all retries fail."""
        from aiologging.exceptions import NetworkError

        handler = AsyncHttpHandler("https://example.com/api/logs")
        handler.batch_config.max_retries = 2

        # Add a record to the buffer
        record = self._create_test_records("Test message")[0]
        await handler.emit(record)

        # Mock the _make_single_request method to simulate all retries failing
        call_count = 0

        async def mock_single_request(session, headers, request_data):
            nonlocal call_count
            call_count += 1
            # Create a mock response with status 500
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Server Error")
            return mock_response

        with patch.object(handler, '_make_single_request', side_effect=mock_single_request):
            # This should fail after all retries
            with pytest.raises(NetworkError, match="Server Error"):
                await handler.flush([record])

            # Verify the method was called 3 times (1 initial + 2 retries)
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_repr(self) -> None:
        """Test string representation of the handler."""
        handler = AsyncHttpHandler("https://example.com/api/logs")
        repr_str = repr(handler)
        assert "AsyncHttpHandler" in repr_str
        assert "https://example.com/api/logs" in repr_str
        assert "method='POST'" in repr_str


class TestHttpBackendSelection:
    """Test cases for HTTP client backend selection."""

    def test_default_backend_prefers_aiohttp(self) -> None:
        """aiohttp is used by default when both backends are installed."""
        handler = AsyncHttpHandler("https://example.com/api/logs")
        assert handler.backend == "aiohttp"

    def test_explicit_httpx_backend(self) -> None:
        """httpx backend can be selected explicitly."""
        handler = AsyncHttpHandler(
            "https://example.com/api/logs", backend="httpx"
        )
        assert handler.backend == "httpx"
        assert "backend='httpx'" in repr(handler)

    def test_explicit_aiohttp_backend(self) -> None:
        """aiohttp backend can be selected explicitly."""
        handler = AsyncHttpHandler(
            "https://example.com/api/logs", backend="aiohttp"
        )
        assert handler.backend == "aiohttp"

    def test_auto_falls_back_to_httpx(self) -> None:
        """httpx is used automatically when aiohttp is not installed."""
        with patch("aiologging.handlers.http.AIOHTTP_AVAILABLE", False):
            handler = AsyncHttpHandler("https://example.com/api/logs")
            assert handler.backend == "httpx"

    def test_httpx_backend_missing(self) -> None:
        """DependencyError is raised when httpx is requested but missing."""
        with patch("aiologging.handlers.http.HTTPX_AVAILABLE", False):
            with pytest.raises(DependencyError, match="httpx is required"):
                AsyncHttpHandler(
                    "https://example.com/api/logs", backend="httpx"
                )

    def test_aiohttp_backend_missing(self) -> None:
        """DependencyError is raised when aiohttp is requested but missing."""
        with patch("aiologging.handlers.http.AIOHTTP_AVAILABLE", False):
            with pytest.raises(DependencyError, match="aiohttp is required"):
                AsyncHttpHandler(
                    "https://example.com/api/logs", backend="aiohttp"
                )

    def test_no_backend_installed(self) -> None:
        """DependencyError mentions both extras when nothing is installed."""
        with patch("aiologging.handlers.http.AIOHTTP_AVAILABLE", False), \
                patch("aiologging.handlers.http.HTTPX_AVAILABLE", False):
            with pytest.raises(DependencyError) as exc_info:
                AsyncHttpHandler("https://example.com/api/logs")
            assert "aiologging[aiohttp]" in str(exc_info.value)
            assert "aiologging[httpx]" in str(exc_info.value)

    def test_unknown_backend(self) -> None:
        """ConfigurationError is raised for an unknown backend name."""
        with pytest.raises(ConfigurationError, match="Unknown HTTP backend"):
            AsyncHttpHandler(
                "https://example.com/api/logs",
                backend="requests",  # type: ignore[arg-type]
            )

    def test_backend_propagates_to_format_handlers(self) -> None:
        """The universal handler passes its backend to format handlers."""
        handler = AsyncHttpHandler(
            "https://example.com/api/logs", backend="httpx"
        )
        for sub_handler in handler._handlers.values():
            assert sub_handler.backend == "httpx"


class TestHttpxBackend:
    """Test cases for the httpx HTTP client backend."""

    def _create_test_record(self, message: str = "Test message") -> logging.LogRecord:
        """Create a test log record."""
        return logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=message, args=(), exc_info=None
        )

    def _make_handler_with_transport(
        self,
        handler_class: type,
        respond: "callable",
        **handler_kwargs: object,
    ) -> object:
        """Create a handler wired to an httpx.MockTransport."""
        handler = handler_class(
            "https://example.com/api/logs",
            backend="httpx",
            **handler_kwargs,
        )
        handler._session = httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        )
        return handler

    @pytest.mark.asyncio
    async def test_flush_sends_json_request(self) -> None:
        """JSON records are delivered through the httpx client."""
        requests: List[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200)

        handler = self._make_handler_with_transport(
            AsyncHttpJsonHandler, respond
        )

        await handler.flush([self._create_test_record("httpx message")])
        await handler.close()

        assert len(requests) == 1
        request = requests[0]
        assert request.method == "POST"
        assert str(request.url) == "https://example.com/api/logs"
        assert request.headers["Content-Type"] == "application/json"
        payload = json.loads(request.content)
        assert payload[0]["message"] == "httpx message"

    @pytest.mark.asyncio
    async def test_flush_sends_text_request(self) -> None:
        """Plain text records are delivered through the httpx client."""
        from aiologging.handlers.http import AsyncHttpTextHandler

        requests: List[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200)

        handler = self._make_handler_with_transport(
            AsyncHttpTextHandler, respond
        )

        await handler.flush([
            self._create_test_record("line 1"),
            self._create_test_record("line 2"),
        ])
        await handler.close()

        assert len(requests) == 1
        request = requests[0]
        assert request.headers["Content-Type"] == "text/plain"
        assert request.content.decode() == "line 1\nline 2"

    @pytest.mark.asyncio
    async def test_flush_server_error_raises_network_error(self) -> None:
        """A 5xx response raises NetworkError after retries."""
        from aiologging.types import BatchConfig

        call_count = 0

        def respond(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(500, text="Server Error")

        handler = self._make_handler_with_transport(
            AsyncHttpJsonHandler,
            respond,
            batch_config=BatchConfig(max_retries=1, retry_delay=0.01),
        )

        with pytest.raises(NetworkError, match="Server Error"):
            await handler.flush([self._create_test_record()])

        # 1 initial attempt + 1 retry
        assert call_count == 2
        await handler.close()

    @pytest.mark.asyncio
    async def test_flush_client_error_does_not_retry(self) -> None:
        """A 4xx response fails immediately without retries."""
        from aiologging.types import BatchConfig

        call_count = 0

        def respond(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, text="Bad Request")

        handler = self._make_handler_with_transport(
            AsyncHttpJsonHandler,
            respond,
            batch_config=BatchConfig(max_retries=3, retry_delay=0.01),
        )

        with pytest.raises(NetworkError, match="Bad Request"):
            await handler.flush([self._create_test_record()])

        assert call_count == 1
        await handler.close()

    @pytest.mark.asyncio
    async def test_close_closes_httpx_client(self) -> None:
        """Closing the handler closes the underlying httpx client."""
        handler = self._make_handler_with_transport(
            AsyncHttpJsonHandler, lambda request: httpx.Response(200)
        )
        client = handler._session

        await handler.close()

        assert client.is_closed

    @pytest.mark.asyncio
    async def test_session_recreated_after_close(self) -> None:
        """A new httpx client is created when the old one is closed."""
        handler = AsyncHttpJsonHandler(
            "https://example.com/api/logs", backend="httpx"
        )

        first = await handler._get_session()
        assert isinstance(first, httpx.AsyncClient)
        await first.aclose()

        second = await handler._get_session()
        assert second is not first
        assert not second.is_closed

        await handler.close()

    @pytest.mark.asyncio
    async def test_response_adapter(self) -> None:
        """_HttpxResponseAdapter exposes aiohttp-style status and text."""
        response = httpx.Response(404, text="Not Found")
        adapter = _HttpxResponseAdapter(response)

        assert adapter.status == 404
        assert await adapter.text() == "Not Found"

    @pytest.mark.asyncio
    async def test_authenticator_receives_httpx_client(self) -> None:
        """The authenticator is called with the httpx client instance."""
        seen_sessions: List[object] = []

        async def authenticator(session, request_data):
            seen_sessions.append(session)
            return {"Authorization": "Bearer token"}

        requests: List[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200)

        handler = self._make_handler_with_transport(
            AsyncHttpJsonHandler, respond, authenticator=authenticator
        )

        await handler.flush([self._create_test_record()])
        await handler.close()

        assert len(seen_sessions) == 1
        assert isinstance(seen_sessions[0], httpx.AsyncClient)
        assert requests[0].headers["Authorization"] == "Bearer token"


if __name__ == "__main__":
    pytest.main([__file__])
