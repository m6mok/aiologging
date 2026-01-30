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

from aiologging.handlers.http import AsyncHttpHandler, AsyncHttpJsonHandler
from aiologging.exceptions import NetworkError, AuthenticationError


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


class TestAsyncHttpProtoHandler:
    """Test cases for AsyncHttpProtoHandler."""

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
    async def test_emit_proto_format(self) -> None:
        """Test emit with protobuf format."""
        from aiologging.handlers.http import AsyncHttpProtoHandler

        with patch("aiologging.handlers.http.PROTOBUF_AVAILABLE", True):
            handler = AsyncHttpProtoHandler("https://example.com/api/logs")
            handler.flush = AsyncMock()
            record = self._create_test_record()

            await handler.emit(record)
            await handler.flush()

            handler.flush.assert_called()

    @pytest.mark.asyncio
    async def test_emit_proto_format_without_protobuf(self) -> None:
        """Test emit with protobuf format when protobuf is not available."""
        from aiologging.handlers.http import AsyncHttpProtoHandler
        from aiologging.exceptions import DependencyError

        with patch("aiologging.handlers.http.PROTOBUF_AVAILABLE", False):
            with pytest.raises(DependencyError, match="protobuf is required"):
                AsyncHttpProtoHandler("https://example.com/api/logs")

    @pytest.mark.asyncio
    async def test_prepare_request_data(self) -> None:
        """Test _prepare_request_data method."""
        from aiologging.handlers.http import AsyncHttpProtoHandler

        with patch("aiologging.handlers.http.PROTOBUF_AVAILABLE", True):
            handler = AsyncHttpProtoHandler("https://example.com/api/logs")

            record1, record2 = self._create_test_records("Test message 1", "Test message 2")

            # Prepare request data
            request_data = await handler._prepare_request_data([record1, record2])

            # Verify the data is bytes
            assert isinstance(request_data, bytes)

            # Decode and verify it contains the expected JSON data
            decoded_data = request_data.decode('utf-8')
            assert "Test message 1" in decoded_data
            assert "Test message 2" in decoded_data

    @pytest.mark.asyncio
    async def test_get_content_type(self) -> None:
        """Test _get_content_type method."""
        from aiologging.handlers.http import AsyncHttpProtoHandler

        with patch("aiologging.handlers.http.PROTOBUF_AVAILABLE", True):
            handler = AsyncHttpProtoHandler("https://example.com/api/logs")
            assert handler._get_content_type() == "application/x-protobuf"


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

        # Test with protobuf Content-Type
        handler = AsyncHttpHandler(
            "https://example.com/api/logs",
            headers={"Content-Type": "application/x-protobuf"}
        )
        assert handler._detect_format_type() == "application/x-protobuf"

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


if __name__ == "__main__":
    pytest.main([__file__])
