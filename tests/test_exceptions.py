"""
Tests for the aiologging exceptions module.
"""

import pytest

from aiologging.exceptions import (
    AiologgingError,
    HandlerError,
    LoggerError,
    ConfigurationError,
    DependencyError,
    AuthenticationError,
    NetworkError,
    FileError,
    RotationError,
    BatchError,
    FormatterError,
    ContextError,
    HandlerClosedError,
    LoggerClosedError,
    BufferFullError,
)


class TestAiologgingError:
    """Test cases for the base AiologgingError class."""

    def test_basic_initialization(self) -> None:
        """Test basic exception initialization."""
        error = AiologgingError("Test message")
        assert str(error) == "Test message"
        assert error.message == "Test message"
        assert error.details == {}

    def test_initialization_with_details(self) -> None:
        """Test exception initialization with details."""
        details = {"key1": "value1", "key2": "value2"}
        error = AiologgingError("Test message", details=details)
        assert str(error) == "Test message (key1=value1 | key2=value2)"
        assert error.message == "Test message"
        assert error.details == details

    def test_initialization_with_kwargs(self) -> None:
        """Test exception initialization with keyword arguments."""
        error = AiologgingError("Test message", key1="value1", key2="value2")
        assert str(error) == "Test message (key1=value1 | key2=value2)"
        assert error.message == "Test message"
        assert error.details == {"key1": "value1", "key2": "value2"}
        assert hasattr(error, "key1")
        assert error.key1 == "value1"
        assert hasattr(error, "key2")
        assert error.key2 == "value2"

    def test_initialization_with_details_and_kwargs(self) -> None:
        """Test exception initialization with both details and kwargs."""
        details = {"key1": "value1"}
        error = AiologgingError("Test message", details=details, key2="value2")
        assert str(error) == "Test message (key1=value1 | key2=value2)"
        assert error.details == {"key1": "value1", "key2": "value2"}

    def test_repr(self) -> None:
        """Test string representation."""
        details = {"key1": "value1"}
        error = AiologgingError("Test message", details=details)
        expected = "AiologgingError('Test message', details={'key1': 'value1'})"
        assert repr(error) == expected


class TestHandlerError:
    """Test cases for the HandlerError class."""

    def test_basic_initialization(self) -> None:
        """Test basic handler error initialization."""
        error = HandlerError("Handler failed")
        assert str(error) == "Handler failed"
        assert error.handler_name is None
        assert error.operation is None

    def test_initialization_with_attributes(self) -> None:
        """Test handler error initialization with attributes."""
        error = HandlerError(
            "Handler failed",
            handler_name="test_handler",
            operation="emit"
        )
        assert "Handler failed" in str(error)
        assert "test_handler" in str(error)
        assert "emit" in str(error)
        assert error.handler_name == "test_handler"
        assert error.operation == "emit"

    def test_inheritance(self) -> None:
        """Test that HandlerError inherits from AiologgingError."""
        error = HandlerError("Test")
        assert isinstance(error, AiologgingError)


class TestLoggerError:
    """Test cases for the LoggerError class."""

    def test_basic_initialization(self) -> None:
        """Test basic logger error initialization."""
        error = LoggerError("Logger failed")
        assert str(error) == "Logger failed"
        assert error.logger_name is None
        assert error.operation is None

    def test_initialization_with_attributes(self) -> None:
        """Test logger error initialization with attributes."""
        error = LoggerError(
            "Logger failed",
            logger_name="test_logger",
            operation="log"
        )
        assert "Logger failed" in str(error)
        assert "test_logger" in str(error)
        assert "log" in str(error)
        assert error.logger_name == "test_logger"
        assert error.operation == "log"

    def test_inheritance(self) -> None:
        """Test that LoggerError inherits from AiologgingError."""
        error = LoggerError("Test")
        assert isinstance(error, AiologgingError)


class TestConfigurationError:
    """Test cases for the ConfigurationError class."""

    def test_basic_initialization(self) -> None:
        """Test basic configuration error initialization."""
        error = ConfigurationError("Invalid config")
        assert str(error) == "Invalid config"
        assert error.config_key is None
        assert error.config_value is None

    def test_initialization_with_attributes(self) -> None:
        """Test configuration error initialization with attributes."""
        error = ConfigurationError(
            "Invalid config",
            config_key="level",
            config_value="INVALID"
        )
        assert "Invalid config" in str(error)
        assert "level" in str(error)
        assert "INVALID" in str(error)
        assert error.config_key == "level"
        assert error.config_value == "INVALID"

    def test_inheritance(self) -> None:
        """Test that ConfigurationError inherits from AiologgingError."""
        error = ConfigurationError("Test")
        assert isinstance(error, AiologgingError)


class TestDependencyError:
    """Test cases for the DependencyError class."""

    def test_basic_initialization(self) -> None:
        """Test basic dependency error initialization."""
        error = DependencyError("Missing dependency")
        assert str(error) == "Missing dependency"
        assert error.dependency_name is None
        assert error.install_command is None

    def test_initialization_with_attributes(self) -> None:
        """Test dependency error initialization with attributes."""
        error = DependencyError(
            "Missing dependency",
            dependency_name="aiohttp",
            install_command="pip install aiohttp"
        )
        assert "Missing dependency" in str(error)
        assert "aiohttp" in str(error)
        assert "pip install aiohttp" in str(error)
        assert error.dependency_name == "aiohttp"
        assert error.install_command == "pip install aiohttp"

    def test_inheritance(self) -> None:
        """Test that DependencyError inherits from AiologgingError."""
        error = DependencyError("Test")
        assert isinstance(error, AiologgingError)


class TestAuthenticationError:
    """Test cases for the AuthenticationError class."""

    def test_basic_initialization(self) -> None:
        """Test basic authentication error initialization."""
        error = AuthenticationError("Auth failed")
        assert str(error) == "Auth failed"
        assert error.auth_type is None
        assert error.status_code is None

    def test_initialization_with_attributes(self) -> None:
        """Test authentication error initialization with attributes."""
        error = AuthenticationError(
            "Auth failed",
            auth_type="basic",
            status_code=401
        )
        assert "Auth failed" in str(error)
        assert "basic" in str(error)
        assert "401" in str(error)
        assert error.auth_type == "basic"
        assert error.status_code == 401

    def test_inheritance(self) -> None:
        """Test that AuthenticationError inherits from AiologgingError."""
        error = AuthenticationError("Test")
        assert isinstance(error, AiologgingError)


class TestNetworkError:
    """Test cases for the NetworkError class."""

    def test_basic_initialization(self) -> None:
        """Test basic network error initialization."""
        error = NetworkError("Network failed")
        assert str(error) == "Network failed"
        assert error.url is None
        assert error.status_code is None
        assert error.timeout is None

    def test_initialization_with_attributes(self) -> None:
        """Test network error initialization with attributes."""
        error = NetworkError(
            "Network failed",
            url="https://example.com",
            status_code=500,
            timeout=30.0
        )
        assert "Network failed" in str(error)
        assert "https://example.com" in str(error)
        assert "500" in str(error)
        assert "30.0" in str(error)
        assert error.url == "https://example.com"
        assert error.status_code == 500
        assert error.timeout == 30.0

    def test_inheritance(self) -> None:
        """Test that NetworkError inherits from AiologgingError."""
        error = NetworkError("Test")
        assert isinstance(error, AiologgingError)


class TestFileError:
    """Test cases for the FileError class."""

    def test_basic_initialization(self) -> None:
        """Test basic file error initialization."""
        error = FileError("File operation failed")
        assert str(error) == "File operation failed"
        assert error.filename is None
        assert error.operation is None
        assert error.errno is None

    def test_initialization_with_attributes(self) -> None:
        """Test file error initialization with attributes."""
        error = FileError(
            "File operation failed",
            filename="/var/log/app.log",
            operation="write",
            errno=13
        )
        assert "File operation failed" in str(error)
        assert "/var/log/app.log" in str(error)
        assert "write" in str(error)
        assert "13" in str(error)
        assert error.filename == "/var/log/app.log"
        assert error.operation == "write"
        assert error.errno == 13

    def test_inheritance(self) -> None:
        """Test that FileError inherits from AiologgingError."""
        error = FileError("Test")
        assert isinstance(error, AiologgingError)


class TestRotationError:
    """Test cases for the RotationError class."""

    def test_basic_initialization(self) -> None:
        """Test basic rotation error initialization."""
        error = RotationError("Rotation failed")
        assert str(error) == "Rotation failed"
        assert error.filename is None
        assert error.rotation_type is None

    def test_initialization_with_attributes(self) -> None:
        """Test rotation error initialization with attributes."""
        error = RotationError(
            "Rotation failed",
            filename="/var/log/app.log",
            rotation_type="size"
        )
        assert "Rotation failed" in str(error)
        assert "/var/log/app.log" in str(error)
        assert "size" in str(error)
        assert error.filename == "/var/log/app.log"
        assert error.rotation_type == "size"

    def test_inheritance(self) -> None:
        """Test that RotationError inherits from FileError."""
        error = RotationError("Test")
        assert isinstance(error, FileError)
        assert isinstance(error, AiologgingError)


class TestBatchError:
    """Test cases for the BatchError class."""

    def test_basic_initialization(self) -> None:
        """Test basic batch error initialization."""
        error = BatchError("Batch failed")
        assert str(error) == "Batch failed"
        assert error.batch_size is None
        assert error.processed_count is None

    def test_initialization_with_attributes(self) -> None:
        """Test batch error initialization with attributes."""
        error = BatchError(
            "Batch failed",
            batch_size=100,
            processed_count=50
        )
        assert "Batch failed" in str(error)
        assert "100" in str(error)
        assert "50" in str(error)
        assert error.batch_size == 100
        assert error.processed_count == 50

    def test_inheritance(self) -> None:
        """Test that BatchError inherits from AiologgingError."""
        error = BatchError("Test")
        assert isinstance(error, AiologgingError)


class TestFormatterError:
    """Test cases for the FormatterError class."""

    def test_basic_initialization(self) -> None:
        """Test basic formatter error initialization."""
        error = FormatterError("Formatting failed")
        assert str(error) == "Formatting failed"
        assert error.formatter_name is None
        assert error.record_summary is None
        assert error.record_details == {}

    def test_initialization_with_attributes(self) -> None:
        """Test formatter error initialization with attributes."""
        record_details = {"level": "INFO", "message": "Test", "extra": "data"}
        error = FormatterError(
            "Formatting failed",
            formatter_name="json_formatter",
            record_details=record_details
        )
        assert "Formatting failed" in str(error)
        assert "json_formatter" in str(error)
        assert "level=INFO" in str(error)
        assert error.formatter_name == "json_formatter"
        assert error.record_details == record_details
        assert error.record_summary is not None
        assert "level=INFO" in error.record_summary

    def test_record_summary_truncation(self) -> None:
        """Test that record summary is truncated for long details."""
        record_details = {f"key{i}": f"value{i}" for i in range(10)}
        error = FormatterError(
            "Formatting failed",
            record_details=record_details
        )
        assert error.record_summary is not None
        assert "..." in error.record_summary  # Should be truncated

    def test_inheritance(self) -> None:
        """Test that FormatterError inherits from AiologgingError."""
        error = FormatterError("Test")
        assert isinstance(error, AiologgingError)


class TestContextError:
    """Test cases for the ContextError class."""

    def test_basic_initialization(self) -> None:
        """Test basic context error initialization."""
        error = ContextError("Context failed")
        assert str(error) == "Context failed"
        assert error.context_type is None
        assert error.operation is None

    def test_initialization_with_attributes(self) -> None:
        """Test context error initialization with attributes."""
        error = ContextError(
            "Context failed",
            context_type="file_handler",
            operation="enter"
        )
        assert "Context failed" in str(error)
        assert "file_handler" in str(error)
        assert "enter" in str(error)
        assert error.context_type == "file_handler"
        assert error.operation == "enter"

    def test_inheritance(self) -> None:
        """Test that ContextError inherits from AiologgingError."""
        error = ContextError("Test")
        assert isinstance(error, AiologgingError)


class TestHandlerClosedError:
    """Test cases for the HandlerClosedError class."""

    def test_basic_initialization(self) -> None:
        """Test basic handler closed error initialization."""
        error = HandlerClosedError("Handler is closed")
        assert str(error) == "Handler is closed"
        assert error.handler_name is None
        assert error.operation is None

    def test_initialization_with_attributes(self) -> None:
        """Test handler closed error initialization with attributes."""
        error = HandlerClosedError(
            "Handler is closed",
            handler_name="test_handler",
            operation="emit"
        )
        assert "Handler is closed" in str(error)
        assert "test_handler" in str(error)
        assert "emit" in str(error)
        assert error.handler_name == "test_handler"
        assert error.operation == "emit"

    def test_inheritance(self) -> None:
        """Test that HandlerClosedError inherits from HandlerError."""
        error = HandlerClosedError("Test")
        assert isinstance(error, HandlerError)
        assert isinstance(error, AiologgingError)


class TestLoggerClosedError:
    """Test cases for the LoggerClosedError class."""

    def test_basic_initialization(self) -> None:
        """Test basic logger closed error initialization."""
        error = LoggerClosedError("Logger is closed")
        assert str(error) == "Logger is closed"
        assert error.logger_name is None
        assert error.operation is None

    def test_initialization_with_attributes(self) -> None:
        """Test logger closed error initialization with attributes."""
        error = LoggerClosedError(
            "Logger is closed",
            logger_name="test_logger",
            operation="info"
        )
        assert "Logger is closed" in str(error)
        assert "test_logger" in str(error)
        assert "info" in str(error)
        assert error.logger_name == "test_logger"
        assert error.operation == "info"

    def test_inheritance(self) -> None:
        """Test that LoggerClosedError inherits from LoggerError."""
        error = LoggerClosedError("Test")
        assert isinstance(error, LoggerError)
        assert isinstance(error, AiologgingError)


class TestBufferFullError:
    """Test cases for the BufferFullError class."""

    def test_basic_initialization(self) -> None:
        """Test basic buffer full error initialization."""
        error = BufferFullError("Buffer is full")
        assert str(error) == "Buffer is full"
        assert error.handler_name is None
        assert error.operation is None
        assert error.buffer_size is None

    def test_initialization_with_attributes(self) -> None:
        """Test buffer full error initialization with attributes."""
        error = BufferFullError(
            "Buffer is full",
            handler_name="memory_handler",
            operation="add",
            buffer_size=1000
        )
        assert "Buffer is full" in str(error)
        assert "memory_handler" in str(error)
        assert "1000" in str(error)
        assert error.handler_name == "memory_handler"
        assert error.operation == "add"
        assert error.buffer_size == 1000

    def test_inheritance(self) -> None:
        """Test that BufferFullError inherits from HandlerError."""
        error = BufferFullError("Test")
        assert isinstance(error, HandlerError)
        assert isinstance(error, AiologgingError)


if __name__ == "__main__":
    pytest.main([__file__])
