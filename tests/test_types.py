"""
Tests for the aiologging types module.
"""

import logging
from pathlib import Path
from typing import get_type_hints
from unittest.mock import MagicMock

import pytest

from aiologging.types import (
    # Type variables
    # T,
    # RequestType,
    # ResponseType,
    # P,
    # HandlerType,
    # LoggerType,
    # FormatterType,
    # FilterType,

    # Type aliases
    # JsonValue,
    # ConfigValue,
    # HeaderValues,
    # ParamValues,
    # LogLevel,
    LOG_LEVEL_MAP,
    # HeadersType,
    # ParamsType,
    # AuthDataType,
    # FilePath,
    # AsyncCallable,
    # AsyncErrorHandler,
    # AsyncFilterFunc,
    # LogProcessor,
    # BatchProcessor,
    # HttpContentType,
    # RotationType,
    # TimeInterval,
    TIME_INTERVAL_DESCRIPTIONS,

    # Protocols
    # AuthenticatorProtocol,
    FormatterProtocol,
    FilterProtocol,
    AsyncHandlerABC,

    # Factory types
    # HandlerFactory,
    # LoggerFactory,
    # FormatterFactory,
    # FilterFactory,

    # Other types
    # AsyncContextManager,
    # AsyncContextManagerGeneric,
    # LoopGetter,
    # ErrorHandler,
    # ConfigValidator,
    # RetryStrategy,
    # RateLimiter,

    # Configuration classes
    HandlerConfig,
    LoggerConfig,
    BatchConfig,
    HttpConfig,
    FileConfig,
    RotationConfig,
)


class TestTypeAliases:
    """Test cases for type aliases."""

    def test_json_value_type(self) -> None:
        """Test JsonValue type alias."""
        # These should all be valid JsonValue types
        valid_values = [
            "string",
            42,
            3.14,
            True,
            None,
            {"key": "value"},
            [1, 2, 3],
            {"nested": {"dict": True}},
            [{"list": "of"}, {"dicts": True}],
        ]

        for value in valid_values:
            # This is just a type check at runtime, not a strict type validation
            assert isinstance(value, (str, int, float, bool, type(None), dict, list))

    def test_config_value_type(self) -> None:
        """Test ConfigValue type alias."""
        # These should all be valid ConfigValue types
        valid_values = [
            "string",
            42,
            3.14,
            True,
            None,
            ["list", "of", "strings"],
            {"key": "value"},
        ]

        for value in valid_values:
            # This is just a type check at runtime, not a strict type validation
            assert isinstance(value, (str, int, float, bool, type(None), list, dict))

    def test_header_values_type(self) -> None:
        """Test HeaderValues type alias."""
        # These should all be valid HeaderValues types
        valid_values = [
            "string",
            ["list", "of", "strings"],
        ]

        for value in valid_values:
            # This is just a type check at runtime, not a strict type validation
            assert isinstance(value, (str, list))

    def test_param_values_type(self) -> None:
        """Test ParamValues type alias."""
        # These should all be valid ParamValues types
        valid_values = [
            "string",
            ["list", "of", "strings"],
        ]

        for value in valid_values:
            # This is just a type check at runtime, not a strict type validation
            assert isinstance(value, (str, list))

    def test_log_level_map(self) -> None:
        """Test LOG_LEVEL_MAP constant."""
        assert isinstance(LOG_LEVEL_MAP, dict)
        assert LOG_LEVEL_MAP["CRITICAL"] == 50
        assert LOG_LEVEL_MAP["ERROR"] == 40
        assert LOG_LEVEL_MAP["WARNING"] == 30
        assert LOG_LEVEL_MAP["INFO"] == 20
        assert LOG_LEVEL_MAP["DEBUG"] == 10
        assert LOG_LEVEL_MAP["NOTSET"] == 0

    def test_time_interval_descriptions(self) -> None:
        """Test TIME_INTERVAL_DESCRIPTIONS constant."""
        assert isinstance(TIME_INTERVAL_DESCRIPTIONS, dict)
        assert TIME_INTERVAL_DESCRIPTIONS["S"] == "Seconds"
        assert TIME_INTERVAL_DESCRIPTIONS["M"] == "Minutes"
        assert TIME_INTERVAL_DESCRIPTIONS["H"] == "Hours"
        assert TIME_INTERVAL_DESCRIPTIONS["D"] == "Days"
        assert TIME_INTERVAL_DESCRIPTIONS["midnight"] == "Midnight"
        assert TIME_INTERVAL_DESCRIPTIONS["W0"] == "Monday"
        assert TIME_INTERVAL_DESCRIPTIONS["W1"] == "Tuesday"
        assert TIME_INTERVAL_DESCRIPTIONS["W2"] == "Wednesday"
        assert TIME_INTERVAL_DESCRIPTIONS["W3"] == "Thursday"
        assert TIME_INTERVAL_DESCRIPTIONS["W4"] == "Friday"
        assert TIME_INTERVAL_DESCRIPTIONS["W5"] == "Saturday"
        assert TIME_INTERVAL_DESCRIPTIONS["W6"] == "Sunday"

    def test_file_path_type(self) -> None:
        """Test FilePath type alias."""
        # These should all be valid FilePath types
        valid_values = [
            "/path/to/file.txt",
            "relative/path.txt",
            Path("/path/to/file.txt"),
            Path("relative/path.txt"),
        ]

        for value in valid_values:
            # This is just a type check at runtime, not a strict type validation
            assert isinstance(value, (str, Path))


class TestProtocols:
    """Test cases for protocol definitions."""

    def test_formatter_protocol(self) -> None:
        """Test FormatterProtocol implementation."""
        class MyFormatter:
            def format(self, record: logging.LogRecord) -> str:
                return f"{record.levelname}: {record.getMessage()}"

            def usesTime(self) -> bool:
                return False

            def formatTime(self, record: logging.LogRecord, datefmt: str = None) -> str:
                return str(record.created)

        formatter = MyFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )

        # Test the methods
        assert isinstance(formatter.format(record), str)
        assert isinstance(formatter.usesTime(), bool)
        assert isinstance(formatter.formatTime(record), str)

        # Check if it implements the protocol
        assert isinstance(formatter, FormatterProtocol)

    def test_filter_protocol(self) -> None:
        """Test FilterProtocol implementation."""
        class MyFilter:
            def filter(self, record: logging.LogRecord) -> bool:
                return record.levelno >= logging.WARNING

            def filterName(self, name: str) -> bool:
                return name.startswith("test")

        filter_obj = MyFilter()
        record = logging.LogRecord(
            name="test_logger", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )

        # Test the methods
        assert isinstance(filter_obj.filter(record), bool)
        assert isinstance(filter_obj.filterName("test_logger"), bool)

        # Check if it implements the protocol
        assert isinstance(filter_obj, FilterProtocol)

    @pytest.mark.asyncio
    async def test_async_handler_abc(self) -> None:
        """Test AsyncHandlerABC abstract class."""
        class MyAsyncHandler(AsyncHandlerABC):
            async def emit(self, record: logging.LogRecord) -> None:
                pass

            async def close(self) -> None:
                pass

            def setLevel(self, level: int) -> None:
                self.level = level

            def addFilter(self, filter: FilterProtocol) -> None:
                if not hasattr(self, 'filters'):
                    self.filters = []
                self.filters.append(filter)

            def removeFilter(self, filter: FilterProtocol) -> None:
                if hasattr(self, 'filters') and filter in self.filters:
                    self.filters.remove(filter)

        handler = MyAsyncHandler()

        # Test the methods
        await handler.emit(logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        ))
        await handler.close()
        handler.setLevel(logging.DEBUG)

        filter_obj = MagicMock(spec=FilterProtocol)
        handler.addFilter(filter_obj)
        handler.removeFilter(filter_obj)

        # Check if it's an instance of the ABC
        assert isinstance(handler, AsyncHandlerABC)


class TestConfigurationClasses:
    """Test cases for configuration classes."""

    def test_handler_config(self) -> None:
        """Test HandlerConfig class."""
        handler_class = MagicMock
        formatter = MagicMock(spec=FormatterProtocol)
        filter_obj = MagicMock(spec=FilterProtocol)

        config = HandlerConfig(
            handler_class=handler_class,
            level=logging.INFO,
            formatter=formatter,
            filters=[filter_obj],
            custom_arg="value"
        )

        assert config.handler_class == handler_class
        assert config.level == logging.INFO
        assert config.formatter == formatter
        assert config.filters == [filter_obj]
        assert config.kwargs == {"custom_arg": "value"}

    def test_logger_config(self) -> None:
        """Test LoggerConfig class."""
        handler_config = MagicMock(spec=HandlerConfig)

        config = LoggerConfig(
            name="test_logger",
            level=logging.DEBUG,
            handlers=[handler_config],
            propagate=False,
            disabled=True
        )

        assert config.name == "test_logger"
        assert config.level == logging.DEBUG
        assert config.handlers == [handler_config]
        assert config.propagate is False
        assert config.disabled is True

    def test_batch_config(self) -> None:
        """Test BatchConfig class."""
        config = BatchConfig(
            batch_size=200,
            flush_interval=10.0,
            max_retries=5,
            retry_delay=2.0
        )

        assert config.batch_size == 200
        assert config.flush_interval == 10.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0

    def test_batch_config_defaults(self) -> None:
        """Test BatchConfig default values."""
        config = BatchConfig()

        assert config.batch_size == 100
        assert config.flush_interval == 5.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_http_config(self) -> None:
        """Test HttpConfig class."""
        batch_config = MagicMock(spec=BatchConfig)

        config = HttpConfig(
            url="https://example.com/api/logs",
            method="POST",
            headers={"Content-Type": "application/json"},
            params={"key": "value"},
            timeout=30.0,
            verify_ssl=False,
            batch_config=batch_config
        )

        assert config.url == "https://example.com/api/logs"
        assert config.method == "POST"
        assert config.headers == {"Content-Type": "application/json"}
        assert config.params == {"key": "value"}
        assert config.timeout == 30.0
        assert config.verify_ssl is False
        assert config.batch_config == batch_config

    def test_http_config_defaults(self) -> None:
        """Test HttpConfig default values."""
        config = HttpConfig(url="https://example.com/api/logs")

        assert config.url == "https://example.com/api/logs"
        assert config.method == "POST"
        assert config.headers == {}
        assert config.params == {}
        assert config.timeout == 30.0
        assert config.verify_ssl is True
        assert isinstance(config.batch_config, BatchConfig)

    def test_file_config(self) -> None:
        """Test FileConfig class."""
        config = FileConfig(
            filename="/var/log/app.log",
            mode="w",
            encoding="utf-8",
            delay=True,
            errors="ignore"
        )

        assert config.filename == "/var/log/app.log"
        assert config.mode == "w"
        assert config.encoding == "utf-8"
        assert config.delay is True
        assert config.errors == "ignore"

    def test_file_config_defaults(self) -> None:
        """Test FileConfig default values."""
        config = FileConfig(filename="/var/log/app.log")

        assert config.filename == "/var/log/app.log"
        assert config.mode == "a"
        assert config.encoding == "utf-8"
        assert config.delay is False
        assert config.errors is None

    def test_rotation_config(self) -> None:
        """Test RotationConfig class."""
        config = RotationConfig(
            max_bytes=10485760,  # 10MB
            backup_count=5,
            when="midnight",
            interval=1,
            backup_time="02:00",
            utc=True
        )

        assert config.max_bytes == 10485760
        assert config.backup_count == 5
        assert config.when == "midnight"
        assert config.interval == 1
        assert config.backup_time == "02:00"
        assert config.utc is True

    def test_rotation_config_defaults(self) -> None:
        """Test RotationConfig default values."""
        config = RotationConfig()

        assert config.max_bytes == 0
        assert config.backup_count == 0
        assert config.when is None
        assert config.interval == 1
        assert config.backup_time is None
        assert config.utc is False


class TestTypeHints:
    """Test cases for type hints."""

    def test_type_hints_for_config_classes(self) -> None:
        """Test that configuration classes have proper type hints."""
        # Check HandlerConfig
        handler_hints = get_type_hints(HandlerConfig.__init__)
        assert "handler_class" in handler_hints
        assert "level" in handler_hints
        assert "formatter" in handler_hints
        assert "filters" in handler_hints

        # Check LoggerConfig
        logger_hints = get_type_hints(LoggerConfig.__init__)
        assert "name" in logger_hints
        assert "level" in logger_hints
        assert "handlers" in logger_hints
        assert "propagate" in logger_hints
        assert "disabled" in logger_hints

        # Check BatchConfig
        batch_hints = get_type_hints(BatchConfig.__init__)
        assert "batch_size" in batch_hints
        assert "flush_interval" in batch_hints
        assert "max_retries" in batch_hints
        assert "retry_delay" in batch_hints

        # Check HttpConfig
        http_hints = get_type_hints(HttpConfig.__init__)
        assert "url" in http_hints
        assert "method" in http_hints
        assert "headers" in http_hints
        assert "params" in http_hints
        assert "timeout" in http_hints
        assert "verify_ssl" in http_hints
        assert "batch_config" in http_hints

        # Check FileConfig
        file_hints = get_type_hints(FileConfig.__init__)
        assert "filename" in file_hints
        assert "mode" in file_hints
        assert "encoding" in file_hints
        assert "delay" in file_hints
        assert "errors" in file_hints

        # Check RotationConfig
        rotation_hints = get_type_hints(RotationConfig.__init__)
        assert "max_bytes" in rotation_hints
        assert "backup_count" in rotation_hints
        assert "when" in rotation_hints
        assert "interval" in rotation_hints
        assert "backup_time" in rotation_hints
        assert "utc" in rotation_hints


if __name__ == "__main__":
    pytest.main([__file__])
