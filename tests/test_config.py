"""
Tests for the configuration module.
"""

import json
import logging
import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

import aiologging
from aiologging.config import (
    ConfigManager,
    get_config_manager,
    configure_from_file,
    configure_from_dict,
    configure_from_env,
    get_configured_logger,
)
from typing import Any, List, Optional

from aiologging.exceptions import ConfigurationError
from aiologging.formatters import TelegramHtmlFormatter
from aiologging.handlers.base import AsyncHandler, BufferedAsyncHandler
from aiologging.handlers.stream import AsyncStreamHandler
from aiologging.handlers.file import AsyncFileHandler
from aiologging.logger import _logger_manager


class DottedPathHandler(AsyncStreamHandler):
    """Handler referenced by dotted path in configuration tests."""

    def __init__(self, tag: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tag = tag


class NotAFormatter:
    """Constructible like a formatter, but without its methods."""

    def __init__(
        self, fmt: "Optional[str]" = None, datefmt: "Optional[str]" = None
    ) -> None:
        self.fmt = fmt
        self.datefmt = datefmt


class CollectingHandler(AsyncHandler):
    """Handler that records everything it receives."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.records: List[logging.LogRecord] = []

    async def _emit(
        self, record: logging.LogRecord, formatted_message: str
    ) -> None:
        self.records.append(record)


class BufferedCollectorHandler(BufferedAsyncHandler):
    """Buffered handler recording what force_flush delivers."""

    last_instance: "Optional[BufferedCollectorHandler]" = None

    def __init__(self, **kwargs: Any) -> None:
        # Large buffer, no auto-flush: records stay buffered until
        # someone (e.g. shutdown) forces a flush
        super().__init__(buffer_size=1000, auto_flush=False, **kwargs)
        self.flushed: List[logging.LogRecord] = []
        BufferedCollectorHandler.last_instance = self

    async def emit(self, record: logging.LogRecord) -> None:
        await self.handle(record)

    async def _emit(
        self, record: logging.LogRecord, formatted_message: str
    ) -> None:
        pass

    async def flush(
        self, records: "Optional[List[logging.LogRecord]]" = None
    ) -> None:
        if records is None:
            await self.force_flush()
            return
        self.flushed.extend(records)


class TestConfigManager:
    """Test cases for ConfigManager."""

    def test_initialization(self) -> None:
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        assert "stream" in manager._handler_registry
        assert "file" in manager._handler_registry
        assert "http" in manager._handler_registry
        assert "async" in manager._logger_registry

    def test_register_handler(self) -> None:
        """Test registering a custom handler."""
        manager = ConfigManager()

        class CustomHandler:
            pass

        manager.register_handler("custom", CustomHandler)
        assert "custom" in manager._handler_registry
        assert manager._handler_registry["custom"] == CustomHandler

    def test_register_logger(self) -> None:
        """Test registering a custom logger."""
        manager = ConfigManager()

        class CustomLogger:
            pass

        manager.register_logger("custom", CustomLogger)
        assert "custom" in manager._logger_registry
        assert manager._logger_registry["custom"] == CustomLogger

    def test_load_from_dict_valid(self) -> None:
        """Test loading configuration from a valid dictionary."""
        manager = ConfigManager()

        config = {
            "version": 1,
            "loggers": {
                "test": {
                    "level": "INFO",
                    "handlers": ["console"]
                }
            },
            "handlers": {
                "console": {
                    "class": "stream",
                    "stream": "stdout"
                }
            }
        }

        manager.load_from_dict(config)
        assert manager._config == config

    def test_load_from_dict_invalid_version(self) -> None:
        """Test loading configuration with invalid version."""
        manager = ConfigManager()

        config = {
            "version": 2,
            "loggers": {},
            "handlers": {}
        }

        with pytest.raises(ConfigurationError):
            manager.load_from_dict(config)

    def test_load_from_dict_missing_version(self) -> None:
        """Test loading configuration without version."""
        manager = ConfigManager()

        config = {
            "loggers": {},
            "handlers": {}
        }

        with pytest.raises(ConfigurationError):
            manager.load_from_dict(config)

    def test_load_from_dict_missing_loggers(self) -> None:
        """Test loading configuration without loggers."""
        manager = ConfigManager()

        config = {
            "version": 1,
            "handlers": {}
        }

        with pytest.raises(ConfigurationError):
            manager.load_from_dict(config)

    def test_load_from_dict_missing_handlers(self) -> None:
        """Test loading configuration without handlers."""
        manager = ConfigManager()

        config = {
            "version": 1,
            "loggers": {}
        }

        with pytest.raises(ConfigurationError):
            manager.load_from_dict(config)

    def test_load_from_file_valid(self) -> None:
        """Test loading configuration from a valid file."""
        manager = ConfigManager()

        config = {
            "version": 1,
            "loggers": {
                "test": {
                    "level": "INFO",
                    "handlers": ["console"]
                }
            },
            "handlers": {
                "console": {
                    "class": "stream",
                    "stream": "stdout"
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_file = f.name

        try:
            manager.load_from_file(temp_file)
            assert manager._config == config
        finally:
            os.unlink(temp_file)

    def test_load_from_file_not_found(self) -> None:
        """Test loading configuration from a non-existent file."""
        manager = ConfigManager()

        with pytest.raises(ConfigurationError):
            manager.load_from_file("/non/existent/file.json")

    def test_load_from_file_invalid_json(self) -> None:
        """Test loading configuration from a file with invalid JSON."""
        manager = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_file = f.name

        try:
            with pytest.raises(ConfigurationError):
                manager.load_from_file(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_from_file_unsupported_format(self) -> None:
        """Test loading configuration from a file with unsupported format."""
        manager = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            temp_file = f.name

        try:
            with pytest.raises(ConfigurationError):
                manager.load_from_file(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_from_env(self) -> None:
        """Test loading configuration from environment variables."""
        manager = ConfigManager()

        # Set environment variables
        os.environ["AIOLogging_loggers_test_level"] = "INFO"
        os.environ["AIOLogging_loggers_test_handlers"] = '["console"]'
        os.environ["AIOLogging_handlers_console_class"] = "stream"
        os.environ["AIOLogging_handlers_console_stream"] = "stdout"

        try:
            manager.load_from_env()

            assert manager._config["version"] == 1
            assert manager._config["loggers"]["test"]["level"] == "INFO"
            assert manager._config["handlers"]["console"]["class"] == "stream"
            assert manager._config["handlers"]["console"]["stream"] == "stdout"
        finally:
            # Clean up environment variables
            del os.environ["AIOLogging_loggers_test_level"]
            del os.environ["AIOLogging_loggers_test_handlers"]
            del os.environ["AIOLogging_handlers_console_class"]
            del os.environ["AIOLogging_handlers_console_stream"]

    def test_get_logger(self) -> None:
        """Test getting a logger from configuration."""
        manager = ConfigManager()

        config = {
            "version": 1,
            "loggers": {
                "test": {
                    "level": "INFO",
                    "handlers": ["console"]
                }
            },
            "handlers": {
                "console": {
                    "class": "stream",
                    "stream": "stdout"
                }
            }
        }

        manager.load_from_dict(config)
        logger = manager.get_logger("test")

        assert logger.name == "test"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], AsyncStreamHandler)

    def test_get_logger_not_configured(self) -> None:
        """Test getting a logger that is not configured."""
        manager = ConfigManager()

        config = {
            "version": 1,
            "loggers": {},
            "handlers": {}
        }

        manager.load_from_dict(config)

        with pytest.raises(ConfigurationError):
            manager.get_logger("nonexistent")

    def test_get_logger_no_config_loaded(self) -> None:
        """Test getting a logger without loading configuration."""
        manager = ConfigManager()

        with pytest.raises(ConfigurationError):
            manager.get_logger("test")

    def test_get_logger_cached(self) -> None:
        """Test that getting the same logger returns the cached instance."""
        manager = ConfigManager()

        config = {
            "version": 1,
            "loggers": {
                "test": {
                    "level": "INFO",
                    "handlers": ["console"]
                }
            },
            "handlers": {
                "console": {
                    "class": "stream",
                    "stream": "stdout"
                }
            }
        }

        manager.load_from_dict(config)
        logger1 = manager.get_logger("test")
        logger2 = manager.get_logger("test")

        assert logger1 is logger2

    def test_get_logger_concurrent_construction(self) -> None:
        """Racing threads must converge on a single logger instance."""
        import threading

        manager = ConfigManager()
        manager.load_from_dict(
            {
                "version": 1,
                "loggers": {
                    "test": {"level": "INFO", "handlers": ["console"]}
                },
                "handlers": {
                    "console": {"class": "stream", "stream": "stdout"}
                },
            }
        )

        racers = 8
        barrier = threading.Barrier(racers)
        seen = []

        def race() -> None:
            barrier.wait()
            seen.append(manager.get_logger("test"))

        threads = [
            threading.Thread(target=race) for _ in range(racers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len({id(logger) for logger in seen}) == 1
        assert len(seen[0].handlers) == 1

    def test_parse_level_string(self) -> None:
        """Test parsing a log level from a string."""
        manager = ConfigManager()

        assert manager._parse_level("INFO") == logging.INFO
        assert manager._parse_level("DEBUG") == logging.DEBUG
        assert manager._parse_level("WARNING") == logging.WARNING
        assert manager._parse_level("ERROR") == logging.ERROR
        assert manager._parse_level("CRITICAL") == logging.CRITICAL

    def test_parse_level_int(self) -> None:
        """Test parsing a log level from an integer."""
        manager = ConfigManager()

        assert manager._parse_level(20) == 20
        assert manager._parse_level(10) == 10

    def test_parse_level_invalid(self) -> None:
        """Test parsing an invalid log level."""
        manager = ConfigManager()

        with pytest.raises(ConfigurationError):
            manager._parse_level("INVALID")

    def test_parse_env_value_json(self) -> None:
        """Test parsing an environment variable value as JSON."""
        manager = ConfigManager()

        assert manager._parse_env_value('{"key": "value"}') == {"key": "value"}
        assert manager._parse_env_value("[1, 2, 3]") == [1, 2, 3]
        assert manager._parse_env_value("true") is True
        assert manager._parse_env_value("false") is False
        assert manager._parse_env_value("null") is None

    def test_parse_env_value_string(self) -> None:
        """Test parsing an environment variable value as a string."""
        manager = ConfigManager()

        assert manager._parse_env_value("simple string") == "simple string"
        assert manager._parse_env_value("{ not valid json }") == "{ not valid json }"

    def test_create_stream_handler(self) -> None:
        """Test creating a stream handler."""
        manager = ConfigManager()

        config = {
            "class": "stream",
            "stream": "stdout"
        }

        handler = manager._create_stream_handler(config, logging.INFO)

        assert isinstance(handler, AsyncStreamHandler)
        assert handler.level == logging.INFO

    def test_create_file_handler(self) -> None:
        """Test creating a file handler."""
        manager = ConfigManager()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name

        try:
            config = {
                "class": "file",
                "filename": filename,
                "mode": "w",
                "encoding": "utf-8"
            }

            handler = manager._create_file_handler(config, logging.DEBUG)

            assert isinstance(handler, AsyncFileHandler)
            assert handler.level == logging.DEBUG
            assert handler.filename == Path(filename)
            assert handler.mode == "w"
            assert handler.encoding == "utf-8"
        finally:
            os.unlink(filename)

    def test_create_file_handler_no_filename(self) -> None:
        """Test creating a file handler without a filename."""
        manager = ConfigManager()

        config = {
            "class": "file"
        }

        with pytest.raises(ConfigurationError):
            manager._create_file_handler(config, logging.DEBUG)

    def test_create_handler_unknown_class(self) -> None:
        """Test creating a handler with an unknown class."""
        manager = ConfigManager()

        config = {
            "class": "unknown"
        }

        with pytest.raises(ConfigurationError):
            manager._create_handler("test")

    def test_create_handler_no_class(self) -> None:
        """Test creating a handler without specifying a class."""
        manager = ConfigManager()

        config = {}

        with pytest.raises(ConfigurationError):
            manager._create_handler("test")

    def test_create_handler_not_configured(self) -> None:
        """Test creating a handler that is not configured."""
        manager = ConfigManager()

        with pytest.raises(ConfigurationError):
            manager._create_handler("nonexistent")

    def test_create_http_handler(self) -> None:
        """Test creating an HTTP handler."""
        manager = ConfigManager()

        config = {
            "class": "http",
            "url": "https://example.com/log",
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "timeout": 60.0,
            "verify_ssl": False
        }

        with patch('aiologging.handlers.http.aiohttp') as mock_aiohttp:
            handler = manager._create_http_handler(config, logging.INFO)

            assert handler is not None
            assert handler.url == "https://example.com/log"
            assert handler.method == "POST"
            assert handler.timeout == 60.0
            assert handler.verify_ssl is False

    def test_create_http_handler_no_url(self) -> None:
        """Test creating an HTTP handler without a URL."""
        manager = ConfigManager()

        config = {
            "class": "http"
        }

        with pytest.raises(ConfigurationError, match="HTTP handler must specify a URL"):
            manager._create_http_handler(config, logging.INFO)

    def test_create_http_json_handler(self) -> None:
        """Test creating an HTTP JSON handler."""
        manager = ConfigManager()

        config = {
            "class": "http_json",
            "url": "https://example.com/log",
            "method": "PUT",
            "params": {"api_key": "secret"},
            "timeout": 45.0
        }

        with patch('aiologging.handlers.http.aiohttp') as mock_aiohttp:
            handler = manager._create_http_json_handler(config, logging.WARNING)

            assert handler is not None
            assert handler.url == "https://example.com/log"
            assert handler.method == "PUT"
            assert handler.timeout == 45.0

    def test_create_http_json_handler_no_url(self) -> None:
        """Test creating an HTTP JSON handler without a URL."""
        manager = ConfigManager()

        config = {
            "class": "http_json"
        }

        with pytest.raises(ConfigurationError, match="HTTP JSON handler must specify a URL"):
            manager._create_http_json_handler(config, logging.INFO)

    def test_create_rotating_file_handler(self) -> None:
        """Test creating a rotating file handler."""
        manager = ConfigManager()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name

        try:
            config = {
                "class": "rotating_file",
                "filename": filename,
                "mode": "a",
                "encoding": "utf-8",
                "max_bytes": 1024,
                "backup_count": 5
            }

            handler = manager._create_rotating_file_handler(config, logging.DEBUG)

            assert handler is not None
            assert handler.filename == Path(filename)
            assert handler.mode == "a"
            assert handler.encoding == "utf-8"
            assert handler.max_bytes == 1024
            assert handler.backup_count == 5
        finally:
            os.unlink(filename)

    def test_create_rotating_file_handler_no_filename(self) -> None:
        """Test creating a rotating file handler without a filename."""
        manager = ConfigManager()

        config = {
            "class": "rotating_file"
        }

        with pytest.raises(ConfigurationError, match="Rotating file handler must specify a filename"):
            manager._create_rotating_file_handler(config, logging.DEBUG)

    def test_create_timed_rotating_file_handler(self) -> None:
        """Test creating a timed rotating file handler."""
        manager = ConfigManager()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name

        try:
            config = {
                "class": "timed_rotating_file",
                "filename": filename,
                "encoding": "utf-8",
                "when": "D",
                "interval": 7,
                "backup_count": 4
            }

            handler = manager._create_timed_rotating_file_handler(config, logging.INFO)

            assert handler is not None
            assert handler.filename == Path(filename)
            assert handler.encoding == "utf-8"
            assert handler.when == "D"
            assert handler.interval == 7
            assert handler.backup_count == 4
        finally:
            os.unlink(filename)

    def test_create_timed_rotating_file_handler_no_filename(self) -> None:
        """Test creating a timed rotating file handler without a filename."""
        manager = ConfigManager()

        config = {
            "class": "timed_rotating_file"
        }

        with pytest.raises(ConfigurationError, match="Timed rotating file handler must specify a filename"):
            manager._create_timed_rotating_file_handler(config, logging.INFO)

    def test_parse_env_value_invalid_json(self) -> None:
        """Test parsing an environment variable with invalid JSON that should be returned as string."""
        manager = ConfigManager()

        # This is invalid JSON but should be returned as string
        assert manager._parse_env_value("{ invalid json }") == "{ invalid json }"

    def test_parse_env_value_complex_json(self) -> None:
        """Test parsing complex JSON values from environment variables."""
        manager = ConfigManager()

        # Test with nested objects
        nested_json = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = manager._parse_env_value(nested_json)
        # The implementation actually parses this JSON and returns the dict
        assert result == {"outer": {"inner": "value"}, "list": [1, 2, 3]}

        # Test with non-string keys in dict - this gets parsed as a dict with string keys
        invalid_dict_json = '{"1": "value", "2": "value2"}'
        result = manager._parse_env_value(invalid_dict_json)
        # The implementation parses this as a regular dict with string keys
        assert result == {"1": "value", "2": "value2"}

        # Test with mixed types in list - this gets parsed as a list
        mixed_list_json = '[1, "string", true, null]'
        result = manager._parse_env_value(mixed_list_json)
        # The implementation parses this as a list with mixed types
        assert result == [1, "string", True, None]


class TestCustomHandlerClasses:
    """Test cases for dotted-path and registered custom handlers."""

    @staticmethod
    def _config_for(handler: dict) -> dict:
        """Build a minimal config with a single handler."""
        return {
            "version": 1,
            "loggers": {
                "test": {
                    "level": "INFO",
                    "handlers": ["custom"]
                }
            },
            "handlers": {"custom": handler}
        }

    def test_dotted_path_handler(self) -> None:
        """Test creating a handler from a dotted path."""
        manager = ConfigManager()
        manager.load_from_dict(self._config_for({
            "class": "tests.test_config.DottedPathHandler",
            "level": "WARNING",
            "tag": "from-config"
        }))

        logger = manager.get_logger("test")

        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, DottedPathHandler)
        assert handler.level == logging.WARNING
        assert handler.tag == "from-config"

    def test_dotted_path_builtin_class(self) -> None:
        """Test that a dotted path to a bundled handler works."""
        manager = ConfigManager()
        manager.load_from_dict(self._config_for({
            "class": "aiologging.handlers.stream.AsyncStreamHandler",
            "level": "INFO"
        }))

        logger = manager.get_logger("test")

        assert isinstance(logger.handlers[0], AsyncStreamHandler)

    def test_registered_custom_handler(self) -> None:
        """Test that a handler registered by name is constructed."""
        manager = ConfigManager()
        manager.register_handler("dotted", DottedPathHandler)
        manager.load_from_dict(self._config_for({
            "class": "dotted",
            "level": "DEBUG",
            "tag": "registered"
        }))

        logger = manager.get_logger("test")

        handler = logger.handlers[0]
        assert isinstance(handler, DottedPathHandler)
        assert handler.tag == "registered"

    def test_dotted_path_module_not_found(self) -> None:
        """Test a dotted path pointing to a missing module."""
        manager = ConfigManager()
        manager.load_from_dict(self._config_for({
            "class": "no.such.module.Handler"
        }))

        with pytest.raises(
            ConfigurationError, match="Cannot import handler class"
        ):
            manager.get_logger("test")

    def test_dotted_path_missing_attribute(self) -> None:
        """Test a dotted path pointing to a missing class."""
        manager = ConfigManager()
        manager.load_from_dict(self._config_for({
            "class": "tests.test_config.NoSuchHandler"
        }))

        with pytest.raises(
            ConfigurationError, match="has no attribute"
        ):
            manager.get_logger("test")

    def test_dotted_path_not_a_handler(self) -> None:
        """Test a dotted path pointing to a non-handler class."""
        manager = ConfigManager()
        manager.load_from_dict(self._config_for({
            "class": "json.JSONDecoder"
        }))

        with pytest.raises(
            ConfigurationError, match="not an AsyncHandler subclass"
        ):
            manager.get_logger("test")

    def test_custom_handler_bad_kwargs(self) -> None:
        """Test that unexpected constructor kwargs raise an error."""
        manager = ConfigManager()
        manager.load_from_dict(self._config_for({
            "class": "tests.test_config.DottedPathHandler",
            "unexpected_option": 42
        }))

        with pytest.raises(
            ConfigurationError, match="Cannot create handler"
        ):
            manager.get_logger("test")


class TestGlobalHierarchyIntegration:
    """Config loggers must live in the global aiologging hierarchy."""

    @staticmethod
    def _config(
        name: str, handler: dict, **logger_extra: Any
    ) -> dict:
        return {
            "version": 1,
            "loggers": {
                name: {
                    "level": "INFO",
                    "handlers": ["h"],
                    **logger_extra,
                }
            },
            "handlers": {"h": handler},
        }

    def test_logger_registered_in_global_hierarchy(self) -> None:
        """The config logger has a parent and is in loggerDict."""
        manager = ConfigManager()
        manager.load_from_dict(self._config(
            "cfgapp.sub", {"class": "stream", "stream": "stderr"}
        ))

        logger = manager.get_logger("cfgapp.sub")

        assert "cfgapp.sub" in _logger_manager.loggerDict
        assert aiologging.getLogger("cfgapp.sub") is logger
        assert logger.parent is aiologging.getLogger("cfgapp")
        assert logger.parent.parent is _logger_manager.root

    def test_existing_logger_is_configured_not_replaced(self) -> None:
        """A logger obtained earlier via getLogger gets the config."""
        early = aiologging.getLogger("cfgearly")
        manager = ConfigManager()
        manager.load_from_dict(self._config(
            "cfgearly",
            {"class": "stream", "stream": "stderr"},
            propagate=False,
        ))

        configured = manager.get_logger("cfgearly")

        assert configured is early
        assert early.level == logging.INFO
        assert early.propagate is False
        assert len(early.handlers) == 1

    @pytest.mark.asyncio
    async def test_propagation_reaches_root_handlers(self) -> None:
        """Records from a config logger reach root handlers."""
        collector = CollectingHandler(level=logging.INFO)
        aiologging.getLogger().addHandler(collector)

        manager = ConfigManager()
        manager.load_from_dict(self._config(
            "cfgprop", {"class": "stream", "stream": "stderr"}
        ))
        logger = manager.get_logger("cfgprop")

        await logger.info("propagated to root")
        await aiologging.flush()

        assert [
            record.getMessage() for record in collector.records
        ] == ["propagated to root"]

    @pytest.mark.asyncio
    async def test_buffered_handler_flushed_on_shutdown(self) -> None:
        """shutdown() force-flushes a config logger's buffered tail."""
        manager = ConfigManager()
        manager.load_from_dict(self._config(
            "cfgbuf",
            {"class": "tests.test_config.BufferedCollectorHandler"},
            propagate=False,
        ))
        logger = manager.get_logger("cfgbuf")
        handler = BufferedCollectorHandler.last_instance
        assert handler is not None

        await logger.info("buffered tail")
        # The under-filled buffer holds the record until shutdown's
        # drain force-flushes every handler in the hierarchy
        await aiologging.shutdown()
        assert [
            record.getMessage() for record in handler.flushed
        ] == ["buffered tail"]

    @pytest.mark.asyncio
    async def test_stale_cache_rebuilt_after_shutdown(self) -> None:
        """After a global shutdown the manager rebuilds the logger."""
        manager = ConfigManager()
        manager.load_from_dict(self._config(
            "cfgcache", {"class": "stream", "stream": "stderr"}
        ))
        before = manager.get_logger("cfgcache")
        await aiologging.shutdown()

        after = manager.get_logger("cfgcache")

        assert after is not before
        assert not after._closed
        assert aiologging.getLogger("cfgcache") is after


class TestFormattersSection:
    """Test cases for the dictConfig-style formatters section."""

    @staticmethod
    def _config(
        formatters: dict, handler_extra: "Optional[dict]" = None
    ) -> dict:
        return {
            "version": 1,
            "formatters": formatters,
            "loggers": {
                "fmtapp": {"level": "INFO", "handlers": ["h"]}
            },
            "handlers": {
                "h": {
                    "class": "stream",
                    "stream": "stderr",
                    **(handler_extra or {}),
                }
            },
        }

    def test_formatter_applied_to_handler(self) -> None:
        """format and datefmt reach the handler's formatter."""
        manager = ConfigManager()
        manager.load_from_dict(self._config(
            {"plain": {
                "format": "%(levelname)s :: %(message)s",
                "datefmt": "%Y",
            }},
            {"formatter": "plain"},
        ))

        logger = manager.get_logger("fmtapp")
        formatter = logger.handlers[0].formatter

        assert isinstance(formatter, logging.Formatter)
        record = logging.LogRecord(
            "fmtapp", logging.INFO, "", 0, "hey", (), None
        )
        assert formatter.format(record) == "INFO :: hey"

    def test_formatter_instance_shared_between_handlers(self) -> None:
        """Handlers referencing the same name share one instance."""
        manager = ConfigManager()
        config = self._config(
            {"plain": {"format": "%(message)s"}},
            {"formatter": "plain"},
        )
        config["handlers"]["h2"] = {
            "class": "stream",
            "stream": "stdout",
            "formatter": "plain",
        }
        config["loggers"]["fmtapp"]["handlers"] = ["h", "h2"]
        manager.load_from_dict(config)

        logger = manager.get_logger("fmtapp")

        assert logger.handlers[0].formatter is logger.handlers[1].formatter

    def test_formatter_class_dotted_path(self) -> None:
        """class resolves a dotted path like TelegramHtmlFormatter."""
        manager = ConfigManager()
        manager.load_from_dict(self._config(
            {"tg": {
                "class": "aiologging.formatters.TelegramHtmlFormatter"
            }},
            {"formatter": "tg"},
        ))

        logger = manager.get_logger("fmtapp")

        assert isinstance(
            logger.handlers[0].formatter, TelegramHtmlFormatter
        )

    def test_formatter_class_not_importable(self) -> None:
        """A missing formatter module raises ConfigurationError."""
        manager = ConfigManager()
        manager.load_from_dict(self._config(
            {"bad": {"class": "no.such.module.Formatter"}},
            {"formatter": "bad"},
        ))
        with pytest.raises(
            ConfigurationError, match="Cannot import formatter class"
        ):
            manager.get_logger("fmtapp")

    def test_formatter_class_not_a_formatter(self) -> None:
        """A class without the formatter methods is rejected."""
        manager = ConfigManager()
        manager.load_from_dict(self._config(
            {"bad": {"class": "tests.test_config.NotAFormatter"}},
            {"formatter": "bad"},
        ))
        with pytest.raises(
            ConfigurationError, match="FormatterProtocol"
        ):
            manager.get_logger("fmtapp")

    def test_unknown_formatter_reference_rejected(self) -> None:
        """A handler referencing a missing formatter fails at load."""
        manager = ConfigManager()
        with pytest.raises(
            ConfigurationError, match="unknown formatter"
        ):
            manager.load_from_dict(self._config(
                {}, {"formatter": "nope"}
            ))

    def test_custom_handler_gets_formatter_not_kwarg(self) -> None:
        """For custom classes 'formatter' is a reference, not a kwarg."""
        manager = ConfigManager()
        manager.load_from_dict(self._config(
            {"plain": {"format": "%(message)s"}},
            {
                "class": "tests.test_config.DottedPathHandler",
                "formatter": "plain",
            },
        ))

        logger = manager.get_logger("fmtapp")
        handler = logger.handlers[0]

        assert isinstance(handler, DottedPathHandler)
        assert isinstance(handler.formatter, logging.Formatter)


class TestStreamTargetKey:
    """Test cases for the stream handler's 'stream' key."""

    @staticmethod
    def _handler_config(target: str) -> dict:
        return {
            "version": 1,
            "loggers": {
                "streamapp": {"level": "INFO", "handlers": ["h"]}
            },
            "handlers": {"h": {"class": "stream", "stream": target}},
        }

    @pytest.mark.parametrize(
        "target,expected",
        [
            ("stdout", "stdout"),
            ("ext://sys.stdout", "stdout"),
            ("stderr", "stderr"),
            ("ext://sys.stderr", "stderr"),
        ],
    )
    def test_stream_target_resolved(
        self, target: str, expected: str
    ) -> None:
        """stdout/stderr and their ext:// aliases are honoured."""
        manager = ConfigManager()
        manager.load_from_dict(self._handler_config(target))

        logger = manager.get_logger("streamapp")

        assert logger.handlers[0].stream is getattr(sys, expected)

    def test_stream_defaults_to_stderr(self) -> None:
        """Without a stream key the handler writes to stderr."""
        manager = ConfigManager()
        config = self._handler_config("stdout")
        del config["handlers"]["h"]["stream"]
        manager.load_from_dict(config)

        logger = manager.get_logger("streamapp")

        assert logger.handlers[0].stream is sys.stderr

    def test_unsupported_stream_rejected(self) -> None:
        """An unknown stream target raises ConfigurationError."""
        manager = ConfigManager()
        manager.load_from_dict(self._handler_config("ext://sys.foo"))
        with pytest.raises(
            ConfigurationError, match="Unsupported stream"
        ):
            manager.get_logger("streamapp")


class TestStrictConfigKeys:
    """Unknown configuration keys must fail loudly."""

    @staticmethod
    def _base_config() -> dict:
        return {
            "version": 1,
            "loggers": {
                "strictapp": {"level": "INFO", "handlers": ["h"]}
            },
            "handlers": {"h": {"class": "stream", "stream": "stderr"}},
        }

    def test_unknown_top_level_key_rejected(self) -> None:
        manager = ConfigManager()
        config = self._base_config()
        config["fromatters"] = {}
        with pytest.raises(
            ConfigurationError, match="Unknown key.*fromatters"
        ):
            manager.load_from_dict(config)

    def test_unknown_logger_key_rejected(self) -> None:
        manager = ConfigManager()
        config = self._base_config()
        config["loggers"]["strictapp"]["propogate"] = False
        with pytest.raises(
            ConfigurationError, match="Unknown key.*propogate"
        ):
            manager.load_from_dict(config)

    def test_unknown_builtin_handler_key_rejected(self) -> None:
        manager = ConfigManager()
        config = self._base_config()
        config["handlers"]["h"]["formater"] = "plain"
        with pytest.raises(
            ConfigurationError, match="Unknown key.*formater"
        ):
            manager.load_from_dict(config)

    def test_unknown_formatter_key_rejected(self) -> None:
        manager = ConfigManager()
        config = self._base_config()
        config["formatters"] = {"plain": {"fmt": "%(message)s"}}
        with pytest.raises(
            ConfigurationError, match="Unknown key.*fmt"
        ):
            manager.load_from_dict(config)

    def test_custom_handler_keys_stay_free_form(self) -> None:
        """Dotted-path handler kwargs are not key-checked at load."""
        manager = ConfigManager()
        config = self._base_config()
        config["handlers"]["h"] = {
            "class": "tests.test_config.DottedPathHandler",
            "tag": "anything-goes",
        }
        manager.load_from_dict(config)

        logger = manager.get_logger("strictapp")

        assert logger.handlers[0].tag == "anything-goes"


class TestGlobalFunctions:
    """Test cases for global configuration functions."""

    def test_get_config_manager(self) -> None:
        """Test getting the global configuration manager."""
        manager = get_config_manager()
        assert isinstance(manager, ConfigManager)

    def test_configure_from_file(self) -> None:
        """Test configuring from a file using the global function."""
        config = {
            "version": 1,
            "loggers": {
                "test": {
                    "level": "INFO",
                    "handlers": ["console"]
                }
            },
            "handlers": {
                "console": {
                    "class": "stream",
                    "stream": "stdout"
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_file = f.name

        try:
            configure_from_file(temp_file)

            manager = get_config_manager()
            assert manager._config == config
        finally:
            os.unlink(temp_file)

    def test_configure_from_dict(self) -> None:
        """Test configuring from a dictionary using the global function."""
        config = {
            "version": 1,
            "loggers": {
                "test": {
                    "level": "INFO",
                    "handlers": ["console"]
                }
            },
            "handlers": {
                "console": {
                    "class": "stream",
                    "stream": "stdout"
                }
            }
        }

        configure_from_dict(config)

        manager = get_config_manager()
        assert manager._config == config

    def test_configure_from_env(self) -> None:
        """Test configuring from environment variables using the global function."""
        # Set environment variables
        os.environ["AIOLogging_loggers_test_level"] = "INFO"
        os.environ["AIOLogging_loggers_test_handlers"] = '["console"]'
        os.environ["AIOLogging_handlers_console_class"] = "stream"
        os.environ["AIOLogging_handlers_console_stream"] = "stdout"

        try:
            configure_from_env()

            manager = get_config_manager()
            assert manager._config["loggers"]["test"]["level"] == "INFO"
            assert manager._config["handlers"]["console"]["class"] == "stream"
            assert manager._config["handlers"]["console"]["stream"] == "stdout"
        finally:
            # Clean up environment variables
            del os.environ["AIOLogging_loggers_test_level"]
            del os.environ["AIOLogging_loggers_test_handlers"]
            del os.environ["AIOLogging_handlers_console_class"]
            del os.environ["AIOLogging_handlers_console_stream"]

    def test_get_configured_logger(self) -> None:
        """Test getting a configured logger using the global function."""
        config = {
            "version": 1,
            "loggers": {
                "test": {
                    "level": "INFO",
                    "handlers": ["console"]
                }
            },
            "handlers": {
                "console": {
                    "class": "stream",
                    "stream": "stdout"
                }
            }
        }

        configure_from_dict(config)
        logger = get_configured_logger("test")

        assert logger.name == "test"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], AsyncStreamHandler)


if __name__ == "__main__":
    pytest.main([__file__])
