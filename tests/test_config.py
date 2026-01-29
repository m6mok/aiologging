"""
Tests for the configuration module.
"""

import json
import logging
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from aiologging.config import (
    ConfigManager,
    get_config_manager,
    configure_from_file,
    configure_from_dict,
    configure_from_env,
    get_configured_logger,
)
from aiologging.exceptions import ConfigurationError
from aiologging.handlers.stream import AsyncStreamHandler
from aiologging.handlers.file import AsyncFileHandler


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
