"""
Configuration management for the aiologging library.

This module provides utilities for managing configuration of loggers and
handlers, including loading from files, environment variables,
and dictionaries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, Type, cast

from .exceptions import ConfigurationError
from .handlers.base import AsyncHandler
from .handlers.stream import AsyncStreamHandler
from .handlers.file import AsyncFileHandler
from .handlers.http import AsyncHttpHandler, AsyncHttpJsonHandler
from .handlers.rotating import (
    AsyncRotatingFileHandler,
    AsyncTimedRotatingFileHandler,
)
from .logger import AsyncLogger
from .types import (
    ConfigValue,
    LogLevel,
    LOG_LEVEL_MAP,
)


class ConfigManager:
    """
    Manager for loading and applying configuration to loggers and handlers.

    This class provides methods to load configuration from various sources
    and apply them to create and configure loggers and handlers.

    Attributes:
        _handler_registry: Registry of available handler classes
        _logger_registry: Registry of available logger classes

    Example:
        >>> config_manager = ConfigManager()
        >>> config_manager.load_from_file("config.json")
        >>> logger = config_manager.get_logger("myapp")
    """

    def __init__(self) -> None:
        """Initialize the configuration manager."""
        self._handler_registry: Dict[str, Type[AsyncHandler]] = {
            "stream": AsyncStreamHandler,
            "file": AsyncFileHandler,
            "http": AsyncHttpHandler,
            "http_json": AsyncHttpJsonHandler,
            "rotating_file": AsyncRotatingFileHandler,
            "timed_rotating_file": AsyncTimedRotatingFileHandler,
        }
        self._logger_registry: Dict[str, Type[AsyncLogger]] = {
            "async": AsyncLogger,
        }
        self._config: Dict[str, Any] = {}
        self._loggers: Dict[str, AsyncLogger] = {}

    def register_handler(
        self, name: str, handler_class: Type[AsyncHandler]
    ) -> None:
        """
        Register a custom handler class.

        Args:
            name: Name to register the handler under
            handler_class: Handler class to register

        Example:
            >>> config_manager.register_handler("custom", MyCustomHandler)
        """
        self._handler_registry[name] = handler_class

    def register_logger(
        self, name: str, logger_class: Type[AsyncLogger]
    ) -> None:
        """
        Register a custom logger class.

        Args:
            name: Name to register the logger under
            logger_class: Logger class to register

        Example:
            >>> config_manager.register_logger("custom", MyCustomLogger)
        """
        self._logger_registry[name] = logger_class

    def load_from_dict(self, config: Dict[str, Any]) -> None:
        """
        Load configuration from a dictionary.

        Args:
            config: Configuration dictionary

        Example:
            >>> config = {
            ...     "version": 1,
            ...     "loggers": {
            ...         "myapp": {
            ...             "level": "INFO",
            ...             "handlers": ["console", "file"]
            ...         }
            ...     },
            ...     "handlers": {
            ...         "console": {
            ...             "class": "stream",
            ...             "level": "INFO",
            ...             "stream": "stdout"
            ...         },
            ...         "file": {
            ...             "class": "file",
            ...             "level": "DEBUG",
            ...             "filename": "app.log",
            ...             "mode": "a"
            ...         }
            ...     }
            ... }
            >>> config_manager.load_from_dict(config)
        """
        self._config = config
        self._validate_config()

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load configuration from a file.

        Args:
            file_path: Path to the configuration file

        Raises:
            ConfigurationError: If the file cannot be loaded or parsed

        Example:
            >>> config_manager.load_from_file("config.json")
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}"
            )

        try:
            with open(file_path, "r") as f:
                if file_path.suffix.lower() == ".json":
                    config = json.load(f)
                else:
                    raise ConfigurationError(
                        "Unsupported configuration "
                        f"file format: {file_path.suffix}"
                    )

            self.load_from_dict(config)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in configuration file: {e}"
            )
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file: {e}")

    def load_from_env(self, prefix: str = "AIOLogging") -> None:
        """
        Load configuration from environment variables.

        Args:
            prefix: Prefix for environment variables

        Example:
            >>> # Set environment variables:
            >>> # AIOLogging_loggers_myapp_level=INFO
            >>> # AIOLogging_handlers_console_class=stream
            >>> # AIOLogging_handlers_console_stream=stdout
            >>> config_manager.load_from_env()
        """
        config = {"version": 1, "loggers": {}, "handlers": {}}

        # Find all environment variables with the specified prefix
        for key, value in os.environ.items():
            if key.startswith(f"{prefix}_"):
                # Remove the prefix and convert to lowercase
                config_key = key[len(prefix) + 1:].lower()

                # Split the key into parts
                parts = config_key.split("_")

                # Navigate to the appropriate location in the config
                current: Dict[str, Any] = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the value
                current[parts[-1]] = self._parse_env_value(value)

        self.load_from_dict(config)

    def get_logger(self, name: str) -> AsyncLogger:
        """
        Get a logger with the specified name,
        configured according to the loaded config.

        Args:
            name: Name of the logger

        Returns:
            Configured logger instance

        Example:
            >>> logger = config_manager.get_logger("myapp")
        """
        if name in self._loggers:
            return self._loggers[name]

        if not self._config:
            raise ConfigurationError("No configuration loaded")

        logger_config = self._config.get("loggers", {}).get(name, {})
        if not logger_config:
            raise ConfigurationError(
                f"No configuration found for logger: {name}"
            )

        # Create the logger
        logger_type = logger_config.get("type", "async")
        logger_class = self._logger_registry.get(logger_type)
        if not logger_class:
            raise ConfigurationError(f"Unknown logger type: {logger_type}")

        # Parse the level
        level = self._parse_level(logger_config.get("level", "INFO"))

        # Create handlers
        handlers = []
        for handler_name in logger_config.get("handlers", []):
            handler = self._create_handler(handler_name)
            if handler:
                handlers.append(handler)

        # Create the logger
        logger = logger_class(
            name=name,
            level=level,
            handlers=handlers,
            propagate=logger_config.get("propagate", True),
            disabled=logger_config.get("disabled", False),
        )

        self._loggers[name] = logger
        return logger

    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        if not isinstance(self._config, dict):
            raise ConfigurationError("Configuration must be a dictionary")

        if "version" not in self._config:
            raise ConfigurationError("Configuration must specify a version")

        if self._config["version"] != 1:
            raise ConfigurationError(
                f"Unsupported configuration version: {self._config['version']}"
            )

        if "loggers" not in self._config:
            raise ConfigurationError("Configuration must specify loggers")

        if "handlers" not in self._config:
            raise ConfigurationError("Configuration must specify handlers")

    def _parse_level(self, level: Union[str, int]) -> int:
        """Parse a log level from a string or integer."""
        if isinstance(level, int):
            return level

        if isinstance(level, str):
            level_upper = cast(LogLevel, level.upper())
            if level_upper in LOG_LEVEL_MAP:
                return LOG_LEVEL_MAP[level_upper]

        raise ConfigurationError(f"Invalid log level: {level}")

    def _parse_env_value(self, value: str) -> ConfigValue:
        """Parse a value from an environment variable."""
        # Try to parse as JSON first
        try:
            parsed: ConfigValue = json.loads(value)
            return parsed
        except json.JSONDecodeError:
            # If not valid JSON, return as string
            return value

    def _create_handler(self, name: str) -> Optional[AsyncHandler]:
        """Create a handler from the configuration."""
        handler_config = self._config.get("handlers", {}).get(name)
        if not handler_config:
            raise ConfigurationError(
                f"No configuration found for handler: {name}"
            )

        handler_type = handler_config.get("class")
        if not handler_type:
            raise ConfigurationError(f"Handler {name} must specify a class")

        handler_class = self._handler_registry.get(handler_type)
        if not handler_class:
            raise ConfigurationError(f"Unknown handler class: {handler_type}")

        # Parse the level
        level = self._parse_level(handler_config.get("level", "INFO"))

        # Create the handler based on its type
        if handler_type == "stream":
            return self._create_stream_handler(handler_config, level)
        elif handler_type == "file":
            return self._create_file_handler(handler_config, level)
        elif handler_type == "http":
            return self._create_http_handler(handler_config, level)
        elif handler_type == "http_json":
            return self._create_http_json_handler(handler_config, level)
        elif handler_type == "rotating_file":
            return self._create_rotating_file_handler(handler_config, level)
        elif handler_type == "timed_rotating_file":
            return self._create_timed_rotating_file_handler(
                handler_config, level
            )
        else:
            raise ConfigurationError(
                f"Unsupported handler type: {handler_type}"
            )

    def _create_stream_handler(
        self, config: Dict[str, Any], level: int
    ) -> AsyncStreamHandler:
        """Create a stream handler from configuration."""
        return AsyncStreamHandler(level=level)

    def _create_file_handler(
        self, config: Dict[str, Any], level: int
    ) -> AsyncFileHandler:
        """Create a file handler from configuration."""
        filename = config.get("filename")
        if not filename:
            raise ConfigurationError("File handler must specify a filename")

        return AsyncFileHandler(
            filename=filename,
            mode=config.get("mode", "a"),
            encoding=config.get("encoding", "utf-8"),
            delay=config.get("delay", False),
            errors=config.get("errors"),
            level=level,
        )

    def _create_http_handler(
        self, config: Dict[str, Any], level: int
    ) -> AsyncHttpHandler:
        """Create an HTTP handler from configuration."""
        url = config.get("url")
        if not url:
            raise ConfigurationError("HTTP handler must specify a URL")

        return AsyncHttpHandler(
            url=url,
            method=config.get("method", "POST"),
            headers=config.get("headers"),
            params=config.get("params"),
            timeout=config.get("timeout", 30.0),
            verify_ssl=config.get("verify_ssl", True),
            level=level,
        )

    def _create_http_json_handler(
        self, config: Dict[str, Any], level: int
    ) -> AsyncHttpJsonHandler:
        """Create an HTTP JSON handler from configuration."""
        url = config.get("url")
        if not url:
            raise ConfigurationError("HTTP JSON handler must specify a URL")

        return AsyncHttpJsonHandler(
            url=url,
            method=config.get("method", "POST"),
            headers=config.get("headers"),
            params=config.get("params"),
            timeout=config.get("timeout", 30.0),
            verify_ssl=config.get("verify_ssl", True),
            level=level,
        )

    def _create_rotating_file_handler(
        self, config: Dict[str, Any], level: int
    ) -> AsyncRotatingFileHandler:
        """Create a rotating file handler from configuration."""
        filename = config.get("filename")
        if not filename:
            raise ConfigurationError(
                "Rotating file handler must specify a filename"
            )

        return AsyncRotatingFileHandler(
            filename=filename,
            mode=config.get("mode", "a"),
            encoding=config.get("encoding", "utf-8"),
            delay=config.get("delay", False),
            errors=config.get("errors"),
            max_bytes=config.get("max_bytes", 0),
            backup_count=config.get("backup_count", 0),
            level=level,
        )

    def _create_timed_rotating_file_handler(
        self, config: Dict[str, Any], level: int
    ) -> AsyncTimedRotatingFileHandler:
        """Create a timed rotating file handler from configuration."""
        filename = config.get("filename")
        if not filename:
            raise ConfigurationError(
                "Timed rotating file handler must specify a filename"
            )

        when = config.get("when", "H")
        interval = config.get("interval", 1)
        backup_count = config.get("backup_count", 0)

        return AsyncTimedRotatingFileHandler(
            filename=filename,
            encoding=config.get("encoding", "utf-8"),
            delay=config.get("delay", False),
            errors=config.get("errors"),
            when=when,
            interval=interval,
            backup_count=backup_count,
            level=level,
        )


# Global configuration manager instance
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.

    Returns:
        Global configuration manager instance

    Example:
        >>> config_manager = get_config_manager()
        >>> config_manager.load_from_file("config.json")
        >>> logger = config_manager.get_logger("myapp")
    """
    return _config_manager


def configure_from_file(file_path: Union[str, Path]) -> None:
    """
    Configure logging from a file using the global configuration manager.

    Args:
        file_path: Path to the configuration file

    Example:
        >>> configure_from_file("config.json")
    """
    _config_manager.load_from_file(file_path)


def configure_from_dict(config: Dict[str, Any]) -> None:
    """
    Configure logging from a dictionary using the global configuration manager.

    Args:
        config: Configuration dictionary

    Example:
        >>> config = {
        ...     "version": 1,
        ...     "loggers": {"myapp": {
        ...         "level": "INFO",
        ...         "handlers": ["console"]
        ...     }},
        ...     "handlers": {"console": {
        ...         "class": "stream",
        ...         "stream": "stdout"
        ...     }}
        ... }
        >>> configure_from_dict(config)
    """
    _config_manager.load_from_dict(config)


def configure_from_env(prefix: str = "AIOLogging") -> None:
    """
    Configure logging from environment variables
    using the global configuration manager.

    Args:
        prefix: Prefix for environment variables

    Example:
        >>> configure_from_env()
    """
    _config_manager.load_from_env(prefix)


def get_configured_logger(name: str) -> AsyncLogger:
    """
    Get a logger configured according to the loaded configuration.

    Args:
        name: Name of the logger

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_configured_logger("myapp")
    """
    return _config_manager.get_logger(name)
