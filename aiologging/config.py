"""
Configuration management for the aiologging library.

This module provides utilities for managing configuration of loggers and
handlers, including loading from files, environment variables,
and dictionaries.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Union, Type, cast

from .exceptions import ConfigurationError
from .handlers.base import AsyncHandler
from .handlers.stream import AsyncStreamHandler
from .handlers.file import AsyncFileHandler
from .handlers.http import AsyncHttpHandler, AsyncHttpJsonHandler
from .handlers.telegram import (
    AsyncTelegramHandler,
    TELEGRAM_MAX_MESSAGE_LENGTH,
)
from .handlers.rotating import (
    AsyncRotatingFileHandler,
    AsyncTimedRotatingFileHandler,
)
from .logger import AsyncLogger, _logger_manager
from .types import (
    ConfigValue,
    FormatterProtocol,
    LogLevel,
    LOG_LEVEL_MAP,
)

# Keys accepted at each level of the configuration dictionary;
# anything else is rejected explicitly instead of being silently
# ignored (a typo in "formater" must not go unnoticed).
_TOP_LEVEL_KEYS: FrozenSet[str] = frozenset(
    ("version", "loggers", "handlers", "formatters")
)
_LOGGER_KEYS: FrozenSet[str] = frozenset(
    ("type", "level", "handlers", "propagate", "disabled")
)
_FORMATTER_KEYS: FrozenSet[str] = frozenset(
    ("class", "format", "datefmt")
)
_COMMON_HANDLER_KEYS = ("class", "level", "formatter")
_HTTP_HANDLER_KEYS = _COMMON_HANDLER_KEYS + (
    "url", "method", "headers", "params", "timeout", "verify_ssl",
    "backend",
)
_BUILTIN_HANDLER_KEYS: Dict[str, FrozenSet[str]] = {
    "stream": frozenset(_COMMON_HANDLER_KEYS + ("stream",)),
    "file": frozenset(
        _COMMON_HANDLER_KEYS
        + ("filename", "mode", "encoding", "delay", "errors")
    ),
    "http": frozenset(_HTTP_HANDLER_KEYS),
    "http_json": frozenset(_HTTP_HANDLER_KEYS),
    "telegram": frozenset(
        _COMMON_HANDLER_KEYS
        + (
            "token", "chat_id", "parse_mode", "disable_notification",
            "message_thread_id", "max_message_length", "api_base_url",
            "timeout", "verify_ssl", "backend",
        )
    ),
    "rotating_file": frozenset(
        _COMMON_HANDLER_KEYS
        + (
            "filename", "mode", "encoding", "delay", "errors",
            "max_bytes", "backup_count",
        )
    ),
    "timed_rotating_file": frozenset(
        _COMMON_HANDLER_KEYS
        + (
            "filename", "encoding", "delay", "errors", "when",
            "interval", "backup_count", "utc",
        )
    ),
}


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
            "telegram": AsyncTelegramHandler,
            "rotating_file": AsyncRotatingFileHandler,
            "timed_rotating_file": AsyncTimedRotatingFileHandler,
        }
        self._logger_registry: Dict[str, Type[AsyncLogger]] = {
            "async": AsyncLogger,
        }
        self._config: Dict[str, Any] = {}
        self._loggers: Dict[str, AsyncLogger] = {}
        # One shared instance per named formatter, like dictConfig
        self._formatters: Dict[str, FormatterProtocol] = {}
        # get_logger is check-then-create: without a lock, racing
        # threads would each build the logger (and its handlers —
        # duplicating every line they write)
        self._get_logger_lock = threading.Lock()

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

        A handler's "class" is either a registered name ("stream",
        "file", ...) or a dotted path to an importable
        ``AsyncHandler`` subclass ("myapp.logging.MyHandler").
        For dotted-path and custom registered classes, all remaining
        keys are passed to the constructor as keyword arguments.

        An optional "formatters" section defines named formatters
        ("format", "datefmt" and an optional "class" dotted path,
        e.g. "aiologging.formatters.TelegramHtmlFormatter"); a
        handler references one via its "formatter" key. Unknown keys
        anywhere in the configuration raise ``ConfigurationError``.

        Args:
            config: Configuration dictionary

        Raises:
            ConfigurationError: If the configuration is malformed or
                contains unknown keys

        Example:
            >>> config = {
            ...     "version": 1,
            ...     "formatters": {
            ...         "plain": {
            ...             "format": "%(levelname)s %(message)s"
            ...         }
            ...     },
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
            ...             "stream": "stdout",
            ...             "formatter": "plain"
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
        self._formatters.clear()
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

        The logger lives in the global aiologging hierarchy (the one
        ``aiologging.getLogger`` serves): it has a ``parent``, its
        records propagate to ancestor handlers, and global
        ``flush()`` / ``shutdown()`` cover its handlers — the loaded
        configuration is applied on top.

        Args:
            name: Name of the logger

        Returns:
            Configured logger instance

        Example:
            >>> logger = config_manager.get_logger("myapp")
        """
        with self._get_logger_lock:
            return self._get_logger_locked(name)

    def _get_logger_locked(self, name: str) -> AsyncLogger:
        cached = self._loggers.get(name)
        # A cached logger closed by a global shutdown() is stale:
        # the hierarchy was reset and fresh instances took its place
        if cached is not None and not cached._closed:
            return cached

        if not self._config:
            raise ConfigurationError("No configuration loaded")

        logger_config = self._config.get("loggers", {}).get(name, {})
        if not logger_config:
            raise ConfigurationError(
                f"No configuration found for logger: {name}"
            )

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

        # Take the logger from the global hierarchy and apply the
        # configuration to it, like stdlib dictConfig — never build
        # a parallel hierarchy that flush()/shutdown() cannot see
        logger = self._materialize_logger(name, logger_class)
        logger.setLevel(level)
        for handler in handlers:
            logger.addHandler(handler)
        logger.propagate = logger_config.get("propagate", True)
        logger.disabled = logger_config.get("disabled", False)

        self._loggers[name] = logger
        return logger

    def _materialize_logger(
        self, name: str, logger_class: Type[AsyncLogger]
    ) -> AsyncLogger:
        """
        Get or create the named logger inside the global hierarchy.

        The default ``AsyncLogger`` comes straight from the global
        manager. An instance of a custom registered class is grafted
        into the hierarchy with the same parent/child wiring, so it
        is equally visible to propagation and ``flush()``.

        Args:
            name: The logger name
            logger_class: The class configured for this logger

        Returns:
            The logger, registered in the global hierarchy

        Raises:
            ConfigurationError: If the name is already taken by a
                logger of an incompatible class
        """
        existing = _logger_manager.loggerDict.get(name)
        if logger_class is AsyncLogger or isinstance(
            existing, logger_class
        ):
            return _logger_manager.getLogger(name)
        if existing is not None:
            raise ConfigurationError(
                f"Logger {name!r} already exists with class "
                f"{type(existing).__name__}, cannot apply type "
                f"{logger_class.__name__}",
                config_key="type",
            )

        logger = logger_class(name=name, manager=_logger_manager)
        _logger_manager.loggerDict[name] = logger
        if "." in name:
            parent = _logger_manager.getLogger(name.rsplit(".", 1)[0])
        else:
            parent = _logger_manager.root
        logger.parent = parent
        parent._children[name] = logger
        return logger

    def _validate_config(self) -> None:
        """
        Validate the loaded configuration.

        Beyond the structural checks, unknown keys at any level are
        rejected explicitly — a typo must fail loudly, not silently
        drop the intended setting.
        """
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

        if not isinstance(self._config["loggers"], dict):
            raise ConfigurationError(
                "The loggers section must be a dictionary"
            )
        if not isinstance(self._config["handlers"], dict):
            raise ConfigurationError(
                "The handlers section must be a dictionary"
            )

        self._reject_unknown_keys(
            self._config, _TOP_LEVEL_KEYS, "configuration"
        )

        for name, logger_config in self._config["loggers"].items():
            if not isinstance(logger_config, dict):
                raise ConfigurationError(
                    f"Logger {name!r} configuration must be a dictionary"
                )
            self._reject_unknown_keys(
                logger_config, _LOGGER_KEYS, f"logger {name!r}"
            )

        formatters = self._config.get("formatters", {})
        if not isinstance(formatters, dict):
            raise ConfigurationError(
                "The formatters section must be a dictionary"
            )
        for name, formatter_config in formatters.items():
            if not isinstance(formatter_config, dict):
                raise ConfigurationError(
                    f"Formatter {name!r} configuration must be a "
                    "dictionary"
                )
            self._reject_unknown_keys(
                formatter_config, _FORMATTER_KEYS, f"formatter {name!r}"
            )

        for name, handler_config in self._config["handlers"].items():
            if not isinstance(handler_config, dict):
                raise ConfigurationError(
                    f"Handler {name!r} configuration must be a dictionary"
                )
            allowed = _BUILTIN_HANDLER_KEYS.get(
                handler_config.get("class", "")
            )
            # Dotted-path and custom registered classes take free-form
            # constructor kwargs — only built-in classes are checked
            if allowed is not None:
                self._reject_unknown_keys(
                    handler_config, allowed, f"handler {name!r}"
                )
            formatter_name = handler_config.get("formatter")
            if (
                formatter_name is not None
                and formatter_name not in formatters
            ):
                raise ConfigurationError(
                    f"Handler {name!r} references unknown formatter: "
                    f"{formatter_name}",
                    config_key="formatter",
                    config_value=formatter_name,
                )

    @staticmethod
    def _reject_unknown_keys(
        config: Dict[str, Any],
        allowed: FrozenSet[str],
        context: str,
    ) -> None:
        """Raise on configuration keys outside the supported set."""
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ConfigurationError(
                f"Unknown key(s) in {context}: {', '.join(unknown)}. "
                f"Supported keys: {', '.join(sorted(allowed))}",
                config_key=unknown[0],
            )

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

        handler_class = self._resolve_handler_class(handler_type)

        # Parse the level
        level = self._parse_level(handler_config.get("level", "INFO"))

        # Create the handler based on its type
        handler: AsyncHandler
        if handler_type == "stream":
            handler = self._create_stream_handler(handler_config, level)
        elif handler_type == "file":
            handler = self._create_file_handler(handler_config, level)
        elif handler_type == "http":
            handler = self._create_http_handler(handler_config, level)
        elif handler_type == "http_json":
            handler = self._create_http_json_handler(handler_config, level)
        elif handler_type == "telegram":
            handler = self._create_telegram_handler(handler_config, level)
        elif handler_type == "rotating_file":
            handler = self._create_rotating_file_handler(
                handler_config, level
            )
        elif handler_type == "timed_rotating_file":
            handler = self._create_timed_rotating_file_handler(
                handler_config, level
            )
        else:
            handler = self._create_custom_handler(
                handler_class, handler_config, level
            )

        formatter_name = handler_config.get("formatter")
        if formatter_name is not None:
            handler.setFormatter(self._get_formatter(formatter_name))
        return handler

    def _get_formatter(self, name: str) -> FormatterProtocol:
        """
        Get (or create and cache) a named formatter, so handlers
        referencing the same name share one instance — mirroring
        stdlib ``dictConfig``.

        Args:
            name: Formatter name from the "formatters" section

        Returns:
            The formatter instance

        Raises:
            ConfigurationError: If the formatter is not configured
                or cannot be constructed
        """
        cached = self._formatters.get(name)
        if cached is not None:
            return cached

        formatter_config = self._config.get("formatters", {}).get(name)
        if formatter_config is None:
            raise ConfigurationError(
                f"No configuration found for formatter: {name}"
            )

        fmt = formatter_config.get("format")
        datefmt = formatter_config.get("datefmt")
        class_path = formatter_config.get("class")
        if class_path is None:
            formatter: FormatterProtocol = logging.Formatter(fmt, datefmt)
        else:
            formatter = self._create_formatter_from_class(
                name, class_path, fmt, datefmt
            )

        self._formatters[name] = formatter
        return formatter

    def _create_formatter_from_class(
        self,
        name: str,
        class_path: str,
        fmt: Optional[str],
        datefmt: Optional[str],
    ) -> FormatterProtocol:
        """
        Instantiate a formatter from a dotted path, like stdlib
        ``dictConfig``'s ``class`` key: the class is called with
        ``(fmt, datefmt)``.

        Args:
            name: Formatter name (for error messages)
            class_path: Dotted path to the formatter class
            fmt: The "format" configuration value
            datefmt: The "datefmt" configuration value

        Returns:
            The formatter instance

        Raises:
            ConfigurationError: If the class cannot be imported,
                constructed, or does not satisfy FormatterProtocol
        """
        if "." not in class_path:
            raise ConfigurationError(
                f"Formatter {name!r} class must be a dotted path: "
                f"{class_path!r}",
                config_key="class",
                config_value=class_path,
            )
        module_name, _, class_name = class_path.rpartition(".")
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ConfigurationError(
                f"Cannot import formatter class {class_path!r}: {e}",
                config_key="class",
                config_value=class_path,
            )
        resolved = getattr(module, class_name, None)
        if resolved is None:
            raise ConfigurationError(
                f"Module {module_name!r} has no attribute "
                f"{class_name!r}",
                config_key="class",
                config_value=class_path,
            )
        try:
            formatter = resolved(fmt, datefmt)
        except TypeError as e:
            raise ConfigurationError(
                f"Cannot create formatter {name!r}: {e}",
                config_value=class_path,
            )
        if not isinstance(formatter, FormatterProtocol):
            raise ConfigurationError(
                f"{class_path} does not satisfy FormatterProtocol",
                config_key="class",
                config_value=class_path,
            )
        return formatter

    def _resolve_handler_class(
        self, handler_type: str
    ) -> Type[AsyncHandler]:
        """
        Resolve a handler class from a registry name or a dotted path.

        Args:
            handler_type: Registered name (e.g. "stream") or dotted
                path to an importable handler class
                (e.g. "myapp.logging.MyHandler")

        Returns:
            The resolved handler class

        Raises:
            ConfigurationError: If the name is neither registered nor
                an importable ``AsyncHandler`` subclass
        """
        handler_class = self._handler_registry.get(handler_type)
        if handler_class is not None:
            return handler_class

        if "." not in handler_type:
            raise ConfigurationError(
                f"Unknown handler class: {handler_type}",
                config_key="class",
                config_value=handler_type,
            )

        module_name, _, class_name = handler_type.rpartition(".")
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ConfigurationError(
                f"Cannot import handler class {handler_type!r}: {e}",
                config_key="class",
                config_value=handler_type,
            )

        resolved = getattr(module, class_name, None)
        if resolved is None:
            raise ConfigurationError(
                f"Module {module_name!r} has no attribute "
                f"{class_name!r}",
                config_key="class",
                config_value=handler_type,
            )
        if not (
            isinstance(resolved, type)
            and issubclass(resolved, AsyncHandler)
        ):
            raise ConfigurationError(
                f"{handler_type} is not an AsyncHandler subclass",
                config_key="class",
                config_value=handler_type,
            )
        return resolved

    def _create_custom_handler(
        self,
        handler_class: Type[AsyncHandler],
        config: Dict[str, Any],
        level: int,
    ) -> AsyncHandler:
        """
        Create a registered or dotted-path handler from configuration.

        Every configuration key except ``class`` and ``level`` is
        passed to the handler constructor as a keyword argument,
        mirroring stdlib ``logging.config.dictConfig``.

        Args:
            handler_class: Handler class to instantiate
            config: Handler configuration dictionary
            level: Parsed logging level

        Returns:
            The created handler instance

        Raises:
            ConfigurationError: If the constructor rejects the
                configured keyword arguments
        """
        kwargs = {
            key: value
            for key, value in config.items()
            if key not in ("class", "level", "formatter")
        }
        try:
            return handler_class(level=level, **kwargs)
        except TypeError as e:
            raise ConfigurationError(
                f"Cannot create handler "
                f"{handler_class.__name__}: {e}",
                config_value=handler_class.__name__,
            )

    def _create_stream_handler(
        self, config: Dict[str, Any], level: int
    ) -> AsyncStreamHandler:
        """
        Create a stream handler from configuration.

        The "stream" key selects the target: "stdout" / "stderr"
        (or the stdlib-style "ext://sys.stdout" / "ext://sys.stderr"
        aliases); stderr is the default.
        """
        target = config.get("stream", "stderr")
        if target in ("stdout", "ext://sys.stdout"):
            stream = sys.stdout
        elif target in ("stderr", "ext://sys.stderr"):
            stream = sys.stderr
        else:
            raise ConfigurationError(
                f"Unsupported stream: {target!r}. Use 'stdout', "
                "'stderr', 'ext://sys.stdout' or 'ext://sys.stderr'",
                config_key="stream",
                config_value=target,
            )
        return AsyncStreamHandler(stream=stream, level=level)

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
            backend=config.get("backend"),
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
            backend=config.get("backend"),
        )

    def _create_telegram_handler(
        self, config: Dict[str, Any], level: int
    ) -> AsyncTelegramHandler:
        """Create a Telegram handler from configuration."""
        token = config.get("token")
        if not token:
            raise ConfigurationError(
                "Telegram handler must specify a token"
            )
        chat_id = config.get("chat_id")
        if chat_id is None:
            raise ConfigurationError(
                "Telegram handler must specify a chat_id"
            )

        return AsyncTelegramHandler(
            token=token,
            chat_id=chat_id,
            parse_mode=config.get("parse_mode"),
            disable_notification=config.get(
                "disable_notification", False
            ),
            message_thread_id=config.get("message_thread_id"),
            max_message_length=config.get(
                "max_message_length", TELEGRAM_MAX_MESSAGE_LENGTH
            ),
            api_base_url=config.get(
                "api_base_url", "https://api.telegram.org"
            ),
            timeout=config.get("timeout", 30.0),
            verify_ssl=config.get("verify_ssl", True),
            level=level,
            backend=config.get("backend"),
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
            utc=config.get("utc", False),
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

    A handler's "class" is either a registered name ("stream",
    "file", ...) or a dotted path to an importable ``AsyncHandler``
    subclass, as in stdlib ``logging.config.dictConfig``.

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
