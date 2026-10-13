"""
Async handlers for aiologging library.

This package provides various async handlers for different output destinations
including streams, files, HTTP endpoints, and more.
"""

from .base import AsyncHandler, BufferedAsyncHandler
from .stream import AsyncStreamHandler
from .file import AsyncFileHandler
from .rotating import AsyncRotatingFileHandler, AsyncTimedRotatingFileHandler
from .http import (
    AsyncHttpHandlerBase,
    AsyncHttpTextHandler,
    AsyncHttpJsonHandler,
    AsyncHttpHandler,
)
from .telegram import AsyncTelegramHandler

__all__ = [
    "AsyncHandler",
    "BufferedAsyncHandler",
    "AsyncStreamHandler",
    "AsyncFileHandler",
    "AsyncRotatingFileHandler",
    "AsyncTimedRotatingFileHandler",
    "AsyncHttpHandlerBase",
    "AsyncHttpTextHandler",
    "AsyncHttpJsonHandler",
    "AsyncHttpHandler",
    "AsyncTelegramHandler",
]


# Standard handler names use the shared stdlib configuration. File rotation
# variants report failures explicitly for confirmed durable delivery.
import logging.handlers as _stdlib_handlers
from typing import Any
from ..extensions.files import (
    RotatingFileHandler as RotatingFileHandler,
    TimedRotatingFileHandler as TimedRotatingFileHandler,
)


def __getattr__(name: str) -> Any:
    return getattr(_stdlib_handlers, name)


__all__ += [
    "BaseRotatingHandler",
    "RotatingFileHandler",
    "TimedRotatingFileHandler",
    "WatchedFileHandler",
    "SocketHandler",
    "DatagramHandler",
    "SysLogHandler",
    "SMTPHandler",
    "NTEventLogHandler",
    "HTTPHandler",
    "BufferingHandler",
    "MemoryHandler",
    "QueueHandler",
    "QueueListener",
]
