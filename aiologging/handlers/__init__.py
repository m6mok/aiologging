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
    AsyncHttpProtoHandler,
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
    "AsyncHttpProtoHandler",
    "AsyncHttpHandler",
    "AsyncTelegramHandler",
]
