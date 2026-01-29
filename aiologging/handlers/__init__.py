"""
Async handlers for aiologging library.

This package provides various async handlers for different output destinations
including streams, files, HTTP endpoints, and more.
"""

from typing import TYPE_CHECKING, Optional, Type, Any

from .base import AsyncHandler, BufferedAsyncHandler

# Helper function to safely import modules
def _safe_import(module_path: str, class_name: str) -> Optional[Type[Any]]:
    """Safely import a class from a module, returning None if import fails."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except (ImportError, AttributeError):
        return None

# Import handlers that don't require optional dependencies
AsyncStreamHandler = _safe_import(".stream", "AsyncStreamHandler")

# Import file handlers (require aiofiles)
if TYPE_CHECKING:
    from .file import AsyncFileHandler
    from .rotating import AsyncRotatingFileHandler, AsyncTimedRotatingFileHandler
    from .http import (
        AsyncHttpHandlerBase,
        AsyncHttpTextHandler,
        AsyncHttpJsonHandler,
        AsyncHttpProtoHandler,
        AsyncHttpHandler,
    )
else:
    # Import file handlers
    AsyncFileHandler = _safe_import(".file", "AsyncFileHandler")

    # Import rotating file handlers
    AsyncRotatingFileHandler = _safe_import(".rotating", "AsyncRotatingFileHandler")
    AsyncTimedRotatingFileHandler = _safe_import(".rotating", "AsyncTimedRotatingFileHandler")

    # Import HTTP handlers
    AsyncHttpHandlerBase = _safe_import(".http", "AsyncHttpHandlerBase")
    AsyncHttpTextHandler = _safe_import(".http", "AsyncHttpTextHandler")
    AsyncHttpJsonHandler = _safe_import(".http", "AsyncHttpJsonHandler")
    AsyncHttpProtoHandler = _safe_import(".http", "AsyncHttpProtoHandler")
    AsyncHttpHandler = _safe_import(".http", "AsyncHttpHandler")

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
]
