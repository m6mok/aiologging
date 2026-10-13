"""Stdlib file/stream handlers with explicit, exception-based confirmation."""

from __future__ import annotations

import fcntl
import logging
import logging.handlers
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from ..journal import prepared_text, handler_recipient


def _render(handler: logging.Handler, record: logging.LogRecord) -> str:
    ready = prepared_text(record, handler_recipient(handler))
    return ready if ready is not None else handler.format(record)


def _sync_file(stream: Any) -> None:
    stream.flush()
    os.fsync(stream.fileno())
    if hasattr(fcntl, "F_FULLFSYNC"):
        fcntl.fcntl(stream.fileno(), fcntl.F_FULLFSYNC)


def _sync_directory(filename: str) -> None:
    descriptor = os.open(Path(filename).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _Confirmed:
    """A successful batch returns True; write failures propagate."""

    batch_size = 32

    def deliver(self, records: Sequence[logging.LogRecord]) -> bool:
        handler: Any = self
        handler.acquire()
        previous = getattr(handler, "_batch_depth", 0)
        handler._batch_depth = previous + 1
        try:
            for record in records:
                handler.emit(record)
            handler.flush()
            if isinstance(handler, logging.FileHandler) and handler.stream:
                _sync_file(handler.stream)
                _sync_directory(handler.baseFilename)
            return True
        finally:
            handler._batch_depth = previous
            handler.release()


class StreamHandler(_Confirmed, logging.StreamHandler[Any]):
    """A StreamHandler whose emit failures cannot hide behind handleError."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream.write(_render(self, record) + self.terminator)
        if not getattr(self, "_batch_depth", 0):
            self.flush()


class FileHandler(_Confirmed, logging.FileHandler):
    """Standard file configuration with flush/fullfsync before confirmation."""

    def emit(self, record: logging.LogRecord) -> None:
        _emit_file(self, record)


def _emit_file(handler: Any, record: logging.LogRecord) -> None:
    if handler.stream is None:
        if handler.mode == "w" and handler._closed:
            raise ValueError("Cannot reopen a closed truncating file handler")
        handler.stream = handler._open()
    handler.stream.write(_render(handler, record) + handler.terminator)
    if not getattr(handler, "_batch_depth", 0):
        _sync_file(handler.stream)
        _sync_directory(handler.baseFilename)


class RotatingFileHandler(_Confirmed, logging.handlers.RotatingFileHandler):
    """Size rotation with propagated write/rollover errors."""

    def emit(self, record: logging.LogRecord) -> None:
        if self.shouldRollover(record):
            if self.stream:
                _sync_file(self.stream)
            self.doRollover()
        _emit_file(self, record)


class TimedRotatingFileHandler(
    _Confirmed, logging.handlers.TimedRotatingFileHandler
):
    """Time rotation with propagated write/rollover errors."""

    def emit(self, record: logging.LogRecord) -> None:
        if self.shouldRollover(record):
            if self.stream:
                _sync_file(self.stream)
            self.doRollover()
        _emit_file(self, record)


for _emitter in (
    StreamHandler.emit,
    FileHandler.emit,
    RotatingFileHandler.emit,
    TimedRotatingFileHandler.emit,
):
    setattr(_emitter, "_prepared_rendering", True)
