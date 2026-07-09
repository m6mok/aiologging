"""
Bridge from standard :mod:`logging` into the aiologging pipeline.

Third-party libraries (aiohttp, sqlalchemy, ...) log through the
standard logging module. :class:`StdlibBridgeHandler` is a regular
synchronous ``logging.Handler`` that forwards their records into the
aiologging queue, so the whole application — your ``await`` calls and
library logs alike — flows through the same async handlers.

Enable it with::

    aiologging.captureStdlib()          # or
    aiologging.basicConfig(capture_stdlib=True)

Records are routed through the aiologging hierarchy under the same
logger name, so per-name handlers and propagation work as usual.
"""

from __future__ import annotations

import logging
from logging import NOTSET, LogRecord
from typing import Optional

from .logger import AsyncLoggerManager, _logger_manager


class StdlibBridgeHandler(logging.Handler):
    """
    Synchronous ``logging.Handler`` that enqueues records for the
    aiologging consumer.

    Safe to call from any thread: records emitted outside the consumer
    loop are handed over via ``loop.call_soon_threadsafe``, and records
    emitted before any event loop exists are buffered until the
    consumer starts. Since ``emit`` cannot await, the "block" overflow
    policy degrades to "drop_old" for bridged records.
    """

    def __init__(
        self,
        level: int = NOTSET,
        manager: Optional[AsyncLoggerManager] = None,
    ) -> None:
        """
        Initialize the bridge handler.

        Args:
            level: Minimum level of stdlib records to forward
            manager: Target manager; None uses the global one
        """
        super().__init__(level)
        self._manager = manager

    def emit(self, record: LogRecord) -> None:
        """Forward a stdlib record into the aiologging queue."""
        try:
            manager = (
                self._manager if self._manager is not None else _logger_manager
            )
            manager.enqueue_from_thread(record)
        except Exception:
            self.handleError(record)


_bridge_handler: Optional[StdlibBridgeHandler] = None


def captureStdlib(capture: bool = True, level: int = NOTSET) -> None:
    """
    Install (or remove) the bridge on the stdlib root logger, in the
    spirit of ``logging.captureWarnings``.

    Args:
        capture: True installs the bridge, False removes it
        level: Minimum level of stdlib records to forward
    """
    global _bridge_handler
    stdlib_root = logging.getLogger()
    if capture:
        if _bridge_handler is None:
            _bridge_handler = StdlibBridgeHandler(level)
            stdlib_root.addHandler(_bridge_handler)
    elif _bridge_handler is not None:
        stdlib_root.removeHandler(_bridge_handler)
        _bridge_handler = None
