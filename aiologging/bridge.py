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

For alerting channels the bridge can additionally deliver critical
records *inline* — synchronously, before they are queued — so "the
pager message leaves before the process dies"::

    aiologging.captureStdlib(inline_level=logging.ERROR)
"""

from __future__ import annotations

import logging
import threading
import time
from logging import NOTSET, LogRecord
from typing import Any, Optional

from .handlers.base import AsyncHandler
from .types import INLINE_SENDER_THREAD_ATTR
from .logger import (
    _IN_CONSUMER,
    _INLINE_HANDLED_ATTR,
    AsyncLoggerManager,
    _logger_manager,
)


class _TokenBucket:
    """
    Thread-safe token bucket bounding how often the bridge may block
    its caller with an inline send.
    """

    def __init__(self, rate_per_minute: float, burst: int) -> None:
        self._rate = rate_per_minute / 60.0
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Take one token if available; never blocks."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._updated) * self._rate,
            )
            self._updated = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class StdlibBridgeHandler(logging.Handler):
    """
    Synchronous ``logging.Handler`` that enqueues records for the
    aiologging consumer.

    Safe to call from any thread: records emitted outside the consumer
    loop are handed over via ``loop.call_soon_threadsafe``, and records
    emitted before any event loop exists are buffered until the
    consumer starts. Since ``emit`` cannot await, the "block" overflow
    policy degrades to "drop_old" for bridged records.

    With ``inline_level`` set, records at or above it are also offered
    — before queueing — to handlers that support synchronous delivery
    (:meth:`AsyncHandler.emit_sync`, e.g. ``AsyncTelegramHandler``).
    The inline path is bounded: ``inline_timeout`` seconds per record,
    and a token bucket of ``inline_burst`` sends refilling at
    ``inline_rate`` per minute, so a burst of errors cannot stall the
    calling thread. Any failure just leaves the record on the normal
    queue path. A handler that received the record inline is skipped
    by its queue worker; the other handlers get it as usual. Note the
    FIFO trade-off: an inline record reaches its sink ahead of older
    queued records — include ``%(asctime)s`` in the formatter so
    readers can reorder.
    """

    def __init__(
        self,
        level: int = NOTSET,
        manager: Optional[AsyncLoggerManager] = None,
        inline_level: Optional[int] = None,
        inline_timeout: float = 3.0,
        inline_burst: int = 1,
        inline_rate: float = 2.0,
    ) -> None:
        """
        Initialize the bridge handler.

        Args:
            level: Minimum level of stdlib records to forward
            manager: Target manager; None uses the global one
            inline_level: Minimum level for inline (synchronous)
                delivery; None disables it
            inline_timeout: Budget in seconds for one inline delivery
            inline_burst: Token bucket capacity for inline sends
            inline_rate: Token refill rate, in sends per minute
        """
        super().__init__(level)
        self._manager = manager
        self._inline_level = inline_level
        self._inline_timeout = inline_timeout
        self._inline_bucket = _TokenBucket(inline_rate, inline_burst)
        self._inline_local = threading.local()

    def handle(self, record: LogRecord) -> bool:
        """
        Filter and emit without the stdlib per-handler lock.

        The bridge is thread-safe on its own (``enqueue_from_thread``
        and the inline guards), while holding a lock across an inline
        send would deadlock: code logging from inside the send thread
        re-enters this handler and blocks on the lock held by the
        caller joining that very thread.
        """
        rv: Any = self.filter(record)
        if isinstance(rv, LogRecord):
            # 3.12+ filters may return a replacement record
            record = rv
        if rv:
            self.emit(record)
        return bool(rv)

    def emit(self, record: LogRecord) -> None:
        """Forward a stdlib record into the aiologging queue."""
        try:
            manager = (
                self._manager if self._manager is not None else _logger_manager
            )
            if (
                self._inline_level is not None
                and record.levelno >= self._inline_level
            ):
                self._deliver_inline(manager, record)
            manager.enqueue_from_thread(record)
        except Exception:
            self.handleError(record)

    def _deliver_inline(
        self, manager: AsyncLoggerManager, record: LogRecord
    ) -> None:
        """
        Offer the record to its handlers' ``emit_sync`` before it is
        queued; each handler that delivers is marked on the record so
        its queue worker skips it later.

        Guards, in order: records emitted by handler I/O itself
        (feedback loop), re-entrant emits on this thread (something
        logging from inside the inline path), and the token bucket.
        """
        if _IN_CONSUMER.get():
            return
        if getattr(self._inline_local, "active", False):
            return
        if getattr(
            threading.current_thread(), INLINE_SENDER_THREAD_ATTR, False
        ):
            # Emitted by code running inside an inline send thread
            return

        logger = manager.getLogger(record.name)
        if logger.disabled or not logger.filter(record):
            return
        handlers, _ = logger._collect_handlers(record)
        capable = [
            handler
            for handler in handlers
            if type(handler).emit_sync is not AsyncHandler.emit_sync
        ]
        if not capable:
            return
        if not self._inline_bucket.try_acquire():
            return

        deadline = time.monotonic() + self._inline_timeout
        self._inline_local.active = True
        try:
            for handler in capable:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    delivered = handler.emit_sync(record, remaining)
                except Exception:
                    delivered = False
                if delivered:
                    marked = getattr(record, _INLINE_HANDLED_ATTR, None)
                    if marked is None:
                        marked = set()
                        setattr(record, _INLINE_HANDLED_ATTR, marked)
                    marked.add(handler)
        finally:
            self._inline_local.active = False


_bridge_handler: Optional[StdlibBridgeHandler] = None


def captureStdlib(
    capture: bool = True,
    level: int = NOTSET,
    inline_level: Optional[int] = None,
    inline_timeout: float = 3.0,
    inline_burst: int = 1,
    inline_rate: float = 2.0,
) -> None:
    """
    Install (or remove) the bridge on the stdlib root logger, in the
    spirit of ``logging.captureWarnings``. Installing again replaces
    the bridge, applying the new settings.

    Args:
        capture: True installs the bridge, False removes it
        level: Minimum level of stdlib records to forward
        inline_level: Minimum level at which bridged records are also
            delivered inline — synchronously, before queueing — to
            handlers that support it (e.g. ``AsyncTelegramHandler``);
            None (default) disables inline delivery
        inline_timeout: Wall-clock budget in seconds for one inline
            delivery
        inline_burst: Number of back-to-back inline sends allowed
            before the rate limit kicks in
        inline_rate: Inline sends allowed per minute after the burst
    """
    global _bridge_handler
    stdlib_root = logging.getLogger()
    if capture:
        if _bridge_handler is not None:
            stdlib_root.removeHandler(_bridge_handler)
        _bridge_handler = StdlibBridgeHandler(
            level,
            inline_level=inline_level,
            inline_timeout=inline_timeout,
            inline_burst=inline_burst,
            inline_rate=inline_rate,
        )
        stdlib_root.addHandler(_bridge_handler)
    elif _bridge_handler is not None:
        stdlib_root.removeHandler(_bridge_handler)
        _bridge_handler = None
