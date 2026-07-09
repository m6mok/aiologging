"""
Async logger implementation for aiologging.

The public API mirrors the standard :mod:`logging` module: attribute and
method names, signatures and semantics follow ``logging.Logger``, with
the logging methods being coroutines.

Records are created synchronously at the call site (so caller
information, ``%``-formatting and ``exc_info`` behave exactly like in
standard logging) and put on a bounded queue. A background consumer
task drains the queue and fans each record out to per-handler dispatch
queues, each drained by its own worker task — the awaiting coroutine
never pays for handler I/O, and one slow handler (e.g. an HTTP sink
honouring a long retry-after) cannot delay delivery to the others.

The consumer starts lazily on the first logged record and is restarted
transparently if the running event loop changes (e.g. one loop per
test). Call ``await aiologging.shutdown()`` before the application
exits to drain the queue and close handlers.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import inspect
import io
import logging
import os
import sys
import time
import traceback
from collections import deque
from contextvars import ContextVar
from logging import CRITICAL, DEBUG, ERROR, INFO, NOTSET, WARNING, LogRecord
from types import FrameType, TracebackType
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

from .exceptions import ContextError
from .handlers.base import AsyncHandler
from .types import (
    AsyncErrorHandler,
    DropPolicyProtocol,
    FilterProtocol,
    RateLimiter,
)

# Accepted forms of the ``exc_info`` argument, mirroring standard logging
ExcInfoType = Union[
    bool,
    BaseException,
    "tuple[type[BaseException], BaseException, Optional[TracebackType]]",
    "tuple[None, None, None]",
    None,
]

# What ``await logger.info(...)`` waits for: "enqueue" resolves as soon
# as the record is on the queue, "await" resolves after the handlers
# have processed the record.
DeliveryMode = Literal["enqueue", "await"]

# What happens when the queue is full: "block" awaits for free space,
# "drop_new" discards the incoming record, "drop_old" discards the
# oldest queued record.
OverflowPolicy = Literal["block", "drop_new", "drop_old"]

_QueueItem = Tuple["AsyncLogger", LogRecord, "Optional[asyncio.Future[None]]"]

_DispatchItem = Tuple["AsyncLogger", LogRecord, "_RecordCompletion"]

# Set inside the consumer task so the stdlib bridge can detect (and
# drop) records emitted by the handlers themselves — otherwise a
# handler whose I/O library logs via stdlib logging would feed its own
# records back into the queue forever.
_IN_CONSUMER: ContextVar[bool] = ContextVar(
    "aiologging_in_consumer", default=False
)

# Attribute set on a record by the stdlib bridge when it delivered the
# record inline (synchronously, before queueing): the set of handlers
# that already received it. Their workers skip the record so it is not
# delivered twice; every other handler gets it from the queue as usual.
_INLINE_HANDLED_ATTR = "_aiologging_inline_handled"

# Used to render exc_info into exc_text when a record is frozen for the
# queue; any stdlib-compatible formatter picks exc_text up later.
_exception_formatter = logging.Formatter()


def _drain_dead_queue(queue: "asyncio.Queue[Any]") -> "List[Any]":
    """
    Take every item out of a queue that may belong to a dead loop.

    ``get_nowait`` cannot be used for this: it wakes the queue's
    pending putters, and waking a putter future bound to a closed
    loop raises ``RuntimeError`` mid-drain — aborting the rescue and
    losing records. Read the internal deque directly instead; the
    queue object is discarded right after, so its waiter bookkeeping
    no longer matters.
    """
    internal = getattr(queue, "_queue", None)
    if internal is None:  # pragma: no cover - non-CPython fallback
        items: List[Any] = []
        try:
            while True:
                items.append(queue.get_nowait())
        except (asyncio.QueueEmpty, RuntimeError):
            pass
        return items
    drained = list(internal)
    internal.clear()
    return drained


def _close_unstarted_task(task: "Optional[asyncio.Task[None]]") -> None:
    """
    Close the coroutine of a task stranded on a closed event loop.

    A consumer/worker task created right before its loop closed never
    runs its first step; when the task is garbage-collected the still
    unstarted coroutine emits ``RuntimeWarning: coroutine ... was
    never awaited``. Closing it silences that. Only tasks whose loop
    is already closed are touched — anywhere else the task can still
    run (or be cancelled) normally.
    """
    if task is None or task.done() or not task.get_loop().is_closed():
        return
    coro = task.get_coro()
    if (
        asyncio.iscoroutine(coro)
        and inspect.getcoroutinestate(coro) == inspect.CORO_CREATED
    ):
        coro.close()


def _check_level(level: Union[int, str]) -> int:
    """Convert a level name or number to a number, like stdlib logging."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        resolved = logging.getLevelName(level)
        if isinstance(resolved, int):
            return resolved
        raise ValueError(f"Unknown level: {level!r}")
    raise TypeError(f"Level not an integer or a valid string: {level!r}")


def _this_file() -> str:
    return __file__


_srcfile = os.path.normcase(_this_file.__code__.co_filename)


def _is_internal_frame(frame: FrameType) -> bool:
    """Signal whether the frame is from this module or import machinery."""
    filename = os.path.normcase(frame.f_code.co_filename)
    return filename == _srcfile or (
        "importlib" in filename and "_bootstrap" in filename
    )


def currentframe() -> Optional[FrameType]:
    """Return the frame of the caller."""
    if hasattr(sys, "_getframe"):
        return sys._getframe(1)
    try:  # pragma: no cover
        raise Exception
    except Exception as exc:
        tb = exc.__traceback__
        if tb is None or tb.tb_frame is None:
            return None
        return tb.tb_frame.f_back


def _freeze_record(record: LogRecord) -> LogRecord:
    """
    Prepare a record for the queue, like ``QueueHandler.prepare``.

    The message is rendered eagerly so later mutation of the caller's
    arguments cannot change it, and ``exc_info`` is converted to
    ``exc_text`` so traceback frames are not kept alive in the queue.
    Standard formatters use ``exc_text`` transparently.
    """
    message = record.getMessage()
    record.message = message
    record.msg = message
    record.args = None
    if record.exc_info is not None:
        if not record.exc_text:
            record.exc_text = _exception_formatter.formatException(
                record.exc_info
            )
        record.exc_info = None
    return record


class LevelAwareDrop:
    """
    Opt-in queue drop policy: under pressure, low-severity records
    are sacrificed before high-severity ones.

    Mirrors rsyslog's ``discardSeverity`` and logback's
    ``discardingThreshold``, both disabled by default — enabling this
    trades completeness of low-level context for the guarantee that
    errors survive overload:

    - an *arriving* record below ``discard_below`` is discarded once
      the queue is at least ``watermark`` full (below the watermark
      low-severity context flows normally);
    - when the queue is completely full, the oldest *queued* record
      below ``discard_below`` is evicted to make room for the
      arriving one; with no such victim the configured overflow
      policy applies as usual.

    Applies to the main record queue and the per-handler dispatch
    queues. Enable it with ``basicConfig(drop_policy=...)`` or by
    setting ``manager.drop_policy``.
    """

    def __init__(
        self,
        discard_below: Union[int, str] = WARNING,
        watermark: float = 0.8,
    ) -> None:
        """
        Initialize the policy.

        Args:
            discard_below: Records strictly below this level (number
                or name) are expendable; the default WARNING makes
                DEBUG and INFO records expendable
            watermark: Queue fill ratio (0..1] at which arriving
                expendable records start being discarded

        Raises:
            ValueError: If watermark is outside (0, 1] or the level
                name is unknown
        """
        if not 0.0 < watermark <= 1.0:
            raise ValueError("watermark must be within (0, 1]")
        self.discard_below = _check_level(discard_below)
        self.watermark = watermark

    def is_expendable(self, record: LogRecord) -> bool:
        """Signal that a queued record may be evicted when full."""
        return record.levelno < self.discard_below

    def should_discard_arriving(
        self, record: LogRecord, qsize: int, capacity: int
    ) -> bool:
        """Signal that an arriving record should be discarded."""
        return (
            capacity > 0
            and record.levelno < self.discard_below
            and qsize >= self.watermark * capacity
        )


class AsyncLoggerMetrics:
    """Metrics collection for async loggers."""

    def __init__(self, enable_metrics: bool = False) -> None:
        self.enable_metrics = enable_metrics
        self._records_created = 0
        self._records_processed = 0
        self._records_dropped = 0
        self._errors_count = 0
        self._last_error_time = 0.0

    def increment_created(self) -> None:
        """Increment created records count."""
        if self.enable_metrics:
            self._records_created += 1

    def increment_processed(self) -> None:
        """Increment processed records count."""
        if self.enable_metrics:
            self._records_processed += 1

    def increment_dropped(self) -> None:
        """Increment dropped records count."""
        if self.enable_metrics:
            self._records_dropped += 1

    def increment_errors(self) -> None:
        """Increment errors count and update last error time."""
        if self.enable_metrics:
            self._errors_count += 1
            self._last_error_time = time.time()

    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Get current metrics."""
        return {
            "records_created": self._records_created,
            "records_processed": self._records_processed,
            "records_dropped": self._records_dropped,
            "errors_count": self._errors_count,
            "last_error_time": self._last_error_time,
        }


class AsyncLogger:
    """
    Async logger with the ``logging.Logger`` API.

    The logging methods (``debug`` … ``critical``, ``log``,
    ``exception``) are coroutines; everything else matches standard
    logging: ``level``, ``parent``, ``propagate``, ``handlers``,
    ``filters`` and ``disabled`` attributes, ``setLevel``,
    ``getEffectiveLevel``, ``isEnabledFor``, ``getChild``,
    ``hasHandlers``, ``addHandler``/``removeHandler``,
    ``addFilter``/``removeFilter``, ``makeRecord`` and ``findCaller``.

    Usage::

        logger = aiologging.getLogger(__name__)
        await logger.info("hello %s", "world")

    Extensions beyond stdlib: ``delivery`` chooses what the await
    guarantees ("enqueue" or "await"), plus optional rate limiting,
    metrics and a custom async error handler.
    """

    def __init__(
        self,
        name: str,
        level: Union[int, str] = NOTSET,
        handlers: Optional[List[AsyncHandler]] = None,
        propagate: bool = True,
        disabled: bool = False,
        rate_limiter: Optional[RateLimiter] = None,
        enable_metrics: bool = False,
        error_handler: Optional[AsyncErrorHandler] = None,
        delivery: Optional[DeliveryMode] = None,
        manager: Optional[AsyncLoggerManager] = None,
    ) -> None:
        """
        Initialize the async logger.

        Args:
            name: The logger name
            level: The logging level (number or name)
            handlers: List of async handlers
            propagate: Whether to propagate to parent loggers
            disabled: Whether the logger is disabled
            rate_limiter: Optional rate limiter for the logger
            enable_metrics: Enable performance metrics collection
            error_handler: Optional error handler for logging errors
            delivery: What ``await logger.info(...)`` waits for; None
                falls back to the manager default
            manager: The owning manager; None uses the global one
        """
        self.name = name
        self.level = _check_level(level)
        self.handlers: List[AsyncHandler] = handlers or []
        self.propagate = propagate
        self.disabled = disabled
        self.parent: Optional[AsyncLogger] = None
        self.filters: List[FilterProtocol] = []
        self.delivery = delivery

        self._rate_limiter = rate_limiter
        self._error_handler = error_handler
        self._manager = manager
        self._children: Dict[str, AsyncLogger] = {}
        self._closed = False

        self._metrics = AsyncLoggerMetrics(enable_metrics)

    @property
    def manager(self) -> AsyncLoggerManager:
        """The manager owning this logger (global one by default)."""
        return self._manager if self._manager is not None else _logger_manager

    # ------------------------------------------------------------------
    # Configuration (sync, stdlib-compatible)
    # ------------------------------------------------------------------

    def setLevel(self, level: Union[int, str]) -> None:
        """Set the logging level for this logger."""
        self.level = _check_level(level)

    def getEffectiveLevel(self) -> int:
        """
        Get the effective level for this logger.

        Walks the hierarchy up to the first logger with a level other
        than NOTSET, like ``logging.Logger.getEffectiveLevel``.
        """
        logger: Optional[AsyncLogger] = self
        while logger is not None:
            if logger.level != NOTSET:
                return logger.level
            logger = logger.parent
        return NOTSET

    def isEnabledFor(self, level: int) -> bool:
        """Check if this logger is enabled for the given level."""
        if self.disabled:
            return False
        if self.manager.disable_level >= level:
            return False
        return level >= self.getEffectiveLevel()

    def getChild(self, suffix: str) -> AsyncLogger:
        """Get a logger which is a descendant to this one."""
        if self.manager.root is not self:
            suffix = ".".join((self.name, suffix))
        return self.manager.getLogger(suffix)

    def getChildren(self) -> Set[AsyncLogger]:
        """Get the immediate children of this logger."""
        return set(self._children.values())

    def addHandler(self, hdlr: AsyncHandler) -> None:
        """Add an async handler to this logger."""
        if hdlr not in self.handlers:
            self.handlers.append(hdlr)

    def removeHandler(self, hdlr: AsyncHandler) -> None:
        """Remove an async handler from this logger."""
        if hdlr in self.handlers:
            self.handlers.remove(hdlr)

    def hasHandlers(self) -> bool:
        """Check if this logger or any of its ancestors has handlers."""
        logger: Optional[AsyncLogger] = self
        while logger is not None:
            if logger.handlers:
                return True
            if not logger.propagate:
                break
            logger = logger.parent
        return False

    def addFilter(self, filter: FilterProtocol) -> None:
        """Add a filter to this logger."""
        if filter not in self.filters:
            self.filters.append(filter)

    def removeFilter(self, filter: FilterProtocol) -> None:
        """Remove a filter from this logger."""
        if filter in self.filters:
            self.filters.remove(filter)

    def filter(self, record: LogRecord) -> bool:
        """Apply this logger's filters; False means drop the record."""
        for filter_obj in self.filters:
            try:
                if not filter_obj.filter(record):
                    return False
            except Exception as e:
                # A failing filter drops the record to stay on the safe side
                sys.stderr.write(
                    f"Filter error in {type(filter_obj).__name__}: {e}\n"
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Logging methods (async, stdlib-compatible signatures)
    # ------------------------------------------------------------------

    async def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a message with severity DEBUG."""
        if self.isEnabledFor(DEBUG):
            await self._log(DEBUG, msg, args, **kwargs)

    async def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a message with severity INFO."""
        if self.isEnabledFor(INFO):
            await self._log(INFO, msg, args, **kwargs)

    async def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a message with severity WARNING."""
        if self.isEnabledFor(WARNING):
            await self._log(WARNING, msg, args, **kwargs)

    async def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Deprecated alias for :meth:`warning`."""
        await self.warning(msg, *args, **kwargs)

    async def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a message with severity ERROR."""
        if self.isEnabledFor(ERROR):
            await self._log(ERROR, msg, args, **kwargs)

    async def exception(
        self,
        msg: Any,
        *args: Any,
        exc_info: ExcInfoType = True,
        **kwargs: Any,
    ) -> None:
        """Log a message with severity ERROR including exception info."""
        await self.error(msg, *args, exc_info=exc_info, **kwargs)

    async def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a message with severity CRITICAL."""
        if self.isEnabledFor(CRITICAL):
            await self._log(CRITICAL, msg, args, **kwargs)

    async def fatal(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Alias for :meth:`critical`."""
        await self.critical(msg, *args, **kwargs)

    async def log(
        self, level: int, msg: Any, *args: Any, **kwargs: Any
    ) -> None:
        """Log a message with the given integer severity level."""
        if not isinstance(level, int):
            raise TypeError("level must be an integer")
        if self.isEnabledFor(level):
            await self._log(level, msg, args, **kwargs)

    async def _log(
        self,
        level: int,
        msg: Any,
        args: Tuple[Any, ...],
        exc_info: ExcInfoType = None,
        extra: Optional[Dict[str, Any]] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        """Create a record and hand it to :meth:`handle`."""
        if self._closed:
            return

        # Apply rate limiting if configured
        if self._rate_limiter is not None and not await self._rate_limiter():
            self._metrics.increment_dropped()
            return

        sinfo: Optional[str] = None
        try:
            fn, lno, func, sinfo = self.findCaller(stack_info, stacklevel)
        except ValueError:  # pragma: no cover - interpreter shutdown
            fn, lno, func = "(unknown file)", 0, "(unknown function)"

        # Normalize exc_info the same way the standard logging module
        # does: True means "use the current exception", an exception
        # instance is expanded into a (type, value, traceback) tuple
        norm_exc_info: Any = None
        if exc_info:
            if isinstance(exc_info, BaseException):
                norm_exc_info = (
                    type(exc_info),
                    exc_info,
                    exc_info.__traceback__,
                )
            elif isinstance(exc_info, tuple):
                norm_exc_info = exc_info
            else:
                norm_exc_info = sys.exc_info()

        record = self.makeRecord(
            self.name,
            level,
            fn,
            lno,
            msg,
            args,
            norm_exc_info,
            func,
            extra,
            sinfo,
        )
        self._metrics.increment_created()
        await self.handle(record)

    def makeRecord(
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: Any,
        args: Tuple[Any, ...],
        exc_info: Any,
        func: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        sinfo: Optional[str] = None,
    ) -> LogRecord:
        """Create a LogRecord; signature matches stdlib ``makeRecord``."""
        rv = logging.LogRecord(
            name, level, fn, lno, msg, args, exc_info, func, sinfo
        )
        if extra is not None:
            for key in extra:
                if (key in ("message", "asctime")) or (key in rv.__dict__):
                    raise KeyError(
                        f"Attempt to overwrite {key!r} in LogRecord"
                    )
                rv.__dict__[key] = extra[key]
        return rv

    def findCaller(
        self, stack_info: bool = False, stacklevel: int = 1
    ) -> Tuple[str, int, str, Optional[str]]:
        """
        Find the caller's source file, line number and function name,
        skipping frames that belong to this module.
        """
        f = currentframe()
        if f is None:
            return "(unknown file)", 0, "(unknown function)", None
        while stacklevel > 0:
            next_f = f.f_back
            if next_f is None:
                break
            f = next_f
            if not _is_internal_frame(f):
                stacklevel -= 1
        co = f.f_code
        sinfo: Optional[str] = None
        if stack_info:
            with io.StringIO() as sio:
                sio.write("Stack (most recent call last):\n")
                traceback.print_stack(f, file=sio)
                sinfo = sio.getvalue().rstrip("\n")
        return co.co_filename, f.f_lineno, co.co_name, sinfo

    # ------------------------------------------------------------------
    # Record processing
    # ------------------------------------------------------------------

    async def handle(self, record: LogRecord) -> None:
        """
        Handle a record: apply this logger's filters and enqueue it for
        the background consumer. What the await guarantees depends on
        the delivery mode ("enqueue" or "await").
        """
        if self._closed or self.disabled:
            return
        if not self.filter(record):
            return
        await self.manager.enqueue(self, record)

    def _collect_handlers(
        self, record: LogRecord
    ) -> Tuple[List[AsyncHandler], int]:
        """
        Collect the handlers a record goes to, walking the hierarchy
        while ``propagate`` is set (like stdlib ``callHandlers``).

        Returns the eligible handlers plus the total number of
        handlers seen (for the lastResort decision).
        """
        handlers: List[AsyncHandler] = []
        found = 0
        logger: Optional[AsyncLogger] = self
        while logger is not None:
            # A closed logger's handlers are closed too — skip them the
            # same way records on a closed logger are dropped
            if not logger._closed:
                for handler in list(logger.handlers):
                    found += 1
                    if record.levelno < handler.level:
                        continue
                    if (
                        getattr(handler, "error_handler", None) is None
                        and logger._error_handler is not None
                    ):
                        handler.error_handler = logger._error_handler
                    handlers.append(handler)
            logger = logger.parent if logger.propagate else None
        return handlers, found

    async def callHandlers(self, record: LogRecord) -> None:
        """
        Pass a record to all relevant handlers, walking the hierarchy
        while ``propagate`` is set (like stdlib ``callHandlers``).

        Awaits each handler in turn; the consumer instead fans records
        out to per-handler workers (see ``_HandlerDispatcher``) so a
        slow handler cannot delay the others. Handler errors are
        reported to stderr and never interrupt dispatch.
        """
        handlers, found = self._collect_handlers(record)
        inline_handled = getattr(record, _INLINE_HANDLED_ATTR, None)
        for handler in handlers:
            if inline_handled is not None and handler in inline_handled:
                continue
            try:
                await handler.handle(record)
            except Exception as e:
                sys.stderr.write(
                    f"Error in handler {type(handler).__name__}: "
                    f"{e}\n"
                    f"Record: {record.getMessage()}\n"
                )

        # Last-resort output, mirroring logging.lastResort
        if found == 0 and record.levelno >= WARNING:
            sys.stderr.write(f"{record.getMessage()}\n")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the logger, its handlers and child loggers."""
        if self._closed:
            return
        self._closed = True

        for handler in self.handlers:
            try:
                await handler.close()
            except Exception as e:
                sys.stderr.write(
                    f"Error closing handler {type(handler).__name__}: {e}\n"
                )

        for child in list(self._children.values()):
            try:
                await child.close()
            except Exception as e:
                sys.stderr.write(
                    f"Error closing child logger {child.name}: {e}\n"
                )

    def get_metrics(self) -> Dict[str, Union[int, float, bool]]:
        """Get performance metrics for the logger."""
        metrics: Dict[str, Union[int, float, bool]] = dict(
            self._metrics.get_metrics()
        )
        metrics.update(
            {
                "handlers_count": len(self.handlers),
                "filters_count": len(self.filters),
                "children_count": len(self._children),
                "closed": self._closed,
                "disabled": self.disabled,
            }
        )
        return metrics

    async def __aenter__(self) -> AsyncLogger:
        """Async context manager entry."""
        if self._closed:
            raise ContextError(
                "Cannot enter context: logger is closed",
                context_type="AsyncLogger",
                operation="__aenter__",
            )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Async context manager exit: flush queued records.

        The logger stays open — loggers are process-wide singletons and
        closing one here would silently kill every other user of the
        same name. Use ``aiologging.shutdown()`` at application exit.
        """
        await self.manager.flush()

    def __repr__(self) -> str:
        level = logging.getLevelName(self.getEffectiveLevel())
        return f"<{self.__class__.__name__} {self.name} ({level})>"


class _RecordCompletion:
    """
    Track one record across its per-handler dispatches.

    Created by the consumer when a record is fanned out; each handler
    worker (or a drop from a full dispatch queue) accounts for one
    dispatch. When the last one completes the record counts as
    processed and the delivery-"await" future, if any, is resolved.
    """

    __slots__ = ("_logger", "_remaining", "fut")

    def __init__(
        self,
        logger: AsyncLogger,
        remaining: int,
        fut: Optional[asyncio.Future[None]],
    ) -> None:
        self._logger = logger
        self._remaining = remaining
        self.fut = fut
        if remaining <= 0:
            self._finish()

    def done_one(self) -> None:
        """Account for one finished (or dropped) handler dispatch."""
        self._remaining -= 1
        if self._remaining <= 0:
            self._finish()

    def _finish(self) -> None:
        self._logger._metrics.increment_processed()
        fut = self.fut
        if fut is not None and not fut.done():
            try:
                fut.set_result(None)
            except RuntimeError:  # pragma: no cover - dead-loop future
                # The awaiting producer died with its event loop
                pass


class _HandlerDispatcher:
    """
    Per-handler dispatch queue plus worker task.

    The consumer fans records out to these instead of awaiting the
    handlers itself, so one slow handler (e.g. an HTTP sink honouring
    a long retry-after) cannot delay delivery to the other handlers.
    Each handler keeps its own FIFO queue, so per-handler ordering is
    preserved.

    Mirrors the consumer's lifecycle: the worker starts lazily, is
    rebuilt when the running event loop changes or the task died, and
    queued plus in-flight items are rescued into ``_pending`` when
    that happens.
    """

    def __init__(
        self, handler: AsyncHandler, manager: AsyncLoggerManager
    ) -> None:
        self._handler = handler
        self._manager = manager
        self._queue: Optional[asyncio.Queue[_DispatchItem]] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Unbounded on purpose: it only ever holds what the rescue
        # moves out of the old queue (≤ queue_size + the in-flight
        # item). A maxlen here silently evicted the in-flight item
        # when the rescued queue was full — an unaccounted loss.
        self._pending: Deque[_DispatchItem] = deque()
        self._in_flight: Optional[_DispatchItem] = None

    def running(self) -> bool:
        """Signal whether the worker task is alive."""
        return self._task is not None and not self._task.done()

    def undelivered(self) -> int:
        """Number of records not yet handled (for the atexit warning)."""
        queued = self._queue.qsize() if self._queue is not None else 0
        return queued + len(self._pending)

    def ensure_worker(self) -> None:
        """
        Start the worker task lazily; rebuild it if the running loop
        changed or the task died, rescuing stuck items.

        Must be called with a running event loop.
        """
        loop = asyncio.get_running_loop()
        if (
            self._task is not None
            and self._loop is loop
            and not self._task.done()
        ):
            return

        _close_unstarted_task(self._task)
        if self._queue is not None:
            if self._in_flight is not None:
                self._pending.appendleft(self._in_flight)
                self._in_flight = None
            self._pending.extend(_drain_dead_queue(self._queue))

        self._loop = loop
        # A rescue can carry queue_size + 1 items (the full old queue
        # plus the in-flight one); size the new queue to fit them all
        # so nothing is lost. Producers still block/shed against the
        # nominal size, and the surplus drains right away.
        maxsize = self._manager.queue_size
        if maxsize:
            maxsize = max(maxsize, len(self._pending))
        self._queue = asyncio.Queue(maxsize=maxsize)

        while self._pending:
            item = self._pending.popleft()
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:  # pragma: no cover - sized above
                self._manager._drop(item[0])
                item[2].done_one()

        self._task = loop.create_task(
            self._run(self._queue),
            name=f"aiologging-handler-{type(self._handler).__name__}",
        )

    async def put(self, item: _DispatchItem) -> None:
        """Enqueue an item, applying the manager's overflow policy."""
        self.ensure_worker()
        queue = self._queue
        if queue is None:  # pragma: no cover - ensure_worker creates it
            return

        manager = self._manager
        if manager.drop_policy is not None:
            if manager._shed_arriving(queue, item[1]):
                manager._account_dropped_item(item)
                return
            if manager._try_put_evicting(queue, item):
                return
            # no expendable victims: fall through to the configured
            # overflow policy below

        if self._manager.overflow == "block":
            await queue.put(item)
        elif self._manager.overflow == "drop_new":
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                self._manager._drop(item[0])
                item[2].done_one()
        else:  # drop_old
            while True:
                try:
                    queue.put_nowait(item)
                    return
                except asyncio.QueueFull:
                    try:
                        old = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        continue
                    queue.task_done()
                    self._manager._drop(old[0])
                    old[2].done_one()

    async def join(self) -> None:
        """Wait until every queued item has been handled."""
        if self._queue is None and not self._pending:
            return
        self.ensure_worker()
        queue = self._queue
        if queue is not None:
            await queue.join()

    async def _run(self, queue: asyncio.Queue[_DispatchItem]) -> None:
        """Drain the dispatch queue, handling records one at a time."""
        loop = asyncio.get_running_loop()
        token = _IN_CONSUMER.set(True)
        try:
            while True:
                item = await queue.get()
                _, record, completion = item
                handled = getattr(record, _INLINE_HANDLED_ATTR, None)
                if handled is not None and self._handler in handled:
                    # The bridge already delivered this record to this
                    # handler inline — account for the dispatch and move on
                    completion.done_one()
                    queue.task_done()
                    continue
                self._in_flight = item
                # _in_flight is cleared only on the completion paths:
                # any BaseException (CancelledError on teardown,
                # GeneratorExit when a dead loop's task is
                # GC-finalized) must leave the item in _in_flight so
                # a rebuilt worker can rescue and redeliver it.
                try:
                    await self._handler.handle(record)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if loop.is_closed():
                        # Teardown, not a handler failure: our loop
                        # died mid-emit and its artifacts raised.
                        # Keep the item for the rebuilt worker.
                        raise
                    sys.stderr.write(
                        f"Error in handler "
                        f"{type(self._handler).__name__}: {e}\n"
                        f"Record: {record.getMessage()}\n"
                    )
                    self._in_flight = None
                    completion.done_one()
                else:
                    self._in_flight = None
                    completion.done_one()
                finally:
                    queue.task_done()
        except RuntimeError:
            # Our loop died under us: the coroutine is being
            # GC-finalized and ``queue.get()``'s cleanup touched the
            # closed loop. Exit quietly — queued and in-flight items
            # are rescued when the worker is rebuilt.
            if not loop.is_closed():
                raise
        finally:
            try:
                _IN_CONSUMER.reset(token)
            except ValueError:
                # GC-finalized after its loop closed: the reset runs
                # in a foreign context. Cosmetic — the var dies with
                # the context the token was created in.
                pass

    async def suspend(self) -> None:
        """
        Cancel the worker, keeping queued records.

        Unlike :meth:`stop` the queue and pending records survive:
        ``ensure_worker`` rescues and redelivers them when records
        flow again (possibly on another loop). Used by ``flush_sync``
        so its private loop can close without pending tasks.
        """
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            _close_unstarted_task(task)
        self._task = None

    async def stop(self) -> None:
        """Cancel the worker and reset the dispatcher state."""
        await self.suspend()
        self._queue = None
        self._loop = None
        self._pending.clear()
        self._in_flight = None


class AsyncLoggerManager:
    """
    Manager owning the logger hierarchy and the record queue.

    Mirrors ``logging.Manager`` for the hierarchy part (``getLogger``
    with dotted names, a ``root`` logger at WARNING) and additionally
    owns the machinery that makes logging asynchronous: a bounded
    ``asyncio.Queue`` plus a background consumer task that fans
    records out to per-handler dispatch queues, each drained by its
    own worker task — so one slow handler cannot delay the others.

    The consumer and the handler workers start lazily on the first
    record and are rebuilt automatically when the running event loop
    changes.
    """

    def __init__(
        self,
        queue_size: int = 10_000,
        overflow: OverflowPolicy = "block",
        delivery: DeliveryMode = "enqueue",
        drop_policy: Optional[DropPolicyProtocol] = None,
    ) -> None:
        self.loggerDict: Dict[str, AsyncLogger] = {}
        self.disable_level: int = NOTSET
        self.queue_size = queue_size
        self.overflow: OverflowPolicy = overflow
        self.delivery: DeliveryMode = delivery
        # Optional level-aware drop policy (e.g. LevelAwareDrop);
        # None keeps the pure FIFO overflow behaviour
        self.drop_policy = drop_policy
        self.root = AsyncLogger("root", level=WARNING, manager=self)

        self._queue: Optional[asyncio.Queue[_QueueItem]] = None
        self._consumer_task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Records that arrived while no event loop was available (e.g.
        # from the stdlib bridge before the app started) and records
        # rescued from a dead loop's queue; drained into the queue
        # when the consumer starts. Unbounded: the cold-start path
        # enforces the queue_size bound itself (with drop accounting),
        # and a rescue never holds more than the old queue plus the
        # in-flight record.
        self._pending: Deque[Tuple[AsyncLogger, LogRecord]] = deque()
        # The record the consumer is currently dispatching; rescued
        # together with the queue when the event loop changes
        self._in_flight: Optional[Tuple[AsyncLogger, LogRecord]] = None
        self._records_dropped = 0
        # One dispatch queue + worker per handler, created on first
        # dispatch, so a slow handler never delays the other handlers
        self._dispatchers: Dict[AsyncHandler, _HandlerDispatcher] = {}

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def getLogger(self, name: str) -> AsyncLogger:
        """
        Get or create a logger with the given hierarchical name.

        Intermediate loggers are created eagerly, so ``parent`` links
        always point to the immediate dotted-name parent.
        """
        if not isinstance(name, str):
            raise TypeError("A logger name must be a string")
        if not name or name == self.root.name:
            return self.root
        if name in self.loggerDict:
            return self.loggerDict[name]

        logger = AsyncLogger(name, manager=self)
        self.loggerDict[name] = logger

        if "." in name:
            parent = self.getLogger(name.rsplit(".", 1)[0])
        else:
            parent = self.root
        logger.parent = parent
        parent._children[name] = logger

        return logger

    # ------------------------------------------------------------------
    # Queue and consumer
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        logger: AsyncLogger,
        record: LogRecord,
        delivery: Optional[DeliveryMode] = None,
    ) -> None:
        """
        Put a record on the queue, applying the overflow policy.

        With delivery "await" the coroutine resolves only after the
        consumer has passed the record to the handlers.
        """
        _freeze_record(record)
        self._ensure_consumer()
        queue = self._queue
        if queue is None:  # pragma: no cover - _ensure_consumer creates it
            return

        mode = delivery or logger.delivery or self.delivery
        fut: Optional[asyncio.Future[None]] = None
        if mode == "await":
            fut = asyncio.get_running_loop().create_future()
        item: _QueueItem = (logger, record, fut)

        placed = False
        if self.drop_policy is not None:
            if self._shed_arriving(queue, record):
                self._account_dropped_item(item)
                return
            placed = self._try_put_evicting(queue, item)
            # no expendable victims: fall through to the configured
            # overflow policy below

        if not placed:
            if self.overflow == "block":
                await queue.put(item)
            elif self.overflow == "drop_new":
                try:
                    queue.put_nowait(item)
                except asyncio.QueueFull:
                    self._drop(logger)
                    if fut is not None:
                        fut.set_result(None)
                    return
            else:  # drop_old
                self._put_drop_old(queue, item)

        if fut is not None:
            await fut

    def _put_drop_old(
        self, queue: asyncio.Queue[_QueueItem], item: _QueueItem
    ) -> None:
        """Put an item, discarding the oldest queued records if full."""
        while True:
            try:
                queue.put_nowait(item)
                return
            except asyncio.QueueFull:
                try:
                    old = queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
                queue.task_done()
                self._drop(old[0])
                old_fut = old[2]
                if old_fut is not None and not old_fut.done():
                    old_fut.set_result(None)

    def _drop(self, logger: Optional[AsyncLogger]) -> None:
        """Account for a dropped record."""
        self._records_dropped += 1
        if logger is not None:
            logger._metrics.increment_dropped()

    def _account_dropped_item(self, item: Any) -> None:
        """
        Account for a dropped queue item and unblock whoever waits
        on it: works for main-queue items (delivery-"await" future)
        and dispatch items (``_RecordCompletion``) alike.
        """
        self._drop(item[0])
        tail = item[2]
        if tail is None:
            return
        if isinstance(tail, _RecordCompletion):
            tail.done_one()
        elif not tail.done():
            tail.set_result(None)

    def _shed_arriving(
        self, queue: asyncio.Queue[Any], record: LogRecord
    ) -> bool:
        """Ask the drop policy whether to discard the arriving record."""
        policy = self.drop_policy
        if policy is None:
            return False
        return policy.should_discard_arriving(
            record, queue.qsize(), queue.maxsize
        )

    def _evict_expendable(self, queue: asyncio.Queue[Any]) -> Optional[Any]:
        """
        Remove and return the oldest queued item whose record the
        drop policy deems expendable; None when the policy is unset
        or no queued record qualifies.

        Reaches into the queue's internal deque — asyncio.Queue can
        only remove the head. ``task_done()`` keeps the ``join()``
        bookkeeping consistent with the silent removal.
        """
        policy = self.drop_policy
        if policy is None:
            return None
        internal = getattr(queue, "_queue", None)
        if internal is None:  # pragma: no cover - CPython detail
            return None
        for index, queued in enumerate(internal):
            if policy.is_expendable(queued[1]):
                del internal[index]
                queue.task_done()
                return queued
        return None

    def _try_put_evicting(
        self, queue: asyncio.Queue[Any], item: Any
    ) -> bool:
        """
        ``put_nowait``, evicting expendable queued records to make
        room; False when the queue stays full (no victims — the
        caller falls back to the configured overflow policy).
        """
        while True:
            try:
                queue.put_nowait(item)
                return True
            except asyncio.QueueFull:
                evicted = self._evict_expendable(queue)
                if evicted is None:
                    return False
                self._account_dropped_item(evicted)

    def enqueue_from_thread(self, record: LogRecord) -> None:
        """
        Thread-safe synchronous enqueue used by the stdlib bridge.

        Routes the record through the aiologging hierarchy under the
        same name. Records emitted from inside the consumer (handler
        I/O libraries logging via stdlib) are dropped to prevent
        feedback loops. Since a sync producer cannot await, the "block"
        policy degrades to "drop_old" here.
        """
        if _IN_CONSUMER.get():
            self._drop(None)
            return

        _freeze_record(record)
        logger = self.getLogger(record.name)
        if logger.disabled or not logger.filter(record):
            return

        loop = self._loop
        try:
            running: Optional[asyncio.AbstractEventLoop] = (
                asyncio.get_running_loop()
            )
        except RuntimeError:
            running = None

        if (
            loop is not None
            and not loop.is_closed()
            and loop.is_running()
            and running is not loop
        ):
            # Emitted from a foreign thread — hop into the consumer loop
            loop.call_soon_threadsafe(self._put_nowait_sync, logger, record)
        elif running is not None:
            self._put_nowait_sync(logger, record)
        else:
            # No loop anywhere yet — buffer until the consumer starts,
            # holding at most queue_size cold records (the oldest are
            # dropped with accounting, mirroring drop_old)
            while len(self._pending) >= self.queue_size > 0:
                try:
                    old_logger, _ = self._pending.popleft()
                except IndexError:  # pragma: no cover - racing thread
                    break
                self._drop(old_logger)
            self._pending.append((logger, record))

    def _put_nowait_sync(
        self, logger: AsyncLogger, record: LogRecord
    ) -> None:
        """Synchronous put for producers that cannot await."""
        self._ensure_consumer()
        queue = self._queue
        if queue is None:  # pragma: no cover - _ensure_consumer creates it
            return
        item: _QueueItem = (logger, record, None)
        if self.drop_policy is not None:
            if self._shed_arriving(queue, record):
                self._drop(logger)
                return
            if self._try_put_evicting(queue, item):
                return
            # no expendable victims: fall through to the sync fallback
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            if self.overflow == "drop_new":
                self._drop(logger)
            else:
                self._put_drop_old(queue, item)

    def _ensure_consumer(self) -> None:
        """
        Start the consumer task lazily; rebuild it if the running loop
        changed (e.g. a new loop per test) or the task died.

        Must be called with a running event loop.
        """
        loop = asyncio.get_running_loop()
        if (
            self._consumer_task is not None
            and self._loop is loop
            and not self._consumer_task.done()
        ):
            return

        # Rebuilding: rescue records stuck in the previous queue (the
        # loop changed or the consumer died), including the one the
        # dead consumer was dispatching when it was interrupted
        _close_unstarted_task(self._consumer_task)
        if self._queue is not None:
            if self._in_flight is not None:
                self._pending.appendleft(self._in_flight)
                self._in_flight = None
            for old_logger, old_record, _ in _drain_dead_queue(
                self._queue
            ):
                self._pending.append((old_logger, old_record))

        self._loop = loop
        # Size the new queue to fit the whole rescue (the full old
        # queue plus the in-flight record); see _HandlerDispatcher.
        maxsize = self.queue_size
        if maxsize:
            maxsize = max(maxsize, len(self._pending))
        self._queue = asyncio.Queue(maxsize=maxsize)

        while self._pending:
            pending_logger, pending_record = self._pending.popleft()
            try:
                self._queue.put_nowait((pending_logger, pending_record, None))
            except asyncio.QueueFull:  # pragma: no cover - sized above
                self._drop(pending_logger)

        self._consumer_task = loop.create_task(
            self._consume(self._queue), name="aiologging-consumer"
        )

    async def _consume(self, queue: asyncio.Queue[_QueueItem]) -> None:
        """Drain the queue, fanning records out to handler workers."""
        loop = asyncio.get_running_loop()
        token = _IN_CONSUMER.set(True)
        try:
            while True:
                logger, record, fut = await queue.get()
                self._in_flight = (logger, record)
                # _in_flight is cleared only on the completion paths:
                # any BaseException (CancelledError on teardown,
                # GeneratorExit when a dead loop's task is
                # GC-finalized) must leave the record in _in_flight
                # so a rebuilt consumer can rescue and redeliver it.
                try:
                    await self._dispatch(logger, record, fut)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if loop.is_closed():
                        # Teardown, not a dispatch failure: our loop
                        # died mid-dispatch and its artifacts raised.
                        # Keep the record for the rebuilt consumer.
                        raise
                    logger._metrics.increment_errors()
                    sys.stderr.write(
                        f"Error in logger {logger.name}: {e}\n"
                    )
                    # The completion may never fire; unblock the
                    # producer awaiting delivery
                    if fut is not None and not fut.done():
                        fut.set_result(None)
                    self._in_flight = None
                else:
                    self._in_flight = None
                finally:
                    queue.task_done()
        except RuntimeError:
            # Our loop died under us: the coroutine is being
            # GC-finalized and ``queue.get()``'s cleanup touched the
            # closed loop. Exit quietly — queued and in-flight
            # records are rescued when the consumer is rebuilt.
            if not loop.is_closed():
                raise
        finally:
            try:
                _IN_CONSUMER.reset(token)
            except ValueError:
                # GC-finalized after its loop closed: the reset runs
                # in a foreign context. Cosmetic — the var dies with
                # the context the token was created in.
                pass

    async def _dispatch(
        self,
        logger: AsyncLogger,
        record: LogRecord,
        fut: Optional[asyncio.Future[None]],
    ) -> None:
        """
        Fan a record out to the dispatch queues of all its handlers.

        The hierarchy walk and level checks happen here, synchronously;
        the handler I/O runs in the per-handler workers, so a slow
        handler only delays its own queue. With the "block" overflow
        policy a full dispatch queue suspends the consumer, which
        propagates backpressure to the main queue and the producers.
        """
        handlers, found = logger._collect_handlers(record)

        # Last-resort output, mirroring logging.lastResort
        if found == 0 and record.levelno >= WARNING:
            sys.stderr.write(f"{record.getMessage()}\n")

        completion = _RecordCompletion(logger, len(handlers), fut)
        for handler in handlers:
            dispatcher = self._dispatchers.get(handler)
            if dispatcher is None:
                dispatcher = _HandlerDispatcher(handler, self)
                self._dispatchers[handler] = dispatcher
            await dispatcher.put((logger, record, completion))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def flush(self, timeout: Optional[float] = None) -> None:
        """
        Wait until every queued record has been handled, then force
        buffered handlers to flush.

        Args:
            timeout: Optional wall-clock bound in seconds; when it
                expires ``asyncio.TimeoutError`` is raised and any
                undelivered records stay queued

        Raises:
            asyncio.TimeoutError: If the drain outlived ``timeout``
        """
        if timeout is None:
            await self._flush()
        else:
            await asyncio.wait_for(self._flush(), timeout)

    async def _flush(self) -> None:
        """Unbounded drain: main queue, dispatch queues, buffers."""
        if self._pending:
            self._ensure_consumer()
        if self._queue is not None:
            self._ensure_consumer()
            queue = self._queue
            if queue is not None:
                await queue.join()

        # The consumer marks a record done only after fanning it out,
        # so once the main queue is joined every record sits in a
        # dispatch queue — join those to wait for the handler I/O
        for dispatcher in list(self._dispatchers.values()):
            await dispatcher.join()

        for logger in [self.root, *self.loggerDict.values()]:
            for handler in logger.handlers:
                force_flush = getattr(handler, "force_flush", None)
                if force_flush is not None:
                    try:
                        await force_flush()
                    except Exception as e:
                        sys.stderr.write(
                            "Error flushing handler "
                            f"{type(handler).__name__}: {e}\n"
                        )

    async def shutdown(self, timeout: Optional[float] = None) -> None:
        """
        Drain the queue, stop the consumer, close every handler and
        reset the hierarchy to a pristine state.

        Loggers obtained before shutdown are closed; ``getLogger``
        returns fresh instances afterwards.

        Args:
            timeout: Optional bound in seconds for the drain phase
                (e.g. a container's termination grace period); when it
                expires the remaining records are dropped and teardown
                proceeds
        """
        try:
            await self.flush(timeout=timeout)
        except asyncio.TimeoutError:
            sys.stderr.write(
                f"aiologging: flush timed out after {timeout}s during "
                "shutdown; undelivered records were dropped\n"
            )
        except Exception as e:
            sys.stderr.write(f"Error flushing during shutdown: {e}\n")

        task = self._consumer_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            _close_unstarted_task(task)
        self._consumer_task = None
        self._queue = None
        self._loop = None
        self._pending.clear()
        self._in_flight = None

        for dispatcher in list(self._dispatchers.values()):
            await dispatcher.stop()
        self._dispatchers.clear()

        for logger in [self.root, *self.loggerDict.values()]:
            try:
                await logger.close()
            except Exception as e:
                sys.stderr.write(
                    f"Error closing logger {logger.name}: {e}\n"
                )

        self.loggerDict.clear()
        self.root = AsyncLogger("root", level=WARNING, manager=self)

    def flush_sync(self, timeout: float = 5.0) -> bool:
        """
        Synchronous emergency drain for contexts without a usable
        event loop: ``finally`` blocks after ``asyncio.run`` returned,
        ``atexit`` hooks, plain synchronous ``main`` functions.

        Runs the drain on a private event loop (the consumer, the
        handler workers and the handler resources all rebuild there
        and are suspended again afterwards). If the manager's loop is
        alive in another thread, the drain is submitted to it instead.

        Best effort by design: failures are reported in the return
        value, not raised, because the caller is typically a dying
        process.

        Args:
            timeout: Wall-clock bound in seconds for the whole drain

        Returns:
            True if everything was delivered within the timeout

        Raises:
            RuntimeError: If called from inside a running event loop
                (``await flush()`` must be used there instead)
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "flush_sync() cannot be called from a running event "
                "loop; use 'await aiologging.flush()' instead"
            )

        loop = self._loop
        if loop is not None and not loop.is_closed() and loop.is_running():
            # The consumer's loop is alive in another thread
            future = asyncio.run_coroutine_threadsafe(self.flush(), loop)
            try:
                future.result(timeout)
                return True
            except concurrent.futures.TimeoutError:
                future.cancel()
                return False
            except Exception:
                return False

        private = asyncio.new_event_loop()
        try:
            return private.run_until_complete(self._drain_bounded(timeout))
        except Exception:
            return False
        finally:
            private.close()

    async def _drain_bounded(self, timeout: float) -> bool:
        """
        Drain under a wall-clock bound, then suspend the consumer and
        the workers so the (private) loop can close without leaving
        pending tasks behind.
        """
        drained = True
        try:
            await self.flush(timeout=timeout)
        except asyncio.TimeoutError:
            drained = False
        except Exception:
            drained = False
        finally:
            await self._suspend()
        return drained

    async def _suspend(self) -> None:
        """
        Cancel the consumer and the handler workers, keeping queued
        records and handler state so logging can resume on a later
        loop. Handler resources bound to the current loop (HTTP
        sessions, files) are released so they do not die with it.
        """
        task = self._consumer_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            _close_unstarted_task(task)
        self._consumer_task = None

        for dispatcher in list(self._dispatchers.values()):
            await dispatcher.suspend()

        for logger in [self.root, *self.loggerDict.values()]:
            for handler in logger.handlers:
                try:
                    await handler._close_resources()
                except Exception:
                    pass

    def undelivered(self) -> int:
        """
        Number of records not yet delivered: queued, pending, in
        flight on dispatchers, or sitting in handler buffers.
        """
        queue = self._queue
        count = (queue.qsize() if queue is not None else 0) + len(
            self._pending
        )
        for dispatcher in self._dispatchers.values():
            count += dispatcher.undelivered()
        for logger in [self.root, *self.loggerDict.values()]:
            for handler in logger.handlers:
                buffered = getattr(handler, "_buffer", None)
                if buffered is not None:
                    count += len(buffered)
                priority = getattr(handler, "_priority_buffer", None)
                if priority is not None:
                    count += len(priority)
        return count

    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Get queue and consumer metrics."""
        return {
            "queue_length": self._queue.qsize() if self._queue else 0,
            "queue_capacity": self.queue_size,
            "pending_records": len(self._pending),
            "records_dropped": self._records_dropped,
            "consumer_running": int(
                self._consumer_task is not None
                and not self._consumer_task.done()
            ),
            "handler_workers_running": sum(
                int(dispatcher.running())
                for dispatcher in self._dispatchers.values()
            ),
        }


# Global logger manager instance
_logger_manager = AsyncLoggerManager()


def getLogger(name: Optional[str] = None) -> AsyncLogger:
    """
    Get or create a logger with the given hierarchical name.

    Without a name (or with "root") the root logger is returned,
    exactly like ``logging.getLogger``.
    """
    if name:
        return _logger_manager.getLogger(name)
    return _logger_manager.root


def disable(level: int = CRITICAL) -> None:
    """
    Disable all logging calls of severity ``level`` and below,
    like ``logging.disable``.
    """
    _logger_manager.disable_level = level


async def flush(timeout: Optional[float] = None) -> None:
    """
    Wait until every queued record has been handled.

    Args:
        timeout: Optional wall-clock bound in seconds
            (``asyncio.TimeoutError`` is raised when it expires)
    """
    await _logger_manager.flush(timeout=timeout)


async def shutdown(timeout: Optional[float] = None) -> None:
    """
    Drain the queue, close all handlers and reset the hierarchy.

    Call this once before the application exits, while the event loop
    is still running.

    Args:
        timeout: Optional bound in seconds for the drain phase; on
            expiry the remaining records are dropped and teardown
            proceeds
    """
    await _logger_manager.shutdown(timeout=timeout)


def flush_sync(timeout: float = 5.0) -> bool:
    """
    Synchronous emergency drain for contexts without a running event
    loop: a ``finally`` after ``asyncio.run``, an ``atexit`` hook, a
    signal-initiated teardown in synchronous code.

    Undelivered records (including buffered ones in HTTP/Telegram
    handlers) are delivered on a private event loop. Best effort:
    returns False instead of raising when the timeout expires or
    delivery fails.

    Args:
        timeout: Wall-clock bound in seconds for the whole drain

    Returns:
        True if everything was delivered within the timeout

    Raises:
        RuntimeError: If called from inside a running event loop;
            use ``await aiologging.flush()`` there instead
    """
    return _logger_manager.flush_sync(timeout=timeout)


def _auto_configure_root() -> None:
    """Add a default stderr handler to root, like stdlib basicConfig."""
    root = _logger_manager.root
    if root.hasHandlers():
        return
    from .handlers.stream import AsyncStreamHandler

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    root.addHandler(AsyncStreamHandler(formatter=formatter))


async def debug(msg: Any, *args: Any, **kwargs: Any) -> None:
    """Log a DEBUG message on the root logger."""
    _auto_configure_root()
    await _logger_manager.root.debug(msg, *args, **kwargs)


async def info(msg: Any, *args: Any, **kwargs: Any) -> None:
    """Log an INFO message on the root logger."""
    _auto_configure_root()
    await _logger_manager.root.info(msg, *args, **kwargs)


async def warning(msg: Any, *args: Any, **kwargs: Any) -> None:
    """Log a WARNING message on the root logger."""
    _auto_configure_root()
    await _logger_manager.root.warning(msg, *args, **kwargs)


async def warn(msg: Any, *args: Any, **kwargs: Any) -> None:
    """Deprecated alias for :func:`warning`."""
    await warning(msg, *args, **kwargs)


async def error(msg: Any, *args: Any, **kwargs: Any) -> None:
    """Log an ERROR message on the root logger."""
    _auto_configure_root()
    await _logger_manager.root.error(msg, *args, **kwargs)


async def exception(
    msg: Any, *args: Any, exc_info: ExcInfoType = True, **kwargs: Any
) -> None:
    """Log an ERROR message with exception info on the root logger."""
    _auto_configure_root()
    await _logger_manager.root.exception(
        msg, *args, exc_info=exc_info, **kwargs
    )


async def critical(msg: Any, *args: Any, **kwargs: Any) -> None:
    """Log a CRITICAL message on the root logger."""
    _auto_configure_root()
    await _logger_manager.root.critical(msg, *args, **kwargs)


fatal = critical


async def log(level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
    """Log a message with the given level on the root logger."""
    _auto_configure_root()
    await _logger_manager.root.log(level, msg, *args, **kwargs)


# Wall-clock budget of the automatic atexit drain; 0 disables it
# (records are then only counted and warned about). Mirrors the
# opt-out convention of comparable delivery SDKs.
_atexit_flush_timeout: float = 2.0


def set_atexit_flush(timeout: float) -> None:
    """
    Configure the automatic drain of undelivered records at
    interpreter exit.

    By default the atexit hook spends up to 2 seconds delivering
    whatever is still queued or buffered (see :func:`flush_sync`).
    Pass 0 to disable the drain and only warn to stderr.

    Note: atexit hooks do not run when the process dies from an
    unhandled signal, ``os._exit`` or SIGKILL — handle SIGTERM (e.g.
    via ``sys.exit``) for the drain to cover container shutdowns.

    Args:
        timeout: Drain budget in seconds; 0 disables the drain
    """
    global _atexit_flush_timeout
    if timeout < 0:
        raise ValueError("timeout must be >= 0")
    _atexit_flush_timeout = timeout


def _drain_at_exit() -> None:
    """Deliver (or at least count) undelivered records at exit."""
    undrained = _logger_manager.undelivered()
    if not undrained:
        return

    if _atexit_flush_timeout > 0:
        try:
            if _logger_manager.flush_sync(timeout=_atexit_flush_timeout):
                return
        except Exception:
            pass
        undrained = _logger_manager.undelivered()
        if not undrained:
            return

    sys.stderr.write(
        f"aiologging: {undrained} log record(s) were never delivered; "
        "call 'await aiologging.shutdown()' before exiting\n"
    )


atexit.register(_drain_at_exit)
