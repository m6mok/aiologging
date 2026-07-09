"""
Async logger implementation for aiologging.

The public API mirrors the standard :mod:`logging` module: attribute and
method names, signatures and semantics follow ``logging.Logger``, with
the logging methods being coroutines.

Records are created synchronously at the call site (so caller
information, ``%``-formatting and ``exc_info`` behave exactly like in
standard logging) and put on a bounded queue. A background consumer
task drains the queue and dispatches records to the async handlers, so
the awaiting coroutine never pays for handler I/O.

The consumer starts lazily on the first logged record and is restarted
transparently if the running event loop changes (e.g. one loop per
test). Call ``await aiologging.shutdown()`` before the application
exits to drain the queue and close handlers.
"""

from __future__ import annotations

import asyncio
import atexit
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

# Set inside the consumer task so the stdlib bridge can detect (and
# drop) records emitted by the handlers themselves — otherwise a
# handler whose I/O library logs via stdlib logging would feed its own
# records back into the queue forever.
_IN_CONSUMER: ContextVar[bool] = ContextVar(
    "aiologging_in_consumer", default=False
)

# Used to render exc_info into exc_text when a record is frozen for the
# queue; any stdlib-compatible formatter picks exc_text up later.
_exception_formatter = logging.Formatter()


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

    async def callHandlers(self, record: LogRecord) -> None:
        """
        Pass a record to all relevant handlers, walking the hierarchy
        while ``propagate`` is set (like stdlib ``callHandlers``).

        Runs inside the consumer task. Handler errors are reported to
        stderr and never interrupt dispatch.
        """
        logger: Optional[AsyncLogger] = self
        found = 0
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
                    try:
                        await handler.handle(record)
                    except Exception as e:
                        sys.stderr.write(
                            f"Error in handler {type(handler).__name__}: "
                            f"{e}\n"
                            f"Record: {record.getMessage()}\n"
                        )
            logger = logger.parent if logger.propagate else None

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


class AsyncLoggerManager:
    """
    Manager owning the logger hierarchy and the record queue.

    Mirrors ``logging.Manager`` for the hierarchy part (``getLogger``
    with dotted names, a ``root`` logger at WARNING) and additionally
    owns the machinery that makes logging asynchronous: a bounded
    ``asyncio.Queue`` plus a single background consumer task that
    dispatches records to handlers.

    The consumer starts lazily on the first enqueued record and is
    rebuilt automatically when the running event loop changes.
    """

    def __init__(
        self,
        queue_size: int = 10_000,
        overflow: OverflowPolicy = "block",
        delivery: DeliveryMode = "enqueue",
    ) -> None:
        self.loggerDict: Dict[str, AsyncLogger] = {}
        self.disable_level: int = NOTSET
        self.queue_size = queue_size
        self.overflow: OverflowPolicy = overflow
        self.delivery: DeliveryMode = delivery
        self.root = AsyncLogger("root", level=WARNING, manager=self)

        self._queue: Optional[asyncio.Queue[_QueueItem]] = None
        self._consumer_task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Records that arrived while no event loop was available (e.g.
        # from the stdlib bridge before the app started); drained into
        # the queue when the consumer starts
        self._pending: Deque[Tuple[AsyncLogger, LogRecord]] = deque(
            maxlen=queue_size
        )
        # The record the consumer is currently dispatching; rescued
        # together with the queue when the event loop changes
        self._in_flight: Optional[Tuple[AsyncLogger, LogRecord]] = None
        self._records_dropped = 0

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
            # No loop anywhere yet — buffer until the consumer starts
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
        if self._queue is not None:
            if self._in_flight is not None:
                self._pending.appendleft(self._in_flight)
                self._in_flight = None
            try:
                while True:
                    old_logger, old_record, _ = self._queue.get_nowait()
                    self._pending.append((old_logger, old_record))
            except asyncio.QueueEmpty:
                pass

        self._loop = loop
        self._queue = asyncio.Queue(maxsize=self.queue_size)

        while self._pending:
            pending_logger, pending_record = self._pending.popleft()
            try:
                self._queue.put_nowait((pending_logger, pending_record, None))
            except asyncio.QueueFull:  # pragma: no cover - same maxlen
                self._drop(pending_logger)

        self._consumer_task = loop.create_task(
            self._consume(self._queue), name="aiologging-consumer"
        )

    async def _consume(self, queue: asyncio.Queue[_QueueItem]) -> None:
        """Drain the queue, dispatching records to handlers."""
        token = _IN_CONSUMER.set(True)
        try:
            while True:
                logger, record, fut = await queue.get()
                self._in_flight = (logger, record)
                interrupted = False
                try:
                    await logger.callHandlers(record)
                    logger._metrics.increment_processed()
                except asyncio.CancelledError:
                    # Dispatch interrupted (loop teardown): leave the
                    # record in _in_flight so a rebuilt consumer can
                    # rescue and redeliver it
                    interrupted = True
                    raise
                except Exception as e:
                    logger._metrics.increment_errors()
                    sys.stderr.write(
                        f"Error in logger {logger.name}: {e}\n"
                    )
                finally:
                    if not interrupted:
                        self._in_flight = None
                    if fut is not None and not fut.done():
                        fut.set_result(None)
                    queue.task_done()
        finally:
            _IN_CONSUMER.reset(token)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """
        Wait until every queued record has been handled, then force
        buffered handlers to flush.
        """
        if self._pending:
            self._ensure_consumer()
        if self._queue is None:
            return
        self._ensure_consumer()
        queue = self._queue
        if queue is not None:
            await queue.join()

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

    async def shutdown(self) -> None:
        """
        Drain the queue, stop the consumer, close every handler and
        reset the hierarchy to a pristine state.

        Loggers obtained before shutdown are closed; ``getLogger``
        returns fresh instances afterwards.
        """
        try:
            await self.flush()
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
        self._consumer_task = None
        self._queue = None
        self._loop = None
        self._pending.clear()
        self._in_flight = None

        for logger in [self.root, *self.loggerDict.values()]:
            try:
                await logger.close()
            except Exception as e:
                sys.stderr.write(
                    f"Error closing logger {logger.name}: {e}\n"
                )

        self.loggerDict.clear()
        self.root = AsyncLogger("root", level=WARNING, manager=self)

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


async def flush() -> None:
    """Wait until every queued record has been handled."""
    await _logger_manager.flush()


async def shutdown() -> None:
    """
    Drain the queue, close all handlers and reset the hierarchy.

    Call this once before the application exits, while the event loop
    is still running.
    """
    await _logger_manager.shutdown()


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


def _warn_undrained_at_exit() -> None:
    """Warn on interpreter exit if records were never flushed."""
    queue = _logger_manager._queue
    undrained = (queue.qsize() if queue is not None else 0) + len(
        _logger_manager._pending
    )
    if undrained:
        sys.stderr.write(
            f"aiologging: {undrained} log record(s) were never delivered; "
            "call 'await aiologging.shutdown()' before exiting\n"
        )


atexit.register(_warn_undrained_at_exit)
