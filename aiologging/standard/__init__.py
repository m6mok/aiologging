"""Stdlib-backed asynchronous API for the durable pipeline.

The underlying loggers remain synchronous for dependencies. Async facades
share their actual levels, handlers, filters, factories and hierarchy.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging as _stdlib
import sys
import threading
import types
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from ..durable import (
    DeliveryError,
    DurableEngine,
    Recipient,
    RecipientUnavailableError,
    _IN_DELIVERY,
    _Worker,
)
from ..extensions.files import FileHandler, StreamHandler
from ..journal import (
    ExtraEncoder,
    JournalError,
    RecordCodec,
    default_journal_path,
    diagnose,
    prepared_text,
    bind_handler,
    handler_recipient,
)

__all__ = list(_stdlib.__all__) + [
    "RootLogger",
    "configureJournal",
    "flush",
    "flush_sync",
    "DurableEngine",
    "Recipient",
]
_engine: DurableEngine | None = None
_guard = threading.RLock()
# Starting this worker on import allows first-use initialization to stay off
# the application loop. It does not open files or start network resources.
_bootstrap = _Worker("aiologging-bootstrap", delivery_context=False)
_staged: contextvars.ContextVar[list[tuple[str, list[str]]] | None] = (
    contextvars.ContextVar("aiologging_staged", default=None)
)
_original_call_handlers = _stdlib.Logger.callHandlers
_original_handler_format = _stdlib.Handler.format
_facades: dict[int, tuple[_stdlib.Logger, AsyncLogger]] = {}
_handler_recipients: dict[int, tuple[_stdlib.Handler, Recipient]] = {}
_ephemeral_ids: dict[int, tuple[_stdlib.Handler, str]] = {}


class _DefaultLimit:
    """Distinguish an omitted limit from explicitly disabling its quota."""


_DEFAULT_LIMIT = _DefaultLimit()


def _application_name() -> str:
    command = str(Path(sys.argv[0] or "python").absolute())
    digest = hashlib.sha256(command.encode()).hexdigest()[:16]
    stem = Path(command).stem or "python"
    return stem + "-" + digest


def configureJournal(
    path: str | Path | None = None,
    *,
    application: str | None = None,
    quota_bytes: int | None | _DefaultLimit = _DEFAULT_LIMIT,
    max_database_bytes: int | None | _DefaultLimit = _DEFAULT_LIMIT,
    extra_encoder: ExtraEncoder | None = None,
    admission_capacity: int = 1024,
    exit_timeout: float = 10.0,
) -> DurableEngine:
    """Configure persistent acceptance once, before producers start.

    Processes of an application must use the same local journal and stable
    recipient IDs. The default path is under macOS Application Support,
    separated by the launching script's absolute path hash.
    """
    global _engine
    with _guard:
        if _engine is not None and not _engine.closed:
            raise JournalError(
                "Shutdown the current journal before replacing it"
            )
        destination = path or default_journal_path(
            application or _application_name()
        )
        _engine = DurableEngine(
            destination,
            quota_bytes=(
                64 * 1024 * 1024
                if isinstance(quota_bytes, _DefaultLimit)
                else quota_bytes
            ),
            max_database_bytes=(
                128 * 1024 * 1024
                if isinstance(max_database_bytes, _DefaultLimit)
                else max_database_bytes
            ),
            _adopt_quota=isinstance(quota_bytes, _DefaultLimit),
            _adopt_maximum=isinstance(max_database_bytes, _DefaultLimit),
            codec=RecordCodec(extra_encoder),
            admission_capacity=admission_capacity,
            exit_timeout=exit_timeout,
        )
        _handler_recipients.clear()
        _ephemeral_ids.clear()
        setattr(_stdlib.Logger, "callHandlers", _call_handlers)
        setattr(_stdlib.Handler, "format", _handler_format)
        _refresh()
        return _engine


def _ensure_engine() -> DurableEngine:
    with _guard:
        if _engine is None or _engine.closed:
            return configureJournal()
        return _engine


def _configured(handler: _stdlib.Handler) -> bool:
    if handler is _stdlib.lastResort or any(
        h is handler for h in _stdlib.root.handlers
    ):
        return True
    return any(
        any(h is handler for h in logger.handlers)
        for logger, _ in list(_facades.values())
    ) or any(
        isinstance(logger, _stdlib.Logger)
        and any(h is handler for h in logger.handlers)
        for logger in list(_stdlib.Logger.manager.loggerDict.values())
    )


def _refresh() -> None:
    """Recover configured destinations without waiting for a new log call."""
    engine = _engine
    if engine is None or engine.closed:
        return
    handlers = list(_stdlib.root.handlers)
    if _stdlib.lastResort is not None:
        handlers.append(_stdlib.lastResort)
    for logger in list(_stdlib.Logger.manager.loggerDict.values()):
        if isinstance(logger, _stdlib.Logger):
            handlers.extend(logger.handlers)
    for logger, _ in list(_facades.values()):
        handlers.extend(logger.handlers)
    seen: set[int] = set()
    for handler in handlers:
        if (
            id(handler) not in seen
            and type(handler).handle is not _stdlib.NullHandler.handle
        ):
            seen.add(id(handler))
            _recipient(handler, engine)


def _identity(handler: _stdlib.Handler) -> str:
    explicit = getattr(handler, "recipient_id", None)
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError("recipient_id must be a nonempty string")
        return explicit
    description: Any
    if isinstance(handler, _stdlib.FileHandler):
        description = [
            "file",
            type(handler).__module__,
            type(handler).__qualname__,
            handler.baseFilename,
            handler.mode,
            handler.encoding,
            handler.errors,
            handler.name,
        ]
    elif isinstance(handler, _stdlib.StreamHandler) and (
        handler.stream is sys.stdout or handler.stream is sys.stderr
    ):
        description = [
            "stream",
            "stdout" if handler.stream is sys.stdout else "stderr",
            handler.name,
        ]
    else:
        # Arbitrary custom targets cannot be inferred safely. A configured
        # name is insufficient: the same name may address another endpoint.
        if id(handler) in _ephemeral_ids:
            return _ephemeral_ids[id(handler)][1]
        identity = "ephemeral:" + uuid.uuid4().hex
        _ephemeral_ids[id(handler)] = (handler, identity)
        diagnose(
            f"{type(handler).__name__} has no recipient_id; "
            "recovery requires explicitly reassigning its retained tasks"
        )
        return identity
    digest = hashlib.sha256(json.dumps(description).encode()).hexdigest()
    return "stdlib:" + digest


def _recipient(handler: _stdlib.Handler, engine: DurableEngine) -> Recipient:
    with _guard:
        cached = _handler_recipients.get(id(handler))
        found = cached[1] if cached is not None else None
        identity = _identity(handler)
        if found is not None and found.identity == identity:
            return found
        strict: Any = getattr(handler, "deliver", None)
        prefiltered = type(handler).handle is _stdlib.Handler.handle

        async def async_deliver(records: Any) -> bool:
            if not _configured(handler) or _identity(handler) != identity:
                raise RecipientUnavailableError(
                    "Original handler is no longer configured"
                )
            return await strict(records) is True

        def deliver(records: Any) -> bool:
            if not _configured(handler) or _identity(handler) != identity:
                raise RecipientUnavailableError(
                    "Original handler is no longer configured"
                )
            if strict is not None:
                return strict(records) is True
            for record in records:
                # Standard/custom handlers keep standard error behaviour.
                # Their hidden emit errors are NOT proof of delivery.
                if prefiltered:
                    handler.acquire()
                    try:
                        handler.emit(record)
                    finally:
                        handler.release()
                else:
                    handler.handle(record)
            return True

        import inspect

        function = (
            async_deliver if inspect.iscoroutinefunction(strict) else deliver
        )
        recipient = Recipient(
            identity,
            function,
            fallback=getattr(handler, "fallback", None),
            primary_timeout=getattr(handler, "primary_timeout", 5.0),
            fallback_timeout=getattr(handler, "fallback_timeout", 5.0),
            retry_delay=getattr(handler, "retry_delay", 1.0),
            batch_size=getattr(handler, "batch_size", 1),
            close=getattr(handler, "aclose", None),
        )
        bind_handler(handler, identity)
        engine.register(recipient, replace=True)
        _handler_recipients[id(handler)] = (handler, recipient)
        return recipient


def _handler_format(
    handler: _stdlib.Handler, record: _stdlib.LogRecord
) -> str:
    frozen = prepared_text(record, handler_recipient(handler))
    return (
        frozen
        if frozen is not None
        else _original_handler_format(handler, record)
    )


def _call_handlers(logger: _stdlib.Logger, record: _stdlib.LogRecord) -> None:
    engine = _engine
    if engine is None or engine.closed:
        _original_call_handlers(logger, record)
        return
    if _IN_DELIVERY.get():
        # Recursion cannot synchronously wait for the worker emitting it.
        # Explicit stderr reporting makes this suppression visible.
        diagnose("recursive handler log was not accepted")
        raise DeliveryError("Logging recursively from a delivery callback")
    routes: dict[str, list[str]] = {}

    def _prepare(handler: _stdlib.Handler) -> None:
        if type(handler).handle is _stdlib.NullHandler.handle:
            return
        selected = record
        if type(handler).handle is _stdlib.Handler.handle:
            result: Any = handler.filter(record)
            if not result:
                return
            if isinstance(result, _stdlib.LogRecord):
                selected = result
        recipient = _recipient(handler, engine)
        can_prepare = (
            getattr(type(handler).emit, "_prepared_rendering", False)
            or (
                getattr(handler, "_format_at_acceptance", False)
                and getattr(
                    getattr(type(handler), "deliver", None),
                    "_prepared_rendering",
                    False,
                )
            )
            or (
                type(handler).format is _handler_format
                and type(handler).emit
                in (_stdlib.StreamHandler.emit, _stdlib.FileHandler.emit)
            )
        )
        encoded = (
            engine.codec.encode_formatted(
                selected, recipient.identity, handler.format(selected)
            )
            if can_prepare
            else engine.codec.encode(selected)
        )
        routes.setdefault(recipient.identity, []).append(encoded)

    def prepare(handler: _stdlib.Handler) -> None:
        try:
            _prepare(handler)
        except Exception as error:
            diagnose(f"record was not accepted: {type(error).__name__}")
            raise

    found = 0
    current: _stdlib.Logger | None = logger
    while current is not None:
        for handler in list(current.handlers):
            found += 1
            if record.levelno >= handler.level:
                prepare(handler)
        current = current.parent if current.propagate else None
    if found == 0:
        if _stdlib.lastResort is not None:
            if record.levelno >= _stdlib.lastResort.level:
                prepare(_stdlib.lastResort)
        elif (
            _stdlib.raiseExceptions
            and not logger.manager.emittedNoHandlerWarning
        ):
            sys.stderr.write(
                f'No handlers could be found for logger "{logger.name}"\n'
            )
            logger.manager.emittedNoHandlerWarning = True
    if not routes:
        return
    try:
        encoded = engine.codec.encode_routes(record, routes)
        recipients = list(routes)
        staged = _staged.get()
        if staged is not None:
            staged.append((encoded, recipients))
        else:
            engine.accept_sync(encoded, recipients)
    except Exception as error:
        diagnose(f"record was not accepted: {type(error).__name__}")
        raise


class _Facade:
    """Delegate configuration and creation to the actual stdlib object."""

    _target: _stdlib.Logger | _stdlib.LoggerAdapter[Any]

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._target, name)
        if isinstance(value, _stdlib.Logger):
            return _wrap(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_target":
            object.__setattr__(self, name, value)
        else:
            setattr(self._target, name, value)
            if name == "handlers":
                _refresh()

    def addHandler(self, handler: _stdlib.Handler) -> None:
        """Share the actual handler object and start recovery immediately."""
        target: Any = self._target
        target.addHandler(handler)
        _refresh()

    def removeHandler(self, handler: _stdlib.Handler) -> None:
        """Remove the destination without redirecting its unfinished tasks."""
        target: Any = self._target
        target.removeHandler(handler)

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> None:
        engine = _engine
        if engine is None or engine.closed:
            engine = await asyncio.wrap_future(
                _bootstrap.submit(_ensure_engine)
            )
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 2
        staged: list[tuple[str, list[str]]] = []
        token = _staged.set(staged)
        try:
            getattr(self._target, method)(*args, **kwargs)
        finally:
            _staged.reset(token)
        for encoded, recipients in staged:
            await engine.accept(encoded, recipients)

    async def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        await self._call("debug", msg, *args, **kwargs)

    async def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        await self._call("info", msg, *args, **kwargs)

    async def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        await self._call("warning", msg, *args, **kwargs)

    async def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        await self._call("warn", msg, *args, **kwargs)

    async def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        await self._call("error", msg, *args, **kwargs)

    async def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        await self._call("exception", msg, *args, **kwargs)

    async def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        await self._call("critical", msg, *args, **kwargs)

    async def fatal(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        await self._call("fatal", msg, *args, **kwargs)

    async def log(
        self, level: int, msg: Any, *args: Any, **kwargs: Any
    ) -> None:
        await self._call("log", level, msg, *args, **kwargs)


class AsyncLogger(_Facade):
    """Async view of a synchronous stdlib Logger, sharing its configuration."""

    _target: _stdlib.Logger

    def __init__(
        self, logger: _stdlib.Logger | str | int, level: int | None = None
    ) -> None:
        if isinstance(logger, _stdlib.Logger):
            self._target = logger
        elif "_target" not in self.__dict__:
            raise TypeError("An async view requires an underlying Logger")
        # Logger/RootLogger.__new__ already initialized a direct-constructor
        # view; Python invokes the returned subtype's __init__ once more.

    def getChild(self, suffix: str) -> AsyncLogger:
        return _wrap(self._target.getChild(suffix))

    def getChildren(self) -> set[AsyncLogger]:
        # The 3.12 API is also available on 3.11 through the stdlib hierarchy.
        target = self._target
        assert isinstance(target, _stdlib.Logger)
        if hasattr(target, "getChildren"):
            return {_wrap(child) for child in target.getChildren()}
        return {
            _wrap(child)
            for child in _stdlib.Logger.manager.loggerDict.values()
            if isinstance(child, _stdlib.Logger) and child.parent is target
        }

    async def handle(self, record: _stdlib.LogRecord) -> None:
        engine = _engine
        if engine is None or engine.closed:
            engine = await asyncio.wrap_future(
                _bootstrap.submit(_ensure_engine)
            )
        staged: list[tuple[str, list[str]]] = []
        token = _staged.set(staged)
        try:
            self._target.handle(record)
        finally:
            _staged.reset(token)
        for encoded, recipients in staged:
            await engine.accept(encoded, recipients)

    async def callHandlers(self, record: _stdlib.LogRecord) -> None:
        engine = _engine
        if engine is None or engine.closed:
            engine = await asyncio.wrap_future(
                _bootstrap.submit(_ensure_engine)
            )
        staged: list[tuple[str, list[str]]] = []
        token = _staged.set(staged)
        try:
            self._target.callHandlers(record)
        finally:
            _staged.reset(token)
        for encoded, recipients in staged:
            await engine.accept(encoded, recipients)


class LoggerAdapter(_Facade):
    """Stdlib LoggerAdapter processing with asynchronous logging methods."""

    _target: _stdlib.LoggerAdapter[Any]

    def __init__(
        self, logger: Any, extra: Mapping[str, Any], **kwargs: Any
    ) -> None:
        target = logger._target if isinstance(logger, _Facade) else logger
        self._target = _stdlib.LoggerAdapter(target, extra, **kwargs)
        # Preserve the documented subclass process hook.
        setattr(self._target, "process", self.process)

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        return _stdlib.LoggerAdapter.process(self._target, msg, kwargs)


# Custom logger classes and record factories remain real stdlib extensions.
class Logger(_stdlib.Logger):
    """Construct an async view; subclasses remain stdlib extension classes."""

    def __new__(cls, name: str, level: int = _stdlib.NOTSET) -> Logger:
        if cls is Logger:
            return cast(Logger, _wrap(_stdlib.Logger(name, level)))
        return super().__new__(cls)

    def __init__(self, name: str, level: int = _stdlib.NOTSET) -> None:
        if "_target" not in self.__dict__:
            super().__init__(name, level)


class RootLogger(_stdlib.RootLogger, Logger):
    """Construct a detached async root; preserve synchronous extensions."""

    def __new__(cls, level: int) -> RootLogger:
        if cls is RootLogger:
            return cast(RootLogger, _wrap(_stdlib.RootLogger(level)))
        return cast(RootLogger, super().__new__(cls, "root", level))

    def __init__(self, level: int) -> None:
        if "_target" not in self.__dict__:
            _stdlib.RootLogger.__init__(self, level)


Handler = _stdlib.Handler
Filter = _stdlib.Filter
Formatter = _stdlib.Formatter
BufferingFormatter = _stdlib.BufferingFormatter
LogRecord = _stdlib.LogRecord
Manager = _stdlib.Manager


def _wrap(logger: _stdlib.Logger) -> AsyncLogger:
    with _guard:
        cached = _facades.get(id(logger))
        if cached is None:
            # Preserve isinstance checks for user-selected stdlib classes,
            # while leaving the synchronous logger in its manager untouched.
            actual_class = type(logger)
            bases: tuple[type[Any], ...]
            if actual_class is _stdlib.Logger:
                bases = (AsyncLogger, Logger)
            elif actual_class is _stdlib.RootLogger:
                bases = (AsyncLogger, RootLogger)
            elif issubclass(actual_class, Logger):
                bases = (AsyncLogger, actual_class)
            else:
                bases = (AsyncLogger, actual_class, Logger)
            facade_type = type("Async" + actual_class.__name__, bases, {})
            facade = cast(AsyncLogger, facade_type(logger))
            _facades[id(logger)] = (logger, facade)
            return facade
        return cached[1]


def getLogger(name: str | None = None) -> AsyncLogger:
    return _wrap(_stdlib.getLogger(name))


def basicConfig(**kwargs: Any) -> None:
    """Use stdlib configuration, including force/filename/style/encoding."""
    _ensure_engine()
    if (
        not _stdlib.root.handlers or kwargs.get("force")
    ) and "handlers" not in kwargs:
        if "filename" in kwargs and "stream" in kwargs:
            raise ValueError(
                "stream and filename must not be specified together"
            )
        options = dict(kwargs)
        filename = options.pop("filename", None)
        mode = options.pop("filemode", "a")
        encoding = options.pop("encoding", None)
        errors = options.pop("errors", "backslashreplace")
        stream = options.pop("stream", None)
        handler = (
            FileHandler(filename, mode, encoding, errors=errors)
            if filename is not None
            else StreamHandler(stream)
        )
        options["handlers"] = [handler]
        try:
            _stdlib.basicConfig(**options)
        except BaseException:
            handler.close()
            raise
    else:
        _stdlib.basicConfig(**kwargs)
    _refresh()


async def flush(timeout: float | None = None) -> None:
    if _engine is not None and not _engine.closed:
        await _engine.flush(timeout)


def flush_sync(timeout: float = 10.0) -> bool:
    return _engine is None or _engine.closed or _engine.flush_sync(timeout)


async def shutdown(timeout: float | None = None) -> None:
    global _engine
    engine = _engine
    if engine is not None:
        await engine.shutdown(timeout)
        if engine.closed:
            with _guard:
                setattr(
                    _stdlib.Logger, "callHandlers", _original_call_handlers
                )
                setattr(_stdlib.Handler, "format", _original_handler_format)
                _engine = None
                _handler_recipients.clear()
    _stdlib.shutdown()


async def _root_log(method: str, *args: Any, **kwargs: Any) -> None:
    if _engine is None or _engine.closed:
        await asyncio.wrap_future(_bootstrap.submit(_ensure_engine))
    if not _stdlib.root.handlers:
        basicConfig()
    kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 2
    await getattr(getLogger(), method)(*args, **kwargs)


async def debug(msg: Any, *args: Any, **kwargs: Any) -> None:
    await _root_log("debug", msg, *args, **kwargs)


async def info(msg: Any, *args: Any, **kwargs: Any) -> None:
    await _root_log("info", msg, *args, **kwargs)


async def warning(msg: Any, *args: Any, **kwargs: Any) -> None:
    await _root_log("warning", msg, *args, **kwargs)


async def warn(msg: Any, *args: Any, **kwargs: Any) -> None:
    await _root_log("warn", msg, *args, **kwargs)


async def error(msg: Any, *args: Any, **kwargs: Any) -> None:
    await _root_log("error", msg, *args, **kwargs)


async def exception(msg: Any, *args: Any, **kwargs: Any) -> None:
    await _root_log("exception", msg, *args, **kwargs)


async def critical(msg: Any, *args: Any, **kwargs: Any) -> None:
    await _root_log("critical", msg, *args, **kwargs)


fatal = critical


async def log(level: int, msg: Any, *args: Any, **kwargs: Any) -> None:
    await _root_log("log", level, msg, *args, **kwargs)


def __getattr__(name: str) -> Any:
    return getattr(_stdlib, name)


root = getLogger()


class _LoggingModule(types.ModuleType):
    """Share documented mutable module settings with synchronous logging."""

    _settings = {
        "lastResort",
        "raiseExceptions",
        "logThreads",
        "logProcesses",
        "logMultiprocessing",
        "logAsyncioTasks",
    }

    def __getattribute__(self, name: str) -> Any:
        if name in _LoggingModule._settings:
            return getattr(_stdlib, name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _LoggingModule._settings:
            setattr(_stdlib, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _LoggingModule
