"""Identical stdlib and durable scenarios, with synchronous dependencies."""

from __future__ import annotations

import asyncio
import copy
import io
import logging

import pytest

import aiologging as aio
import aiologging.config as config
from aiologging.journal import JournalQuotaError, RecordEncodingError
from aiologging.standard import _original_call_handlers


class Capture(logging.Handler):
    recipient_id = "capture"

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(copy.copy(record))


async def test_same_calls_fields_and_configuration_shared(configured):
    engine, logger = configured
    capture = Capture()
    logger.addHandler(capture)
    # pytest 9.1 also inserts capture handlers on non-propagating loggers.
    assert any(handler is capture for handler in logger.handlers)
    assert aio.getLogger(logger.name) is logger
    assert logging.getLogger(logger.name).handlers is logger.handlers
    await logger.info("value %s", {"a": 1}, extra={"key": [1, 2]})
    logging.getLogger(logger.name).warning("dependency")
    await aio.flush(3)
    assert [r.getMessage() for r in capture.records] == [
        "value {'a': 1}",
        "dependency",
    ]
    assert (
        capture.records[0].funcName
        == "test_same_calls_fields_and_configuration_shared"
    )
    assert capture.records[0].pathname == __file__
    assert capture.records[0].key == [1, 2]
    logger.setLevel("ERROR")
    await logger.info("disabled")
    logging.getLogger(logger.name).info("dependency disabled")
    await aio.flush(3)
    assert len(capture.records) == 2


async def test_factory_custom_logger_and_adapter_hooks(configured):
    engine, logger = configured
    capture = Capture()

    class Custom(logging.Logger):
        def makeRecord(self, *args, **kwargs):
            result = super().makeRecord(*args, **kwargs)
            result.from_custom = True
            return result

    factory = logging.getLogRecordFactory()

    def create(*args, **kwargs):
        result = factory(*args, **kwargs)
        result.from_factory = 42
        return result

    aio.setLogRecordFactory(create)
    aio.setLoggerClass(Custom)
    custom = aio.getLogger("durable-tests-custom")
    assert isinstance(custom, Custom)
    custom.handlers = [capture]
    custom.propagate = False
    custom.setLevel(logging.INFO)
    try:
        await custom.info("custom")

        class Adapter(aio.LoggerAdapter):
            def process(self, msg, kwargs):
                kwargs["extra"] = {"adapted": self.extra["key"]}
                return "prefix " + msg, kwargs

        adapter = Adapter(custom, {"key": "value"})
        await adapter.info("adapter")
        await aio.flush(3)
        assert capture.records[0].from_custom is True
        assert capture.records[0].from_factory == 42
        assert capture.records[1].getMessage() == "prefix adapter"
        assert capture.records[1].adapted == "value"
        assert (
            capture.records[1].funcName
            == "test_factory_custom_logger_and_adapter_hooks"
        )
    finally:
        custom.handlers = []


async def test_filters_mutation_replacement_duplicates_match_stdlib(
    configured,
):
    engine, logger = configured
    first = Capture()
    second = Capture()
    second.recipient_id = "capture-second"

    def mutate(record):
        record.msg += " first"
        return True

    def replacement(record):
        changed = copy.copy(record)
        changed.msg += " replaced"
        return changed

    first.addFilter(mutate)
    second.addFilter(replacement)
    logger.handlers = [first, second, first]
    await logger.info("original")
    await aio.flush(3)
    actual = [r.getMessage() for r in first.records], [
        r.getMessage() for r in second.records
    ]
    # Reference uses the same handler order and the running Python version's
    # actual filter semantics (replacement only exists in 3.12+).
    reference_first = Capture()
    reference_second = Capture()
    reference_first.addFilter(mutate)
    reference_second.addFilter(replacement)
    reference = logging.Logger("reference")
    reference.handlers = [reference_first, reference_second, reference_first]
    _original_call_handlers(
        reference,
        logging.LogRecord("reference", 20, "x", 1, "original", (), None),
    )
    expected = [r.getMessage() for r in reference_first.records], [
        r.getMessage() for r in reference_second.records
    ]
    assert actual == expected


async def test_dictconfig_fileconfig_and_incremental_reconfiguration(
    configured, tmp_path
):
    engine, logger = configured
    definition = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {"format": "{levelname}:{message}", "style": "{"}
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": str(tmp_path / "file"),
                "formatter": "plain",
            }
        },
        "loggers": {
            logger.name: {
                "level": "INFO",
                "handlers": ["file"],
                "propagate": False,
            }
        },
    }
    config.dictConfig(definition)
    await logger.info("dict")
    await aio.flush(3)
    config.dictConfig(
        {
            "version": 1,
            "incremental": True,
            "loggers": {logger.name: {"level": "ERROR"}},
        }
    )
    await logger.info("ignored")
    await logger.error("incremental")
    await aio.flush(3)
    assert (tmp_path / "file").read_text() == "INFO:dict\nERROR:incremental\n"
    text = f"""
[loggers]
keys=root,app
[handlers]
keys=file
[formatters]
keys=plain
[logger_root]
level=WARNING
handlers=
[logger_app]
level=INFO
handlers=file
qualname={logger.name}
propagate=0
[handler_file]
class=FileHandler
args=({str(tmp_path / 'ini')!r}, 'a')
formatter=plain
[formatter_plain]
format=%(message)s
"""
    config.fileConfig(io.StringIO(text), disable_existing_loggers=False)
    await logger.info("ini")
    await aio.flush(3)
    assert (tmp_path / "ini").read_text() == "ini\n"


async def test_basicconfig_module_functions_and_caller(configured, tmp_path):
    engine, logger = configured
    output = tmp_path / "output"
    aio.basicConfig(
        filename=output,
        encoding="utf-8",
        errors="strict",
        style="$",
        format="$message",
        level="INFO",
        force=True,
    )
    await aio.info("root")
    logging.info("sync dependency")
    await aio.flush(3)
    assert output.read_text() == "root\nsync dependency\n"
    capture = Capture()
    logging.root.handlers = [capture]
    await aio.warning("caller")
    await aio.flush(3)
    assert (
        capture.records[0].funcName
        == "test_basicconfig_module_functions_and_caller"
    )


async def test_acceptance_errors_propagate_from_sync_and_async(configured):
    engine, logger = configured
    logger.handlers = [Capture()]
    with pytest.raises(RecordEncodingError):
        await logger.info("unsupported", extra={"value": object()})
    with pytest.raises(RecordEncodingError):
        logging.getLogger(logger.name).info(
            "unsupported", extra={"value": object()}
        )
    engine.journal.quota_bytes = 1
    with pytest.raises(JournalQuotaError):
        await logger.info("quota")
    with pytest.raises(JournalQuotaError):
        logging.getLogger(logger.name).info("quota")
    assert engine.journal.record_ids() == []


async def test_mutable_settings_hierarchy_and_reconfiguration(configured):
    engine, logger = configured
    old = aio.raiseExceptions
    aio.raiseExceptions = False
    assert logging.raiseExceptions is False
    aio.raiseExceptions = old
    capture = Capture()
    logger.handlers = [capture]
    child = logger.getChild("child")
    child.handlers = []
    child.propagate = True
    child.setLevel(logging.NOTSET)
    assert child.parent is logger
    assert child in logger.getChildren()
    child.addFilter(lambda r: r.getMessage() != "filtered")
    await child.info("filtered")
    await child.info("accepted")
    await aio.flush(3)
    assert [r.getMessage() for r in capture.records] == ["accepted"]
    child.filters = []
    logger.removeHandler(capture)
    await child.info("last resort below threshold")
    assert await engine.pending() == {}


async def test_all_documented_names_available(configured):
    import aiologging.handlers as handlers
    import logging.handlers

    for name in logging.__all__:
        assert hasattr(aio, name), name
    for name in (
        "SocketHandler",
        "DatagramHandler",
        "SysLogHandler",
        "SMTPHandler",
        "HTTPHandler",
        "QueueHandler",
        "QueueListener",
        "MemoryHandler",
        "WatchedFileHandler",
        "RotatingFileHandler",
        "TimedRotatingFileHandler",
    ):
        assert hasattr(handlers, name)


async def test_restart_recovers_configured_handler_without_new_log_call(
    configured, tmp_path
):
    engine, logger = configured
    from aiologging.journal import Journal, RecordCodec
    from aiologging.standard import _identity

    handler = aio.FileHandler(tmp_path / "recovered")
    record = logging.LogRecord(
        logger.name, 20, "x", 1, "old accepted", (), None
    )
    with Journal(engine.journal.path) as journal:
        record_id = journal.append(
            RecordCodec().encode(record), [_identity(handler)]
        )
    # Adding the original destination starts its sender, even if no producer
    # ever emits a new message during this process lifetime.
    logger.addHandler(handler)
    await aio.flush(3)
    assert record_id not in engine.journal.record_ids()
    assert (tmp_path / "recovered").read_text() == "old accepted\n"


async def test_changed_destination_does_not_receive_original_backlog(
    configured, tmp_path
):
    engine, logger = configured
    from aiologging.journal import Journal, RecordCodec
    from aiologging.standard import _identity

    original = aio.FileHandler(tmp_path / "old")
    record = logging.LogRecord(
        logger.name, 20, "x", 1, "original route", (), None
    )
    with Journal(engine.journal.path) as journal:
        record_id = journal.append(
            RecordCodec().encode(record), [_identity(original)]
        )
    replacement = aio.FileHandler(tmp_path / "new")
    logger.handlers = [replacement]
    await logger.info("new route")
    with pytest.raises(aio.DeliveryError):
        await aio.flush(0.05)
    assert record_id in engine.journal.record_ids()
    assert (tmp_path / "new").read_text() == "new route\n"
    logger.addHandler(original)
    await aio.flush(3)
    assert (tmp_path / "old").read_text() == "original route\n"


async def test_direct_logger_root_and_extension_base_construction(configured):
    engine, _ = configured
    direct = aio.Logger("detached", logging.INFO)
    capture = Capture()
    direct.addHandler(capture)
    await direct.info("detached")
    detached_root = aio.RootLogger(logging.INFO)
    second = Capture()
    second.recipient_id = "detached-root"
    detached_root.handlers = [second]
    await detached_root.info("root detached")
    assert isinstance(direct, aio.Logger)
    assert isinstance(detached_root, aio.RootLogger)
    assert isinstance(aio.getLogger(), aio.RootLogger)

    class Custom(aio.Logger):
        def makeRecord(self, *args, **kwargs):
            record = super().makeRecord(*args, **kwargs)
            record.custom_base = True
            return record

    aio.setLoggerClass(Custom)
    custom = aio.getLogger("own-custom-base")
    custom.handlers = [capture]
    custom.setLevel(logging.INFO)
    custom.propagate = False
    try:
        await custom.info("custom base")
        await aio.flush(3)
        assert [r.getMessage() for r in capture.records] == [
            "detached",
            "custom base",
        ]
        assert capture.records[-1].custom_base is True
        assert [r.getMessage() for r in second.records] == ["root detached"]
        assert isinstance(custom, Custom)
    finally:
        custom.handlers = []
        direct.handlers = []
        detached_root.handlers = []


async def test_first_async_call_initializes_disk_off_application_loop(
    tmp_path, monkeypatch
):
    import aiologging.standard as implementation

    monkeypatch.setattr(
        implementation,
        "default_journal_path",
        lambda name: tmp_path / "journal",
    )
    logger = aio.getLogger("first-use")
    logger.handlers = [Capture()]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        await logger.info("first use")
        await aio.flush(3)
        assert (tmp_path / "journal").exists()
    finally:
        await aio.shutdown(1)
        logger.handlers = []


async def test_recursive_dependency_call_is_rejected_not_falsely_accepted(
    configured,
):
    engine, logger = configured
    errors = []

    class Confirmed(logging.Handler):
        recipient_id = "recursion-contract"

        def deliver(self, records):
            try:
                logging.getLogger(logger.name).info("recursive")
            except aio.DeliveryError as error:
                errors.append(error)
            return True

    logger.handlers = [Confirmed()]
    await logger.info("original")
    await aio.flush(3)
    assert len(errors) == 1
    assert await engine.pending() == {}


async def test_implicit_defaults_adopt_persisted_quota_after_restart(tmp_path):
    path = tmp_path / "journal"
    engine = aio.configureJournal(path)
    engine.journal.quota_bytes = 8192
    await aio.shutdown(1)
    restarted = aio.configureJournal(path)
    try:
        assert restarted.journal.quota_bytes == 8192
    finally:
        await aio.shutdown(1)


async def test_distinct_file_handlers_same_path_keep_their_formatter(
    configured, tmp_path
):
    _, logger = configured
    path = tmp_path / "same-output"
    first = aio.FileHandler(path)
    second = aio.FileHandler(path)
    first.setFormatter(aio.Formatter("first:%(message)s"))
    second.setFormatter(aio.Formatter("second:%(message)s"))
    logger.handlers = [first, second]
    await logger.info("original")
    await aio.flush(3)
    assert path.read_text() == "first:original\nsecond:original\n"
    assert logger.handlers[0] is first and logger.handlers[1] is second


async def test_formatted_output_is_frozen_before_commit_wait(
    configured, tmp_path
):
    engine, logger = configured
    import threading

    path = tmp_path / "output"
    handler = aio.FileHandler(path)
    handler.setFormatter(aio.Formatter("old:%(message)s"))
    logger.handlers = [handler]
    started = threading.Event()
    release = threading.Event()

    def block():
        started.set()
        assert release.wait(3)

    engine._journal_worker.submit(block)
    task = asyncio.create_task(logger.info("first"))
    try:
        await asyncio.sleep(0)
        assert started.wait(1)
        handler.setFormatter(aio.Formatter("new:%(message)s"))
        release.set()
        await task
        await logger.info("second")
        await aio.flush(3)
        assert path.read_text() == "old:first\nnew:second\n"
    finally:
        release.set()


async def test_unhashable_logger_handler_and_record_extensions(
    configured, tmp_path
):
    _, _ = configured

    class UnhashableLogger(logging.Logger):
        __hash__ = None

    class UnhashableHandler(Capture):
        __hash__ = None
        recipient_id = "unhashable"

        def createLock(self):
            # Stdlib's default fork-lock WeakSet itself requires hashing.
            # This thread-safe capture has the documented custom lock hook.
            self.lock = None

    class UnhashableRecord(logging.LogRecord):
        __hash__ = None

    aio.setLoggerClass(UnhashableLogger)
    aio.setLogRecordFactory(UnhashableRecord)
    logger = aio.getLogger("unhashable-logger")
    handler = UnhashableHandler()
    file = aio.FileHandler(tmp_path / "unhashable")
    logger.handlers = [handler, file]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        await logger.info("works")
        await aio.flush(3)
        assert handler.records[0].getMessage() == "works"
        assert (tmp_path / "unhashable").read_text() == "works\n"
        assert aio.getLogger(logger.name) is logger
    finally:
        logger.handlers = []


async def test_nullhandler_remains_an_intentional_noop(configured):
    engine, logger = configured
    logger.handlers = [aio.NullHandler()]
    await logger.info(
        "intentionally disabled", extra={"not_serializable": object()}
    )
    assert await engine.pending() == {}
    assert engine.journal.record_ids() == []
