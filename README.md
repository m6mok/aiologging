# aiologging

Asynchronous logging for macOS and Python 3.11–3.15. Standard logging
configuration, factories, filters, formatters and synchronous dependencies
share one hierarchy. Awaiting a logging operation confirms durable SQLite
acceptance for eligible records; disabled/filtered calls keep standard
behaviour. Handler I/O runs independently of the application's event loop.

```python
import asyncio
import aiologging as logging

logging.configureJournal(application="my-application")  # optional
logging.basicConfig(filename="application.log", level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    await logger.info("Hello %s", "world")

asyncio.run(main())  # ordinary exit drains; shutdown is optional
```

## Guarantees and recovery

Accepted messages are stored in one persistent journal shared by processes
of the application. Each destination has its own confirmation state. Original
routes survive reconfiguration. Old tasks run before new tasks for the same
destination; an unavailable destination does not stop another destination.
Delivery is at least once for confirmed adapters: crash recovery may duplicate
a message. Network-side ordering across failures is not guaranteed.

The default journal lives in `~/Library/Application Support/aiologging/`, in
an application directory derived from the launching script's absolute path.
Set `application` or an explicit path for applications using multiple entry
points. Defaults reserve at most 64 MiB for unfinished snapshots/routes and
128 MiB for SQLite pages. Limits are configurable, shared across processes,
and never evict unfinished records. Implicit defaults adopt persisted limits
on restart; Journal.quota_bytes can adjust the shared logical limit live. Rollback files need additional disk space.
Acceptance errors raise and are reported on stderr. `extra` must contain JSON
values; custom objects require an explicit `extra_encoder`.

Files and streams require no external packages. HTTP/Telegram are optional:

```sh
pip install aiologging[httpx]  # or aiologging[aiohttp]
```

```python
from aiologging import FileHandler, HTTPHandler

backup = FileHandler("fallback.log")
network = HTTPHandler(
    "https://example.org/logs", fallback=backup.deliver,
    primary_timeout=5, fallback_timeout=5,
)
logging.basicConfig(handlers=[network], level=logging.INFO)
```

Fallback is mandatory for our network adapters. A successful fallback is
remembered once per task and retains the primary task for later recovery.
Sessions are created by a factory on the library loop, never supplied as
already-running application sessions. See the offline examples for mocks.

`logging` dependencies use the same handlers and wait for commit in their
calling thread. Arbitrary stdlib/custom handlers retain ordinary behaviour,
including hidden `handleError` failures: completion is **not universal proof
of delivery**. Our file/stream/network adapters and custom `deliver(records)`
methods returning exactly `True` use explicit confirmation. `Logger` subclasses and
`Handler` remain synchronous stdlib extension bases; direct Logger/RootLogger
construction produces an async view, while `getLogger` returns an
async view of the actual logger. Configuration and handler identity are
shared; custom synchronous logger methods remain usable through the view.

`aiologging.config` supports standard `dictConfig`/`fileConfig` and configuration
hooks. `aiologging.handlers` exposes the standard handler names, with confirmed
file rotation variants. Mutation of levels, filters, handlers, propagation,
formatters and module flags uses the real stdlib state. Custom handler targets
need a stable `recipient_id` for automatic cross-process recovery. A name alone
cannot safely identify an arbitrary target; unnamed targets have a diagnosed
process-local identity and retain their backlog for explicit reassignment.

`await logging.flush(timeout)` waits for primary confirmation and raises if
unfinished work remains. `await logging.shutdown(timeout)` is optional.
Neither exit hooks nor timeouts stop `SIGKILL`, `os._exit`, or arbitrary blocking
custom code. The journal provides restart recovery for accepted records.
Use multiprocessing **spawn**, the macOS default; engines/connections cannot
be inherited through fork.

## Development and migration

- [Architecture and recovery](docs/architecture.md)
- [Migration and compatibility boundaries](docs/migration.md)
- [Development and validation](docs/development.md)
- [Measured defaults](docs/durable-measurements.md)
- [Offline examples](examples/)

The previous queue-only API is retained in `aiologging.legacy`, and its
configuration helpers in `aiologging.legacy_config`. That API's dropping,
inline delivery, and loop-rescue behaviour does not provide durable acceptance.
Historical tests and stress scenarios continue to check it explicitly.
The package version has not been changed and no release has been created.

MIT licensed. Historical release notes below describe the previous API.

## Changelog

### 0.2.11

Telegram HTML safety and ConfigManager integration with the global
hierarchy (preparing the library for services replacing
`python-telegram-handler` + `dictConfig`):

- New `TelegramHtmlFormatter` (`aiologging.formatters`): HTML-escapes
  the message, logger name, `funcName`, the traceback (wrapped in
  `<pre>`) and stack info, while the format template keeps its own
  Telegram markup; template and `datefmt` are configurable, default
  template matches python-telegram-handler's `HtmlFormatter`
- `AsyncTelegramHandler` splitting/batching is entity-safe with
  `parse_mode="HTML"`: a message boundary never falls inside a tag
  or `&...;` entity, and open tags are closed at the end of a part
  and reopened (with attributes) in the next — an over-long record
  can no longer produce a 400 that loses the whole batch. Markdown
  modes keep the hard split (documented limitation)
- Fixed: `ConfigManager` built loggers outside the global hierarchy —
  no `parent`, no propagation to root, and global
  `flush()`/`shutdown()` never force-flushed their buffered handlers,
  so the undersized tail of a Telegram batch could be lost at
  shutdown. Config loggers now come from the global `getLogger()`
  with the configuration applied on top, stdlib-dictConfig-style
- `configure_from_dict` parity: a `formatters` section (`format`,
  `datefmt`, optional `class` dotted path) referenced by handlers
  via `formatter`; the `stream` handler honours
  `stdout`/`stderr`/`ext://sys.*` targets (the key was silently
  ignored before); unknown configuration keys now raise
  `ConfigurationError` instead of being silently ignored
- Removed `AsyncHttpProtoHandler` and the `protobuf` extra: the
  serialization was a JSON stub mislabelled as protobuf and
  `proto_message_class` was unused — fictional functionality is
  worse than absent functionality
- Fixed timed-rotation backup sorting for `when="S"`: the
  `YYYY-MM-DD_HH-MM-SS` suffix never parsed and always fell back to
  file mtime; rotating handlers coverage extended (timed schedule
  branches for `midnight`/`W0`–`W6`/`at_time`/`utc`, rotation with
  an unavailable `tell()`)

### 0.2.10

Two correctness fixes found by extending the stress harness (fuzz
seed sweep and the new `surface` scenario category):

- The stdlib bridge under `overflow="block"` degraded to `drop_old`
  on a full queue, so bridge traffic could evict records already
  accepted from async producers — and resolve their
  `delivery="await"` futures without delivery. The bridge now
  degrades to `drop_new`: the arriving bridged record is shed (with
  accounting) and queued records are never touched; the cold-start
  buffer picks its victim by the configured policy the same way
- `ConfigManager.get_logger` had a check-then-create race: threads
  requesting the same logger concurrently could each construct it,
  attaching duplicate handlers that would double-write every record;
  construction is now serialized with a lock
- New stress category `surface` (6 scenarios): content integrity
  (delivered text and `exc_info` tracebacks byte-checked), stream
  handlers sharing one buffer, retry exhaustion firing
  `error_handler` exactly once per record, secret redaction on the
  error path, and — in a child interpreter — the ConfigManager
  pipeline and the global `captureStdlib()` install path

### 0.2.9

Three delivery-guarantee fixes, all found by the new stress harness
(`stress/`, see `docs/stress-testing.md`):

- The atexit drain delivered zero file records: aiofiles needs
  executors, which are already shut down at interpreter exit —
  `AsyncFileHandler` now falls back to blocking file I/O there
- `flush_sync` after an event-loop death lost the in-flight records
  when the queues were full: rescuing a dead queue no longer wakes
  its dead-loop putters, teardown errors are no longer mistaken for
  handler failures, and the rescue never evicts records silently
- `force_flush()` (and therefore `flush()`, `flush_sync()` and the
  atexit drain) flushed at most one `max_batch_size` batch, leaving
  the rest buffered while reporting success — it now drains the
  buffers completely

### 0.2.8

- Silenced cosmetic stderr noise from consumer/worker coroutines
  GC-finalized after their event loop died (guarded `ContextVar`
  reset, unstarted coroutines closed, quiet exit when `queue.get()`
  cleanup touches a closed loop); no records were ever lost

### 0.2.7

- Opt-in `LevelAwareDrop` queue policy (`basicConfig(drop_policy=...)`
  or `manager.drop_policy`): above the `watermark` fill ratio
  arriving records below `discard_below` are discarded, and on a full
  queue the oldest expendable record is evicted — a DEBUG/INFO burst
  can no longer push an ERROR out of the queue; mirrors rsyslog
  `discardSeverity` / logback `discardingThreshold`, default
  behaviour unchanged

### 0.2.6

- Inline delivery of critical bridged records:
  `captureStdlib(inline_level=logging.ERROR)` delivers matching
  records synchronously — before queueing — so the pager message
  leaves even if the process dies on the next line
- `AsyncHandler.emit_sync(record, timeout)` opt-in hook (implemented
  by the Telegram handler with a deadline-joined helper thread)
- Bounded blocking: `inline_timeout` per record plus a token bucket
  (`inline_burst`, `inline_rate`); failures fall back to the queue
  path, a dedup marker prevents double delivery

### 0.2.5

- `flush_sync(timeout=5.0) -> bool` — synchronous emergency drain
  without a running loop (private loop, or
  `run_coroutine_threadsafe` when the manager's loop is alive in
  another thread); best-effort, never raises
- `flush(timeout=)` raises `asyncio.TimeoutError` on expiry;
  `shutdown(timeout=)` bounds the drain and always completes teardown
- Automatic atexit drain of undelivered records (2 s budget,
  opt-out via `set_atexit_flush(0)` / `basicConfig(atexit_flush=0)`)

### 0.2.4

- `configure_from_dict` accepts a dotted path to an `AsyncHandler`
  subclass in `"class"` (stdlib `dictConfig`-style), passing the
  remaining keys to the constructor
- Fixed `register_handler`: registered custom classes no longer fail
  with "Unsupported handler type"

### 0.2.3

- Per-handler dispatch queues, each drained by its own worker task
  (`QueueListener` style): a stuck sink only backs up its own queue,
  other handlers keep delivering
- Semantics preserved: per-handler ordering, `delivery="await"`
  resolves after the last handler, overflow policies apply per
  queue, workers survive event-loop changes

### 0.2.2

- `AsyncTelegramHandler`: batches records into `sendMessage` calls
  within the 4096-char limit, honours `retry_after` on 429, redacts
  the bot token in errors and `repr`
- `_retry_delay_from_response` hook on HTTP handlers lets the retry
  policy inspect the response

### 0.2.1

- httpx as an alternative HTTP backend alongside aiohttp
  (`backend="httpx"` or auto-detection)

### 0.2.0

**Breaking**: the logging pipeline is now queue-based and the API strictly follows the standard `logging` module.

- Records are enqueued at the call site and dispatched by a background consumer — `await logger.info(...)` no longer waits for handler I/O
- Configurable `delivery` ("enqueue"/"await") and `overflow` ("block"/"drop_new"/"drop_old") via `basicConfig` or per logger
- `await aiologging.flush()` / `await aiologging.shutdown()` lifecycle; the consumer starts lazily and survives event loop changes
- Stdlib bridge: `captureStdlib()` / `basicConfig(capture_stdlib=True)` routes standard `logging` records through async handlers
- Module-level convenience coroutines (`aiologging.info(...)`, ...) and `aiologging.disable(level)`
- API aligned with `logging.Logger`: `getEffectiveLevel`, `getChild`, `getChildren`, `hasHandlers`, `makeRecord`, `findCaller`, `stacklevel` support, `parent`/`propagate`/`filters` attributes; `getLogger()` without arguments returns the root logger
- Removed: `Logger.setFormatter`/`getFormatter` (formatters live on handlers), `getLevel` (use `getEffectiveLevel`), `disable()`/`enable()` methods (use the `disabled` attribute or `aiologging.disable`), `setParent`/`getParent`, `getRootLogger`, `getLoggerContext`, `log_async`
- `async with logger:` now flushes instead of closing the shared logger instance
- Handlers can be constructed outside a running event loop on Python 3.9

### 0.1.1

- Python 3.13 and 3.14 support
- Replace deprecated `asyncio.iscoroutinefunction` with `inspect.iscoroutinefunction`
- Remove dead Python 3.8 compatibility branches

### 0.1.0

- Initial release
- Full async logging API
- Stream, file, and HTTP handlers
- File rotation support
- Extensible authentication
- Optional dependencies
- Strict type checking
