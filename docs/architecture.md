# Architecture

aiologging mirrors the standard `logging` module — same names,
signatures, hierarchy and semantics — with the logging methods being
coroutines. The core design decision: **record creation is synchronous
at the call site, handler I/O is asynchronous in a background task**.

## The record path

```
await logger.info("msg %s", x)
  │ sync, at the call site:
  │   level check → rate limit → findCaller → makeRecord
  │   (so caller info, %-formatting, exc_info behave exactly
  │    like stdlib logging)
  ▼
logger.handle(record)
  │   logger filters → _freeze_record → manager.enqueue
  ▼
asyncio.Queue (bounded, default 10 000)
  │   overflow policy: "block" | "drop_new" | "drop_old"
  ▼
consumer task ("aiologging-consumer", one per manager)
  │   logger.callHandlers(record): walks the hierarchy while
  │   propagate is set; handler errors go to stderr and never
  │   interrupt dispatch
  ▼
async handlers (stream / file / rotating / HTTP / Telegram)
```

Key pieces, all in `aiologging/logger.py`:

- **`_freeze_record`** — before queueing, the message is rendered
  eagerly (`record.getMessage()`) and `exc_info` is converted to
  `exc_text`, so later mutation of caller args can't change the
  message and traceback frames aren't kept alive in the queue.
  Mirrors `QueueHandler.prepare`.
- **Delivery modes** (`DeliveryMode`): `"enqueue"` (default) — the
  await resolves once the record is queued; `"await"` — a future is
  attached to the queue item and resolved by the consumer after
  handlers processed the record. Selectable per call, per logger, or
  manager-wide.
- **Overflow policies** (`OverflowPolicy`): `"block"` awaits free
  space, `"drop_new"` discards the incoming record, `"drop_old"`
  discards the oldest queued one. Drops are counted in metrics.
- **Lazy, self-healing consumer** — `_ensure_consumer()` starts the
  task on the first record and rebuilds it when the running event
  loop changes (e.g. one loop per test) or the task died. Records
  stuck in the old queue — including the in-flight one — are rescued
  into `_pending` and redelivered.
- **`_IN_CONSUMER` ContextVar** — set inside the consumer task so the
  stdlib bridge can drop records emitted *by handler I/O libraries
  themselves* (e.g. aiohttp logging during an HTTP flush), preventing
  infinite feedback loops.
- **`AsyncLoggerManager`** — owns the hierarchy (`getLogger` with
  dotted names, root at WARNING, eager intermediate loggers), the
  queue and the consumer. The global instance is
  `logger._logger_manager`. `shutdown()` drains the queue, cancels
  the consumer, closes every handler and resets the hierarchy to a
  pristine state; `flush()` is `queue.join()` plus `force_flush()`
  on buffered handlers.
- **atexit hook** warns to stderr if records were never delivered
  (user forgot `await aiologging.shutdown()`).

## Stdlib bridge (`aiologging/bridge.py`)

`StdlibBridgeHandler` is a *synchronous* `logging.Handler` attached to
the stdlib root by `captureStdlib()`. It forwards third-party records
into the aiologging queue via `manager.enqueue_from_thread`, which is
thread-safe: records from a foreign thread hop into the consumer loop
with `call_soon_threadsafe`; records emitted before any loop exists
are buffered in `_pending` until the consumer starts. Because a sync
producer cannot await, the `"block"` policy degrades to `"drop_old"`
for bridged records.

## Handler class hierarchy (`aiologging/handlers/`)

```
AsyncHandlerABC (types.py)
└── AsyncHandler (base.py)
    │     level/filters/formatter, retry with optional
    │     retry_strategy, rate limiting, metrics, close()
    ├── AsyncStreamHandler (stream.py)
    │   └── AsyncStandardStreamHandler
    ├── AsyncFileHandler (file.py, requires aiofiles)
    │   └── AsyncFileHandlerWithRotation
    │       ├── AsyncRotatingFileHandler (rotating.py, size)
    │       └── AsyncTimedRotatingFileHandler (time)
    └── BufferedAsyncHandler (base.py)
        │     buffer + periodic auto-flush task, priority buffer,
        │     adaptive sizing; subclasses implement flush(records)
        └── AsyncHttpHandlerBase (http.py)
            │     dual backend (aiohttp preferred / httpx fallback,
            │     _HttpxResponseAdapter unifies responses), lazy
            │     session via LazyLock, authenticator protocol,
            │     batch retries with exponential backoff
            ├── AsyncHttpTextHandler
            ├── AsyncHttpJsonHandler
            ├── AsyncHttpProtoHandler (requires protobuf)
            ├── AsyncHttpHandler (auto-detects format)
            └── AsyncTelegramHandler (telegram.py)
                  Bot API sendMessage, 4096-char chunking,
                  429 retry_after, token redaction
```

Extension hooks on `AsyncHttpHandlerBase` (added in 0.2.2):

- `_retry_delay_from_response(status, error_text) -> Optional[float]`
  — lets the retry policy see the response. Returning a delay forces
  a retry after that delay even for 4xx (used by Telegram for 429
  with `parameters.retry_after`); returning None keeps the default
  policy (exponential backoff for 5xx, no retry for 4xx).
- `_redacted_url()` — the URL as shown in error messages and repr;
  override when the URL embeds a secret (Telegram hides the bot
  token as `/bot***/`).

## Supporting modules

- `aiologging/types.py` — protocols (`FormatterProtocol`,
  `FilterProtocol`, `AuthenticatorProtocol`), type aliases
  (`HttpBackendType`, `LogLevel`, …) and config containers
  (`BatchConfig`, `HttpConfig`, `FileConfig`, `RotationConfig`,
  `HandlerConfig`, `LoggerConfig`).
- `aiologging/exceptions.py` — hierarchy under `AiologgingError`;
  every exception carries a structured `details` dict plus keyword
  context set as attributes (e.g. `NetworkError(url=…,
  status_code=…)`).
- `aiologging/config.py` — `ConfigManager` builds loggers/handlers
  from dicts, JSON files or env vars; string registry maps `"stream"`,
  `"file"`, `"http"`, `"http_json"`, `"telegram"`, `"rotating_file"`,
  `"timed_rotating_file"` to classes.
- `aiologging/utils.py` — `LazyLock` (an `asyncio.Lock` created on
  first use, because on Python 3.9 `asyncio.Lock()` binds to the
  current loop at construction), `handle_error_with_fallback`,
  `ErrorContext`.
- `aiologging/__init__.py` — public API, `basicConfig`, the
  `create_*_handler` factories, `__version__`.

## Artifacts — never edit

`build/` (contains stale copies of the package under `build/lib/`),
`dist/`, `htmlcov/`, `*.egg-info/`, `.venv*`, `.mypy_cache*`,
`.pytest_cache*`.
