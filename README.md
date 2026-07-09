# aiologging

[![CI](https://github.com/m6mok/aiologging/actions/workflows/ci.yml/badge.svg)](https://github.com/m6mok/aiologging/actions/workflows/ci.yml)

Asynchronous logging library for Python (3.9–3.14). The API mirrors the standard `logging` module — same method names, signatures, hierarchy and semantics — with the logging methods being coroutines. Records are created at the call site and put on a queue; a background consumer fans them out to per-handler workers that perform the I/O, so `await logger.info(...)` never waits for a file write or an HTTP request.

## Features

- **logging-compatible API**: same methods, levels, hierarchy, filters and `LogRecord` semantics as the standard `logging` module
- **Background consumer**: handler I/O happens off the calling coroutine's path in per-handler workers, so one slow handler never delays the others; everything starts lazily and survives event loop changes
- **Configurable delivery**: `await` resolves on enqueue (default) or after handlers processed the record (`delivery="await"`)
- **Configurable backpressure**: bounded queue with `block` (default), `drop_new` or `drop_old` overflow policies
- **Stdlib bridge**: `captureStdlib()` routes third-party library logs (aiohttp, sqlalchemy, ...) through the same async handlers
- **Async Handlers**: non-blocking I/O for streams, files (with size/time rotation), HTTP endpoints with extensible authentication, and Telegram chats
- **Buffered Handlers**: batch processing for high-volume logging
- **Performance Metrics**: built-in metrics for loggers, handlers and the queue
- **Strict Type Checking**: full mypy support with type hints

## Installation

### Basic Installation

```bash
pip install aiologging
```

### With Optional Dependencies

```bash
# For file handlers
pip install aiologging[aiofiles]

# For HTTP handlers (pick one backend)
pip install aiologging[aiohttp]
pip install aiologging[httpx]

# For Protobuf support
pip install aiologging[protobuf]

# All dependencies
pip install aiologging[all]

# Development dependencies
pip install aiologging[dev]
```

## Quick Start

```python
import asyncio
import aiologging

async def main():
    aiologging.basicConfig(level=aiologging.INFO)

    logger = aiologging.getLogger("app")
    await logger.info("Application started")
    await logger.warning("Something might be wrong")
    await logger.error("An error occurred: %s", "details")

    # once, before the loop goes away: drain the queue, close handlers
    await aiologging.shutdown()

asyncio.run(main())
```

Module-level convenience functions work like in standard logging:

```python
await aiologging.warning("Logged on the root logger")
```

### How it works

`await logger.info(...)` synchronously creates the `LogRecord` (so caller info, `%`-formatting and `exc_info` behave exactly like standard logging) and puts it on a bounded queue. A background task drains the queue and fans each record out to per-handler dispatch queues, each drained by its own worker — so a slow handler (say, an HTTP sink honouring a long retry-after) delays only its own queue, never the other handlers. Per-handler ordering is preserved. By default the `await` resolves as soon as the record is enqueued — logging is nearly free for the calling coroutine.

Lifecycle:

- the consumer and the handler workers start lazily on the first logged record and are rebuilt transparently if the event loop changes (e.g. one loop per test) — including handler resources such as HTTP sessions and open files;
- `await aiologging.flush(timeout=...)` waits until everything queued has been handled (optionally bounded; raises `asyncio.TimeoutError` on expiry);
- `await aiologging.shutdown(timeout=...)` drains the queue and closes all handlers — call it once at application exit; `timeout` bounds the drain phase to fit e.g. a container's termination grace period;
- `aiologging.flush_sync(timeout=5.0)` is the synchronous emergency drain for contexts without a running loop (a `finally` after `asyncio.run`, plain sync code): it delivers queued *and buffered* records on a private event loop and returns `False` instead of raising when the budget expires;
- at interpreter exit an automatic drain spends up to 2 seconds delivering whatever is left, then warns on stderr about anything undelivered. Configure with `basicConfig(atexit_flush=...)` or `aiologging.set_atexit_flush()`; `0` disables the drain. atexit does not run when the process dies from an unhandled signal — handle SIGTERM (e.g. `sys.exit`) for this to cover container shutdowns.

### Delivery and backpressure

```python
aiologging.basicConfig(
    level=aiologging.INFO,
    queue_size=10_000,     # queue capacity
    overflow="block",      # "block" | "drop_new" | "drop_old"
    delivery="enqueue",    # "enqueue" | "await"
)

# per-logger override for critical logs: the await resolves only
# after the handlers have processed the record
audit = aiologging.getLogger("app.audit")
audit.delivery = "await"
```

- `overflow="block"` (default): when the queue is full, `await logger.info(...)` waits for free space — no records are lost.
- `overflow="drop_new"` / `"drop_old"`: the call never waits; the incoming or the oldest record is discarded and counted in metrics.

### Capturing stdlib logging

Third-party libraries log via the standard `logging` module. Route their records through your async handlers:

```python
aiologging.basicConfig(level=aiologging.INFO, capture_stdlib=True)
# or:
aiologging.captureStdlib()
```

Bridged records are routed through the aiologging hierarchy under the same logger name, so per-name handlers and propagation work as usual. The bridge is thread-safe and buffers records emitted before the event loop starts.

For alerting channels the bridge can also deliver critical records **inline** — synchronously, before they are queued — so the pager message leaves even when the record is emitted right before the process dies (an `except` block before a `raise`, a SIGTERM handler):

```python
aiologging.captureStdlib(inline_level=logging.ERROR)
# or: aiologging.basicConfig(capture_stdlib=True, inline_level=logging.ERROR)
```

Records at or above `inline_level` are offered to handlers that support synchronous delivery (`emit_sync`, currently `AsyncTelegramHandler`, which POSTs via `urllib` in a helper thread joined with the deadline). The inline path is bounded — `inline_timeout` (default 3 s) per record, plus a token bucket of `inline_burst` sends (default 1) refilling at `inline_rate` per minute (default 2) — so a burst of errors cannot stall the calling thread. On any failure the record simply stays on the normal queue path; a handler that received it inline is skipped by its queue worker, other handlers deliver as usual. Note the FIFO trade-off: an inline record reaches the chat ahead of older queued records — include `%(asctime)s` in the formatter so readers can reorder.

## Examples

Complete runnable examples live in the [examples/](examples/) directory:

- [examples/basic_usage.py](examples/basic_usage.py) — levels, `basicConfig`, `exc_info`, delivery modes, `flush`/`shutdown`
- [examples/stdlib_capture.py](examples/stdlib_capture.py) — routing third-party (stdlib `logging`) records through async handlers, including from threads and before the loop starts
- [examples/file_logging.py](examples/file_logging.py) — file handler, size- and time-based rotation
- [examples/http_logging.py](examples/http_logging.py) — JSON batches over HTTP with a custom authenticator (includes a local test collector)
- [examples/telegram_logging.py](examples/telegram_logging.py) — sending records to a Telegram chat, including rate-limit handling (includes a local Bot API stand-in)
- [examples/config_usage.py](examples/config_usage.py) — configuring loggers from a dictionary
- [examples/emergency_flush.py](examples/emergency_flush.py) — delivery at process death: `shutdown(timeout=...)`, `flush_sync`, the automatic atexit drain
- [examples/inline_alerts.py](examples/inline_alerts.py) — inline (synchronous) Telegram delivery of critical bridged records via `captureStdlib(inline_level=...)`

## Handlers

### Stream Handler

```python
import sys
import aiologging

async def main():
    logger = aiologging.getLogger("app")
    logger.addHandler(aiologging.AsyncStreamHandler(sys.stdout))

    await logger.info("This goes to stdout")
    await aiologging.shutdown()
```

### File Handler (requires aiofiles)

```python
import aiologging

async def main():
    logger = aiologging.getLogger("app")
    logger.addHandler(aiologging.AsyncFileHandler("app.log"))

    await logger.info("This goes to app.log")
    await aiologging.shutdown()
```

### Rotating File Handler (requires aiofiles)

```python
import aiologging

async def main():
    logger = aiologging.getLogger("app")

    # Size-based rotation
    logger.addHandler(aiologging.AsyncRotatingFileHandler(
        "app.log",
        max_bytes=1024*1024,  # 1MB
        backup_count=5,
    ))

    # Time-based rotation
    logger.addHandler(aiologging.AsyncTimedRotatingFileHandler(
        "timed.log",
        when="midnight",
        backup_count=7,
    ))

    await logger.info("This will be rotated")
    await aiologging.shutdown()
```

### HTTP Handler (requires aiohttp or httpx)

```python
import aiologging

async def main():
    logger = aiologging.getLogger("app")
    logger.addHandler(aiologging.AsyncHttpHandler(
        "https://api.example.com/logs",
        headers={"Authorization": "Bearer token"},
    ))

    await logger.info("This will be sent via HTTP")
    await aiologging.shutdown()
```

HTTP handlers work on top of either `aiohttp` or `httpx`. By default
aiohttp is used when installed, with a fallback to httpx. The backend
can also be selected explicitly:

```python
http_handler = aiologging.AsyncHttpHandler(
    "https://api.example.com/logs",
    backend="httpx",  # or "aiohttp"
)
```

### Telegram Handler (requires aiohttp or httpx)

Sends log records to a Telegram chat via the Bot API. Buffered
records are combined into as few `sendMessage` calls as possible,
each within the 4096-character limit; a `429 Too Many Requests`
response is retried after the delay the Bot API returns in
`retry_after`.

```python
import aiologging

async def main():
    logger = aiologging.getLogger("app")
    logger.addHandler(aiologging.AsyncTelegramHandler(
        token="123456:ABC-DEF...",   # from @BotFather
        chat_id="-1001234567890",    # or "@channelname"
        level=aiologging.ERROR,
        parse_mode=None,             # or "HTML" / "MarkdownV2"
    ))

    await logger.error("This will be sent to Telegram")
    await aiologging.shutdown()
```

### Custom Authentication

```python
import aiologging

async def oauth_authenticator(session, request_data):
    """Custom OAuth authentication."""
    token = await refresh_oauth_token()
    return {"Authorization": f"Bearer {token}"}

http_handler = aiologging.AsyncHttpHandler(
    "https://api.example.com/logs",
    authenticator=oauth_authenticator,
)
```

## HTTP Handler Formats

### JSON Handler

```python
json_handler = aiologging.AsyncHttpJsonHandler(
    "https://api.example.com/logs"
)
```

### Text Handler

```python
text_handler = aiologging.AsyncHttpTextHandler(
    "https://api.example.com/logs"
)
```

### Protobuf Handler (requires protobuf)

```python
proto_handler = aiologging.AsyncHttpProtoHandler(
    "https://api.example.com/logs"
)
```

### Universal Handler (auto-detect format)

```python
universal_handler = aiologging.AsyncHttpHandler(
    "https://api.example.com/logs",
    format_type="application/json"  # Optional: auto-detected if not specified
)
```

## Advanced Usage

### Custom Filters

```python
import aiologging

class CustomFilter:
    def filter(self, record):
        return "important" in record.getMessage()

logger = aiologging.getLogger("app")
logger.addFilter(CustomFilter())
```

### Custom Formatters

Formatters live on handlers, exactly like in standard logging:

```python
import logging
import aiologging

handler = aiologging.AsyncStreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
aiologging.getLogger("app").addHandler(handler)
```

### Error Handling

```python
import aiologging

async def error_handler(record, exception):
    """Custom error handler for failed log operations."""
    print(f"Failed to log {record.getMessage()}: {exception}")

handler = aiologging.AsyncStreamHandler()
handler.error_handler = error_handler
```

### Performance Metrics

```python
from aiologging.logger import _logger_manager

logger = aiologging.getLogger("app")
print(logger.get_metrics())            # per-logger counters
print(_logger_manager.get_metrics())   # queue length, dropped records

for handler in logger.handlers:
    print(handler.get_metrics())       # per-handler counters
```

### Batch Processing

```python
import aiologging
from aiologging.types import BatchConfig

http_handler = aiologging.AsyncHttpHandler(
    "https://api.example.com/logs",
    batch_config=BatchConfig(
        batch_size=100,
        flush_interval=5.0,
        max_retries=3,
    ),
)
```

## Configuration Management

### Configuration from Dictionary

```python
import asyncio
import aiologging

config = {
    "version": 1,
    "loggers": {
        "myapp": {
            "level": "INFO",
            "handlers": ["console", "file"]
        }
    },
    "handlers": {
        "console": {
            "class": "stream",
            "level": "INFO",
            "stream": "stdout"
        },
        "file": {
            "class": "file",
            "level": "DEBUG",
            "filename": "app.log",
            "mode": "a"
        }
    }
}

aiologging.configure_from_dict(config)

async def main():
    logger = aiologging.get_configured_logger("myapp")
    await logger.info("This uses configured logger")
    await aiologging.shutdown()

asyncio.run(main())
```

A handler's `"class"` is either a built-in name (`stream`, `file`,
`http`, `http_json`, `telegram`, `rotating_file`,
`timed_rotating_file`) or a dotted path to your own `AsyncHandler`
subclass, as in stdlib `logging.config.dictConfig`:

```python
config["handlers"]["custom"] = {
    "class": "myapp.logging.MyHandler",  # imported at build time
    "level": "WARNING",
    "endpoint": "https://example.com",   # extra keys -> constructor
}
```

### Configuration from File

```python
aiologging.configure_from_file("logging_config.json")
```

## Migration from Standard Logging

```python
# before
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
logger.info("Message")

# after
import aiologging

aiologging.basicConfig(level=aiologging.INFO)
logger = aiologging.getLogger("app")
await logger.info("Message")            # + await

await aiologging.shutdown()             # once, at application exit
```

Key differences:

1. **Await the logging methods** — `debug`/`info`/.../`critical`, `log` and `exception` are coroutines; everything else (`setLevel`, `addHandler`, `getChild`, ...) stays synchronous.
2. **Call `await aiologging.shutdown()` at exit** — the queue must be drained while the event loop is still running.
3. **Handlers are async** — use the `Async*` handler classes (or bridge stdlib records with `captureStdlib()`).

## API Reference

### Logger Methods

Async (require `await`):

- `await logger.log(level, msg, *args, **kwargs)`
- `await logger.debug/info/warning/error/critical(msg, *args, **kwargs)`
- `await logger.exception(msg, *args, exc_info=True, **kwargs)`

All accept the standard keyword arguments: `exc_info`, `extra`, `stack_info`, `stacklevel`.

Sync (identical to `logging.Logger`):

- `setLevel(level)`, `getEffectiveLevel()`, `isEnabledFor(level)`
- `addHandler(h)` / `removeHandler(h)` / `hasHandlers()`
- `addFilter(f)` / `removeFilter(f)` / `filter(record)`
- `getChild(suffix)` / `getChildren()`
- `makeRecord(...)`, `findCaller(stack_info, stacklevel)`
- attributes: `name`, `level`, `parent`, `propagate`, `handlers`, `filters`, `disabled`

### Module Functions

- `getLogger(name=None)` — hierarchical loggers; no name returns the root logger
- `basicConfig(level, format, datefmt, handlers, force, queue_size, overflow, delivery, capture_stdlib, inline_level, atexit_flush)`
- `await flush(timeout=None)` — wait until every queued record has been handled
- `await shutdown(timeout=None)` — drain the queue and close all handlers
- `flush_sync(timeout=5.0)` — synchronous emergency drain without a running loop; returns `True` when fully delivered
- `set_atexit_flush(timeout)` — budget of the automatic drain at interpreter exit (default 2.0; 0 disables)
- `disable(level)` — like `logging.disable`
- `captureStdlib(capture=True, level=NOTSET, inline_level=None, inline_timeout=3.0, inline_burst=1, inline_rate=2.0)` — bridge stdlib logging records; `inline_level` enables synchronous inline delivery of critical records
- `await debug/info/warning/error/exception/critical/log(...)` — root-logger convenience coroutines

### Handler Classes

- `AsyncHandler` - Base async handler
- `AsyncStreamHandler` - Stream output handler
- `AsyncFileHandler` - File output handler (requires aiofiles)
- `AsyncRotatingFileHandler` - Size-based rotation (requires aiofiles)
- `AsyncTimedRotatingFileHandler` - Time-based rotation (requires aiofiles)
- `AsyncHttpHandler` - Universal HTTP handler (requires aiohttp or httpx)
- `AsyncHttpTextHandler` - Plain text HTTP handler (requires aiohttp or httpx)
- `AsyncHttpJsonHandler` - JSON HTTP handler (requires aiohttp or httpx)
- `AsyncHttpProtoHandler` - Protobuf HTTP handler (requires aiohttp or httpx, plus protobuf)
- `StdlibBridgeHandler` - Sync `logging.Handler` forwarding records into aiologging

### Configuration Classes

- `BatchConfig` - Batch processing configuration
- `FileConfig` - File handler configuration
- `HttpConfig` - HTTP handler configuration
- `LoggerConfig` - Logger configuration
- `RotationConfig` - Rotation configuration for file handlers

### Exception Classes

- `AiologgingError` - Base exception for all aiologging errors
- `HandlerError` - Base exception for handler errors
- `ConfigurationError` - Configuration-related errors
- `DependencyError` - Missing optional dependencies
- `AuthenticationError` - Authentication failures
- `NetworkError` - Network-related errors
- `FileError` - File operation errors
- `RotationError` - File rotation errors
- `BatchError` - Batch processing errors
- `FormatterError` - Formatting errors
- `LoggerError` - Logger operation errors
- `ContextError` - Context manager errors

## Performance Considerations

1. **Delivery mode**: the default `enqueue` makes logging nearly free for the caller; reserve `delivery="await"` for records that must be confirmed
2. **Backpressure**: pick the overflow policy consciously — `block` never loses records, `drop_*` never stalls the application
3. **Buffering**: use buffered/batching handlers for high-volume logging
4. **Metrics Collection**: enable metrics to monitor drops and errors
5. **Rate Limiting**: use rate limiters to prevent log flooding
6. **Shutdown**: always `await aiologging.shutdown()` before exit so nothing queued is lost

## Testing

```bash
# Install development dependencies
pip install aiologging[dev]

# Run tests
pytest

# Run tests with coverage
pytest --cov=aiologging

# Run type checking
mypy aiologging
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and type checking succeeds
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

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
