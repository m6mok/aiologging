# Adding a handler

The checklist below is the full integration path; the 0.2.2 Telegram
handler is a complete worked example — mirror its commit when in
doubt.

## 1. Pick a base class

| Base | When |
| --- | --- |
| `AsyncHandler` | one record at a time (stream-like sinks) |
| `BufferedAsyncHandler` | batching helps (network, bulk writes) |
| `AsyncHttpHandlerBase` | the sink is an HTTP endpoint |

- `AsyncHandler` subclasses implement
  `_emit(record, formatted_message)` and, if they hold resources,
  `_close_resources()`.
- `BufferedAsyncHandler` subclasses implement `flush(records)`;
  `emit()` is `await self.handle(record)` and `_emit()` is a no-op
  (see any HTTP handler).
- `AsyncHttpHandlerBase` subclasses implement
  `_prepare_request_data(records)` and `_get_content_type()`, and may
  override:
  - `_retry_delay_from_response(status, error_text)` — inspect the
    error response; return a delay in seconds to force a retry (even
    on 4xx), or None for the default policy;
  - `_redacted_url()` — hide URL-embedded secrets in errors/repr;
  - `_make_request_with_retries(...)` — e.g. to fan one batch out
    into several requests (Telegram sends one request per message
    chunk by looping over payloads and calling `super()`).

## 2. Wire it into the library

1. New module in `aiologging/handlers/`, e.g.
   `aiologging/handlers/foo.py`.
2. Export from `aiologging/handlers/__init__.py`
   (import + `__all__`).
3. Export from `aiologging/__init__.py`: add to the
   `from .handlers import (...)` block and to `__all__`.
4. Optional but conventional: a `create_foo_handler(...)` factory in
   `aiologging/__init__.py` (add to `__all__` too).
5. Register in `ConfigManager` (`aiologging/config.py`): entry in
   `_handler_registry`, branch in `_create_handler`, and a
   `_create_foo_handler(config, level)` method that validates
   required keys with `ConfigurationError`.

## 3. Validate configuration early

Reject bad constructor arguments with `ConfigurationError`
(`config_key=`, `config_value=`) at `__init__` time, not at first
emit. Missing optional dependencies raise `DependencyError` with the
`pip install aiologging[extra]` command.

## 4. Tests

`tests/test_foo_handler.py`, following
`tests/test_telegram_handler.py`:

- init/validation cases (including secret redaction in `repr` if
  applicable);
- payload/formatting unit tests calling the private helpers directly;
- delivery tests through `httpx.MockTransport` injected into
  `handler._session` (never the real network);
- retry behaviour with `BatchConfig(max_retries=…, retry_delay=0.01)`
  and a call counter;
- a `ConfigManager` round-trip if the handler is registered there.

Mind the global 80% coverage gate — an untested module fails the
suite.

## 5. Example and docs

- Runnable, offline, self-terminating example in `examples/` (CI
  executes it): fake the external service with a local aiohttp
  server; end with `await aiologging.shutdown()`.
- README: bullet in Features if user-visible, entry in the Examples
  list, a section under "Handlers" with a minimal snippet.
- If behaviour or hooks changed in the base classes, update
  `docs/architecture.md`.
