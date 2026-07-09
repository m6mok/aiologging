# Development

## Environments

Local dev uses `uv` with one virtualenv per supported Python version
(`.venv-3.9` … `.venv-3.14`), driven by the Makefile:

```bash
make setup-envs        # (re)create all virtualenvs with the dev extra
make quick-test        # pytest on the newest Python (3.14)
make quick-mypy        # mypy strict on the newest Python
make test-all          # pytest on 3.9–3.14
make mypy-all          # mypy on 3.9–3.14
make test-python3.11   # one specific version
make clean             # remove caches, coverage, virtualenvs
```

The pattern behind the targets — use it for ad-hoc commands:

```bash
VIRTUAL_ENV=.venv-3.14 uv run --active --python 3.14 <command>
```

Single test file or test:

```bash
VIRTUAL_ENV=.venv-3.14 uv run --active --python 3.14 \
    pytest tests/test_telegram_handler.py -k test_name -q
```

Lint:

```bash
VIRTUAL_ENV=.venv-3.14 uv run --active --python 3.14 \
    flake8 aiologging tests examples
```

## Testing

Pytest is configured in `pyproject.toml`:

- `asyncio_mode = "auto"` — async test functions need no marker
  (existing tests still carry `@pytest.mark.asyncio`; both work).
- Coverage is always on with a **hard gate of 80%**
  (`--cov-fail-under=80`). A full `pytest tests/` run fails below
  that, so new modules need tests in the same change. Running a
  single file with `-k`/path also triggers the gate — ignore the
  coverage failure in that case or pass `--no-cov`.

Patterns used in the suite:

- **No network, ever.** HTTP-facing handlers are tested with
  `httpx.MockTransport` injected directly into the handler:

  ```python
  handler = AsyncHttpJsonHandler(url, backend="httpx")
  handler._session = httpx.AsyncClient(
      transport=httpx.MockTransport(respond)
  )
  ```

  See `tests/test_http_handlers.py` (`TestHttpxBackend`) and
  `tests/test_telegram_handler.py`.
- Optional-dependency branches are exercised by patching the
  module-level flags, e.g.
  `patch("aiologging.handlers.http.AIOHTTP_AVAILABLE", False)`.
- Records for tests are built with `logging.LogRecord(...)` directly
  (helper `_create_record` in most test files).
- Fast retry tests set `BatchConfig(max_retries=…, retry_delay=0.01)`
  and count calls in the mock transport.

## Examples are part of CI

CI executes **every** `examples/*.py` (`for example in examples/*.py`).
An example must therefore:

- be self-contained and offline — stand in for external services with
  a local aiohttp server (see `examples/http_logging.py` and
  `examples/telegram_logging.py`, which fakes the Telegram Bot API
  including a 429 response);
- terminate on its own (`asyncio.run(main())`, no infinite loops);
- end with `await aiologging.shutdown()` to drain the queue.

## CI pipeline (`.github/workflows/ci.yml`)

On every push/PR to `master`/`main`/`develop`:

1. **test** — matrix over Python 3.9–3.14: flake8 (errors-only pass
   is blocking, full pass is advisory), `mypy aiologging`, pytest
   with coverage, Codecov upload.
2. **examples** — runs all examples on 3.12 with `[all]` extras.
3. **build** — `python -m build` + `twine check`.
