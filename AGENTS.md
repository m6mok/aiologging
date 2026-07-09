# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

aiologging — asynchronous logging library for Python (3.9–3.14) that
mirrors the standard `logging` API with coroutine logging methods.
Records are enqueued at the call site; a background consumer fans
them out to per-handler worker queues that perform the I/O. Zero required dependencies; optional extras: `aiofiles`,
`aiohttp` / `httpx`, `protobuf`.

## Quick commands

```bash
make quick-test        # pytest on the newest Python (3.14)
make quick-mypy        # mypy strict on the newest Python
make test-all          # pytest on all supported versions (3.9–3.14)

# ad-hoc (single test, flake8, examples):
VIRTUAL_ENV=.venv-3.14 uv run --active --python 3.14 <command>
```

## Documentation

Detailed guides live in [docs/](docs/):

- [docs/architecture.md](docs/architecture.md) — the record path
  (queue, consumer, delivery/overflow modes), stdlib bridge, handler
  class hierarchy and extension hooks, module map.
- [docs/development.md](docs/development.md) — environments and
  Makefile targets, testing patterns (`httpx.MockTransport`, no
  network), the 80% coverage gate, examples-run-in-CI rule, CI
  pipeline.
- [docs/conventions.md](docs/conventions.md) — mypy strict at
  Python 3.9, 79-char lines, docstring format, optional-dependency
  and error-handling patterns, `LazyLock`, secret redaction.
- [docs/adding-a-handler.md](docs/adding-a-handler.md) — base-class
  choice, full integration checklist (exports, factory, ConfigManager,
  tests, example, README).
- [docs/releases.md](docs/releases.md) — versioning (0.2.x, version
  in two files), commit message format `type: summary (X.Y.Z)`,
  CI/release pipeline.

## Hard rules

- Never touch build artifacts: `build/` (stale package copies in
  `build/lib/`), `dist/`, `htmlcov/`, `*.egg-info/`, `.venv*`.
- Tests and examples must not use the network; examples must run
  offline and terminate (CI executes every `examples/*.py`).
- Keep `pyproject.toml` `version` and `aiologging/__init__.py`
  `__version__` in sync; never bump versions or create releases
  unprompted.
- `aiologging/` must pass mypy strict at `python_version = 3.9`.
