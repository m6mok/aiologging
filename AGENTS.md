# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

aiologging — asynchronous logging library for macOS and Python (3.11–3.15) that
mirrors the standard `logging` API with coroutine logging methods.
Records/routes are frozen at the call site and durably committed to a
shared SQLite journal. A prestarted library loop and per-recipient blocking
workers deliver ordered batches with independent ACKs. The previous pipeline
is isolated in aiologging.legacy. Zero required dependencies; optional HTTP
transports: aiohttp or httpx. Files use stdlib Python.


## Quick commands

```bash
make quick-test        # pytest on the newest Python (3.15)
make quick-mypy        # mypy strict on the newest Python
make test-all          # pytest on all supported versions (3.11–3.15)

# ad-hoc (single test, flake8, examples):
VIRTUAL_ENV=.venv-3.14 uv run --active --python 3.14 <command>
```

## Documentation

Detailed guides live in [docs/](docs/):

- [docs/architecture.md](docs/architecture.md) — the record path
  (acceptance, shared journal, recipient ownership, fallback), stdlib
  views, handler confirmation and lifecycle.
- [docs/development.md](docs/development.md) — environments and
  Makefile targets, testing patterns (`httpx.MockTransport`, no
  network), the 80% coverage gate, examples-run-in-CI rule, CI
  pipeline.
- [docs/conventions.md](docs/conventions.md) — mypy strict at
  Python 3.11, 79-char lines, docstring format, optional-dependency
  and error-handling patterns, `LazyLock`, secret redaction.
- [docs/adding-a-handler.md](docs/adding-a-handler.md) — base-class
  choice, full integration checklist (exports, stdlib configuration,
  tests, example, README).
- [docs/stress-testing.md](docs/stress-testing.md) — the `stress/`
  harness (`make stress` / `make stress-quick`): scenario categories,
  correctness invariants, how to add a scenario.
- [docs/releases.md](docs/releases.md) — versioning (0.2.x, version
  in two files), commit message format `type: summary (X.Y.Z)`,
  CI/release pipeline.

## TODO

(Legacy roadmap: currently empty — the 2026-07 stress-testing roadmap is complete;
see [docs/stress-testing.md](docs/stress-testing.md) for the
resulting harness. It found and fixed two real delivery bugs: the
atexit drain losing the whole file backlog, and the dead-loop rescue
losing in-flight records. Reminder: `make stress-quick` gates CI,
the full `make stress` stays on-demand only.)

The durable path adds tests/durable/ and python -m stress.durable.

## Durable compatibility

- Read docs/migration.md for snapshot, custom handler, identity and fork limits.
- No silent eviction or false success in durable mode.
- No new worker/executor creation during atexit.
- Treat ordinary stdlib handleError as standard processing, not verified delivery.
- Benchmark only modes with equal commit guarantees.

## Hard rules

- Never touch build artifacts: `build/` (stale package copies in
  `build/lib/`), `dist/`, `htmlcov/`, `*.egg-info/`, `.venv*`.
- Tests and examples must not use the network; examples must run
  offline and terminate (CI executes every `examples/*.py`).
- Keep `pyproject.toml` `version` and `aiologging/__init__.py`
  `__version__` in sync; never bump versions or create releases
  unprompted.
- `aiologging/` must pass mypy strict at `python_version = 3.11`.
