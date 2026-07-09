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
- [docs/stress-testing.md](docs/stress-testing.md) — the `stress/`
  harness (`make stress` / `make stress-quick`): scenario categories,
  correctness invariants, how to add a scenario.
- [docs/releases.md](docs/releases.md) — versioning (0.2.x, version
  in two files), commit message format `type: summary (X.Y.Z)`,
  CI/release pipeline.

## TODO

Stress-testing roadmap (from the 2026-07 harness review; see
[docs/stress-testing.md](docs/stress-testing.md) for context).

Priority — CI integration:

- [x] Add `make stress-quick` (~11 s, 25 scenarios) to the CI
  pipeline as a required invariant gate. `make stress` (the full
  multi-minute run) stays on-demand only — never add it to CI.

Harness fixes:

- [x] Enforce `Scenario.timeout` for sync (plain-function) scenarios
  in `stress/runner.py` — today only coroutine scenarios are wrapped
  in `wait_for`, so a deadlocked `flush_sync` scenario hangs the
  whole run (run the function in a thread with `join(timeout)`).
- [ ] Perf baselines: compare throughput metrics against a checked-in
  baseline with a generous tolerance (or at least keep run history) —
  today a 5x regression still reports OK.
- [ ] Sample RSS (`resource.getrusage`) alongside `tracemalloc` in
  `HeapSampler` — C-level allocations (httpx/aiofiles) are invisible.
- [ ] Add `--repeat N` to `python -m stress run` for hunting rare
  races (`chaos.loop_switch`, the thread-swarm scenarios).
- [ ] Replace the tautological "no batch was delivered twice" check
  in `chaos.http_mock_endpoint` with a set-based duplicate check on
  `(producer, seq)`; let `Context.new_manager` accept `drop_policy`.

New scenarios (delivery-guarantee gaps first):

- [x] atexit drain (0.2.5): subprocess scenario — a script logs to a
  file handler and exits without `shutdown()`; verify the file
  contents. The only untested headline delivery guarantee.
  (Found and fixed a real bug: `AsyncFileHandler` lost the whole
  backlog at atexit because aiofiles needs executors, which are dead
  during interpreter shutdown — now falls back to blocking I/O.)
- [x] D2 inline error path: `urlopen` raising (ConnectionError /
  HTTP 500) must fall back to the queue exactly once; plus a mixed
  INFO+ERROR stream through the bridge (all current D2 scenarios are
  pure-ERROR).
- [x] D3 matrix: `LevelAwareDrop` over `drop_new` (only `drop_old` is
  covered), overload via `enqueue_from_thread` with a drop policy.
  (Watermark-boundary behaviour is already pinned by unit tests in
  `tests/test_drop_policy.py` — no stress scenario needed.)
- [ ] `delivery="await"` under faults: flaky/slow sink, full queue
  with drop policies, and cancellation of an awaiting caller (record
  neither lost nor duplicated).
- [ ] `AsyncTimedRotatingFileHandler` churn (soak covers only
  size-based rotation).
- [ ] Seeded chaos-mix fuzz scenario: random levels / payload sizes /
  interleaved `flush` + `flush_sync` / injected sink failures /
  occasional loop churn, checked only by the accounting identity
  (`sent == delivered + dropped`, duplicates ≤ in-flight); report the
  seed in metrics.

Lower priority:

- [ ] Runtime topology churn (add/removeHandler mid-stream), noisy
  vs. quiet logger fairness on the shared queue, manager reuse after
  `shutdown()`, large payloads / `exc_info` tracebacks (current
  payloads are 32–96 B), aiohttp backend (mock `_session.post`; no
  MockTransport equivalent).

## Hard rules

- Never touch build artifacts: `build/` (stale package copies in
  `build/lib/`), `dist/`, `htmlcov/`, `*.egg-info/`, `.venv*`.
- Tests and examples must not use the network; examples must run
  offline and terminate (CI executes every `examples/*.py`).
- Keep `pyproject.toml` `version` and `aiologging/__init__.py`
  `__version__` in sync; never bump versions or create releases
  unprompted.
- `aiologging/` must pass mypy strict at `python_version = 3.9`.
