# Stress testing

The `stress/` package is a standalone harness that pushes the library
well past normal load and verifies both performance numbers and
correctness invariants. It is **not** part of the pytest suite (no
coverage gate, not collected by CI's `pytest tests/`) and, like the
tests, it never touches the network — HTTP scenarios run against
`httpx.MockTransport`, file scenarios write to a per-scenario
temporary directory.

CI runs the scaled-down smoke run (`--quick`, ~10 s) as a required
invariant gate (the `stress` job in `ci.yml`). The full `make stress`
run takes minutes and is on-demand only — do not add it to CI.

## Running

```bash
make stress            # full run (a few minutes), JSON report in logs/
make stress-quick      # scaled-down smoke run (~10 seconds)

# ad-hoc:
VIRTUAL_ENV=.venv-3.14 uv run --active --python 3.14 \
    python -m stress run [patterns ...] [--quick] [--json PATH]
VIRTUAL_ENV=.venv-3.14 uv run --active --python 3.14 \
    python -m stress list
```

`patterns` are substring filters (`overload`, `chaos.loop`, …).
Exit code is non-zero when any scenario FAILs or ERRORs; SKIPs
(missing optional dependency) are fine.

Expect stderr noise from the loop-churn scenarios (`chaos.loop_switch`,
`chaos.flush_sync_rescue`): they close event loops with tasks still
pending on purpose, so asyncio prints "Task was destroyed but it is
pending!" and similar. That is part of the scenario, not a failure.

## Layout

```
stress/
├── __main__.py     CLI (list / run)
├── runner.py       registry, Context, per-scenario isolation
├── sinks.py        instrumented handlers: CollectorHandler
│                   (delay / fail_every / (producer, seq) tracking),
│                   HangingHandler
├── workload.py     producers: concurrent burst, fixed-rate paced,
│                   stdlib-record factory for bridge scenarios
├── metrics.py      percentiles, HeapSampler (tracemalloc), wait_until
├── report.py       console report + JSON dump
└── scenarios/      one module per category
```

Every scenario runs on a fresh event loop with its own
`AsyncLoggerManager` instances (`ctx.new_manager(...)`), never the
global one, so scenarios cannot contaminate each other. Every produced
record carries `extra={"seq": i, "producer": p}`, which lets sinks
verify per-producer ordering and identify exactly which records were
dropped, lost or duplicated.

## Categories and invariants

- **throughput** — call-site vs end-to-end rate with concurrent
  producers, per-record latency of delivery mode `"await"`, fan-out
  to several handler workers. Invariants: nothing dropped with the
  `block` policy, per-producer ordering preserved.
- **overload** — tiny queues plus slow sinks. Invariants: drop
  accounting balances exactly (`sent == delivered + dropped`),
  `drop_old` keeps the newest records (sentinel technique), `block`
  loses nothing, `flush(timeout)` / `shutdown(timeout)` stay within
  their bounds under a backlog.
- **soak** — sustained paced load. Invariants: the Python heap
  plateaus (HeapSampler compares late samples with early ones),
  produce/flush cycles do not accumulate asyncio tasks, size-based
  rotation keeps `backup_count` and file sizes honest.
- **chaos** — fault injection: a sink failing every N-th attempt
  (retries must mask it without reordering), a hanging sink (must not
  stall others nor block `shutdown(timeout)`), an HTTP endpoint
  rejecting every 4th request (batch retries must deliver everything
  exactly once), killing the event loop mid-stream (the rescue path
  must lose nothing; ≤ the in-flight records may be duplicated),
  `flush_sync` draining a backlog with no loop at all, and six
  threads flooding `enqueue_from_thread`.

## Adding a scenario

1. Pick the category module in `stress/scenarios/` (or add a new one
   and import it from `stress/scenarios/__init__.py`).
2. Register with `@scenario("category.name", timeout=...)`. The
   function takes a `Context`; make it a coroutine unless it must
   manage event loops itself (then a plain function — the runner
   executes those in a daemon thread so the timeout holds even if
   the scenario wedges).
3. Scale the workload with `ctx.n(full, quick)` so `--quick` stays
   fast; create managers only via `ctx.new_manager(...)` (the runner
   shuts them down); report numbers via `ctx.metrics[...]` and
   invariants via `ctx.check(name, ok, detail)`; raise `Skip("...")`
   when an optional dependency is missing.
4. No network, no global `aiologging` state, and thresholds generous
   enough not to flake on a loaded machine — prefer exact accounting
   identities over timing assertions where possible.
