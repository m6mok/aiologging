# Architecture

## Call to durable acceptance

The public async view delegates record creation to the actual synchronous
stdlib logger. Stdlib factories, logger filters, caller/stacklevel, adapters,
levels and propagation remain authoritative. A context variable collects
routes during an async call; a synchronous dependency uses the same routing
hook and waits for commit in its calling thread.

The hook installed on `logging.Logger.callHandlers` captures destinations at
acceptance. Documented custom overrides still run and may call `super()`.
Handler filters using ordinary `Handler.handle` run at acceptance: each
recipient gets its frozen replacement/mutation snapshot. Stock stream/file
handlers and our text network adapters also freeze their final formatted
output at acceptance; receiver metadata is stored outside user extra fields. Repeated visits to
the same handler are retained. Custom overrides of `handle` retain their own
processing on the handler worker. Completely overridden dispatch owns its
own behaviour, as it does in stdlib.

```
producer -> stdlib creation/filter/route -> snapshot codec
         -> bounded admission queue -> journal worker -> SQLite COMMIT
         <- acceptance future resolves only here

recipient worker -> OS lock -> oldest journal batch -> I/O -> recipient ACK
                                                -> fallback ACK on failure
```

The admission queue is bounded; overload is an explicit rejection. Queued
accepts can share a transaction, up to 32 already available records, with no
intentional delay for a lone producer. A failed transaction rejects the whole
group. Cancellation of an async wait does not undo work already submitted;
callers with cancelled waits cannot infer whether acceptance committed.

## Journal and ownership

`journal.py` uses only stdlib sqlite3, JSON and macOS file locks. SQLite uses
DELETE rollback journals, synchronous FULL and fullfsync ON. Initial schema
publication uses an initialization lock, atomic rename and directory fsync.
Opening verifies schema/version, SQLite integrity, foreign keys and accounting
in one read snapshot. Record and route checksums detect valid-JSON/metadata changes as well as
structural corruption. Corrupt, partial, foreign-schema and inconsistent
journals are preserved and raise; there is no automatic replacement/reset.

`records` assigns monotonic IDs; `deliveries` stores one primary claim and
fallback confirmation per original destination. Completed destinations are
removed independently; a record is removed only after all primary tasks
have ACKed. Fallback never removes a primary task. After a primary failure, fallback
walks its own oldest unfinished snapshots within a bounded pass, so a blocked
primary head cannot starve newer fallback tasks. Encoded record/route bytes
are charged transactionally until a record is fully complete. SQLite page
limits are separate from logical reservation; rollback files and metadata
still require additional free space. Limits are persistent and inconsistent
explicit settings between processes raise. A live quota_bytes adjustment is
shared transactionally; implicit defaults adopt the persisted limit on restart.

Recipient ownership is an exclusive OS flock held across claim, I/O and ACK.
It has no expiry. A process crash releases its lock; another process recovers
that destination's oldest tasks with a new fencing token. Closed/inherited
leases cannot ACK. A crash after send and before ACK permits duplicate sends.
Lock files are never unlinked during ownership, preventing inode races.
SQLite IDs define the order of attempts per destination; absolute ordering
at a remote service across retries is not claimed.

## Delivery and lifecycle

`durable.py` starts a journal thread and a dedicated event loop during
configuration, and an I/O worker before each recipient begins sending.
Workers are daemon threads with explicit future queues, not ThreadPoolExecutor.
No threads or executors are first created from atexit. Async transport factories
run on the library loop. Blocking callbacks each have their own worker.

Ordinary exit drains through these existing resources, then closes transports.
Default total exit budget is 10 seconds; our network recipients have separate
5-second primary and fallback budgets. Configured limits can be changed.
Failed tasks remain persistent after budgets run out. Unavailable original
handlers release ownership so another process with that destination can
recover them. Removed targets never receive another target's old task.

A timeout does not forcibly stop arbitrary Python blocking code. Its live
worker retains ownership until it finishes, and cleanup completes later.
This prevents a second process from sending concurrently with the timed-out
worker. Use multiprocessing spawn; inherited engines/connections reject use.

## Compatibility and confirmation

Confirmed file handlers propagate failures and flush/fullfsync before ACK;
rotation also syncs the old file and directory changes. Streams confirm a
successful write/flush. Optional HTTP confirms a successful response, and
Telegram additionally requires `ok: true`. Partial batched sends may repeat
successful earlier items after a crash/failure; this is at-least-once delivery.

Synchronous stdlib/custom handlers preserve their usual error behaviour. An
ordinary call returning from `emit` may have hidden an error in `handleError`;
the library records standard processing, not verified delivery. Implement
`deliver(records) -> True` or use confirmed adapters for strict guarantees.

The synchronous actual logger stays in the stdlib manager. Async facades are
cached and share underlying handler/filter collections. Logger subclasses and Handler remain synchronous extension bases; direct
Logger/RootLogger construction produces an async view. `getLogger` views preserve subtype checks
for the configured logger class; parent/getChild/getChildren return cached
async views. See migration.md for direct construction and custom-dispatch
boundaries. The previous independent in-memory hierarchy remains isolated in
`legacy.py`, `legacy_config.py`, `logger.py`, bridge and Async* handler modules;
see legacy-architecture.md for its historical behaviour.
