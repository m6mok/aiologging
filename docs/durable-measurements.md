# Durable measurements

Measured on 2026-10-13, macOS arm64, Python 3.14.6. See
[durable-benchmark.json](durable-benchmark.json) for every sample and environment.
Each case accepts 128 records. This small local run is evidence for choosing
bounded batching, not an SLA or a general performance claim. Repeat longer
runs on the application workload before capacity planning.

All compared modes use SQLite DELETE, synchronous FULL, fullfsync ON and
retain every unfinished ID. There is no comparison with a memory-only or
flush-without-fsync logger. The benchmark measures the journal engine with
pre-encoded records, not complete application formatting or remote delivery.
It includes sizes 128/4096/65536 bytes, 1/8/32 async producers, and four
concurrent spawned processes. Acceptance latency is measured until commit;
a 1 ms application heartbeat records responsiveness while producers run.

Selected 32-producer cases:

| Text bytes | Commit batch limit | Records/s | Acceptance p95 ms | Max heartbeat gap ms |
| --- | --- | --- | --- | --- |
| 128 | 1 | 91.2 | 359.207 | 1.938 |
| 128 | 32 | 1306.6 | 28.766 | 3.789 |
| 4096 | 1 | 78.8 | 416.632 | 3.121 |
| 4096 | 32 | 1091.1 | 33.007 | 5.966 |
| 65536 | 1 | 84.5 | 413.407 | 4.226 |
| 65536 | 32 | 524.5 | 67.54 | 22.204 |

## Defaults and capacity

The maximum commit batch is 32 already queued records. It does not impose a
batching delay on a lone producer. The concurrent measurements show why
bounded group commit is useful; single-producer variation also shows why
small local samples cannot justify a latency guarantee. Confirmed file/stream
and network adapters use a batch limit of 32; protocol/server limits and
actual payload sizes may require lower limits.

The public default logical quota is 64 MiB and the SQLite page limit 128 MiB.
The raw journal samples include the actual database/reserved byte counts;
they show additional SQLite/index/claim overhead, especially for short text.
The two limits bound different resources; the physical limit provides a
separate backstop rather than an assumed fixed expansion ratio. Rollback
journals and filesystem metadata still require spare disk space.

The 64 MiB value is a configurable storage policy, not a retention-time or
throughput promise. Public per-handler snapshots and frozen formatted text
add more bytes than this raw journal benchmark. Unicode escaping, extra,
exception/stack text, multiple destinations and repeated handler visits also
change capacity. Do not derive a universal record count from text length.

Admission capacity 1024 bounds the number of pending operations, not their
combined byte size. Oversized records/quota exhaustion explicitly fail; no
unfinished record is evicted. Measure large records and tune admission and
quota for the application. Journal.quota_bytes adjusts the shared logical
limit transactionally without deleting existing records. Persisted limits
are adopted on restart when configureJournal arguments are omitted.

Reproduce with python -m stress.durable --records 1000 --output report.json.
The CLI checks accepted-ID conservation but never imposes timing thresholds.
