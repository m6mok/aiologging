"""Offline FULL-synchronous benchmark: python -m stress.durable.

Every compared mode returns acceptance only after durable commit. This is a
measurement tool, not a timing gate or an advertised latency guarantee.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from aiologging.durable import DurableEngine
from aiologging.journal import Journal, RecordCodec


def _percentile(values: list[float], percentile: float) -> float:
    return sorted(values)[min(len(values) - 1, int(len(values) * percentile))]


async def _case(
    size: int, producers: int, count: int, group: int
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        engine = DurableEngine(
            Path(directory) / "journal", commit_batch_size=group
        )
        record = logging.LogRecord(
            "bench", 20, "bench.py", 1, "x" * size, (), None
        )
        encoded = RecordCodec().encode(record)
        latencies: list[float] = []
        gaps: list[float] = []
        running = True

        async def heartbeat() -> None:
            previous = time.perf_counter()
            while running:
                await asyncio.sleep(0.001)
                now = time.perf_counter()
                gaps.append((now - previous) * 1000)
                previous = now

        async def produce(number: int) -> None:
            for _ in range(number):
                started = time.perf_counter()
                await engine.accept(encoded, ["unconfigured"])
                latencies.append((time.perf_counter() - started) * 1000)

        ticker = asyncio.create_task(heartbeat())
        start = time.perf_counter()
        try:
            await asyncio.gather(
                *[
                    produce(count // producers + (i < count % producers))
                    for i in range(producers)
                ]
            )
            elapsed = time.perf_counter() - start
            assert len(engine.journal.record_ids()) == count
            disk_bytes = engine.journal.path.stat().st_size
            logical_bytes = engine.journal.used_bytes
        finally:
            running = False
            await ticker
            await engine.shutdown(0)
        return {
            "record_text_bytes": size,
            "encoded_record_bytes": len(encoded.encode()),
            "producers": producers,
            "commit_batch_limit": group,
            "records": count,
            "records_per_second": round(count / elapsed, 1),
            "acceptance_ms": {
                "median": round(statistics.median(latencies), 3),
                "p95": round(_percentile(latencies, 0.95), 3),
                "p99": round(_percentile(latencies, 0.99), 3),
            },
            "application_heartbeat_max_gap_ms": round(max(gaps, default=0), 3),
            "database_bytes": disk_bytes,
            "reserved_bytes": logical_bytes,
        }


def _process(path: str, count: int, size: int, output: Any) -> None:
    encoded = RecordCodec().encode(
        logging.LogRecord("bench", 20, "x", 1, "x" * size, (), None)
    )
    start = time.perf_counter()
    with Journal(path) as journal:
        ids = [journal.append(encoded, ["a"]) for _ in range(count)]
    output.put((ids, time.perf_counter() - start))


def _multiprocess(count: int) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "journal")
        queue = context.Queue()
        processes = [
            context.Process(target=_process, args=(path, count, 1024, queue))
            for _ in range(4)
        ]
        started = time.perf_counter()
        for process in processes:
            process.start()
        accepted: list[int] = []
        times = []
        for _ in processes:
            ids, duration = queue.get(timeout=60)
            accepted.extend(ids)
            times.append(duration)
        for process in processes:
            process.join(60)
            assert process.exitcode == 0
        elapsed = time.perf_counter() - started
        queue.close()
        with Journal(path) as journal:
            assert journal.record_ids() == sorted(accepted)
            assert len(set(accepted)) == count * 4
        return {
            "processes": 4,
            "records": count * 4,
            "records_per_second_including_spawn": round(
                count * 4 / elapsed, 1
            ),
            "producer_seconds": times,
        }


async def main(count: int, output: str | None) -> None:
    cases = []
    for size in (128, 4096, 65536):
        for producers in (1, 8, 32):
            for group in (1, 32):
                cases.append(await _case(size, producers, count, group))
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "guarantees": (
            "SQLite DELETE, synchronous FULL, fullfsync ON; "
            "unfinished records retained"
        ),
        "cases": cases,
        "multiprocess": _multiprocess(max(10, count // 4)),
    }
    text = json.dumps(report, indent=2)
    if output:
        Path(output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.records < 32:
        parser.error("--records must be at least 32")
    asyncio.run(main(args.records, args.output))
