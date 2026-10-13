"""Delivery ownership, lifecycle, fallback and application-loop isolation."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import threading

import pytest

from aiologging.durable import DeliveryError, DurableEngine, Recipient
from aiologging.journal import (
    Journal,
    JournalError,
    JournalQuotaError,
    RecordCodec,
)


def payload(message="hello"):
    return RecordCodec().encode(
        logging.LogRecord("app", 20, "caller.py", 42, message, (), None)
    )


async def eventually(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_acceptance_responsive_during_disk_commit(tmp_path):
    engine = DurableEngine(tmp_path / "journal")
    started = threading.Event()
    release = threading.Event()
    append = engine.journal.append_many

    def slow_append(*args):
        started.set()
        assert release.wait(3)
        return append(*args)

    engine.journal.append_many = slow_append
    acceptance = asyncio.create_task(engine.accept(payload(), ["absent"]))
    try:
        await eventually(started.is_set)
        # If append ran on the application loop, these tasks could not run.
        ticks = []
        for _ in range(10):
            await asyncio.sleep(0)
            ticks.append(1)
        assert len(ticks) == 10
        assert not acceptance.done()
        release.set()
        record_id = await acceptance
        assert engine.journal.record_ids() == [record_id]
        with pytest.raises(DeliveryError):
            await engine.flush(0.1)
    finally:
        release.set()
        await engine.shutdown(0.1)


async def test_workers_and_resources_belong_to_library_loop(tmp_path):
    engine = DurableEngine(tmp_path / "journal")
    application_loop = asyncio.get_running_loop()
    loops = []
    threads = []
    messages = []

    async def deliver(records):
        loops.append(asyncio.get_running_loop())
        messages.extend(r.getMessage() for r in records)
        return True

    def blocking(records):
        threads.append(threading.get_ident())
        return True

    engine.register(Recipient("async", deliver, batch_size=20))
    engine.register(Recipient("blocking", blocking))
    try:
        for i in range(12):
            await engine.accept(payload(str(i)), ["async", "blocking"])
        await engine.flush(3)
        assert messages == [str(i) for i in range(12)]
        assert all(loop is engine._loop for loop in loops)
        assert application_loop not in loops
        assert all(t != threading.get_ident() for t in threads)
    finally:
        assert await engine.shutdown(1)


async def test_fallback_once_keeps_primary_then_primary_recovers(tmp_path):
    path = tmp_path / "journal"
    engine = DurableEngine(path)
    attempts = []
    fallbacks = []
    available = threading.Event()

    async def primary(records):
        attempts.append(records[0].getMessage())
        return available.is_set()

    async def fallback(records):
        fallbacks.extend(r.getMessage() for r in records)
        return True

    engine.register(Recipient("a", primary, fallback, retry_delay=0.01))
    try:
        record_id = await engine.accept(payload(), ["a"])
        await eventually(lambda: len(attempts) >= 3)
        assert fallbacks == ["hello"]
        assert engine.journal.record_ids() == [record_id]
        with pytest.raises(DeliveryError):
            await engine.flush(0.01)
        available.set()
        await engine.flush(3)
        assert engine.journal.record_ids() == []
        assert fallbacks == ["hello"]
    finally:
        await engine.shutdown(0.1)


async def test_failed_recipient_does_not_block_others_or_skip_older(tmp_path):
    engine = DurableEngine(tmp_path / "journal")
    failures = []
    successes = []

    async def fail(records):
        failures.append(records[0].getMessage())
        return False

    async def succeed(records):
        successes.extend(r.getMessage() for r in records)
        return True

    engine.register(Recipient("bad", fail, retry_delay=0.01))
    engine.register(Recipient("good", succeed))
    try:
        ids = [
            await engine.accept(payload(str(i)), ["bad", "good"])
            for i in range(5)
        ]
        await eventually(lambda: len(successes) == 5)
        assert successes == [str(i) for i in range(5)]
        assert set(failures) == {"0"}
        assert engine.journal.record_ids() == ids
        assert await engine.pending() == {"bad": 5}
    finally:
        assert not await engine.shutdown(0.05)


async def test_two_engines_share_one_sender_and_recover_unconfigured(tmp_path):
    path = tmp_path / "journal"
    messages = []
    active = 0
    maximum = 0

    async def deliver(records):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.002)
        messages.extend(r.getMessage() for r in records)
        active -= 1
        return True

    first = DurableEngine(path)
    second = DurableEngine(path)
    first.register(Recipient("a", deliver))
    second.register(Recipient("a", deliver))
    try:
        for i in range(10):
            await (first if i % 2 else second).accept(payload(str(i)), ["a"])
        await first.flush(3)
        assert messages == [str(i) for i in range(10)]
        assert maximum == 1
    finally:
        await first.shutdown(1)
        await second.shutdown(1)
    with Journal(path) as journal:
        record_id = journal.append(payload("old"), ["original"])
    third = DurableEngine(path)
    try:
        await third.accept(payload("new"), ["different"])
        with pytest.raises(DeliveryError):
            await third.flush(0.01)
        assert record_id in third.journal.record_ids()
        third.register(Recipient("original", deliver))
        await eventually(lambda: "old" in messages)
        assert "new" not in messages
    finally:
        await third.shutdown(0.01)


async def test_quota_and_delivery_recursion_are_explicit_errors(tmp_path):
    engine = DurableEngine(tmp_path / "journal", quota_bytes=1)
    try:
        with pytest.raises(JournalQuotaError):
            await engine.accept(payload(), ["a"])
        with pytest.raises(JournalQuotaError):
            engine.accept_sync(payload(), ["a"])
        assert engine.journal.record_ids() == []
    finally:
        await engine.shutdown(0)
    engine = DurableEngine(tmp_path / "second")
    errors = []

    def recursive(records):
        try:
            engine.accept_sync(payload("recursive"), ["a"])
        except DeliveryError as error:
            errors.append(error)
        return True

    engine.register(Recipient("a", recursive))
    try:
        await engine.accept(payload(), ["a"])
        await engine.flush(3)
        assert len(errors) == 1
    finally:
        await engine.shutdown(1)


async def test_timed_out_blocking_io_does_not_release_live_ownership(tmp_path):
    path = tmp_path / "journal"
    engine = DurableEngine(path)
    started = threading.Event()
    release = threading.Event()

    def block(records):
        started.set()
        assert release.wait(5)
        return True

    engine.register(Recipient("a", block, primary_timeout=0.01))
    record_id = await engine.accept(payload(), ["a"])
    try:
        await eventually(started.is_set)
        assert not await engine.shutdown(0.02)
        with Journal(path) as other:
            assert other.acquire("a") is None
            assert other.record_ids() == [record_id]
        release.set()
        await eventually(lambda: engine._closed)
        with Journal(path) as other:
            with other.acquire("a"):
                # A late unconfirmed operation may have succeeded, but a
                # timeout cannot turn its journal task into a silent loss.
                assert other.record_ids() == [record_id]
    finally:
        release.set()


async def test_cancelled_wait_does_not_cancel_committed_work(tmp_path):
    engine = DurableEngine(tmp_path / "journal")
    started = threading.Event()
    release = threading.Event()
    original = engine.journal.append_many

    def append(*args):
        started.set()
        assert release.wait(3)
        return original(*args)

    engine.journal.append_many = append
    task = asyncio.create_task(engine.accept(payload(), ["a"]))
    try:
        await eventually(started.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await eventually(lambda: len(engine.journal.record_ids()) == 1)
    finally:
        release.set()
        await engine.shutdown(0)


def test_plain_exit_drains_without_manual_shutdown_or_new_threads(tmp_path):
    path = tmp_path / "journal"
    output = tmp_path / "output"
    code = """
import asyncio, sys, threading
from aiologging.durable import DurableEngine, Recipient
from aiologging.journal import RecordCodec
import logging
engine = DurableEngine(sys.argv[1])
def deliver(records):
    with open(sys.argv[2], "a") as stream:
        stream.write("\\n".join(r.getMessage() for r in records) + "\\n")
        stream.flush()
    return True
engine.register(Recipient("a", deliver, batch_size=100))
async def main():
    for i in range(30):
        record = logging.LogRecord("app", 20, "x", 1, str(i), (), None)
        await engine.accept(RecordCodec().encode(record), ["a"])
asyncio.run(main())
def forbidden(*args, **kwargs):
    raise AssertionError("Started a thread during exit")
threading.Thread.start = forbidden
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(path), str(output)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert process.returncode == 0, process.stderr
    assert output.read_text().splitlines() == [str(i) for i in range(30)]
    with Journal(path) as journal:
        assert journal.pending() == {}


def test_configuration_validation_and_duplicate_identity(tmp_path):
    with pytest.raises(ValueError):
        DurableEngine(tmp_path / "j", admission_capacity=0)
    with pytest.raises(ValueError):
        Recipient("", lambda records: True)
    with pytest.raises(ValueError):
        Recipient("a", lambda records: True, batch_size=0)
    engine = DurableEngine(tmp_path / "j")
    recipient = Recipient("a", lambda records: True)
    engine.register(recipient)
    engine.register(recipient)
    with pytest.raises(ValueError):
        engine.register(Recipient("a", lambda records: True))
    assert engine.flush_sync(1)
    assert engine.shutdown_sync(1)
    assert engine.shutdown_sync(1)
    with pytest.raises(JournalError):
        engine.accept_sync(payload(), ["a"])


async def test_concurrent_acceptance_uses_a_single_confirmed_group_commit(
    tmp_path,
):
    engine = DurableEngine(tmp_path / "journal", commit_batch_size=32)
    started = threading.Event()
    release = threading.Event()
    sizes = []
    append = engine.journal.append_many

    def measured(records):
        sizes.append(len(records))
        return append(records)

    def hold_actor():
        started.set()
        assert release.wait(3)

    engine.journal.append_many = measured
    engine._journal_worker.submit(hold_actor)
    try:
        await eventually(started.is_set)
        tasks = [
            asyncio.create_task(engine.accept(payload(str(i)), ["a"]))
            for i in range(8)
        ]
        await asyncio.sleep(0)
        assert not any(task.done() for task in tasks)
        release.set()
        ids = await asyncio.gather(*tasks)
        assert sizes == [8]
        assert engine.journal.record_ids() == ids
    finally:
        release.set()
        await engine.shutdown(0)


async def test_primary_head_failure_does_not_starve_later_fallback_tasks(
    tmp_path,
):
    engine = DurableEngine(tmp_path / "journal")
    primary = []
    backups = []
    available = threading.Event()

    async def deliver(records):
        primary.extend(r.getMessage() for r in records)
        return available.is_set()

    async def fallback(records):
        backups.extend(r.getMessage() for r in records)
        return True

    engine.register(
        Recipient("a", deliver, fallback, batch_size=1, retry_delay=0.01)
    )
    try:
        ids = [await engine.accept(payload(str(i)), ["a"]) for i in range(5)]
        await eventually(lambda: len(backups) == 5)
        assert backups == [str(i) for i in range(5)]
        assert set(primary) == {"0"}
        assert engine.journal.record_ids() == ids
        await asyncio.sleep(0.05)
        assert len(backups) == 5
        available.set()
        await engine.flush(3)
        assert primary[-5:] == [str(i) for i in range(5)]
        assert len(backups) == 5
    finally:
        await engine.shutdown(1)


def test_public_api_exit_without_shutdown_keeps_rendered_order(tmp_path):
    path = tmp_path / "journal"
    output = tmp_path / "output"
    code = """
import asyncio, sys, threading
import aiologging as logging
logging.configureJournal(sys.argv[1])
logging.basicConfig(
    filename=sys.argv[2], level=logging.INFO, format="%(message)s"
)
async def main():
    for i in range(40):
        await logging.info("record %d", i)
asyncio.run(main())
def forbidden(*args, **kwargs):
    raise AssertionError("New thread during interpreter exit")
threading.Thread.start = forbidden
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), str(output)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text().splitlines() == [
        f"record {i}" for i in range(40)
    ]
    with Journal(path) as journal:
        assert journal.pending() == {}


async def test_replacement_closes_old_resources_after_inflight_work(tmp_path):
    engine = DurableEngine(tmp_path / "journal")
    started = threading.Event()
    release = threading.Event()
    closed = []

    async def old(records):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        assert closed == []
        return True

    async def close_old():
        closed.append("old")

    async def current(records):
        return True

    engine.register(Recipient("a", old, close=close_old))
    try:
        await engine.accept(payload(), ["a"])
        await eventually(started.is_set)
        engine.register(Recipient("a", current), replace=True)
        assert closed == []
        release.set()
        await eventually(lambda: closed == ["old"])
        await engine.accept(payload("new"), ["a"])
        await engine.flush(3)
    finally:
        release.set()
        await engine.shutdown(1)
