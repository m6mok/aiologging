"""Recovery invariants using real SQLite, OS locks and spawned processes."""

from __future__ import annotations

import json
import logging
import multiprocessing
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from aiologging.journal import (
    Journal,
    JournalCorruptError,
    JournalError,
    JournalQuotaError,
    OwnershipError,
    RecordCodec,
    RecordEncodingError,
    default_journal_path,
)


def payload(message="hello", **extra):
    record = logging.LogRecord("app", 20, __file__, 42, message, (), None)
    record.__dict__.update(extra)
    return RecordCodec().encode(record)


def producer(path, count, output):
    with Journal(path) as journal:
        ids = [
            journal.append(payload(str(i)), ["a", "b"]) for i in range(count)
        ]
    output.put(ids)


def contender(path, output):
    with Journal(path) as journal:
        lease = journal.acquire("a")
        output.put(lease is not None)
        if lease:
            lease.close()


def test_accept_commit_reopen_and_independent_ack(tmp_path):
    path = tmp_path / "journal"
    journal = Journal(path)
    first, second = journal.append_many(
        [(payload("first"), ["a", "b"]), (payload("second"), ["a"])]
    )
    journal.close()
    with Journal(path) as journal:
        assert journal.record_ids() == [first, second]
        assert journal.pending() == {"a": 2, "b": 1}
        with journal.acquire("a") as lease:
            batch = journal.claim(lease, 10)
            assert [task.record_id for task in batch] == [first, second]
            journal.acknowledge(lease, [first, second])
        assert journal.record_ids() == [first]
        with journal.acquire("b") as lease:
            assert journal.claim(lease)[0].record_id == first
            journal.acknowledge(lease, [first])
        assert journal.pending() == {}
        assert journal.used_bytes == 0
        assert journal.append(payload("third"), ["b"]) > second


def test_fallback_ack_survives_reopen_keeps_primary(tmp_path):
    path = tmp_path / "journal"
    with Journal(path) as journal:
        record_id = journal.append(payload(), ["primary"])
        with journal.acquire("primary") as lease:
            assert not journal.claim(lease)[0].fallback_done
            journal.acknowledge(lease, [record_id], fallback=True)
    with Journal(path) as journal:
        assert journal.record_ids() == [record_id]
        with journal.acquire("primary") as lease:
            assert journal.claim(lease)[0].fallback_done
            journal.acknowledge(lease, [record_id])
        assert journal.record_ids() == []


def test_claims_require_live_current_owner(tmp_path):
    path = tmp_path / "journal"
    with Journal(path) as first, Journal(path) as second:
        record_id = first.append(payload(), ["a"])
        lease = first.acquire("a")
        assert second.acquire("a") is None
        with pytest.raises(OwnershipError):
            second.claim(lease)
        with pytest.raises(OwnershipError):
            first.acknowledge(lease, [record_id])
        first.claim(lease)
        lease.close()
        with pytest.raises(OwnershipError):
            first.acknowledge(lease, [record_id])
        with second.acquire("a") as successor:
            assert second.claim(successor)[0].record_id == record_id
            second.acknowledge(successor, [record_id])


def test_process_lock_no_time_expiry_and_no_duplicate_live_owner(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    with Journal(tmp_path / "journal") as journal:
        journal.append(payload(), ["a"])
        lease = journal.acquire("a")
        queue = ctx.Queue()
        process = ctx.Process(target=contender, args=(journal.path, queue))
        process.start()
        assert queue.get(timeout=20) is False
        process.join(20)
        assert process.exitcode == 0
        lease.close()
        process = ctx.Process(target=contender, args=(journal.path, queue))
        process.start()
        assert queue.get(timeout=20) is True
        process.join(20)
        assert process.exitcode == 0
        queue.close()


def test_concurrent_processes_accept_unique_ordered_ids(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    path = tmp_path / "journal"
    processes = [
        ctx.Process(target=producer, args=(path, 15, queue)) for _ in range(4)
    ]
    for process in processes:
        process.start()
    accepted = []
    for _ in processes:
        ids = queue.get(timeout=30)
        assert ids == sorted(ids)
        accepted.extend(ids)
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    queue.close()
    assert len(set(accepted)) == 60
    with Journal(path) as journal:
        assert journal.record_ids() == sorted(accepted)
        with journal.acquire("a") as lease:
            assert [d.record_id for d in journal.claim(lease, 100)] == sorted(
                accepted
            )


# The child reports a committed ID before its caller kills it. The separate
# stdout handshake makes a killed-but-uncommitted ID impossible to mistake
# for accepted. No timing sleeps are needed to pick a crash boundary.
CRASH_CHILD = """
import contextlib, os, sys
from aiologging.journal import Journal, RecordCodec
import logging
path, boundary = sys.argv[1:]
journal = Journal(path)
payload = RecordCodec().encode(
    logging.LogRecord("app", 20, "x", 1, "x", (), None)
)
if boundary == "before_commit":
    original = journal._transaction
    @contextlib.contextmanager
    def transaction():
        with original() as connection:
            yield connection
            print("WRITTEN", flush=True)
            sys.stdin.readline()
    journal._transaction = transaction
    journal.append(payload, ["a"])
else:
    record_id = journal.append(payload, ["a"])
    print(record_id, flush=True)
    lease = journal.acquire("a")
    journal.claim(lease)
    if boundary == "after_send":
        with open(path + ".sink", "a") as sink:
            sink.write(str(record_id) + "\\n")
            sink.flush()
            os.fsync(sink.fileno())
    elif boundary == "after_ack":
        journal.acknowledge(lease, [record_id])
    elif boundary == "during_ack":
        original = journal._transaction
        @contextlib.contextmanager
        def transaction():
            with original() as connection:
                yield connection
                print("BOUNDARY", flush=True)
                sys.stdin.readline()
        journal._transaction = transaction
        journal.acknowledge(lease, [record_id])
    print("BOUNDARY", flush=True)
    sys.stdin.readline()
"""


@pytest.mark.parametrize(
    "boundary",
    ["before_commit", "after_commit", "after_send", "during_ack", "after_ack"],
)
def test_sigkill_at_transaction_send_and_ack_boundaries(tmp_path, boundary):
    path = tmp_path / "journal"
    # Start from a valid DB so before_commit tests hot-journal recovery.
    Journal(path).close()
    process = subprocess.Popen(
        [sys.executable, "-c", CRASH_CHILD, str(path), boundary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = process.stdout.readline().strip()
        if boundary != "before_commit":
            record_id = int(line)
            assert process.stdout.readline().strip() == "BOUNDARY"
        else:
            assert line == "WRITTEN"
        process.kill()
        _, errors = process.communicate(timeout=20)
        assert process.returncode == -signal.SIGKILL, errors
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=20)
    with Journal(path) as journal:
        if boundary in ("before_commit", "after_ack"):
            assert journal.record_ids() == []
        else:
            assert journal.record_ids() == [record_id]
            with journal.acquire("a") as lease:
                assert journal.claim(lease)[0].record_id == record_id
                if boundary == "after_send":
                    # Re-delivery is allowed; disappearance before ACK is not.
                    assert (
                        Path(str(path) + ".sink").read_text()
                        == f"{record_id}\n"
                    )
                journal.acknowledge(lease, [record_id])


def test_quota_atomic_group_rollback_and_space_reclamation(tmp_path, capsys):
    encoded = payload()
    size = len(encoded.encode()) + 65
    with Journal(tmp_path / "journal", quota_bytes=size * 2) as journal:
        record_id = journal.append(encoded, ["a"])
        with pytest.raises(JournalQuotaError):
            journal.append_many([(encoded, ["a"]), (encoded, ["a"])])
        assert journal.record_ids() == [record_id]
        assert journal.used_bytes == size
        assert "not accepted" in capsys.readouterr().err
        with journal.acquire("a") as lease:
            journal.claim(lease)
            journal.acknowledge(lease, [record_id])
        assert journal.used_bytes == 0
        journal.append_many([(encoded, ["a"]), (encoded, ["a"])])


def test_sqlite_full_and_readonly_propagate_without_success(tmp_path):
    path = tmp_path / "journal"
    with Journal(path, max_database_bytes=65536) as journal:
        record_id = journal.append(payload(), ["a"])
        with pytest.raises(JournalQuotaError):
            journal.append(payload("x" * 100000), ["a"])
        assert journal.record_ids() == [record_id]
        journal._connection.execute("PRAGMA query_only=ON")
        with pytest.raises(JournalError):
            journal.append(payload(), ["a"])
        assert journal.record_ids() == [record_id]
    with Journal(path) as journal:
        assert journal.record_ids() == [record_id]


@pytest.mark.parametrize(
    "content", [b"", b"garbage", b"SQLite format 3\x00" + b"\0" * 20]
)
def test_corrupt_or_partial_journal_never_replaced(tmp_path, content):
    path = tmp_path / "journal"
    path.write_bytes(content)
    with pytest.raises(JournalCorruptError):
        Journal(path)
    assert path.read_bytes() == content


def test_unknown_schema_and_broken_accounting_preserved(tmp_path):
    path = tmp_path / "journal"
    Journal(path).close()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    original = path.read_bytes()
    with pytest.raises(JournalCorruptError):
        Journal(path)
    assert path.read_bytes() == original
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=1")
        connection.execute("UPDATE usage SET bytes=100")
    with pytest.raises(JournalCorruptError):
        Journal(path)


def test_codec_freezes_message_exception_stack_and_extra():
    mutable = {"nested": [1]}
    try:
        raise ValueError("original error")
    except ValueError:
        record = logging.LogRecord(
            "app",
            40,
            "caller.py",
            123,
            "value %s",
            ({"x": 1},),
            sys.exc_info(),
            "caller",
            "stack at caller",
        )
        record.extra = mutable
        encoded = RecordCodec().encode(record)
    mutable["nested"].append(2)
    restored = RecordCodec().decode(encoded)
    assert restored.getMessage() == "value {'x': 1}"
    assert "ValueError: original error" in restored.exc_text
    assert restored.exc_info is None
    assert restored.stack_info == "stack at caller"
    assert (restored.pathname, restored.lineno, restored.funcName) == (
        "caller.py",
        123,
        "caller",
    )
    assert restored.extra == {"nested": [1]}


@pytest.mark.parametrize("extra", [object(), (1, 2), {1: "bad"}, float("nan")])
def test_extra_requires_explicit_converter(extra):
    record = logging.LogRecord("app", 20, "x", 1, "hello", (), None)
    record.extra = extra
    with pytest.raises(RecordEncodingError):
        RecordCodec().encode(record)
    codec = RecordCodec(lambda value: str(value))
    assert isinstance(codec.decode(codec.encode(record)).extra, str)
    with pytest.raises(RecordEncodingError):
        RecordCodec(lambda value: value).encode(record)


def test_decode_corruption_no_factories_executed(monkeypatch):
    monkeypatch.setattr(
        logging, "_logRecordFactory", lambda *a: pytest.fail("factory")
    )
    assert RecordCodec().decode(payload()).getMessage() == "hello"
    for encoded in ("{}", "not json", '{"version":99,"fields":{}}'):
        with pytest.raises(JournalCorruptError):
            RecordCodec().decode(encoded)
    data = json.loads(payload())
    data["fields"]["exc_info"] = "unsafe"
    with pytest.raises(JournalCorruptError):
        RecordCodec().decode(json.dumps(data))


def test_reconfiguration_routes_only_move_explicitly(tmp_path):
    with Journal(tmp_path / "journal") as journal:
        record_id = journal.append(payload(), ["old"])
        assert journal.pending() == {"old": 1}
        lease = journal.acquire("old")
        with pytest.raises(OwnershipError):
            journal.reassign("old", "new")
        lease.close()
        journal.reassign("old", "new")
        assert journal.pending() == {"new": 1}
        with journal.acquire("new") as lease:
            assert journal.claim(lease)[0].record_id == record_id
            journal.acknowledge(lease, [record_id])


def test_configuration_validation_and_closed_journal(tmp_path):
    with pytest.raises(ValueError):
        Journal(tmp_path / "j", quota_bytes=0)
    with pytest.raises(ValueError):
        Journal(tmp_path / "j", max_database_bytes=0)
    (tmp_path / "file").write_text("not a directory")
    with pytest.raises(JournalError):
        Journal(tmp_path / "file" / "j")
    journal = Journal(tmp_path / "j")
    with pytest.raises(ValueError):
        journal.append(payload(), [])
    with pytest.raises(ValueError):
        journal.append(payload(), [""])
    with journal.acquire("a") as lease:
        with pytest.raises(ValueError):
            journal.claim(lease, 0)
    journal.close()
    journal.close()
    with pytest.raises(JournalError):
        journal.append(payload(), ["a"])
    with pytest.raises(ValueError):
        default_journal_path("../escape")
    assert str(default_journal_path("example")).endswith(
        "Application Support/aiologging/example/journal.sqlite3"
    )


def test_quota_configuration_shared_on_reopen(tmp_path):
    path = tmp_path / "journal"
    encoded = payload()
    quota = len(encoded.encode()) + 65
    with Journal(path, quota_bytes=quota) as first:
        first.append(encoded, ["a"])
        with Journal(path) as second:
            assert second.quota_bytes == quota
            with pytest.raises(JournalQuotaError):
                second.append(encoded, ["a"])
        with pytest.raises(JournalError, match="limits differ"):
            Journal(path, quota_bytes=quota * 2)


def test_explicit_reassignment_moves_saved_handler_snapshots_and_accounting(
    tmp_path,
):
    codec = RecordCodec()
    record = logging.LogRecord("app", 20, "x", 1, "original message", (), None)
    encoded = codec.encode_routes(record, {"old": [codec.encode(record)]})
    path = tmp_path / "journal"
    with Journal(path) as journal:
        record_id = journal.append(encoded, ["old"])
        before = journal.used_bytes
        journal.reassign("old", "longer-new-id")
        assert journal.used_bytes > before
        with journal.acquire("longer-new-id") as lease:
            task = journal.claim(lease)[0]
            assert task.record_id == record_id
            assert (
                codec.decode_delivery(task.payload, "longer-new-id")[
                    0
                ].getMessage()
                == "original message"
            )
    # Accounting remains canonical after the target/snapshot rewrite.
    with Journal(path) as journal:
        with journal.acquire("longer-new-id") as lease:
            journal.claim(lease)
            journal.acknowledge(lease, [record_id])
        assert journal.used_bytes == 0


def test_valid_json_bit_change_is_detected_without_deleting_accepted_record(
    tmp_path,
):
    path = tmp_path / "journal"
    with Journal(path) as journal:
        record_id = journal.append(payload("original"), ["a"])
    # Valid JSON and a structurally valid SQLite DB are insufficient to
    # detect this change without the persistent record checksum.
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE records SET payload=? WHERE id=?",
            (payload("changed"), record_id),
        )
    original_bytes = path.read_bytes()
    with pytest.raises(JournalCorruptError, match="checksum"):
        Journal(path)
    assert path.read_bytes() == original_bytes
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT id FROM records").fetchall() == [
            (record_id,)
        ]


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE deliveries SET recipient='wrong' WHERE recipient='a'",
        "UPDATE deliveries SET fallback_done=1",
        "DELETE FROM deliveries WHERE recipient='b'",
    ],
)
def test_route_or_fallback_corruption_is_detected_without_loss(
    tmp_path, mutation
):
    path = tmp_path / "journal"
    with Journal(path) as journal:
        record_id = journal.append(payload(), ["a", "b"])
    with sqlite3.connect(path) as connection:
        connection.execute(mutation)
    before = path.read_bytes()
    with pytest.raises(JournalCorruptError, match="routing checksum"):
        Journal(path)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT id FROM records").fetchall() == [
            (record_id,)
        ]


def test_live_quota_change_is_shared_and_never_evicts(tmp_path):
    path = tmp_path / "journal"
    encoded = payload()
    size = len(encoded.encode()) + 65
    with Journal(path, quota_bytes=size) as first, Journal(path) as second:
        record_id = first.append(encoded, ["a"])
        with pytest.raises(JournalQuotaError):
            second.append(encoded, ["a"])
        first.quota_bytes = size * 2
        assert second.quota_bytes == size * 2
        second.append(encoded, ["a"])
        second.quota_bytes = 1
        assert record_id in first.record_ids()
        with pytest.raises(JournalQuotaError):
            first.append(encoded, ["a"])
        with pytest.raises(ValueError):
            first.quota_bytes = 0
