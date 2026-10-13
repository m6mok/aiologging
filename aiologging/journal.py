"""Durable, shared delivery journal. No network or third-party imports.

SQLite transactions define acceptance order. Recipient locks have no expiry:
only closing the lock or process death allows another sender to take over.
This module deliberately does not perform handler I/O.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import threading
import uuid
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import AiologgingError

_SCHEMA_VERSION = 1
_APPLICATION_ID = 0x41494F4C
_SCHEMA = """
CREATE TABLE records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL,
    checksum TEXT NOT NULL CHECK(length(checksum) = 64),
    routing_checksum TEXT NOT NULL CHECK(length(routing_checksum) = 64),
    size INTEGER NOT NULL CHECK(size >= 0)
);
CREATE TABLE deliveries (
    record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    recipient TEXT NOT NULL,
    claim TEXT,
    fallback_done INTEGER NOT NULL DEFAULT 0 CHECK(fallback_done IN (0, 1)),
    PRIMARY KEY(record_id, recipient)
);
CREATE INDEX recipient_order ON deliveries(recipient, record_id);
CREATE TABLE limits (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    quota_bytes INTEGER,
    max_database_bytes INTEGER
);
CREATE TABLE usage (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    bytes INTEGER NOT NULL CHECK(bytes >= 0)
);
INSERT INTO usage VALUES (1, 0);
CREATE TRIGGER record_removed AFTER DELETE ON records BEGIN
    UPDATE usage SET bytes = bytes - OLD.size WHERE singleton = 1;
END;
"""


class JournalError(AiologgingError):
    """Acceptance or journal operation failed; never a successful write."""


class JournalCorruptError(JournalError):
    """An existing journal needs operator attention; it is preserved."""


class JournalQuotaError(JournalError):
    """The configured limit prevents acceptance of another record."""


class RecordEncodingError(JournalError):
    """A record cannot be represented by the recovery codec."""


class OwnershipError(JournalError):
    """A closed, inherited or foreign recipient lock was used."""


def _record_checksum(record_id: int, payload: str) -> str:
    return hashlib.sha256(
        str(record_id).encode() + b"\0" + payload.encode()
    ).hexdigest()


ExtraEncoder = Callable[[Any], Any]
_formatted_records: weakref.WeakKeyDictionary[
    logging.LogRecord, tuple[str, str]
] = weakref.WeakKeyDictionary()
_formatted_lock = threading.RLock()
_handler_tokens: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}


def bind_handler(handler: logging.Handler, recipient: str) -> None:
    """Bind rendering outside user objects, including unhashable handlers."""
    key = id(handler)

    def remove(reference: weakref.ReferenceType[Any]) -> None:
        with _formatted_lock:
            current = _handler_tokens.get(key)
            if current is not None and current[0] is reference:
                _handler_tokens.pop(key, None)

    with _formatted_lock:
        _handler_tokens[key] = (weakref.ref(handler, remove), recipient)


def handler_recipient(handler: logging.Handler) -> str | None:
    with _formatted_lock:
        current = _handler_tokens.get(id(handler))
        return (
            current[1]
            if current is not None and current[0]() is handler
            else None
        )


def prepared_text(
    record: logging.LogRecord, recipient: str | None
) -> str | None:
    """Read internal rendering metadata without occupying user extra names."""
    if type(record) is not logging.LogRecord:
        return None
    with _formatted_lock:
        prepared = _formatted_records.get(record)
        return (
            prepared[1]
            if prepared is not None and prepared[0] == recipient
            else None
        )


def _routing_checksum(connection: sqlite3.Connection, record_id: int) -> str:
    routes = connection.execute(
        "SELECT recipient, fallback_done FROM deliveries "
        "WHERE record_id=? ORDER BY recipient",
        (record_id,),
    ).fetchall()
    return hashlib.sha256(
        json.dumps(routes, separators=(",", ":")).encode()
    ).hexdigest()


def diagnose(message: str) -> None:
    """Report failures without feeding them back into logging."""
    try:
        sys.stderr.write(f"aiologging: {message}\n")
        sys.stderr.flush()
    except Exception:
        # Preserve the original exception even if stderr is unavailable.
        pass


def default_journal_path(application: str) -> Path:
    """Return the macOS persistent path for an explicit application name."""
    if (
        not application
        or application in (".", "..")
        or any(c in application for c in ("/", "\\", "\x00"))
    ):
        raise ValueError("application must be a nonempty directory name")
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "aiologging"
        / application
        / "journal.sqlite3"
    )


def _json_value(value: Any, encoder: ExtraEncoder | None) -> Any:
    """Require lossless JSON values instead of silently stringifying extra."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_json_value(item, encoder) for item in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _json_value(v, encoder) for k, v in value.items()}
    if encoder is not None:
        # Converter results must themselves be serializable. Do not recurse
        # into the converter forever if it returns the original object.
        return _json_value(encoder(value), None)
    raise RecordEncodingError(
        f"Unsupported record value {type(value).__name__}; "
        "configure an extra_encoder"
    )


class RecordCodec:
    """Freeze message, exception and caller fields into versioned JSON."""

    def __init__(self, extra_encoder: ExtraEncoder | None = None) -> None:
        self.extra_encoder = extra_encoder

    def encode(
        self, record: logging.LogRecord, *, ready_message: str | None = None
    ) -> str:
        """Encode at the call site, before arguments or extra can change."""
        try:
            fields = dict(record.__dict__)
            fields["msg"] = (
                record.getMessage() if ready_message is None else ready_message
            )
            fields["message"] = fields["msg"]
            fields["args"] = None
            if record.exc_info and not fields.get("exc_text"):
                fields["exc_text"] = logging.Formatter().formatException(
                    record.exc_info
                )
            fields["exc_info"] = None
            return json.dumps(
                {
                    "version": 1,
                    "fields": _json_value(fields, self.extra_encoder),
                },
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        except Exception as error:
            if isinstance(error, RecordEncodingError):
                raise
            raise RecordEncodingError("Cannot encode log record") from error

    def encode_formatted(
        self, record: logging.LogRecord, recipient: str, text: str
    ) -> str:
        """Freeze final text while preserving all record fields."""
        if not isinstance(text, str):
            raise RecordEncodingError("Formatter must return a string")
        data = json.loads(
            self.encode(record, ready_message=getattr(record, "message", None))
        )
        data["formatted"] = [recipient, text]
        return json.dumps(data, ensure_ascii=True, separators=(",", ":"))

    def encode_routes(
        self,
        record: logging.LogRecord,
        routes: Mapping[str, Sequence[str]],
    ) -> str:
        """Persist per-handler filter results and repeated handler visits."""
        data = json.loads(self.encode(record))
        data["routes"] = {
            recipient: [json.loads(snapshot) for snapshot in snapshots]
            for recipient, snapshots in routes.items()
        }
        return json.dumps(data, ensure_ascii=True, separators=(",", ":"))

    def decode_delivery(
        self, payload: str, recipient: str
    ) -> list[logging.LogRecord]:
        """Restore the records selected for an original handler."""
        self.decode(payload)
        data = json.loads(payload)
        if "routes" not in data:
            return [self.decode(payload)]
        try:
            snapshots = data["routes"][recipient]
            if not isinstance(snapshots, list) or not snapshots:
                raise ValueError("invalid route snapshots")
            return [
                self.decode(json.dumps(snapshot)) for snapshot in snapshots
            ]
        except (KeyError, ValueError, TypeError) as error:
            raise JournalCorruptError(
                "Invalid stored destination snapshot"
            ) from error

    def decode(self, payload: str) -> logging.LogRecord:
        """Restore data without executing factories or unpickling objects."""
        try:
            data = json.loads(payload)
            if data["version"] != 1 or not isinstance(data["fields"], dict):
                raise ValueError("unsupported record schema")
            fields = data["fields"]
            if (
                not isinstance(fields["name"], str)
                or not isinstance(fields["msg"], str)
                or not isinstance(fields["levelno"], int)
                or fields["args"] is not None
                or fields["exc_info"] is not None
            ):
                raise ValueError("invalid record fields")
            record = logging.LogRecord("", 0, "", 0, "", (), None)
            record.__dict__.update(fields)
            if "formatted" in data:
                rendering = data["formatted"]
                if (
                    not isinstance(rendering, list)
                    or len(rendering) != 2
                    or not all(isinstance(value, str) for value in rendering)
                ):
                    raise ValueError("invalid rendering metadata")
                with _formatted_lock:
                    _formatted_records[record] = (rendering[0], rendering[1])
            return record
        except (ValueError, TypeError, KeyError) as error:
            raise JournalCorruptError("Invalid stored record") from error


@dataclass(frozen=True)
class Delivery:
    """An ordered task whose primary and fallback ACKs are independent."""

    record_id: int
    recipient: str
    payload: str
    fallback_done: bool


class RecipientLease:
    """An OS lock and fencing token, owned by one Journal and process."""

    def __init__(self, journal: Journal, recipient: str, fd: int) -> None:
        self.journal = journal
        self.recipient = recipient
        self.token = uuid.uuid4().hex
        self._fd: int | None = fd
        self._pid = os.getpid()
        self._lock = threading.RLock()

    def close(self) -> None:
        """Release ownership. Never unlink a live lock inode."""
        with self._lock:
            if self._fd is not None:
                # An inherited descriptor must not unlock the parent's lock.
                if self._pid == os.getpid():
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> RecipientLease:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class Journal:
    """SQLite FULL-synchronous journal with independent recipient ACKs.

    Args:
        path: A local disk path shared by processes of one application.
        quota_bytes: Maximum encoded bytes of unfinished records and routes.
            None requires the operator to manage capacity. No unmeasured
            default limit is imposed.
        max_database_bytes: Optional SQLite page limit. The transient rollback
            journal and filesystem metadata also need disk space.
        timeout: Maximum SQLite lock wait; errors always reach the caller.

    The logical quota is transactional across processes. Fully acknowledged
    records are removed; unfinished records are never evicted. Connections
    cannot be inherited across fork. Use a new Journal in the child.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        quota_bytes: int | None = None,
        max_database_bytes: int | None = None,
        timeout: float = 30.0,
        _adopt_quota: bool = False,
        _adopt_maximum: bool = False,
    ) -> None:
        if quota_bytes is not None and quota_bytes <= 0:
            raise ValueError("quota_bytes must be positive")
        if max_database_bytes is not None and max_database_bytes <= 0:
            raise ValueError("max_database_bytes must be positive")
        self.path = Path(path).absolute()
        self._quota_bytes = quota_bytes
        self.max_database_bytes = max_database_bytes
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._closed = False
        self._connection: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            lock_fd = os.open(
                str(self.path) + ".init.lock", os.O_CREAT | os.O_RDWR, 0o600
            )
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                if not self.path.exists():
                    self._initialize(timeout)
                elif self.path.stat().st_size == 0:
                    raise JournalCorruptError("Existing journal is empty")
                # mode=rw prevents SQLite from silently replacing a missing DB.
                self._connection = sqlite3.connect(
                    self.path.as_uri() + "?mode=rw",
                    uri=True,
                    timeout=timeout,
                    isolation_level=None,
                    check_same_thread=False,
                )
                self._configure(self._connection)
                # Every integrity/accounting read must use the same snapshot:
                # other processes may keep committing during our open.
                self._connection.execute("BEGIN")
                try:
                    self._validate(self._connection)
                finally:
                    self._connection.execute("ROLLBACK")
                stored_quota, stored_maximum = self._connection.execute(
                    "SELECT quota_bytes, max_database_bytes FROM limits"
                ).fetchone()
                if (
                    not _adopt_quota
                    and quota_bytes is not None
                    and quota_bytes != stored_quota
                    or not _adopt_maximum
                    and max_database_bytes is not None
                    and max_database_bytes != stored_maximum
                ):
                    raise JournalError(
                        "Journal limits differ between processes"
                    )
                self._quota_bytes = stored_quota
                max_database_bytes = stored_maximum
                if max_database_bytes is not None:
                    page_size = self._connection.execute(
                        "PRAGMA page_size"
                    ).fetchone()[0]
                    pages = max_database_bytes // page_size
                    current = self._connection.execute(
                        "PRAGMA page_count"
                    ).fetchone()[0]
                    if pages < current:
                        raise JournalQuotaError(
                            "Database already exceeds max_database_bytes"
                        )
                    self._connection.execute(f"PRAGMA max_page_count={pages}")
            finally:
                os.close(lock_fd)
        except (OSError, sqlite3.Error, JournalError) as error:
            if self._connection is not None:
                self._connection.close()
            self._closed = True
            diagnose(f"journal open failed: {error}")
            if isinstance(error, JournalError):
                raise
            if isinstance(error, sqlite3.DatabaseError) and getattr(
                error, "sqlite_errorcode", None
            ) in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB):
                raise JournalCorruptError(
                    "Cannot open corrupt journal"
                ) from error
            raise JournalError("Cannot open journal") from error

    @staticmethod
    def _configure(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA foreign_keys=ON")
        # DELETE avoids an unbounded WAL when a reader holds a snapshot.
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA fullfsync=ON")

    def _initialize(self, timeout: float) -> None:
        temporary = self.path.with_name(
            self.path.name + "." + uuid.uuid4().hex
        )
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(fd)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(temporary, timeout=timeout)
            self._configure(connection)
            connection.executescript(
                "BEGIN IMMEDIATE;\n"
                + _SCHEMA
                + "INSERT INTO limits VALUES (1, "
                + (
                    str(self.quota_bytes)
                    if self.quota_bytes is not None
                    else "NULL"
                )
                + ", "
                + (
                    str(self.max_database_bytes)
                    if self.max_database_bytes is not None
                    else "NULL"
                )
                + ");\n"
                + f"PRAGMA application_id={_APPLICATION_ID};\n"
                + f"PRAGMA user_version={_SCHEMA_VERSION};\nCOMMIT;"
            )
            connection.close()
            connection = None
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if connection is not None:
                connection.close()
            # An unpublished initialization file has no accepted records.
            temporary.unlink(missing_ok=True)
            Path(str(temporary) + "-journal").unlink(missing_ok=True)

    @staticmethod
    def _validate(connection: sqlite3.Connection) -> None:
        if (
            connection.execute("PRAGMA application_id").fetchone()[0]
            != _APPLICATION_ID
            or connection.execute("PRAGMA user_version").fetchone()[0]
            != _SCHEMA_VERSION
        ):
            raise JournalCorruptError("Unknown journal schema; file preserved")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise JournalCorruptError("Journal integrity check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchone():
            raise JournalCorruptError("Journal contains orphan deliveries")
        for record_id, payload, checksum, routing in connection.execute(
            "SELECT id, payload, checksum, routing_checksum FROM records"
        ):
            if checksum != _record_checksum(record_id, payload):
                raise JournalCorruptError("Stored record checksum mismatch")
            if routing != _routing_checksum(connection, record_id):
                raise JournalCorruptError("Stored routing checksum mismatch")
        usage = connection.execute("SELECT bytes FROM usage").fetchall()
        actual = connection.execute(
            "SELECT COALESCE(SUM(size), 0) FROM records"
        ).fetchone()[0]
        limits = connection.execute(
            "SELECT singleton, quota_bytes, max_database_bytes FROM limits"
        ).fetchall()
        if len(limits) != 1 or limits[0][0] != 1:
            raise JournalCorruptError("Invalid journal limits")
        if usage != [(actual,)]:
            raise JournalCorruptError("Journal accounting mismatch")
        if connection.execute(
            "SELECT id FROM records WHERE NOT EXISTS "
            "(SELECT 1 FROM deliveries WHERE record_id=records.id) LIMIT 1"
        ).fetchone():
            raise JournalCorruptError("Journal contains unrouted records")

    def _check_open(self) -> sqlite3.Connection:
        if self._closed or self._connection is None:
            raise JournalError("Journal is closed")
        if os.getpid() != self._pid:
            raise OwnershipError("Create a new Journal after fork")
        return self._connection

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._check_open()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.execute("COMMIT")
            except BaseException as error:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                if isinstance(error, sqlite3.Error):
                    diagnose(f"journal transaction failed: {error}")
                    if (
                        getattr(error, "sqlite_errorcode", None)
                        == sqlite3.SQLITE_FULL
                    ):
                        raise JournalQuotaError(
                            "SQLite storage is full"
                        ) from error
                    raise JournalError("Journal transaction failed") from error
                raise

    def append(self, payload: str, recipients: Sequence[str]) -> int:
        """Return an ID only after the record and every route are committed."""
        return self.append_many([(payload, recipients)])[0]

    def append_many(
        self, records: Sequence[tuple[str, Sequence[str]]]
    ) -> list[int]:
        """Group commit: either all submitted records are accepted or none."""
        prepared: list[tuple[str, tuple[str, ...], int]] = []
        for payload, recipients in records:
            RecordCodec().decode(payload)
            destinations = tuple(dict.fromkeys(recipients))
            if not destinations or any(
                not isinstance(r, str) or not r or "\x00" in r
                for r in destinations
            ):
                raise ValueError("Each record requires nonempty recipient IDs")
            for recipient in destinations:
                RecordCodec().decode_delivery(payload, recipient)
            size = len(payload.encode("utf-8")) + sum(
                len(r.encode("utf-8")) + 64 for r in destinations
            )
            prepared.append((payload, destinations, size))
        ids: list[int] = []
        with self._transaction() as connection:
            used = connection.execute("SELECT bytes FROM usage").fetchone()[0]
            total = sum(size for _, _, size in prepared)
            if (
                self.quota_bytes is not None
                and used + total > self.quota_bytes
            ):
                diagnose("journal quota exceeded; record was not accepted")
                raise JournalQuotaError(
                    "Unfinished records exceed quota_bytes"
                )
            for payload, recipients, size in prepared:
                cursor = connection.execute(
                    "INSERT INTO records(payload, size, checksum, "
                    "routing_checksum) VALUES (?, ?, ?, ?)",
                    (payload, size, "0" * 64, "0" * 64),
                )
                record_id = cursor.lastrowid
                assert record_id is not None
                connection.execute(
                    "UPDATE records SET checksum=? WHERE id=?",
                    (_record_checksum(record_id, payload), record_id),
                )
                ids.append(record_id)
                connection.executemany(
                    "INSERT INTO deliveries(record_id, recipient) VALUES "
                    "(?, ?)",
                    [(record_id, recipient) for recipient in recipients],
                )
                connection.execute(
                    "UPDATE records SET routing_checksum=? WHERE id=?",
                    (_routing_checksum(connection, record_id), record_id),
                )
            connection.execute("UPDATE usage SET bytes=bytes+?", (total,))
        return ids

    def acquire(self, recipient: str) -> RecipientLease | None:
        """Try to become the only live sender for this destination."""
        with self._lock:
            self._check_open()
            digest = hashlib.sha256(recipient.encode()).hexdigest()
            fd = os.open(
                self.path.with_name(self.path.name + "." + digest + ".lock"),
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                return None
            except BaseException:
                os.close(fd)
                raise
            return RecipientLease(self, recipient, fd)

    @contextlib.contextmanager
    def _owned(self, lease: RecipientLease) -> Iterator[sqlite3.Connection]:
        with lease._lock:
            if (
                lease.journal is not self
                or lease._fd is None
                or lease._pid != os.getpid()
            ):
                raise OwnershipError("Recipient ownership is no longer valid")
            with self._transaction() as connection:
                yield connection

    def claim(
        self,
        lease: RecipientLease,
        limit: int = 1,
        *,
        fallback: bool = False,
        through: int | None = None,
    ) -> list[Delivery]:
        """Claim the oldest unfinished tasks, recovering any dead owner's work.

        The caller keeps the lease across I/O and ACK. Retry the first failed
        batch before claiming a later one to preserve order of attempts.
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._owned(lease) as connection:
            selection = " AND d.fallback_done=0" if fallback else ""
            maximum = through if through is not None else 9223372036854775807
            rows = connection.execute(
                "SELECT r.id, r.payload, d.fallback_done, r.checksum, "
                "r.routing_checksum FROM records r "
                "JOIN deliveries d ON r.id=d.record_id "
                "WHERE d.recipient=? AND r.id<=?"
                + selection
                + " ORDER BY r.id LIMIT ?",
                (lease.recipient, maximum, limit),
            ).fetchall()
            for row in rows:
                if row[3] != _record_checksum(row[0], row[1]):
                    raise JournalCorruptError(
                        "Stored record checksum mismatch"
                    )
                if row[4] != _routing_checksum(connection, row[0]):
                    raise JournalCorruptError(
                        "Stored routing checksum mismatch"
                    )
            connection.executemany(
                "UPDATE deliveries SET claim=? WHERE record_id=? AND "
                "recipient=?",
                [(lease.token, row[0], lease.recipient) for row in rows],
            )
        return [
            Delivery(row[0], lease.recipient, row[1], bool(row[2]))
            for row in rows
        ]

    def high_watermark(self, lease: RecipientLease) -> int:
        """Bound a fallback pass so incoming records cannot starve primary."""
        with self._owned(lease) as connection:
            maximum: int = connection.execute(
                "SELECT COALESCE(MAX(record_id), 0) FROM deliveries WHERE "
                "recipient=?",
                (lease.recipient,),
            ).fetchone()[0]
            return maximum

    def acknowledge(
        self,
        lease: RecipientLease,
        record_ids: Sequence[int],
        *,
        fallback: bool = False,
    ) -> None:
        """ACK only claimed tasks. Fallback ACK never removes the primary."""
        with self._owned(lease) as connection:
            for record_id in dict.fromkeys(record_ids):
                owned = connection.execute(
                    "SELECT 1 FROM deliveries WHERE record_id=? "
                    "AND recipient=? AND claim=?",
                    (record_id, lease.recipient, lease.token),
                ).fetchone()
                if owned is None:
                    raise OwnershipError("ACK requires the current claim")
                if fallback:
                    connection.execute(
                        "UPDATE deliveries SET fallback_done=1 "
                        "WHERE record_id=? AND recipient=?",
                        (record_id, lease.recipient),
                    )
                else:
                    connection.execute(
                        "DELETE FROM deliveries WHERE record_id=? AND "
                        "recipient=?",
                        (record_id, lease.recipient),
                    )
                    connection.execute(
                        "DELETE FROM records WHERE id=? AND NOT EXISTS "
                        "(SELECT 1 FROM deliveries WHERE record_id=?)",
                        (record_id, record_id),
                    )
                connection.execute(
                    "UPDATE records SET routing_checksum=? WHERE id=?",
                    (_routing_checksum(connection, record_id), record_id),
                )

    def pending(self) -> Mapping[str, int]:
        """Return unfinished counts, including destinations not configured."""
        with self._lock:
            connection = self._check_open()
            return dict(
                connection.execute(
                    "SELECT recipient, COUNT(*) FROM deliveries GROUP BY "
                    "recipient"
                ).fetchall()
            )

    def record_ids(self) -> list[int]:
        """Return IDs retained until every mandatory recipient has ACKed."""
        with self._lock:
            return [
                row[0]
                for row in self._check_open().execute(
                    "SELECT id FROM records ORDER BY id"
                )
            ]

    @property
    def quota_bytes(self) -> int | None:
        """The shared logical limit, including explicit live adjustments."""
        with self._lock:
            if self._connection is None or self._closed:
                return self._quota_bytes
            value: int | None = (
                self._check_open()
                .execute("SELECT quota_bytes FROM limits")
                .fetchone()[0]
            )
            return value

    @quota_bytes.setter
    def quota_bytes(self, value: int | None) -> None:
        """Change the shared quota without deleting any unfinished record."""
        if value is not None and value <= 0:
            raise ValueError("quota_bytes must be positive")
        with self._transaction() as connection:
            connection.execute("UPDATE limits SET quota_bytes=?", (value,))
        self._quota_bytes = value

    @property
    def used_bytes(self) -> int:
        """Return transactional logical usage, not an estimate of file size."""
        with self._lock:
            value: int = (
                self._check_open()
                .execute("SELECT bytes FROM usage")
                .fetchone()[0]
            )
            return value

    def reassign(self, old: str, new: str) -> None:
        """Explicitly move unfinished routes while both senders are stopped.

        Never transfer fallback success: the new destination has a different
        fallback contract. Acquiring both locks prevents concurrent delivery.
        """
        if not old or not new or old == new:
            raise ValueError("Distinct nonempty destination IDs are required")
        with contextlib.ExitStack() as stack:
            leases: dict[str, RecipientLease] = {}
            for name in sorted((old, new)):
                lease = self.acquire(name)
                if lease is None:
                    raise OwnershipError(
                        "Stop both recipients before reassign"
                    )
                leases[name] = stack.enter_context(lease)
            with self._owned(leases[old]) as connection:
                # A uniqueness collision must not delete a required route.
                rows = connection.execute(
                    "SELECT r.id, r.payload FROM records r JOIN deliveries d "
                    "ON r.id=d.record_id WHERE d.recipient=?",
                    (old,),
                ).fetchall()
                updates = []
                for record_id, payload in rows:
                    data = json.loads(payload)
                    if "routes" in data:
                        snapshots = data["routes"].pop(old)
                        for snapshot in snapshots:
                            if "formatted" in snapshot:
                                snapshot["formatted"][0] = new
                        data["routes"][new] = snapshots
                        updated = json.dumps(
                            data, ensure_ascii=True, separators=(",", ":")
                        )
                    else:
                        updated = payload
                    delta = (
                        len(updated.encode())
                        - len(payload.encode())
                        + len(new.encode())
                        - len(old.encode())
                    )
                    updates.append(
                        (
                            updated,
                            delta,
                            _record_checksum(record_id, updated),
                            record_id,
                        )
                    )
                total_delta = sum(update[1] for update in updates)
                used = connection.execute(
                    "SELECT bytes FROM usage"
                ).fetchone()[0]
                if (
                    self.quota_bytes is not None
                    and used + total_delta > self.quota_bytes
                ):
                    raise JournalQuotaError("Reassignment exceeds quota_bytes")
                connection.executemany(
                    "UPDATE records SET payload=?, size=size+?, "
                    "checksum=? WHERE id=?",
                    updates,
                )
                connection.execute(
                    "UPDATE usage SET bytes=bytes+?", (total_delta,)
                )
                connection.execute(
                    "UPDATE deliveries SET recipient=?, claim=NULL, "
                    "fallback_done=0 WHERE recipient=?",
                    (new, old),
                )
                for _, _, _, record_id in updates:
                    connection.execute(
                        "UPDATE records SET routing_checksum=? WHERE id=?",
                        (_routing_checksum(connection, record_id), record_id),
                    )

    def close(self) -> None:
        """Close SQLite after users have stopped their senders."""
        with self._lock:
            if not self._closed:
                self._check_open().close()
                self._closed = True

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
