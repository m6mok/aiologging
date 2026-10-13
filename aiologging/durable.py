"""Durable delivery engine, independent of an application's event loop.

The journal thread, event loop and recipient I/O workers start before they
are needed at interpreter exit. No ThreadPoolExecutor is used: its shutdown
hook runs before ordinary atexit callbacks on supported Python versions.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextvars
import inspect
import os
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .journal import (
    Delivery,
    Journal,
    JournalError,
    RecordCodec,
    RecipientLease,
    diagnose,
)

T = TypeVar("T")
_IN_DELIVERY: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "aiologging_durable_delivery", default=False
)


class DeliveryError(JournalError):
    """A mandatory destination has not confirmed delivery."""


class AcceptanceBusyError(JournalError):
    """Bounded admission queue is full; the record was not accepted."""


class RecipientUnavailableError(DeliveryError):
    """The original destination is absent; release ownership, retain work."""


class _Worker:
    """Prestarted daemon worker with futures, available during atexit."""

    def __init__(
        self, name: str, capacity: int = 0, *, delivery_context: bool = True
    ) -> None:
        self._delivery_context = delivery_context
        self._jobs: queue.Queue[
            tuple[Callable[[], Any], concurrent.futures.Future[Any]] | None
        ] = queue.Queue(capacity)
        self._gate = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True
        )
        self._thread.start()

    def submit(
        self, operation: Callable[[], T]
    ) -> concurrent.futures.Future[T]:
        future: concurrent.futures.Future[T] = concurrent.futures.Future()
        with self._gate:
            if self._closed:
                raise JournalError("Background worker is closed")
            try:
                self._jobs.put_nowait((operation, future))
            except queue.Full as error:
                diagnose("admission queue full; record was not accepted")
                raise AcceptanceBusyError("Admission queue is full") from error
        return future

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            operation, future = job
            # A cancelled wait does not cancel durable work already submitted.
            if not future.set_running_or_notify_cancel():
                continue
            token = _IN_DELIVERY.set(self._delivery_context)
            try:
                future.set_result(operation())
            except BaseException as error:
                future.set_exception(error)
            finally:
                _IN_DELIVERY.reset(token)

    def close(self) -> None:
        with self._gate:
            self._closed = True
        # This is called after all submitted work has completed.
        self._jobs.put(None)


@dataclass(frozen=True)
class _Append:
    journal: Journal
    payload: str
    recipients: tuple[str, ...]

    def __call__(self) -> int:
        return self.journal.append(self.payload, self.recipients)


class _JournalWorker(_Worker):
    """Group already queued accepts without delaying a lone producer."""

    def __init__(self, capacity: int, group_size: int) -> None:
        self._group_size = group_size
        super().__init__("aiologging-journal", capacity)

    def _run(self) -> None:
        carried: (
            tuple[Callable[[], Any], concurrent.futures.Future[Any]] | None
        ) = None
        while True:
            job = carried if carried is not None else self._jobs.get()
            carried = None
            if job is None:
                return
            operation, future = job
            group = [job]
            if isinstance(operation, _Append):
                while len(group) < self._group_size:
                    try:
                        following = self._jobs.get_nowait()
                    except queue.Empty:
                        break
                    if following is None:
                        self._jobs.put(None)
                        break
                    if not isinstance(following[0], _Append):
                        carried = following
                        break
                    group.append(following)
            running = [
                (op, f) for op, f in group if f.set_running_or_notify_cancel()
            ]
            if not running:
                continue
            token = _IN_DELIVERY.set(True)
            try:
                if isinstance(operation, _Append):
                    operations = [
                        op for op, _ in running if isinstance(op, _Append)
                    ]
                    results = operation.journal.append_many(
                        [(op.payload, op.recipients) for op in operations]
                    )
                else:
                    results = [operation()]
                for (_, result_future), result in zip(running, results):
                    result_future.set_result(result)
            except BaseException as error:
                for _, result_future in running:
                    result_future.set_exception(error)
            finally:
                _IN_DELIVERY.reset(token)


@dataclass(frozen=True)
class Recipient:
    """One stable destination with an explicit delivery-success contract.

    deliver returns True only on confirmed success, or raises/returns False.
    Coroutine functions run on the engine loop. Blocking functions run on
    a dedicated, already started worker. A callable returning an awaitable
    is also supported; creating loop-bound resources belongs in deliver.

    Fallback delivery is recorded separately. Network adapters must provide
    it explicitly. Time budgets cannot forcibly stop arbitrary blocking
    Python code: its worker retains ownership until the code has returned.
    """

    identity: str
    deliver: Callable[[Sequence[Any]], Any]
    fallback: Callable[[Sequence[Any]], Any] | None = None
    primary_timeout: float = 5.0
    fallback_timeout: float = 5.0
    batch_size: int = 1
    retry_delay: float = 1.0
    close: Callable[[], Any] | None = None

    def __post_init__(self) -> None:
        if not self.identity or "\x00" in self.identity:
            raise ValueError("A stable recipient identity is required")
        if (
            self.primary_timeout <= 0
            or self.fallback_timeout <= 0
            or self.retry_delay < 0
            or self.batch_size < 1
        ):
            raise ValueError("Invalid delivery budget, batch or retry delay")


@dataclass
class _Sender:
    recipient: Recipient
    worker: _Worker
    task: asyncio.Task[None] | None = None
    lease: RecipientLease | None = None
    retire: bool = False
    wake: asyncio.Event | None = None
    retired: list[Recipient] = field(default_factory=list)


class DurableEngine:
    """A complete journal-to-recipient pipeline with explicit acceptance.

    Constructor starts every core resource synchronously. Create it during
    configuration, before logging starts. Async acceptance performs no disk
    I/O on the application loop. Synchronous acceptance blocks its caller
    until commit. Success means persistence, not universal handler delivery.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        quota_bytes: int | None = None,
        max_database_bytes: int | None = None,
        admission_capacity: int = 1024,
        exit_timeout: float = 10.0,
        commit_batch_size: int = 32,
        codec: RecordCodec | None = None,
        _adopt_quota: bool = False,
        _adopt_maximum: bool = False,
    ) -> None:
        if admission_capacity < 1 or exit_timeout < 0 or commit_batch_size < 1:
            raise ValueError("Invalid admission capacity or exit budget")
        self._pid = os.getpid()
        self.codec = codec or RecordCodec()
        self.exit_timeout = exit_timeout
        self._gate = threading.RLock()
        self._closed = False
        self._closing = False
        self._shutdown_future: concurrent.futures.Future[bool] | None = None
        self._senders: dict[str, _Sender] = {}
        self._diagnosed_missing: set[str] = set()
        self._journal_worker = _JournalWorker(
            admission_capacity, commit_batch_size
        )
        try:
            self.journal = self._journal_worker.submit(
                lambda: Journal(
                    path,
                    quota_bytes=quota_bytes,
                    max_database_bytes=max_database_bytes,
                    _adopt_quota=_adopt_quota,
                    _adopt_maximum=_adopt_maximum,
                )
            ).result()
        except BaseException:
            self._journal_worker.close()
            raise
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()
            self._loop.close()

        self._thread = threading.Thread(
            target=run, name="aiologging-loop", daemon=True
        )
        self._thread.start()
        ready.wait()
        atexit.register(self._at_exit)

    @property
    def closed(self) -> bool:
        """Whether background resources have finished closing."""
        return self._closed

    def _check(self) -> None:
        if os.getpid() != self._pid:
            raise JournalError("Create a new DurableEngine after fork")
        if self._closed or self._closing:
            raise JournalError("DurableEngine is closed or closing")
        if _IN_DELIVERY.get():
            raise DeliveryError(
                "Logging from a delivery callback would recurse into its "
                "journal"
            )

    def register(self, recipient: Recipient, *, replace: bool = False) -> None:
        """Start a recipient before shutdown; reject identity collisions.

        An identity must describe the original target, not just its current
        logger name. Removing/replacing a handler never redirects old tasks.
        """
        with self._gate:
            self._check()
            existing = self._senders.get(recipient.identity)
            if existing is not None:
                if existing.recipient is recipient:
                    return
                if replace:
                    existing.retired.append(existing.recipient)
                    existing.recipient = recipient
                    if existing.wake is not None:
                        self._loop.call_soon_threadsafe(existing.wake.set)
                    return
                raise ValueError("Recipient identity is already registered")
            sender = _Sender(
                recipient, _Worker("aiologging-io-" + recipient.identity)
            )
            self._senders[recipient.identity] = sender
            future = asyncio.run_coroutine_threadsafe(
                self._start_sender(sender), self._loop
            )
            future.result()

    async def _start_sender(self, sender: _Sender) -> None:
        sender.wake = asyncio.Event()
        sender.task = asyncio.create_task(self._send(sender))

    def accept_sync(self, payload: str, recipients: Sequence[str]) -> int:
        """Wait until commit in a synchronous producer's calling thread."""
        with self._gate:
            self._check()
            future = self._journal_worker.submit(
                _Append(self.journal, payload, tuple(recipients))
            )
        return future.result()

    async def accept(self, payload: str, recipients: Sequence[str]) -> int:
        """Await durable acceptance without blocking the application loop."""
        with self._gate:
            self._check()
            future = self._journal_worker.submit(
                _Append(self.journal, payload, tuple(recipients))
            )
        # Shield retains the commit future if the application cancels its wait.
        return await asyncio.shield(asyncio.wrap_future(future))

    async def _journal(self, operation: Callable[[], T]) -> T:
        # Internal work shares the journal actor; it never blocks the I/O loop.
        while True:
            try:
                future = self._journal_worker.submit(operation)
                break
            except AcceptanceBusyError:
                await asyncio.sleep(0.01)
        return await asyncio.wrap_future(future)

    async def _invoke(
        self,
        sender: _Sender,
        function: Callable[..., Any],
        *args: Any,
    ) -> Any:
        token = _IN_DELIVERY.set(True)
        try:
            if inspect.iscoroutinefunction(function):
                return await function(*args)
            result = await asyncio.wrap_future(
                sender.worker.submit(lambda: function(*args))
            )
            if inspect.isawaitable(result):
                return await result
            return result
        finally:
            _IN_DELIVERY.reset(token)

    async def _attempt(
        self,
        sender: _Sender,
        function: Callable[..., Any],
        deliveries: Sequence[Delivery],
        timeout: float,
    ) -> bool | None:
        records = [
            record
            for delivery in deliveries
            for record in self.codec.decode_delivery(
                delivery.payload, delivery.recipient
            )
        ]
        operation = asyncio.create_task(
            self._invoke(sender, function, records)
        )
        try:
            done, _ = await asyncio.wait([operation], timeout=timeout)
            if not done:
                diagnose(
                    f"delivery budget exceeded for {sender.recipient.identity}"
                )
                if inspect.iscoroutinefunction(function):
                    operation.cancel()
                # Await even a timed-out blocking call before releasing its OS
                # lock. Otherwise another process could send concurrently.
                try:
                    await operation
                except (Exception, asyncio.CancelledError):
                    pass
                return False
            result = operation.result()
            if result is not True:
                raise DeliveryError("Recipient did not confirm success")
            return True
        except RecipientUnavailableError:
            diagnose(
                f"recipient {sender.recipient.identity} is not configured; "
                f"tasks retained"
            )
            return None
        except Exception as error:
            diagnose(
                f"delivery failed for {sender.recipient.identity}: "
                f"{type(error).__name__}"
            )
            return False
        except asyncio.CancelledError:
            # Shutdown must not unlock while a blocking operation
            # is still live.
            try:
                await asyncio.shield(operation)
            except (Exception, asyncio.CancelledError):
                pass
            raise

    async def _pause(self, sender: _Sender, delay: float) -> None:
        if sender.retire:
            return
        assert sender.wake is not None
        sender.wake.clear()
        try:
            await asyncio.wait_for(sender.wake.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    async def _send(self, sender: _Sender) -> None:
        recipient = sender.recipient
        try:
            while not sender.retire:
                await self._close_retired(sender)
                recipient = sender.recipient
                if sender.lease is None:
                    sender.lease = await self._journal(
                        lambda: self.journal.acquire(recipient.identity)
                    )
                    if sender.lease is None:
                        await self._pause(sender, 0.05)
                        continue
                lease = sender.lease
                deliveries = await self._journal(
                    lambda: self.journal.claim(lease, recipient.batch_size)
                )
                if not deliveries:
                    await self._pause(sender, 0.02)
                    continue
                delivered = await self._attempt(
                    sender,
                    recipient.deliver,
                    deliveries,
                    recipient.primary_timeout,
                )
                if delivered is None:
                    lease.close()
                    sender.lease = None
                    await self._pause(sender, recipient.retry_delay)
                    continue
                if delivered:
                    await self._journal(
                        lambda: self.journal.acknowledge(
                            lease,
                            [delivery.record_id for delivery in deliveries],
                        )
                    )
                else:
                    if recipient.fallback is not None:
                        await self._fallback_pass(sender, lease, recipient)
                    await self._pause(sender, recipient.retry_delay)
        except Exception as error:
            diagnose(
                f"sender stopped for {recipient.identity}; tasks retained: "
                f"{type(error).__name__}"
            )
        finally:
            if sender.lease is not None:
                sender.lease.close()
                sender.lease = None

    async def _close_retired(self, sender: _Sender) -> None:
        """Close replaced resources after their previous I/O has finished."""
        while sender.retired:
            previous = sender.retired.pop(0)
            if previous.close is not None:
                try:
                    await self._invoke(sender, previous.close)
                except Exception as error:
                    diagnose(
                        f"retired recipient close failed: {type(error).__name__}"
                    )

    async def _fallback_pass(
        self, sender: _Sender, lease: RecipientLease, recipient: Recipient
    ) -> None:
        """An unavailable primary head must not starve later fallback tasks."""
        assert recipient.fallback is not None
        maximum = await self._journal(
            lambda: self.journal.high_watermark(lease)
        )
        deadline = time.monotonic() + recipient.fallback_timeout
        while time.monotonic() < deadline and not sender.retire:
            pending = await self._journal(
                lambda: self.journal.claim(
                    lease, recipient.batch_size, fallback=True, through=maximum
                )
            )
            if not pending:
                return
            budget = deadline - time.monotonic()
            if budget <= 0 or not await self._attempt(
                sender, recipient.fallback, pending, budget
            ):
                return
            await self._journal(
                lambda: self.journal.acknowledge(
                    lease,
                    [delivery.record_id for delivery in pending],
                    fallback=True,
                )
            )

    async def pending(self) -> dict[str, int]:
        """Inspect every pending route, even without a matching sender."""
        self._check()
        return dict(
            await asyncio.wrap_future(
                self._journal_worker.submit(self.journal.pending)
            )
        )

    async def _drain(self, timeout: float | None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            pending = await self._journal(self.journal.pending)
            if not pending:
                return True
            missing = set(pending) - self._senders.keys()
            for identity in missing - self._diagnosed_missing:
                diagnose(
                    f"recipient {identity} is not configured; tasks retained"
                )
            self._diagnosed_missing.update(missing)
            if missing or (
                deadline is not None and time.monotonic() >= deadline
            ):
                return False
            await asyncio.sleep(0.02)

    async def flush(self, timeout: float | None = None) -> None:
        """Wait for primary ACKs. Fallback does not make flush successful."""
        self._check()
        complete = await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(self._drain(timeout), self._loop)
        )
        if not complete:
            raise DeliveryError(
                "Unfinished primary deliveries remain in journal"
            )

    def flush_sync(self, timeout: float = 10.0) -> bool:
        """Drain on the existing loop, also after asyncio.run exits."""
        self._check()
        return asyncio.run_coroutine_threadsafe(
            self._drain(timeout), self._loop
        ).result()

    async def _stop(self, timeout: float) -> bool:
        complete = await self._drain(timeout)
        for sender in self._senders.values():
            sender.retire = True
            if sender.wake is not None:
                sender.wake.set()
        finishing = asyncio.create_task(self._finish_stop())
        done, pending = await asyncio.wait([finishing], timeout=0.1)
        if pending:
            # Live workers keep their OS locks and finish closing resources
            # later. No timeout lets a second process steal their tasks.
            diagnose("delivery still running; journal retained")
            return False
        finishing.result()
        return complete

    async def _finish_stop(self) -> None:
        tasks = [s.task for s in self._senders.values() if s.task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for sender in self._senders.values():
            await self._close_retired(sender)
            if sender.recipient.close is not None:
                try:
                    await self._invoke(sender, sender.recipient.close)
                except Exception as error:
                    diagnose(f"recipient close failed: {type(error).__name__}")
            sender.worker.close()
        await self._journal(self.journal.close)
        self._journal_worker.close()
        self._closed = True
        atexit.unregister(self._at_exit)
        if self._shutdown_future is not None and self._shutdown_future.done():
            self._loop.call_soon(self._loop.stop)

    def _submit_stop(self, budget: float) -> concurrent.futures.Future[bool]:
        future = asyncio.run_coroutine_threadsafe(
            self._stop(budget), self._loop
        )
        self._shutdown_future = future

        def stop_loop(completed: concurrent.futures.Future[bool]) -> None:
            if self.closed:
                self._loop.call_soon_threadsafe(self._loop.stop)

        future.add_done_callback(stop_loop)
        return future

    def shutdown_sync(self, timeout: float | None = None) -> bool:
        """Stop without starting threads. Return whether all tasks ACKed."""
        if self._closed or os.getpid() != self._pid:
            return self._closed
        with self._gate:
            if self._closing:
                return False
            self._closing = True
        budget = self.exit_timeout if timeout is None else timeout
        future = self._submit_stop(budget)
        try:
            complete = future.result(timeout=budget + 1.0)
            if self.closed:
                self._thread.join(1.0)
                atexit.unregister(self._at_exit)
            if not complete:
                diagnose("shutdown left unfinished primary tasks in journal")
            return complete
        except concurrent.futures.TimeoutError:
            diagnose("shutdown budget exhausted; tasks retained in journal")
            return False

    async def shutdown(self, timeout: float | None = None) -> bool:
        """Async shutdown keeps the application loop responsive."""
        if self._closed:
            return True
        with self._gate:
            self._check()
            self._closing = True
        budget = self.exit_timeout if timeout is None else timeout
        complete = await asyncio.wrap_future(self._submit_stop(budget))
        if self.closed:
            atexit.unregister(self._at_exit)
        if not complete:
            diagnose("shutdown left unfinished primary tasks in journal")
        return complete

    def _at_exit(self) -> None:
        # In a forked child, never wait for the parent's vanished threads.
        if os.getpid() == self._pid:
            self.shutdown_sync()
