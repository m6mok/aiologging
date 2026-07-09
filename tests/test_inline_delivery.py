"""
Inline (synchronous) delivery of critical bridged records.

``captureStdlib(inline_level=ERROR)``: the pager message must leave
before the process dies. Records at or above ``inline_level`` are
offered to handlers implementing ``emit_sync`` (Telegram posts via
urllib in a helper thread bounded by the deadline) before they are
queued; a token bucket bounds how long the calling thread may block,
and every failure falls back to the normal queue path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

import aiologging
from aiologging import AsyncTelegramHandler, captureStdlib
from aiologging.bridge import _TokenBucket, StdlibBridgeHandler
from aiologging.logger import _IN_CONSUMER, _INLINE_HANDLED_ATTR
from aiologging.types import BatchConfig

from .test_logger import RecordingHandler


def make_record(
    msg: str = "boom",
    level: int = logging.ERROR,
    name: str = "pager",
) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, None, None)


def make_telegram_handler(
    queue_posts: Optional[List[str]] = None, **kwargs: Any
) -> AsyncTelegramHandler:
    """
    Telegram handler that never flushes on its own and whose queue
    path (async batch flush) goes through an httpx mock transport.
    """

    def respond(request: httpx.Request) -> httpx.Response:
        if queue_posts is not None:
            queue_posts.append(json.loads(request.content)["text"])
        return httpx.Response(200)

    handler = AsyncTelegramHandler(
        token="123456:test-token",
        chat_id="42",
        backend="httpx",
        batch_config=BatchConfig(batch_size=100, flush_interval=3600.0),
        **kwargs,
    )
    handler._create_session = (  # type: ignore[method-assign]
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond))
    )
    return handler


class FakeResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def install_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    requests: List[Dict[str, Any]],
    delay: float = 0.0,
    error: Optional[Exception] = None,
    side_effect: Optional[Callable[[], None]] = None,
) -> None:
    """Replace urllib.request.urlopen with an offline fake."""

    def fake_urlopen(
        request: Any, timeout: Optional[float] = None, context: Any = None
    ) -> FakeResponse:
        if delay:
            time.sleep(delay)
        if side_effect is not None:
            side_effect()
        if error is not None:
            raise error
        requests.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


class TestTelegramEmitSync:
    def test_delivers_and_reports_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        install_urlopen(monkeypatch, requests)
        handler = make_telegram_handler()

        assert handler.emit_sync(make_record("Service crashed"), 5.0)
        assert len(requests) == 1
        assert requests[0]["chat_id"] == "42"
        assert "Service crashed" in requests[0]["text"]

    def test_network_failure_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        install_urlopen(monkeypatch, requests, error=OSError("refused"))
        handler = make_telegram_handler()

        assert handler.emit_sync(make_record(), 5.0) is False
        assert requests == []

    def test_deadline_bounds_blocking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hanging send (e.g. DNS) must not block past the budget."""
        install_urlopen(monkeypatch, [], delay=0.5)
        handler = make_telegram_handler()

        started = time.monotonic()
        assert handler.emit_sync(make_record(), 0.05) is False
        assert time.monotonic() - started < 0.4

    def test_respects_handler_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        install_urlopen(monkeypatch, requests)
        handler = make_telegram_handler(level=logging.ERROR)

        record = make_record("just info", level=logging.INFO)
        assert handler.emit_sync(record, 5.0) is False
        assert requests == []

    def test_long_record_is_chunked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        install_urlopen(monkeypatch, requests)
        handler = make_telegram_handler(max_message_length=10)

        assert handler.emit_sync(make_record("x" * 25), 5.0)
        assert len(requests) == 3

    def test_closed_handler_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        install_urlopen(monkeypatch, requests)
        handler = make_telegram_handler()
        handler._closed = True

        assert handler.emit_sync(make_record(), 5.0) is False
        assert requests == []

    def test_base_handler_does_not_support_it(self) -> None:
        assert RecordingHandler().emit_sync(make_record(), 5.0) is False


class TestTokenBucket:
    def test_burst_then_empty(self) -> None:
        bucket = _TokenBucket(rate_per_minute=60.0, burst=2)
        assert bucket.try_acquire()
        assert bucket.try_acquire()
        assert not bucket.try_acquire()

    def test_refills_over_time(self) -> None:
        bucket = _TokenBucket(rate_per_minute=2.0, burst=1)
        assert bucket.try_acquire()
        assert not bucket.try_acquire()
        bucket._updated -= 30.0  # pretend 30 s passed: +1 token
        assert bucket.try_acquire()
        assert not bucket.try_acquire()


class TestBridgeInline:
    def _setup_pager(
        self, queue_posts: Optional[List[str]] = None, **capture_kwargs: Any
    ) -> AsyncTelegramHandler:
        handler = make_telegram_handler(queue_posts)
        pager = aiologging.getLogger("pager")
        pager.addHandler(handler)
        captureStdlib(True, **capture_kwargs)
        return handler

    def test_error_delivered_inline_without_a_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        install_urlopen(monkeypatch, requests)
        queue_posts: List[str] = []
        self._setup_pager(queue_posts, inline_level=logging.ERROR)

        # sync context, no event loop anywhere — the pager case
        logging.getLogger("pager").error("Service crashed")
        assert len(requests) == 1
        assert "Service crashed" in requests[0]["text"]

        # the queued copy is marked and must not be delivered twice
        asyncio.run(aiologging.flush())
        assert queue_posts == []

    def test_other_handlers_still_get_the_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        install_urlopen(monkeypatch, requests)
        self._setup_pager(inline_level=logging.ERROR)
        recorder = RecordingHandler()
        aiologging.getLogger("pager").addHandler(recorder)

        logging.getLogger("pager").error("Service crashed")
        assert len(requests) == 1

        asyncio.run(aiologging.flush())
        assert recorder.messages == ["Service crashed"]

    def test_below_inline_level_takes_queue_path_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        install_urlopen(monkeypatch, requests)
        queue_posts: List[str] = []
        self._setup_pager(queue_posts, inline_level=logging.ERROR)

        logging.getLogger("pager").warning("just a warning")
        assert requests == []

        asyncio.run(aiologging.flush())
        assert queue_posts == ["just a warning"]

    def test_token_bucket_exhaustion_falls_back_to_queue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        install_urlopen(monkeypatch, requests)
        queue_posts: List[str] = []
        self._setup_pager(
            queue_posts,
            inline_level=logging.ERROR,
            inline_burst=1,
            inline_rate=0.0,
        )

        logging.getLogger("pager").error("first")
        logging.getLogger("pager").error("second")
        assert len(requests) == 1
        assert "first" in requests[0]["text"]

        asyncio.run(aiologging.flush())
        assert queue_posts == ["second"]

    def test_inline_timeout_falls_back_to_queue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_urlopen(monkeypatch, [], delay=0.5)
        queue_posts: List[str] = []
        self._setup_pager(
            queue_posts,
            inline_level=logging.ERROR,
            inline_timeout=0.05,
        )

        started = time.monotonic()
        logging.getLogger("pager").error("slow pipe")
        assert time.monotonic() - started < 0.4

        asyncio.run(aiologging.flush())
        assert queue_posts == ["slow pipe"]

    def test_reentrant_logging_does_not_recurse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        nested = threading.Event()

        def log_from_inside_send() -> None:
            if not nested.is_set():
                nested.set()
                logging.getLogger("pager").error("nested")

        install_urlopen(
            monkeypatch, requests, side_effect=log_from_inside_send
        )
        # a generous bucket so the guard, not the limiter, is tested
        self._setup_pager(
            inline_level=logging.ERROR, inline_burst=5, inline_rate=60.0
        )

        logging.getLogger("pager").error("outer")
        assert nested.is_set()
        assert len(requests) == 1
        assert "outer" in requests[0]["text"]

    def test_records_from_consumer_are_not_inlined(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: List[Dict[str, Any]] = []
        install_urlopen(monkeypatch, requests)
        handler = make_telegram_handler()
        aiologging.getLogger("pager").addHandler(handler)
        bridge = StdlibBridgeHandler(inline_level=logging.ERROR)

        token = _IN_CONSUMER.set(True)
        try:
            bridge.emit(make_record("from handler I/O"))
        finally:
            _IN_CONSUMER.reset(token)
        assert requests == []

    def test_marker_lists_the_inline_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_urlopen(monkeypatch, [])
        handler = make_telegram_handler()
        aiologging.getLogger("pager").addHandler(handler)
        bridge = StdlibBridgeHandler(inline_level=logging.ERROR)

        record = make_record()
        bridge.emit(record)
        assert getattr(record, _INLINE_HANDLED_ATTR) == {handler}

    def test_capture_stdlib_replaces_the_bridge(self) -> None:
        from aiologging import bridge as bridge_module

        captureStdlib(True)
        first = bridge_module._bridge_handler
        assert first is not None and first._inline_level is None

        captureStdlib(True, inline_level=logging.ERROR)
        second = bridge_module._bridge_handler
        assert second is not None
        assert second._inline_level == logging.ERROR

        stdlib_root = logging.getLogger()
        bridges = [
            h
            for h in stdlib_root.handlers
            if isinstance(h, StdlibBridgeHandler)
        ]
        assert bridges == [second]

    def test_basic_config_forwards_inline_level(self) -> None:
        from aiologging import bridge as bridge_module

        aiologging.basicConfig(
            capture_stdlib=True, inline_level=logging.ERROR
        )
        bridge = bridge_module._bridge_handler
        assert bridge is not None
        assert bridge._inline_level == logging.ERROR


if __name__ == "__main__":
    pytest.main([__file__])
