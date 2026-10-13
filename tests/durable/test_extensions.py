"""Real files and offline HTTP transports, including failed primary ACKs."""

from __future__ import annotations

import asyncio
import io
import threading

import httpx
import pytest

from aiologging.durable import DeliveryError
from aiologging.extensions import (
    FileHandler,
    HTTPHandler,
    HTTPTextHandler,
    RotatingFileHandler,
    StreamHandler,
    TelegramHandler,
    TimedRotatingFileHandler,
)
import aiologging as aio


async def test_strict_files_rotation_and_stream(configured, tmp_path):
    engine, logger = configured
    stream = io.StringIO()
    handler = StreamHandler(stream)
    handler.recipient_id = "stream"
    file = FileHandler(tmp_path / "file", encoding="utf-8", delay=True)
    rotation = RotatingFileHandler(
        tmp_path / "rotate", maxBytes=20, backupCount=2
    )
    timed = TimedRotatingFileHandler(
        tmp_path / "timed", when="D", backupCount=1
    )
    logger.handlers = [handler, file, rotation, timed]
    for i in range(3):
        await logger.info("record %d", i)
    await aio.flush(3)
    assert stream.getvalue() == "record 0\nrecord 1\nrecord 2\n"
    assert (tmp_path / "file").read_text() == stream.getvalue()
    assert (tmp_path / "rotate.1").exists()
    assert (tmp_path / "timed").read_text() == stream.getvalue()


async def test_file_failure_retains_primary(configured, tmp_path):
    engine, logger = configured
    handler = FileHandler(tmp_path / "missing" / "file", delay=True)
    logger.handlers = [handler]
    await logger.info("accepted on disk")
    with pytest.raises(DeliveryError):
        await aio.flush(0.05)
    assert len(engine.journal.record_ids()) == 1


async def test_http_owned_loop_fallback_once_and_recovery(configured):
    engine, logger = configured
    calls = []
    ready = threading.Event()
    fallback = []
    factory_loops = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200 if ready.is_set() else 503)

    def factory():
        factory_loops.append(asyncio.get_running_loop())
        return httpx.AsyncClient(transport=httpx.MockTransport(respond))

    def backup(records):
        fallback.extend(r.getMessage() for r in records)
        return True

    handler = HTTPHandler(
        "https://offline.invalid/receive",
        fallback=backup,
        session_factory=factory,
        retry_delay=0.01,
    )
    logger.handlers = [handler]
    await logger.info("network record")
    async with asyncio.timeout(3):
        while len(calls) < 3:
            await asyncio.sleep(0.01)
    assert fallback == ["network record"]
    assert len(engine.journal.record_ids()) == 1
    assert factory_loops == [engine._loop]
    ready.set()
    await aio.flush(3)
    assert fallback == ["network record"]
    assert await engine.pending() == {}


async def test_http_text_primary_budget_and_failed_fallback(configured):
    engine, logger = configured
    attempted = []
    fallback_calls = []

    async def respond(request):
        attempted.append(request)
        await asyncio.sleep(2)
        return httpx.Response(200)

    def fallback(records):
        fallback_calls.append(records[0].getMessage())
        return False

    handler = HTTPTextHandler(
        "https://offline.invalid",
        fallback=fallback,
        session_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ),
        primary_timeout=0.02,
        fallback_timeout=0.02,
        retry_delay=0.01,
    )
    logger.handlers = [handler]
    await logger.info("text")
    with pytest.raises(DeliveryError):
        await aio.flush(0.1)
    assert fallback_calls
    assert attempted[0].content == b"text"
    assert len(engine.journal.record_ids()) == 1
    assert not engine.journal.claim(
        engine._senders[handler.recipient_id].lease
    )[0].fallback_done


async def test_telegram_chunking_ack_validation_and_secret_redaction(
    configured,
):
    engine, logger = configured
    bodies = []

    def respond(request):
        import json

        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    handler = TelegramHandler(
        "TOP_SECRET",
        123,
        fallback=lambda records: True,
        session_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ),
    )
    assert "TOP_SECRET" not in repr(handler)
    logger.handlers = [handler]
    await logger.info("🙂" * 3000)
    await aio.flush(3)
    assert len(bodies) == 2
    assert "".join(body["text"] for body in bodies) == "🙂" * 3000
    assert all(
        len(body["text"].encode("utf-16-le")) // 2 <= 4096 for body in bodies
    )
    assert all(body["chat_id"] == 123 for body in bodies)


def test_network_fallback_is_required_and_sessions_are_factories():
    with pytest.raises(TypeError):
        HTTPHandler("https://offline.invalid")
    with pytest.raises(ValueError):
        HTTPHandler("https://offline.invalid", fallback=None)
    with pytest.raises(TypeError):
        HTTPHandler(
            "https://offline.invalid",
            fallback=lambda r: True,
            session=object(),
        )
    with pytest.raises(ValueError):
        HTTPHandler(
            "https://offline.invalid",
            fallback=lambda r: True,
            backend="unknown",
        )
