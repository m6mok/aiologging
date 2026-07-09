"""
Inline delivery of critical bridged records (captureStdlib inline_level).

The alerting case: ``getLogger("log")`` → Telegram is the operator's
only pager, and the records that matter most — "Service crashed"
right before a ``raise``, "Received SIGTERM" in a signal handler —
are emitted from synchronous code, often when no event loop is
running (or the loop is about to die). With ``inline_level`` set, the
stdlib bridge delivers such records synchronously — before queueing —
to handlers that support it (``AsyncTelegramHandler`` posts via
urllib in a helper thread joined with the deadline). The inline path
is bounded by ``inline_timeout`` and a token bucket
(``inline_burst``/``inline_rate``), and any failure just leaves the
record on the normal queue path.

A local HTTP server plays the Telegram Bot API, so the example runs
offline.

Run:
    pip install aiologging[aiohttp]   # or [httpx]
    python examples/inline_alerts.py
"""

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, List

import aiologging
from aiologging.types import BatchConfig

TOKEN = "123456:example-token"

received: List[str] = []


class BotApiStub(BaseHTTPRequestHandler):
    """Pretend to be the Bot API sendMessage method."""

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        received.append(payload["text"])
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the example output clean


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), BotApiStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    pager = aiologging.getLogger("log")
    pager.addHandler(
        aiologging.AsyncTelegramHandler(
            token=TOKEN,
            chat_id="-1001234567890",
            api_base_url=f"http://127.0.0.1:{port}",
            batch_config=BatchConfig(batch_size=10, flush_interval=1.0),
        )
    )
    # ERROR and above also go inline; include %(asctime)s in real
    # formatters — an inline record reaches the chat ahead of older
    # queued ones
    aiologging.captureStdlib(True, inline_level=logging.ERROR)

    # Synchronous context, no event loop anywhere — e.g. an except
    # block right before re-raising, or a SIGTERM handler. The record
    # is POSTed to the Bot API before this call returns; the queued
    # copy is marked so the handler will not deliver it twice.
    logging.getLogger("log").error("Service crashed, restarting")
    print(f"delivered inline, before any event loop: {received}")

    # The inline budget is spent (burst=1 by default, refill 2/min),
    # so the next record takes the normal queue path; flush_sync
    # drains it on a private loop — same guarantee, slightly later.
    logging.getLogger("log").error("Received SIGTERM")
    aiologging.flush_sync(timeout=5.0)
    print(f"after flush_sync: {received}")

    aiologging.captureStdlib(False)
    asyncio.run(aiologging.shutdown())
    server.shutdown()


if __name__ == "__main__":
    main()
