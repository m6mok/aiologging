"""
HTTP logging (requires the ``aiohttp`` extra).

Starts a tiny local aiohttp server that plays the role of a log
collector, then sends log records to it as JSON batches with a custom
authenticator.

The HTTP handlers can also run on top of ``httpx`` instead of aiohttp:
install the ``httpx`` extra and pass ``backend="httpx"`` to the handler.

Run:
    pip install aiologging[aiohttp]
    python examples/http_logging.py
"""

import asyncio
from typing import Any, Dict

from aiohttp import web

import aiologging
from aiologging.types import BatchConfig


async def collector(request: web.Request) -> web.Response:
    """Pretend to be a log collection endpoint."""
    payload = await request.json()
    auth = request.headers.get("Authorization", "<missing>")
    print(f"[collector] auth={auth} received {len(payload)} record(s)")
    for record in payload:
        print(f"[collector]   {record['levelname']}: {record['message']}")
    return web.Response(text="ok")


async def token_authenticator(
    session: Any, request_data: Any
) -> Dict[str, str]:
    """Custom authenticator: fetch/refresh a token and return headers."""
    token = "example-token"  # e.g. await refresh_oauth_token()
    return {"Authorization": f"Bearer {token}"}


async def main() -> None:
    # Start the local collector
    app = web.Application()
    app.router.add_post("/logs", collector)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8080)
    await site.start()

    try:
        logger = aiologging.getLogger("app.http")
        logger.setLevel(aiologging.INFO)

        handler = aiologging.AsyncHttpJsonHandler(
            "http://127.0.0.1:8080/logs",
            authenticator=token_authenticator,
            # Records are buffered and sent in batches
            batch_config=BatchConfig(batch_size=5, flush_interval=1.0),
        )
        logger.addHandler(handler)

        for i in range(12):
            await logger.info("HTTP log message %d", i)

        # shutdown() drains the queue and flushes the remaining batch
        await aiologging.shutdown()
        await asyncio.sleep(0.2)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
