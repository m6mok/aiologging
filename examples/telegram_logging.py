"""
Telegram logging (requires the ``aiohttp`` extra).

Starts a tiny local aiohttp server that plays the role of the
Telegram Bot API — including a 429 (Too Many Requests) response with
a ``retry_after`` hint — then sends log records to it with
AsyncTelegramHandler.

Against the real Bot API, drop ``api_base_url`` and pass the token
from @BotFather and a real chat id:

    handler = aiologging.AsyncTelegramHandler(
        token=os.environ["BOT_TOKEN"],
        chat_id=os.environ["CHAT_ID"],
        level=aiologging.ERROR,
    )

Run:
    pip install aiologging[aiohttp]
    python examples/telegram_logging.py
"""

import asyncio

from aiohttp import web

import aiologging
from aiologging.types import BatchConfig

TOKEN = "123456:example-token"


request_count = 0


async def send_message(request: web.Request) -> web.Response:
    """Pretend to be the Bot API sendMessage method."""
    global request_count
    request_count += 1

    # Simulate rate limiting on the first request; the handler
    # retries after the delay from parameters.retry_after
    if request_count == 1:
        print("[bot api] 429 Too Many Requests (retry_after=1)")
        return web.json_response(
            {
                "ok": False,
                "error_code": 429,
                "description": "Too Many Requests: retry after 1",
                "parameters": {"retry_after": 1},
            },
            status=429,
        )

    payload = await request.json()
    text = payload["text"]
    print(
        f"[bot api] message to chat {payload['chat_id']} "
        f"({len(text)} chars):"
    )
    for line in text.splitlines():
        print(f"[bot api]   {line}")
    return web.json_response({"ok": True})


async def main() -> None:
    # Start the local Bot API stand-in
    app = web.Application()
    app.router.add_post(f"/bot{TOKEN}/sendMessage", send_message)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8081)
    await site.start()

    try:
        logger = aiologging.getLogger("app.telegram")
        logger.setLevel(aiologging.INFO)

        handler = aiologging.AsyncTelegramHandler(
            token=TOKEN,
            chat_id="-1001234567890",
            api_base_url="http://127.0.0.1:8081",
            # Records are buffered and combined into as few
            # messages as possible (4096-char limit per message)
            batch_config=BatchConfig(batch_size=5, flush_interval=1.0),
        )
        logger.addHandler(handler)

        for i in range(7):
            await logger.info("Telegram log message %d", i)

        # shutdown() drains the queue and flushes the remaining batch
        await aiologging.shutdown()
        await asyncio.sleep(0.2)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
