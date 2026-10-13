"""Offline Telegram transport with a separately confirmed file fallback."""

import asyncio
import json
import httpx
import aiologging as logging
from _support import setup

path = setup()
backup = logging.FileHandler(path / "fallback.log")


def respond(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    print("Mock Telegram chat", body["chat_id"], "received", body["text"])
    return httpx.Response(200, json={"ok": True})


handler = logging.TelegramHandler(
    "OFFLINE_TOKEN",
    "demo-chat",
    fallback=backup.deliver,
    session_factory=lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ),
)
logging.basicConfig(handlers=[handler], level=logging.INFO)


async def main() -> None:
    await logging.error("An example alert")
    await logging.flush(5)


if __name__ == "__main__":
    asyncio.run(main())
