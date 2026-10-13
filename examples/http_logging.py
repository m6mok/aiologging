"""Offline HTTP batch example. No sockets are opened."""

import asyncio
import httpx
import aiologging as logging
from _support import setup

path = setup()
backup = logging.FileHandler(path / "fallback.log")


def respond(request: httpx.Request) -> httpx.Response:
    print("Mock endpoint received", len(request.content), "bytes")
    return httpx.Response(200)


handler = logging.HTTPHandler(
    "https://offline.invalid/logs",
    fallback=backup.deliver,
    session_factory=lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ),
)
logging.basicConfig(handlers=[handler], level=logging.INFO)


async def main() -> None:
    await logging.info("HTTP batch")
    await logging.flush(5)


if __name__ == "__main__":
    asyncio.run(main())
