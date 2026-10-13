"""Fallback preserves FIFO and leaves the original primary task pending."""

import asyncio
import httpx
import aiologging as logging
from _support import setup

path = setup()
backup = logging.FileHandler(path / "fallback.log")
available = False


def respond(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200 if available else 503)


handler = logging.HTTPHandler(
    "https://offline.invalid",
    fallback=backup.deliver,
    retry_delay=0.01,
    session_factory=lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ),
)
logging.basicConfig(handlers=[handler], level=logging.INFO)


async def main() -> None:
    global available
    await logging.error("Primary is temporarily unavailable")
    try:
        await logging.flush(0.1)
    except logging.DeliveryError:
        print("Fallback:", (path / "fallback.log").read_text())
    available = True
    await logging.flush(5)
    print("Primary recovered; its journal task is now acknowledged")


if __name__ == "__main__":
    asyncio.run(main())
