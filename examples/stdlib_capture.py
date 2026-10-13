"""Synchronous dependencies share the asynchronous application's config."""

import asyncio
import logging as stdlib
import aiologging as logging
from _support import setup

setup()
logging.basicConfig(level=logging.INFO)


async def main() -> None:
    stdlib.getLogger("dependency").info(
        "waits for durable commit in this thread"
    )
    await logging.getLogger("application").info(
        "awaits commit off this event loop"
    )
    await logging.flush(5)


if __name__ == "__main__":
    asyncio.run(main())
