"""Durable mode rejects over-quota records explicitly instead of dropping."""

import asyncio
import aiologging as logging
from _support import setup

path = setup(quota_bytes=4096)
logging.basicConfig(filename=path / "app.log", level=logging.INFO)


async def main() -> None:
    try:
        await logging.info("x" * 10000)
    except logging.JournalQuotaError:
        print("Large record was explicitly rejected; no successful acceptance")
    await logging.info("A small record fits")
    await logging.flush(5)


if __name__ == "__main__":
    asyncio.run(main())
