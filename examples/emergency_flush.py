"""The library loop can drain after the application's loop has closed."""

import asyncio
import aiologging as logging
from _support import setup

path = setup()
logging.basicConfig(filename=path / "app.log", level=logging.INFO)


async def main() -> None:
    for i in range(5):
        await logging.info("record %d", i)


if __name__ == "__main__":
    asyncio.run(main())
    assert logging.flush_sync(5)
    print((path / "app.log").read_text())
