"""Files use the standard library, with no aiofiles dependency."""

import asyncio
import aiologging as logging
from _support import setup

path = setup()
logging.basicConfig(filename=path / "application.log", level=logging.INFO)


async def main() -> None:
    await logging.info("A confirmed file write")
    await logging.flush(5)
    print((path / "application.log").read_text())


if __name__ == "__main__":
    asyncio.run(main())
