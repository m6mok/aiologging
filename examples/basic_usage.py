"""No manual shutdown: accepted messages drain on ordinary process exit."""

import asyncio
import aiologging as logging
from _support import setup

setup()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("example")


async def main() -> None:
    await logger.info("Hello %s", "world")
    await asyncio.gather(*(logger.info("record %d", i) for i in range(5)))


if __name__ == "__main__":
    asyncio.run(main())
