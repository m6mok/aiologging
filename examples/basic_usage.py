"""
Basic aiologging usage.

Shows the recommended usage pattern (``async with`` + ``await``),
logging levels, formatters and ``basicConfig``.

Run:
    python examples/basic_usage.py
"""

import asyncio
import logging

import aiologging


async def main() -> None:
    # Configure the root logger once, similar to logging.basicConfig()
    aiologging.basicConfig(
        level=aiologging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # The recommended pattern: the context manager closes the logger
    # (and its handlers) on exit
    async with aiologging.getLogger("app") as logger:
        await logger.info("Application started")
        await logger.warning("Something might be wrong")
        await logger.error("An error occurred: %s", "example")

        # exc_info=True attaches the current exception, just like in
        # the standard logging module
        try:
            1 / 0
        except ZeroDivisionError:
            await logger.exception("Unhandled arithmetic error")

    # One-off logging without managing the logger lifecycle
    await aiologging.log_async("app.oneoff", logging.INFO, "Fire and forget")


if __name__ == "__main__":
    asyncio.run(main())
