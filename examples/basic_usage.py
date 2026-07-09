"""
Basic aiologging usage.

The API mirrors the standard logging module: get a logger, await the
logging methods. Records are enqueued at the call site and written by
a background consumer, so the await never waits for handler I/O.
Call ``aiologging.shutdown()`` once at exit to drain the queue.

Run:
    python examples/basic_usage.py
"""

import asyncio

import aiologging


async def main() -> None:
    # Configure the root logger once, like logging.basicConfig()
    aiologging.basicConfig(
        level=aiologging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Loggers work exactly like logging.getLogger(...); the logging
    # methods are awaitable and resolve as soon as the record is
    # enqueued — actual I/O happens in the background
    logger = aiologging.getLogger("app")
    await logger.info("Application started")
    await logger.warning("Something might be wrong")
    await logger.error("An error occurred: %s", "example")

    # exc_info=True attaches the current exception, just like in the
    # standard logging module
    try:
        1 / 0
    except ZeroDivisionError:
        await logger.exception("Unhandled arithmetic error")

    # For critical records you can wait for actual delivery: with
    # delivery="await" the coroutine resolves only after the handlers
    # have processed the record
    audit = aiologging.getLogger("app.audit")
    audit.delivery = "await"
    await audit.warning("This line is on disk/stream when we get here")

    # Module-level convenience functions log via the root logger
    await aiologging.warning("Fire and forget on the root logger")

    # flush() waits until everything queued so far has been handled
    await aiologging.flush()

    # Drain the queue and close handlers before the loop goes away
    await aiologging.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
