"""
Routing standard logging through aiologging.

Third-party libraries (aiohttp, sqlalchemy, ...) log via the standard
``logging`` module. With ``capture_stdlib=True`` their records are
bridged into the aiologging queue and flow through the same async
handlers as your own ``await`` calls — including records emitted from
other threads and records logged before the event loop started.

Run:
    python examples/stdlib_capture.py
"""

import asyncio
import logging
import threading

import aiologging

# A stand-in for a third-party library that knows nothing about
# aiologging and logs via the standard logging module
library_logger = logging.getLogger("some.library")


def blocking_worker() -> None:
    """Simulates library code running in a worker thread."""
    library_logger.warning("hello from a worker thread")


async def main() -> None:
    aiologging.basicConfig(
        level=aiologging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        capture_stdlib=True,  # or: aiologging.captureStdlib()
    )

    # Your own async logging
    app_logger = aiologging.getLogger("app")
    await app_logger.info("Application started")

    # Library code logging via stdlib — same handlers, same format.
    # Records keep their logger name and are routed through the
    # aiologging hierarchy, so per-name configuration still applies
    library_logger.warning("stdlib record from the main thread")

    # The bridge is thread-safe
    thread = threading.Thread(target=blocking_worker)
    thread.start()
    thread.join()

    await aiologging.shutdown()


if __name__ == "__main__":
    # The bridge can be installed before any event loop exists:
    # records are buffered and delivered once the consumer starts
    aiologging.captureStdlib()
    library_logger.warning("logged before the event loop started")

    asyncio.run(main())
