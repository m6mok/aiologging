"""
Level-aware overflow (LevelAwareDrop): errors survive queue pressure.

By default aiologging never drops by severity — the overflow policy
("block" / "drop_new" / "drop_old") treats all records alike, so a
burst of DEBUG/INFO can push an ERROR out of a full queue. For
alerting-critical services the opt-in ``LevelAwareDrop`` policy
sacrifices low-severity context instead (mirroring rsyslog's
``discardSeverity`` and logback's ``discardingThreshold``, both of
which are opt-in too):

- above the ``watermark`` fill ratio, arriving records below
  ``discard_below`` are discarded;
- on a completely full queue the oldest expendable queued record is
  evicted to make room for the arriving one;
- with no expendable victims the configured overflow policy applies
  unchanged.

The queue here is tiny and the records are logged back-to-back (the
consumer gets no chance to drain in between), so the pressure is
reproducible.

Run:
    python examples/drop_policy.py
"""

import asyncio
import logging
from typing import List

import aiologging
from aiologging import LevelAwareDrop


class CollectingHandler(aiologging.AsyncHandler):
    """Keep delivered messages in memory so we can show the outcome."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: List[str] = []

    async def _emit(
        self, record: logging.LogRecord, formatted_message: str
    ) -> None:
        self.messages.append(
            f"{record.levelname}: {record.getMessage()}"
        )


async def main() -> None:
    manager = aiologging.AsyncLoggerManager(
        queue_size=4,
        overflow="drop_old",
        drop_policy=LevelAwareDrop(
            discard_below=logging.WARNING,  # DEBUG/INFO are expendable
            watermark=0.5,  # shed arriving INFO at >= 50% fill
        ),
    )
    logger = manager.getLogger("app")
    logger.setLevel(logging.DEBUG)
    handler = CollectingHandler()
    logger.addHandler(handler)

    # A burst under pressure: two INFO records fit below the
    # watermark, the rest of the INFO burst is shed, and the ERRORs
    # first fill the queue, then evict the queued INFO context
    for i in range(6):
        await logger.info("context %d", i)
    await logger.error("Source failing")
    await logger.error("Cannot write heartbeat")
    await logger.error("Service crashed")

    await manager.flush()
    print("delivered:")
    for message in handler.messages:
        print(f"  {message}")
    print("(every ERROR survived; excess INFO context was sacrificed)")

    await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
