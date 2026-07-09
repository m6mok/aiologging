"""
File logging with rotation (requires the ``aiofiles`` extra).

Shows a plain file handler, size-based rotation and time-based rotation.

Run:
    pip install aiologging[aiofiles]
    python examples/file_logging.py
"""

import asyncio
import logging
from pathlib import Path

import aiologging

LOG_DIR = Path("logs")


async def main() -> None:
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    logger = aiologging.getLogger("app.files")
    logger.setLevel(aiologging.DEBUG)

    # Plain append-mode file handler
    plain = aiologging.AsyncFileHandler(
        LOG_DIR / "app.log", formatter=formatter
    )
    logger.addHandler(plain)

    # Size-based rotation: keep up to 3 backups of 1 KiB each
    rotating = aiologging.AsyncRotatingFileHandler(
        LOG_DIR / "rotating.log",
        max_bytes=1024,
        backup_count=3,
        formatter=formatter,
    )
    logger.addHandler(rotating)

    # Time-based rotation: a new file every midnight, keep a week
    timed = aiologging.AsyncTimedRotatingFileHandler(
        LOG_DIR / "timed.log",
        when="midnight",
        backup_count=7,
        formatter=formatter,
    )
    logger.addHandler(timed)

    for i in range(50):
        await logger.info("Message %03d - watch rotating.log roll over", i)

    # Drain the queue and close the file handlers
    await aiologging.shutdown()

    print(f"Done, see the {LOG_DIR}/ directory:")
    for path in sorted(LOG_DIR.iterdir()):
        print(f"  {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
