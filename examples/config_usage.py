"""Standard dictConfig, including factories and named handler settings."""

import asyncio
import aiologging as logging
from aiologging.config import dictConfig
from _support import setup

path = setup()
dictConfig(
    {
        "version": 1,
        "formatters": {"plain": {"format": "{name}: {message}", "style": "{"}},
        "handlers": {
            "file": {
                "class": "aiologging.FileHandler",
                "filename": str(path / "app.log"),
                "encoding": "utf-8",
                "formatter": "plain",
            }
        },
        "root": {"level": "INFO", "handlers": ["file"]},
    }
)


async def main() -> None:
    await logging.getLogger("configured").info("configured through stdlib")
    await logging.flush(5)
    print((path / "app.log").read_text())


if __name__ == "__main__":
    asyncio.run(main())
