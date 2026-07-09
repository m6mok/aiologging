"""
Configuring loggers from a dictionary.

The same structure works from a JSON file via
``aiologging.configure_from_file("config.json")`` and from environment
variables via ``aiologging.configure_from_env()``.

Run:
    python examples/config_usage.py
"""

import asyncio

import aiologging

CONFIG = {
    "version": 1,
    "loggers": {
        "myapp": {
            "level": "INFO",
            "handlers": ["console", "file"],
        }
    },
    "handlers": {
        "console": {
            "class": "stream",
            "level": "INFO",
        },
        "file": {
            "class": "file",
            "level": "DEBUG",
            "filename": "logs/configured.log",
            "mode": "a",
        },
    },
}


async def main() -> None:
    aiologging.configure_from_dict(CONFIG)

    logger = aiologging.get_configured_logger("myapp")
    await logger.info("Configured via dictionary")
    await logger.debug("This goes to the file handler only")

    await aiologging.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
