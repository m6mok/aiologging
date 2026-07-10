"""
Configuring loggers from a dictionary.

The same structure works from a JSON file via
``aiologging.configure_from_file("config.json")`` and from environment
variables via ``aiologging.configure_from_env()``.

Configured loggers live in the global aiologging hierarchy: they have
a parent, propagate records to ancestor handlers, and are covered by
``aiologging.flush()`` / ``aiologging.shutdown()``. Named formatters
from the "formatters" section are referenced by handlers through the
"formatter" key; unknown configuration keys raise an explicit error.

Run:
    python examples/config_usage.py
"""

import asyncio

import aiologging

CONFIG = {
    "version": 1,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
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
            "stream": "stdout",
            "formatter": "detailed",
        },
        "file": {
            "class": "file",
            "level": "DEBUG",
            "filename": "logs/configured.log",
            "mode": "a",
            "formatter": "detailed",
        },
    },
}


async def main() -> None:
    aiologging.configure_from_dict(CONFIG)

    logger = aiologging.get_configured_logger("myapp")
    await logger.info("Configured via dictionary")
    await logger.debug("This goes to the file handler only")

    # The configured logger is part of the global hierarchy
    assert aiologging.getLogger("myapp") is logger
    assert logger.parent is aiologging.getLogger()

    await aiologging.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
