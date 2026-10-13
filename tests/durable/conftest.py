"""Isolate the shared stdlib hierarchy and durable engine for API tests."""

import asyncio
import logging
import pytest
import aiologging as aio


@pytest.fixture
async def configured(tmp_path):
    root_handlers = logging.root.handlers[:]
    root_level = logging.root.level
    factory = logging.getLogRecordFactory()
    logger_class = logging.getLoggerClass()
    last_resort = logging.lastResort
    logging.root.handlers = []
    engine = aio.configureJournal(tmp_path / "journal")
    logger = aio.getLogger("durable-tests")
    logger.handlers = []
    logger.filters = []
    logger.disabled = False
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    try:
        yield engine, logger
    finally:
        await aio.shutdown(0.05)
        async with asyncio.timeout(3):
            while not engine.closed:
                await asyncio.sleep(0.01)
        logger.handlers = []
        logger.filters = []
        logging.root.handlers = root_handlers
        logging.root.setLevel(root_level)
        logging.setLogRecordFactory(factory)
        logging.setLoggerClass(logger_class)
        logging.lastResort = last_resort
