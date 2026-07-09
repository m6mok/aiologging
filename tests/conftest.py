"""
Shared test fixtures.

aiologging keeps global state (the logger manager with its queue and
consumer task, the stdlib bridge, the disable level), so every test
gets a clean slate afterwards.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import aiologging
from aiologging.bridge import captureStdlib


@pytest.fixture(autouse=True)
def _reset_aiologging():
    yield
    captureStdlib(False)
    aiologging.disable(logging.NOTSET)
    # The test's own event loop is already gone; a fresh one both
    # exercises loop-change recovery and drains whatever is left.
    # Run it on an explicit loop without touching the thread-local
    # current loop: asyncio.run() would leave it unset, which breaks
    # sync creation of asyncio primitives on Python 3.9.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(aiologging.shutdown())
    finally:
        loop.close()
