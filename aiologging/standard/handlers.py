"""The documented logging.handlers surface with standard behaviour.

These are synchronous stdlib handlers, run by durable recipient workers.
They retain their ordinary error behaviour, including handleError. Strict
confirmation is available through aiologging.extensions or a custom deliver
method which returns True only after delivery.
"""

import logging.handlers as _stdlib
from typing import Any

__all__ = [name for name in dir(_stdlib) if not name.startswith("_")]


def __getattr__(name: str) -> Any:
    return getattr(_stdlib, name)
