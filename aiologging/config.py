"""The standard logging.config API, sharing aiologging's real loggers."""

import logging.config as _stdlib
import sys
import types
from typing import Any
from logging.config import listen as listen, stopListening as stopListening

__all__ = [
    "dictConfig",
    "dictConfigClass",
    "fileConfig",
    "listen",
    "stopListening",
]
dictConfigClass = _stdlib.dictConfigClass


def dictConfig(config: dict[str, Any]) -> None:
    """Configure through stdlib, then resume every configured destination."""
    from .standard import _refresh

    _stdlib.dictConfig(config)
    _refresh()


def fileConfig(*args: Any, **kwargs: Any) -> None:
    """Read a standard file configuration and resume its destinations."""
    from .standard import _refresh

    _stdlib.fileConfig(*args, **kwargs)
    _refresh()


def __getattr__(name: str) -> Any:
    return getattr(_stdlib, name)


class _ConfigModule(types.ModuleType):
    def __getattribute__(self, name: str) -> Any:
        if name == "dictConfigClass":
            return _stdlib.dictConfigClass
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "dictConfigClass":
            _stdlib.dictConfigClass = value
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ConfigModule
