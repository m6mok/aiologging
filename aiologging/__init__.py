"""Asynchronous stdlib logging with durable, shared SQLite acceptance.

Configure once, then use the standard logging names with await on logging
operations. Synchronous dependencies use the same underlying configuration.
The previous memory-only pipeline is available in aiologging.legacy.
"""

from .standard import *  # noqa: F401,F403
from .standard import __all__ as _standard_all
from .standard import __getattr__ as __getattr__  # noqa: F401
from .extensions import (
    FileHandler as FileHandler,
    StreamHandler as StreamHandler,
    HTTPHandler as HTTPHandler,
    HTTPTextHandler as HTTPTextHandler,
    TelegramHandler as TelegramHandler,
)
from .journal import (
    Journal as Journal,
    JournalError as JournalError,
    JournalCorruptError as JournalCorruptError,
    JournalQuotaError as JournalQuotaError,
    RecordCodec as RecordCodec,
    RecordEncodingError as RecordEncodingError,
)
from .durable import DeliveryError as DeliveryError
from .formatters import (
    TelegramHtmlFormatter as TelegramHtmlFormatter,
    escape_html as escape_html,
)

__version__ = "0.2.11"
__author__ = "Evgenii Dementev (m6mok)"
__license__ = "MIT"
__all__ = _standard_all + [
    "FileHandler",
    "StreamHandler",
    "HTTPHandler",
    "HTTPTextHandler",
    "TelegramHandler",
    "Journal",
    "JournalError",
    "JournalCorruptError",
    "JournalQuotaError",
    "RecordCodec",
    "RecordEncodingError",
    "DeliveryError",
    "TelegramHtmlFormatter",
    "escape_html",
]

# Documented mutable logging flags must also work on the top-level module.
import sys as _sys
from .standard import _LoggingModule

_sys.modules[__name__].__class__ = _LoggingModule
