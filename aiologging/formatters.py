"""
Formatters bundled with aiologging.

Currently a single formatter lives here: ``TelegramHtmlFormatter``,
an HTML-safe formatter for ``AsyncTelegramHandler`` used with
``parse_mode="HTML"``. Any stdlib-compatible formatter (an object
satisfying ``FormatterProtocol``) can be used instead.
"""

from __future__ import annotations

import copy
import logging
from logging import LogRecord
from typing import Optional

# Default template, mirroring the HtmlFormatter of
# python-telegram-handler for drop-in parity
DEFAULT_TELEGRAM_HTML_FORMAT = (
    "<code>%(asctime)s</code> <b>%(levelname)s</b>\n"
    "From %(name)s:%(funcName)s\n"
    "%(message)s"
)


def escape_html(text: str) -> str:
    """
    Escape the characters Telegram HTML parse mode reserves.

    Args:
        text: The raw text to escape

    Returns:
        The text with ``&``, ``<`` and ``>`` replaced by entities
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class TelegramHtmlFormatter(logging.Formatter):
    """
    Formatter producing valid Telegram HTML (``parse_mode="HTML"``).

    The record fields that carry user-controlled text — the rendered
    message, ``name``, ``funcName``, the exception text and the stack
    info — are HTML-escaped before the template is interpolated, so
    the template itself may carry markup while record content can
    never break it. A rendered traceback is appended wrapped in
    ``<pre>…</pre>``, like the stack info.

    Compatible with ``FormatterProtocol``; pass it to any handler,
    typically ``AsyncTelegramHandler``.

    Example:
        >>> handler = AsyncTelegramHandler(
        ...     token=token,
        ...     chat_id=chat_id,
        ...     parse_mode="HTML",
        ...     formatter=TelegramHtmlFormatter(),
        ... )
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
    ) -> None:
        """
        Initialize the formatter.

        Args:
            fmt: %-style format template; may contain Telegram HTML
                markup (default: ``DEFAULT_TELEGRAM_HTML_FORMAT``)
            datefmt: ``time.strftime`` date format for ``%(asctime)s``
        """
        super().__init__(fmt or DEFAULT_TELEGRAM_HTML_FORMAT, datefmt)

    def format(self, record: LogRecord) -> str:
        """
        Format a record with HTML-escaped dynamic fields.

        Args:
            record: The log record to format

        Returns:
            The formatted message, valid for ``parse_mode="HTML"``
        """
        # The record is shared with the other handlers — escape a copy
        escaped = copy.copy(record)
        escaped.msg = escape_html(record.getMessage())
        escaped.args = None
        escaped.name = escape_html(record.name)
        escaped.funcName = escape_html(str(record.funcName))

        exc_text = record.exc_text
        if not exc_text and record.exc_info:
            exc_text = self.formatException(record.exc_info)
        # Pre-rendered and escaped here so the stdlib append path
        # (which uses exc_text verbatim) cannot emit raw markup
        escaped.exc_info = None
        escaped.exc_text = (
            f"<pre>{escape_html(exc_text)}</pre>" if exc_text else None
        )
        if record.stack_info:
            escaped.stack_info = (
                f"<pre>{escape_html(record.stack_info)}</pre>"
            )
        return super().format(escaped)
