"""
Tests for aiologging.formatters (TelegramHtmlFormatter).
"""

import logging
import sys
from typing import Optional

import pytest

from aiologging.formatters import (
    DEFAULT_TELEGRAM_HTML_FORMAT,
    TelegramHtmlFormatter,
    escape_html,
)


def _record(
    message: str = "hello",
    name: str = "app",
    func: str = "run",
    level: int = logging.INFO,
    exc_info: Optional[tuple] = None,
) -> logging.LogRecord:
    """Create a log record with a controllable funcName."""
    record = logging.LogRecord(
        name=name, level=level, pathname="", lineno=0,
        msg=message, args=(), exc_info=exc_info, func=func,
    )
    return record


class TestEscapeHtml:
    """Test cases for the escape_html helper."""

    def test_escapes_reserved_characters(self) -> None:
        """&, < and > become entities; & first to avoid doubling."""
        assert escape_html("a < b & c > d") == (
            "a &lt; b &amp; c &gt; d"
        )

    def test_plain_text_unchanged(self) -> None:
        """Text without reserved characters passes through."""
        assert escape_html("plain text 123") == "plain text 123"


class TestTelegramHtmlFormatter:
    """Test cases for TelegramHtmlFormatter."""

    def test_default_template(self) -> None:
        """The default template mirrors python-telegram-handler."""
        formatter = TelegramHtmlFormatter()
        out = formatter.format(_record("hi"))
        assert out.startswith("<code>")
        assert "<b>INFO</b>" in out
        assert "From app:run" in out
        assert out.endswith("hi")

    def test_message_escaped(self) -> None:
        """Reserved characters in the message become entities."""
        formatter = TelegramHtmlFormatter()
        out = formatter.format(_record("x < y & z > w"))
        assert "x &lt; y &amp; z &gt; w" in out
        assert "x < y" not in out

    def test_name_and_funcname_escaped(self) -> None:
        """Logger name and funcName are escaped too."""
        formatter = TelegramHtmlFormatter()
        out = formatter.format(
            _record("hi", name="mod<ule>", func="<lambda>")
        )
        assert "From mod&lt;ule&gt;:&lt;lambda&gt;" in out

    def test_percent_args_rendered_before_escaping(self) -> None:
        """%-args are interpolated first, then escaped."""
        formatter = TelegramHtmlFormatter()
        record = logging.LogRecord(
            name="app", level=logging.INFO, pathname="", lineno=0,
            msg="value=%s", args=("<b>",), exc_info=None, func="run",
        )
        assert "value=&lt;b&gt;" in formatter.format(record)

    def test_exception_escaped_and_wrapped_in_pre(self) -> None:
        """A traceback (with <module> frames) is escaped in <pre>."""
        try:
            exec(compile("raise ValueError('a < b')", "<string>", "exec"))
        except ValueError:
            exc_info = sys.exc_info()
        record = _record("boom", exc_info=exc_info)
        out = TelegramHtmlFormatter().format(record)
        assert "<pre>" in out and out.endswith("</pre>")
        assert "&lt;module&gt;" in out
        assert "a &lt; b" in out
        assert "<module>" not in out

    def test_frozen_exc_text_escaped(self) -> None:
        """exc_text set by the queue freeze path is escaped too."""
        record = _record("boom")
        record.exc_text = 'File "<string>", line 1, in <module>'
        out = TelegramHtmlFormatter().format(record)
        assert "&lt;module&gt;" in out
        assert "<module>" not in out
        # The record shared with other handlers is left untouched
        assert record.exc_text == (
            'File "<string>", line 1, in <module>'
        )

    def test_stack_info_escaped(self) -> None:
        """stack_info is escaped and wrapped in <pre>."""
        record = _record("hi")
        record.stack_info = "Stack: <frame>"
        out = TelegramHtmlFormatter().format(record)
        assert "<pre>Stack: &lt;frame&gt;</pre>" in out

    def test_custom_template_and_datefmt(self) -> None:
        """fmt and datefmt are configurable."""
        formatter = TelegramHtmlFormatter(
            fmt="<i>%(asctime)s</i> %(message)s",
            datefmt="%Y",
        )
        out = formatter.format(_record("msg"))
        assert out.startswith("<i>")
        # datefmt "%Y" renders just the four-digit year
        year = out[len("<i>"):out.index("</i>")]
        assert len(year) == 4 and year.isdigit()
        assert out.endswith("msg")

    def test_uses_time_with_default_template(self) -> None:
        """The default template includes %(asctime)s."""
        assert "%(asctime)s" in DEFAULT_TELEGRAM_HTML_FORMAT
        assert TelegramHtmlFormatter().usesTime()

    def test_satisfies_formatter_protocol(self) -> None:
        """The formatter satisfies FormatterProtocol structurally."""
        from aiologging.types import FormatterProtocol

        assert isinstance(TelegramHtmlFormatter(), FormatterProtocol)


if __name__ == "__main__":
    pytest.main([__file__])
