"""
Tests for the Telegram handler.
"""

import json
import logging
import re
from typing import Any, Callable, List

import httpx
import pytest

from aiologging.config import ConfigManager
from aiologging.formatters import TelegramHtmlFormatter
from aiologging.handlers.telegram import (
    AsyncTelegramHandler,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    _split_html,
)
from aiologging.exceptions import ConfigurationError, NetworkError
from aiologging.types import BatchConfig

TOKEN = "123456:ABC-secret"
CHAT_ID = "-1001234567890"


def _create_record(
    message: str = "Test message", level: int = logging.INFO
) -> logging.LogRecord:
    """Create a test log record."""
    return logging.LogRecord(
        name="test", level=level, pathname="", lineno=0,
        msg=message, args=(), exc_info=None
    )


def _make_handler(
    respond: Callable[[httpx.Request], httpx.Response],
    **handler_kwargs: Any,
) -> AsyncTelegramHandler:
    """Create a Telegram handler wired to an httpx.MockTransport."""
    handler = AsyncTelegramHandler(
        TOKEN,
        CHAT_ID,
        backend="httpx",
        **handler_kwargs,
    )
    handler._session = httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    )
    return handler


class TestAsyncTelegramHandlerInit:
    """Test cases for AsyncTelegramHandler initialization."""

    def test_initialization(self) -> None:
        """The sendMessage URL is built from token and base URL."""
        handler = AsyncTelegramHandler(TOKEN, CHAT_ID)
        assert handler.url == (
            f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        )
        assert handler.method == "POST"
        assert handler.chat_id == CHAT_ID
        assert handler.parse_mode is None
        assert handler.max_message_length == TELEGRAM_MAX_MESSAGE_LENGTH

    def test_custom_api_base_url(self) -> None:
        """A local Bot API server URL can be used (trailing slash ok)."""
        handler = AsyncTelegramHandler(
            TOKEN, CHAT_ID, api_base_url="http://localhost:8081/"
        )
        assert handler.url == (
            f"http://localhost:8081/bot{TOKEN}/sendMessage"
        )

    def test_empty_token_rejected(self) -> None:
        """An empty token raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="bot token"):
            AsyncTelegramHandler("", CHAT_ID)

    def test_empty_chat_id_rejected(self) -> None:
        """An empty chat_id raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="chat_id"):
            AsyncTelegramHandler(TOKEN, "")

    def test_zero_chat_id_allowed(self) -> None:
        """A numeric chat_id of 0 is not treated as missing."""
        handler = AsyncTelegramHandler(TOKEN, 0)
        assert handler.chat_id == 0

    def test_invalid_max_message_length(self) -> None:
        """max_message_length outside 1..4096 raises."""
        with pytest.raises(ConfigurationError, match="max_message_length"):
            AsyncTelegramHandler(TOKEN, CHAT_ID, max_message_length=0)
        with pytest.raises(ConfigurationError, match="max_message_length"):
            AsyncTelegramHandler(TOKEN, CHAT_ID, max_message_length=4097)

    def test_repr_hides_token(self) -> None:
        """The bot token does not appear in repr."""
        handler = AsyncTelegramHandler(TOKEN, CHAT_ID)
        repr_str = repr(handler)
        assert "AsyncTelegramHandler" in repr_str
        assert TOKEN not in repr_str
        assert "/bot***/" in repr_str


class TestTelegramMessageBuilding:
    """Test cases for message chunking and payloads."""

    def test_records_combined_into_one_message(self) -> None:
        """Short records are joined with newlines into one message."""
        handler = AsyncTelegramHandler(TOKEN, CHAT_ID)
        records = [_create_record("line 1"), _create_record("line 2")]
        assert handler._build_messages(records) == ["line 1\nline 2"]

    def test_messages_split_at_limit(self) -> None:
        """A new message starts when the limit would be exceeded."""
        handler = AsyncTelegramHandler(
            TOKEN, CHAT_ID, max_message_length=10
        )
        records = [
            _create_record("aaaa"),
            _create_record("bbbb"),
            _create_record("cccc"),
        ]
        # "aaaa\nbbbb" is exactly 9 chars; adding "\ncccc" exceeds 10
        assert handler._build_messages(records) == ["aaaa\nbbbb", "cccc"]

    def test_long_record_hard_split(self) -> None:
        """A single record longer than the limit is hard-split."""
        handler = AsyncTelegramHandler(
            TOKEN, CHAT_ID, max_message_length=10
        )
        records = [_create_record("x" * 25)]
        assert handler._build_messages(records) == [
            "x" * 10, "x" * 10, "x" * 5
        ]

    @pytest.mark.asyncio
    async def test_payload_fields(self) -> None:
        """The sendMessage payload carries the configured fields."""
        handler = AsyncTelegramHandler(
            TOKEN,
            CHAT_ID,
            parse_mode="HTML",
            disable_notification=True,
            message_thread_id=7,
        )
        payloads = await handler._prepare_request_data(
            [_create_record("hello")]
        )
        assert payloads == [{
            "chat_id": CHAT_ID,
            "text": "hello",
            "parse_mode": "HTML",
            "disable_notification": True,
            "message_thread_id": 7,
        }]

    @pytest.mark.asyncio
    async def test_payload_minimal(self) -> None:
        """Optional fields are omitted from the payload by default."""
        handler = AsyncTelegramHandler(TOKEN, CHAT_ID)
        payloads = await handler._prepare_request_data(
            [_create_record("hello")]
        )
        assert payloads == [{"chat_id": CHAT_ID, "text": "hello"}]

    def test_content_type(self) -> None:
        """The Bot API is spoken to in JSON."""
        handler = AsyncTelegramHandler(TOKEN, CHAT_ID)
        assert handler._get_content_type() == "application/json"


_HTML_TOKEN = re.compile(
    r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)((?:\s[^<>]*)?)>"
    r"|&(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);"
)

_KNOWN_TAGS = {"b", "i", "u", "s", "code", "pre", "a"}


def _assert_valid_telegram_html(text: str) -> None:
    """Fail on raw <, >, &, unknown tags or unbalanced tags."""
    stack: List[str] = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if char in "<&":
            match = _HTML_TOKEN.match(text, pos)
            assert match is not None, (
                f"raw {char!r} at {pos}: {text[pos:pos + 20]!r}"
            )
            name = match.group(2)
            if name is not None:
                name = name.lower()
                assert name in _KNOWN_TAGS, f"unknown tag {name!r}"
                if match.group(1):
                    assert stack and stack[-1] == name, (
                        f"unbalanced </{name}> in {text!r}"
                    )
                    stack.pop()
                else:
                    stack.append(name)
            pos = match.end()
        else:
            assert char != ">", f"raw '>' at {pos}: {text!r}"
            pos += 1
    assert not stack, f"unclosed tags {stack} in {text!r}"


class TestHtmlEntitySafeSplitting:
    """Test cases for the HTML-aware splitter (parse_mode="HTML")."""

    def test_short_text_unsplit(self) -> None:
        """Text within the limit is returned as a single message."""
        handler = AsyncTelegramHandler(
            TOKEN, CHAT_ID, parse_mode="HTML"
        )
        assert handler._split_text("<b>short</b>") == ["<b>short</b>"]

    def test_tag_closed_and_reopened_across_parts(self) -> None:
        """A tag spanning the boundary is closed and reopened."""
        parts = _split_html("<b>" + "x" * 30 + "</b>", 20)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 20
            _assert_valid_telegram_html(part)
        assert parts[0].startswith("<b>") and parts[0].endswith("</b>")
        assert parts[1].startswith("<b>")
        # No payload characters were lost in the split
        payload = "".join(re.sub(r"</?b>", "", part) for part in parts)
        assert payload == "x" * 30

    def test_boundary_never_inside_entity(self) -> None:
        """A part boundary cannot cut an &...; entity in half."""
        text = "ab&amp;" * 20
        for limit in range(8, 16):
            for part in _split_html(text, limit):
                assert len(part) <= limit
                _assert_valid_telegram_html(part)

    def test_boundary_never_inside_tag(self) -> None:
        """A part boundary cannot cut a <tag> token in half."""
        text = "<code>ab</code>" * 10
        for limit in range(16, 24):
            for part in _split_html(text, limit):
                assert len(part) <= limit
                _assert_valid_telegram_html(part)

    def test_link_reopened_with_attributes(self) -> None:
        """An <a href=...> is reopened with its attributes intact."""
        opening = '<a href="https://example.com">'
        parts = _split_html(opening + "y" * 60 + "</a>", 45)
        assert len(parts) > 1
        for part in parts:
            assert part.startswith(opening)
            assert part.endswith("</a>")

    def test_nested_tags_reopened_in_order(self) -> None:
        """Nested open tags close in reverse and reopen in order."""
        parts = _split_html("<b><i>" + "z" * 40 + "</i></b>", 25)
        assert len(parts) > 1
        assert parts[0].endswith("</i></b>")
        assert parts[1].startswith("<b><i>")

    def test_unknown_tag_like_text_kept_atomic(self) -> None:
        """A non-Telegram token like <module> is never cut in half."""
        text = ("word " * 4) + "<module>" + ("word " * 4)
        for limit in range(9, 20):
            for part in _split_html(text, limit):
                assert len(part) <= limit
                # the pseudo-tag survives whole in exactly one part
            joined = _split_html(text, limit)
            assert sum("<module>" in part for part in joined) == 1

    def test_record_split_at_bot_api_limit(self) -> None:
        """A formatted record with markup splits safely at 4096."""
        handler = AsyncTelegramHandler(
            TOKEN, CHAT_ID, parse_mode="HTML",
            formatter=TelegramHtmlFormatter(),
        )
        record = _create_record("long & <detail> " * 400)
        messages = handler._build_messages([record])
        assert len(messages) > 1
        for message in messages:
            assert len(message) <= TELEGRAM_MAX_MESSAGE_LENGTH
            _assert_valid_telegram_html(message)

    def test_batched_records_stay_valid(self) -> None:
        """Several records batched into one message remain valid."""
        handler = AsyncTelegramHandler(
            TOKEN, CHAT_ID, parse_mode="HTML",
            formatter=TelegramHtmlFormatter(),
        )
        records = [
            _create_record(f"line {i}: a<b & c>d") for i in range(5)
        ]
        messages = handler._build_messages(records)
        assert len(messages) == 1
        _assert_valid_telegram_html(messages[0])

    def test_non_html_mode_keeps_hard_split(self) -> None:
        """Without parse_mode="HTML" the split stays character-based."""
        handler = AsyncTelegramHandler(
            TOKEN, CHAT_ID, parse_mode="MarkdownV2",
            max_message_length=10,
        )
        assert handler._split_text("x" * 25) == [
            "x" * 10, "x" * 10, "x" * 5
        ]

    @pytest.mark.asyncio
    async def test_html_payloads_sent_are_valid(self) -> None:
        """End to end: what leaves the handler is valid, escaped HTML."""
        requests: List[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        handler = _make_handler(
            respond,
            parse_mode="HTML",
            formatter=TelegramHtmlFormatter(),
            max_message_length=200,
        )

        await handler.flush([
            _create_record("tick <index> & sons " * 20),
            _create_record("second & <last>"),
        ])
        await handler.close()

        assert len(requests) > 1
        for request in requests:
            payload = json.loads(request.content)
            assert payload["parse_mode"] == "HTML"
            assert len(payload["text"]) <= 200
            _assert_valid_telegram_html(payload["text"])


class TestTelegramSending:
    """Test cases for delivery, rate limiting and errors."""

    @pytest.mark.asyncio
    async def test_flush_sends_send_message_request(self) -> None:
        """Records are delivered to the sendMessage endpoint."""
        requests: List[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        handler = _make_handler(respond)

        await handler.flush([
            _create_record("line 1"), _create_record("line 2")
        ])
        await handler.close()

        assert len(requests) == 1
        request = requests[0]
        assert request.method == "POST"
        assert str(request.url) == (
            f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        )
        payload = json.loads(request.content)
        assert payload == {
            "chat_id": CHAT_ID, "text": "line 1\nline 2"
        }

    @pytest.mark.asyncio
    async def test_flush_sends_multiple_messages(self) -> None:
        """An over-long batch is delivered as several messages."""
        requests: List[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        handler = _make_handler(respond, max_message_length=10)

        await handler.flush([
            _create_record("aaaa"),
            _create_record("bbbb"),
            _create_record("cccc"),
        ])
        await handler.close()

        texts = [
            json.loads(request.content)["text"] for request in requests
        ]
        assert texts == ["aaaa\nbbbb", "cccc"]

    @pytest.mark.asyncio
    async def test_429_retries_with_retry_after(self) -> None:
        """A 429 response is retried after parameters.retry_after."""
        call_count = 0

        def respond(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, json={
                    "ok": False,
                    "error_code": 429,
                    "description": "Too Many Requests: retry after 0",
                    "parameters": {"retry_after": 0},
                })
            return httpx.Response(200, json={"ok": True})

        handler = _make_handler(
            respond,
            batch_config=BatchConfig(max_retries=2, retry_delay=60.0),
        )

        # retry_delay=60 would stall the test if the retry_after hint
        # (0 seconds) were not used for the sleep
        await handler.flush([_create_record()])
        await handler.close()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_429_exhausts_retries(self) -> None:
        """Persistent 429 responses fail after max_retries."""
        call_count = 0

        def respond(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(429, json={
                "ok": False,
                "error_code": 429,
                "parameters": {"retry_after": 0},
            })

        handler = _make_handler(
            respond,
            batch_config=BatchConfig(max_retries=2, retry_delay=60.0),
        )

        with pytest.raises(NetworkError, match="429"):
            await handler.flush([_create_record()])

        assert call_count == 3
        await handler.close()

    @pytest.mark.asyncio
    async def test_other_client_errors_do_not_retry(self) -> None:
        """A 400 response fails immediately without retries."""
        call_count = 0

        def respond(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, json={
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: chat not found",
            })

        handler = _make_handler(
            respond,
            batch_config=BatchConfig(max_retries=3, retry_delay=0.01),
        )

        with pytest.raises(NetworkError, match="chat not found"):
            await handler.flush([_create_record()])

        assert call_count == 1
        await handler.close()

    @pytest.mark.asyncio
    async def test_network_error_hides_token(self) -> None:
        """The bot token does not leak into error messages."""

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Server Error")

        handler = _make_handler(
            respond,
            batch_config=BatchConfig(max_retries=0, retry_delay=0.01),
        )

        with pytest.raises(NetworkError) as exc_info:
            await handler.flush([_create_record()])

        assert TOKEN not in str(exc_info.value)
        assert "/bot***/" in str(exc_info.value)
        await handler.close()

    def test_retry_delay_parsing(self) -> None:
        """retry_after is read from the 429 body, with fallbacks."""
        handler = AsyncTelegramHandler(
            TOKEN, CHAT_ID,
            batch_config=BatchConfig(retry_delay=1.5),
        )

        body = json.dumps({"parameters": {"retry_after": 7}})
        assert handler._retry_delay_from_response(429, body) == 7.0

        # 429 without a usable hint still retries with the default delay
        assert handler._retry_delay_from_response(429, "not json") == 1.5
        assert handler._retry_delay_from_response(429, "{}") == 1.5

        # Other statuses keep the default policy
        assert handler._retry_delay_from_response(500, body) is None
        assert handler._retry_delay_from_response(400, body) is None


class TestTelegramConfig:
    """Test cases for creating the handler via ConfigManager."""

    def test_create_from_config(self) -> None:
        """The 'telegram' handler class is available in configs."""
        manager = ConfigManager()
        manager.load_from_dict({
            "version": 1,
            "loggers": {
                "app": {"level": "ERROR", "handlers": ["tg"]},
            },
            "handlers": {
                "tg": {
                    "class": "telegram",
                    "level": "ERROR",
                    "token": TOKEN,
                    "chat_id": CHAT_ID,
                    "parse_mode": "HTML",
                },
            },
        })

        logger = manager.get_logger("app")
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, AsyncTelegramHandler)
        assert handler.chat_id == CHAT_ID
        assert handler.parse_mode == "HTML"
        assert handler.level == logging.ERROR

    def test_config_requires_token(self) -> None:
        """A config without a token raises ConfigurationError."""
        manager = ConfigManager()
        manager.load_from_dict({
            "version": 1,
            "loggers": {"app": {"handlers": ["tg"]}},
            "handlers": {
                "tg": {"class": "telegram", "chat_id": CHAT_ID},
            },
        })
        with pytest.raises(ConfigurationError, match="token"):
            manager.get_logger("app")


if __name__ == "__main__":
    pytest.main([__file__])
