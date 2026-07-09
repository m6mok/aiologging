"""
Tests for the Telegram handler.
"""

import json
import logging
from typing import Any, Callable, List

import httpx
import pytest

from aiologging.config import ConfigManager
from aiologging.handlers.telegram import (
    AsyncTelegramHandler,
    TELEGRAM_MAX_MESSAGE_LENGTH,
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
