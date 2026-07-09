"""
Async Telegram handler for aiologging.

This module provides an async handler that sends log records to a
Telegram chat via the Bot API ``sendMessage`` method, respecting the
4096-character message limit and the ``retry_after`` hint of
429 (Too Many Requests) responses.
"""

from __future__ import annotations

import json
import ssl
import threading
import urllib.request
from logging import LogRecord, NOTSET
from typing import Any, Dict, List, Optional, Union

from ..exceptions import ConfigurationError
from ..types import (
    INLINE_SENDER_THREAD_ATTR,
    BatchConfig,
    ErrorHandler,
    FilterProtocol,
    FormatterProtocol,
    HeadersType,
    HttpBackendType,
)
from .http import AsyncHttpHandlerBase

# Hard limit of the Bot API for the text of a single message
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

TELEGRAM_API_BASE_URL = "https://api.telegram.org"


class AsyncTelegramHandler(AsyncHttpHandlerBase):
    """
    Async handler that sends log records to a Telegram chat.

    Buffered records are formatted, joined with newlines and split
    into as few ``sendMessage`` calls as possible, each within the
    4096-character limit (a single record longer than the limit is
    hard-split). A 429 response is retried after the delay the Bot
    API returns in ``parameters.retry_after``.

    Note:
        With ``parse_mode`` set (e.g. "HTML" or "MarkdownV2") the
        formatter is responsible for escaping the message text;
        splitting an over-long record may also break markup entities.

    Example:
        >>> handler = AsyncTelegramHandler(
        ...     token="123456:ABC-DEF...",
        ...     chat_id="-1001234567890",
        ...     level=logging.ERROR,
        ... )
        >>> logger.addHandler(handler)
    """

    def __init__(
        self,
        token: str,
        chat_id: Union[str, int],
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
        message_thread_id: Optional[int] = None,
        max_message_length: int = TELEGRAM_MAX_MESSAGE_LENGTH,
        api_base_url: str = TELEGRAM_API_BASE_URL,
        headers: Optional[HeadersType] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[List[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
        batch_config: Optional[BatchConfig] = None,
        backend: Optional[HttpBackendType] = None,
    ) -> None:
        """
        Initialize the async Telegram handler.

        Args:
            token: The bot token from @BotFather
            chat_id: Target chat: unique identifier or "@channelname"
            parse_mode: Message formatting mode ("HTML", "Markdown",
                        "MarkdownV2"); plain text if None
            disable_notification: Send messages silently
            message_thread_id: Target thread of a forum supergroup
            max_message_length: Maximum characters per message
                                (1..4096, the Bot API hard limit)
            api_base_url: Base URL of the Bot API (override for a
                          local Bot API server or tests)
            headers: Additional HTTP headers
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions
            batch_config: Configuration for batch processing
            backend: HTTP client backend to use ('aiohttp' or 'httpx');
                     if None, aiohttp is used when installed,
                     falling back to httpx

        Raises:
            ConfigurationError: If token, chat_id or
                                max_message_length is invalid
            DependencyError: If no async HTTP client is installed
        """
        if not token:
            raise ConfigurationError(
                "Telegram handler requires a bot token",
                config_key="token",
            )
        if chat_id is None or chat_id == "":
            raise ConfigurationError(
                "Telegram handler requires a chat_id",
                config_key="chat_id",
            )
        if not 0 < max_message_length <= TELEGRAM_MAX_MESSAGE_LENGTH:
            raise ConfigurationError(
                "max_message_length must be within "
                f"1..{TELEGRAM_MAX_MESSAGE_LENGTH}",
                config_key="max_message_length",
                config_value=max_message_length,
            )

        url = f"{api_base_url.rstrip('/')}/bot{token}/sendMessage"

        super().__init__(
            url=url,
            method="POST",
            headers=headers,
            timeout=timeout,
            verify_ssl=verify_ssl,
            level=level,
            formatter=formatter,
            filters=filters,
            error_handler=error_handler,
            batch_config=batch_config,
            backend=backend,
        )

        self.token = token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self.disable_notification = disable_notification
        self.message_thread_id = message_thread_id
        self.max_message_length = max_message_length

    async def emit(self, record: LogRecord) -> None:
        """
        Emit a log record by adding it to the buffer.

        Args:
            record: The log record to emit
        """
        await self.handle(record)

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        """Emit a log record (buffered handler uses flush instead)."""
        # This is a buffered handler, so individual records
        # are not emitted directly
        # They are collected and sent in batches via flush()
        pass

    def _redacted_url(self) -> str:
        """URL with the bot token hidden, for errors and repr."""
        return self.url.replace(f"/bot{self.token}/", "/bot***/")

    def _split_text(self, text: str) -> List[str]:
        """Split a single over-long text into message-sized parts."""
        if len(text) <= self.max_message_length:
            return [text]
        return [
            text[i:i + self.max_message_length]
            for i in range(0, len(text), self.max_message_length)
        ]

    def _build_messages(self, records: List[LogRecord]) -> List[str]:
        """
        Combine formatted records into as few messages as possible.

        Records are joined with newlines; a new message is started
        when adding the next record would exceed the message length
        limit.

        Args:
            records: The log records to combine

        Returns:
            The list of message texts to send
        """
        messages: List[str] = []
        current = ""
        for record in records:
            for part in self._split_text(self.format(record)):
                if not current:
                    current = part
                elif (
                    len(current) + 1 + len(part) <= self.max_message_length
                ):
                    current = f"{current}\n{part}"
                else:
                    messages.append(current)
                    current = part
        if current:
            messages.append(current)
        return messages

    def _build_payload(self, text: str) -> Dict[str, Any]:
        """Build the ``sendMessage`` JSON payload for one message."""
        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
        }
        if self.parse_mode is not None:
            payload["parse_mode"] = self.parse_mode
        if self.disable_notification:
            payload["disable_notification"] = True
        if self.message_thread_id is not None:
            payload["message_thread_id"] = self.message_thread_id
        return payload

    def emit_sync(self, record: LogRecord, timeout: float) -> bool:
        """
        Deliver one record synchronously via ``urllib``, bypassing
        the queue — the stdlib bridge calls this for critical records
        when ``captureStdlib(inline_level=...)`` is set.

        The POST runs in a short-lived helper thread joined with the
        deadline: urllib's own ``timeout`` does not cover DNS
        resolution (it starts before the socket exists), so a bare
        call could block the caller far longer than requested. On any
        failure — timeout, network error, non-2xx response — False is
        returned and the record stays on the normal queue path.

        Args:
            record: The log record to deliver
            timeout: Wall-clock budget in seconds for all chunks

        Returns:
            True if every message chunk got a 2xx response in time
        """
        if self._closed or timeout <= 0:
            return False
        if not self.filter(record):
            return False
        try:
            payloads = [
                self._build_payload(text)
                for text in self._build_messages([record])
            ]
        except Exception:
            return False
        if not payloads:
            return False

        outcome: List[bool] = []

        def send() -> None:
            outcome.append(
                all(self._post_sync(p, timeout) for p in payloads)
            )

        thread = threading.Thread(
            target=send, name="aiologging-telegram-inline", daemon=True
        )
        # Anything logging from inside the send must not trigger a
        # nested inline delivery (the bridge checks this attribute)
        setattr(thread, INLINE_SENDER_THREAD_ATTR, True)
        thread.start()
        thread.join(timeout)
        return bool(outcome) and outcome[0]

    def _post_sync(self, payload: Dict[str, Any], timeout: float) -> bool:
        """POST one ``sendMessage`` payload with urllib; True on 2xx."""
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**self.headers, "Content-Type": "application/json"},
            method="POST",
        )
        context: Optional[ssl.SSLContext] = None
        if not self.verify_ssl:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=context
            ) as response:
                return bool(200 <= response.status < 300)
        except Exception:
            return False

    async def _prepare_request_data(
        self, records: List[LogRecord]
    ) -> List[Dict[str, Any]]:
        """Prepare one ``sendMessage`` payload per message chunk."""
        return [
            self._build_payload(text)
            for text in self._build_messages(records)
        ]

    def _get_content_type(self) -> str:
        """Get the content type for the Bot API requests."""
        return "application/json"

    async def _make_request_with_retries(
        self,
        session: Any,
        headers: HeadersType,
        request_data: Any,
        records: List[LogRecord],
    ) -> None:
        """
        Send each prepared ``sendMessage`` payload with retries.

        A batch of records may need several messages; each is sent
        (and retried) independently.

        Args:
            session: The HTTP client (aiohttp.ClientSession
                     or httpx.AsyncClient)
            headers: Request headers
            request_data: The list of ``sendMessage`` payloads
            records: The original log records (for error reporting)

        Raises:
            NetworkError: If a message cannot be delivered
        """
        for payload in request_data:
            await super()._make_request_with_retries(
                session, headers, payload, records
            )

    def _retry_delay_from_response(
        self, status: int, error_text: str
    ) -> Optional[float]:
        """
        Honour the ``retry_after`` hint of a 429 Bot API response.

        Args:
            status: The HTTP status code of the response
            error_text: The response body as text

        Returns:
            The delay in seconds before the next attempt, or None
            to keep the default retry policy
        """
        if status != 429:
            return None
        try:
            body = json.loads(error_text)
            retry_after = body.get("parameters", {}).get("retry_after")
        except (ValueError, AttributeError):
            retry_after = None
        if isinstance(retry_after, (int, float)) and retry_after >= 0:
            return float(retry_after)
        # 429 without a usable hint: still retry, with the default delay
        return self.batch_config.retry_delay
