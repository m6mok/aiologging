"""Optional HTTP and Telegram adapters. Resources belong to the engine loop."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import importlib.util
import json
import logging
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any
from ..journal import diagnose
from ..durable import _IN_DELIVERY
from .files import _render


def _transport_filter(record: logging.LogRecord) -> bool:
    """Do not feed our own transport diagnostics back into its HTTP sink."""
    return not _IN_DELIVERY.get()


def _protect_transport_logs() -> None:
    for name in list(logging.Logger.manager.loggerDict) + [
        "httpx",
        "httpcore",
        "aiohttp",
    ]:
        if name.startswith(("httpx", "httpcore", "aiohttp")):
            logging.getLogger(name).addFilter(_transport_filter)


def _fallback_identity(
    function: Callable[..., Any], explicit: str | None
) -> Any:
    if explicit:
        return explicit
    owner = getattr(function, "__self__", None)
    if isinstance(owner, logging.FileHandler):
        return ["file", owner.baseFilename, owner.name]
    if isinstance(owner, logging.StreamHandler):
        if owner.stream is sys.stdout or owner.stream is sys.stderr:
            return [
                "stream",
                "stdout" if owner.stream is sys.stdout else "stderr",
                owner.name,
            ]
    identity = getattr(owner, "recipient_id", None)
    if identity:
        return identity
    diagnose(
        "custom network fallback needs fallback_id or recipient_id for "
        "automatic restart recovery"
    )
    return "ephemeral:" + uuid.uuid4().hex


class HTTPHandler(logging.Handler):
    """Confirmed batched HTTP delivery, with a required explicit fallback.

    Args:
        url: Original endpoint, included only as a hash in recipient identity.
        fallback: A callable returning True only after successful delivery.
            A confirmed file handler's deliver method is a useful fallback.
        backend: Optional installed transport, httpx or aiohttp.
        session_factory: A factory called on the library loop. Ready-made
            sessions from an application's event loop are not accepted.
        recipient_id: Explicit stable destination identity, if needed.
        primary_timeout: Total primary-attempt budget, including transport I/O.
        fallback_timeout: Separate fallback-attempt budget.
    """

    def __init__(
        self,
        url: str,
        *,
        fallback: Callable[[Sequence[logging.LogRecord]], Any],
        backend: str | None = None,
        session_factory: Callable[[], Any] | None = None,
        recipient_id: str | None = None,
        fallback_id: str | None = None,
        headers: Mapping[str, str] | None = None,
        primary_timeout: float = 5.0,
        fallback_timeout: float = 5.0,
        batch_size: int = 32,
        retry_delay: float = 1.0,
        level: int = logging.NOTSET,
    ) -> None:
        super().__init__(level)
        if not callable(fallback):
            raise ValueError("HTTP delivery requires an explicit fallback")
        if backend is None:
            if session_factory is not None or importlib.util.find_spec(
                "httpx"
            ):
                backend = "httpx"
            elif importlib.util.find_spec("aiohttp"):
                backend = "aiohttp"
            else:
                raise ImportError(
                    "Install aiologging[httpx] or aiologging[aiohttp]"
                )
        if backend not in ("httpx", "aiohttp"):
            raise ValueError("backend must be httpx or aiohttp")
        if primary_timeout <= 0 or fallback_timeout <= 0 or batch_size < 1:
            raise ValueError("Invalid delivery budget or batch size")
        self._url = url
        self._fallback = fallback
        self.backend = backend
        self.session_factory = session_factory
        self._headers = MappingProxyType(dict(headers or {}))
        self.primary_timeout = primary_timeout
        self.fallback_timeout = fallback_timeout
        self.batch_size = batch_size
        self.retry_delay = retry_delay
        backup_identity = (
            "explicit-recipient-contract"
            if recipient_id
            else _fallback_identity(fallback, fallback_id)
        )
        descriptor = json.dumps(
            [
                url,
                sorted(self.headers.items()),
                type(self).__qualname__,
                getattr(self, "chat_id", None),
                getattr(self, "parse_mode", None),
                backup_identity,
            ]
        )
        self.recipient_id = recipient_id or (
            "http:" + hashlib.sha256(descriptor.encode()).hexdigest()
        )
        self._session: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def url(self) -> str:
        """Original endpoint; replace the handler to change the destination."""
        return self._url

    @property
    def headers(self) -> Mapping[str, str]:
        return self._headers

    @property
    def fallback(self) -> Callable[[Sequence[logging.LogRecord]], Any]:
        return self._fallback

    async def _get_session(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError(
                "HTTP resources belong to the library event loop"
            )
        if self._session is None:
            _protect_transport_logs()
            if self.session_factory is not None:
                session = self.session_factory()
                if inspect.isawaitable(session):
                    session = await session
            elif self.backend == "httpx":
                import httpx

                _protect_transport_logs()

                session = httpx.AsyncClient(timeout=self.primary_timeout)
            else:
                import aiohttp

                _protect_transport_logs()

                session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.primary_timeout)
                )
            self._session = session
            self._loop = loop
            if getattr(session, "_loop", loop) is not loop:
                self._session = None
                self._loop = None
                raise RuntimeError(
                    "Do not supply sessions from an application loop"
                )
            _protect_transport_logs()
        return self._session

    def _body(self, records: Sequence[logging.LogRecord]) -> dict[str, Any]:
        return {"json": [record.__dict__ for record in records]}

    async def _request(self, body: dict[str, Any]) -> bool:
        session = await self._get_session()
        response = await session.request(
            "POST", self.url, headers=self.headers, **body
        )
        if self.backend == "aiohttp":
            try:
                await response.read()
                return bool(200 <= response.status < 300)
            finally:
                response.release()
        return bool(200 <= response.status_code < 300)

    async def deliver(self, records: Sequence[logging.LogRecord]) -> bool:
        """True confirms an HTTP 2xx response for the complete batch."""
        return await self._request(self._body(records))

    def emit(self, record: logging.LogRecord) -> None:
        raise RuntimeError("Configure aiologging before using an HTTP adapter")

    async def aclose(self) -> None:
        if self._session is not None:
            if self.backend == "httpx":
                await self._session.aclose()
            else:
                await self._session.close()
            self._session = None
            self._loop = None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} recipient={self.recipient_id}>"


class HTTPTextHandler(HTTPHandler):
    """Send formatted records as one newline-separated body."""

    _format_at_acceptance = True

    def _body(self, records: Sequence[logging.LogRecord]) -> dict[str, Any]:
        text = "\n".join(_render(self, record) for record in records)
        return {"content" if self.backend == "httpx" else "data": text}


class TelegramHandler(HTTPHandler):
    """Telegram sendMessage adapter with chunking and independent fallback."""

    _format_at_acceptance = True

    def __init__(
        self,
        token: str,
        chat_id: str | int,
        *,
        fallback: Callable[[Sequence[logging.LogRecord]], Any],
        api_base_url: str = "https://api.telegram.org",
        parse_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._chat_id = chat_id
        self._parse_mode = parse_mode
        url = api_base_url.rstrip("/") + "/bot" + token + "/sendMessage"
        super().__init__(url, fallback=fallback, **kwargs)

    @property
    def chat_id(self) -> str | int:
        return self._chat_id

    @property
    def parse_mode(self) -> str | None:
        return self._parse_mode

    async def deliver(self, records: Sequence[logging.LogRecord]) -> bool:
        # Python's character count is conservative for Telegram's UTF-16
        # entities, so count UTF-16 units when constructing 4096-unit chunks.
        text = "\n".join(_render(self, record) for record in records)
        chunks: list[str] = []
        current: list[str] = []
        units = 0
        for character in text or " ":
            length = 2 if ord(character) > 0xFFFF else 1
            if units + length > 4096:
                chunks.append("".join(current))
                current = []
                units = 0
            current.append(character)
            units += length
        chunks.append("".join(current))
        for chunk in chunks:
            body: dict[str, Any] = {"chat_id": self.chat_id, "text": chunk}
            if self.parse_mode:
                body["parse_mode"] = self.parse_mode
            session = await self._get_session()
            response = await session.request(
                "POST", self.url, json=body, headers=self.headers
            )
            if self.backend == "aiohttp":
                try:
                    data = await response.json()
                    status = response.status
                finally:
                    response.release()
            else:
                data = response.json()
                status = response.status_code
            if not 200 <= status < 300 or data.get("ok") is not True:
                if status == 429:
                    delay = data.get("parameters", {}).get("retry_after", 0)
                    if isinstance(delay, (int, float)) and delay > 0:
                        await asyncio.sleep(delay)
                return False
        return True


setattr(HTTPHandler.deliver, "_prepared_rendering", True)
setattr(TelegramHandler.deliver, "_prepared_rendering", True)
