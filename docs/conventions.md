# Code conventions

## Typing

- **mypy strict with `python_version = "3.11"`** (`[tool.mypy]` in
  `pyproject.toml`). Everything in `aiologging/` must type-check on
  3.11 semantics:
  - modern unions and collection generics are supported;
  - prefer collections.abc for runtime protocols, and keep optional
    dependencies out of the core;
  - all functions fully annotated, including tests' helper functions
    when practical (only `aiologging/` is mypy-checked).
- Public callable contracts are `Protocol` classes in `types.py`
  (`FormatterProtocol`, `FilterProtocol`, `AuthenticatorProtocol`),
  not ABCs — stdlib objects like `logging.Formatter` satisfy them
  structurally.

## Style

- **79-character lines** in library code (flake8 default; CI's
  blocking flake8 pass checks errors only, but keep the tree clean at
  default settings).
- Every module, class and method carries a docstring; public ones use
  sectioned docstrings (`Args:` / `Returns:` / `Raises:`, plus
  `Example:` where a snippet helps). Follow the file you are editing.
- stdlib-compatible API keeps stdlib naming (`getLogger`, `setLevel`,
  `addHandler`, `camelCase` where logging uses it); internal/new code
  uses snake_case. Private methods are `_underscore`-prefixed.

## Recurring patterns

- **Optional dependencies** — import at module top in
  `try/except ImportError`, set a module-level flag, raise
  `DependencyError` with the install command on use:

  ```python
  try:
      import aiohttp
      AIOHTTP_AVAILABLE = True
  except ImportError:
      AIOHTTP_AVAILABLE = False

  def _check_aiohttp() -> None:
      if not AIOHTTP_AVAILABLE:
          raise DependencyError(
              "aiohttp is required for ...",
              dependency_name="aiohttp",
              install_command="pip install aiologging[aiohttp]",
          )
  ```

  Tests patch the flag to exercise both branches. New optional deps
  also need an extra in `pyproject.toml` (and inclusion in `all` and
  `dev`).
- **Exceptions** — subclass the right branch of `AiologgingError`
  and pass structured context as keywords
  (`NetworkError(..., url=…, status_code=…)`); they land in
  `.details` and as attributes. Never raise bare `Exception`.
- **Resource ownership** — async resources are created on the library loop.
  OS recipient locks span I/O and ACK. Do not use expiry to steal a live
  sender's task. Legacy Async* implementations still use utils.LazyLock.
- **Acceptance and delivery errors** — never return false acceptance.
  Producer failures raise and diagnose on stderr. Delivery failures preserve
  the journal task and diagnose; confirmed handlers propagate write errors
  instead of swallowing them. Legacy Async* handlers retain their historical
  handle_error_with_fallback behaviour only in the compatibility API.
- **Secrets** — must not leak into exception text or `repr`. For
  URL-embedded secrets override `_redacted_url()`
  (see `AsyncTelegramHandler`).
- **Legacy buffered handlers** — implement `emit()` as
  `await self.handle(record)` and a no-op `_emit()`; real sending
  happens in `flush(records)`. The base class owns buffering,
  auto-flush and retries.
