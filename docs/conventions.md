# Code conventions

## Typing

- **mypy strict with `python_version = "3.9"`** (`[tool.mypy]` in
  `pyproject.toml`). Everything in `aiologging/` must type-check on
  3.9 semantics:
  - no `X | Y` unions or `list[str]` in runtime positions — use
    `typing.Optional/Union/List/Dict` (modern syntax is fine inside
    `from __future__ import annotations` files, but stay consistent
    with the surrounding code);
  - `ParamSpec` / `TypeAlias` come from `typing_extensions` under a
    `sys.version_info` check (see the top of `types.py`);
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
- **Locks** — use `utils.LazyLock`, not `asyncio.Lock()`, in any
  constructor that may run without a running event loop (Python 3.9
  binds a Lock to the loop at construction time).
- **Error handling in handlers** — never let logging crash the app:
  catch, count in metrics, route through
  `handle_error_with_fallback` / the configured `error_handler`,
  fall back to `sys.stderr.write`.
- **Secrets** — must not leak into exception text or `repr`. For
  URL-embedded secrets override `_redacted_url()`
  (see `AsyncTelegramHandler`).
- **Buffered handlers** — implement `emit()` as
  `await self.handle(record)` and a no-op `_emit()`; real sending
  happens in `flush(records)`. The base class owns buffering,
  auto-flush and retries.
