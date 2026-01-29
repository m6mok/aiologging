"""
Async file handler for aiologging.

This module provides async handlers for writing log records to files
using aiofiles for non-blocking file I/O operations.
"""

from __future__ import annotations

import asyncio
import os
from logging import LogRecord, NOTSET
from pathlib import Path
from typing import Any, Optional, Union, Protocol, runtime_checkable

from ..exceptions import DependencyError, FileError, HandlerError
from ..types import ErrorHandler, FilterProtocol, FormatterProtocol
from ..utils import log_error_to_stderr
from .base import AsyncHandler

# Define a protocol for aiofiles file objects
@runtime_checkable
class AsyncFileProtocol(Protocol):
    """Protocol for aiofiles file objects."""
    async def write(self, s: str) -> None: ...
    async def flush(self) -> None: ...
    async def close(self) -> None: ...
    async def tell(self) -> int: ...

# Try to import aiofiles
try:
    import aiofiles
    import aiofiles.threadpool
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None  # type: ignore


def _check_aiofiles() -> None:
    """Check if aiofiles is available."""
    if not AIOFILES_AVAILABLE:
        raise DependencyError(
            "aiofiles is required for AsyncFileHandler. "
            "Install it with: pip install aiologging[aiofiles]",
            dependency_name="aiofiles",
            install_command="pip install aiologging[aiofiles]",
        )


class AsyncFileHandler(AsyncHandler):
    """
    Async handler that writes log records to a file.

    This handler uses aiofiles for non-blocking file I/O operations,
    making it suitable for high-performance logging applications.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        encoding: str = "utf-8",
        errors: Optional[str] = None,
        delay: bool = False,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[list[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
    ) -> None:
        """
        Initialize the async file handler.

        Args:
            filename: The path to the log file
            mode: The file opening mode (defaults to 'a' for append)
            encoding: The file encoding (defaults to 'utf-8')
            errors: The error handling strategy for encoding
            delay: Whether to delay file opening until the first emit
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions

        Raises:
            DependencyError: If aiofiles is not installed
            FileError: If the file path is invalid
        """
        _check_aiofiles()

        super().__init__(level, formatter, filters, error_handler)

        self.filename = Path(filename)
        self.mode = mode
        self.encoding = encoding
        self.errors = errors or "strict"
        self.delay = delay
        self._file: Optional[AsyncFileProtocol] = None  # aiofiles file object
        self._write_lock = asyncio.Lock()

        # Validate the file path
        self._validate_file_path()

        # Open the file immediately if not delayed
        if not delay:
            self._schedule_file_open()

    def _validate_file_path(self) -> None:
        """Validate the file path and create parent directories if needed."""
        try:
            # Create parent directories if they don't exist
            self.filename.parent.mkdir(parents=True, exist_ok=True)

            # Check if we can write to the directory
            if not os.access(self.filename.parent, os.W_OK):
                raise FileError(
                    f"Cannot write to directory: {self.filename.parent}",
                    filename=str(self.filename),
                    operation="validate_path",
                    errno=13,  # Permission denied
                )

        except OSError as e:
            raise FileError(
                f"Invalid file path: {e}",
                filename=str(self.filename),
                operation="validate_path",
                errno=e.errno,
            ) from e

    def _schedule_file_open(self) -> None:
        """Schedule file opening if an event loop is running."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            loop.create_task(self._open_file())
        except RuntimeError:
            # No running event loop, so we'll open the file on first emit
            pass

    async def _open_file(self) -> None:
        """Open the log file for writing."""
        if self._file is not None:
            return

        try:
            # Use a simpler approach to avoid aiofiles type issues
            self._file = await aiofiles.threadpool.open(
                str(self.filename),
                mode=self.mode,
                encoding=self.encoding,
                errors=self.errors,
            )  # type: ignore
            # Type: ignore for aiofiles.open type issues
            # The actual type is correct at runtime, mypy just has trouble with it
        except Exception as e:
            raise FileError(
                f"Failed to open file: {e}",
                filename=str(self.filename),
                operation="open",
                errno=getattr(e, 'errno', None),
            ) from e

    async def _ensure_file_open(self) -> None:
        """Ensure the file is open, opening it if necessary."""
        if self._file is None:
            await self._open_file()

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        """
        Emit a log record to the file.

        Args:
            record: The log record to emit
            formatted_message: The formatted log message

        Raises:
            HandlerError: If writing to the file fails
        """
        try:
            # Ensure file is open
            if self.delay:
                await self._ensure_file_open()

            if self._file is None:
                raise HandlerError(
                    "File is not open",
                    handler_name=type(self).__name__,
                    operation="_emit",
                    details={"filename": str(self.filename)},
                )

            msg = self._prepare_message(formatted_message)

            async with self._write_lock:
                await self._file.write(msg)
                await self._file.flush()

        except Exception as e:
            raise HandlerError(
                f"Failed to emit log record to file: {e}",
                handler_name=type(self).__name__,
                operation="_emit",
                details={
                    "filename": str(self.filename),
                    "record_level": record.levelname,
                    "record_message": record.getMessage(),
                },
            ) from e

    def _prepare_message(self, formatted_message: str) -> str:
        """Prepare the message for writing by ensuring it ends with a newline."""
        msg = formatted_message
        if not msg.endswith('\n'):
            msg += '\n'
        return msg

    async def _close_resources(self) -> None:
        """Close the file if it's open."""
        if self._file is not None:
            try:
                await self._file.close()
            except Exception as e:
                # Log the error but don't raise it during cleanup
                log_error_to_stderr(f"Error closing file {self.filename}", e)
            finally:
                self._file = None

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        return (
            f"{self.__class__.__name__}(filename='{self.filename}', "
            f"mode='{self.mode}', level={self.level}, "
            f"formatter={type(self.formatter).__name__ if self.formatter else None})"
        )

    async def __aenter__(self) -> AsyncFileHandler:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


class AsyncFileHandlerWithRotation(AsyncFileHandler):
    """
    Base class for async file handlers with rotation support.

    This class provides common functionality for file handlers that
    need to rotate log files based on size or time.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        encoding: str = "utf-8",
        errors: Optional[str] = None,
        delay: bool = False,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[list[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
        backup_count: int = 0,
    ) -> None:
        """
        Initialize the async file handler with rotation support.

        Args:
            filename: The path to the log file
            mode: The file opening mode (defaults to 'a' for append)
            encoding: The file encoding (defaults to 'utf-8')
            errors: The error handling strategy for encoding
            delay: Whether to delay file opening until the first emit
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions
            backup_count: Number of backup files to keep
        """
        super().__init__(
            filename, mode, encoding, errors, delay, level, formatter, filters, error_handler
        )
        self.backup_count = backup_count
        self._rotation_lock = asyncio.Lock()

    async def _rotate_file(self) -> None:
        """
        Rotate the log file.

        This method must be implemented by subclasses to define
        the specific rotation logic.
        """
        raise NotImplementedError

    def _get_backup_filename(self, backup_number: int) -> Path:
        """
        Get the filename for a backup file.

        Args:
            backup_number: The backup number (1 for the first backup)

        Returns:
            The path for the backup file
        """
        if backup_number == 1:
            return self.filename.with_suffix(self.filename.suffix + ".1")
        else:
            return self.filename.with_suffix(f"{self.filename.suffix}.{backup_number}")

    def _log_file_error(self, message: str, file_path: Path) -> None:
        """Log file operation errors to stderr."""
        log_error_to_stderr(f"{message}: {file_path}")

    async def _remove_old_backups(self) -> None:
        """Remove old backup files if backup_count is exceeded."""
        if self.backup_count <= 0:
            return

        # Remove backups that exceed the backup count
        for i in range(self.backup_count + 1, 100):  # Reasonable upper limit
            backup_file = self._get_backup_filename(i)
            if backup_file.exists():
                try:
                    backup_file.unlink()
                except OSError as e:
                    # Log the error but don't raise it
                    self._log_file_error(f"Failed to remove old backup file: {e}", backup_file)
            else:
                break

    async def _rotate_backups(self) -> None:
        """Rotate existing backup files."""
        if self.backup_count <= 0:
            return

        # Rotate existing backups
        for i in range(self.backup_count, 0, -1):
            current_backup = self._get_backup_filename(i)
            next_backup = self._get_backup_filename(i + 1)

            if current_backup.exists():
                if i >= self.backup_count:
                    # Remove the oldest backup if it exceeds the count
                    try:
                        current_backup.unlink()
                    except OSError as e:
                        self._log_file_error(f"Failed to remove backup file: {e}", current_backup)
                else:
                    # Move the backup to the next number
                    try:
                        current_backup.rename(next_backup)
                    except OSError as e:
                        self._log_file_error(f"Failed to rotate backup file: {e}", current_backup)

    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        """
        Emit a log record to the file, with rotation if needed.

        Args:
            record: The log record to emit
            formatted_message: The formatted log message
        """
        # Check if rotation is needed before emitting
        if await self._should_rotate(record):
            async with self._rotation_lock:
                # Double-check rotation condition after acquiring lock
                if await self._should_rotate(record):
                    await self._rotate_file()

        # Emit the record
        await super()._emit(record, formatted_message)

    async def _should_rotate(self, record: LogRecord) -> bool:
        """
        Check if the file should be rotated.

        This method must be implemented by subclasses to define
        the specific rotation conditions.

        Args:
            record: The log record being emitted

        Returns:
            True if the file should be rotated
        """
        raise NotImplementedError
