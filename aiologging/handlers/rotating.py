"""
Async rotating file handlers for aiologging.

This module provides async handlers for writing log records to files
with automatic rotation based on size or time intervals.
"""

from __future__ import annotations

import time
from datetime import datetime
from logging import LogRecord, NOTSET
from pathlib import Path
from typing import Optional, Union

from ..exceptions import RotationError
from ..types import (
    ErrorHandler,
    FilterProtocol,
    FormatterProtocol,
    TimeInterval,
)
from .file import AsyncFileHandlerWithRotation, _check_aiofiles


class AsyncRotatingFileHandler(AsyncFileHandlerWithRotation):
    """
    Async handler that rotates log files when they reach a specified size.

    This handler is similar to the standard logging.RotatingFileHandler but
    provides async functionality for non-blocking file I/O operations.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        max_bytes: int = 0,
        backup_count: int = 0,
        encoding: str = "utf-8",
        errors: Optional[str] = None,
        delay: bool = False,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[list[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
    ) -> None:
        """
        Initialize the async rotating file handler.

        Args:
            filename: The path to the log file
            mode: The file opening mode (defaults to 'a' for append)
            max_bytes: Maximum file size in bytes
                       before rotation (0 = no rotation)
            backup_count: Number of backup files to keep
            encoding: The file encoding (defaults to 'utf-8')
            errors: The error handling strategy for encoding
            delay: Whether to delay file opening until the first emit
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions

        Raises:
            DependencyError: If aiofiles is not installed
            RotationError: If rotation parameters are invalid
        """
        _check_aiofiles()

        if max_bytes <= 0:
            raise RotationError(
                "max_bytes must be positive for rotation",
                filename=str(filename),
                rotation_type="size",
                details={"max_bytes": max_bytes},
            )

        if backup_count < 0:
            raise RotationError(
                "backup_count must be non-negative",
                filename=str(filename),
                rotation_type="size",
                details={"backup_count": backup_count},
            )

        super().__init__(
            filename,
            mode,
            encoding,
            errors,
            delay,
            level,
            formatter,
            filters,
            error_handler,
            backup_count,
        )
        self.max_bytes = max_bytes

    async def _should_rotate(self, record: LogRecord) -> bool:
        """
        Check if the file should be rotated based on size.

        Args:
            record: The log record being emitted

        Returns:
            True if the file should be rotated
        """
        if self.max_bytes <= 0:
            return False

        try:
            if self._file is None:
                return False

            # Get current file size
            current_pos = int(await self._file.tell())
            return current_pos >= self.max_bytes

        except (OSError, AttributeError):
            return False

    async def _rotate_file(self) -> None:
        """Rotate the log file based on size."""
        if self._file is None:
            return

        try:
            # Close the current file
            await self._file.close()
            self._file = None

            # Rotate existing backups
            await self._rotate_backups()

            # Move current file to backup
            backup_file = self._get_backup_filename(1)
            if self.filename.exists():
                try:
                    self.filename.rename(backup_file)
                except OSError as e:
                    raise RotationError(
                        f"Failed to rotate file: {e}",
                        filename=str(self.filename),
                        rotation_type="size",
                    ) from e

            # Remove old backups if needed
            await self._remove_old_backups()

            # Reopen the file
            await self._open_file()

        except Exception as e:
            # Try to reopen the file even if rotation failed
            try:
                await self._open_file()
            except Exception:
                pass

            if isinstance(e, RotationError):
                raise
            else:
                raise RotationError(
                    f"Unexpected error during rotation: {e}",
                    filename=str(self.filename),
                    rotation_type="size",
                ) from e

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        formatter: Union[str, None] = None
        if self.formatter:
            formatter = type(self.formatter).__name__
        return (
            f"{self.__class__.__name__}(filename='{self.filename}', "
            f"max_bytes={self.max_bytes}, backup_count={self.backup_count}, "
            f"level={self.level}, "
            f"formatter={formatter})"
        )


class AsyncTimedRotatingFileHandler(AsyncFileHandlerWithRotation):
    """
    Async handler that rotates log files at specific time intervals.

    This handler is similar to the standard logging.TimedRotatingFileHandler
    but provides async functionality for non-blocking file I/O operations.
    """

    # Time interval mappings
    TIME_INTERVALS = {
        "S": 1,  # seconds
        "M": 60,  # minutes
        "H": 3600,  # hours
        "D": 86400,  # days
        "midnight": 86400,  # days at midnight
        "W0": 604800,  # Monday (7 days)
        "W1": 604800,  # Tuesday
        "W2": 604800,  # Wednesday
        "W3": 604800,  # Thursday
        "W4": 604800,  # Friday
        "W5": 604800,  # Saturday
        "W6": 604800,  # Sunday
    }

    def __init__(
        self,
        filename: Union[str, Path],
        when: TimeInterval = "H",
        interval: int = 1,
        backup_count: int = 0,
        encoding: str = "utf-8",
        errors: Optional[str] = None,
        delay: bool = False,
        utc: bool = False,
        at_time: Optional[datetime] = None,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[list[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
    ) -> None:
        """
        Initialize the async timed rotating file handler.

        Args:
            filename: The path to the log file
            when: The type of interval
                  ('S', 'M', 'H', 'D', 'midnight', 'W0'-'W6')
            interval: The interval between rotations
            backup_count: Number of backup files to keep
            encoding: The file encoding (defaults to 'utf-8')
            errors: The error handling strategy for encoding
            delay: Whether to delay file opening until the first emit
            utc: Whether to use UTC time for rotation
            at_time: Specific time for 'midnight' or weekly rotations
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions

        Raises:
            DependencyError: If aiofiles is not installed
            RotationError: If rotation parameters are invalid
        """
        _check_aiofiles()

        if when not in self.TIME_INTERVALS:
            raise RotationError(
                (
                    f"Invalid 'when' value: {when}. "
                    f"Must be one of {list(self.TIME_INTERVALS.keys())}"
                ),
                filename=str(filename),
                rotation_type="time",
                details={"when": when},
            )

        if interval <= 0:
            raise RotationError(
                "interval must be positive",
                filename=str(filename),
                rotation_type="time",
                details={"interval": interval},
            )

        if backup_count < 0:
            raise RotationError(
                "backup_count must be non-negative",
                filename=str(filename),
                rotation_type="time",
                details={"backup_count": backup_count},
            )

        super().__init__(
            filename,
            "a",
            encoding,
            errors,
            delay,
            level,
            formatter,
            filters,
            error_handler,
            backup_count,
        )
        self.when = when
        self.interval = interval
        self.utc = utc
        self.at_time = at_time
        self._rollover_at: float = self._compute_rollover_time()

    def _compute_rollover_time(self) -> float:
        """Compute the next rollover time based on the interval type."""
        current_time = time.time()

        if self.when == "midnight":
            # Rotate at midnight
            if self.utc:
                t = time.gmtime(current_time)
            else:
                t = time.localtime(current_time)

            current_hour = t[3]
            current_minute = t[4]
            current_second = t[5]

            # Time until midnight
            r = 86400 - (
                current_hour * 3600 + current_minute * 60 + current_second
            )

            if self.at_time is not None:
                # Use specific time
                at_hour = self.at_time.hour
                at_minute = self.at_time.minute
                at_second = self.at_time.second

                # Compute time until the specified time
                target_seconds = at_hour * 3600 + at_minute * 60 + at_second
                current_seconds = (
                    current_hour * 3600 + current_minute * 60 + current_second
                )

                if target_seconds > current_seconds:
                    r = target_seconds - current_seconds
                else:
                    r = 86400 - (current_seconds - target_seconds)

            return current_time + r

        elif self.when.startswith("W"):
            # Weekly rotation
            if self.utc:
                t = time.gmtime(current_time)
            else:
                t = time.localtime(current_time)

            current_weekday = t[6]  # Monday = 0, Sunday = 6
            target_weekday = int(self.when[1])

            # Compute days until target weekday
            days_until_target = (target_weekday - current_weekday) % 7
            if days_until_target == 0:
                days_until_target = 7  # Next week

            r = days_until_target * 86400

            if self.at_time is not None:
                # Adjust for specific time
                at_hour = self.at_time.hour
                at_minute = self.at_time.minute
                at_second = self.at_time.second

                current_hour = t[3]
                current_minute = t[4]
                current_second = t[5]

                target_seconds = at_hour * 3600 + at_minute * 60 + at_second
                current_seconds = (
                    current_hour * 3600 + current_minute * 60 + current_second
                )

                if target_seconds > current_seconds:
                    r += target_seconds - current_seconds
                else:
                    r += 86400 - (current_seconds - target_seconds)

            return current_time + r

        else:
            # 'S', 'M', 'H', 'D' - simple intervals
            interval_seconds = self.TIME_INTERVALS[self.when] * self.interval
            time_until_rotation: float = interval_seconds - (
                current_time % interval_seconds
            )
            return current_time + time_until_rotation

    async def _should_rotate(self, record: LogRecord) -> bool:
        """
        Check if the file should be rotated based on time.

        Args:
            record: The log record being emitted

        Returns:
            True if the file should be rotated
        """
        return time.time() >= self._rollover_at

    async def _rotate_file(self) -> None:
        """Rotate the log file based on time."""
        if self._file is None:
            return

        try:
            # Close the current file
            await self._file.close()
            self._file = None

            # Get the current time for the suffix
            if self.utc:
                time_tuple = time.gmtime(time.time())
            else:
                time_tuple = time.localtime(time.time())

            # Create the suffix based on the rotation type
            if self.when.startswith("W"):
                # Weekly rotation - use weekday
                suffix = time.strftime("%Y-%m-%d", time_tuple)
            else:
                # Other intervals - use appropriate format
                if self.when == "S":
                    suffix = time.strftime("%Y-%m-%d_%H-%M-%S", time_tuple)
                elif self.when == "M":
                    suffix = time.strftime("%Y-%m-%d_%H-%M", time_tuple)
                elif self.when == "H":
                    suffix = time.strftime("%Y-%m-%d_%H", time_tuple)
                else:  # 'D' or 'midnight'
                    suffix = time.strftime("%Y-%m-%d", time_tuple)

            # Rotate existing backups
            await self._rotate_backups_with_suffix(suffix)

            # Move current file to backup with suffix
            backup_file = self.filename.with_suffix(
                f"{self.filename.suffix}.{suffix}"
            )
            if self.filename.exists():
                try:
                    self.filename.rename(backup_file)
                except OSError as e:
                    raise RotationError(
                        f"Failed to rotate file: {e}",
                        filename=str(self.filename),
                        rotation_type="time",
                    ) from e

            # Remove old backups if needed
            await self._remove_old_backups_with_suffix()

            # Compute next rollover time
            self._rollover_at = self._compute_rollover_time()

            # Reopen the file
            await self._open_file()

        except Exception as e:
            # Try to reopen the file even if rotation failed
            try:
                await self._open_file()
            except Exception:
                pass

            if isinstance(e, RotationError):
                raise
            else:
                raise RotationError(
                    f"Unexpected error during rotation: {e}",
                    filename=str(self.filename),
                    rotation_type="time",
                ) from e

    def _get_timestamp_from_filename(self, file_path: Path) -> tuple[int, ...]:
        """
        Extract timestamp from filename suffix and parse it.

        Returns a tuple that can be used for sorting files chronologically.
        """
        suffix = file_path.suffix[1:]  # Remove the leading dot
        try:
            # Try to parse the timestamp based on the rotation interval
            if self.when == "S":
                # Format: YYYY-MM-DD_HH-MM-SS
                return tuple(map(lambda x: int(x), suffix.split('_')))
            elif self.when == "M":
                # Format: YYYY-MM-DD_HH-MM
                parts = suffix.split('_')
                date_parts = tuple(map(lambda x: int(x), parts[0].split('-')))
                time_parts = tuple(map(lambda x: int(x), parts[1].split('-')))
                return date_parts + time_parts
            elif self.when == "H":
                # Format: YYYY-MM-DD_HH
                parts = suffix.split('_')
                date_parts = tuple(map(lambda x: int(x), parts[0].split('-')))
                return date_parts + (int(parts[1]),)
            else:  # 'D', 'midnight', or weekly
                # Format: YYYY-MM-DD
                return tuple(map(lambda x: int(x), suffix.split('-')))
        except (ValueError, IndexError):
            # Fallback to modification time if parsing fails
            return (int(file_path.stat().st_mtime),)

    async def _rotate_backups_with_suffix(self, suffix: str) -> None:
        """Rotate existing backup files with time suffixes."""
        if self.backup_count <= 0:
            return

        # Find existing backup files with time suffixes
        backup_files = []
        for file_path in self.filename.parent.glob(f"{self.filename.name}.*"):
            if file_path != self.filename:
                backup_files.append(file_path)

        # Sort by the timestamp in the filename (newest first)
        backup_files.sort(key=self._get_timestamp_from_filename, reverse=True)

        # Remove excess backups
        for i, backup_file in enumerate(backup_files):
            if i >= self.backup_count:
                try:
                    backup_file.unlink()
                except OSError as e:
                    import sys

                    sys.stderr.write(
                        "Failed to remove "
                        f"old backup file {backup_file}: {e}\n"
                    )

    async def _remove_old_backups_with_suffix(self) -> None:
        """
        Remove old backup files
        with time suffixes if backup_count is exceeded.
        """
        if self.backup_count <= 0:
            return

        # Find all backup files
        backup_files = []
        for file_path in self.filename.parent.glob(f"{self.filename.name}.*"):
            if file_path != self.filename:
                backup_files.append(file_path)

        # Sort by the timestamp in the filename (newest first)
        backup_files.sort(key=self._get_timestamp_from_filename, reverse=True)

        # Remove excess backups
        for i, backup_file in enumerate(backup_files):
            if i >= self.backup_count:
                try:
                    backup_file.unlink()
                except OSError as e:
                    import sys

                    sys.stderr.write(
                        "Failed to remove "
                        f"old backup file {backup_file}: {e}\n"
                    )

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        formatter: Union[str, None] = None
        if self.formatter:
            formatter = type(self.formatter).__name__
        return (
            f"{self.__class__.__name__}(filename='{self.filename}', "
            f"when='{self.when}', interval={self.interval}, "
            f"backup_count={self.backup_count}, level={self.level}, "
            f"formatter={formatter})"
        )
