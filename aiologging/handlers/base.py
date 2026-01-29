"""
Base async handler implementation for aiologging.

This module provides the abstract base class for all async handlers
in the aiologging library with enhanced performance and error handling.
"""

from __future__ import annotations

import asyncio
import sys
import time
from abc import abstractmethod
from logging import LogRecord, NOTSET
from typing import Any, List, Optional, Union
from weakref import WeakSet

from ..exceptions import FormatterError, HandlerClosedError
from ..types import (
    AsyncHandlerABC,
    ErrorHandler,
    FilterProtocol,
    FormatterProtocol,
    RateLimiter,
    RetryStrategy,
)
from ..utils import handle_error_with_fallback


class AsyncHandlerMetrics:
    """Metrics collection for async handlers."""

    def __init__(self, enable_metrics: bool = False) -> None:
        self.enable_metrics = enable_metrics
        self._records_processed = 0
        self._records_dropped = 0
        self._errors_count = 0
        self._last_error_time = 0.0

    def increment_processed(self) -> None:
        """Increment processed records count."""
        if self.enable_metrics:
            self._records_processed += 1

    def increment_dropped(self) -> None:
        """Increment dropped records count."""
        if self.enable_metrics:
            self._records_dropped += 1

    def increment_errors(self) -> None:
        """Increment errors count and update last error time."""
        if self.enable_metrics:
            self._errors_count += 1
            self._last_error_time = time.time()

    def get_metrics(self) -> dict[str, Union[int, float]]:
        """Get current metrics."""
        return {
            "records_processed": self._records_processed,
            "records_dropped": self._records_dropped,
            "errors_count": self._errors_count,
            "last_error_time": self._last_error_time,
        }


class AsyncHandler(AsyncHandlerABC):
    """
    Base class for all async handlers in aiologging with enhanced performance.

    This class provides the common functionality for all async handlers,
    including level filtering, formatting, error handling, rate limiting,
    and performance metrics.
    """

    # Class-level configuration for better performance
    _default_retry_attempts: int = 3
    _default_retry_delay: float = 0.1
    _default_filter_cache_size: int = 1000

    def __init__(
        self,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[List[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
        rate_limiter: Optional[RateLimiter] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        retry_attempts: Optional[int] = None,
        retry_delay: Optional[float] = None,
        enable_metrics: bool = False,
    ) -> None:
        """
        Initialize the async handler with enhanced configuration options.

        Args:
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions
            rate_limiter: Optional rate limiter function
            retry_strategy: Optional retry strategy function
            retry_attempts: Number of retry attempts for failed operations
            retry_delay: Delay between retry attempts in seconds
            enable_metrics: Enable performance metrics collection
        """
        self.level = level
        self.formatter = formatter
        self.filters = filters or []
        self.error_handler = error_handler
        self._rate_limiter = rate_limiter
        self._retry_strategy = retry_strategy
        self._retry_attempts = retry_attempts or self._default_retry_attempts
        self._retry_delay = retry_delay or self._default_retry_delay

        self._closed = False
        self._lock = asyncio.Lock()

        # Performance optimizations
        self._filter_cache: dict[int, bool] = {}
        self._filter_cache_size = self._default_filter_cache_size

        # Metrics collection
        self._metrics = AsyncHandlerMetrics(enable_metrics)

        # Reference tracking for proper cleanup
        self._loggers: WeakSet[Any] = WeakSet()

    def setLevel(self, level: int) -> None:
        """Set the logging level for this handler and clear filter cache."""
        self.level = level
        self._filter_cache.clear()

    def addFilter(self, filter: FilterProtocol) -> None:
        """Add a filter to this handler and clear cache."""
        self.filters.append(filter)
        self._filter_cache.clear()

    def removeFilter(self, filter: FilterProtocol) -> None:
        """Remove a filter from this handler and clear cache."""
        if filter in self.filters:
            self.filters.remove(filter)
            self._filter_cache.clear()

    def setFormatter(self, formatter: FormatterProtocol) -> None:
        """Set the formatter for this handler."""
        self.formatter = formatter

    def format(self, record: LogRecord) -> str:
        """
        Format a log record.

        Args:
            record: The log record to format

        Returns:
            The formatted log message

        Raises:
            FormatterError: If formatting fails
        """
        if self.formatter is None:
            return record.getMessage()

        try:
            return self.formatter.format(record)
        except Exception as e:
            raise FormatterError(
                f"Failed to format log record: {e}",
                formatter_name=type(self.formatter).__name__,
                record_details={
                    "name": record.name,
                    "level": record.levelname,
                    "message": record.getMessage(),
                },
            ) from e

    def filter(self, record: LogRecord) -> bool:
        """
        Apply all filters to a log record with caching for better performance.

        Args:
            record: The log record to filter

        Returns:
            True if the record should be processed, False otherwise
        """
        # Quick level check
        if record.levelno < self.level:
            return False

        # Use cache for filter results if enabled
        record_id = id(record)
        if record_id in self._filter_cache:
            return self._filter_cache[record_id]

        # Apply filters
        result = True
        for filter_obj in self.filters:
            try:
                if not filter_obj.filter(record):
                    result = False
                    break
            except Exception as e:
                # Log filter error but continue processing
                sys.stderr.write(f"Filter error in {type(filter_obj).__name__}: {e}\n")
                result = False
                break

        # Cache the result
        if len(self._filter_cache) >= self._filter_cache_size:
            # Clear oldest entries
            self._filter_cache.clear()
        self._filter_cache[record_id] = result

        return result

    def isEnabledFor(self, level: int) -> bool:
        """
        Check if this handler is enabled for the given level.

        Args:
            level: The logging level to check

        Returns:
            True if the handler is enabled for the level
        """
        return level >= self.level

    async def handle(self, record: LogRecord) -> None:
        """
        Handle a log record with enhanced error handling and retry logic.

        This method applies filters, formats the record, and emits it.

        Args:
            record: The log record to handle

        Raises:
            HandlerClosedError: If the handler is closed.
        """
        if self._closed:
            raise HandlerClosedError(
                "Handler is closed and cannot handle records",
                handler_name=type(self).__name__,
                operation="handle",
            )

        if not self.filter(record):
            return

        if not self.isEnabledFor(record.levelno):
            return

        # Apply rate limiting if configured
        if self._rate_limiter and not await self._rate_limiter():
            self._metrics.increment_dropped()
            return

        try:
            formatted_message = self.format(record)
            await self._emit_with_retry(record, formatted_message)
            self._metrics.increment_processed()
        except Exception as e:
            self._metrics.increment_errors()
            await self._handle_error(record, e)

    async def emit(self, record: LogRecord) -> None:
        """
        Emit a log record with enhanced error handling.

        This method formats the record and calls the actual emit implementation.

        Args:
            record: The log record to emit
        """
        try:
            formatted_message = self.format(record)
            await self._emit_with_retry(record, formatted_message)
            self._metrics.increment_processed()
        except Exception as e:
            self._metrics.increment_errors()
            await self._handle_error(record, e)

    async def _emit_with_retry(self, record: LogRecord, formatted_message: str) -> None:
        """
        Emit a record with retry logic.

        Args:
            record: The log record to emit.
            formatted_message: The formatted log message.
        """
        last_exception = None

        for attempt in range(self._retry_attempts + 1):
            try:
                await self._emit(record, formatted_message)
                return
            except Exception as e:
                last_exception = e

                # Check if we should retry
                if self._should_retry(attempt, e):
                    await self._wait_before_retry(attempt)
                else:
                    break

        # All retries failed
        raise last_exception  # type: ignore[misc]

    def _should_retry(self, attempt: int, exception: Exception) -> bool:
        """
        Determine if we should retry based on the attempt and exception.

        Args:
            attempt: The current attempt number (0-based)
            exception: The exception that occurred

        Returns:
            True if we should retry, False otherwise
        """
        return (
            attempt < self._retry_attempts and (
                self._retry_strategy is None or
                self._retry_strategy(attempt + 1, exception)
            )
        )

    async def _wait_before_retry(self, attempt: int) -> None:
        """
        Wait before retrying with exponential backoff.

        Args:
            attempt: The current attempt number (0-based)
        """
        await asyncio.sleep(self._retry_delay * (2 ** attempt))

    async def _handle_error(self, record: LogRecord, error: Exception) -> None:
        """
        Handle errors during record processing.

        Args:
            record: The log record being processed.
            error: The exception that occurred.
        """
        fallback_message = f"Error in handler {type(self).__name__}: {error}"
        await handle_error_with_fallback(
            record, error, self.error_handler, fallback_message
        )

    @abstractmethod
    async def _emit(self, record: LogRecord, formatted_message: str) -> None:
        """
        Emit a log record.

        This method must be implemented by subclasses to define
        how the log record is actually processed (written to file,
        sent over network, etc.).

        Args:
            record: The log record to emit
        """
        raise NotImplementedError

    async def close(self) -> None:
        """
        Close the handler and release any resources.

        This method should be called when the handler is no longer needed.
        """
        async with self._lock:
            if not self._closed:
                self._closed = True
                await self._close_resources()

        # Clean up references and caches
        self._loggers.clear()
        self._filter_cache.clear()

    async def _close_resources(self) -> None:
        """
        Close handler-specific resources.

        Subclasses should override this method to clean up their
        specific resources (close files, network connections, etc.).
        """
        pass

    def get_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for the handler.

        Returns:
            Dictionary containing performance metrics.
        """
        metrics = self._metrics.get_metrics()
        metrics.update({
            "filter_cache_size": len(self._filter_cache),
            "closed": self._closed,
        })
        return metrics

    def __repr__(self) -> str:
        """Return a string representation of the handler."""
        return (
            f"{self.__class__.__name__}(level={self.level}, "
            f"formatter={type(self.formatter).__name__ if self.formatter else None}, "
            f"filters={len(self.filters)}, "
            f"metrics={self.get_metrics()})"
        )


class BufferedAsyncHandler(AsyncHandler):
    """
    Enhanced base class for async handlers that support buffering.

    This handler buffers log records and processes them in batches
    for better performance with high-volume logging. It includes
    advanced features like adaptive buffering, priority handling,
    and enhanced error recovery.
    """

    # Class-level configuration for better performance
    _default_buffer_size: int = 100
    _default_flush_interval: float = 5.0
    _default_max_batch_size: int = 1000
    _default_adaptive_threshold: float = 0.8

    def __init__(
        self,
        level: int = NOTSET,
        formatter: Optional[FormatterProtocol] = None,
        filters: Optional[List[FilterProtocol]] = None,
        error_handler: Optional[ErrorHandler] = None,
        rate_limiter: Optional[RateLimiter] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        retry_attempts: Optional[int] = None,
        retry_delay: Optional[float] = None,
        enable_metrics: bool = False,
        buffer_size: Optional[int] = None,
        flush_interval: Optional[float] = None,
        auto_flush: bool = True,
        max_batch_size: Optional[int] = None,
        adaptive_buffering: bool = False,
        priority_flush: bool = False,
    ) -> None:
        """
        Initialize the buffered async handler with enhanced configuration.

        Args:
            level: The logging level for this handler
            formatter: The formatter to use for log records
            filters: List of filters to apply to log records
            error_handler: Optional error handler for exceptions
            rate_limiter: Optional rate limiter function
            retry_strategy: Optional retry strategy function
            retry_attempts: Number of retry attempts for failed operations
            retry_delay: Delay between retry attempts in seconds
            enable_metrics: Enable performance metrics collection
            buffer_size: Maximum number of records to buffer
            flush_interval: Time interval in seconds to auto-flush
            auto_flush: Whether to automatically flush on interval
            max_batch_size: Maximum batch size for processing
            adaptive_buffering: Enable adaptive buffer sizing
            priority_flush: Enable priority-based flushing
        """
        super().__init__(
            level, formatter, filters, error_handler, rate_limiter,
            retry_strategy, retry_attempts, retry_delay, enable_metrics
        )

        self.buffer_size = buffer_size or self._default_buffer_size
        self.flush_interval = flush_interval or self._default_flush_interval
        self.auto_flush = auto_flush
        self.max_batch_size = max_batch_size or self._default_max_batch_size
        self.adaptive_buffering = adaptive_buffering
        self.priority_flush = priority_flush

        # Buffer management
        self._buffer: List[LogRecord] = []
        self._priority_buffer: List[LogRecord] = []
        self._flush_task: Optional[asyncio.Task[None]] = None
        self._buffer_lock = asyncio.Lock()

        # Adaptive buffering metrics
        self._avg_processing_time = 0.0
        self._last_flush_duration = 0.0
        self._adaptive_buffer_size = self.buffer_size

        # Enhanced metrics
        self._batches_processed = 0
        self._total_records_in_batches = 0
        self._priority_flushes = 0

    async def handle(self, record: LogRecord) -> None:
        """
        Handle a log record by adding it to the appropriate buffer.

        Args:
            record: The log record to handle

        Raises:
            HandlerClosedError: If the handler is closed.
        """
        if self._closed:
            raise HandlerClosedError(
                "Handler is closed and cannot handle records",
                handler_name=type(self).__name__,
                operation="handle",
            )

        if not self.filter(record):
            return

        if not self.isEnabledFor(record.levelno):
            return

        # Apply rate limiting if configured
        if self._rate_limiter and not await self._rate_limiter():
            self._metrics.increment_dropped()
            return

        async with self._buffer_lock:
            # Use priority buffer for high-priority records if enabled
            if (self.priority_flush and
                record.levelno >= 40 and  # ERROR and CRITICAL
                len(self._priority_buffer) < self.buffer_size // 4):
                self._priority_buffer.append(record)
            else:
                self._buffer.append(record)

            # Start auto-flush task if needed
            if self.auto_flush and self._flush_task is None:
                self._flush_task = asyncio.create_task(self._auto_flush())

            # Adaptive buffer sizing
            if self.adaptive_buffering:
                await self._adjust_buffer_size()

            # Flush if buffer is full
            current_buffer_size = len(self._buffer) + len(self._priority_buffer)
            effective_buffer_size = self._adaptive_buffer_size if self.adaptive_buffering else self.buffer_size

            if current_buffer_size >= effective_buffer_size:
                await self._flush_buffer()

    async def _auto_flush(self) -> None:
        """Enhanced auto-flush task that runs periodically with adaptive timing."""
        while not self._closed and self.auto_flush:
            # Adaptive flush interval based on processing time
            flush_interval = self.flush_interval
            if self.adaptive_buffering and self._last_flush_duration > 0:
                # Adjust interval based on last flush duration
                if self._last_flush_duration > flush_interval * 0.5:
                    flush_interval *= 1.2  # Increase interval if processing is slow
                elif self._last_flush_duration < flush_interval * 0.1:
                    flush_interval *= 0.8  # Decrease interval if processing is fast

            await asyncio.sleep(flush_interval)

            # Check if we need to flush
            if self._buffer or self._priority_buffer:
                await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        """Enhanced buffer flushing with priority handling and batch optimization."""
        start_time = time.time()

        # Get records to flush
        all_records = self._get_records_to_flush()

        if not all_records:
            return

        try:
            await self._flush_with_retry(all_records)

            # Update metrics
            self._batches_processed += 1
            self._total_records_in_batches += len(all_records)
            # Metrics are already updated in the parent class

        except Exception as e:
            self._metrics.increment_errors()
            await self._handle_batch_error(all_records, e)
        finally:
            # Update performance metrics
            self._update_performance_metrics(start_time)

    def _get_records_to_flush(self) -> list[LogRecord]:
        """
        Get records to flush from priority and regular buffers.

        Returns:
            List of records to flush, with priority records first
        """
        # Process priority buffer first
        priority_records = []
        if self._priority_buffer:
            priority_records = self._priority_buffer.copy()
            self._priority_buffer.clear()
            self._priority_flushes += 1

        # Process regular buffer
        regular_records = []
        if self._buffer:
            # Optimize batch size based on performance
            batch_size = self._calculate_optimal_batch_size()
            regular_records = self._buffer[:batch_size]
            self._buffer = self._buffer[batch_size:]

        # Combine records with priority first
        return priority_records + regular_records

    def _calculate_optimal_batch_size(self) -> int:
        """
        Calculate the optimal batch size based on performance metrics.

        Returns:
            The optimal batch size
        """
        batch_size = min(len(self._buffer), self.max_batch_size)
        if self.adaptive_buffering and self._avg_processing_time > 0:
            # Adjust batch size based on average processing time
            target_time = 1.0  # Target 1 second per batch
            optimal_batch = int(target_time / self._avg_processing_time)
            batch_size = min(batch_size, max(1, optimal_batch))
        return batch_size

    def _update_performance_metrics(self, start_time: float) -> None:
        """
        Update performance metrics after a flush operation.

        Args:
            start_time: The start time of the flush operation
        """
        self._last_flush_duration = time.time() - start_time
        if self._batches_processed > 0:
            self._avg_processing_time = (
                (self._avg_processing_time * (self._batches_processed - 1) +
                 self._last_flush_duration) / self._batches_processed
            )

    async def _flush_with_retry(self, records: List[LogRecord]) -> None:
        """
        Flush records with retry logic.

        Args:
            records: The records to flush.
        """
        last_exception = None

        for attempt in range(self._retry_attempts + 1):
            try:
                await self.flush(records)
                return
            except Exception as e:
                last_exception = e

                # Check if we should retry
                if attempt < self._retry_attempts and (
                    self._retry_strategy is None or
                    self._retry_strategy(attempt + 1, e)
                ):
                    await asyncio.sleep(self._retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    break

        # All retries failed
        raise last_exception  # type: ignore[misc]

    async def _handle_batch_error(self, records: List[LogRecord], error: Exception) -> None:
        """
        Handle errors during batch processing.

        Args:
            records: The records that failed to process.
            error: The exception that occurred.
        """
        if self.error_handler:
            for record in records:
                try:
                    await self.error_handler(record, error)
                except Exception:
                    pass  # Avoid infinite error loops
        else:
            import traceback
            sys.stderr.write(
                f"Error flushing buffer in {type(self).__name__}: {error}\n"
                f"Records affected: {len(records)}\n"
                f"Traceback: {traceback.format_exc()}\n"
            )

    async def _adjust_buffer_size(self) -> None:
        """
        Adjust buffer size adaptively based on performance metrics.
        """
        if not self.adaptive_buffering:
            return

        # Adjust buffer size based on processing performance
        if self._avg_processing_time > 0:
            # If processing is slow, reduce buffer size
            if self._avg_processing_time > 2.0:  # More than 2 seconds per batch
                self._adaptive_buffer_size = max(
                    self.buffer_size // 2,
                    10  # Minimum buffer size
                )
            # If processing is fast, increase buffer size
            elif self._avg_processing_time < 0.5:  # Less than 0.5 seconds per batch
                self._adaptive_buffer_size = min(
                    self.buffer_size * 2,
                    self.max_batch_size
                )
            else:
                # Reset to default
                self._adaptive_buffer_size = self.buffer_size

    async def flush(self, records: List[LogRecord]) -> None:
        """
        Flush a batch of log records.

        This method must be implemented by subclasses to define
        how the batch of records is processed.

        Args:
            records: The list of log records to flush
        """
        raise NotImplementedError

    async def force_flush(self) -> None:
        """Force flush all buffers immediately."""
        async with self._buffer_lock:
            await self._flush_buffer()

    async def close(self) -> None:
        """Close the handler and flush any remaining records."""
        # Cancel auto-flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Flush remaining records
        await self._flush_buffer()

        # Call parent close
        await super().close()

    async def _close_resources(self) -> None:
        """Close handler-specific resources."""
        # Cancel any remaining tasks
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()

    def get_metrics(self) -> dict[str, Any]:
        """
        Get enhanced performance metrics for the buffered handler.

        Returns:
            Dictionary containing performance metrics.
        """
        base_metrics = super().get_metrics()
        buffer_metrics = {
            "buffer_size": len(self._buffer),
            "priority_buffer_size": len(self._priority_buffer),
            "max_buffer_size": self.buffer_size,
            "adaptive_buffer_size": self._adaptive_buffer_size if self.adaptive_buffering else None,
            "batches_processed": self._batches_processed,
            "total_records_in_batches": self._total_records_in_batches,
            "priority_flushes": self._priority_flushes,
            "avg_processing_time": self._avg_processing_time,
            "last_flush_duration": self._last_flush_duration,
            "auto_flush_enabled": self.auto_flush,
            "adaptive_buffering_enabled": self.adaptive_buffering,
            "priority_flush_enabled": self.priority_flush,
        }
        return {**base_metrics, **buffer_metrics}
