"""
Tests for file handlers.
"""

import asyncio
import logging
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiologging.handlers.file import AsyncFileHandler
from aiologging.exceptions import HandlerError


class TestAsyncFileHandler:
    """Test cases for AsyncFileHandler."""

    @pytest.mark.asyncio
    async def test_initialization_with_filename(self) -> None:
        """Test handler initialization with filename."""
        filename = self._create_temp_file()
        try:
            handler = AsyncFileHandler(filename)
            self._assert_default_handler_properties(handler, Path(filename))
        finally:
            os.unlink(filename)

    @pytest.mark.asyncio
    async def test_initialization_with_path(self) -> None:
        """Test handler initialization with Path object."""
        filename = self._create_temp_file()
        try:
            path = Path(filename)
            handler = AsyncFileHandler(path)
            self._assert_default_handler_properties(handler, path)
        finally:
            os.unlink(filename)

    @pytest.mark.asyncio
    async def test_initialization_with_parameters(self) -> None:
        """Test handler initialization with parameters."""
        filename = self._create_temp_file()
        try:
            handler = AsyncFileHandler(
                filename=filename,
                mode="w",
                encoding="ascii",
                delay=True,
                errors="ignore"
            )
            assert handler.filename == Path(filename)
            assert handler.mode == "w"
            assert handler.encoding == "ascii"
            assert handler.delay is True
            assert handler.errors == "ignore"
        finally:
            os.unlink(filename)

    def _create_temp_file(self) -> str:
        """Create a temporary file and return its name."""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        return temp_file.name

    def _assert_default_handler_properties(self, handler: AsyncFileHandler, filename: Path) -> None:
        """Assert default handler properties."""
        assert handler.filename == filename
        assert handler.mode == "a"
        assert handler.encoding == "utf-8"
        assert handler.delay is False
        assert handler._file is None
        assert handler._closed is False

    def _create_test_record(self, message: str = "Test message", level: int = logging.INFO) -> logging.LogRecord:
        """Create a test log record."""
        return logging.LogRecord(
            name="test", level=level, pathname="", lineno=0,
            msg=message, args=(), exc_info=None
        )

    @pytest.mark.asyncio
    async def test_ensure_file_open(self) -> None:
        """Test _ensure_file_open method."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name

        try:
            handler = AsyncFileHandler(filename, delay=True)

            # File should not be open initially
            assert handler._file is None

            # Call _ensure_file_open
            await handler._ensure_file_open()

            # File should be open now
            assert handler._file is not None

            # Close the file
            await handler.close()
        finally:
            os.unlink(filename)

    @pytest.mark.asyncio
    async def test_emit_without_delay(self) -> None:
        """Test emit without delay (file opened immediately)."""
        filename = self._create_temp_file()
        try:
            handler = AsyncFileHandler(filename, delay=False)
            record = self._create_test_record()

            await handler.emit(record)
            self._assert_message_in_file(filename, "Test message")
            await handler.close()
        finally:
            os.unlink(filename)

    @pytest.mark.asyncio
    async def test_emit_with_delay(self) -> None:
        """Test emit with delay (file opened on first emit)."""
        filename = self._create_temp_file()
        try:
            handler = AsyncFileHandler(filename, delay=True)

            # File should not be open initially
            assert handler._file is None

            record = self._create_test_record()
            await handler.emit(record)

            # File should be open now
            assert handler._file is not None

            self._assert_message_in_file(filename, "Test message")
            await handler.close()
        finally:
            os.unlink(filename)

    def _assert_message_in_file(self, filename: str, message: str) -> None:
        """Assert that a message is present in the file."""
        with open(filename, "r") as f:
            content = f.read()
            assert message in content

    @pytest.mark.asyncio
    async def test_emit_with_exception(self) -> None:
        """Test emit with exception in record."""
        filename = self._create_temp_file()
        try:
            handler = AsyncFileHandler(filename)

            try:
                raise ValueError("Test exception")
            except ValueError:
                record = logging.LogRecord(
                    name="test", level=logging.ERROR, pathname="", lineno=0,
                    msg="Error message", args=(), exc_info=True
                )

                await handler.emit(record)
                self._assert_message_in_file(filename, "Error message")

            await handler.close()
        finally:
            os.unlink(filename)

    @pytest.mark.asyncio
    async def test_emit_file_error(self) -> None:
        """Test emit with file error."""
        filename = self._create_temp_file()
        try:
            handler = AsyncFileHandler(filename, delay=True)
            mock_file = self._create_mock_file_with_error()
            handler._file = mock_file

            record = self._create_test_record()

            # The error should be caught and handled by the error handler
            with patch("sys.stderr") as stderr:
                await handler.emit(record)
                stderr.write.assert_called()

            await handler.close()
        finally:
            os.unlink(filename)

    def _create_mock_file_with_error(self) -> AsyncMock:
        """Create a mock file object that raises an error on write."""
        mock_file = AsyncMock()
        mock_file.write = AsyncMock(side_effect=IOError("Write error"))
        mock_file.flush = AsyncMock()
        return mock_file

    @pytest.mark.asyncio
    async def test_emit_closed_handler(self) -> None:
        """Test emit with closed handler."""
        filename = self._create_temp_file()
        try:
            handler = AsyncFileHandler(filename)
            await handler.close()

            record = self._create_test_record()

            with pytest.raises(HandlerError):
                await handler.handle(record)  # Use handle instead of emit to check _closed
        finally:
            os.unlink(filename)

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing handler."""
        filename = self._create_temp_file()
        try:
            handler = AsyncFileHandler(filename)

            # Emit a record to open the file
            record = self._create_test_record()
            await handler.emit(record)

            # File should be open
            assert handler._file is not None
            assert handler._closed is False

            # Close the handler
            await handler.close()

            # File should be closed and handler marked as closed
            assert handler._file is None
            assert handler._closed is True
        finally:
            os.unlink(filename)

    @pytest.mark.asyncio
    async def test_close_already_closed(self) -> None:
        """Test closing an already closed handler."""
        filename = self._create_temp_file()
        try:
            handler = AsyncFileHandler(filename)
            await handler.close()

            # Closing again should not raise an error
            await handler.close()
        finally:
            os.unlink(filename)

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test handler as context manager."""
        filename = self._create_temp_file()
        try:
            handler = AsyncFileHandler(filename)
            async with handler:
                assert isinstance(handler, AsyncFileHandler)
                assert handler._closed is False

                record = self._create_test_record()
                await handler.emit(record)

            # Handler should be closed after exiting context
            assert handler._closed is True

            # Verify the message was written
            self._assert_message_in_file(filename, "Test message")
        finally:
            os.unlink(filename)

    @pytest.mark.asyncio
    async def test_file_not_exists_delay_false(self) -> None:
        """Test handler with non-existent file and delay=False."""
        filename = "/tmp/non_existent_file.log"

        # Make sure the file doesn't exist
        if os.path.exists(filename):
            os.unlink(filename)

        try:
            # This should create the file
            handler = AsyncFileHandler(filename, delay=False)

            # Wait a bit for the file to be opened
            await asyncio.sleep(0.1)

            # File should be open
            assert handler._file is not None

            # Emit a record
            record = self._create_test_record()
            await handler.emit(record)

            await handler.close()

            # File should exist now
            assert os.path.exists(filename)

            # Verify the message was written
            with open(filename, "r") as f:
                content = f.read()
                assert "Test message" in content
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    @pytest.mark.asyncio
    async def test_file_not_exists_delay_true(self) -> None:
        """Test handler with non-existent file and delay=True."""
        filename = "/tmp/non_existent_file.log"

        # Make sure the file doesn't exist
        if os.path.exists(filename):
            os.unlink(filename)

        try:
            handler = AsyncFileHandler(filename, delay=True)

            # File should not be open initially
            assert handler._file is None

            # Emit a record
            record = self._create_test_record()
            await handler.emit(record)

            # File should be open now
            assert handler._file is not None

            await handler.close()

            # File should exist now
            assert os.path.exists(filename)

            # Verify the message was written
            with open(filename, "r") as f:
                content = f.read()
                assert "Test message" in content
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    @pytest.mark.asyncio
    async def test_directory_not_exists(self) -> None:
        """Test handler with non-existent directory."""
        dirname = "/tmp/non_existent_dir"
        filename = os.path.join(dirname, "test.log")

        # Make sure the directory doesn't exist
        if os.path.exists(dirname):
            os.rmdir(dirname)

        try:
            # This should create the directory and file
            handler = AsyncFileHandler(filename)

            # Emit a record
            record = self._create_test_record()
            await handler.emit(record)

            await handler.close()

            # Directory and file should exist now
            assert os.path.exists(dirname)
            assert os.path.exists(filename)

            # Verify the message was written
            with open(filename, "r") as f:
                content = f.read()
                assert "Test message" in content
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
            if os.path.exists(dirname):
                os.rmdir(dirname)


if __name__ == "__main__":
    pytest.main([__file__])
