# aiologging

Asynchronous logging library for Python with full compatibility to the standard logging module but with async methods requiring await.

## Features

- **Full API Compatibility**: Drop-in replacement for standard logging with async methods
- **Async Handlers**: Non-blocking I/O for streams, files, and HTTP endpoints
- **File Rotation**: Size and time-based log rotation with async support
- **HTTP Handlers**: Send logs to HTTP endpoints with extensible authentication
- **Buffered Handlers**: High-performance batch processing for high-volume logging
- **Performance Metrics**: Built-in metrics collection for monitoring logging performance
- **Error Handling**: Comprehensive error handling with custom exception types
- **Configuration Management**: Flexible configuration from files, dictionaries, or environment variables
- **Optional Dependencies**: Install only what you need
- **Strict Type Checking**: Full mypy support with type hints
- **Context Manager Support**: Safe resource management with `async with`

## Installation

### Basic Installation

```bash
pip install aiologging
```

### With Optional Dependencies

```bash
# For file handlers
pip install aiologging[aiofiles]

# For HTTP handlers
pip install aiologging[aiohttp]

# For Protobuf support
pip install aiologging[protobuf]

# All dependencies
pip install aiologging[all]

# Development dependencies
pip install aiologging[dev]
```

## Quick Start

### Basic Usage

```python
import asyncio
import aiologging

async def main():
    async with aiologging.getLogger("app") as logger:
        await logger.info("Application started")
        await logger.warning("Something might be wrong")
        await logger.error("An error occurred")

asyncio.run(main())
```

### Basic Configuration

```python
import aiologging

# Configure basic logging (similar to logging.basicConfig)
aiologging.basicConfig(
    level=aiologging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

async def main():
    async with aiologging.getLogger("app") as logger:
        await logger.info("This will be logged to stderr")
```

### Using Convenience Functions

```python
import asyncio
import aiologging

async def main():
    # Create handlers using convenience functions
    stream_handler = aiologging.create_stream_handler(level=aiologging.INFO)
    file_handler = aiologging.create_file_handler("app.log", level=aiologging.DEBUG)

    # Get logger and add handlers
    async with aiologging.getLogger("app") as logger:
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

        await logger.info("This goes to both stderr and app.log")

asyncio.run(main())
```

## Handlers

### Stream Handler

```python
import aiologging
import sys

async def main():
    async with aiologging.getLogger("app") as logger:
        # Add stdout handler
        stdout_handler = aiologging.AsyncStreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)

        await logger.info("This goes to stdout")
```

### File Handler (requires aiofiles)

```python
import aiologging

async def main():
    async with aiologging.getLogger("app") as logger:
        # Add file handler
        file_handler = aiologging.AsyncFileHandler("app.log")
        logger.addHandler(file_handler)

        await logger.info("This goes to app.log")
```

### Rotating File Handler (requires aiofiles)

```python
import aiologging

async def main():
    async with aiologging.getLogger("app") as logger:
        # Size-based rotation
        rotating_handler = aiologging.AsyncRotatingFileHandler(
            "app.log",
            max_bytes=1024*1024,  # 1MB
            backup_count=5
        )
        logger.addHandler(rotating_handler)

        # Time-based rotation
        timed_handler = aiologging.AsyncTimedRotatingFileHandler(
            "app.log",
            when="midnight",
            backup_count=7
        )
        logger.addHandler(timed_handler)

        await logger.info("This will be rotated")
```

### HTTP Handler (requires aiohttp)

```python
import aiologging

async def main():
    async with aiologging.getLogger("app") as logger:
        # Basic HTTP handler
        http_handler = aiologging.AsyncHttpHandler(
            "https://api.example.com/logs",
            headers={"Authorization": "Bearer token"}
        )
        logger.addHandler(http_handler)

        await logger.info("This will be sent via HTTP")
```

### Custom Authentication

```python
import aiologging

async def oauth_authenticator(session, request_data):
    """Custom OAuth authentication."""
    # Refresh token logic here
    token = await refresh_oauth_token()
    return {"Authorization": f"Bearer {token}"}

async def main():
    async with aiologging.getLogger("app") as logger:
        http_handler = aiologging.AsyncHttpHandler(
            "https://api.example.com/logs",
            authenticator=oauth_authenticator
        )
        logger.addHandler(http_handler)

        await logger.info("This uses custom authentication")
```

## HTTP Handler Formats

### JSON Handler

```python
json_handler = aiologging.AsyncHttpJsonHandler(
    "https://api.example.com/logs"
)
```

### Text Handler

```python
text_handler = aiologging.AsyncHttpTextHandler(
    "https://api.example.com/logs"
)
```

### Protobuf Handler (requires protobuf)

```python
proto_handler = aiologging.AsyncHttpProtoHandler(
    "https://api.example.com/logs"
)
```

### Universal Handler (auto-detect format)

```python
universal_handler = aiologging.AsyncHttpHandler(
    "https://api.example.com/logs",
    format_type="application/json"  # Optional: auto-detected if not specified
)
```

## Advanced Usage

### Custom Filters

```python
import aiologging

class CustomFilter:
    def filter(self, record):
        return "important" in record.getMessage()

async def main():
    async with aiologging.getLogger("app") as logger:
        logger.addFilter(CustomFilter())

        await logger.info("This is important")  # Will be logged
        await logger.info("This is not")        # Will be filtered
```

### Custom Formatters

```python
import logging
import aiologging

async def main():
    async with aiologging.getLogger("app") as logger:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        handler = aiologging.AsyncStreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        await logger.info("Formatted message")
```

### Error Handling

```python
import aiologging

async def error_handler(record, exception):
    """Custom error handler for failed log operations."""
    print(f"Failed to log {record.getMessage()}: {exception}")

async def main():
    async with aiologging.getLogger("app") as logger:
        handler = aiologging.AsyncStreamHandler()
        handler.error_handler = error_handler
        logger.addHandler(handler)

        await logger.info("This has custom error handling")
```

### Performance Metrics

```python
import aiologging

async def main():
    # Create logger with metrics enabled
    async with aiologging.getLogger("app") as logger:
        # Get metrics for the logger
        metrics = logger.get_metrics()
        print(f"Logger metrics: {metrics}")

        # Get metrics for handlers
        for handler in logger.handlers:
            if hasattr(handler, 'get_metrics'):
                handler_metrics = handler.get_metrics()
                print(f"Handler metrics: {handler_metrics}")

asyncio.run(main())
```

### Batch Processing

```python
import aiologging
from aiologging.types import BatchConfig

async def main():
    async with aiologging.getLogger("app") as logger:
        # Configure batch processing for HTTP handler
        batch_config = BatchConfig(
            batch_size=100,
            flush_interval=5.0,
            max_retries=3
        )

        http_handler = aiologging.AsyncHttpHandler(
            "https://api.example.com/logs",
            batch_config=batch_config
        )
        logger.addHandler(http_handler)

        # Log many messages - they'll be sent in batches
        for i in range(150):
            await logger.info(f"Message {i}")
```

## Configuration Management

### Configuration from Dictionary

```python
import aiologging

config = {
    "version": 1,
    "loggers": {
        "myapp": {
            "level": "INFO",
            "handlers": ["console", "file"]
        }
    },
    "handlers": {
        "console": {
            "class": "stream",
            "level": "INFO",
            "stream": "stdout"
        },
        "file": {
            "class": "file",
            "level": "DEBUG",
            "filename": "app.log",
            "mode": "a"
        }
    }
}

# Configure from dictionary
aiologging.configure_from_dict(config)

async def main():
    logger = aiologging.get_configured_logger("myapp")
    await logger.info("This uses configured logger")

asyncio.run(main())
```

### Configuration from File

```python
import aiologging

# Configure from JSON file
aiologging.configure_from_file("logging_config.json")

async def main():
    logger = aiologging.get_configured_logger("myapp")
    await logger.info("This uses logger configured from file")

asyncio.run(main())
```

## Migration from Standard Logging

### Standard Logging

```python
import logging

logger = logging.getLogger("app")
logger.info("Message")
```

### aiologging

```python
import aiologging

async def main():
    async with aiologging.getLogger("app") as logger:
        await logger.info("Message")
```

### Key Differences

1. **Async Context**: Use `async with` for proper resource management
2. **Await Methods**: All logging methods require `await`
3. **Async Handlers**: All handlers are non-blocking
4. **Type Safety**: Full type hints and mypy compliance
5. **Error Handling**: Enhanced error handling with custom exception types
6. **Performance Metrics**: Built-in metrics collection for monitoring

## API Reference

### Logger Methods

All standard logging methods are available as async:

- `await logger.log(level, msg, *args, **kwargs)`
- `await logger.debug(msg, *args, **kwargs)`
- `await logger.info(msg, *args, **kwargs)`
- `await logger.warning(msg, *args, **kwargs)`
- `await logger.error(msg, *args, **kwargs)`
- `await logger.critical(msg, *args, **kwargs)`
- `await logger.exception(msg, *args, **kwargs)`

### Handler Classes

- `AsyncHandler` - Base async handler
- `AsyncStreamHandler` - Stream output handler
- `AsyncFileHandler` - File output handler (requires aiofiles)
- `AsyncRotatingFileHandler` - Size-based rotation (requires aiofiles)
- `AsyncTimedRotatingFileHandler` - Time-based rotation (requires aiofiles)
- `AsyncHttpHandler` - Universal HTTP handler (requires aiohttp)
- `AsyncHttpTextHandler` - Plain text HTTP handler (requires aiohttp)
- `AsyncHttpJsonHandler` - JSON HTTP handler (requires aiohttp)
- `AsyncHttpProtoHandler` - Protobuf HTTP handler (requires aiohttp, protobuf)

### Configuration Classes

- `BatchConfig` - Batch processing configuration
- `FileConfig` - File handler configuration
- `HttpConfig` - HTTP handler configuration
- `LoggerConfig` - Logger configuration
- `RotationConfig` - Rotation configuration for file handlers

### Exception Classes

- `AiologgingError` - Base exception for all aiologging errors
- `HandlerError` - Base exception for handler errors
- `ConfigurationError` - Configuration-related errors
- `DependencyError` - Missing optional dependencies
- `AuthenticationError` - Authentication failures
- `NetworkError` - Network-related errors
- `FileError` - File operation errors
- `RotationError` - File rotation errors
- `BatchError` - Batch processing errors
- `FormatterError` - Formatting errors
- `LoggerError` - Logger operation errors
- `ContextError` - Context manager errors

## Performance Considerations

1. **Buffering**: Use buffered handlers for high-volume logging
2. **Batch Processing**: Configure appropriate batch sizes for HTTP handlers
3. **Async I/O**: All I/O operations are non-blocking
4. **Resource Management**: Always use context managers for proper cleanup
5. **Metrics Collection**: Enable metrics to monitor performance
6. **Rate Limiting**: Use rate limiters to prevent log flooding
7. **Adaptive Buffering**: Enable adaptive buffering for optimal performance

## Testing

```bash
# Install development dependencies
pip install aiologging[dev]

# Run tests
pytest

# Run tests with coverage
pytest --cov=aiologging

# Run type checking
mypy aiologging
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and type checking succeeds
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### 0.1.0

- Initial release
- Full async logging API
- Stream, file, and HTTP handlers
- File rotation support
- Extensible authentication
- Optional dependencies
- Strict type checking
