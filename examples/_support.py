"""Keep demo journals isolated and alive until the engine's exit drain."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import aiologging

_directories: list[TemporaryDirectory[str]] = []


def setup(**kwargs: Any) -> Path:
    directory = TemporaryDirectory(prefix="aiologging-demo-")
    _directories.append(directory)
    path = Path(directory.name)
    aiologging.configureJournal(path / "journal.sqlite3", **kwargs)
    return path
