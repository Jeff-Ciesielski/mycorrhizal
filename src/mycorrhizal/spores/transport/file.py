#!/usr/bin/env python3
"""
Spores File Transports

File-based transports for OCEL log records.
Includes both synchronous (blocking I/O) and asynchronous (async I/O) implementations.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Optional


class SyncFileTransport:
    """
    Synchronous file transport using blocking I/O.

    Writes log records to a file in JSONL format (one JSON object per line).
    Thread-safe with a lock for concurrent writes.
    """

    def __init__(self, filepath: str | Path):
        """
        Initialize the file transport.

        Args:
            filepath: Path to the log file. Will be created if it doesn't exist.
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, 'a', encoding='utf-8')
        self._lock = threading.Lock()

    def send(self, data: bytes, content_type: str) -> None:
        """
        Write data to file (blocking call).

        Args:
            data: Encoded log record data
            content_type: MIME content type (e.g., "application/json")
        """
        with self._lock:
            self._file.write(data.decode('utf-8') + '\n')
            self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        with self._lock:
            if self._file and not self._file.closed:
                self._file.close()
                self._file = None

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.close()


class AsyncFileTransport:
    """
    Asynchronous file transport using async I/O.

    Writes log records to a file in JSONL format (one JSON object per line).
    Uses asyncio.to_thread() for non-blocking async file I/O.

    This is the async version of SyncFileTransport - use it with get_spore_async().
    """

    def __init__(self, filepath: str | Path):
        """
        Initialize the async file transport.

        Args:
            filepath: Path to the log file. Will be created if it doesn't exist.
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        # Open file in append mode, will be managed by asyncio.to_thread
        self._file = open(self.filepath, 'a', encoding='utf-8')
        self._lock = asyncio.Lock()

    async def send(self, data: bytes, content_type: str) -> None:
        """
        Write data to file asynchronously.

        Uses asyncio.to_thread() to run blocking I/O in a thread pool,
        avoiding the need for external dependencies like aiofiles.

        Args:
            data: Encoded log record data
            content_type: MIME content type (e.g., "application/json")
        """
        async with self._lock:
            # Run blocking file I/O in thread pool
            await asyncio.to_thread(self._write_sync, data)

    def _write_sync(self, data: bytes) -> None:
        """
        Synchronous write helper - runs in thread pool.

        Args:
            data: Encoded log record data
        """
        self._file.write(data.decode('utf-8') + '\n')
        self._file.flush()

    async def close(self) -> None:
        """Close the file handle asynchronously."""
        async with self._lock:
            if self._file and not self._file.closed:
                await asyncio.to_thread(self._file.close)
                self._file = None

    async def __aenter__(self):
        """Async context manager support."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager support."""
        await self.close()
