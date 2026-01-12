#!/usr/bin/env python3
"""
Spores Transport Interface

Protocols for transporting encoded LogRecords.
Separate sync and async protocols for clarity.
"""

from __future__ import annotations

from typing import Protocol


class SyncTransport(Protocol):
    """
    Protocol for synchronous transports (blocking I/O).

    Sync transports use blocking send() - they write data immediately
    and block until complete. Use these with get_spore_sync().

    Examples: file writes, blocking HTTP requests, etc.
    """

    def send(self, data: bytes, content_type: str) -> None:
        """
        Send encoded data to the transport destination (blocking).

        Args:
            data: The encoded log record data
            content_type: MIME content type (e.g., "application/json")
        """
        ...

    def close(self) -> None:
        """Close the transport and release resources."""
        ...


class AsyncTransport(Protocol):
    """
    Protocol for asynchronous transports (async I/O).

    Async transports use non-blocking async send() - they await completion.
    Use these with get_spore_async().

    Examples: async file writes, async HTTP requests, message queues, etc.
    """

    async def send(self, data: bytes, content_type: str) -> None:
        """
        Send encoded data to the transport destination (async).

        Args:
            data: The encoded log record data
            content_type: MIME content type (e.g., "application/json")
        """
        ...

    async def close(self) -> None:
        """Close the transport and release resources."""
        ...


# Backward compatibility alias
Transport = AsyncTransport
