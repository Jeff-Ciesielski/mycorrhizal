#!/usr/bin/env python3
"""
Spores Transport Interface

Abstract base class for transporting encoded LogRecords.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Callable, Optional


class Transport(Protocol):
    """
    Protocol for transports that send encoded LogRecords.

    Transports handle the actual delivery of log records to consumers
    (e.g., HTTP POST, files, message queues, etc.).
    """

    async def send(self, data: bytes, content_type: str) -> None:
        """
        Send encoded data to the transport destination.

        Args:
            data: The encoded log record data
            content_type: MIME content type (e.g., "application/json")
        """
        ...

    def is_async(self) -> bool:
        """
        Return True if this transport is async (non-blocking).

        Async transports don't block the caller when sending data.
        They typically use background threads, queues, or async I/O.

        Returns:
            True if transport is async, False if synchronous
        """
        ...

    def close(self) -> None:
        """Close the transport and release resources."""
        ...
