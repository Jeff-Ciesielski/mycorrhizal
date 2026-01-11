#!/usr/bin/env python3
"""
Spores Encoder Interface

Abstract base class for encoding LogRecords to bytes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from ..models import LogRecord


class Encoder(Protocol):
    """
    Protocol for encoders that convert LogRecords to bytes.

    Encoders serialize OCEL LogRecords for transport to OCEL consumers.
    """

    def encode(self, record: LogRecord) -> bytes:
        """
        Encode a LogRecord to bytes.

        Args:
            record: The LogRecord to encode

        Returns:
            Serialized bytes representation
        """
        ...

    def content_type(self) -> str:
        """
        Return the MIME content type for this encoding.

        Returns:
            Content type string (e.g., "application/json")
        """
        ...
