#!/usr/bin/env python3
"""
Spores JSON Encoder

JSON encoder for OCEL LogRecords.
Compatible with the Go OCEL library format.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, TYPE_CHECKING

from .base import Encoder
from ..models import LogRecord, Event, Object, Relationship, EventAttributeValue, ObjectAttributeValue


class JSONEncoder(Encoder):
    """
    JSON encoder for OCEL LogRecords.

    Produces OCEL-compatible JSON with Unix float64 timestamps.

    Example output:
        {
          "event": {
            "id": "evt-123",
            "type": "process_item",
            "activity": null,
            "time": 1705689296.123456,
            "attributes": [
              {"name": "priority", "value": "high", "type": "string"}
            ],
            "relationships": [
              {"objectId": "obj-456", "qualifier": "input"}
            ]
          },
          "object": null
        }

    Or for objects:
        {
          "event": null,
          "object": {
            "id": "obj-456",
            "type": "WorkItem",
            "attributes": [
              {"name": "status", "value": "pending", "type": "string", "time": 1705689296.123456}
            ],
            "relationships": []
          }
        }
    """

    def __init__(self):
        """
        Initialize JSONEncoder.

        Timestamps are always encoded as Unix float64 timestamps.
        """
        pass

    def encode(self, record: LogRecord) -> bytes:
        """
        Encode a LogRecord to JSON bytes.

        Args:
            record: The LogRecord to encode

        Returns:
            JSON-encoded bytes
        """
        # Convert LogRecord to dict with union structure (both fields, one null)
        if record.event is not None:
            data = {"event": self._event_to_dict(record.event), "object": None}
        else:
            data = {"event": None, "object": self._object_to_dict(record.object)}

        # Encode to JSON
        return json.dumps(data, separators=(',', ':')).encode('utf-8')

    def content_type(self) -> str:
        """Return the JSON content type."""
        return "application/json"

    def _event_to_dict(self, event: Event) -> Dict[str, Any]:
        """Convert an Event to dict for JSON serialization."""
        return {
            "id": event.id,
            "type": event.type,
            "activity": event.activity,
            "time": self._format_datetime(event.time),
            "attributes": [
                self._event_attr_to_dict(name, attr)
                for name, attr in event.attributes.items()
            ],
            "relationships": [
                self._relationship_to_dict(qualifier, rel)
                for qualifier, rel in event.relationships.items()
            ]
        }

    def _object_to_dict(self, obj: Object) -> Dict[str, Any]:
        """Convert an Object to dict for JSON serialization."""
        return {
            "id": obj.id,
            "type": obj.type,
            "attributes": [
                self._object_attr_to_dict(name, attr)
                for name, attr in obj.attributes.items()
            ],
            "relationships": [
                self._relationship_to_dict(qualifier, rel)
                for qualifier, rel in obj.relationships.items()
            ]
        }

    def _event_attr_to_dict(self, name: str, attr: EventAttributeValue) -> Dict[str, Any]:
        """Convert EventAttributeValue to dict."""
        return {
            "name": attr.name or name,
            "value": attr.value,
            "type": attr.type
        }

    def _object_attr_to_dict(self, name: str, attr: ObjectAttributeValue) -> Dict[str, Any]:
        """Convert ObjectAttributeValue to dict."""
        result = {
            "name": attr.name or name,
            "value": attr.value,
            "type": attr.type
        }
        if attr.time is not None:
            result["time"] = self._format_datetime(attr.time)
        else:
            result["time"] = None
        return result

    def _relationship_to_dict(self, qualifier: str, rel: Relationship) -> Dict[str, str]:
        """Convert Relationship to dict."""
        return {
            "objectId": rel.object_id,
            "qualifier": qualifier
        }

    def _format_datetime(self, dt: datetime) -> float:
        """
        Format datetime as Unix float64 timestamp.

        Args:
            dt: The datetime to format

        Returns:
            Unix float64 timestamp (seconds since epoch)
        """
        return dt.timestamp()
