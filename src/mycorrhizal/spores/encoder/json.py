#!/usr/bin/env python3
"""
Spores JSON Encoder

JSON encoder for OCEL LogRecords.
Compatible with the Go OCEL library format.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

from .base import Encoder
from ..models import LogRecord, Event, Object, Relationship, EventAttributeValue, ObjectAttributeValue


class JSONEncoder(Encoder):
    """
    JSON encoder for OCEL LogRecords.

    Produces OCEL-compatible JSON.

    Example output:
        {
          "event": {
            "id": "evt-123",
            "type": "process_item",
            "activity": null,
            "time": "2025-01-11T12:34:56.789000000Z",
            "attributes": [
              {"name": "priority", "value": "high", "type": "string", "time": null}
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
              {"name": "status", "value": "pending", "type": "string", "time": null}
            ],
            "relationships": []
          }
        }
    """

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

    def _format_datetime(self, dt: datetime) -> str:
        """
        Format datetime to RFC3339Nano.

        Args:
            dt: The datetime to format

        Returns:
            ISO format string with nanosecond precision
        """
        # Python's isoformat() produces RFC3339-compatible output
        # For nanosecond precision, we need to ensure it has 9 digits
        iso = dt.isoformat()

        # Add microseconds if not present (Python 3.11+ has timespec='nanoseconds')
        if '.' not in iso:
            iso += '.000000000'

        # Ensure we have 9 digits of fractional seconds
        if '.' in iso:
            main, frac = iso.split('.')
            # Pad or truncate to 9 digits
            frac = (frac + '0' * 9)[:9]
            iso = f"{main}.{frac}"

        # Ensure timezone is Z (UTC) or offset
        if iso.endswith('+00:00'):
            iso = iso[:-6] + 'Z'

        return iso
