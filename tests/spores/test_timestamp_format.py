#!/usr/bin/env python3
"""Tests for Spores timestamp encoding (Unix float64 only)."""

import sys
sys.path.insert(0, "src")

import pytest
import asyncio
import json
from datetime import datetime, timezone
from pydantic import BaseModel

from mycorrhizal.spores import (
    configure, get_config,
    Event, Object, LogRecord,
    EventAttributeValue, ObjectAttributeValue,
    Relationship,
    get_spore_async, get_spore_sync,
)
from mycorrhizal.spores.encoder import JSONEncoder
from mycorrhizal.spores.transport import Transport, SyncTransport


# ============================================================================
# Test Fixtures
# ============================================================================

class MockTransport(Transport):
    """Mock async transport for testing."""

    def __init__(self):
        self.records = []

    async def send(self, data: bytes, content_type: str) -> None:
        """Store records instead of sending."""
        record = json.loads(data.decode('utf-8'))
        self.records.append(record)

    def is_async(self) -> bool:
        return False

    async def close(self) -> None:
        pass


class MockSyncTransport(SyncTransport):
    """Mock sync transport for testing."""

    def __init__(self):
        self.records = []

    def send(self, data: bytes, content_type: str) -> None:
        """Store records instead of sending."""
        record = json.loads(data.decode('utf-8'))
        self.records.append(record)

    def close(self) -> None:
        pass


@pytest.fixture
def reset_config():
    """Reset global config after each test."""
    yield
    # Reset config by reimporting the module
    import importlib
    import mycorrhizal.spores.core as core
    importlib.reload(core)


# ============================================================================
# Unix Float64 Format Tests
# ============================================================================

def test_default_timestamp_is_unix_float64(reset_config):
    """Test that timestamps are Unix float64 by default."""
    transport = MockSyncTransport()
    configure(transport=transport)
    encoder = JSONEncoder()

    # Create event with known timestamp
    dt = datetime(2025, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
    event = Event(
        id="evt-1",
        type="test_event",
        time=dt,
        attributes={
            "priority": EventAttributeValue(
                name="priority",
                value="high",
                type="string"
            )
        }
    )

    data = encoder.encode(LogRecord(event=event))
    parsed = json.loads(data)

    # Check timestamp is a number
    assert "event" in parsed
    assert isinstance(parsed["event"]["time"], (int, float))

    # Verify it's the correct Unix timestamp
    expected_ts = dt.timestamp()
    assert abs(parsed["event"]["time"] - expected_ts) < 1e-6


def test_unix_float64_event_timestamp(reset_config):
    """Test event timestamp in Unix float64 format."""
    transport = MockSyncTransport()
    configure(transport=transport)

    encoder = JSONEncoder()

    # Create event with known timestamp
    dt = datetime(2025, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
    event = Event(
        id="evt-1",
        type="test_event",
        time=dt,
        attributes={
            "priority": EventAttributeValue(
                name="priority",
                value="high",
                type="string"
            )
        }
    )

    data = encoder.encode(LogRecord(event=event))
    parsed = json.loads(data)

    # Check timestamp is a number
    assert "event" in parsed
    assert isinstance(parsed["event"]["time"], (int, float))

    # Verify it's the correct Unix timestamp
    expected_ts = dt.timestamp()
    assert abs(parsed["event"]["time"] - expected_ts) < 1e-6


def test_unix_float64_object_attribute_timestamp(reset_config):
    """Test object attribute timestamp in Unix float64 format."""
    transport = MockSyncTransport()
    configure(transport=transport)

    encoder = JSONEncoder()

    # Create object with timestamped attribute
    dt = datetime(2025, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
    obj = Object(
        id="obj-1",
        type="TestObject",
        attributes={
            "status": ObjectAttributeValue(
                name="status",
                value="active",
                type="string",
                time=dt
            )
        }
    )

    data = encoder.encode(LogRecord(object=obj))
    parsed = json.loads(data)

    # Check attribute timestamp is a number
    assert "object" in parsed
    attrs = parsed["object"]["attributes"]
    status_attr = next(a for a in attrs if a["name"] == "status")
    assert isinstance(status_attr["time"], (int, float))

    # Verify it's the correct Unix timestamp
    expected_ts = dt.timestamp()
    assert abs(status_attr["time"] - expected_ts) < 1e-6


def test_unix_float64_microsecond_precision(reset_config):
    """Test that Unix float64 format preserves microsecond precision (Python datetime limit)."""
    transport = MockSyncTransport()
    configure(transport=transport)

    encoder = JSONEncoder()

    # Create event with microsecond precision (Python's max precision)
    dt = datetime(2025, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
    event = Event(id="evt-1", type="test", time=dt)

    data = encoder.encode(LogRecord(event=event))
    parsed = json.loads(data)

    # Unix timestamps from Python have microsecond precision
    # Verify the fractional seconds part
    ts = parsed["event"]["time"]
    expected_ts = dt.timestamp()
    # Should be very close (within microsecond precision)
    assert abs(ts - expected_ts) < 1e-6

    # Check that fractional part contains microseconds
    fractional = ts - int(ts)
    # 123456 microseconds = 0.123456 seconds
    assert abs(fractional - 0.123456) < 1e-6


def test_unix_float64_epoch(reset_config):
    """Test Unix timestamp for epoch (1970-01-01 UTC)."""
    transport = MockSyncTransport()
    configure(transport=transport)

    encoder = JSONEncoder()

    # Epoch timestamp
    dt = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    event = Event(id="evt-1", type="test", time=dt)

    data = encoder.encode(LogRecord(event=event))
    parsed = json.loads(data)

    # Unix epoch should be 0.0
    assert parsed["event"]["time"] == 0.0


def test_unix_float64_negative_timestamp(reset_config):
    """Test Unix timestamp for dates before epoch."""
    transport = MockSyncTransport()
    configure(transport=transport)

    encoder = JSONEncoder()

    # Date before epoch (1969-12-31 23:59:59 UTC = -1 second)
    dt = datetime(1969, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    event = Event(id="evt-1", type="test", time=dt)

    data = encoder.encode(LogRecord(event=event))
    parsed = json.loads(data)

    # Should be negative
    assert parsed["event"]["time"] == -1.0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_end_to_end_unix_float64(reset_config):
    """Test complete logging pipeline with Unix float64 format."""
    transport = MockTransport()
    configure(transport=transport)

    spore = get_spore_async(__name__)

    @spore.log_event(event_type="TestEvent")
    async def test_function():
        return "result"

    await test_function()
    await asyncio.sleep(0.01)  # Wait for async logging

    assert len(transport.records) >= 1
    event_record = [r for r in transport.records if r.get("event") is not None][0]

    # Check timestamp is number
    assert isinstance(event_record["event"]["time"], (int, float))


# ============================================================================
# Acorn Compatibility Tests
# ============================================================================

def test_unix_format_for_acorn(reset_config):
    """Test that Unix format is compatible with Acorn OCEL."""
    transport = MockSyncTransport()
    configure(transport=transport)

    encoder = JSONEncoder()

    # Create a realistic event
    dt = datetime(2025, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
    event = Event(
        id="evt-order-created-123",
        type="OrderCreated",
        activity="create_order",
        time=dt,
        attributes={
            "order_id": EventAttributeValue(name="order_id", value="ORD-123", type="string"),
            "total": EventAttributeValue(name="total", value="99.99", type="float"),
        },
        relationships={
            "order": Relationship(object_id="obj-order-123", qualifier="order")
        }
    )

    data = encoder.encode(LogRecord(event=event))
    parsed = json.loads(data)

    # Validate Acorn-compatible structure
    assert parsed["event"]["id"] == "evt-order-created-123"
    assert parsed["event"]["type"] == "OrderCreated"
    assert isinstance(parsed["event"]["time"], (int, float))
    assert len(parsed["event"]["attributes"]) == 2
    assert len(parsed["event"]["relationships"]) == 1

    # Verify timestamp can be parsed back
    ts = parsed["event"]["time"]
    reconstructed_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    # Should match within microsecond precision
    assert abs((reconstructed_dt - dt).total_seconds()) < 1e-6


# ============================================================================
# Edge Cases
# ============================================================================

def test_null_object_attribute_time(reset_config):
    """Test that null object attribute timestamps are handled correctly."""
    transport = MockSyncTransport()
    configure(transport=transport)

    encoder = JSONEncoder()

    # Object attribute with null time
    obj = Object(
        id="obj-1",
        type="TestObject",
        attributes={
            "status": ObjectAttributeValue(
                name="status",
                value="active",
                type="string",
                time=None  # Null timestamp
            )
        }
    )

    data = encoder.encode(LogRecord(object=obj))
    parsed = json.loads(data)

    # Should be null
    attrs = parsed["object"]["attributes"]
    status_attr = next(a for a in attrs if a["name"] == "status")
    assert status_attr["time"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
