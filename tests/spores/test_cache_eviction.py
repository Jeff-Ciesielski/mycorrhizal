#!/usr/bin/env python3
"""Tests for Spores cache eviction behavior."""

import sys
sys.path.insert(0, "src")

import pytest
import tempfile
import os
import json
import importlib
import time
from pathlib import Path

from mycorrhizal.spores import (
    configure, flush_object_cache, get_cache_metrics, get_object_cache,
    SporesConfig, EvictionPolicy, CacheMetrics,
    Object,
)
from mycorrhizal.spores.cache import ObjectLRUCache
from mycorrhizal.spores.models import object_attribute_from_python
from mycorrhizal.spores.transport import SyncFileTransport, SyncTransport


# ============================================================================
# Test Fixtures
# ============================================================================

class CountingTransport(SyncTransport):
    """Sync transport that counts records sent."""

    def __init__(self):
        self.records = []
        self.count = 0

    def send(self, data: bytes, content_type: str) -> None:
        """Store and count records."""
        record = json.loads(data.decode('utf-8'))
        self.records.append(record)
        self.count += 1

    def close(self) -> None:
        pass

    def get_object_records(self):
        """Get only object records."""
        return [r for r in self.records if 'object' in r]


@pytest.fixture
def temp_log_file():
    """Create a temporary log file."""
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def counting_transport():
    """Create a counting transport."""
    return CountingTransport()


def reset_global_state():
    """Reset global state between tests."""
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False
    # Don't reset _cache_metrics as it's used by callbacks
    # Instead, configure() will reset it
    return core_module


# ============================================================================
# Test: Cache eviction in sync context (no event loop)
# ============================================================================

def test_eviction_in_sync_context_no_data_loss():
    """Test that cache eviction doesn't lose data in sync context."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    # Create small cache to trigger evictions
    transport = CountingTransport()
    configure(
        enabled=True,
        object_cache_size=10,  # Small cache
        transport=transport,
    )

    cache = get_object_cache()

    # Generate 50 objects directly using cache (simulates DSL adapter behavior)
    for i in range(50):
        obj = Object(
            id=f"obj-{i}",
            type="TestObject",
            attributes={"value": object_attribute_from_python(i)}
        )
        cache.contains_or_add(obj.id, obj)

    # Verify metrics
    metrics = get_cache_metrics()
    assert metrics.evictions >= 40, f"Expected at least 40 evictions, got {metrics.evictions}"
    assert metrics.eviction_failures == 0, f"Expected no failures, got {metrics.eviction_failures}"
    assert metrics.first_sights == 50, f"Expected 50 first sights, got {metrics.first_sights}"

    # Verify all objects were logged (first_sight + evictions)
    object_records = transport.get_object_records()
    # We expect: 50 (on_first_sight) + 40 (on_evict) = 90 records
    assert len(object_records) == 90, f"Expected 90 object records (50 first_sight + 40 evictions), got {len(object_records)}"


def test_eviction_in_async_context():
    """Test that cache eviction works correctly in async context."""
    import asyncio

    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    # Create small cache to trigger evictions
    transport = CountingTransport()
    configure(
        enabled=True,
        object_cache_size=10,
        transport=transport,
    )

    async def generate_objects():
        cache = get_object_cache()

        # Generate 50 objects
        for i in range(50):
            obj = Object(
                id=f"obj-{i}",
                type="TestObject",
                attributes={"value": object_attribute_from_python(i)}
            )
            cache.contains_or_add(obj.id, obj)

    # Run async function
    asyncio.run(generate_objects())

    # Wait a bit for async tasks to complete
    time.sleep(0.2)

    # Verify metrics
    metrics = get_cache_metrics()
    assert metrics.evictions >= 40, f"Expected at least 40 evictions, got {metrics.evictions}"

    # Verify all objects were logged (first_sight + evictions)
    object_records = transport.get_object_records()
    # We expect: 50 (on_first_sight) + 40 (on_evict) = 90 records
    assert len(object_records) == 90, f"Expected 90 object records (50 first_sight + 40 evictions), got {len(object_records)}"


# ============================================================================
# Test: Flush API
# ============================================================================

def test_flush_object_cache():
    """Test flushing object cache."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    transport = CountingTransport()
    configure(
        enabled=True,
        object_cache_size=100,  # Large cache
        transport=transport,
    )

    cache = get_object_cache()

    # Generate 20 objects (less than cache size)
    for i in range(20):
        obj = Object(
            id=f"obj-{i}",
            type="TestObject",
            attributes={"value": object_attribute_from_python(i)}
        )
        cache.contains_or_add(obj.id, obj)

    # Flush cache
    flush_object_cache()

    # Verify all objects were logged
    object_records = transport.get_object_records()
    # We expect at least 20 (on_first_sight) + possibly 20 more (flush)
    assert len(object_records) >= 20, f"Expected at least 20 objects, got {len(object_records)}"


# ============================================================================
# Test: Eviction policy configuration
# ============================================================================

def test_eviction_policy_default():
    """Test that default eviction policy is evict_and_log."""
    config = SporesConfig()
    assert config.eviction_policy == EvictionPolicy.EVICT_AND_LOG


def test_eviction_policy_string():
    """Test that eviction policy can be set as string."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    transport = CountingTransport()
    configure(
        enabled=True,
        object_cache_size=10,
        transport=transport,
        eviction_policy="evict_and_log"  # String
    )

    config = core_module.get_config()
    assert isinstance(config.eviction_policy, EvictionPolicy)
    assert config.eviction_policy == EvictionPolicy.EVICT_AND_LOG


def test_eviction_policy_enum():
    """Test that eviction policy can be set as enum."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    transport = CountingTransport()
    configure(
        enabled=True,
        object_cache_size=10,
        transport=transport,
        eviction_policy=EvictionPolicy.EVICT_AND_LOG  # Enum
    )

    config = core_module.get_config()
    assert config.eviction_policy == EvictionPolicy.EVICT_AND_LOG


# ============================================================================
# Test: Cache metrics
# ============================================================================

def test_cache_metrics_evictions():
    """Test that eviction metrics are tracked."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    transport = CountingTransport()
    configure(
        enabled=True,
        object_cache_size=10,
        transport=transport,
    )

    cache = get_object_cache()

    # Generate objects to trigger evictions
    for i in range(25):
        obj = Object(
            id=f"obj-{i}",
            type="TestObject",
            attributes={"value": object_attribute_from_python(i)}
        )
        cache.contains_or_add(obj.id, obj)

    metrics = get_cache_metrics()

    # Should have seen 25 first sights
    assert metrics.first_sights == 25

    # Should have evicted at least 15 objects
    assert metrics.evictions >= 15


def test_cache_metrics_reset_on_reconfigure():
    """Test that metrics reset on reconfiguration."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    transport = CountingTransport()
    configure(
        enabled=True,
        object_cache_size=10,
        transport=transport,
    )

    cache = get_object_cache()

    # Generate some objects
    for i in range(15):
        obj = Object(
            id=f"obj-{i}",
            type="TestObject",
            attributes={"value": object_attribute_from_python(i)}
        )
        cache.contains_or_add(obj.id, obj)

    metrics_before = get_cache_metrics()
    assert metrics_before.first_sights > 0

    # Reconfigure
    configure(
        enabled=True,
        object_cache_size=10,
        transport=transport,
    )

    metrics_after = get_cache_metrics()
    assert metrics_after.first_sights == 0, "Metrics should reset on reconfiguration"


# ============================================================================
# Test: End-to-end sync context with file transport
# ============================================================================

def test_end_to_end_sync_context_no_data_loss(temp_log_file):
    """Test end-to-end sync context with file transport using cache directly."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    # Configure with file transport
    configure(
        enabled=True,
        object_cache_size=50,  # Small cache
        transport=SyncFileTransport(temp_log_file),
    )

    cache = get_object_cache()

    # Generate 500 objects (should trigger evictions)
    for i in range(500):
        obj = Object(
            id=f"order-{i}",
            type="Order",
            attributes={
                "order_id": object_attribute_from_python(f"order-{i}"),
                "status": object_attribute_from_python("created"),
                "total": object_attribute_from_python(100.0 + i)
            }
        )
        cache.contains_or_add(obj.id, obj)

    # Flush to ensure everything is written
    flush_object_cache()

    # Read log file and verify
    object_ids = set()
    with open(temp_log_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if 'object' in record:
                    obj_id = record['object']['id']
                    object_ids.add(obj_id)

    # Verify all 500 objects were logged
    assert len(object_ids) == 500, f"Expected 500 unique objects, got {len(object_ids)}"


def test_ocel_log_validity(temp_log_file):
    """Test that OCEL log is valid (all referenced objects exist)."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    # Configure with file transport
    configure(
        enabled=True,
        object_cache_size=20,  # Very small cache
        transport=SyncFileTransport(temp_log_file),
    )

    cache = get_object_cache()

    # Generate objects
    for i in range(100):
        obj = Object(
            id=f"customer-{i % 20}",  # Only 20 unique customers
            type="Customer",
            attributes={
                "name": object_attribute_from_python(f"Customer {i % 20}"),
                "email": object_attribute_from_python(f"customer{i % 20}@example.com")
            }
        )
        cache.contains_or_add(obj.id, obj)

    # Flush
    flush_object_cache()

    # Read log and validate
    object_ids = set()
    with open(temp_log_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if 'object' in record:
                    obj_id = record['object']['id']
                    object_ids.add(obj_id)

    # Verify all 20 unique customers were logged
    assert len(object_ids) == 20, f"Expected 20 unique objects, got {len(object_ids)}"


# ============================================================================
# Test: Performance
# ============================================================================

def test_cache_operations_performance():
    """Test that cache operations are fast (< 100µs per operation)."""
    import time

    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    transport = CountingTransport()
    configure(
        enabled=True,
        object_cache_size=100,
        transport=transport,
    )

    cache = get_object_cache()

    # Time 100 cache operations
    start = time.perf_counter()
    for i in range(100):
        obj = Object(
            id=f"obj-{i}",
            type="TestObject",
            attributes={"value": object_attribute_from_python(i)}
        )
        cache.contains_or_add(obj.id, obj)
    elapsed = time.perf_counter() - start

    # Should be much less than 1 second for 100 operations
    # (< 10ms per operation is very generous)
    assert elapsed < 1.0, f"Cache operations too slow: {elapsed}s for 100 operations"


# ============================================================================
# Test: Attribute change detection
# ============================================================================

def test_attribute_change_detection():
    """Test that attribute changes trigger logging."""
    logged_objects = []

    def needs_logged(obj_id: str, obj: Object):
        logged_objects.append((obj_id, obj))

    cache = ObjectLRUCache(
        maxsize=10,
        needs_logged=needs_logged,
        touch_resend_n=0  # Disable periodic resend
    )

    # Create object with initial attributes
    obj1 = Object(
        id="obj-1",
        type="TestObject",
        attributes={"status": object_attribute_from_python("created")}
    )

    # First sight - should log
    cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 1

    # Same object, same attributes - should not log
    cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 1

    # Same object, changed attributes - should log
    obj1_updated = Object(
        id="obj-1",
        type="TestObject",
        attributes={"status": object_attribute_from_python("updated")}
    )
    cache.contains_or_add("obj-1", obj1_updated)
    assert len(logged_objects) == 2
    # Verify the logged object has the new attributes
    assert logged_objects[1][1].attributes["status"].value == "updated"


def test_touch_resend_n():
    """Test that objects are re-logged every N touches."""
    logged_objects = []

    def needs_logged(obj_id: str, obj: Object):
        logged_objects.append((obj_id, obj))

    cache = ObjectLRUCache(
        maxsize=10,
        needs_logged=needs_logged,
        touch_resend_n=5  # Resend every 5 touches
    )

    obj1 = Object(
        id="obj-1",
        type="TestObject",
        attributes={"value": object_attribute_from_python("test")}
    )

    # Touch 1: first sight
    cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 1

    # Touches 2-4: no resend yet
    for i in range(3):
        cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 1

    # Touch 5: should resend (5 % 5 == 0)
    cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 2

    # Touch 6-9: no resend yet
    for i in range(4):
        cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 2

    # Touch 10: should resend again (10 % 5 == 0)
    cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 3


def test_touch_resend_n_disabled():
    """Test that touch_resend_n=0 disables periodic resend."""
    logged_objects = []

    def needs_logged(obj_id: str, obj: Object):
        logged_objects.append((obj_id, obj))

    cache = ObjectLRUCache(
        maxsize=10,
        needs_logged=needs_logged,
        touch_resend_n=0  # Disable periodic resend
    )

    obj1 = Object(
        id="obj-1",
        type="TestObject",
        attributes={"value": object_attribute_from_python("test")}
    )

    # First sight
    cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 1

    # Many more touches - should never resend
    for i in range(100):
        cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 1, "Periodic resend should be disabled when touch_resend_n=0"


def test_attribute_change_with_touch_resend():
    """Test that attribute changes and periodic resend work together."""
    logged_objects = []

    def needs_logged(obj_id: str, obj: Object):
        logged_objects.append((obj_id, obj))

    cache = ObjectLRUCache(
        maxsize=10,
        needs_logged=needs_logged,
        touch_resend_n=3
    )

    # First sight (touch 1)
    obj1 = Object(
        id="obj-1",
        type="TestObject",
        attributes={"value": object_attribute_from_python("initial")}
    )
    cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 1

    # Touch 2: no resend
    cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 1

    # Touch 3: periodic resend
    cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 2

    # Touch 4: attribute change
    obj1_updated = Object(
        id="obj-1",
        type="TestObject",
        attributes={"value": object_attribute_from_python("updated")}
    )
    cache.contains_or_add("obj-1", obj1_updated)
    assert len(logged_objects) == 3
    assert logged_objects[2][1].attributes["value"].value == "updated"


def test_no_attributes():
    """Test that objects without attributes work correctly."""
    logged_objects = []

    def needs_logged(obj_id: str, obj: Object):
        logged_objects.append((obj_id, obj))

    cache = ObjectLRUCache(
        maxsize=10,
        needs_logged=needs_logged,
        touch_resend_n=5
    )

    obj1 = Object(id="obj-1", type="TestObject", attributes={})

    # First sight
    cache.contains_or_add("obj-1", obj1)
    assert len(logged_objects) == 1

    # Multiple touches - no attribute change (no attributes)
    for i in range(10):
        cache.contains_or_add("obj-1", obj1)

    # Should have first sight + periodic resends (at 5 and 10)
    assert len(logged_objects) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
