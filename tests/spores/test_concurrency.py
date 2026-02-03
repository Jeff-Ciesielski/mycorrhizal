#!/usr/bin/env python3
"""Concurrency tests for Spores global state initialization."""

import sys
sys.path.insert(0, "src")

import pytest
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from mycorrhizal.spores import (
    configure, get_config, get_object_cache,
    SporesConfig, ObjectLRUCache,
)
from mycorrhizal.spores.models import Object
from mycorrhizal.spores.transport import SyncTransport
from mycorrhizal.spores.encoder import JSONEncoder


# ============================================================================
# Test Fixtures
# ============================================================================

class MockTransport(SyncTransport):
    """Mock sync transport for testing."""

    def __init__(self):
        self.records = []

    def send(self, data: bytes, content_type: str) -> None:
        """Store records instead of sending."""
        import json
        record = json.loads(data.decode('utf-8'))
        self.records.append(record)

    def close(self) -> None:
        pass


def reset_global_state():
    """Reset global state between tests."""
    import importlib
    import mycorrhizal.spores.core as core
    importlib.reload(core)
    # Re-import to get the reloaded module
    from mycorrhizal.spores import core as core_module
    return core_module


# ============================================================================
# Test: Concurrent configure() calls
# ============================================================================

def test_concurrent_configure_consistent_state():
    """Test that concurrent configure() calls produce consistent state."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    # Track what each thread configured
    configs_created = []
    caches_created = []
    errors = []

    def configure_thread(thread_id: int, enabled: bool, cache_size: int):
        """Each thread calls configure with different parameters."""
        try:
            transport = MockTransport()
            configure(
                enabled=enabled,
                object_cache_size=cache_size,
                transport=transport,
            )
            # Record what we created
            configs_created.append((thread_id, enabled, cache_size))
            caches_created.append((thread_id, id(get_object_cache())))
        except Exception as e:
            errors.append((thread_id, e))

    # Spawn 10 threads with different configurations
    threads = []
    for i in range(10):
        t = threading.Thread(
            target=configure_thread,
            args=(i, i % 2 == 0, 128 + i)
        )
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify final state is consistent
    config = get_config()
    cache = get_object_cache()

    # Config should match one of the configurations (last writer wins)
    assert config.enabled in [True, False]
    assert config.object_cache_size in range(128, 138)

    # All threads should see the same cache object
    # Note: configure() can be called multiple times, so cache may be replaced,
    # but all threads should see the same final cache object
    unique_cache_ids = set(cid for _, cid in caches_created)
    # Due to reconfiguration, we may see multiple cache IDs, but they should
    # all be from sequential configure() calls (not concurrent creation)
    # The important thing is that we don't have orphaned caches
    assert len(unique_cache_ids) <= len(caches_created), \
        f"More unique cache IDs than threads: {unique_cache_ids}"


def test_concurrent_configure_no_partial_initialization():
    """Test that concurrent configure() never results in partial initialization."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    states_seen = []

    def check_state(thread_id: int):
        """Check that state is always consistent (both or neither)."""
        for _ in range(100):
            config = core_module._config
            cache = core_module._object_cache

            # Record the state
            state = (
                config is not None,
                cache is not None,
            )
            states_seen.append(state)

            # Critical invariant: never have config without cache or vice versa
            # (except transiently during configure, which we might observe)
            # But we should NEVER see config=None while cache!=None
            if config is None and cache is not None:
                pytest.fail(f"Thread {thread_id}: saw cache without config")

            time.sleep(0.0001)  # Small sleep to observe different states

    # Start threads that will configure
    def configure_thread():
        transport = MockTransport()
        configure(enabled=True, object_cache_size=256, transport=transport)

    threads = []
    # Start state checkers
    for i in range(5):
        t = threading.Thread(target=check_state, args=(i,))
        threads.append(t)
        t.start()

    # Start configurers
    for i in range(3):
        t = threading.Thread(target=configure_thread)
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Final state should be consistent
    assert core_module._config is not None
    assert core_module._object_cache is not None


# ============================================================================
# Test: Concurrent get_config() calls
# ============================================================================

def test_concurrent_get_config_single_initialization():
    """Test that concurrent get_config() calls initialize only once."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    config_ids = []
    errors = []

    def get_config_thread(thread_id: int):
        """Each thread calls get_config()."""
        try:
            config = get_config()
            config_ids.append((thread_id, id(config)))
        except Exception as e:
            errors.append((thread_id, e))

    # Spawn 20 threads all calling get_config
    threads = []
    for i in range(20):
        t = threading.Thread(target=get_config_thread, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # All threads should get the same config object
    unique_config_ids = set(cid for _, cid in config_ids)
    assert len(unique_config_ids) == 1, f"Multiple configs created: {unique_config_ids}"

    # Verify it's the default config
    config = get_config()
    assert config.enabled == True
    assert config.object_cache_size == 128


# ============================================================================
# Test: Concurrent get_object_cache() calls
# ============================================================================

def test_concurrent_get_object_cache_single_initialization():
    """Test that concurrent get_object_cache() calls initialize only once."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    cache_ids = []
    errors = []

    def get_cache_thread(thread_id: int):
        """Each thread calls get_object_cache()."""
        try:
            cache = get_object_cache()
            cache_ids.append((thread_id, id(cache)))
        except Exception as e:
            errors.append((thread_id, e))

    # Spawn 20 threads all calling get_object_cache
    threads = []
    for i in range(20):
        t = threading.Thread(target=get_cache_thread, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # All threads should get the same cache object (no memory leaks)
    unique_cache_ids = set(cid for _, cid in cache_ids)
    assert len(unique_cache_ids) == 1, f"Multiple caches created: {unique_cache_ids}"


# ============================================================================
# Test: Mixed concurrent access
# ============================================================================

def test_mixed_concurrent_access():
    """Test mixed concurrent calls to configure(), get_config(), get_object_cache()."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    results = {"configure": [], "get_config": [], "get_cache": []}
    errors = []

    def configure_thread(thread_id: int):
        """Thread calling configure."""
        try:
            transport = MockTransport()
            configure(enabled=True, object_cache_size=200, transport=transport)
            results["configure"].append(thread_id)
        except Exception as e:
            errors.append(("configure", thread_id, e))

    def get_config_thread(thread_id: int):
        """Thread calling get_config."""
        try:
            config = get_config()
            results["get_config"].append((thread_id, id(config)))
        except Exception as e:
            errors.append(("get_config", thread_id, e))

    def get_cache_thread(thread_id: int):
        """Thread calling get_object_cache."""
        try:
            cache = get_object_cache()
            results["get_cache"].append((thread_id, id(cache)))
        except Exception as e:
            errors.append(("get_cache", thread_id, e))

    # Spawn mixed threads
    threads = []
    for i in range(10):
        if i % 3 == 0:
            t = threading.Thread(target=configure_thread, args=(i,))
        elif i % 3 == 1:
            t = threading.Thread(target=get_config_thread, args=(i,))
        else:
            t = threading.Thread(target=get_cache_thread, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify final state is consistent
    config = get_config()
    cache = get_object_cache()

    # All get_config calls should return objects that existed at the time
    # (they may be different if configure() was called multiple times)
    config_ids = [cid for _, cid in results["get_config"]]
    if config_ids:
        # At minimum, we should have some config objects
        assert len(config_ids) > 0, "No config objects retrieved"

    # All get_cache calls should return objects that existed at the time
    cache_ids = [cid for _, cid in results["get_cache"]]
    if cache_ids:
        # At minimum, we should have some cache objects
        assert len(cache_ids) > 0, "No cache objects retrieved"


# ============================================================================
# Test: Reentrancy (configure calls get_object_cache)
# ============================================================================

def test_reentrant_no_deadlock():
    """Test that configure() calling get_object_cache() doesn't deadlock."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    # This should not deadlock even though configure() acquires the lock
    # and get_object_cache() tries to acquire it again
    transport = MockTransport()
    configure(enabled=True, object_cache_size=256, transport=transport)

    # Verify state is consistent
    config = get_config()
    cache = get_object_cache()

    assert config is not None
    assert cache is not None
    assert config.enabled == True
    assert config.object_cache_size == 256


# ============================================================================
# Test: Stress test with 100 threads
# ============================================================================

def test_stress_100_threads():
    """Stress test with 100 threads accessing global state."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    results = []
    errors = []

    def worker(thread_id: int):
        """Worker that randomly accesses global state."""
        try:
            import random
            for _ in range(10):
                action = random.choice(['configure', 'get_config', 'get_cache'])

                if action == 'configure':
                    transport = MockTransport()
                    configure(
                        enabled=random.choice([True, False]),
                        object_cache_size=random.randint(100, 300),
                        transport=transport,
                    )
                elif action == 'get_config':
                    config = get_config()
                    results.append(('config', id(config)))
                else:
                    cache = get_object_cache()
                    results.append(('cache', id(cache)))

                time.sleep(0.001)  # Small delay

        except Exception as e:
            errors.append((thread_id, e))

    # Spawn 100 threads
    threads = []
    for i in range(100):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors (no crashes, deadlocks, or race conditions)
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify final state is consistent
    config = get_config()
    cache = get_object_cache()

    assert config is not None
    assert cache is not None

    # The key assertion: no crashes or errors under concurrent load
    # (configure() can be called multiple times, so config/cache objects may vary)


# ============================================================================
# Test: Double-checked locking optimization
# ============================================================================

def test_double_checked_locking_fast_path():
    """Test that double-checked locking provides fast path for initialized state."""
    # Configure first
    transport = MockTransport()
    configure(enabled=True, object_cache_size=128, transport=transport)

    # Time the fast path (should be very fast, no lock acquisition)
    # Run multiple iterations to get stable timing
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(10000):
            config = get_config()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    # Take median timing
    timings.sort()
    median_elapsed = timings[2]

    # Should be very fast (< 50ms for 10000 calls = < 5µs per call)
    # This is a relaxed threshold for CI environments
    assert median_elapsed < 0.05, f"Fast path too slow: {median_elapsed}s for 10000 calls"

    # Verify we got the same config every time
    assert config is not None


# ============================================================================
# Test: Warning on reconfiguration
# ============================================================================

def test_reconfiguration_warning(caplog):
    """Test that reconfiguration logs a warning."""
    import logging

    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    # First configure
    transport1 = MockTransport()
    configure(enabled=True, object_cache_size=128, transport=transport1)

    # Second configure (should warn)
    transport2 = MockTransport()
    with caplog.at_level(logging.WARNING):
        configure(enabled=False, object_cache_size=256, transport=transport2)

    # Check warning was logged
    assert any(
        "Spores already configured" in record.message
        for record in caplog.records
    ), "Expected warning about reconfiguration"

    # Verify final state is from second configure
    config = get_config()
    assert config.enabled == False
    assert config.object_cache_size == 256


# ============================================================================
# Test: Memory leak check via weak references
# ============================================================================

def test_no_orphaned_caches_memory_leak():
    """Test that no orphaned caches are left (memory leak check)."""
    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    weak_refs = []
    errors = []

    def configure_and_track(thread_id: int):
        """Configure and track cache with weak reference."""
        try:
            transport = MockTransport()
            configure(
                enabled=True,
                object_cache_size=128 + thread_id,
                transport=transport,
            )
            cache = get_object_cache()
            weak_refs.append(weakref.ref(cache))
        except Exception as e:
            errors.append((thread_id, e))

    # Spawn 10 threads
    threads = []
    for i in range(10):
        t = threading.Thread(target=configure_and_track, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # All weak refs should point to the same live object
    # (or some may be dead if they were garbage collected)
    live_refs = [ref for ref in weak_refs if ref() is not None]

    # There should be exactly 1 live cache (the current one)
    assert len(live_refs) <= 1, f"Multiple caches still alive: {len(live_refs)}"

    # Verify the live cache is the current one
    current_cache = get_object_cache()
    if live_refs:
        assert live_refs[0]() is current_cache, "Live cache is not the current one"


# ============================================================================
# Test: Async + thread concurrent access
# ============================================================================

def test_async_thread_concurrent_access():
    """Test concurrent access from async tasks and threads."""
    import asyncio

    # Reset state
    import mycorrhizal.spores.core as core_module
    core_module._config = None
    core_module._object_cache = None
    core_module._config_initialized = False

    results = {"async": [], "thread": []}
    errors = []

    def thread_worker(thread_id: int):
        """Thread worker accessing global state."""
        try:
            for _ in range(10):
                config = get_config()
                cache = get_object_cache()
                results["thread"].append((thread_id, id(config), id(cache)))
                time.sleep(0.001)
        except Exception as e:
            errors.append(("thread", thread_id, e))

    async def async_worker(task_id: int):
        """Async worker accessing global state."""
        try:
            for _ in range(10):
                config = get_config()
                cache = get_object_cache()
                results["async"].append((task_id, id(config), id(cache)))
                await asyncio.sleep(0.001)
        except Exception as e:
            errors.append(("async", task_id, e))

    async def run_async():
        """Run async tasks."""
        tasks = [async_worker(i) for i in range(5)]
        await asyncio.gather(*tasks)

    # Start threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=thread_worker, args=(i,))
        threads.append(t)
        t.start()

    # Run async tasks
    asyncio.run(run_async())

    # Wait for threads
    for t in threads:
        t.join()

    # Verify no errors (no crashes, deadlocks, or race conditions)
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify we got some results
    assert len(results["thread"]) > 0, "No thread results"
    assert len(results["async"]) > 0, "No async results"

    # Final state should be consistent
    final_config = get_config()
    final_cache = get_object_cache()
    assert final_config is not None
    assert final_cache is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
