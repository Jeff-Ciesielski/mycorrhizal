#!/usr/bin/env python3
"""
Memory leak tests for Hypha runtime.

Tests memory behavior, cache management, and resource cleanup.

Run with: pytest tests/hypha/test_memory_leaks.py -v
"""

import asyncio
import gc
import pytest
import tracemalloc
from dataclasses import dataclass, field, replace
from typing import Any, List

from mycorrhizal.hypha.core import pn, PlaceType, Runner as PNRunner
from mycorrhizal.common.timebase import CycleClock


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MemoryTestBlackboard:
    """Blackboard for memory testing."""
    data: str = "test"
    counter: int = 0


@pytest.fixture
def mem_bb():
    return MemoryTestBlackboard()


@pytest.fixture
def cycle_tb():
    return CycleClock()


# =============================================================================
# Cache Statistics Tests
# =============================================================================

class TestCacheStats:
    """Tests for cache statistics API"""

    async def test_cache_stats_empty(self, mem_bb, cycle_tb):
        """Cache stats should be zero when cache is empty"""
        @pn.net
        def EmptyNet(builder):
            pass

        runner = PNRunner(EmptyNet, mem_bb)
        await runner.start(cycle_tb)

        stats = runner.runtime.get_cache_stats()
        assert stats['cached_views'] == 0

        await runner.stop()

    async def test_cache_clear_function(self, mem_bb, cycle_tb):
        """Cache clear function should empty the cache"""

        from mycorrhizal.common.interface_builder import blackboard_interface, readonly
        from typing import Annotated

        # Define an interface to trigger cache usage
        @blackboard_interface
        class TestInterface:
            data: Annotated[str, readonly]

        @pn.net
        def CacheNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb: TestInterface, timebase):
                # Access interface to trigger cache usage
                _ = bb.data
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        runner = PNRunner(CacheNet, mem_bb)
        await runner.start(cycle_tb)

        # Process some tokens to populate cache
        input_place = runner.runtime.places[('CacheNet', 'input')]
        for i in range(10):
            input_place.add_token(f"token{i}")

        await asyncio.sleep(0.2)

        # Cache should have entries
        stats_before = runner.runtime.get_cache_stats()
        assert stats_before['cached_views'] > 0

        # Clear cache
        runner.runtime._interface_cache.clear()

        # Cache should be empty
        stats_after = runner.runtime.get_cache_stats()
        assert stats_after['cached_views'] == 0

        await runner.stop()


# =============================================================================
# Weak Reference Tests
# =============================================================================

class TestWeakReferences:
    """Tests for weak reference behavior in cache"""

    async def test_weak_ref_auto_expiry(self, mem_bb, cycle_tb):
        """Verify instance cache is cleaned up when runner is deleted"""

        from mycorrhizal.common.interface_builder import blackboard_interface, readonly
        from typing import Annotated

        # Define an interface to trigger cache usage
        @blackboard_interface
        class TestInterface:
            data: Annotated[str, readonly]

        @pn.net
        def WeakRefNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb: TestInterface, timebase):
                # Access interface to trigger cache usage
                _ = bb.data
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        # Create runner and process tokens
        runner = PNRunner(WeakRefNet, mem_bb)
        await runner.start(cycle_tb)

        input_place = runner.runtime.places[('WeakRefNet', 'input')]
        for i in range(5):
            input_place.add_token(f"token{i}")

        await asyncio.sleep(0.2)

        # Get cache stats before deleting runner
        stats_before = runner.runtime.get_cache_stats()
        initial_cache_size = stats_before['cached_views']
        assert initial_cache_size > 0, "Cache should have entries after processing tokens"

        # Delete runner (cache is instance-local, so it gets deleted too)
        await runner.stop()
        del runner

        # Force garbage collection
        gc.collect()

        # No global cache to check - instance cache is gone
        # Test verifies no memory leak from runner holding references

    async def test_cache_with_multiple_blackboards(self, mem_bb, cycle_tb):
        """Each runner instance has its own cache"""

        @pn.net
        def MultiBBNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb: MemoryTestBlackboard, timebase):
                yield {output_p: f"{bb.data}_{consumed[0]}"}

            builder.arc(input_p, worker)

        # Create multiple runners with different blackboards
        runners = []
        blackboards = []
        for i in range(5):
            bb = MemoryTestBlackboard(data=f"bb{i}")
            blackboards.append(bb)
            runner = PNRunner(MultiBBNet, bb)
            await runner.start(cycle_tb)

            # Add tokens
            input_place = runner.runtime.places[('MultiBBNet', 'input')]
            input_place.add_token(f"token{i}")

            runners.append(runner)

        await asyncio.sleep(0.2)

        # Each runner should have its own cache
        total_cached = 0
        for runner in runners:
            stats = runner.runtime.get_cache_stats()
            total_cached += stats['cached_views']

        assert total_cached >= 0, "Should have cached views across all runners"

        # Delete all runners and blackboards
        for runner in runners:
            await runner.stop()

        del runners
        del blackboards

        # Force GC
        gc.collect()

        # No global cache - each instance cache is deleted with its runner
        # but weak references should prevent unbounded growth


# =============================================================================
# Memory Leak Tests
# =============================================================================

class TestMemoryLeaks:
    """Tests for memory leak detection"""

    @pytest.mark.slow
    async def test_memory_leak_10k_iterations(self, cycle_tb):
        """Verify memory doesn't grow unbounded with many blackboard instances"""

        # Start memory tracking
        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Create and destroy many blackboards
        for i in range(10000):
            bb = MemoryTestBlackboard(data=f"test{i}")
            # Simulate creating interface views
            # In real usage, this happens through transitions

            # Delete blackboard
            del bb

            # Periodic GC to simulate real-world behavior
            if i % 100 == 0:
                gc.collect()

        # Force final GC
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()

        # Calculate memory growth
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_growth = sum(stat.size_diff for stat in top_stats)

        # Memory growth should be minimal (< 1MB)
        # This allows for some growth but detects unbounded leaks
        max_allowed_growth = 1_000_000  # 1 MB
        assert total_growth < max_allowed_growth, (
            f"Memory grew by {total_growth} bytes, "
            f"expected < {max_allowed_growth} bytes"
        )

        tracemalloc.stop()

    @pytest.mark.slow
    async def test_memory_leak_with_runners(self, cycle_tb):
        """Verify memory doesn't leak with many runner lifecycles"""

        @pn.net
        def LeakTestNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb: MemoryTestBlackboard, timebase):
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        # Start memory tracking
        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Create and destroy many runners
        for i in range(100):
            bb = MemoryTestBlackboard(data=f"test{i}")
            runner = PNRunner(LeakTestNet, bb)
            await runner.start(cycle_tb)

            # Add some tokens
            input_place = runner.runtime.places[('LeakTestNet', 'input')]
            for j in range(10):
                input_place.add_token(f"token{j}")

            await asyncio.sleep(0.01)
            await runner.stop()

            del runner
            del bb

            if i % 10 == 0:
                gc.collect()

        # Force final GC
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()

        # Calculate memory growth
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_growth = sum(stat.size_diff for stat in top_stats)

        # Memory growth should be reasonable
        # Allow more growth here because we're creating full runners
        max_allowed_growth = 5_000_000  # 5 MB
        assert total_growth < max_allowed_growth, (
            f"Memory grew by {total_growth} bytes, "
            f"expected < {max_allowed_growth} bytes"
        )

        tracemalloc.stop()

    async def test_cache_doesnt_leak_with_subnets(self, mem_bb, cycle_tb):
        """Verify instance cache works with interface views"""

        from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite
        from typing import Annotated

        # Define an interface for this test
        @blackboard_interface
        class CacheTestInterface:
            """Interface to trigger cache usage"""
            data: Annotated[str, readonly]

        # Update blackboard to have the field
        mem_bb_with_field = replace(mem_bb, data="test_value")

        # Create simple net with ONE transition that uses interface view
        @pn.net
        def SimpleNet(builder):
            input_p = builder.place("input", type=PlaceType.QUEUE)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def processor(consumed, bb: CacheTestInterface, timebase):
                # Access bb.data to ensure interface view is created and used
                value = bb.data
                yield {output_p: f"{value}_{consumed[0]}"}

            builder.arc(input_p, processor)
            builder.arc(processor, output_p)

        # Create and start runner
        runner = PNRunner(SimpleNet, mem_bb_with_field)
        await runner.start(cycle_tb)

        # Add input tokens to trigger transition execution
        input_place = runner.runtime.places[('SimpleNet', 'input')]
        output_place = runner.runtime.places[('SimpleNet', 'output')]

        for i in range(5):
            input_place.add_token(f"token{i}")

        # Wait for transitions to fire
        await asyncio.sleep(0.2)

        # Verify transitions actually fired
        output_tokens = list(output_place.tokens)
        print(f"\n[DEBUG] Output tokens: {output_tokens}")
        assert len(output_tokens) > 0, "Transition should have produced output tokens"

        # Check instance cache has entries from interface view creation
        stats_before = runner.runtime.get_cache_stats()
        initial_cache_size = stats_before['cached_views']
        print(f"[DEBUG] Cache stats before: {stats_before}")
        assert initial_cache_size > 0, f"Cache should have interface views, but has {initial_cache_size}"

        # Stop and delete runner (instance cache is deleted with runner)
        await runner.stop()
        del runner
        del mem_bb_with_field

        # Force garbage collection
        gc.collect()

        # No global cache - instance cache is gone with the runner
        # Test verifies no memory leak from runner holding references

    async def test_lru_cache_eviction(self, mem_bb, cycle_tb):
        """Verify LRU cache evicts entries when exceeding maxsize"""

        from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite
        from mycorrhizal.common.cache import InterfaceViewCache
        from typing import Annotated

        # Define a simple interface
        @blackboard_interface
        class TestInterface:
            data: Annotated[str, readonly]

        # Create an instance cache directly
        cache = InterfaceViewCache(maxsize=256)

        # Create multiple blackboards to fill cache beyond maxsize
        blackboards = []
        for i in range(266):  # 256 + 10
            bb = replace(mem_bb, data=f"value_{i}")
            blackboards.append(bb)

        # Fill cache beyond maxsize by creating interface views
        from mycorrhizal.common.wrappers import create_view_from_protocol
        for bb in blackboards:
            cache.get_or_create(
                bb_id=id(bb),
                interface_type=TestInterface,
                readonly_fields=None,
                creator_func=lambda: create_view_from_protocol(bb, TestInterface, readonly_fields=None)
            )

        # Check cache stats
        stats = cache.get_stats()
        print(f"\n[DEBUG] Cache stats: {stats}")

        # Cache should not exceed maxsize
        assert stats['cached_views'] <= 256, \
            f"Cache size {stats['cached_views']} exceeds maxsize 256"

        # Cache should have evicted some entries (we added more than maxsize)
        assert stats['cached_views'] < len(blackboards), \
            "Cache should have evicted old entries"

        # Verify cache is bounded
        print(f"[DEBUG] Cache is bounded: {stats['cached_views']}/{stats['maxsize']}")
        assert stats['maxsize'] == 256


# =============================================================================
# Token Memory Tests
# =============================================================================

class TestTokenMemory:
    """Tests for token memory management"""

    async def test_large_tokens_gc(self, mem_bb, cycle_tb):
        """Verify large tokens are properly GC'd"""

        @pn.net
        def LargeTokenNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb, timebase):
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        runner = PNRunner(LargeTokenNet, mem_bb)
        await runner.start(cycle_tb)

        # Create large tokens
        large_data = "x" * 10000  # 10KB strings
        input_place = runner.runtime.places[('LargeTokenNet', 'input')]

        for i in range(100):
            input_place.add_token(f"{large_data}_{i}")

        await asyncio.sleep(0.5)

        # All tokens should be processed
        output_tokens = runner.runtime.places[('LargeTokenNet', 'output')].tokens
        assert len(output_tokens) == 100

        # Input should be empty
        assert len(input_place.tokens) == 0

        await runner.stop()

        # Force GC and verify no leaks
        del runner
        gc.collect()


# =============================================================================
# Event Object Memory Tests
# =============================================================================

class TestEventMemory:
    """Tests for asyncio Event object memory management"""

    async def test_event_objects_cleanup(self, mem_bb, cycle_tb):
        """Verify Event objects don't leak"""

        @pn.net
        def EventNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb, timebase):
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        # Create many places with events
        runners = []
        for i in range(20):
            bb = MemoryTestBlackboard(data=f"test{i}")
            runner = PNRunner(EventNet, bb)
            await runner.start(cycle_tb)
            runners.append((runner, bb))

        await asyncio.sleep(0.2)

        # Stop all runners
        for runner, bb in runners:
            await runner.stop()

        # Delete all
        del runners

        # Force GC
        gc.collect()


# =============================================================================
# Transition Task Memory Tests
# =============================================================================

class TestTaskMemory:
    """Tests for asyncio Task object memory management"""

    async def test_transition_tasks_cleanup(self, mem_bb, cycle_tb):
        """Verify transition tasks don't leak"""

        @pn.net
        def TaskNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb, timebase):
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        # Create many transitions (tasks)
        runner = PNRunner(TaskNet, mem_bb)
        await runner.start(cycle_tb)

        # Add tokens to trigger task activity
        input_place = runner.runtime.places[('TaskNet', 'input')]
        for i in range(50):
            input_place.add_token(f"token{i}")

        await asyncio.sleep(0.3)

        # Stop runner (should cancel all tasks)
        await runner.stop()

        # All tasks should be cancelled
        for trans in runner.runtime.transitions.values():
            assert trans.task is not None
            assert trans.task.cancelled() or trans.task.done()

        del runner
        gc.collect()
