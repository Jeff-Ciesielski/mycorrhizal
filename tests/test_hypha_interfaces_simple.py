#!/usr/bin/env python3
"""
Simple Hypha interface integration test demonstrating that:
1. Interface-based transitions work
2. Readonly fields are enforced
3. Backward compatibility is maintained
"""

import sys
sys.path.insert(0, "src")

import pytest
import asyncio
from typing import Annotated
from pydantic import BaseModel

from mycorrhizal.hypha.core import pn, PlaceType, Runner as PNRunner, NetBuilder
from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite
from mycorrhizal.common.wrappers import AccessControlError


class TestBlackboard(BaseModel):
    """Test blackboard"""
    max_tasks: int = 10
    tasks_completed: int = 0


@blackboard_interface
class TaskInterface:
    """Interface for task processing"""
    max_tasks: Annotated[int, readonly]
    tasks_completed: Annotated[int, readwrite]


def test_hypha_interface_integration():
    """Test basic interface integration with Hypha transitions"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def process_with_interface(consumed, bb: TaskInterface, timebase):
            # Verify we can read readonly field
            assert bb.max_tasks == 10

            # Verify we can write readwrite field
            bb.tasks_completed += len(consumed)

            # Try to modify readonly field - should fail
            try:
                bb.max_tasks = 999
                raise AssertionError("Should not be able to modify readonly field")
            except (AccessControlError, AttributeError):
                pass  # Expected

            yield pn.put(output_place, f"processed_{consumed[0]}")

        builder.arc(input_place, process_with_interface)
        builder.arc(process_with_interface, output_place)

    bb = TestBlackboard()
    runner = PNRunner(TestNet, bb)

    async def run():
        from mycorrhizal.common.timebase import MonotonicClock
        tb = MonotonicClock()

        await runner.start(tb)

        # Add a token to input place
        runtime = runner.runtime
        runtime.places[("TestNet", "input")].add_token("task1")

        # Give time for processing
        await asyncio.sleep(0.1)

        await runner.stop()

    asyncio.run(run())

    # Verify the transition ran and enforced constraints
    assert bb.tasks_completed == 1
    assert bb.max_tasks == 10  # Unchanged (readonly protected)


def test_hypha_backward_compatibility():
    """Test that transitions without interfaces still work (backward compatibility)"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def process_no_interface(consumed, bb: TestBlackboard, timebase):
            # Should have full access without interface
            bb.tasks_completed += 1
            bb.max_tasks = 999  # Can modify with full access
            yield pn.put(output_place, "done")

        builder.arc(input_place, process_no_interface)
        builder.arc(process_no_interface, output_place)

    bb = TestBlackboard()
    runner = PNRunner(TestNet, bb)

    async def run():
        from mycorrhizal.common.timebase import MonotonicClock
        tb = MonotonicClock()

        await runner.start(tb)

        runtime = runner.runtime
        runtime.places[("TestNet", "input")].add_token("task1")

        await asyncio.sleep(0.1)

        await runner.stop()

    asyncio.run(run())

    # Full access worked
    assert bb.tasks_completed == 1
    assert bb.max_tasks == 999  # Modified successfully


if __name__ == "__main__":
    print("Testing Hypha interface integration...")
    test_hypha_interface_integration()
    print("✓ Interface integration test passed")

    test_hypha_backward_compatibility()
    print("✓ Backward compatibility test passed")

    print("\nAll tests passed!")
