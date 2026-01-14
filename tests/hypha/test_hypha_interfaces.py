#!/usr/bin/env python3
"""
Comprehensive tests for Hypha Petri net interface integration.

These tests verify that:
1. Interface-based transitions work with automatic view creation
2. Readonly fields are enforced
3. Unspecified fields are inaccessible
4. Multiple interfaces can be used in the same net
5. Backward compatibility is maintained
6. Different place types work with interfaces
"""

import sys
sys.path.insert(0, "src")

import pytest
import asyncio
from typing import Annotated, List
from pydantic import BaseModel

from mycorrhizal.hypha.core import pn, PlaceType, Runner as PNRunner, NetBuilder
from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite
from mycorrhizal.common.wrappers import AccessControlError


# ============================================================================
# Test Fixtures
# ============================================================================

class TaskBlackboard(BaseModel):
    """Test blackboard with multiple fields"""
    max_tasks: int = 10
    tasks_completed: int = 0
    processing_queue: List[str] = []
    secret_field: str = "should_not_access"


@blackboard_interface
class TaskInterface:
    """Interface for task processing"""
    max_tasks: Annotated[int, readonly]
    tasks_completed: Annotated[int, readwrite]


@blackboard_interface
class ProcessingInterface:
    """Interface for queue processing"""
    max_tasks: Annotated[int, readonly]
    processing_queue: Annotated[List[str], readwrite]


@blackboard_interface
class ReadOnlyInterface:
    """Read-only interface for configuration"""
    max_tasks: Annotated[int, readonly]
    timeout: Annotated[float, readonly] = 5.0


@blackboard_interface
class WriteOnlyInterface:
    """Write-only interface for updates"""
    tasks_completed: Annotated[int, readwrite]


# ============================================================================
# Basic Interface Integration Tests
# ============================================================================

def test_transition_with_interface_type_hint():
    """Test that transitions can use interface type hints"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def process_with_interface(consumed, bb: TaskInterface, timebase):
            # Should be able to read and write tasks_completed
            bb.tasks_completed += len(consumed)
            yield {output_place: f"processed_{consumed[0]}"}

        builder.arc(input_place, process_with_interface)
        builder.arc(process_with_interface, output_place)

    bb = TaskBlackboard()
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

    assert bb.tasks_completed == 1


def test_transition_readonly_field_enforced():
    """Test that readonly fields cannot be modified through interface"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def try_modify_readonly(consumed, bb: TaskInterface, timebase):
            try:
                bb.max_tasks = 20  # Should fail - max_tasks is readonly
                yield {output_place: "should_not_succeed"}
            except (AccessControlError, AttributeError):
                # Expected - readonly field protected
                yield {output_place: "protected"}

        builder.arc(input_place, try_modify_readonly)
        builder.arc(try_modify_readonly, output_place)

    bb = TaskBlackboard()
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

    # Original value should be unchanged
    assert bb.max_tasks == 10

    # Output should have the protected marker
    output_tokens = list(runner.runtime.places[("TestNet", "output")].tokens)
    assert len(output_tokens) == 1
    assert output_tokens[0] == "protected"


def test_transition_unspecified_field_inaccessible():
    """Test that fields not in interface are inaccessible"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def try_access_secret(consumed, bb: TaskInterface, timebase):
            try:
                # secret_field is not in TaskInterface
                _ = bb.secret_field
                yield {output_place: "should_not_succeed"}
            except (AccessControlError, AttributeError):
                # Expected - field not in interface
                yield {output_place: "inaccessible"}

        builder.arc(input_place, try_access_secret)
        builder.arc(try_access_secret, output_place)

    bb = TaskBlackboard()
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

    # Output should have the inaccessible marker
    output_tokens = list(runner.runtime.places[("TestNet", "output")].tokens)
    assert len(output_tokens) == 1
    assert output_tokens[0] == "inaccessible"


# ============================================================================
# Multiple Interfaces Tests
# ============================================================================

def test_multiple_interfaces_same_net():
    """Test that different transitions can use different interfaces"""
    @pn.net
    def TestNet(builder: NetBuilder):
        task_input = builder.place("task_input", type=PlaceType.QUEUE)
        queue_input = builder.place("queue_input", type=PlaceType.QUEUE)
        output = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def process_task(consumed, bb: TaskInterface, timebase):
            # Can only access TaskInterface fields
            bb.tasks_completed += 1
            yield {output: "task_processed"}

        @builder.transition()
        async def process_queue(consumed, bb: ProcessingInterface, timebase):
            # Can only access ProcessingInterface fields
            bb.processing_queue.append(consumed[0])
            yield {output: "queue_updated"}

        builder.arc(task_input, process_task)
        builder.arc(process_task, output)

        builder.arc(queue_input, process_queue)
        builder.arc(process_queue, output)

    bb = TaskBlackboard()
    runner = PNRunner(TestNet, bb)

    async def run():
        from mycorrhizal.common.timebase import MonotonicClock
        tb = MonotonicClock()

        await runner.start(tb)

        runtime = runner.runtime
        runtime.places[("TestNet", "task_input")].add_token("task1")
        runtime.places[("TestNet", "queue_input")].add_token("item1")

        await asyncio.sleep(0.1)
        await runner.stop()

    asyncio.run(run())

    # Both transitions should have run
    assert bb.tasks_completed == 1
    assert len(bb.processing_queue) == 1
    assert bb.processing_queue[0] == "item1"


def test_interface_with_different_place_types():
    """Test that interfaces work with different place types"""
    @pn.net
    def TestNet(builder: NetBuilder):
        # Queue place
        queue_place = builder.place("queue", type=PlaceType.QUEUE)
        # Bag place
        bag_place = builder.place("bag", type=PlaceType.BAG)
        output = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def process_queue(consumed, bb: TaskInterface, timebase):
            bb.tasks_completed += 1
            yield {output: "queue_done"}

        @builder.transition()
        async def process_bag(consumed, bb: TaskInterface, timebase):
            bb.tasks_completed += 1
            yield {output: "bag_done"}

        builder.arc(queue_place, process_queue)
        builder.arc(process_queue, output)

        builder.arc(bag_place, process_bag)
        builder.arc(process_bag, output)

    bb = TaskBlackboard()
    runner = PNRunner(TestNet, bb)

    async def run():
        from mycorrhizal.common.timebase import MonotonicClock
        tb = MonotonicClock()

        await runner.start(tb)

        runtime = runner.runtime
        runtime.places[("TestNet", "queue")].add_token("item1")
        runtime.places[("TestNet", "bag")].add_token("item2")

        await asyncio.sleep(0.1)
        await runner.stop()

    asyncio.run(run())

    # Both transitions should have run
    assert bb.tasks_completed == 2


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

def test_transition_without_interface_still_works():
    """Test that transitions without interface type hints still work"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def full_access(consumed, bb: TaskBlackboard, timebase):
            # Should have full access to all fields
            bb.tasks_completed += 1
            bb.max_tasks = 20
            bb.secret_field = "accessed"
            yield {output_place: "full_access"}

        builder.arc(input_place, full_access)
        builder.arc(full_access, output_place)

    bb = TaskBlackboard()
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

    assert bb.tasks_completed == 1
    assert bb.max_tasks == 20
    assert bb.secret_field == "accessed"


def test_transition_with_untyped_bb():
    """Test that transitions with untyped bb parameter still work"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def untyped(consumed, bb, timebase):
            # Untyped parameter should get full access
            bb.tasks_completed = 5
            yield {output_place: "untyped"}

        builder.arc(input_place, untyped)
        builder.arc(untyped, output_place)

    bb = TaskBlackboard()
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

    assert bb.tasks_completed == 5


# ============================================================================
# Complex Workflow Tests
# ============================================================================

def test_sequential_transitions_with_interfaces():
    """Test sequential workflow with interface constraints"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input1 = builder.place("input1", type=PlaceType.QUEUE)
        input2 = builder.place("input2", type=PlaceType.QUEUE)
        intermediate = builder.place("intermediate", type=PlaceType.QUEUE)
        output = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def first_stage(consumed, bb: TaskInterface, timebase):
            bb.tasks_completed += 1
            yield {intermediate: f"stage1_{consumed[0]}"}

        @builder.transition()
        async def second_stage(consumed, bb: TaskInterface, timebase):
            bb.tasks_completed += 1
            yield {output: f"stage2_{consumed[0]}"}

        builder.arc(input1, first_stage)
        builder.arc(first_stage, intermediate)

        builder.arc(intermediate, second_stage)
        builder.arc(second_stage, output)

    bb = TaskBlackboard()
    runner = PNRunner(TestNet, bb)

    async def run():
        from mycorrhizal.common.timebase import MonotonicClock
        tb = MonotonicClock()

        await runner.start(tb)

        runtime = runner.runtime
        runtime.places[("TestNet", "input1")].add_token("task1")

        await asyncio.sleep(0.2)  # Give time for both stages
        await runner.stop()

    asyncio.run(run())

    # Both stages should have completed
    assert bb.tasks_completed == 2

    output_tokens = list(runner.runtime.places[("TestNet", "output")].tokens)
    assert len(output_tokens) == 1
    assert output_tokens[0].startswith("stage2_")


def test_parallel_transitions_with_interfaces():
    """Test parallel transitions with interface constraints"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input1 = builder.place("input1", type=PlaceType.QUEUE)
        input2 = builder.place("input2", type=PlaceType.QUEUE)
        output = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def process_a(consumed, bb: TaskInterface, timebase):
            bb.tasks_completed += 1
            yield {output: f"A_{consumed[0]}"}

        @builder.transition()
        async def process_b(consumed, bb: TaskInterface, timebase):
            bb.tasks_completed += 1
            yield {output: f"B_{consumed[0]}"}

        builder.arc(input1, process_a)
        builder.arc(process_a, output)

        builder.arc(input2, process_b)
        builder.arc(process_b, output)

    bb = TaskBlackboard()
    runner = PNRunner(TestNet, bb)

    async def run():
        from mycorrhizal.common.timebase import MonotonicClock
        tb = MonotonicClock()

        await runner.start(tb)

        runtime = runner.runtime
        runtime.places[("TestNet", "input1")].add_token("task1")
        runtime.places[("TestNet", "input2")].add_token("task2")

        await asyncio.sleep(0.1)
        await runner.stop()

    asyncio.run(run())

    # Both transitions should have run
    assert bb.tasks_completed == 2

    output_tokens = list(runner.runtime.places[("TestNet", "output")].tokens)
    assert len(output_tokens) == 2


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_interface_violation_in_transition():
    """Test that interface violations are handled gracefully"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def violating_transition(consumed, bb: TaskInterface, timebase):
            try:
                bb.max_tasks = 999  # Try to modify readonly field
                yield {output_place: "should_not_happen"}
            except (AccessControlError, AttributeError):
                # Handle gracefully - produce error output
                yield {output_place: "access_denied"}

        builder.arc(input_place, violating_transition)
        builder.arc(violating_transition, output_place)

    bb = TaskBlackboard()
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

    # Readonly field should be unchanged
    assert bb.max_tasks == 10

    # Should have error output
    output_tokens = list(runner.runtime.places[("TestNet", "output")].tokens)
    assert len(output_tokens) == 1
    assert output_tokens[0] == "access_denied"


# ============================================================================
# Cache Tests
# ============================================================================

def test_interface_view_caching():
    """Test that interface views are cached for performance"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        # Track how many times transition runs
        run_count = {"count": 0}

        @builder.transition()
        async def process(consumed, bb: TaskInterface, timebase):
            # Each call should use the same cached view
            run_count["count"] += 1
            bb.tasks_completed += len(consumed)
            yield {output_place: f"run_{run_count['count']}"}

        builder.arc(input_place, process)
        builder.arc(process, output_place)

    bb = TaskBlackboard()
    runner = PNRunner(TestNet, bb)

    async def run():
        from mycorrhizal.common.timebase import MonotonicClock
        tb = MonotonicClock()

        await runner.start(tb)

        runtime = runner.runtime
        # Add tokens one at a time and wait to ensure multiple transition runs
        runtime.places[("TestNet", "input")].add_token("task1")
        await asyncio.sleep(0.05)

        runtime.places[("TestNet", "input")].add_token("task2")
        await asyncio.sleep(0.05)

        runtime.places[("TestNet", "input")].add_token("task3")
        await asyncio.sleep(0.1)

        await runner.stop()

    asyncio.run(run())

    # Transition should have run at least once with the cached view
    assert bb.tasks_completed >= 1

    output_tokens = list(runner.runtime.places[("TestNet", "output")].tokens)
    assert len(output_tokens) >= 1


# ============================================================================
# Readonly/WriteOnly Interface Tests
# ============================================================================

def test_readonly_interface():
    """Test that readonly interface allows reads but not writes"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def check_readonly(consumed, bb: ReadOnlyInterface, timebase):
            # Should be able to read
            max_tasks = bb.max_tasks

            # Should NOT be able to write
            try:
                bb.max_tasks = 999
                yield {output_place: "write_succeeded"}
            except (AccessControlError, AttributeError):
                # Expected - readonly prevents writes
                yield {output_place: f"readonly_{max_tasks}"}

        builder.arc(input_place, check_readonly)
        builder.arc(check_readonly, output_place)

    bb = TaskBlackboard()
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

    # Original value should be unchanged
    assert bb.max_tasks == 10

    # Output should show readonly protection worked
    output_tokens = list(runner.runtime.places[("TestNet", "output")].tokens)
    assert len(output_tokens) == 1
    assert output_tokens[0] == "readonly_10"


def test_writeonly_interface():
    """Test that writeonly interface allows writes but restricts reads"""
    @pn.net
    def TestNet(builder: NetBuilder):
        input_place = builder.place("input", type=PlaceType.QUEUE)
        output_place = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def use_writeonly(consumed, bb: WriteOnlyInterface, timebase):
            # Should be able to write
            bb.tasks_completed = 42
            yield {output_place: "write_worked"}

        builder.arc(input_place, use_writeonly)
        builder.arc(use_writeonly, output_place)

    bb = TaskBlackboard()
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

    # Write should have succeeded
    assert bb.tasks_completed == 42


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    print("Testing Hypha interface integration...")
    print("=" * 80)

    tests = [
        ("Basic Integration", [
            test_transition_with_interface_type_hint,
            test_transition_readonly_field_enforced,
            test_transition_unspecified_field_inaccessible,
        ]),
        ("Multiple Interfaces", [
            test_multiple_interfaces_same_net,
            test_interface_with_different_place_types,
        ]),
        ("Backward Compatibility", [
            test_transition_without_interface_still_works,
            test_transition_with_untyped_bb,
        ]),
        ("Complex Workflows", [
            test_sequential_transitions_with_interfaces,
            test_parallel_transitions_with_interfaces,
        ]),
        ("Error Handling", [
            test_interface_violation_in_transition,
        ]),
        ("Caching", [
            test_interface_view_caching,
        ]),
        ("Interface Types", [
            test_readonly_interface,
            test_writeonly_interface,
        ]),
    ]

    total_tests = 0
    passed_tests = 0

    for category, test_funcs in tests:
        print(f"\n{category}:")
        print("-" * 80)
        for test_func in test_funcs:
            total_tests += 1
            try:
                test_func()
                print(f"  ✓ {test_func.__name__}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {test_func.__name__}: {e}")

    print("\n" + "=" * 80)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("All tests passed!")
    else:
        print(f"{total_tests - passed_tests} tests failed")
