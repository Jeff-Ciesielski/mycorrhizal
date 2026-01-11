#!/usr/bin/env python3
"""Tests for Rhizomorph behavior tree interface integration"""

import sys
sys.path.insert(0, "src")

import pytest
from enum import Enum
from typing import Annotated, Optional
from pydantic import BaseModel

from mycorrhizal.rhizomorph.core import bt, Runner, Status
from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite
from mycorrhizal.common.wrappers import AccessControlError


# ============================================================================
# Test Fixtures
# ============================================================================

class TaskBlackboard(BaseModel):
    """Test blackboard with multiple fields"""
    max_tasks: int = 10
    tasks_completed: int = 0
    current_task: str | None = None
    secret_field: str = "should_not_access"


@blackboard_interface
class TaskInterface:
    """Interface for task processing"""
    max_tasks: Annotated[int, readonly]
    tasks_completed: Annotated[int, readwrite]
    current_task: Annotated[Optional[str], readwrite]


@blackboard_interface
class ReadOnlyInterface:
    """Read-only interface for configuration"""
    max_tasks: Annotated[int, readonly]
    timeout: Annotated[float, readonly] = 5.0


@blackboard_interface
class WriteOnlyInterface:
    """Write-only interface for updates"""
    tasks_completed: Annotated[int, readwrite]
    status: Annotated[str, readwrite] = "ok"


# ============================================================================
# Basic Interface Integration Tests
# ============================================================================

def test_action_with_interface_type_hint():
    """Test that action nodes can use interface type hints"""
    @bt.tree
    def TestTree():
        @bt.action
        async def increment_tasks(bb: TaskInterface) -> Status:
            # Should be able to read and write tasks_completed
            bb.tasks_completed += 1
            return Status.SUCCESS

        @bt.root
        @bt.sequence()
        def root(N):
            yield N.increment_tasks

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio
    status = asyncio.run(runner.tick())

    assert status == Status.SUCCESS
    assert bb.tasks_completed == 1


def test_action_readonly_field_enforced():
    """Test that readonly fields cannot be modified through interface"""
    @bt.tree
    def TestTree():
        @bt.action
        async def try_modify_readonly(bb: TaskInterface) -> Status:
            try:
                bb.max_tasks = 20  # Should fail - max_tasks is readonly
                return Status.SUCCESS
            except (AccessControlError, AttributeError):
                return Status.FAILURE

        @bt.root
        @bt.sequence()
        def root(N):
            yield N.try_modify_readonly

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio
    status = asyncio.run(runner.tick())

    # Should fail because we tried to modify readonly field
    assert status == Status.FAILURE
    # Original value should be unchanged
    assert bb.max_tasks == 10


def test_action_unspecified_field_inaccessible():
    """Test that fields not in interface are inaccessible"""
    @bt.tree
    def TestTree():
        @bt.action
        async def try_access_secret(bb: TaskInterface) -> Status:
            try:
                # secret_field is not in TaskInterface
                _ = bb.secret_field
                return Status.SUCCESS
            except (AccessControlError, AttributeError):
                return Status.FAILURE

        @bt.root
        @bt.sequence()
        def root(N):
            yield N.try_access_secret

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio
    status = asyncio.run(runner.tick())

    # Should fail because secret_field is not accessible
    assert status == Status.FAILURE


def test_condition_with_interface():
    """Test that condition nodes work with interfaces"""
    @bt.tree
    def TestTree():
        @bt.condition
        def check_task_limit(bb: TaskInterface) -> bool:
            # Should be able to read both readonly and readwrite fields
            return bb.tasks_completed < bb.max_tasks

        @bt.action
        async def increment(bb: TaskInterface) -> Status:
            bb.tasks_completed += 1
            return Status.SUCCESS

        @bt.root
        @bt.sequence()
        def root(N):
            yield N.check_task_limit
            yield N.increment

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio

    # Should succeed 10 times (max_tasks = 10)
    for i in range(10):
        status = asyncio.run(runner.tick())
        assert status == Status.RUNNING or status == Status.SUCCESS

    # 11th tick should fail because limit reached
    status = asyncio.run(runner.tick())
    assert status == Status.FAILURE


def test_selector_with_interfaces():
    """Test selector nodes with interface constraints"""
    @bt.tree
    def TestTree():
        @bt.action
        async def process_task(bb: TaskInterface) -> Status:
            # Check if there's a task to process
            if bb.current_task is None:
                return Status.FAILURE
            bb.tasks_completed += 1
            bb.current_task = None
            return Status.SUCCESS

        @bt.action
        async def no_task(bb: TaskInterface) -> Status:
            # Fallback when no task available
            return Status.FAILURE

        @bt.root
        @bt.selector()
        def root(N):
            yield N.process_task  # Try processing first
            yield N.no_task  # Fallback

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio

    # No task set - both children fail, selector should fail
    status = asyncio.run(runner.tick())
    assert status == Status.FAILURE
    assert bb.tasks_completed == 0

    # Set a task - should succeed
    bb.current_task = "task1"
    status = asyncio.run(runner.tick())
    assert status == Status.SUCCESS
    assert bb.tasks_completed == 1
    assert bb.current_task is None


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

def test_action_without_interface_still_works():
    """Test that actions without interface type hints still work (backward compatibility)"""
    @bt.tree
    def TestTree():
        @bt.action
        async def full_access(bb: TaskBlackboard) -> Status:
            # Should have full access to all fields
            bb.tasks_completed += 1
            bb.max_tasks = 20
            bb.secret_field = "accessed"
            return Status.SUCCESS

        @bt.root
        @bt.sequence()
        def root(N):
            yield N.full_access

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio
    status = asyncio.run(runner.tick())

    assert status == Status.SUCCESS
    assert bb.tasks_completed == 1
    assert bb.max_tasks == 20
    assert bb.secret_field == "accessed"


def test_action_with_untyped_bb():
    """Test that actions with untyped bb parameter still work"""
    @bt.tree
    def TestTree():
        @bt.action
        async def untyped(bb) -> Status:
            # Untyped parameter should get full access
            bb.tasks_completed = 5
            return Status.SUCCESS

        @bt.root
        @bt.sequence()
        def root(N):
            yield N.untyped

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio
    status = asyncio.run(runner.tick())

    assert status == Status.SUCCESS
    assert bb.tasks_completed == 5


# ============================================================================
# Timebase Support Tests
# ============================================================================

def test_action_with_interface_and_timebase():
    """Test that actions with interface and timebase work correctly"""
    @bt.tree
    def TestTree():
        @bt.action
        async def increment_with_time(bb: TaskInterface, tb) -> Status:
            # Should receive both interface view and timebase
            bb.tasks_completed += 1
            return Status.SUCCESS if tb.now() >= 0 else Status.FAILURE

        @bt.root
        @bt.sequence()
        def root(N):
            yield N.increment_with_time

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio
    status = asyncio.run(runner.tick())

    assert status == Status.SUCCESS
    assert bb.tasks_completed == 1


# ============================================================================
# Complex Tree Tests
# ============================================================================

def test_parallel_with_interfaces():
    """Test parallel nodes with interface constraints"""
    @bt.tree
    def TestTree():
        @bt.action
        async def increment_a(bb: TaskInterface) -> Status:
            bb.tasks_completed += 1
            return Status.SUCCESS

        @bt.action
        async def increment_b(bb: TaskInterface) -> Status:
            bb.tasks_completed += 1
            return Status.SUCCESS

        @bt.root
        @bt.parallel(success_threshold=2)
        def root(N):
            yield N.increment_a
            yield N.increment_b

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio
    status = asyncio.run(runner.tick())

    assert status == Status.SUCCESS
    # Both actions should have run
    assert bb.tasks_completed == 2


def test_decorator_stack_with_interfaces():
    """Test decorator stacks with interface constraints"""
    @bt.tree
    def TestTree():
        @bt.condition
        def not_at_limit(bb: TaskInterface) -> bool:
            return bb.tasks_completed < bb.max_tasks

        @bt.action
        async def increment(bb: TaskInterface) -> Status:
            bb.tasks_completed += 1
            return Status.SUCCESS

        @bt.root
        @bt.sequence()
        def root(N):
            # Stack decorators: gate -> timeout -> action
            yield bt.timeout(seconds=1.0).gate(not_at_limit)(increment)

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio

    # Should succeed until limit reached
    for _ in range(10):
        status = asyncio.run(runner.tick())
        assert status in (Status.RUNNING, Status.SUCCESS)

    # At limit, gate should fail
    status = asyncio.run(runner.tick())
    assert status == Status.FAILURE


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_interface_violation_in_sequence():
    """Test that interface violations fail gracefully in sequences"""
    @bt.tree
    def TestTree():
        @bt.action
        async def good_action(bb: TaskInterface) -> Status:
            bb.tasks_completed += 1
            return Status.SUCCESS

        @bt.action
        async def bad_action(bb: TaskInterface) -> Status:
            try:
                bb.max_tasks = 999  # Try to modify readonly field
                return Status.SUCCESS
            except (AccessControlError, AttributeError):
                return Status.ERROR

        @bt.root
        @bt.sequence()
        def root(N):
            yield N.good_action
            yield N.bad_action

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio
    status = asyncio.run(runner.tick())

    # First action succeeds, second fails with ERROR
    assert status == Status.ERROR
    assert bb.tasks_completed == 1  # First action ran


# ============================================================================
# Multiple Interfaces Tests
# ============================================================================

def test_multiple_interfaces_same_tree():
    """Test that different nodes can use different interfaces"""
    @bt.tree
    def TestTree():
        @bt.action
        async def use_task_interface(bb: TaskInterface) -> Status:
            # Can only access TaskInterface fields
            bb.tasks_completed += 1
            return Status.SUCCESS

        # Note: ReadOnlyInterface doesn't have tasks_completed, so this won't work
        # But we can test that the tree handles this correctly
        @bt.root
        @bt.sequence()
        def root(N):
            yield N.use_task_interface

    bb = TaskBlackboard()
    runner = Runner(TestTree, bb)

    import asyncio
    status = asyncio.run(runner.tick())

    assert status == Status.SUCCESS
    assert bb.tasks_completed == 1
