#!/usr/bin/env python3
"""
Test trace mode for Rhizomorph behavior tree system.

Tests that the trace logger correctly logs action and condition execution.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import pytest

from mycorrhizal.rhizomorph.core import (
    bt,
    Status,
    Runner,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class TestBlackboard:
    """Blackboard for trace tests."""
    value: int = 0
    should_pass: bool = True
    log: List[str] = field(default_factory=list)


class TraceCapture(logging.Handler):
    """Custom logging handler to capture log records."""

    def __init__(self):
        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def trace_logger():
    """Create a logger with a capture handler for testing."""
    logger = logging.getLogger("bt.trace.test")
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    logger.handlers.clear()

    # Add capture handler
    capture = TraceCapture()
    logger.addHandler(capture)

    yield logger, capture

    # Cleanup
    logger.removeHandler(capture)


@pytest.fixture
def test_bb():
    return TestBlackboard()


# =============================================================================
# Test Tree Definitions
# =============================================================================


@bt.tree
def SimpleTraceTree():
    """Simple tree with action and condition for trace testing."""

    @bt.condition
    def check_value(bb: TestBlackboard) -> bool:
        return bb.should_pass

    @bt.action
    def increment_value(bb: TestBlackboard) -> Status:
        bb.value += 1
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield check_value
        yield increment_value


@bt.tree
def FailingTraceTree():
    """Tree with failing condition for trace testing."""

    @bt.condition
    def always_fail(bb: TestBlackboard) -> bool:
        return False

    @bt.action
    def never_reached(bb: TestBlackboard) -> Status:
        bb.value = 999
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield always_fail
        yield never_reached


@bt.tree
def MultiActionTree():
    """Tree with multiple actions to test trace ordering."""

    @bt.action
    def action_one(bb: TestBlackboard) -> Status:
        bb.value = 1
        return Status.SUCCESS

    @bt.action
    def action_two(bb: TestBlackboard) -> Status:
        bb.value = 2
        return Status.SUCCESS

    @bt.action
    def action_three(bb: TestBlackboard) -> Status:
        bb.value = 3
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield action_one
        yield action_two
        yield action_three


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.asyncio
async def test_trace_mode_disabled(trace_logger, test_bb):
    """Test that no trace logs are emitted when trace is None."""
    logger, capture = trace_logger

    runner = Runner(SimpleTraceTree, bb=test_bb, trace=None)
    result = await runner.tick_until_complete()

    assert result == Status.SUCCESS
    assert test_bb.value == 1
    assert len(capture.records) == 0


@pytest.mark.asyncio
async def test_trace_action_success(trace_logger, test_bb):
    """Test that trace logs capture action execution with SUCCESS status."""
    logger, capture = trace_logger

    runner = Runner(SimpleTraceTree, bb=test_bb, trace=logger)
    result = await runner.tick_until_complete()

    assert result == Status.SUCCESS
    assert test_bb.value == 1

    # Check trace logs
    messages = [record.getMessage() for record in capture.records]
    assert len(messages) == 2

    # First should be condition check
    assert any("condition:" in msg and "check_value" in msg for msg in messages)
    assert any("SUCCESS" in msg for msg in messages)

    # Second should be action
    assert any("action:" in msg and "increment_value" in msg for msg in messages)


@pytest.mark.asyncio
async def test_trace_condition_failure(trace_logger, test_bb):
    """Test that trace logs capture condition FAILURE."""
    logger, capture = trace_logger
    test_bb.should_pass = False

    runner = Runner(FailingTraceTree, bb=test_bb, trace=logger)
    result = await runner.tick_until_complete()

    assert result == Status.FAILURE
    assert test_bb.value == 0  # Action should not execute

    # Check trace logs - should only have condition
    messages = [record.getMessage() for record in capture.records]
    assert len(messages) == 1
    assert "condition:" in messages[0]
    assert "always_fail" in messages[0]
    assert "FAILURE" in messages[0]


@pytest.mark.asyncio
async def test_trace_multiple_actions(trace_logger, test_bb):
    """Test that trace logs capture multiple actions in order."""
    logger, capture = trace_logger

    runner = Runner(MultiActionTree, bb=test_bb, trace=logger)
    result = await runner.tick_until_complete()

    assert result == Status.SUCCESS
    # Final value after all actions is 3
    assert test_bb.value == 3

    # Check trace logs
    messages = [record.getMessage() for record in capture.records]
    assert len(messages) == 3

    # Check order and content
    assert "action:" in messages[0]
    assert "action_one" in messages[0]
    assert "SUCCESS" in messages[0]

    assert "action:" in messages[1]
    assert "action_two" in messages[1]
    assert "SUCCESS" in messages[1]

    assert "action:" in messages[2]
    assert "action_three" in messages[2]
    assert "SUCCESS" in messages[2]


@pytest.mark.asyncio
async def test_trace_fully_qualified_name(trace_logger, test_bb):
    """Test that trace logs use fully qualified function names."""
    logger, capture = trace_logger

    runner = Runner(SimpleTraceTree, bb=test_bb, trace=logger)
    result = await runner.tick_until_complete()

    assert result == Status.SUCCESS

    messages = [record.getMessage() for record in capture.records]

    # Should contain module name
    assert any("test_trace_mode" in msg for msg in messages)
    assert any("check_value" in msg for msg in messages)
    assert any("increment_value" in msg for msg in messages)

    # Should contain fully qualified name with dots
    assert any("tests.rhizomorph" in msg or "test_trace_mode" in msg for msg in messages)


@pytest.mark.asyncio
async def test_trace_format(trace_logger, test_bb):
    """Test that trace logs use the correct format: 'type: name | STATUS'."""
    logger, capture = trace_logger

    runner = Runner(SimpleTraceTree, bb=test_bb, trace=logger)
    result = await runner.tick_until_complete()

    assert result == Status.SUCCESS

    messages = [record.getMessage() for record in capture.records]

    # Check format: should contain " | " separator
    for msg in messages:
        assert " | " in msg
        parts = msg.split(" | ")
        assert len(parts) == 2
        # First part should be "action: name" or "condition: name"
        assert ":" in parts[0]
        # Second part should be status name
        assert parts[1] in ["SUCCESS", "FAILURE", "RUNNING", "ERROR", "CANCELLED"]


@pytest.mark.asyncio
async def test_trace_with_status_enum_return(trace_logger, test_bb):
    """Test trace logging when action returns Status enum directly."""
    logger, capture = trace_logger

    @bt.tree
    def StatusEnumTree():
        @bt.action
        def returns_status(bb: TestBlackboard) -> Status:
            bb.value = 42
            return Status.SUCCESS

        @bt.root
        @bt.sequence
        def root():
            yield returns_status

    runner = Runner(StatusEnumTree, bb=test_bb, trace=logger)
    result = await runner.tick_until_complete()

    assert result == Status.SUCCESS
    assert test_bb.value == 42

    messages = [record.getMessage() for record in capture.records]
    assert len(messages) == 1
    assert "action:" in messages[0]
    assert "returns_status" in messages[0]
    assert "SUCCESS" in messages[0]


@pytest.mark.asyncio
async def test_trace_with_bool_return(trace_logger, test_bb):
    """Test trace logging when action returns bool."""
    logger, capture = trace_logger

    @bt.tree
    def BoolReturnTree():
        @bt.action
        def returns_true(bb: TestBlackboard) -> bool:
            bb.value = 100
            return True

        @bt.root
        @bt.sequence
        def root():
            yield returns_true

    runner = Runner(BoolReturnTree, bb=test_bb, trace=logger)
    result = await runner.tick_until_complete()

    assert result == Status.SUCCESS
    assert test_bb.value == 100

    messages = [record.getMessage() for record in capture.records]
    assert len(messages) == 1
    assert "action:" in messages[0]
    assert "returns_true" in messages[0]
    assert "SUCCESS" in messages[0]


@pytest.mark.asyncio
async def test_trace_with_none_return(trace_logger, test_bb):
    """Test trace logging when action returns None (treated as SUCCESS)."""
    logger, capture = trace_logger

    @bt.tree
    def NoneReturnTree():
        @bt.action
        def returns_none(bb: TestBlackboard) -> None:
            bb.value = 200

        @bt.root
        @bt.sequence
        def root():
            yield returns_none

    runner = Runner(NoneReturnTree, bb=test_bb, trace=logger)
    result = await runner.tick_until_complete()

    assert result == Status.SUCCESS
    assert test_bb.value == 200

    messages = [record.getMessage() for record in capture.records]
    assert len(messages) == 1
    assert "action:" in messages[0]
    assert "returns_none" in messages[0]
    assert "SUCCESS" in messages[0]
