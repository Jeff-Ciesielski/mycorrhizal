#!/usr/bin/env python3
"""Integration tests for Mycelium composition patterns with Septum FSMs."""

import sys
sys.path.insert(0, "src")

import asyncio
import pytest
from enum import Enum, auto
from pydantic import BaseModel

from mycorrhizal.septum.core import (
    septum,
    StateMachine,
    LabeledTransition,
    StateConfiguration,
    SharedContext,
    StateMachineComplete,
)


# ============================================================================
# FSM-in-BT Integration Test
# ============================================================================


@septum.state(config=StateConfiguration(can_dwell=True))
def FSMIdleState():
    """Idle state for FSM-in-BT test."""

    @septum.events
    class Events(Enum):
        START = auto()
        STOP = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        should_start = ctx.common.get("should_start", False)
        if should_start:
            return Events.START
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.START, FSMActiveState),
        ]


@septum.state()
def FSMActiveState():
    """Active state for FSM-in-BT test."""

    @septum.events
    class Events(Enum):
        CONTINUE = auto()
        DONE = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        count = ctx.common.get("count", 0)
        if count < 3:
            ctx.common["count"] = count + 1
            return Events.CONTINUE
        return Events.DONE

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.CONTINUE, FSMActiveState),
            LabeledTransition(Events.DONE, FSMIdleState),
        ]


@pytest.mark.asyncio
async def test_fsm_state_basic():
    """Test basic FSM state transitions."""
    fsm = StateMachine(initial_state=FSMIdleState)
    await fsm.initialize()

    # Initially in idle state (state name includes module prefix)
    assert "FSMIdleState" in fsm.current_state.name

    # Start the FSM
    fsm.context.common["should_start"] = True
    await fsm.tick()

    # Should transition to active
    assert "FSMActiveState" in fsm.current_state.name

    # Run a few ticks
    for _ in range(5):
        await fsm.tick()

    # Should have counted
    assert fsm.context.common.get("count", 0) > 0


@pytest.mark.asyncio
async def test_multiple_fsm_instances():
    """Test multiple FSM instances running concurrently."""
    async def run_single_fsm(fsm_id: int):
        """Run FSM to completion."""
        fsm = StateMachine(initial_state=FSMIdleState)
        await fsm.initialize()

        # Start the FSM
        fsm.context.common["should_start"] = True
        fsm.context.common["fsm_id"] = fsm_id

        # Run for a few ticks
        for _ in range(10):
            await fsm.tick()

        return fsm_id

    # Create 50 FSMs
    fsms = [run_single_fsm(i) for i in range(50)]

    # Run concurrently
    results = await asyncio.gather(*fsms)

    # Verify all completed
    assert len(results) == 50
    assert set(results) == set(range(50))


# ============================================================================
# BT-in-FSM Integration Test (simplified without full Mycelium)
# ============================================================================


@septum.state()
def ErrorState():
    """Error state that simulates BT decision making."""

    @septum.events
    class Events(Enum):
        RETRY = auto()
        FAIL = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Simulate BT logic: check error conditions and decide
        has_error = ctx.common.get("has_error", False)
        can_recover = ctx.common.get("can_recover", False)

        if not has_error:
            return Events.FAIL

        if can_recover:
            # Simulated BT recovery success
            ctx.common["recovery_attempted"] = True
            return Events.RETRY

        return Events.FAIL

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.RETRY, RetryState),
            LabeledTransition(Events.FAIL, FailState),
        ]


@septum.state()
def RetryState():
    """Retry state."""

    @septum.events
    class Events(Enum):
        DONE = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        ctx.common["retry_count"] = ctx.common.get("retry_count", 0) + 1
        return Events.DONE

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.DONE, FSMIdleState),
        ]


@septum.state()
def FailState():
    """Failure state."""

    @septum.events
    class Events(Enum):
        DONE = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        ctx.common["failed"] = True
        return Events.DONE

    @septum.transitions
    def transitions():
        return []


@pytest.mark.asyncio
async def test_bt_decision_simulation():
    """Test FSM with simulated BT decision making."""
    fsm = StateMachine(initial_state=ErrorState)
    await fsm.initialize()

    # Test 1: No error
    fsm.context.common["has_error"] = False
    await fsm.tick()
    assert "FailState" in fsm.current_state.name

    # Test 2: Error but can recover
    fsm2 = StateMachine(initial_state=ErrorState)
    await fsm2.initialize()
    fsm2.context.common["has_error"] = True
    fsm2.context.common["can_recover"] = True
    await fsm2.tick()
    assert "RetryState" in fsm2.current_state.name
    assert fsm2.context.common.get("recovery_attempted") == True

    # Test 3: Error cannot recover
    fsm3 = StateMachine(initial_state=ErrorState)
    await fsm3.initialize()
    fsm3.context.common["has_error"] = True
    fsm3.context.common["can_recover"] = False
    await fsm3.tick()
    assert "FailState" in fsm3.current_state.name


# ============================================================================
# FSM Lifecycle Test
# ============================================================================


@septum.state()
def LifecycleInitState():
    """Initial state for lifecycle testing."""

    @septum.events
    class Events(Enum):
        NEXT = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        ctx.common["initialized"] = True
        return Events.NEXT

    @septum.on_enter
    async def on_enter(ctx: SharedContext):
        ctx.common["enter_count"] = ctx.common.get("enter_count", 0) + 1

    @septum.on_leave
    async def on_leave(ctx: SharedContext):
        ctx.common["leave_count"] = ctx.common.get("leave_count", 0) + 1

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.NEXT, LifecycleActiveState),
        ]


@septum.state(config=StateConfiguration(terminal=True))
def LifecycleActiveState():
    """Active state for lifecycle testing."""

    @septum.on_state
    async def on_state(ctx: SharedContext):
        ctx.common["active"] = True
        # Return None to stay in this terminal state

    @septum.transitions
    def transitions():
        return []


@pytest.mark.asyncio
async def test_fsm_lifecycle():
    """Test FSM lifecycle management."""
    fsm = StateMachine(initial_state=LifecycleInitState)
    await fsm.initialize()

    # Verify initialized
    assert fsm._initialized == True
    assert fsm.current_state is not None

    # Run lifecycle - first tick transitions to LifecycleActiveState
    # Note: LifecycleActiveState is terminal, so on_state isn't called there
    try:
        await fsm.tick()
    except StateMachineComplete:
        # FSM reached terminal state
        pass

    # Check lifecycle hooks
    assert fsm.context.common.get("initialized") == True
    assert fsm.context.common.get("enter_count") == 1
    assert fsm.context.common.get("leave_count") == 1
    # Verify we're in the terminal state
    assert "LifecycleActiveState" in fsm.current_state.name


# ============================================================================
# Data Flow Test
# ============================================================================


@septum.state()
def DataProducerState():
    """State that produces data."""

    @septum.events
    class Events(Enum):
        PRODUCE = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Produce data
        ctx.common["data"] = ctx.common.get("data", 0) + 1
        return Events.PRODUCE

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.PRODUCE, DataConsumerState),
        ]


@septum.state()
def DataConsumerState():
    """State that consumes data."""

    @septum.events
    class Events(Enum):
        CONSUME = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Consume data
        data = ctx.common.get("data", 0)
        ctx.common["consumed"] = ctx.common.get("consumed", 0) + data
        return Events.CONSUME

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.CONSUME, DataProducerState),
        ]


@pytest.mark.asyncio
async def test_fsm_data_flow():
    """Test data flows correctly between composed systems."""
    fsm = StateMachine(initial_state=DataProducerState)
    await fsm.initialize()

    # Run for a few cycles
    for _ in range(5):
        await fsm.tick()

    # Verify data flow
    assert fsm.context.common.get("data", 0) > 0
    assert fsm.context.common.get("consumed", 0) > 0
    assert fsm.context.common["consumed"] >= fsm.context.common["data"]


# ============================================================================
# Stress Test: Many Concurrent FSMs
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_fsm_stress():
    """Stress test with many concurrent FSM instances."""
    async def run_fsm(fsm_id: int):
        """Run FSM through lifecycle."""
        fsm = StateMachine(initial_state=LifecycleInitState)
        await fsm.initialize()

        # Run a few ticks - catch StateMachineComplete
        for _ in range(10):
            try:
                await fsm.tick()
            except StateMachineComplete:
                # FSM reached terminal state, stop ticking
                break

        return fsm_id

    # Create 100 concurrent FSMs
    tasks = [run_fsm(i) for i in range(100)]

    # Run all concurrently
    results = await asyncio.gather(*tasks)

    # Verify all completed
    assert len(results) == 100
    assert all(r == i for i, r in enumerate(results))
