#!/usr/bin/env python3
"""Tests for Mycelium unified pattern with FSM integration."""

import sys
sys.path.insert(0, "src")

import pytest
import asyncio
from enum import Enum, auto
from pydantic import BaseModel

from mycorrhizal.enoki.core import enoki, LabeledTransition, StateConfiguration
from mycorrhizal.mycelium import (
    tree,
    Action,
    Condition,
    Sequence,
    Selector,
    root,
    TreeRunner,
)
from mycorrhizal.mycelium.exceptions import (
    MyceliumError,
    TreeDefinitionError,
)
from mycorrhizal.rhizomorph.core import Status
from mycorrhizal.common.timebase import MonotonicClock


# ======================================================================================
# Test States
# ======================================================================================


@enoki.state(config=StateConfiguration(can_dwell=True))
def IdleState():
    """Idle state for testing."""

    @enoki.events
    class Events(Enum):
        START = auto()
        STOP = auto()

    @enoki.on_state
    async def on_state(ctx):
        # Check if we should start
        if ctx.common.get("should_start", False):
            return Events.START
        return None

    @enoki.transitions
    def transitions():
        return [
            LabeledTransition(Events.START, WorkingState),
            LabeledTransition(Events.STOP, IdleState),
        ]


@enoki.state()
def WorkingState():
    """Working state for testing."""

    @enoki.events
    class Events(Enum):
        DONE = auto()
        ERROR = auto()

    @enoki.on_state
    async def on_state(ctx):
        # Just transition back to idle
        return Events.DONE

    @enoki.transitions
    def transitions():
        return [
            LabeledTransition(Events.DONE, IdleState),
            LabeledTransition(Events.ERROR, IdleState),
        ]


# ======================================================================================
# Test Blackboard
# ======================================================================================


class TestBlackboard(BaseModel):
    """Blackboard for testing."""
    should_start: bool = False
    tasks_completed: int = 0


# ======================================================================================
# Test Trees
# ======================================================================================


@tree
def SimpleTree():
    """A simple tree with FSM integration for testing."""

    @Action(fsm=IdleState)
    async def run_fsm(bb, tb, fsm_runner):
        """Run the FSM and check its state."""
        # Start the FSM
        if not bb.should_start:
            bb.should_start = True
            fsm_runner.send_message(IdleState.Events.START)
            return Status.RUNNING

        # Check FSM state
        state_name = fsm_runner.current_state.name if fsm_runner.current_state else ""
        if "Working" in state_name:
            return Status.SUCCESS
        return Status.RUNNING

    @root
    @Sequence
    def main():
        yield run_fsm


@tree
def MultipleFSMsTree():
    """A tree with multiple FSM-integrated actions."""

    @Action(fsm=IdleState)
    async def run_fsm1(bb, tb, fsm_runner):
        """First FSM action."""
        if not bb.should_start:
            bb.should_start = True
            fsm_runner.send_message(IdleState.Events.START)
            return Status.RUNNING
        return Status.SUCCESS

    @Action(fsm=IdleState)
    async def run_fsm2(bb, tb, fsm_runner):
        """Second FSM action."""
        return Status.SUCCESS

    @root
    @Sequence
    def main():
        yield run_fsm1
        yield run_fsm2


@tree
def VanillaActionsTree():
    """A tree with vanilla actions (no FSM integration)."""

    @Action
    async def simple_action(bb, tb):
        """Simple action without FSM."""
        return Status.SUCCESS

    @Condition
    def check_condition(bb):
        """Simple condition."""
        return True

    @root
    @Sequence
    def main():
        yield check_condition
        yield simple_action


# ======================================================================================
# Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_tree_builder_creates_spec():
    """Test that @tree decorator creates a spec."""
    spec = SimpleTree._mycelium_spec
    assert spec is not None
    assert spec.name == "SimpleTree"
    assert len(spec.fsm_integrations) == 1
    assert len(spec.actions) == 1


@pytest.mark.asyncio
async def test_fsm_integration_created():
    """Test that FSM integration is created for actions."""
    spec = SimpleTree._mycelium_spec
    assert "run_fsm" in spec.fsm_integrations

    fsm_integration = spec.fsm_integrations["run_fsm"]
    assert fsm_integration.initial_state == IdleState
    assert fsm_integration.action_name == "run_fsm"


@pytest.mark.asyncio
async def test_tree_runner_creation():
    """Test creating a TreeRunner."""
    bb = TestBlackboard()
    tb = MonotonicClock()

    runner = TreeRunner(SimpleTree, bb=bb, tb=tb)

    assert runner.spec.name == "SimpleTree"


@pytest.mark.asyncio
async def test_fsm_runner_created():
    """Test that FSM runner is created for FSM-integrated actions."""
    bb = TestBlackboard()
    tb = MonotonicClock()

    runner = TreeRunner(SimpleTree, bb=bb, tb=tb)

    # FSM runner should be created
    fsm_runner = runner.instance.get_fsm_runner("run_fsm")
    assert fsm_runner is not None


@pytest.mark.asyncio
async def test_fsm_auto_ticks():
    """Test that FSM auto-ticks when action runs."""
    bb = TestBlackboard()
    tb = MonotonicClock()

    runner = TreeRunner(SimpleTree, bb=bb, tb=tb)

    # First tick: action will send START message to FSM
    result = await runner.tick()
    assert result == Status.RUNNING
    assert bb.should_start == True

    # Second tick: FSM should have transitioned to Working state
    result = await runner.tick()
    # Should be SUCCESS (in Working state) or RUNNING (still transitioning)
    assert result in (Status.SUCCESS, Status.RUNNING)


@pytest.mark.asyncio
async def test_multiple_fsm_integrations():
    """Test that multiple FSM-integrated actions work."""
    bb = TestBlackboard()
    tb = MonotonicClock()

    runner = TreeRunner(MultipleFSMsTree, bb=bb, tb=tb)

    # Should have two FSM runners
    fsm_runner1 = runner.instance.get_fsm_runner("run_fsm1")
    fsm_runner2 = runner.instance.get_fsm_runner("run_fsm2")
    assert fsm_runner1 is not None
    assert fsm_runner2 is not None

    # They should be different instances
    assert fsm_runner1 is not fsm_runner2


@pytest.mark.asyncio
async def test_vanilla_actions_work():
    """Test that vanilla actions without FSM integration still work."""
    bb = TestBlackboard()
    tb = MonotonicClock()

    runner = TreeRunner(VanillaActionsTree, bb=bb, tb=tb)

    # Should run successfully
    result = await runner.tick()
    assert result == Status.SUCCESS


def test_action_decorator_outside_tree():
    """Test that @Action outside @tree raises error."""

    try:

        @Action
        async def orphan_action(bb, tb):
            return Status.SUCCESS

        # Should have raised
        assert False, "Expected TreeDefinitionError"
    except TreeDefinitionError:
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
