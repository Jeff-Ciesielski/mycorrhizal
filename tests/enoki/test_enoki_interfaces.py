#!/usr/bin/env python3
"""
Simple tests for Enoki state machine interface integration
"""

import sys
sys.path.insert(0, "src")

import pytest
import asyncio
from enum import Enum, auto
from typing import Annotated
from pydantic import BaseModel

from mycorrhizal.enoki.core import (
    enoki,
    StateMachine,
    StateConfiguration,
    SharedContext,
    LabeledTransition,
    Again,
    StateMachineComplete,
)
from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite
from mycorrhizal.common.wrappers import AccessControlError


# ============================================================================
# Test Fixtures
# ============================================================================

class TestBlackboard(BaseModel):
    """Test blackboard"""
    max_steps: int = 10
    steps_taken: int = 0
    secret: str = "protected"


@blackboard_interface
class TestInterface:
    """Interface for state machine"""
    max_steps: Annotated[int, readonly]
    steps_taken: Annotated[int, readwrite]


# ============================================================================
# Tests
# ============================================================================

def test_enoki_interface_integration():
    """Test basic interface integration with state machine"""
    @enoki.state()
    def ProcessingState():
        class T(Enum):
            CONTINUE = auto()
            DONE = auto()

        @enoki.on_state
        async def on_state(ctx: SharedContext[TestInterface]):
            # Can read readonly
            assert ctx.common.max_steps == 10

            # Can write readwrite
            ctx.common.steps_taken += 1

            if ctx.common.steps_taken >= 3:
                return T.DONE
            return T.CONTINUE

        @enoki.transitions
        def transitions():
            return [
                LabeledTransition(T.CONTINUE, Again),
                LabeledTransition(T.DONE, TerminalState),
            ]

    @enoki.state(config=StateConfiguration(terminal=True))
    def TerminalState():
        @enoki.on_state
        async def on_state(ctx: SharedContext[TestInterface]):
            pass

        @enoki.transitions
        def transitions():
            return []

    bb = TestBlackboard()
    sm = StateMachine(initial_state=ProcessingState, common_data=bb)

    async def run():
        await sm.initialize()
        try:
            await sm.run()
        except StateMachineComplete:
            pass  # Expected for terminal state

    asyncio.run(run())

    assert bb.steps_taken == 3
    assert bb.max_steps == 10


def test_enoki_readonly_enforcement():
    """Test that readonly fields are protected"""
    @enoki.state()
    def TestState():
        class T(Enum):
            DONE = auto()

        @enoki.on_state
        async def on_state(ctx: SharedContext[TestInterface]):
            try:
                ctx.common.max_steps = 999  # Should fail
                raise AssertionError("Should not modify readonly")
            except (AccessControlError, AttributeError):
                pass
            return T.DONE

        @enoki.transitions
        def transitions():
            return [LabeledTransition(T.DONE, TerminalState)]

    @enoki.state(config=StateConfiguration(terminal=True))
    def TerminalState():
        @enoki.on_state
        async def on_state(ctx: SharedContext[TestInterface]):
            pass

        @enoki.transitions
        def transitions():
            return []

    bb = TestBlackboard()
    sm = StateMachine(initial_state=TestState, common_data=bb)

    async def run():
        await sm.initialize()
        try:
            await sm.run()
        except StateMachineComplete:
            pass  # Expected

    asyncio.run(run())

    assert bb.max_steps == 10  # Unchanged


def test_enoki_backward_compatibility():
    """Test that code without interfaces still works"""
    @enoki.state()
    def TestState():
        class T(Enum):
            DONE = auto()

        @enoki.on_state
        async def on_state(ctx: SharedContext):
            # Full access without interface
            ctx.common.steps_taken = 5
            ctx.common.max_steps = 999
            ctx.common.secret = "accessed"
            return T.DONE

        @enoki.transitions
        def transitions():
            return [LabeledTransition(T.DONE, TerminalState)]

    @enoki.state(config=StateConfiguration(terminal=True))
    def TerminalState():
        @enoki.on_state
        async def on_state(ctx: SharedContext):
            pass

        @enoki.transitions
        def transitions():
            return []

    bb = TestBlackboard()
    sm = StateMachine(initial_state=TestState, common_data=bb)

    async def run():
        await sm.initialize()
        try:
            await sm.run()
        except StateMachineComplete:
            pass  # Expected

    asyncio.run(run())

    assert bb.steps_taken == 5
    assert bb.max_steps == 999
    assert bb.secret == "accessed"


def test_enoki_unspecified_field_inaccessible():
    """Test that fields not in interface are inaccessible"""
    @enoki.state()
    def TestState():
        class T(Enum):
            DONE = auto()

        @enoki.on_state
        async def on_state(ctx: SharedContext[TestInterface]):
            try:
                _ = ctx.common.secret  # Should fail
                raise AssertionError("Should not access unspecified field")
            except (AccessControlError, AttributeError):
                pass
            return T.DONE

        @enoki.transitions
        def transitions():
            return [LabeledTransition(T.DONE, TerminalState)]

    @enoki.state(config=StateConfiguration(terminal=True))
    def TerminalState():
        @enoki.on_state
        async def on_state(ctx: SharedContext[TestInterface]):
            pass

        @enoki.transitions
        def transitions():
            return []

    bb = TestBlackboard()
    sm = StateMachine(initial_state=TestState, common_data=bb)

    async def run():
        await sm.initialize()
        try:
            await sm.run()
        except StateMachineComplete:
            pass  # Expected

    asyncio.run(run())


if __name__ == "__main__":
    print("Testing Enoki interface integration...")

    test_enoki_interface_integration()
    print("✓ Interface integration test passed")

    test_enoki_readonly_enforcement()
    print("✓ Readonly enforcement test passed")

    test_enoki_backward_compatibility()
    print("✓ Backward compatibility test passed")

    test_enoki_unspecified_field_inaccessible()
    print("✓ Unspecified field test passed")

    print("\nAll tests passed!")
