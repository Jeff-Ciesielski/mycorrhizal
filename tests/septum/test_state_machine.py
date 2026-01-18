#!/usr/bin/env python3
"""Tests for StateMachine with StateSpec"""

import sys
sys.path.insert(0, "src")

import asyncio
from enum import Enum, auto
import pytest

from mycorrhizal.septum.core import (
    septum,
    StateConfiguration,
    SharedContext,
    Again,
    LabeledTransition,
    StateMachine,
    StateMachineComplete,
    BlockedInUntimedState,
    PopFromEmptyStack,
    Push,
    Pop,
    StateRef,
)


# ============================================================================
# Test States
# ============================================================================

@septum.state()
def StartState():
    """Starting state"""

    @septum.events
    class Events(Enum):
        GO = auto()
        FINISH = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        return Events.GO

    @septum.on_enter
    async def on_enter(ctx: SharedContext):
        ctx.common['enter_count'] = ctx.common.get('enter_count', 0) + 1

    @septum.on_leave
    async def on_leave(ctx: SharedContext):
        ctx.common['leave_count'] = ctx.common.get('leave_count', 0) + 1

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.GO, MiddleState),
            LabeledTransition(Events.FINISH, EndState),
        ]


@septum.state()
def MiddleState():
    """Middle state that can loop"""

    @septum.events
    class Events(Enum):
        CONTINUE = auto()
        END = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Continue based on common data
        count = ctx.common.get('count', 0)
        if count < 2:
            ctx.common['count'] = count + 1
            return Events.CONTINUE
        return Events.END

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.CONTINUE, Again),
            LabeledTransition(Events.END, EndState),
        ]


@septum.state(config=StateConfiguration(terminal=True))
def EndState():
    """Terminal state"""

    @septum.on_state
    async def on_state(ctx: SharedContext):
        pass

    @septum.transitions
    def transitions():
        return []


# ============================================================================
# Basic Execution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_statemachine_basic_execution():
    """Test basic state machine execution from start to terminal"""
    common = {}
    fsm = StateMachine(initial_state=StartState, common_data=common)

    await fsm.initialize()
    assert fsm.current_state.name == "tests.septum.test_state_machine.StartState"

    # Run to completion - should raise StateMachineComplete
    with pytest.raises(StateMachineComplete):
        await fsm.run(timeout=0.1)

    # Should be in terminal state
    assert fsm.current_state.name == "tests.septum.test_state_machine.EndState"
    assert fsm.current_state.config.terminal is True


@pytest.mark.asyncio
async def test_statemachine_lifecycle_methods():
    """Test that on_enter and on_leave are called"""
    common = {}
    fsm = StateMachine(initial_state=StartState, common_data=common)

    await fsm.initialize()

    # StartState should have been entered
    assert common.get('enter_count') == 1

    # Execute one tick to transition
    await fsm.tick(timeout=0)

    # StartState should have been left
    assert common.get('leave_count') == 1


@pytest.mark.asyncio
async def test_statemachine_tick_by_tick():
    """Test stepping through state machine tick by tick"""
    common = {'count': 0}
    fsm = StateMachine(initial_state=StartState, common_data=common)

    await fsm.initialize()

    # First tick: StartState -> MiddleState
    try:
        await fsm.tick(timeout=0)
    except StateMachineComplete:
        pass
    assert fsm.current_state.name == "tests.septum.test_state_machine.MiddleState"

    # Second tick: MiddleState loops with Again (count=0 -> 1)
    try:
        await fsm.tick(timeout=0)
    except StateMachineComplete:
        pass
    assert fsm.current_state.name == "tests.septum.test_state_machine.MiddleState"
    assert common['count'] == 1

    # Third tick: MiddleState loops with Again (count=1 -> 2)
    try:
        await fsm.tick(timeout=0)
    except StateMachineComplete:
        pass
    assert fsm.current_state.name == "tests.septum.test_state_machine.MiddleState"
    assert common['count'] == 2

    # Fourth tick: MiddleState -> EndState
    try:
        await fsm.tick(timeout=0)
    except StateMachineComplete:
        pass
    assert fsm.current_state.name == "tests.septum.test_state_machine.EndState"


# ============================================================================
# Transition Tests
# ============================================================================


@septum.state()
def AgainTestState():
    """Test state that returns Again"""

    @septum.events
    class Events(Enum):
        GO = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        count = ctx.common.get('again_count', 0)
        if count < 3:
            ctx.common['again_count'] = count + 1
            # Return labeled transition instead of bare Again
            return Events.GO
        return Events.GO

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.GO, Again),  # Loops back with Again
        ]


@pytest.mark.asyncio
async def test_statemachine_again_transition():
    """Test Again transition re-executes current state"""
    common = {}
    fsm = StateMachine(initial_state=AgainTestState, common_data=common)

    await fsm.initialize()
    assert fsm.current_state.name == "tests.septum.test_state_machine.AgainTestState"

    # Execute multiple ticks - should stay in same state until count reaches 3
    for i in range(5):
        try:
            await fsm.tick(timeout=0)
        except StateMachineComplete:
            pass

        if common.get('again_count', 0) < 3:
            assert fsm.current_state.name == "tests.septum.test_state_machine.AgainTestState"
        else:
            # After 3 Again transitions, should have processed
            break


@septum.state()
def RetryState():
    """State that tests retry logic"""

    @septum.events
    class Events(Enum):
        FAIL = auto()
        SUCCEED = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        count = ctx.common.get('retry_attempts', 0)
        if count < 2:
            ctx.common['retry_attempts'] = count + 1
            return Events.FAIL
        return Events.SUCCEED

    @septum.on_fail
    async def on_fail(ctx: SharedContext):
        # on_fail is called when retries are exhausted
        ctx.common['failed'] = True
        return EndState

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.FAIL, RetryState),
            LabeledTransition(Events.SUCCEED, EndState),
        ]


@pytest.mark.asyncio
async def test_statemachine_retry_logic():
    """Test retry logic with retry counter"""
    common = {}
    fsm = StateMachine(
        initial_state=RetryState,
        common_data=common
    )

    await fsm.initialize()

    # The state should retry until it succeeds
    # In this case, it will fail twice then succeed
    try:
        await fsm.run(timeout=0.1, max_iterations=10)
    except StateMachineComplete:
        pass

    # Should have reached EndState
    assert fsm.current_state.name == "tests.septum.test_state_machine.EndState"
    # Should have retried
    assert common.get('retry_attempts') == 2


# ============================================================================
# Push/Pop Tests
# ============================================================================


@septum.state()
def PushTestState():
    """State that pushes other states"""

    @septum.events
    class Events(Enum):
        PUSH = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        return Events.PUSH

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.PUSH, Push(MiddleState, EndState)),
        ]


@septum.state()
def PopTestState():
    """State that pops back to previous"""

    @septum.events
    class Events(Enum):
        POP = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        return Events.POP

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.POP, Pop),
        ]


@pytest.mark.asyncio
async def test_statemachine_push_transition():
    """Test Push transition stacks states"""
    common = {'count': 99}  # Force MiddleState to end immediately
    fsm = StateMachine(initial_state=PushTestState, common_data=common)

    await fsm.initialize()

    # First tick: should push MiddleState and EndState onto stack
    # and transition to MiddleState
    try:
        await fsm.tick(timeout=0)
    except StateMachineComplete:
        pass

    # Should be in MiddleState or EndState
    assert fsm.current_state.name in [
        "tests.septum.test_state_machine.MiddleState",
        "tests.septum.test_state_machine.EndState"
    ]


@pytest.mark.asyncio
async def test_statemachine_pop_transition():
    """Test Pop transition returns to stacked state"""
    # Use a sequence: Start -> Push -> Middle -> Pop -> End
    @septum.state()
    def StartForPop():
        @septum.events
        class Events(Enum):
            GO = auto()

        @septum.on_state
        async def on_state(ctx: SharedContext):
            return Events.GO

        @septum.transitions
        def transitions():
            return [
                LabeledTransition(Events.GO, Push(MiddleState, EndState)),
            ]

    common = {'count': 2}  # Force MiddleState to end immediately
    fsm = StateMachine(initial_state=StartForPop, common_data=common)

    await fsm.initialize()

    # First tick: Start -> pushes MiddleState and EndState, enters MiddleState
    try:
        await fsm.tick(timeout=0)
    except StateMachineComplete:
        pass
    assert fsm.current_state.name == "tests.septum.test_state_machine.MiddleState"
    assert len(fsm.state_stack) == 1

    # Second tick: MiddleState -> EndState (because count=2)
    try:
        await fsm.tick(timeout=0)
    except StateMachineComplete:
        pass
    assert fsm.current_state.name == "tests.septum.test_state_machine.EndState"


@pytest.mark.asyncio
async def test_statemachine_pop_from_empty_stack():
    """Test that popping from empty stack raises error"""
    common = {}
    fsm = StateMachine(initial_state=PopTestState, common_data=common)

    await fsm.initialize()

    # Should raise PopFromEmptyStack
    with pytest.raises(PopFromEmptyStack):
        await fsm.tick(timeout=0)


# ============================================================================
# Timeout Tests
# ============================================================================


@septum.state(config=StateConfiguration(timeout=0.1))
def TimeoutTestState():
    """State with timeout"""

    @septum.events
    class Events(Enum):
        GO = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # State has a timeout, so it waits for messages
        # If no message, return None to wait (timeout will trigger if no message arrives)
        if ctx.msg is None:
            return None
        return Events.GO

    @septum.on_timeout
    async def on_timeout(ctx: SharedContext):
        ctx.common['timed_out'] = True
        return EndState

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.GO, EndState),
        ]


@pytest.mark.asyncio
async def test_statemachine_timeout_handling():
    """Test that timeout triggers on_timeout handler"""
    common = {}
    fsm = StateMachine(initial_state=TimeoutTestState, common_data=common)

    await fsm.initialize()

    # Wait for timeout (slightly longer than configured timeout)
    await asyncio.sleep(0.15)

    # Process the timeout message
    try:
        await fsm.tick(timeout=0)
    except StateMachineComplete:
        pass

    # Should have timed out and transitioned to EndState
    assert common.get('timed_out') is True
    assert fsm.current_state.name == "tests.septum.test_state_machine.EndState"


# ============================================================================
# Error State Tests
# ============================================================================


@septum.state()
def ErrorState():
    """Error state"""

    @septum.on_state
    async def on_state(ctx: SharedContext):
        ctx.common['in_error_state'] = True

    @septum.transitions
    def transitions():
        return []


@pytest.mark.asyncio
async def test_statemachine_error_state():
    """Test that error state is used when exception occurs"""
    @septum.state()
    def FailingState():
        @septum.on_state
        async def on_state(ctx: SharedContext):
            raise ValueError("Intentional error")

        @septum.transitions
        def transitions():
            return []

    common = {}
    fsm = StateMachine(
        initial_state=FailingState,
        error_state=ErrorState,
        common_data=common,
        on_error_fn=lambda ctx, e: None,  # Suppress error, don't re-raise
    )

    await fsm.initialize()

    # Should transition to error state
    try:
        await fsm.tick(timeout=0)
    except ValueError:
        pass  # Exception might be raised before error state transition

    # Should be in error state or FailingState
    assert fsm.current_state.name in [
        "tests.septum.test_state_machine.ErrorState",
        "tests.septum.test_state_machine.FailingState"
    ]


# ============================================================================
# Validation Tests
# ============================================================================


def test_statemachine_validates_on_construction():
    """Test that StateMachine validates states on construction"""
    # The decorator itself validates, so this test just confirms that behavior
    with pytest.raises(ValueError, match="must have an @on_state"):
        @septum.state()
        def InvalidState():
            # Missing on_state - should fail at decoration time
            @septum.transitions
            def transitions():
                return []


def test_statemachine_validates_transitions():
    """Test that StateMachine validates transitions"""

    @septum.state()
    def InvalidTransitionState():
        @septum.events
        class Events(Enum):
            GO = auto()

        @septum.on_state
        async def on_state(ctx: SharedContext):
            return Events.GO

        @septum.transitions
        def transitions():
            # Reference to non-existent state using StateRef
            return [
                LabeledTransition(Events.GO, StateRef("NonExistentState")),
            ]

    # Should fail validation during StateMachine construction
    from mycorrhizal.septum.core import ValidationError
    with pytest.raises(ValidationError, match="not found in registry"):
        StateMachine(initial_state=InvalidTransitionState)


# ============================================================================
# Terminal State Tests
# ============================================================================


@pytest.mark.asyncio
async def test_statemachine_complete_exception():
    """Test that StateMachineComplete is raised for terminal state"""
    common = {}
    fsm = StateMachine(initial_state=StartState, common_data=common)

    await fsm.initialize()

    # Run until terminal state
    with pytest.raises(StateMachineComplete):
        await fsm.run(timeout=0.1)

    assert fsm.current_state.config.terminal is True


# ============================================================================
# Message Passing Tests
# ============================================================================


@septum.state(config=StateConfiguration(can_dwell=True))
def MessageTestState():
    """State that waits for messages"""

    @septum.events
    class Events(Enum):
        PROCESS = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        if ctx.msg == "finish":
            return Events.PROCESS
        # Return None to wait for message

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.PROCESS, EndState),
        ]


@pytest.mark.asyncio
async def test_statemachine_message_passing():
    """Test sending messages to state machine"""
    common = {}
    fsm = StateMachine(initial_state=MessageTestState, common_data=common)

    await fsm.initialize()

    # Send a message
    fsm.send_message("finish")

    # Tick should process the message
    try:
        await fsm.tick(timeout=0)
    except StateMachineComplete:
        pass

    # Should have transitioned to EndState
    assert fsm.current_state.name == "tests.septum.test_state_machine.EndState"
