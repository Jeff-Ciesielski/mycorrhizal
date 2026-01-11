#!/usr/bin/env python3
"""Basic test of the new decorator-based Enoki API"""

import sys
sys.path.insert(0, "src")

from enum import Enum, auto
from mycorrhizal.enoki.core import (
    enoki,
    StateConfiguration,
    StateSpec,
    SharedContext,
    Again,
    LabeledTransition,
    Push,
    Pop,
    get_all_states,
    get_state,
)

# Define some states using the decorator API

@enoki.state(config=StateConfiguration(timeout=5.0))
def InitialState():
    """The initial state"""

    @enoki.events
    class Events(Enum):
        GO = auto()
        DONE = auto()

    @enoki.on_state
    async def on_state(ctx: SharedContext):
        print(f"In InitialState! msg={ctx.msg}")
        ctx.msg = None
        return Events.GO

    @enoki.transitions
    def transitions():
        return [
            LabeledTransition(Events.GO, ProcessingState),
            LabeledTransition(Events.DONE, CompleteState),
        ]


@enoki.state()
def ProcessingState():
    """A processing state"""

    @enoki.events
    class Events(Enum):
        CONTINUE = auto()
        DONE = auto()

    @enoki.on_state
    async def on_state(ctx: SharedContext):
        print(f"In ProcessingState! msg={ctx.msg}")
        ctx.msg = None
        return Events.DONE

    @enoki.transitions
    def transitions():
        return [
            LabeledTransition(Events.CONTINUE, Again),
            LabeledTransition(Events.DONE, CompleteState),
        ]


@enoki.state(config=StateConfiguration(terminal=True))
def CompleteState():
    """Terminal state"""

    @enoki.on_state
    async def on_state(ctx: SharedContext):
        print("In CompleteState - we're done!")

    @enoki.transitions
    def transitions():
        return []


# Tests

def test_state_registration():
    """Test that states are properly registered"""
    states = get_all_states()
    # Check that we have at least our 3 states (global registry may have more from other tests)
    assert len(states) >= 3

    # Check that all expected states are registered
    # Note: module name is 'tests.test_decorator_basic' when run via pytest
    assert "tests.enoki.test_decorator_basic.InitialState" in states
    assert "tests.enoki.test_decorator_basic.ProcessingState" in states
    assert "tests.enoki.test_decorator_basic.CompleteState" in states


def test_state_spec_properties():
    """Test that StateSpec has all required properties"""
    # Note: module name includes 'tests.' prefix when run via pytest
    assert InitialState.name == "tests.enoki.test_decorator_basic.InitialState"
    assert InitialState.base_name == "InitialState"
    assert InitialState.config.timeout == 5.0
    assert InitialState.config.terminal == False

    assert CompleteState.config.terminal == True


def test_state_has_on_state():
    """Test that states have on_state handlers"""
    assert InitialState.on_state is not None
    assert ProcessingState.on_state is not None
    assert CompleteState.on_state is not None


def test_state_has_transitions():
    """Test that states have transitions defined"""
    initial_trans = InitialState.get_transitions()
    assert len(initial_trans) == 2
    assert all(isinstance(t, LabeledTransition) for t in initial_trans)

    processing_trans = ProcessingState.get_transitions()
    assert len(processing_trans) == 2

    complete_trans = CompleteState.get_transitions()
    assert len(complete_trans) == 0


def test_state_has_events():
    """Test that states can have event enums"""
    assert InitialState.Events is not None
    assert ProcessingState.Events is not None
    assert CompleteState.Events is None  # No Events enum defined


def test_state_is_state_transition():
    """Test that StateSpec is a StateTransition"""
    from mycorrhizal.enoki.core import StateTransition

    assert isinstance(InitialState, StateTransition)
    # Can't use issubclass with dataclass instances


def test_state_equality():
    """Test state equality"""
    # Note: module name includes 'tests.' prefix when run via pytest
    same_state = get_state("tests.enoki.test_decorator_basic.InitialState")
    assert InitialState == same_state
    assert not (InitialState == "wrong name")
    assert InitialState != ProcessingState


def test_state_name_property():
    """Test that StateSpec name property works correctly"""
    # Note: module name includes 'tests.' prefix when run via pytest
    assert InitialState.name == "tests.enoki.test_decorator_basic.InitialState"
    assert InitialState.base_name == "InitialState"
