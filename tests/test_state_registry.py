#!/usr/bin/env python3
"""Tests for StateRegistry with StateSpec"""

import sys
sys.path.insert(0, "src")

from enum import Enum, auto
import pytest

from mycorrhizal.enoki.core import (
    enoki,
    StateConfiguration,
    SharedContext,
    Again,
    LabeledTransition,
    StateRef,
    Push,
    Pop,
    StateRegistry,
    ValidationError,
    get_all_states,
)


# Define test states

@enoki.state()
def StateA():
    """First state"""

    @enoki.events
    class Events(Enum):
        GO_TO_B = auto()
        GO_TO_C = auto()

    @enoki.on_state
    async def on_state(ctx: SharedContext):
        return Events.GO_TO_B

    @enoki.transitions
    def transitions():
        return [
            LabeledTransition(Events.GO_TO_B, StateB),
            LabeledTransition(Events.GO_TO_C, StateC),
        ]


@enoki.state()
def StateB():
    """Second state"""

    @enoki.events
    class Events(Enum):
        BACK_TO_A = auto()
        AGAIN = auto()

    @enoki.on_state
    async def on_state(ctx: SharedContext):
        return Events.AGAIN

    @enoki.transitions
    def transitions():
        return [
            LabeledTransition(Events.BACK_TO_A, StateA),
            LabeledTransition(Events.AGAIN, Again),
        ]


@enoki.state(config=StateConfiguration(terminal=True))
def StateC():
    """Terminal state"""

    @enoki.on_state
    async def on_state(ctx: SharedContext):
        pass

    @enoki.transitions
    def transitions():
        return []


# Tests

def test_registry_resolve_state_spec():
    """Test resolving a StateSpec directly"""
    registry = StateRegistry()

    # Should return the same StateSpec
    resolved = registry.resolve_state(StateA)
    assert resolved is StateA
    assert resolved.name == StateA.name


def test_registry_resolve_state_ref():
    """Test resolving a StateRef string"""
    registry = StateRegistry()

    # Create a StateRef
    state_ref = StateRef("tests.test_state_registry.StateB")

    # Should resolve from global registry
    resolved = registry.resolve_state(state_ref)
    assert resolved is not None
    assert resolved.name == "tests.test_state_registry.StateB"


def test_registry_resolve_string():
    """Test resolving a state name string"""
    registry = StateRegistry()

    # Resolve by string name
    resolved = registry.resolve_state("tests.test_state_registry.StateC")
    assert resolved is not None
    assert resolved.name == "tests.test_state_registry.StateC"
    assert resolved.config.terminal is True


def test_registry_resolve_unknown_state():
    """Test that resolving unknown state raises error"""
    registry = StateRegistry()

    with pytest.raises(ValidationError, match="not found in registry"):
        registry.resolve_state("UnknownState")


def test_registry_resolve_validate_only():
    """Test validate_only mode returns None for unknown states"""
    registry = StateRegistry()

    resolved = registry.resolve_state("UnknownState", validate_only=True)
    assert resolved is None


def test_registry_get_transitions():
    """Test getting transition mapping for a state"""
    registry = StateRegistry()

    transitions = registry.get_transitions(StateA)

    # Should have transitions for both enum values and names
    assert StateA.Events.GO_TO_B in transitions
    assert "GO_TO_B" in transitions
    assert transitions[StateA.Events.GO_TO_B] is StateB

    assert StateA.Events.GO_TO_C in transitions
    assert "GO_TO_C" in transitions
    assert transitions[StateA.Events.GO_TO_C] is StateC


def test_registry_get_transitions_caching():
    """Test that transition mappings are cached"""
    registry = StateRegistry()

    # First call
    transitions1 = registry.get_transitions(StateB)
    # Second call should return cached version
    transitions2 = registry.get_transitions(StateB)

    assert transitions1 is transitions2


def test_registry_validate_all_states():
    """Test validating all reachable states"""
    registry = StateRegistry()

    result = registry.validate_all_states(StateA)

    assert result.valid is True
    assert len(result.errors) == 0
    assert "tests.test_state_registry.StateA" in result.discovered_states
    assert "tests.test_state_registry.StateB" in result.discovered_states
    assert "tests.test_state_registry.StateC" in result.discovered_states


def test_registry_validate_with_error_state():
    """Test validation with an error state"""
    registry = StateRegistry()

    result = registry.validate_all_states(StateA, error_state=StateC)

    assert result.valid is True
    assert "tests.test_state_registry.StateC" in result.discovered_states


def test_registry_validate_includes_continuations():
    """Test that Again and other continuations are validated"""
    registry = StateRegistry()

    result = registry.validate_all_states(StateB)

    # StateB has an Again transition, which should be valid
    assert result.valid is True

    transitions = registry.get_transitions(StateB)
    # Check that Again is in the transition values (as class, not instance)
    assert Again in transitions.values()
    # Also check by string key
    assert "AGAIN" in transitions
    assert transitions["AGAIN"] == Again


def test_registry_warns_about_unlabeled_transitions():
    """Test that unlabeled direct state transitions generate warnings"""
    # Create a state with unlabeled transition
    @enoki.state()
    def UnlabeledState():
        @enoki.on_state
        async def on_state(ctx: SharedContext):
            return None

        @enoki.transitions
        def transitions():
            # Direct state reference without label
            return [StateRef("tests.test_state_registry.StateA")]

        return {
            "on_state": on_state,
            "transitions": transitions,
        }

    registry = StateRegistry()

    result = registry.validate_all_states(UnlabeledState)

    # Should have a warning about unlabeled transition
    assert len(result.warnings) > 0
    assert any("without a label" in w for w in result.warnings)


def test_registry_requires_on_state():
    """Test that states without on_state raise an error at decoration time"""
    # The decorator should catch missing on_state at decoration time
    with pytest.raises(ValueError, match="must have an @on_state decorated method"):
        @enoki.state()
        def InvalidState():
            @enoki.transitions
            def transitions():
                return []

            return {
                "transitions": transitions,
                # Missing on_state!
            }


def test_registry_warns_about_timeout_without_handler():
    """Test that timeout without on_timeout generates a warning"""
    @enoki.state(config=StateConfiguration(timeout=5.0))
    def TimeoutWithoutHandler():
        @enoki.on_state
        async def on_state(ctx: SharedContext):
            return None

        @enoki.transitions
        def transitions():
            return []

        return {
            "on_state": on_state,
            "transitions": transitions,
            # Missing on_timeout!
        }

    registry = StateRegistry()

    result = registry.validate_all_states(TimeoutWithoutHandler)

    # Should warn about missing on_timeout
    assert any("timeout but no on_timeout" in w for w in result.warnings)


def test_registry_validates_invalid_timeout():
    """Test that invalid timeout values are caught"""
    @enoki.state(config=StateConfiguration(timeout=-1.0))
    def InvalidTimeout():
        @enoki.on_state
        async def on_state(ctx: SharedContext):
            return None

        @enoki.transitions
        def transitions():
            return []

        return {
            "on_state": on_state,
            "transitions": transitions,
        }

    registry = StateRegistry()

    result = registry.validate_all_states(InvalidTimeout)

    assert result.valid is False
    assert any("invalid timeout" in e for e in result.errors)
