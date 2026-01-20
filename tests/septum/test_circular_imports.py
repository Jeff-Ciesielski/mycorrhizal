"""Test string-based state references for breaking circular imports.

This test proves that:
1. Module A can import Module B and reference its states directly
2. Module B can reference Module A's states via string (avoiding circular import)
3. The global registry enables cross-module string resolution
4. State machines can validate and run with circular dependencies
"""

import pytest
import asyncio
import importlib
from enum import Enum, auto
from mycorrhizal.septum.core import (
    StateMachine,
    septum,
    _clear_state_registry,
    _clear_interface_view_cache,
    LabeledTransition,
    get_all_states,
)


class TestCircularImports:
    """Test that string-based state references break circular imports."""

    @staticmethod
    def _load_fixtures():
        """Reload fixture modules to re-register states after conftest clears registry."""
        import tests.septum.fixtures.circular_a as circular_a
        import tests.septum.fixtures.circular_b as circular_b

        # Force reload to re-execute module-level code (state registration)
        importlib.reload(circular_a)
        importlib.reload(circular_b)

        return circular_a.StateA, circular_b.StateB

    def test_can_import_both_modules(self):
        """Verify both modules can be imported without circular import errors."""
        # This should work without ImportError or circular reference errors
        StateA, StateB = self._load_fixtures()

        # Both states should be registered
        from mycorrhizal.septum.core import get_all_states
        all_states = get_all_states()

        assert "tests.septum.fixtures.circular_a.StateA" in all_states
        assert "tests.septum.fixtures.circular_b.StateB" in all_states

    @pytest.mark.asyncio
    async def test_string_reference_resolution(self):
        """Test that string references are resolved correctly at validation time."""
        StateA, _ = self._load_fixtures()

        # Create FSM - validation should resolve string reference
        # StateB references StateA via string "tests.septum.fixtures.circular_a.StateA"
        # This should not raise ValidationError
        fsm = StateMachine(initial_state=StateA)

        # Verify initialization succeeds (validation passed)
        await fsm.initialize()

        # Current state should be StateA
        assert fsm.current_state.name == "tests.septum.fixtures.circular_a.StateA"

    @pytest.mark.asyncio
    async def test_transitions_across_modules(self):
        """Test that transitions work across modules with mixed reference types."""
        StateA, StateB = self._load_fixtures()

        fsm = StateMachine(initial_state=StateA)
        await fsm.initialize()

        # Initial state
        assert fsm.current_state.name == "tests.septum.fixtures.circular_a.StateA"

        # Transition to StateB (direct reference in transitions)
        fsm.send_message("go")
        await fsm.tick()
        assert fsm.current_state.name == "tests.septum.fixtures.circular_b.StateB"

        # Transition back to StateA (string reference in transitions!)
        fsm.send_message("back")
        await fsm.tick()
        assert fsm.current_state.name == "tests.septum.fixtures.circular_a.StateA"

    @pytest.mark.asyncio
    async def test_no_circular_import_error(self):
        """Verify that importing modules doesn't cause circular import errors.

        This is the key test - if circular imports existed, the imports would fail.
        """
        # Import module A (which imports module B)
        import tests.septum.fixtures.circular_a as circular_a

        # Import module B (which does NOT import module A, uses string reference)
        import tests.septum.fixtures.circular_b as circular_b

        # If we got here, no circular import occurred!
        assert circular_a.StateA is not None
        assert circular_b.StateB is not None

        # Verify StateB has string reference to StateA
        transitions = circular_b.StateB.get_transitions()
        assert len(transitions) == 1

        # The transition should have a StateRef target
        from mycorrhizal.septum.core import StateRef
        transition = transitions[0]
        assert isinstance(transition.transition, StateRef)
        assert transition.transition.state == "tests.septum.fixtures.circular_a.StateA"

    @pytest.mark.asyncio
    async def test_registry_serves_as_global_phone_book(self):
        """Test that the global registry acts as a shared 'phone book' for all modules.

        This verifies the key insight: string references require a shared registry
        that all modules can query. A per-module registry would break this.
        """
        StateA, StateB = self._load_fixtures()
        from mycorrhizal.septum.core import get_all_states

        # Get all states from global registry
        all_states = get_all_states()

        # Both modules' states should be in the SAME registry
        assert "tests.septum.fixtures.circular_a.StateA" in all_states
        assert "tests.septum.fixtures.circular_b.StateB" in all_states

        # Verify they're the same objects
        assert all_states["tests.septum.fixtures.circular_a.StateA"] is StateA
        assert all_states["tests.septum.fixtures.circular_b.StateB"] is StateB

        # Create FSM from either state - both should work
        fsm_a = StateMachine(initial_state=StateA)
        await fsm_a.initialize()

        fsm_b = StateMachine(initial_state=StateB)
        await fsm_b.initialize()

        # Verify both FSMs can run successfully
        assert fsm_a.current_state.name == "tests.septum.fixtures.circular_a.StateA"
        assert fsm_b.current_state.name == "tests.septum.fixtures.circular_b.StateB"


class TestRegistryCleanup:
    """Test that registry cleanup prevents test pollution."""

    @pytest.mark.asyncio
    async def test_registry_cleanup_isolates_tests(self):
        """Verify that _clear_state_registry() prevents test pollution.

        This test saves the current registry state, tests with a clean registry,
        then restores the original state to avoid polluting other tests.
        """
        # Save current registry state
        original_states = get_all_states().copy()

        try:
            # Clear registry for this test
            _clear_state_registry()

            # Registry should be empty now
            states_before = get_all_states()
            assert len(states_before) == 0

            # Define a temporary state
            @septum.state()
            def TempState():
                class Events(Enum):
                    DONE = auto()

                @septum.on_state
                async def on_state(ctx):
                    return Events.DONE

                @septum.transitions
                def transitions():
                    return []

            # State should be registered with its full qualified name
            states_during = get_all_states()
            assert len(states_during) > 0

            # Find the state - it will have a fully qualified name
            temp_state_key = None
            for key in states_during.keys():
                if "TempState" in key:
                    temp_state_key = key
                    break

            assert temp_state_key is not None, f"TempState not found in {list(states_during.keys())}"

        finally:
            # Restore original registry state to avoid polluting other tests
            _clear_state_registry()
            for state_name, state_spec in original_states.items():
                # Re-register states from the global registry dict
                from mycorrhizal.septum.core import _state_registry
                _state_registry[state_name] = state_spec

    @pytest.mark.asyncio
    async def test_multiple_isolated_fsms(self):
        """Test that multiple FSMs can coexist in the registry without interference.

        This test verifies that FSMs with different state definitions can
        coexist in the global registry without interfering with each other,
        thanks to fully-qualified state names.
        """
        # Define FSM 1 states
        @septum.state()
        def Fsm1State1():
            class Events(Enum):
                GO = auto()

            @septum.on_state
            async def on_state(ctx):
                return Events.GO

            @septum.transitions
            def transitions():
                return [LabeledTransition(Events.GO, Fsm1State2)]

        @septum.state()
        def Fsm1State2():
            @septum.on_state
            async def on_state(ctx):
                return None

            @septum.transitions
            def transitions():
                return []

        # Create FSM 1 using global registry
        fsm1 = StateMachine(initial_state=Fsm1State1)
        await fsm1.initialize()

        # Define FSM 2 states
        @septum.state()
        def Fsm2State1():
            class Events(Enum):
                GO = auto()

            @septum.on_state
            async def on_state(ctx):
                return Events.GO

            @septum.transitions
            def transitions():
                return [LabeledTransition(Events.GO, Fsm2State2)]

        @septum.state()
        def Fsm2State2():
            @septum.on_state
            async def on_state(ctx):
                return None

            @septum.transitions
            def transitions():
                return []

        # Create FSM 2 using global registry
        fsm2 = StateMachine(initial_state=Fsm2State1)
        await fsm2.initialize()

        # Verify both FSMs work correctly
        # They can coexist in the global registry without interfering
        # because they have fully-qualified names

        # FSM 1 should work
        assert fsm1.current_state.name == "tests.septum.test_circular_imports.TestRegistryCleanup.test_multiple_isolated_fsms.<locals>.Fsm1State1"

        # FSM 2 should work
        assert fsm2.current_state.name == "tests.septum.test_circular_imports.TestRegistryCleanup.test_multiple_isolated_fsms.<locals>.Fsm2State1"

        # Both FSMs' states should be in the global registry
        current_states = get_all_states()
        state_names = list(current_states.keys())

        fsm2_states = [s for s in state_names if "Fsm2State" in s]
        fsm1_states = [s for s in state_names if "Fsm1State" in s]

        assert len(fsm2_states) == 2, f"Expected 2 FSM2 states, got {len(fsm2_states)}: {fsm2_states}"
        assert len(fsm1_states) == 2, f"Expected 2 FSM1 states, got {len(fsm1_states)}: {fsm1_states}"


