#!/usr/bin/env python3
"""
Enoki Comprehensive Tests

Comprehensive test suite for the class-method-based Enoki architecture.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
import time
import gevent
from gevent import sleep, spawn
from enum import auto

# Import the Enoki modules
from mycorrhizal.enoki import (
    State,
    StateMachine,
    TransitionName,
    GlobalTransitions,
    StateConfiguration,
    SharedContext,
    TimeoutMessage,
    Push,
    Pop,
    ValidationError,
    BlockedInUntimedState,
    PopFromEmptyStack,
    StateMachineComplete
)

# Import our testing utilities
from mycorrhizal.enoki.testing_utils import (
    EnokiTestCase,
    StateMachineTestCase,
    make_mock_context,
    run_state_on_state,
    simulate_state_lifecycle,
    run_fsm_scenario,
)


class TestStringResolution(EnokiTestCase):
    """Test basic state transition behaviors in  architecture."""

    class TestTransitions(TransitionName):
        GO_TO_B = "go_to_b"

    class StateA(State):
        CONFIG = StateConfiguration()

        @classmethod
        def transitions(cls):
            return {
                # NOTE: For string resolution to work, states CANNOT be declared
                # inside a function as they will be cleaned up / GCd. They only exist within
                # the context of that function
                TestStringResolution.TestTransitions.GO_TO_B: "test_enoki.TestStringResolution.StateB",
                GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
            }

        @classmethod
        def on_state(cls, ctx: SharedContext):
            return TestStringResolution.TestTransitions.GO_TO_B

    class StateB(State):
        CONFIG = StateConfiguration()

        @classmethod
        def transitions(cls):
            return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

        @classmethod
        def on_state(cls, ctx: SharedContext):
            return GlobalTransitions.UNHANDLED

    def test_transition_to_string_state_reference(self):
        """States should be able to transition via string references."""

        result = run_fsm_scenario(TestStringResolution.StateA, timeout=0)
        self.assertEqual(result.get("final_state"), TestStringResolution.StateB)


class TestBasicStateTransitions(EnokiTestCase):
    def test_unhandled_returns_unhandled(self):
        """Returning UNHANDLED should result in UNHANDLED."""

        class TestState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        self.assertNoTransition(TestState)

    def test_push_single_state(self):
        """Push with single state should work correctly."""

        class TestTransitions(TransitionName):
            PUSH_B = "push_b"

        class StateA(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.PUSH_B: Push("enoki_tests.StateB"),
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.PUSH_B

        class StateB(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        result = run_state_on_state(StateA)
        self.assertEqual(result, TestTransitions.PUSH_B)

        # Verify the transition mapping
        transitions = StateA.transitions()
        push_target = transitions[TestTransitions.PUSH_B]
        self.assertIsInstance(push_target, Push)

    def test_push_multiple_states(self):
        """Push with multiple states should work correctly."""

        class TestTransitions(TransitionName):
            PUSH_MULTI = "push_multi"

        class StateA(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.PUSH_MULTI: Push(StateB, StateC),
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.PUSH_MULTI

        class StateB(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        class StateC(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        result = run_state_on_state(StateA)
        self.assertEqual(result, TestTransitions.PUSH_MULTI)

        # Verify the transition mapping
        transitions = StateA.transitions()
        push_target = transitions[TestTransitions.PUSH_MULTI]
        self.assertIsInstance(push_target, Push)
        self.assertEqual(len(push_target.states), 2)

    def test_pop_transition(self):
        """Pop transitions should work correctly."""

        class TestTransitions(TransitionName):
            POP_BACK = "pop_back"

        class TestState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.POP_BACK: Pop,
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP_BACK

        result = run_state_on_state(TestState)
        self.assertEqual(result, TestTransitions.POP_BACK)

        # Verify the transition mapping
        transitions = TestState.transitions()
        pop_target = transitions[TestTransitions.POP_BACK]
        self.assertIs(pop_target, Pop)


class TestStateLifecycle(EnokiTestCase):
    """Test state lifecycle methods in  architecture."""

    def test_on_enter_called(self):
        """on_enter should be callable and not crash."""

        class TestState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_enter(cls, ctx: SharedContext):
                ctx.common.entered = True

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        self.assertLifecycleMethodCalled(TestState, "on_enter")

    def test_on_leave_called(self):
        """on_leave should be callable and not crash."""

        class TestState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_leave(cls, ctx: SharedContext):
                ctx.common.left = True

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        self.assertLifecycleMethodCalled(TestState, "on_leave")

    def test_on_fail_called_on_retry_limit(self):
        """on_fail should be callable and handle retry exhaustion."""

        class TestTransitions(TransitionName):
            FAILED = "failed"

        class TestState(State):
            CONFIG = StateConfiguration(retries=3)

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.FAILED: "enoki_tests.ErrorState",
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_fail(cls, ctx: SharedContext):
                ctx.common.failed = True
                return TestTransitions.FAILED

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        class ErrorState(State):
            CONFIG = StateConfiguration(terminal=True)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        context = make_mock_context()
        result = TestState.on_fail(context)
        self.assertEqual(result, TestTransitions.FAILED)
        self.assertTrue(context.common.failed)


class TestStateConfiguration(EnokiTestCase):
    """Test StateConfiguration and state properties."""

    def test_timeout_configuration(self):
        """States should properly configure timeouts."""

        class TimeoutState(State):
            CONFIG = StateConfiguration(timeout=10.0)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        self.assertStateConfiguration(TimeoutState, timeout=10.0)

    def test_retry_configuration(self):
        """States should properly configure retry limits."""

        class RetryState(State):
            CONFIG = StateConfiguration(retries=5)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        self.assertStateConfiguration(RetryState, retries=5)

    def test_terminal_configuration(self):
        """States should properly configure terminal status."""

        class TerminalState(State):
            CONFIG = StateConfiguration(terminal=True)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        self.assertStateConfiguration(TerminalState, terminal=True)

    def test_can_dwell_configuration(self):
        """States should properly configure dwelling capability."""

        class DwellState(State):
            CONFIG = StateConfiguration(can_dwell=True)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        self.assertStateConfiguration(DwellState, can_dwell=True)


class TestSharedContext(EnokiTestCase):
    """Test SharedContext functionality."""

    def test_shared_state_modification(self):
        """States should be able to modify shared state."""

        class TestState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                ctx.common.test_value = 42
                return GlobalTransitions.UNHANDLED

        self.assertSharedStateModified(TestState, "test_value", 42)

    def test_message_handling(self):
        """States should be able to access and respond to messages."""

        class TestState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                if ctx.msg and ctx.msg.get("action") == "test":
                    ctx.common.received_test = True
                return GlobalTransitions.UNHANDLED

        context = make_mock_context(msg={"action": "test"})
        run_state_on_state(TestState, context)
        self.assertTrue(context.common.received_test)

    def test_send_message_functionality(self):
        """States should be able to send messages."""

        class TestState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                ctx.send_message({"response": "sent"})
                return GlobalTransitions.UNHANDLED

        context = make_mock_context()
        run_state_on_state(TestState, context)

        # Verify send_message was called
        context.send_message.assert_called_once_with({"response": "sent"})

    def test_log_functionality(self):
        """States should be able to log messages."""

        class TestState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                ctx.log("Test log message")
                return GlobalTransitions.UNHANDLED

        context = make_mock_context()
        run_state_on_state(TestState, context)

        # Verify log was called
        context.log.assert_called_once_with("Test log message")


class TestStateMachineLifecycle(StateMachineTestCase):
    """Test StateMachine lifecycle and behavior."""

    def test_fsm_starts_in_initial_state(self):
        """FSM should start in the specified initial state."""

        class InitialState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        sm = StateMachine(InitialState)
        self.assertEqual(sm.current_state, InitialState)

    def test_fsm_transitions_between_states(self):
        """FSM should transition between states correctly."""

        class TestTransitions(TransitionName):
            GO_TO_B = "go_to_b"

        class StateA(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.GO_TO_B: StateB,
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.GO_TO_B

        class StateB(State):
            CONFIG = StateConfiguration(can_dwell=True)  # Prevent blocking

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        sm = StateMachine(StateA)
        sm.tick()

        self.assertEqual(sm.current_state, StateB)

    def test_fsm_completes_on_terminal_state(self):
        """FSM should complete when reaching a terminal state."""

        class TerminalState(State):
            CONFIG = StateConfiguration(terminal=True)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        self.assertFSMCompletes(TerminalState)

    def test_fsm_blocks_in_untimed_state(self):
        """FSM should raise BlockedInUntimedState for non-dwelling states."""

        class TestTransitions(TransitionName):
            Nada = "Nada"

        class BlockingState(State):
            CONFIG = StateConfiguration()  # No timeout, no dwelling

            @classmethod
            def transitions(cls):
                return {TestTransitions.Nada: None}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                pass

        self.assertFSMBlocks(BlockingState)

    def test_fsm_reset_functionality(self):
        """FSM reset should return to initial state and clear stack."""

        class TestTransitions(TransitionName):
            PUSH_B = "push_b"

        class StateA(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.PUSH_B: Push(StateB),
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.PUSH_B

        class StateB(State):
            CONFIG = StateConfiguration(can_dwell=True)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        print(StateA.name())
        sm = StateMachine(StateA)
        sm.tick()  # Should transition to StateB

        # Verify state changed
        self.assertEqual(sm.current_state, StateB)

        # Reset and verify
        sm.reset()
        self.assertEqual(sm.current_state, StateA)
        self.assertEqual(len(sm.state_stack), 0)


class TestStackOperations(StateMachineTestCase):
    """Test push/pop stack operations."""

    def test_push_multiple_states_uses_stack(self):
        """Pushing multiple states should use the state stack correctly."""

        class TestTransitions(TransitionName):
            PUSH_MULTI = "push_multi"
            POP_BACK = "pop_back"

        class StateA(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.PUSH_MULTI: Push(StateB, StateC),
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.PUSH_MULTI

        class StateB(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.POP_BACK: Pop,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                ctx.common["B_entered"] = True
                return TestTransitions.POP_BACK

        class StateC(State):
            CONFIG = StateConfiguration(terminal=True)

            @classmethod
            def on_state(cls, ctx: SharedContext):
                ctx.common["C_entered"] = True
                return GlobalTransitions.UNHANDLED

        cd = {"B_entered": False, "C_entered": False}
        sm = StateMachine(StateA, common_data=cd)

        # First tick: A -> B, push C
        with self.assertRaises(StateMachineComplete):
            sm.tick()
        self.assertEqual(sm.current_state, StateC)
        self.assertEqual(len(sm.state_stack), 0)
        self.assertTrue(cd['B_entered'])
        self.assertTrue(cd['C_entered'])

    def test_pop_from_empty_stack_handled(self):
        """Popping from empty stack should be handled gracefully."""

        class TestTransitions(TransitionName):
            POP_BACK = "pop_back"

        class PopState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.POP_BACK: Pop,
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP_BACK

        sm = StateMachine(PopState)

        # This should handle the empty stack gracefully
        # The exact behavior depends on implementation
        with self.assertRaises(PopFromEmptyStack):
            sm.tick()


class TestValidation(StateMachineTestCase):
    """Test state machine validation."""

    def test_missing_state_reference_validation(self):
        """StateMachine should validate state references during construction."""

        class BadState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: "nonexistent.module.FakeState"}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        self.assertValidationError(BadState)

    def test_invalid_transition_enum_validation(self):
        """StateMachine should validate transition enums."""

        class BadState(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    "not_an_enum": "some_state",  # Invalid key
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        self.assertValidationError(BadState)

    def test_timeout_without_handler_warning(self):
        """States with timeout but no timeout handler should generate warnings."""

        class TimeoutState(State):
            CONFIG = StateConfiguration(timeout=5.0)

            @classmethod
            def transitions(cls):
                return {
                    # Missing GlobalTransitions.TIMEOUT handler
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        # This should succeed but generate warnings
        try:
            sm = StateMachine(TimeoutState)
            # If no exception, the validation passed with warnings
        except ValidationError:
            self.fail("Should not raise ValidationError for missing timeout handler")


class TestComplexScenarios(StateMachineTestCase):
    """Test complex real-world-like scenarios."""

    def test_timeout(self):
        """Test that timeouts work"""

        class TimeoutTransitions(TransitionName):
            RETURN_TO_IDLE = auto()
            GO_TO_LONG_STATE = auto()

        class Idle(State):
            CONFIG = StateConfiguration(can_dwell=True)

            @classmethod
            def transitions(cls):
                return {TimeoutTransitions.GO_TO_LONG_STATE: LongState}

            @classmethod
            def on_state(cls, ctx):
                return TimeoutTransitions.GO_TO_LONG_STATE

        class LongState(State):
            CONFIG = StateConfiguration(timeout=0.2)

            @classmethod
            def transitions(cls):
                return {TimeoutTransitions.RETURN_TO_IDLE: Idle}

            @classmethod
            def on_state(cls, ctx):
                pass

            @classmethod
            def on_timeout(cls, ctx):
                print("timed out")
                ctx.common["timed_out"] = True
                return TimeoutTransitions.RETURN_TO_IDLE

        # Run the scenario
        scenario_result = run_fsm_scenario(
            Idle, max_iterations=5, common_data={"timed_out": False}, timeout=0.1
        )

        self.assertTrue(scenario_result["ctx"].common.get("timed_out"))

    def test_imaging_workflow_scenario(self):
        """Test a complex imaging workflow with multiple states."""

        class ImagingTransitions(TransitionName):
            LIGHTING_SET = "lighting_set"
            PICTURE_TAKEN = "picture_taken"
            FAILED = "failed"

        class SetImageLighting(State):
            CONFIG = StateConfiguration(timeout=10.0, retries=3)

            @classmethod
            def transitions(cls):
                return {
                    ImagingTransitions.LIGHTING_SET: TakePicture,
                    ImagingTransitions.FAILED: ErrorState,
                    GlobalTransitions.TIMEOUT: GlobalTransitions.RETRY,
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_enter(cls, ctx: SharedContext):
                ctx.log("Setting up lighting...")
                # Simulate async setup
                spawn(cls._simulate_setup, ctx)

            @classmethod
            def _simulate_setup(cls, ctx: SharedContext):
                sleep(0.1)  # Short simulation
                ctx.send_message({"type": "lighting_ready"})

            @classmethod
            def on_state(cls, ctx: SharedContext):
                if isinstance(ctx.msg, TimeoutMessage):
                    return GlobalTransitions.TIMEOUT
                elif ctx.msg and ctx.msg.get("type") == "lighting_ready":
                    return ImagingTransitions.LIGHTING_SET
                return GlobalTransitions.UNHANDLED

            @classmethod
            def on_timeout(cls, ctx: SharedContext):
                ctx.log("Lighting setup timed out")
                return GlobalTransitions.RETRY

            @classmethod
            def on_fail(cls, ctx: SharedContext):
                ctx.log("Lighting setup failed after all retries")
                return ImagingTransitions.FAILED

        class TakePicture(State):
            CONFIG = StateConfiguration(timeout=15.0, retries=2)

            @classmethod
            def transitions(cls):
                return {
                    ImagingTransitions.PICTURE_TAKEN: ProcessingComplete,
                    ImagingTransitions.FAILED: ErrorState,
                    GlobalTransitions.TIMEOUT: GlobalTransitions.RETRY,
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_enter(cls, ctx: SharedContext):
                ctx.log("Taking picture...")
                spawn(cls._simulate_capture, ctx)

            @classmethod
            def _simulate_capture(cls, ctx: SharedContext):
                sleep(0.1)  # Short simulation
                ctx.send_message({"type": "picture_complete", "success": True})

            @classmethod
            def on_state(cls, ctx: SharedContext):
                if isinstance(ctx.msg, TimeoutMessage):
                    return GlobalTransitions.TIMEOUT
                elif ctx.msg and ctx.msg.get("type") == "picture_complete":
                    if ctx.msg.get("success"):
                        return ImagingTransitions.PICTURE_TAKEN
                    else:
                        return ImagingTransitions.FAILED
                return GlobalTransitions.UNHANDLED

        class ProcessingComplete(State):
            CONFIG = StateConfiguration(terminal=True)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                ctx.log("Processing complete!")
                return GlobalTransitions.UNHANDLED

        class ErrorState(State):
            CONFIG = StateConfiguration(terminal=True)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                ctx.log("Error state reached")
                return GlobalTransitions.UNHANDLED

        # Run the scenario
        scenario_result = run_fsm_scenario(
            SetImageLighting, max_iterations=20, common_data={"images_captured": 0}
        )

        # Verify the scenario completed successfully
        self.assertTrue(
            scenario_result["completed"]
            or scenario_result["final_state"] in [ProcessingComplete, ErrorState],
            f"Scenario should complete or reach error state. Final: {scenario_result}",
        )

    def test_menu_navigation_with_push_pop(self):
        """Test menu navigation using push/pop operations."""

        class MenuTransitions(TransitionName):
            ENTER_SUBMENU = "enter_submenu"
            BACK = "back"
            EXIT = "exit"

        class MainMenu(State):
            CONFIG = StateConfiguration(can_dwell=True)

            @classmethod
            def transitions(cls):
                return {
                    MenuTransitions.ENTER_SUBMENU: Push(SubMenu, MainMenu),
                    MenuTransitions.EXIT: ProcessingComplete,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                if ctx.msg and ctx.msg.get("action") == "enter_submenu":
                    return MenuTransitions.ENTER_SUBMENU
                elif ctx.msg and ctx.msg.get("action") == "exit":
                    return MenuTransitions.EXIT

        class SubMenu(State):
            CONFIG = StateConfiguration(can_dwell=True)

            @classmethod
            def transitions(cls):
                return {
                    MenuTransitions.BACK: Pop,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                if ctx.msg and ctx.msg.get("action") == "back":
                    return MenuTransitions.BACK

        class ProcessingComplete(State):
            CONFIG = StateConfiguration(terminal=True)

            @classmethod
            def on_state(cls, ctx: SharedContext):
                pass

        # Test the menu navigation
        messages = [{"action": "enter_submenu"}, {"action": "back"}, {"action": "exit"}]

        scenario_result = run_fsm_scenario(
            MainMenu, messages=messages, max_iterations=10, timeout=0
        )
        
        self.assertEqual(scenario_result['error'], None)
        # Should reach ProcessingComplete
        self.assertEqual(scenario_result["final_state"], ProcessingComplete)
        self.assertTrue(scenario_result["completed"])


class TestAnalysisAndVisualization(StateMachineTestCase):
    """Test analysis and visualization features."""

    def test_state_discovery(self):
        """StateMachine should discover all reachable states."""

        class TestTransitions(TransitionName):
            GO_TO_B = "go_to_b"
            GO_TO_C = "go_to_c"

        class StateA(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.GO_TO_B: StateB,
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.GO_TO_B

        class StateB(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.GO_TO_C: StateC,
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.GO_TO_C

        class StateC(State):
            CONFIG = StateConfiguration(terminal=True)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        sm = StateMachine(StateA)

        # Should discover all three states
        expected_states = [StateA, StateB, StateC]
        self.assertFSMAnalysis(sm, expected_states=expected_states)

    def test_state_grouping(self):
        """StateMachine should group states by base class."""

        class ImagingState(State):
            """Base class for imaging states."""

            pass

        class ImagingTransitions(TransitionName):
            NEXT = "next"

        class CaptureImage(ImagingState):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    ImagingTransitions.NEXT: ProcessImage,
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return ImagingTransitions.NEXT

        class ProcessImage(ImagingState):
            CONFIG = StateConfiguration(terminal=True)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        sm = StateMachine(CaptureImage)

        # Should group under ImagingState
        groups = sm.state_groups
        self.assertIn("ImagingState", groups)

    def test_push_pop_analysis(self):
        """StateMachine should analyze push/pop relationships."""

        class TestTransitions(TransitionName):
            PUSH_B = "push_b"
            POP_BACK = "pop_back"

        class StateA(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.PUSH_B: Push(StateB, StateA),
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.PUSH_B

        class StateB(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.POP_BACK: Pop,
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP_BACK

        sm = StateMachine(StateA)

        # Should analyze push/pop relationships
        relationships = sm.get_push_pop_relationships()

        # StateB should have push sources and pop targets
        stateb_name = StateB.name()
        if stateb_name in relationships:
            stateb_info = relationships[stateb_name]
            self.assertTrue(stateb_info.get("is_pushed_to", False))
            self.assertTrue(stateb_info.get("has_pop", False))

    def test_mermaid_flowchart_generation(self):
        """StateMachine should generate Mermaid flowcharts."""

        class TestTransitions(TransitionName):
            GO_TO_B = "go_to_b"

        class StateA(State):
            CONFIG = StateConfiguration()

            @classmethod
            def transitions(cls):
                return {
                    TestTransitions.GO_TO_B: StateB,
                    GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
                }

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return TestTransitions.GO_TO_B

        class StateB(State):
            CONFIG = StateConfiguration(terminal=True)

            @classmethod
            def transitions(cls):
                return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

            @classmethod
            def on_state(cls, ctx: SharedContext):
                return GlobalTransitions.UNHANDLED

        sm = StateMachine(StateA)

        # Should generate a flowchart
        flowchart = sm.generate_mermaid_flowchart()
        self.assertIsInstance(flowchart, str)
        self.assertIn("flowchart", flowchart.lower())
        self.assertIn("StateA", flowchart)
        self.assertIn("StateB", flowchart)


if __name__ == "__main__":
    # Run specific test groups
    import sys

    if len(sys.argv) > 1:
        # Run specific test class
        test_class = sys.argv[1]
        suite = unittest.TestLoader().loadTestsFromName(f"__main__.{test_class}")
    else:
        # Run all tests
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
